//! 3D velocity perturbation hook (`VelocityPerturbation3D`), smooth `SubductionPlumeModel3D`,
//! Um & Thurber (1987) 3-point pseudo-bending ray path perturbation, and Damped Huber-IRLS 4D
//! earthquake hypocenter inverter (`locate_hypocenter`) with analytical $4\times 4$ covariance
//! and 3D confidence ellipsoids.

use crate::calculators::{spherical_great_circle_point, ArrivalPath, RayPathPoint};
use crate::ellipticity::arrival_ellipticity_correction_s;
use crate::models::{MaterialProperty, WaveType};
use crate::phase::{RawArrival, SeismicPhase, DEFAULT_RAY_PARAM_TOL};
use crate::tau_model::TauModel;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;
use earthsnail_core::event::{Arrival as CoreArrival, Origin, OriginQuality, Pick};
use earthsnail_core::geodesics::{
    geocentric_central_angle_deg, gps2dist_azimuth_ellipsoid, try_gps2dist_azimuth, Ellipsoid,
};
use earthsnail_core::time::UtcTimestamp;
use earthsnail_core::{math, EarthSnailError, Result};

/// Hook for 3D lateral and radial seismic velocity perturbations $\delta v / v_0$ layered on top
/// of a 1D spherical [`TauModel`].
pub trait VelocityPerturbation3D: Sync {
    /// Returns fractional velocity perturbation $\delta v / v_0$ (dimensionless, e.g. `+0.04` for
    /// a `+4%` fast slab or `-0.03` for a `-3%` slow plume) at geographic `(lat_deg, lon_deg)`,
    /// `depth_km`, and `wave_type` (`WaveType::P` or `WaveType::S`).
    fn dv_over_v(&self, lat_deg: f64, lon_deg: f64, depth_km: f64, wave_type: WaveType) -> f64;
}

/// Smooth $C^\infty$ 3D mantle heterogeneity model combining a dipping high-velocity subduction
/// slab and a buoyant low-velocity mantle plume conduit + upper-mantle mushroom head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubductionPlumeModel3D {
    /// Trench reference geographic latitude in degrees.
    pub slab_trench_lat_deg: f64,
    /// Trench reference geographic longitude in degrees.
    pub slab_trench_lon_deg: f64,
    /// Trench strike azimuth clockwise from North in degrees (dip direction is `strike + 90 deg`).
    pub slab_strike_deg: f64,
    /// Slab dip angle from horizontal in `(0, 90)` degrees.
    pub slab_dip_deg: f64,
    /// Gaussian half-thickness perpendicular to the slab plane in kilometers.
    pub slab_half_thickness_km: f64,
    /// Gaussian half-length along strike in kilometers.
    pub slab_half_length_km: f64,
    /// Maximum slab penetration depth in kilometers.
    pub slab_max_depth_km: f64,
    /// Peak fractional P-wave anomaly in the slab core (e.g. `+0.045`).
    pub slab_dv_p: f64,
    /// Peak fractional S-wave anomaly in the slab core (e.g. `+0.070`).
    pub slab_dv_s: f64,
    /// Mantle plume axis geographic latitude in degrees.
    pub plume_lat_deg: f64,
    /// Mantle plume axis geographic longitude in degrees.
    pub plume_lon_deg: f64,
    /// Plume lower/mid-mantle conduit Gaussian radius in kilometers.
    pub plume_conduit_radius_km: f64,
    /// Plume upper-mantle mushroom head center depth in kilometers.
    pub plume_head_depth_km: f64,
    /// Plume head horizontal Gaussian radius in kilometers.
    pub plume_head_radius_km: f64,
    /// Plume head vertical Gaussian half-thickness in kilometers.
    pub plume_head_half_thickness_km: f64,
    /// Maximum depth of the plume root in kilometers (e.g. CMB `2889 km`).
    pub plume_max_depth_km: f64,
    /// Peak fractional P-wave anomaly in the plume (`< 0`, e.g. `-0.025`).
    pub plume_dv_p: f64,
    /// Peak fractional S-wave anomaly in the plume (`< 0`, e.g. `-0.045`).
    pub plume_dv_s: f64,
}

impl Default for SubductionPlumeModel3D {
    fn default() -> Self {
        Self::pacific_ring_default()
    }
}

impl SubductionPlumeModel3D {
    /// Canonical Northwest Pacific dipping slab (Japan Trench) + central Pacific hotspot plume.
    pub const fn pacific_ring_default() -> Self {
        Self {
            slab_trench_lat_deg: 38.0,
            slab_trench_lon_deg: 144.0,
            slab_strike_deg: 200.0, // SSW strike -> WNW dip (290 deg) beneath Honshu/Sea of Japan
            slab_dip_deg: 30.0,
            slab_half_thickness_km: 45.0,
            slab_half_length_km: 900.0,
            slab_max_depth_km: 660.0,
            slab_dv_p: 0.045,
            slab_dv_s: 0.070,
            plume_lat_deg: 19.5,
            plume_lon_deg: -155.5,
            plume_conduit_radius_km: 150.0,
            plume_head_depth_km: 200.0,
            plume_head_radius_km: 450.0,
            plume_head_half_thickness_km: 120.0,
            plume_max_depth_km: 2889.0,
            plume_dv_p: -0.025,
            plume_dv_s: -0.045,
        }
    }
}

impl VelocityPerturbation3D for SubductionPlumeModel3D {
    fn dv_over_v(&self, lat_deg: f64, lon_deg: f64, depth_km: f64, wave_type: WaveType) -> f64 {
        if depth_km < 0.0 {
            return 0.0;
        }

        // 1. Dipping slab contribution in local tangent frame at trench reference point:
        let slab_amp = match wave_type {
            WaveType::P => self.slab_dv_p,
            WaveType::S => self.slab_dv_s,
        };
        let mut dv_total = 0.0;
        if slab_amp != 0.0 && depth_km <= self.slab_max_depth_km * 1.5 {
            let (dn_km, de_km) = local_ne_offset_km(
                self.slab_trench_lat_deg,
                self.slab_trench_lon_deg,
                lat_deg,
                lon_deg,
            );
            let strike_rad = self.slab_strike_deg.to_radians();
            let (ss, cs) = math::sin_cos(strike_rad);
            // Along-strike coordinate s_along and downdip horizontal coordinate x_dip (strike + 90 deg):
            let s_along = dn_km * cs + de_km * ss;
            let x_dip = -dn_km * ss + de_km * cs;

            let dip_rad = self.slab_dip_deg.to_radians();
            let (sdip, cdip) = math::sin_cos(dip_rad);
            // Distance downdip along slab mid-plane (starts ~15 km below trench):
            let z_rel = depth_km - 15.0;
            let downdip_s = x_dip * cdip + z_rel * sdip;
            let normal_d = -x_dip * sdip + z_rel * cdip;

            let max_downdip = (self.slab_max_depth_km - 15.0).max(10.0) / sdip.max(0.1);
            let clamped_s = downdip_s.clamp(0.0, max_downdip);
            let edge_excess = downdip_s - clamped_s;

            let u_norm = normal_d / self.slab_half_thickness_km.max(1.0);
            let u_edge = edge_excess / self.slab_half_thickness_km.max(1.0);
            let u_strike = s_along / self.slab_half_length_km.max(1.0);
            let arg = u_norm * u_norm + u_edge * u_edge + u_strike * u_strike;
            if arg < 36.0 {
                dv_total += slab_amp * math::exp(-0.5 * arg);
            }
        }

        // 2. Mantle plume conduit + upper-mantle head contribution:
        let plume_amp = match wave_type {
            WaveType::P => self.plume_dv_p,
            WaveType::S => self.plume_dv_s,
        };
        if plume_amp != 0.0 && depth_km <= self.plume_max_depth_km {
            let r_surf_km = great_circle_chord_km(
                self.plume_lat_deg,
                self.plume_lon_deg,
                lat_deg,
                lon_deg,
                (6371.0 - depth_km).max(100.0),
            );
            let u_cond = r_surf_km / self.plume_conduit_radius_km.max(1.0);
            let w_depth = if depth_km < 40.0 {
                (depth_km / 40.0).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let cond_term = if u_cond * u_cond < 36.0 {
                w_depth * math::exp(-0.5 * u_cond * u_cond)
            } else {
                0.0
            };

            let u_hr = r_surf_km / self.plume_head_radius_km.max(1.0);
            let u_hz =
                (depth_km - self.plume_head_depth_km) / self.plume_head_half_thickness_km.max(1.0);
            let head_arg = u_hr * u_hr + u_hz * u_hz;
            let head_term = if head_arg < 36.0 {
                0.8 * math::exp(-0.5 * head_arg)
            } else {
                0.0
            };

            let combined = (cond_term + head_term).min(1.25);
            dv_total += plume_amp * combined;
        }

        dv_total
    }
}

#[inline]
fn local_ne_offset_km(lat0_deg: f64, lon0_deg: f64, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let phi0 = lat0_deg.to_radians();
    let lam0 = lon0_deg.to_radians();
    let phi = lat_deg.to_radians();
    let lam = lon_deg.to_radians();
    let (sp0, cp0) = math::sin_cos(phi0);
    let (sl0, cl0) = math::sin_cos(lam0);
    let (sp, cp) = math::sin_cos(phi);
    let (sl, cl) = math::sin_cos(lam);

    let u = [cp * cl, cp * sl, sp];
    let e_n = [-sp0 * cl0, -sp0 * sl0, cp0];
    let e_e = [-sl0, cl0, 0.0];
    let dn = 6371.0 * (u[0] * e_n[0] + u[1] * e_n[1] + u[2] * e_n[2]);
    let de = 6371.0 * (u[0] * e_e[0] + u[1] * e_e[1]);
    (dn, de)
}

#[inline]
fn great_circle_chord_km(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    radius_km: f64,
) -> f64 {
    let p1 = lat1_deg.to_radians();
    let l1 = lon1_deg.to_radians();
    let p2 = lat2_deg.to_radians();
    let l2 = lon2_deg.to_radians();
    let (sp1, cp1) = math::sin_cos(p1);
    let (sl1, cl1) = math::sin_cos(l1);
    let (sp2, cp2) = math::sin_cos(p2);
    let (sl2, cl2) = math::sin_cos(l2);
    let u1 = [cp1 * cl1, cp1 * sl1, sp1];
    let u2 = [cp2 * cl2, cp2 * sl2, sp2];
    let cross = [
        u1[1] * u2[2] - u1[2] * u2[1],
        u1[2] * u2[0] - u1[0] * u2[2],
        u1[0] * u2[1] - u1[1] * u2[0],
    ];
    let dot = u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2];
    let sin_ang = math::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    radius_km * math::atan2(sin_ang, dot)
}

/// Result of 3D ray perturbation (`Fermat` path integral + Um & Thurber 1987 3-point pseudo-bending).
#[derive(Debug, Clone, PartialEq)]
pub struct PerturbedRayPath3D {
    /// Original 1D TauP reference travel time in seconds.
    pub time_1d_s: f64,
    /// First-order Fermat path-integral 3D travel time along the unbent 1D path in seconds.
    pub time_fermat_3d_s: f64,
    /// 3D travel time after Um & Thurber (1987) 3-point pseudo-bending in seconds (`<= time_fermat_3d_s`).
    pub time_bent_3d_s: f64,
    /// Maximum 3D spatial displacement of the bent ray relative to the 1D path in kilometers.
    pub max_ray_shift_km: f64,
    /// Updated 3D geographic ray-path points (`lat_deg`, `lon_deg`, `depth_km`, `radius_km`, `time_s`).
    pub path_points: Vec<RayPathPoint>,
}

#[inline]
fn geo_depth_to_xyz(lat_deg: f64, lon_deg: f64, depth_km: f64, r_planet: f64) -> [f64; 3] {
    let r = (r_planet - depth_km).max(0.0);
    let (sp, cp) = math::sin_cos(lat_deg.to_radians());
    let (sl, cl) = math::sin_cos(lon_deg.to_radians());
    [r * cp * cl, r * cp * sl, r * sp]
}

#[inline]
fn xyz_to_geo_depth(xyz: [f64; 3], r_planet: f64) -> (f64, f64, f64) {
    let r_xy = math::hypot(xyz[0], xyz[1]);
    let r = math::hypot(r_xy, xyz[2]);
    let lat_deg = math::atan2(xyz[2], r_xy).to_degrees();
    let lon_deg = math::atan2(xyz[1], xyz[0]).to_degrees();
    let depth_km = (r_planet - r).clamp(0.0, r_planet);
    (lat_deg, lon_deg, depth_km)
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    math::sqrt(dx * dx + dy * dy + dz * dz)
}

fn eval_velocity_3d(
    model: &TauModel,
    pert3d: &dyn VelocityPerturbation3D,
    xyz: [f64; 3],
    wave_type: WaveType,
    prefer_below: bool,
) -> f64 {
    let r_planet = model.radius_of_planet;
    let (lat, lon, depth) = xyz_to_geo_depth(xyz, r_planet);
    let prop = match wave_type {
        WaveType::P => MaterialProperty::PVelocity,
        WaveType::S => MaterialProperty::SVelocity,
    };
    let mut v1d = if prefer_below {
        model.s_mod.v_mod.evaluate_below(depth, prop).unwrap_or(8.0)
    } else {
        model.s_mod.v_mod.evaluate_above(depth, prop).unwrap_or(8.0)
    };
    if v1d <= 1e-6 {
        // Fluid outer core fallback if S requested inadvertently:
        v1d = model
            .s_mod
            .v_mod
            .evaluate_below(depth, MaterialProperty::PVelocity)
            .unwrap_or(8.0)
            .max(0.1);
    }
    let dv = pert3d.dv_over_v(lat, lon, depth, wave_type);
    let factor = (1.0 + dv).clamp(0.25, 2.5);
    v1d * factor
}

/// Evaluates Fermat's first-order 3D path-integral travel-time perturbation AND Um & Thurber
/// (1987) 3-point pseudo-bending ray deformation on top of a 1D geographic [`ArrivalPath`].
pub fn perturb_ray_path_3d(
    model: &TauModel,
    arrival_path: &ArrivalPath,
    pert3d: &dyn VelocityPerturbation3D,
    sweeps: usize,
) -> Result<PerturbedRayPath3D> {
    let pts = &arrival_path.path_points;
    let n = pts.len();
    if n < 2 {
        return Ok(PerturbedRayPath3D {
            time_1d_s: arrival_path.arrival.time_s,
            time_fermat_3d_s: arrival_path.arrival.time_s,
            time_bent_3d_s: arrival_path.arrival.time_s,
            max_ray_shift_km: 0.0,
            path_points: pts.clone(),
        });
    }

    for pt in pts {
        if pt.lat_deg.is_none() || pt.lon_deg.is_none() {
            return Err(EarthSnailError::InvalidParameter(
                "perturb_ray_path_3d requires geographic (lat_deg, lon_deg) coordinates from get_ray_paths_geo".into(),
            ));
        }
    }

    let r_planet = model.radius_of_planet;
    let default_s =
        arrival_path.arrival.name.starts_with('S') || arrival_path.arrival.name.starts_with('s');

    // Infer segment wave type from local 1D slowness / phase character:
    let mut seg_wave = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        let mid_d = 0.5 * (pts[k].depth_km + pts[k + 1].depth_km);
        let dt = pts[k + 1].time_s - pts[k].time_s;
        if model.s_mod.depth_in_fluid(mid_d) {
            seg_wave.push(WaveType::P);
        } else if dt > 1e-9 {
            let p1 = geo_depth_to_xyz(
                pts[k].lat_deg.unwrap(),
                pts[k].lon_deg.unwrap(),
                pts[k].depth_km,
                r_planet,
            );
            let p2 = geo_depth_to_xyz(
                pts[k + 1].lat_deg.unwrap(),
                pts[k + 1].lon_deg.unwrap(),
                pts[k + 1].depth_km,
                r_planet,
            );
            let v_est = dist3(p1, p2) / dt;
            let vp = model
                .s_mod
                .v_mod
                .evaluate_below(mid_d, MaterialProperty::PVelocity)
                .unwrap_or(8.0);
            let vs = model
                .s_mod
                .v_mod
                .evaluate_below(mid_d, MaterialProperty::SVelocity)
                .unwrap_or(4.5);
            if (v_est - vs).abs() < (v_est - vp).abs() {
                seg_wave.push(WaveType::S);
            } else {
                seg_wave.push(WaveType::P);
            }
        } else {
            seg_wave.push(if default_s { WaveType::S } else { WaveType::P });
        }
    }

    // 1. Exact first-order Fermat path-integral travel time on the 1D ray:
    let mut time_fermat_3d_s = 0.0;
    for k in 0..(n - 1) {
        let dt0 = pts[k + 1].time_s - pts[k].time_s;
        if dt0 <= 0.0 {
            continue;
        }
        let x0 = geo_depth_to_xyz(
            pts[k].lat_deg.unwrap(),
            pts[k].lon_deg.unwrap(),
            pts[k].depth_km,
            r_planet,
        );
        let x1 = geo_depth_to_xyz(
            pts[k + 1].lat_deg.unwrap(),
            pts[k + 1].lon_deg.unwrap(),
            pts[k + 1].depth_km,
            r_planet,
        );
        let xm = [
            0.5 * (x0[0] + x1[0]),
            0.5 * (x0[1] + x1[1]),
            0.5 * (x0[2] + x1[2]),
        ];
        let (lat_m, lon_m, dep_m) = xyz_to_geo_depth(xm, r_planet);
        let dv = pert3d
            .dv_over_v(lat_m, lon_m, dep_m, seg_wave[k])
            .clamp(-0.75, 1.5);
        time_fermat_3d_s += dt0 / (1.0 + dv);
    }
    // Preserve any ellipticity offset already applied to arrival.time_s:
    let ellip_off = arrival_path.arrival.ellipticity_corr_s.unwrap_or(0.0);
    time_fermat_3d_s += ellip_off;

    // 2. Um & Thurber (1987) 3-point pseudo-bending along continuous medium sub-segments:
    let orig_xyz: Vec<[f64; 3]> = pts
        .iter()
        .map(|p| geo_depth_to_xyz(p.lat_deg.unwrap(), p.lon_deg.unwrap(), p.depth_km, r_planet))
        .collect();
    let mut bent_xyz = orig_xyz.clone();

    // Pin endpoints, wave-type conversions, and first-order discontinuity pierce vertices
    // (free surface, Moho, CMB, ICB) so Snell kinks at sharp interfaces are preserved.
    let mut pinned = vec![false; n];
    pinned[0] = true;
    pinned[n - 1] = true;
    for k in 1..(n - 1) {
        let d = pts[k].depth_km;
        let on_major_discon = (d - 0.0).abs() < 1e-6
            || (d - model.moho_depth).abs() < 1e-6
            || (d - model.cmb_depth).abs() < 1e-6
            || (d - model.iocb_depth).abs() < 1e-6;
        let mode_flip = seg_wave[k - 1] != seg_wave[k];
        let zero_step = dist3(orig_xyz[k - 1], orig_xyz[k]) < 1e-3
            || dist3(orig_xyz[k], orig_xyz[k + 1]) < 1e-3;
        if on_major_discon || mode_flip || zero_step {
            pinned[k] = true;
        }
    }

    let pair_travel_time = |xa: [f64; 3], xb: [f64; 3], xc: [f64; 3], wtype: WaveType| -> f64 {
        let va = eval_velocity_3d(model, pert3d, xa, wtype, true);
        let vb = eval_velocity_3d(model, pert3d, xb, wtype, true);
        let vc = eval_velocity_3d(model, pert3d, xc, wtype, true);
        let lab = dist3(xa, xb);
        let lbc = dist3(xb, xc);
        0.5 * lab * (1.0 / va + 1.0 / vb) + 0.5 * lbc * (1.0 / vb + 1.0 / vc)
    };

    let raw_unbent_chord_time: f64 = (0..(n - 1))
        .map(|k| {
            let w = seg_wave[k];
            let va = eval_velocity_3d(model, pert3d, orig_xyz[k], w, true);
            let vb = eval_velocity_3d(model, pert3d, orig_xyz[k + 1], w, true);
            0.5 * dist3(orig_xyz[k], orig_xyz[k + 1]) * (1.0 / va + 1.0 / vb)
        })
        .sum();

    let fd_h = 2.0_f64; // 2 km central-difference stencil for 3D velocity gradient
    for _sweep in 0..sweeps {
        let mut max_step = 0.0_f64;
        for k in 1..(n - 1) {
            if pinned[k] {
                continue;
            }
            let xa = bent_xyz[k - 1];
            let xc = bent_xyz[k + 1];
            let chord = [xc[0] - xa[0], xc[1] - xa[1], xc[2] - xa[2]];
            let chord_len =
                math::sqrt(chord[0] * chord[0] + chord[1] * chord[1] + chord[2] * chord[2]);
            if chord_len < 1e-3 {
                continue;
            }
            let half_l = 0.5 * chord_len;
            let t_hat = [
                chord[0] / chord_len,
                chord[1] / chord_len,
                chord[2] / chord_len,
            ];
            let x_mid = [
                0.5 * (xa[0] + xc[0]),
                0.5 * (xa[1] + xc[1]),
                0.5 * (xa[2] + xc[2]),
            ];
            let wtype = seg_wave[k];

            let v1 = eval_velocity_3d(model, pert3d, xa, wtype, true);
            let v2 = eval_velocity_3d(model, pert3d, xc, wtype, true);
            let vm = eval_velocity_3d(model, pert3d, x_mid, wtype, true);

            // 3D central-difference velocity gradient at x_mid:
            let mut grad_v = [0.0; 3];
            for axis in 0..3 {
                let mut xp = x_mid;
                let mut xm = x_mid;
                xp[axis] += fd_h;
                xm[axis] -= fd_h;
                let vp = eval_velocity_3d(model, pert3d, xp, wtype, true);
                let vn = eval_velocity_3d(model, pert3d, xm, wtype, true);
                grad_v[axis] = (vp - vn) / (2.0 * fd_h);
            }

            let dot_gt = grad_v[0] * t_hat[0] + grad_v[1] * t_hat[1] + grad_v[2] * t_hat[2];
            let g_perp = [
                grad_v[0] - dot_gt * t_hat[0],
                grad_v[1] - dot_gt * t_hat[1],
                grad_v[2] - dot_gt * t_hat[2],
            ];
            let g_norm =
                math::sqrt(g_perp[0] * g_perp[0] + g_perp[1] * g_perp[1] + g_perp[2] * g_perp[2]);

            let x_old = bent_xyz[k];
            let t_old = pair_travel_time(xa, x_old, xc, wtype);

            let x_cand = if g_norm > 1e-12 {
                let n_hat = [g_perp[0] / g_norm, g_perp[1] / g_norm, g_perp[2] / g_norm];
                // Um & Thurber (1987, Eq. 6) cancellation-free offset Rc along +grad_v_perp:
                let c_slow = 0.5 * (1.0 / v1 + 1.0 / v2);
                let a_coef = (c_slow * vm + 1.0) / (4.0 * c_slow);
                let l2_term = (half_l * half_l) / (2.0 * c_slow * vm);
                let rc = (g_norm * l2_term)
                    / (a_coef + math::sqrt(a_coef * a_coef + g_norm * g_norm * l2_term));
                let rc_clamped = rc.min(0.5 * half_l);
                [
                    x_mid[0] + rc_clamped * n_hat[0],
                    x_mid[1] + rc_clamped * n_hat[1],
                    x_mid[2] + rc_clamped * n_hat[2],
                ]
            } else {
                x_mid
            };

            // Safeguarded line search from x_old toward x_cand ensuring strict Fermat time decrease
            // and keeping the ray inside the planet interior:
            let mut alpha = 1.0;
            while alpha >= 0.125 {
                let x_try = [
                    (1.0 - alpha) * x_old[0] + alpha * x_cand[0],
                    (1.0 - alpha) * x_old[1] + alpha * x_cand[1],
                    (1.0 - alpha) * x_old[2] + alpha * x_cand[2],
                ];
                let r_try = math::hypot(math::hypot(x_try[0], x_try[1]), x_try[2]);
                if r_try <= r_planet {
                    let t_try = pair_travel_time(xa, x_try, xc, wtype);
                    if t_try <= t_old {
                        let step = dist3(x_old, x_try);
                        if step > max_step {
                            max_step = step;
                        }
                        bent_xyz[k] = x_try;
                        break;
                    }
                }
                alpha *= 0.5;
            }
        }
        if max_step < 1e-4 {
            break;
        }
    }

    let raw_bent_chord_time: f64 = (0..(n - 1))
        .map(|k| {
            let w = seg_wave[k];
            let va = eval_velocity_3d(model, pert3d, bent_xyz[k], w, true);
            let vb = eval_velocity_3d(model, pert3d, bent_xyz[k + 1], w, true);
            0.5 * dist3(bent_xyz[k], bent_xyz[k + 1]) * (1.0 / va + 1.0 / vb)
        })
        .sum();

    // Subtract the non-negative 2nd-order Fermat bending reduction from the exact 1D-anchored
    // Fermat 3D time so discretization chord bias of the 1D mesh cancels identically!
    let bending_gain_s = (raw_unbent_chord_time - raw_bent_chord_time).max(0.0);
    let time_bent_3d_s = time_fermat_3d_s - bending_gain_s;

    let mut max_ray_shift_km = 0.0_f64;
    let mut updated_pts = Vec::with_capacity(n);
    let mut cum_t = 0.0_f64;
    let ev_lat = pts[0].lat_deg.unwrap();
    let ev_lon = pts[0].lon_deg.unwrap();

    for k in 0..n {
        let shift = dist3(orig_xyz[k], bent_xyz[k]);
        if shift > max_ray_shift_km {
            max_ray_shift_km = shift;
        }
        if k > 0 {
            let w = seg_wave[k - 1];
            let va = eval_velocity_3d(model, pert3d, bent_xyz[k - 1], w, true);
            let vb = eval_velocity_3d(model, pert3d, bent_xyz[k], w, true);
            cum_t += 0.5 * dist3(bent_xyz[k - 1], bent_xyz[k]) * (1.0 / va + 1.0 / vb);
        }
        let (lat, lon, depth) = xyz_to_geo_depth(bent_xyz[k], r_planet);
        let dist_rad = if k == 0 {
            0.0
        } else {
            great_circle_chord_km(ev_lat, ev_lon, lat, lon, 1.0)
        };
        updated_pts.push(RayPathPoint {
            p_s_per_rad: pts[k].p_s_per_rad,
            time_s: cum_t,
            dist_rad,
            dist_deg: dist_rad.to_degrees(),
            depth_km: depth,
            radius_km: r_planet - depth,
            lat_deg: Some(lat),
            lon_deg: Some(lon),
        });
    }

    Ok(PerturbedRayPath3D {
        time_1d_s: arrival_path.arrival.time_s,
        time_fermat_3d_s,
        time_bent_3d_s,
        max_ray_shift_km,
        path_points: updated_pts,
    })
}

/// Single seismic phase arrival observation at a known station used by [`locate_hypocenter`].
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseArrivalObservation {
    /// Station or channel identifier (e.g. `"IU.ANMO"`).
    pub station_id: String,
    /// Station geographic latitude in `[-90, 90]` degrees.
    pub station_lat_deg: f64,
    /// Station geographic longitude in `[-180, 180]` degrees.
    pub station_lon_deg: f64,
    /// Station elevation above sea level in meters (`0.0` at sea level).
    pub station_elevation_m: f64,
    /// Observed seismic phase label (e.g. `"P"`, `"S"`, `"Pn"`, `"pP"`, `"PKP"`, `"PKIKP"`).
    pub phase: String,
    /// Observed arrival time in seconds relative to a chosen reference epoch.
    pub arrival_time_s: f64,
    /// Symmetric $1\sigma$ pick standard deviation in seconds (defaults to `0.1 s` if `<= 0`).
    pub uncertainty_s: f64,
}

impl PhaseArrivalObservation {
    /// Convenience constructor for a sea-level station observation.
    pub fn new(
        station_id: impl Into<String>,
        station_lat_deg: f64,
        station_lon_deg: f64,
        phase: impl Into<String>,
        arrival_time_s: f64,
        uncertainty_s: f64,
    ) -> Self {
        Self {
            station_id: station_id.into(),
            station_lat_deg,
            station_lon_deg,
            station_elevation_m: 0.0,
            phase: phase.into(),
            arrival_time_s,
            uncertainty_s,
        }
    }

    /// Converts an [`earthsnail_core::Pick`] into a [`PhaseArrivalObservation`] relative to
    /// `reference_epoch`.
    pub fn from_pick(
        pick: &Pick,
        reference_epoch: UtcTimestamp,
        station_lat_deg: f64,
        station_lon_deg: f64,
        station_elevation_m: f64,
        default_phase: &str,
    ) -> Self {
        let phase = pick
            .phase_hint
            .as_deref()
            .unwrap_or(default_phase)
            .to_string();
        let arrival_time_s = pick.time.duration_secs_f64_since(reference_epoch);
        let uncertainty_s = pick.time_uncertainty_s.unwrap_or(0.1).max(1e-3);
        Self {
            station_id: pick.channel.to_string(),
            station_lat_deg,
            station_lon_deg,
            station_elevation_m,
            phase,
            arrival_time_s,
            uncertainty_s,
        }
    }
}

/// Configuration for the Damped Huber-IRLS 4D Hypocenter Inverter (`locate_hypocenter`).
#[derive(Clone)]
pub struct HypocenterConfig<'a> {
    /// Apply WGS84 geocentric central angle + Dziewonski-Gilbert ellipticity corrections.
    pub apply_ellipticity: bool,
    /// Optional 3D mantle/crustal velocity perturbation model (`Fermat` + pseudo-bending).
    pub perturbation_3d: Option<&'a dyn VelocityPerturbation3D>,
    /// Dimensionless Huber tuning constant $k$ on standardized residuals $|r_i / \sigma_i|$
    /// (`1.345` gives 95% asymptotic Gaussian efficiency while bounding outlier leverage).
    pub huber_k: f64,
    /// Tikhonov regularization parameter $\lambda > 0$ used in
    /// $\boldsymbol{\Sigma} = \hat{\sigma}^2 (\mathbf{G}^T \mathbf{W} \mathbf{G} + \lambda \mathbf{I}_4)^{-1}$.
    pub damping_lambda: f64,
    /// Maximum Gauss-Newton / Levenberg-Marquardt IRLS iterations.
    pub max_iterations: usize,
    /// Convergence threshold on 4D parameter step norm in kilometers/seconds.
    pub step_tol: f64,
    /// Minimum permitted focal depth in kilometers (`>= 0.0`).
    pub min_depth_km: f64,
    /// Maximum permitted focal depth in kilometers (`<= 750.0` typical).
    pub max_depth_km: f64,
}

impl<'a> Default for HypocenterConfig<'a> {
    fn default() -> Self {
        Self {
            apply_ellipticity: false,
            perturbation_3d: None,
            huber_k: 1.345,
            damping_lambda: 1e-4,
            max_iterations: 40,
            step_tol: 1e-6,
            min_depth_km: 0.0,
            max_depth_km: 700.0,
        }
    }
}

impl<'a> core::fmt::Debug for HypocenterConfig<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HypocenterConfig")
            .field("apply_ellipticity", &self.apply_ellipticity)
            .field("has_perturbation_3d", &self.perturbation_3d.is_some())
            .field("huber_k", &self.huber_k)
            .field("damping_lambda", &self.damping_lambda)
            .field("max_iterations", &self.max_iterations)
            .field("step_tol", &self.step_tol)
            .field("min_depth_km", &self.min_depth_km)
            .field("max_depth_km", &self.max_depth_km)
            .finish()
    }
}

/// Per-observation residual, ray geometry, and final Huber weight at the converged hypocenter.
#[derive(Debug, Clone, PartialEq)]
pub struct StationResidual {
    /// Station identifier.
    pub station_id: String,
    /// Matched phase name.
    pub phase: String,
    /// Epicentral distance from converged hypocenter to station in degrees.
    pub distance_deg: f64,
    /// Event-to-station azimuth clockwise from North in degrees.
    pub azimuth_deg: f64,
    /// Takeoff angle at source measured from downward vertical in degrees.
    pub takeoff_angle_deg: f64,
    /// Predicted arrival time $t_0 + T_{\text{pred}}$ in seconds.
    pub predicted_time_s: f64,
    /// Travel-time residual $r_i = t_i^{\text{obs}} - t_i^{\text{pred}}$ in seconds.
    pub residual_s: f64,
    /// Final Huber IRLS weight $w_{H,i} \in (0, 1]$ (`< 1` indicates down-weighted outlier).
    pub huber_weight: f64,
}

/// 3D spatial confidence ellipsoid derived from eigendecomposition of the $3\times 3$ `(North, East, Down)`
/// block of the posterior covariance matrix $\boldsymbol{\Sigma}$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceEllipsoid {
    /// $1\sigma$ semi-axis lengths `[semi_major_km, semi_intermediate_km, semi_minor_km]`
    /// sorted descending (`>= 0`).
    pub semi_axes_1sigma_km: [f64; 3],
    /// 95% confidence ($\sqrt{\chi^2_3(0.95)} \approx 2.7955$) semi-axis lengths in kilometers.
    pub semi_axes_95pct_km: [f64; 3],
    /// Orthonormal principal-axis unit vectors (columns/rows in `[North_km, East_km, Down_km]`).
    pub principal_axes_ned: [[f64; 3]; 3],
}

/// Converged 4D hypocenter solution `(lat_deg, lon_deg, depth_km, origin_time_s)` with analytical
/// $4\times 4$ covariance matrix and confidence ellipsoid.
#[derive(Debug, Clone, PartialEq)]
pub struct HypocenterSolution {
    /// Converged hypocenter geographic latitude in `[-90, 90]` degrees.
    pub lat_deg: f64,
    /// Converged hypocenter geographic longitude in `[-180, 180]` degrees.
    pub lon_deg: f64,
    /// Converged focal depth below sea level in kilometers.
    pub depth_km: f64,
    /// Converged earthquake origin time in seconds relative to the input reference epoch.
    pub origin_time_s: f64,
    /// Weighted root-mean-square travel-time residual in seconds.
    pub weighted_rms_s: f64,
    /// Estimated posterior variance factor $\hat{\sigma}^2$ (`s^2`).
    pub variance_scale_s2: f64,
    /// Closed-form $4\times 4$ posterior covariance matrix
    /// $\boldsymbol{\Sigma} = \hat{\sigma}^2 (\mathbf{G}^T \mathbf{W} \mathbf{G} + \lambda \mathbf{I}_4)^{-1}$
    /// in `[North_km, East_km, Down_km, OriginTime_s]` coordinates.
    pub covariance_nedt: [[f64; 4]; 4],
    /// 3D spatial confidence ellipsoid over `[North_km, East_km, Down_km]`.
    pub ellipsoid: ConfidenceEllipsoid,
    /// Number of IRLS Gauss-Newton iterations executed.
    pub iterations: usize,
    /// Largest azimuthal gap across stations in degrees.
    pub azimuthal_gap_deg: f64,
    /// Per-observation residuals and Huber weights.
    pub residuals: Vec<StationResidual>,
}

impl HypocenterSolution {
    /// Converts this `HypocenterSolution` into a QuakeML-compatible [`earthsnail_core::Origin`]
    /// anchored to `reference_epoch`.
    pub fn to_origin(&self, reference_epoch: UtcTimestamp) -> Origin {
        let origin_time = reference_epoch
            .add_secs_f64(self.origin_time_s)
            .unwrap_or(reference_epoch);
        let lat_unc_km = math::sqrt(self.covariance_nedt[0][0].max(0.0));
        let lon_unc_km = math::sqrt(self.covariance_nedt[1][1].max(0.0));
        let dep_unc_m = math::sqrt(self.covariance_nedt[2][2].max(0.0)) * 1_000.0;
        let time_unc_s = math::sqrt(self.covariance_nedt[3][3].max(0.0));

        let mut min_d = f64::INFINITY;
        let mut max_d = 0.0_f64;
        let mut used_count = 0u32;
        let mut arrivals = Vec::with_capacity(self.residuals.len());

        for sr in &self.residuals {
            min_d = min_d.min(sr.distance_deg);
            max_d = max_d.max(sr.distance_deg);
            if sr.huber_weight > 0.0 {
                used_count += 1;
            }
            arrivals.push(CoreArrival {
                pick_id: Some(sr.station_id.clone()),
                phase: sr.phase.clone(),
                azimuth_deg: Some(sr.azimuth_deg),
                distance_deg: Some(sr.distance_deg),
                takeoff_angle_deg: Some(sr.takeoff_angle_deg),
                time_residual_s: Some(sr.residual_s),
                time_weight: Some(sr.huber_weight),
            });
        }

        Origin {
            public_id: None,
            time: origin_time,
            latitude: self.lat_deg,
            longitude: self.lon_deg,
            depth_m: self.depth_km * 1_000.0,
            time_uncertainty_s: Some(time_unc_s),
            latitude_uncertainty_km: Some(lat_unc_km),
            longitude_uncertainty_km: Some(lon_unc_km),
            depth_uncertainty_m: Some(dep_unc_m),
            quality: Some(OriginQuality {
                associated_phase_count: Some(self.residuals.len() as u32),
                used_phase_count: Some(used_count),
                standard_error_s: Some(self.weighted_rms_s),
                azimuthal_gap_deg: Some(self.azimuthal_gap_deg),
                minimum_distance_deg: min_d.is_finite().then_some(min_d),
                maximum_distance_deg: (!self.residuals.is_empty()).then_some(max_d),
            }),
            arrivals,
        }
    }
}

struct CompiledPhaseGroup {
    primary: Vec<SeismicPhase>,
    fallback: Vec<SeismicPhase>,
}

struct PreparedDepthPhases {
    dm: TauModel,
    groups: Vec<CompiledPhaseGroup>,
}

#[derive(Clone, Copy)]
struct RayPrediction {
    travel_time_s: f64,
    distance_deg: f64,
    azimuth_deg: f64,
    takeoff_angle_deg: f64,
    ray_param_s_per_rad: f64,
    source_vel_km_s: f64,
}

fn build_depth_phases(
    base_model: &TauModel,
    depth_km: f64,
    unique_labels: &[String],
) -> Result<PreparedDepthPhases> {
    let dm = if (base_model.source_depth - depth_km).abs() == 0.0 {
        base_model.clone()
    } else {
        base_model.with_source_depth(depth_km)?
    };

    let mut groups = Vec::with_capacity(unique_labels.len());
    for label in unique_labels {
        let (prim_names, fb_names): (&[&str], &[&str]) = match label.as_str() {
            "P" => (&["P", "p"], &["Pn", "Pdiff", "PKIKP"]),
            "S" => (&["S", "s"], &["Sn", "Sdiff", "SKS"]),
            _ => (&[], &[]),
        };

        let mut primary = Vec::new();
        let mut fallback = Vec::new();
        if !prim_names.is_empty() {
            for &pn in prim_names {
                if let Ok(sp) = SeismicPhase::new(pn, &dm) {
                    if sp.has_arrivals() {
                        primary.push(sp);
                    }
                }
            }
            for &fnm in fb_names {
                if let Ok(sp) = SeismicPhase::new(fnm, &dm) {
                    if sp.has_arrivals() {
                        fallback.push(sp);
                    }
                }
            }
        } else {
            let sp = SeismicPhase::new(label, &dm)?;
            if sp.has_arrivals() {
                primary.push(sp);
            }
            if label == "p" {
                if let Ok(sp_turn) = SeismicPhase::new("P", &dm) {
                    if sp_turn.has_arrivals() {
                        fallback.push(sp_turn);
                    }
                }
            } else if label == "s" {
                if let Ok(sp_turn) = SeismicPhase::new("S", &dm) {
                    if sp_turn.has_arrivals() {
                        fallback.push(sp_turn);
                    }
                }
            }
        }

        groups.push(CompiledPhaseGroup { primary, fallback });
    }

    Ok(PreparedDepthPhases { dm, groups })
}

#[allow(clippy::too_many_arguments)]
fn predict_single_observation(
    prep: &PreparedDepthPhases,
    group_idx: usize,
    ev_lat_deg: f64,
    ev_lon_deg: f64,
    st_lat_deg: f64,
    st_lon_deg: f64,
    st_elev_m: f64,
    config: &HypocenterConfig<'_>,
) -> Option<RayPrediction> {
    let r_planet = prep.dm.radius_of_planet;
    let sph = gps2dist_azimuth_ellipsoid(
        ev_lat_deg,
        ev_lon_deg,
        st_lat_deg,
        st_lon_deg,
        Ellipsoid {
            a_m: r_planet * 1_000.0,
            f: 0.0,
        },
    );
    let (lookup_dist_deg, base_az_deg) = if config.apply_ellipticity {
        let wgs = try_gps2dist_azimuth(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg).ok()?;
        (
            geocentric_central_angle_deg(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg),
            wgs.azimuth_deg,
        )
    } else {
        (sph.distance_deg_spherical, sph.azimuth_deg)
    };

    let grp = &prep.groups[group_idx];
    fn find_earliest_in_slice<'p>(
        phases: &'p [SeismicPhase],
        dm: &TauModel,
        lookup_dist_deg: f64,
    ) -> Option<(&'p SeismicPhase, RawArrival)> {
        let mut best: Option<(&'p SeismicPhase, RawArrival)> = None;
        for sp in phases {
            if let Ok(arrs) = sp.calc_time(dm, lookup_dist_deg, DEFAULT_RAY_PARAM_TOL) {
                for raw in arrs {
                    if best.as_ref().is_none_or(|(_, b)| raw.time_s < b.time_s) {
                        best = Some((sp, raw));
                    }
                }
            }
        }
        best
    }

    let (sp, raw) = find_earliest_in_slice(&grp.primary, &prep.dm, lookup_dist_deg)
        .or_else(|| find_earliest_in_slice(&grp.fallback, &prep.dm, lookup_dist_deg))?;

    let purist_deg = raw.purist_dist_rad.to_degrees();
    let rem = math::rem_euclid(purist_deg, 360.0);
    let ray_az = if rem > 180.0 {
        math::rem_euclid(base_az_deg + 180.0, 360.0)
    } else {
        math::rem_euclid(base_az_deg, 360.0)
    };

    let mut t_pred = raw.time_s;
    if config.apply_ellipticity {
        if let Ok(dt_el) = arrival_ellipticity_correction_s(sp, &prep.dm, &raw, ev_lat_deg, ray_az)
        {
            t_pred += dt_el;
        }
    }

    // Station topographic elevation static correction (positive elevation -> extra upward leg):
    if st_elev_m.abs() > 1e-3 {
        let is_p_rec = sp.wave_type.last().copied().unwrap_or(true);
        let prop_rec = if is_p_rec {
            MaterialProperty::PVelocity
        } else {
            MaterialProperty::SVelocity
        };
        if let Ok(v_surf) = prep.dm.s_mod.v_mod.evaluate_below(0.0, prop_rec) {
            if v_surf > 0.1 {
                let cos_inc = math::cos(raw.incident_angle_deg.to_radians())
                    .abs()
                    .max(0.2);
                t_pred += (st_elev_m * 1e-3) * cos_inc / v_surf;
            }
        }
    }

    // Optional 3D velocity anomaly path integral:
    if let Some(pert3d) = config.perturbation_3d {
        if let Ok(td_pts) = sp.calc_path_for_arrival(&prep.dm, &raw) {
            let wp_az = if rem > 180.0 {
                math::rem_euclid(sph.azimuth_deg + 180.0, 360.0)
            } else {
                math::rem_euclid(sph.azimuth_deg, 360.0)
            };
            let is_p_start = sp.wave_type.first().copied().unwrap_or(true);
            let mut dt_3d = 0.0;
            for k in 0..td_pts.len().saturating_sub(1) {
                let dt0 = td_pts[k + 1].time - td_pts[k].time;
                if dt0 <= 0.0 {
                    continue;
                }
                let mid_dist = 0.5 * (td_pts[k].dist + td_pts[k + 1].dist);
                let mid_dep = 0.5 * (td_pts[k].depth + td_pts[k + 1].depth);
                let (lat_m, lon_m) =
                    spherical_great_circle_point(ev_lat_deg, ev_lon_deg, wp_az, mid_dist);
                let wtype = if prep.dm.s_mod.depth_in_fluid(mid_dep) || is_p_start {
                    WaveType::P
                } else {
                    WaveType::S
                };
                let dv = pert3d
                    .dv_over_v(lat_m, lon_m, mid_dep, wtype)
                    .clamp(-0.75, 1.5);
                dt_3d += dt0 * (1.0 / (1.0 + dv) - 1.0);
            }
            t_pred += dt_3d;
        }
    }

    // Source-side velocity matching takeoff_angle_deg for exact dT/dz = -cos(iota)/v_src:
    let is_p_src = sp.wave_type.first().copied().unwrap_or(true);
    let prop_src = if is_p_src {
        MaterialProperty::PVelocity
    } else {
        MaterialProperty::SVelocity
    };
    let down_src = sp.down_going.first().copied().unwrap_or(true);
    let v_src = if down_src {
        prep.dm
            .s_mod
            .v_mod
            .evaluate_below(prep.dm.source_depth, prop_src)
            .unwrap_or(8.0)
    } else {
        prep.dm
            .s_mod
            .v_mod
            .evaluate_above(prep.dm.source_depth, prop_src)
            .unwrap_or(8.0)
    }
    .max(0.2);

    Some(RayPrediction {
        travel_time_s: t_pred,
        distance_deg: sph.distance_deg_spherical,
        azimuth_deg: ray_az,
        takeoff_angle_deg: raw.takeoff_angle_deg,
        ray_param_s_per_rad: raw.ray_param_s_per_rad,
        source_vel_km_s: v_src,
    })
}

#[inline]
fn huber_rho(u: f64, k: f64) -> f64 {
    let au = u.abs();
    if au <= k {
        0.5 * u * u
    } else {
        k * (au - 0.5 * k)
    }
}

#[inline]
fn huber_weight(u: f64, k: f64) -> f64 {
    let au = u.abs();
    if au <= k || au == 0.0 {
        1.0
    } else {
        k / au
    }
}

/// Computes a robust origin-time estimate $t_0$ from $(t_i^{\text{obs}} - T_i^{\text{pred}}, \sigma_i)$
/// using exact median initialization followed by 3 scalar Huber-IRLS polish iterations.
fn robust_origin_time(diffs: &[(f64, f64)], huber_k: f64) -> f64 {
    if diffs.is_empty() {
        return 0.0;
    }
    let mut vals: Vec<f64> = diffs.iter().map(|&(d, _)| d).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    let m = vals.len();
    let mut t0 = if m % 2 == 1 {
        vals[m / 2]
    } else {
        0.5 * (vals[m / 2 - 1] + vals[m / 2])
    };

    for _ in 0..3 {
        let mut num = 0.0;
        let mut den = 0.0;
        for &(d, sig) in diffs {
            let u = (d - t0) / sig;
            let w = huber_weight(u, huber_k) / (sig * sig);
            num += w * d;
            den += w;
        }
        if den > 0.0 {
            t0 = num / den;
        }
    }
    t0
}

fn generate_seed_epicenters(observations: &[PhaseArrivalObservation]) -> Vec<(f64, f64)> {
    let mut seeds = Vec::with_capacity(80);

    // 1. 3D unit-vector centroid of all stations:
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for obs in observations {
        let (sp, cp) = math::sin_cos(obs.station_lat_deg.to_radians());
        let (sl, cl) = math::sin_cos(obs.station_lon_deg.to_radians());
        sx += cp * cl;
        sy += cp * sl;
        sz += sp;
    }
    let norm = math::hypot(math::hypot(sx, sy), sz);
    let (cent_lat, cent_lon) = if norm > 1e-9 {
        (
            math::atan2(sz, math::hypot(sx, sy)).to_degrees(),
            math::atan2(sy, sx).to_degrees(),
        )
    } else {
        (
            observations[0].station_lat_deg,
            observations[0].station_lon_deg,
        )
    };
    seeds.push((cent_lat, cent_lon));

    // Concentric rings around network centroid:
    for &ring_deg in &[0.6, 2.0, 6.0, 18.0] {
        let ring_rad = ring_deg * PI / 180.0;
        for k in 0..8 {
            let az = (k as f64) * 45.0;
            seeds.push(spherical_great_circle_point(
                cent_lat, cent_lon, az, ring_rad,
            ));
        }
    }

    // 2. Top-4 earliest-arrival stations (plus small 0.5 deg offset so epicentral distance != 0):
    let mut order: Vec<usize> = (0..observations.len()).collect();
    order.sort_by(|&i, &j| {
        observations[i]
            .arrival_time_s
            .total_cmp(&observations[j].arrival_time_s)
    });
    let mut added_sta = 0usize;
    for idx in order {
        let lat = observations[idx].station_lat_deg;
        let lon = observations[idx].station_lon_deg;
        if seeds
            .iter()
            .all(|&(slat, slon)| (slat - lat).abs() > 0.05 || (slon - lon).abs() > 0.05)
        {
            seeds.push((lat, lon));
            seeds.push(spherical_great_circle_point(
                lat,
                lon,
                37.0,
                0.5_f64.to_radians(),
            ));
            added_sta += 1;
            if added_sta >= 4 {
                break;
            }
        }
    }

    // 3. 32-point golden-angle Fibonacci sphere for global teleseismic coverage:
    let n_fib = 32usize;
    let golden_angle_deg = 180.0 * (3.0 - math::sqrt(5.0));
    for i in 0..n_fib {
        let z = 1.0 - 2.0 * ((i as f64) + 0.5) / (n_fib as f64);
        let lat = math::asin(z.clamp(-1.0, 1.0)).to_degrees();
        let lon = math::rem_euclid((i as f64) * golden_angle_deg + 180.0, 360.0) - 180.0;
        seeds.push((lat, lon));
    }

    seeds
}

/// Damped Huber-IRLS 4D Hypocenter Inverter (`locate_hypocenter`).
///
/// Combines a multi-scale Fibonacci + station-centroid seed search across focal depths with
/// Levenberg-Marquardt / Tikhonov-damped Gauss-Newton Huber-IRLS over
/// `(lat_deg, lon_deg, depth_km, origin_time_s)`, returning analytical $4\times 4$ posterior
/// covariance $\boldsymbol{\Sigma} = \hat{\sigma}^2 (\mathbf{G}^T \mathbf{W} \mathbf{G} + \lambda \mathbf{I}_4)^{-1}$
/// and 3D confidence ellipsoid semi-axes.
pub fn locate_hypocenter(
    model: &TauModel,
    observations: &[PhaseArrivalObservation],
    config: &HypocenterConfig<'_>,
) -> Result<HypocenterSolution> {
    if observations.len() < 4 {
        return Err(EarthSnailError::InvalidParameter(alloc::format!(
            "locate_hypocenter requires at least 4 phase observations to solve for (lat, lon, depth, t0), got {}",
            observations.len()
        )));
    }

    for obs in observations {
        if !obs.station_lat_deg.is_finite()
            || !obs.station_lon_deg.is_finite()
            || obs.station_lat_deg.abs() > 90.0
            || !obs.arrival_time_s.is_finite()
        {
            return Err(EarthSnailError::InvalidParameter(alloc::format!(
                "invalid coordinate or arrival_time_s on station '{}'",
                obs.station_id
            )));
        }
    }

    let min_z = config.min_depth_km.clamp(0.0, model.radius_of_planet);
    let max_z = config
        .max_depth_km
        .clamp(min_z + 1.0, model.radius_of_planet - 10.0);
    let huber_k = config.huber_k.max(0.5);
    let base_lambda = config.damping_lambda.max(1e-12);

    // Deduplicate phase labels so each distinct phase is compiled only once per trial depth!
    let mut unique_labels: Vec<String> = Vec::new();
    let mut obs_group_idx: Vec<usize> = Vec::with_capacity(observations.len());
    for obs in observations {
        let trimmed = obs.phase.trim();
        if trimmed.is_empty() {
            return Err(EarthSnailError::PhaseParse(
                "empty phase name in PhaseArrivalObservation".into(),
            ));
        }
        if let Some(pos) = unique_labels.iter().position(|s| s == trimmed) {
            obs_group_idx.push(pos);
        } else {
            unique_labels.push(trimmed.to_string());
            obs_group_idx.push(unique_labels.len() - 1);
        }
    }

    // Reference pick uncertainty scale sigma_ref (median of valid pick uncertainties):
    let mut sigs: Vec<f64> = observations
        .iter()
        .map(|o| {
            if o.uncertainty_s.is_finite() && o.uncertainty_s > 1e-4 {
                o.uncertainty_s
            } else {
                0.1
            }
        })
        .collect();
    let mut sorted_sigs = sigs.clone();
    sorted_sigs.sort_by(|a, b| a.total_cmp(b));
    let sigma_ref = sorted_sigs[sorted_sigs.len() / 2];
    for s in &mut sigs {
        *s = s.max(1e-4);
    }

    // Stage 1: Coarse 3D + Depth Grid Search (compiled once per depth!)
    // Use fast spherical 1D lookup during coarse seed screening, then full config during IRLS.
    let seed_config = HypocenterConfig {
        apply_ellipticity: config.apply_ellipticity,
        perturbation_3d: None,
        ..*config
    };
    let seed_epicenters = generate_seed_epicenters(observations);
    let raw_depths: [f64; 9] = [2.0, 10.0, 25.0, 50.0, 100.0, 200.0, 350.0, 500.0, 620.0];
    let mut trial_depths: Vec<f64> = Vec::with_capacity(10);
    for &d in &raw_depths {
        let cd = d.clamp(min_z, max_z);
        if trial_depths.iter().all(|&x| (x - cd).abs() > 0.5) {
            trial_depths.push(cd);
        }
    }

    let mut best_seed = (
        seed_epicenters[0].0,
        seed_epicenters[0].1,
        trial_depths[0],
        0.0,
    );
    let mut best_seed_loss = f64::INFINITY;
    let missing_penalty = huber_rho(25.0, huber_k);

    for &z_try in &trial_depths {
        let prep = build_depth_phases(model, z_try, &unique_labels)?;
        let mut diffs = Vec::with_capacity(observations.len());
        for &(lat_try, lon_try) in &seed_epicenters {
            diffs.clear();
            let mut missing = 0usize;
            for (i, obs) in observations.iter().enumerate() {
                if let Some(pred) = predict_single_observation(
                    &prep,
                    obs_group_idx[i],
                    lat_try,
                    lon_try,
                    obs.station_lat_deg,
                    obs.station_lon_deg,
                    obs.station_elevation_m,
                    &seed_config,
                ) {
                    diffs.push((obs.arrival_time_s - pred.travel_time_s, sigs[i]));
                } else {
                    missing += 1;
                }
            }
            if diffs.len() < 4 {
                continue;
            }
            let t0_try = robust_origin_time(&diffs, huber_k);
            let mut loss = (missing as f64) * missing_penalty;
            for &(d, sig) in &diffs {
                loss += huber_rho((d - t0_try) / sig, huber_k);
            }
            if loss < best_seed_loss {
                best_seed_loss = loss;
                best_seed = (lat_try, lon_try, z_try, t0_try);
            }
        }
    }

    // Refine local 3x3x3 grid around best_seed (+/- 1.5 deg, +/- 20 km) so deep/crustal seeds
    // start right inside the quadratic Gauss-Newton basin!
    let (s_lat, s_lon, s_z, _) = best_seed;
    for &dz in &[-20.0, -7.0, 0.0, 7.0, 20.0] {
        let z_try = (s_z + dz).clamp(min_z, max_z);
        let prep = build_depth_phases(model, z_try, &unique_labels)?;
        let mut local_pts = vec![(s_lat, s_lon)];
        for &r_deg in &[0.4, 1.2] {
            for k in 0..8 {
                local_pts.push(spherical_great_circle_point(
                    s_lat,
                    s_lon,
                    (k as f64) * 45.0,
                    r_deg * PI / 180.0,
                ));
            }
        }
        let mut diffs = Vec::with_capacity(observations.len());
        for (lat_try, lon_try) in local_pts {
            diffs.clear();
            let mut missing = 0usize;
            for (i, obs) in observations.iter().enumerate() {
                if let Some(pred) = predict_single_observation(
                    &prep,
                    obs_group_idx[i],
                    lat_try,
                    lon_try,
                    obs.station_lat_deg,
                    obs.station_lon_deg,
                    obs.station_elevation_m,
                    config,
                ) {
                    diffs.push((obs.arrival_time_s - pred.travel_time_s, sigs[i]));
                } else {
                    missing += 1;
                }
            }
            if diffs.len() < 4 {
                continue;
            }
            let t0_try = robust_origin_time(&diffs, huber_k);
            let mut loss = (missing as f64) * missing_penalty;
            for &(d, sig) in &diffs {
                loss += huber_rho((d - t0_try) / sig, huber_k);
            }
            if loss < best_seed_loss {
                best_seed_loss = loss;
                best_seed = (lat_try, lon_try, z_try, t0_try);
            }
        }
    }

    // Stage 2: Damped Gauss-Newton / Levenberg-Marquardt Huber-IRLS
    let (mut cur_lat, mut cur_lon, mut cur_depth, mut cur_t0) = best_seed;
    let mut lm_lambda = base_lambda;
    let mut cur_prep = build_depth_phases(model, cur_depth, &unique_labels)?;
    let r_planet = model.radius_of_planet;

    let eval_state = |prep: &PreparedDepthPhases,
                      lat: f64,
                      lon: f64,
                      t0: f64|
     -> (f64, Vec<Option<RayPrediction>>) {
        let mut preds = Vec::with_capacity(observations.len());
        let mut loss = 0.0;
        for (i, obs) in observations.iter().enumerate() {
            if let Some(p) = predict_single_observation(
                prep,
                obs_group_idx[i],
                lat,
                lon,
                obs.station_lat_deg,
                obs.station_lon_deg,
                obs.station_elevation_m,
                config,
            ) {
                let res = obs.arrival_time_s - (t0 + p.travel_time_s);
                loss += huber_rho(res / sigs[i], huber_k);
                preds.push(Some(p));
            } else {
                loss += missing_penalty;
                preds.push(None);
            }
        }
        (loss, preds)
    };

    let (mut cur_loss, mut cur_preds) = eval_state(&cur_prep, cur_lat, cur_lon, cur_t0);
    let mut iter_executed = 0usize;

    for iter in 0..config.max_iterations {
        iter_executed = iter + 1;

        // Assemble G^T W G (4x4) and G^T W r (4x1) in normalized residual weights:
        // W_ii = w_H(r_i / sig_i) * (sigma_ref / sig_i)^2
        let mut gtwg = [[0.0_f64; 4]; 4];
        let mut gtwr = [0.0_f64; 4];
        let mut valid_count = 0usize;

        for (i, obs) in observations.iter().enumerate() {
            let Some(pred) = cur_preds[i] else {
                continue;
            };
            valid_count += 1;
            let res = obs.arrival_time_s - (cur_t0 + pred.travel_time_s);
            let wh = huber_weight(res / sigs[i], huber_k);
            let rel = sigma_ref / sigs[i];
            let w = wh * rel * rel;

            let (saz, caz) = math::sin_cos(pred.azimuth_deg.to_radians());
            let g0 = -pred.ray_param_s_per_rad * caz / r_planet; // dT / dx_N (s/km)
            let g1 = -pred.ray_param_s_per_rad * saz / r_planet; // dT / dx_E (s/km)
            let g2 = -math::cos(pred.takeoff_angle_deg.to_radians()) / pred.source_vel_km_s; // dT / dz (s/km)
            let g3 = 1.0_f64; // dt_pred / dt0

            let g = [g0, g1, g2, g3];
            for r in 0..4 {
                gtwr[r] += w * g[r] * res;
                for c in 0..4 {
                    gtwg[r][c] += w * g[r] * g[c];
                }
            }
        }

        if valid_count < 4 {
            return Err(EarthSnailError::Numerical(
                "fewer than 4 valid predicted phase arrivals at current hypocenter".into(),
            ));
        }

        // Try Levenberg-Marquardt step (increasing damping if crossing a discontinuity overshoots):
        let mut accepted = false;
        let mut step_norm = 0.0_f64;

        for _sub in 0..8 {
            let mut a = gtwg;
            for (d, row) in a.iter_mut().enumerate() {
                row[d] += lm_lambda;
            }
            let Some(dm) = solve_4x4(a, gtwr) else {
                lm_lambda *= 10.0;
                continue;
            };

            // Cap single-iteration spatial jump to 250 km so global rays stay smoothly tracked:
            let horiz_km = math::hypot(dm[0], dm[1]);
            let full_spatial = math::hypot(horiz_km, dm[2]);
            let scale = if full_spatial > 250.0 {
                250.0 / full_spatial
            } else {
                1.0
            };
            let dn = dm[0] * scale;
            let de = dm[1] * scale;
            let dz = dm[2] * scale;
            let dt0 = dm[3] * scale;

            let step_h = math::hypot(dn, de);
            let step_az = math::atan2(de, dn).to_degrees();
            let (try_lat, try_lon) =
                spherical_great_circle_point(cur_lat, cur_lon, step_az, step_h / r_planet);
            let try_depth = (cur_depth + dz).clamp(min_z, max_z);
            let try_t0 = cur_t0 + dt0;

            let try_prep = if (try_depth - cur_depth).abs() == 0.0 {
                cur_prep.dm.clone()
            } else {
                let p = build_depth_phases(model, try_depth, &unique_labels)?;
                let (try_loss, try_preds) = eval_state(&p, try_lat, try_lon, try_t0);
                if try_loss <= cur_loss + 1e-12 * (1.0 + cur_loss) {
                    step_norm = math::sqrt(
                        dn * dn
                            + de * de
                            + (try_depth - cur_depth) * (try_depth - cur_depth)
                            + dt0 * dt0,
                    );
                    cur_lat = try_lat;
                    cur_lon = try_lon;
                    cur_depth = try_depth;
                    cur_t0 = try_t0;
                    cur_prep = p;
                    cur_loss = try_loss;
                    cur_preds = try_preds;
                    lm_lambda = (lm_lambda * 0.35).max(base_lambda * 0.01);
                    accepted = true;
                    break;
                } else {
                    lm_lambda *= 5.0;
                    continue;
                }
            };
            let _ = try_prep;
            let (try_loss, try_preds) = eval_state(&cur_prep, try_lat, try_lon, try_t0);
            if try_loss <= cur_loss + 1e-12 * (1.0 + cur_loss) {
                step_norm = math::sqrt(dn * dn + de * de + dt0 * dt0);
                cur_lat = try_lat;
                cur_lon = try_lon;
                cur_t0 = try_t0;
                cur_loss = try_loss;
                cur_preds = try_preds;
                lm_lambda = (lm_lambda * 0.35).max(base_lambda * 0.01);
                accepted = true;
                break;
            } else {
                lm_lambda *= 5.0;
            }
        }

        if !accepted || step_norm < config.step_tol {
            break;
        }
    }

    // Stage 3: Final Analytical 4x4 Covariance Matrix & 3D Confidence Ellipsoid
    let mut gtwg_final = [[0.0_f64; 4]; 4];
    let mut sum_w_r2 = 0.0_f64;
    let mut sum_wh = 0.0_f64;
    let mut sum_w_norm = 0.0_f64;
    let mut residuals = Vec::with_capacity(observations.len());
    let mut azimuths = Vec::with_capacity(observations.len());

    for (i, obs) in observations.iter().enumerate() {
        let Some(pred) = cur_preds[i] else {
            continue;
        };
        let pred_arr = cur_t0 + pred.travel_time_s;
        let res = obs.arrival_time_s - pred_arr;
        let wh = huber_weight(res / sigs[i], huber_k);
        let rel = sigma_ref / sigs[i];
        let w = wh * rel * rel;

        sum_w_r2 += w * res * res;
        sum_wh += wh;
        sum_w_norm += w;
        azimuths.push(pred.azimuth_deg);

        let (saz, caz) = math::sin_cos(pred.azimuth_deg.to_radians());
        let g = [
            -pred.ray_param_s_per_rad * caz / r_planet,
            -pred.ray_param_s_per_rad * saz / r_planet,
            -math::cos(pred.takeoff_angle_deg.to_radians()) / pred.source_vel_km_s,
            1.0,
        ];
        for r in 0..4 {
            for c in 0..4 {
                gtwg_final[r][c] += w * g[r] * g[c];
            }
        }

        residuals.push(StationResidual {
            station_id: obs.station_id.clone(),
            phase: obs.phase.clone(),
            distance_deg: pred.distance_deg,
            azimuth_deg: pred.azimuth_deg,
            takeoff_angle_deg: pred.takeoff_angle_deg,
            predicted_time_s: pred_arr,
            residual_s: res,
            huber_weight: wh,
        });
    }

    let weighted_rms_s = if sum_w_norm > 0.0 {
        math::sqrt(sum_w_r2 / sum_w_norm)
    } else {
        0.0
    };
    let dof = (sum_wh - 4.0).max(1.0);
    // Combine empirical posterior variance with a positive a-priori floor so noise-free synthetic
    // inversions still produce a strictly positive-definite covariance matrix:
    let aposteriori_s2 = sum_w_r2 / dof;
    let floor_sd = sigma_ref * 0.1;
    let floor_s2 = (floor_sd * floor_sd).max(1e-8);
    let variance_scale_s2 = aposteriori_s2.max(floor_s2);

    let mut reg_normal = gtwg_final;
    for (d, row) in reg_normal.iter_mut().enumerate() {
        row[d] += base_lambda;
    }
    let inv_normal = invert_4x4(reg_normal).ok_or_else(|| {
        EarthSnailError::Numerical("singular damped normal matrix in locate_hypocenter".into())
    })?;

    let mut covariance_nedt = [[0.0_f64; 4]; 4];
    for (r, cov_row) in covariance_nedt.iter_mut().enumerate() {
        for (c, cell) in cov_row.iter_mut().enumerate() {
            *cell = variance_scale_s2 * inv_normal[r][c];
        }
    }
    // Enforce exact matrix symmetry:
    #[allow(clippy::needless_range_loop)]
    for r in 0..4 {
        for c in (r + 1)..4 {
            let sym = 0.5 * (covariance_nedt[r][c] + covariance_nedt[c][r]);
            covariance_nedt[r][c] = sym;
            covariance_nedt[c][r] = sym;
        }
    }

    let cov3 = [
        [
            covariance_nedt[0][0],
            covariance_nedt[0][1],
            covariance_nedt[0][2],
        ],
        [
            covariance_nedt[1][0],
            covariance_nedt[1][1],
            covariance_nedt[1][2],
        ],
        [
            covariance_nedt[2][0],
            covariance_nedt[2][1],
            covariance_nedt[2][2],
        ],
    ];
    let ellipsoid = compute_confidence_ellipsoid_3x3(cov3);
    let azimuthal_gap_deg = compute_azimuthal_gap(&mut azimuths);

    Ok(HypocenterSolution {
        lat_deg: cur_lat,
        lon_deg: cur_lon,
        depth_km: cur_depth,
        origin_time_s: cur_t0,
        weighted_rms_s,
        variance_scale_s2,
        covariance_nedt,
        ellipsoid,
        iterations: iter_executed,
        azimuthal_gap_deg,
        residuals,
    })
}

fn compute_azimuthal_gap(azimuths: &mut [f64]) -> f64 {
    if azimuths.is_empty() {
        return 360.0;
    }
    azimuths.sort_by(|a, b| a.total_cmp(b));
    let mut max_gap = 0.0_f64;
    for k in 0..(azimuths.len() - 1) {
        let gap = azimuths[k + 1] - azimuths[k];
        if gap > max_gap {
            max_gap = gap;
        }
    }
    let wrap_gap = (azimuths[0] + 360.0) - azimuths[azimuths.len() - 1];
    max_gap.max(wrap_gap)
}

/// Solves $4\times 4$ linear system $A x = b$ via Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve_4x4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let mut piv = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..4 {
            let v = a[row][col].abs();
            if v > max_val {
                max_val = v;
                piv = row;
            }
        }
        if max_val < 1e-18 || !max_val.is_finite() {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let inv_piv = 1.0 / a[col][col];
        for row in (col + 1)..4 {
            let factor = a[row][col] * inv_piv;
            a[row][col] = 0.0;
            for k in (col + 1)..4 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0_f64; 4];
    for row in (0..4).rev() {
        let mut sum = b[row];
        for k in (row + 1)..4 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

/// Computes the exact $4\times 4$ matrix inverse $A^{-1}$ by solving against the 4 canonical unit vectors.
fn invert_4x4(a: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut inv = [[0.0_f64; 4]; 4];
    for col in 0..4 {
        let mut e = [0.0_f64; 4];
        e[col] = 1.0;
        let x = solve_4x4(a, e)?;
        for row in 0..4 {
            inv[row][col] = x[row];
        }
    }
    Some(inv)
}

/// Cyclic Jacobi eigensolver for a real symmetric $3\times 3$ covariance matrix, returning sorted
/// $1\sigma$ and 95% confidence ellipsoid semi-axes and orthonormal principal axes.
fn compute_confidence_ellipsoid_3x3(mut d: [[f64; 3]; 3]) -> ConfidenceEllipsoid {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

    for _sweep in 0..24 {
        let off = d[0][1].abs() + d[0][2].abs() + d[1][2].abs();
        if off <= 1e-15 * (d[0][0].abs() + d[1][1].abs() + d[2][2].abs()) {
            break;
        }
        for &(p, q) in &PAIRS {
            let apq = d[p][q];
            if apq.abs() <= 1e-18 {
                continue;
            }
            let diff = d[q][q] - d[p][p];
            let t = if diff.abs() > 1e15 * apq.abs() {
                apq / diff
            } else {
                let tau = diff / (2.0 * apq);
                let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                sign / (tau.abs() + math::sqrt(1.0 + tau * tau))
            };
            let c = 1.0 / math::sqrt(1.0 + t * t);
            let s = t * c;

            let dpp = d[p][p] - t * apq;
            let dqq = d[q][q] + t * apq;
            d[p][p] = dpp;
            d[q][q] = dqq;
            d[p][q] = 0.0;
            d[q][p] = 0.0;

            let r = 3 - p - q;
            let dpr = c * d[p][r] - s * d[q][r];
            let dqr = s * d[p][r] + c * d[q][r];
            d[p][r] = dpr;
            d[r][p] = dpr;
            d[q][r] = dqr;
            d[r][q] = dqr;

            for v_row in &mut v {
                let v_rp = c * v_row[p] - s * v_row[q];
                let v_rq = s * v_row[p] + c * v_row[q];
                v_row[p] = v_rp;
                v_row[q] = v_rq;
            }
        }
    }

    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| d[j][j].total_cmp(&d[i][i]));

    // sqrt(chi2_3(0.95)) = sqrt(7.814727903251179) = 2.795483486501251
    const CHI2_3_95_SQRT: f64 = 2.795_483_486_501_251;
    let mut semi_1s = [0.0_f64; 3];
    let mut semi_95 = [0.0_f64; 3];
    let mut axes = [[0.0_f64; 3]; 3];

    for (out_idx, &eig_idx) in order.iter().enumerate() {
        let s1 = math::sqrt(d[eig_idx][eig_idx].max(0.0));
        semi_1s[out_idx] = s1;
        semi_95[out_idx] = s1 * CHI2_3_95_SQRT;
        axes[out_idx] = [v[0][eig_idx], v[1][eig_idx], v[2][eig_idx]];
    }

    ConfidenceEllipsoid {
        semi_axes_1sigma_km: semi_1s,
        semi_axes_95pct_km: semi_95,
        principal_axes_ned: axes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subduction_plume_3d_and_pseudo_bending_reduce_travel_time() {
        let model = crate::tau_model::test_iasp91_model();
        let pert = SubductionPlumeModel3D::pacific_ring_default();

        // Fast slab core is positive dv/v, plume core is negative dv/v:
        let dv_slab = pert.dv_over_v(38.5, 141.5, 120.0, WaveType::P);
        let dv_plume = pert.dv_over_v(19.5, -155.5, 200.0, WaveType::S);
        assert!(dv_slab > 0.01, "expected fast slab > +1%, got {dv_slab}");
        assert!(
            dv_plume < -0.02,
            "expected slow plume < -2%, got {dv_plume}"
        );

        let paths = model
            .get_ray_paths_geo(36.0, 143.0, 60.0, 41.0, 139.0, &["P"], false)
            .unwrap();
        assert_eq!(paths.len(), 1);
        let res = perturb_ray_path_3d(&model, &paths[0], &pert, 4).unwrap();
        assert!(
            res.time_bent_3d_s <= res.time_fermat_3d_s + 1e-12,
            "pseudo-bending must not increase Fermat time: {} vs {}",
            res.time_bent_3d_s,
            res.time_fermat_3d_s
        );
    }

    #[test]
    fn locate_hypocenter_recovers_deep_source_and_rejects_glitches() {
        let model = crate::tau_model::test_iasp91_model();
        let true_lat = 37.85;
        let true_lon = 141.65;
        let true_depth = 142.5; // non-10 km intermediate-depth Wadati-Benioff focus!
        let true_t0 = 125.4;

        let stations = [
            ("STA1", 38.10, 141.20),
            ("STA2", 36.20, 140.10),
            ("STA3", 39.70, 141.95),
            ("STA4", 35.65, 144.20),
            ("STA5", 41.80, 143.10),
            ("STA6", 33.50, 137.80),
            ("STA7", 43.20, 135.40),
            ("STA8", 27.10, 142.20),
        ];
        let active_stations = if cfg!(miri) {
            &stations[..4]
        } else {
            &stations[..]
        };

        let mut obs = Vec::new();
        for &(id, slat, slon) in active_stations {
            for &(label, query) in &[("P", &["P", "p"][..]), ("S", &["S", "s"][..])] {
                let arrs = model
                    .get_travel_times_geo(true_lat, true_lon, true_depth, slat, slon, query, false)
                    .unwrap();
                if let Some(a) = arrs.first() {
                    obs.push(PhaseArrivalObservation::new(
                        id,
                        slat,
                        slon,
                        label,
                        true_t0 + a.time_s,
                        0.05,
                    ));
                }
            }
        }
        // Add teleseismic pP depth phase at 45 deg for rock-solid focal depth constraint:
        let pp_arrs = model
            .get_travel_times_geo(
                true_lat,
                true_lon,
                true_depth,
                68.0,
                -150.0,
                &["pP", "P"],
                false,
            )
            .unwrap();
        for a in pp_arrs {
            obs.push(PhaseArrivalObservation::new(
                "TEL1",
                68.0,
                -150.0,
                a.name,
                true_t0 + a.time_s,
                0.05,
            ));
        }

        let cfg = if cfg!(miri) {
            HypocenterConfig {
                min_depth_km: 135.0,
                max_depth_km: 150.0,
                max_iterations: 6,
                ..HypocenterConfig::default()
            }
        } else {
            HypocenterConfig::default()
        };
        let sol_clean = locate_hypocenter(&model, &obs, &cfg).unwrap();
        let horiz_err_km = great_circle_chord_km(
            true_lat,
            true_lon,
            sol_clean.lat_deg,
            sol_clean.lon_deg,
            6371.0,
        );
        let depth_err_km = (sol_clean.depth_km - true_depth).abs();
        let total_err_km = math::hypot(horiz_err_km, depth_err_km);
        assert!(
            total_err_km < 0.01,
            "noise-free 3D hypocenter error {total_err_km} km >= 0.01 km (sol={sol_clean:?})"
        );
        assert!((sol_clean.origin_time_s - true_t0).abs() < 1e-3);
        assert!(sol_clean.ellipsoid.semi_axes_1sigma_km[2] > 0.0);

        // Inject two massive clock-glitch outliers (+18.0 s and -14.0 s):
        if !cfg!(miri) {
            let mut glitch_obs = obs.clone();
            glitch_obs[2].arrival_time_s += 18.0;
            glitch_obs[9].arrival_time_s -= 14.0;
            let sol_glitch = locate_hypocenter(&model, &glitch_obs, &cfg).unwrap();
            let h_err = great_circle_chord_km(
                true_lat,
                true_lon,
                sol_glitch.lat_deg,
                sol_glitch.lon_deg,
                6371.0,
            );
            let z_err = (sol_glitch.depth_km - true_depth).abs();
            let tot_glitch_err = math::hypot(h_err, z_err);
            assert!(
                tot_glitch_err < 1.0,
                "Huber-IRLS hypocenter error under +/-18 s glitches was {tot_glitch_err} km"
            );
            assert!(
                sol_glitch.residuals[2].huber_weight < 0.02,
                "expected outlier weight < 0.02, got {}",
                sol_glitch.residuals[2].huber_weight
            );
        }
    }
}
