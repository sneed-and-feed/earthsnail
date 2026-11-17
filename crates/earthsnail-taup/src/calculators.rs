//! High-level TauP calculators: `get_travel_times`, `get_travel_times_geo`, `get_pierce_points`,
//! `get_pierce_points_geo`, `get_ray_paths`, `get_ray_paths_geo`, `get_travel_time_curves`, and
//! [`TauEngine`] with source-depth phase caching.

use crate::ellipticity::arrival_ellipticity_correction_s;
use crate::phase::{
    parse_phase_list, RawArrival, SeismicPhase, DEFAULT_PATH_RAY_PARAM_TOL, DEFAULT_RAY_PARAM_TOL,
};
use crate::tau_model::{TauModel, TimeDist};
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::f64::consts::PI;
use earthsnail_core::geodesics::{
    geocentric_central_angle_deg, gps2dist_azimuth_ellipsoid, try_gps2dist_azimuth, Ellipsoid,
};
use earthsnail_core::{math, EarthSnailError, Result};

/// High-level seismic phase arrival returned by `get_travel_times` and `get_travel_times_geo`.
#[derive(Debug, Clone, PartialEq)]
pub struct SeismicArrival {
    /// Requested phase name (e.g. `"P"`, `"PKP"`, `"P410s"`, `"3.5kmps"`).
    pub name: String,
    /// Discontinuity-snapped purist phase name (e.g. `"P400.0s"` on `prem`).
    pub purist_name: String,
    /// Arrival travel time in seconds (including WGS84 ellipticity correction when requested).
    pub time_s: f64,
    /// Spherical ray parameter $p$ in `s/rad`.
    pub ray_param_s_per_rad: f64,
    /// Spherical ray parameter in `s/deg` (`p * pi / 180`).
    pub ray_param_s_per_deg: f64,
    /// Takeoff angle at source measured from downward vertical in `[0, 180]` degrees.
    pub takeoff_angle_deg: f64,
    /// Incident angle at receiver measured from vertical in `[0, 180]` degrees.
    pub incident_angle_deg: f64,
    /// Requested epicentral distance in degrees.
    pub distance_deg: f64,
    /// True unwrapped great-circle travel distance in degrees (`purist_dist_rad * 180 / pi`).
    pub purist_distance_deg: f64,
    /// Source focal depth in kilometers.
    pub source_depth_km: f64,
    /// Receiver burial depth in kilometers.
    pub receiver_depth_km: f64,
    /// Ray departure azimuth clockwise from North in `[0, 360)` degrees (populated by `*_geo`).
    pub azimuth_deg: Option<f64>,
    /// Station-to-source back-azimuth in `[0, 360)` degrees (populated by `*_geo`).
    pub back_azimuth_deg: Option<f64>,
    /// Applied Dziewonski-Gilbert WGS84 ellipticity correction $\delta t_{\text{ellip}}$ in seconds.
    pub ellipticity_corr_s: Option<f64>,
}

/// Discontinuity or turning-point pierce waypoint along an arrival's ray path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PiercePoint {
    /// Spherical ray parameter $p$ in `s/rad`.
    pub p_s_per_rad: f64,
    /// Cumulative travel time to this pierce point in seconds.
    pub time_s: f64,
    /// Cumulative angular distance from source in radians.
    pub dist_rad: f64,
    /// Cumulative angular distance from source in degrees.
    pub dist_deg: f64,
    /// Depth below free surface in kilometers.
    pub depth_km: f64,
    /// Optional geographic latitude in `[-90, 90]` degrees along the great circle.
    pub lat_deg: Option<f64>,
    /// Optional geographic longitude in `[-180, 180]` degrees along the great circle.
    pub lon_deg: Option<f64>,
}

/// Phase arrival paired with its ordered sequence of discontinuity & turning [`PiercePoint`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrivalPierce {
    /// Underlying phase arrival metadata.
    pub arrival: SeismicArrival,
    /// Ordered pierce points from source (`time_s = 0`) to receiver.
    pub pierce_points: Vec<PiercePoint>,
}

/// Dense trajectory sample along a 1D spherical ray path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayPathPoint {
    /// Spherical ray parameter $p$ in `s/rad`.
    pub p_s_per_rad: f64,
    /// Cumulative travel time from source in seconds.
    pub time_s: f64,
    /// Cumulative angular distance from source in radians.
    pub dist_rad: f64,
    /// Cumulative angular distance from source in degrees.
    pub dist_deg: f64,
    /// Depth below free surface in kilometers.
    pub depth_km: f64,
    /// Spherical radius $r = R_E - z$ in kilometers.
    pub radius_km: f64,
    /// Optional geographic latitude in `[-90, 90]` degrees along the great circle.
    pub lat_deg: Option<f64>,
    /// Optional geographic longitude in `[-180, 180]` degrees along the great circle.
    pub lon_deg: Option<f64>,
}

/// Phase arrival paired with its dense spherical [`RayPathPoint`] trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrivalPath {
    /// Underlying phase arrival metadata.
    pub arrival: SeismicArrival,
    /// Ordered ray-path trajectory samples from source to receiver.
    pub path_points: Vec<RayPathPoint>,
}

/// Sample point on a continuous parametric $\tau(p)$ / $T(\Delta)$ branch curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TauPCurvePoint {
    /// Spherical ray parameter $p$ in `s/rad`.
    pub ray_param_s_per_rad: f64,
    /// Delay time / intercept $\tau(p) = T(p) - p\Delta(p)$ in seconds.
    pub tau_s: f64,
    /// Travel time $T(p)$ in seconds.
    pub time_s: f64,
    /// Unwrapped angular distance $\Delta(p)$ in radians.
    pub dist_rad: f64,
    /// Unwrapped angular distance in degrees.
    pub dist_deg: f64,
}

/// Continuous sub-branch of a phase's travel-time & $\tau(p)$ curve (split across LVZ shadow zones).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseCurveBranch {
    /// Sub-branch index (`0`, `1`, ... across any low-velocity shadow zones).
    pub branch_index: usize,
    /// Ordered parametric $(p, \tau(p), T(p), \Delta(p))$ curve samples.
    pub points: Vec<TauPCurvePoint>,
}

/// Complete $T(\Delta)$ and $\tau(p)$ branch curve representation for one compiled seismic phase.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTravelTimeCurve {
    /// Requested phase name.
    pub name: String,
    /// Discontinuity-snapped purist phase name.
    pub purist_name: String,
    /// Source focal depth in kilometers.
    pub source_depth_km: f64,
    /// Minimum valid ray parameter (`s/rad`).
    pub min_ray_param_s_per_rad: f64,
    /// Maximum valid ray parameter (`s/rad`).
    pub max_ray_param_s_per_rad: f64,
    /// Minimum observable distance in degrees.
    pub min_distance_deg: f64,
    /// Maximum observable distance in degrees.
    pub max_distance_deg: f64,
    /// Continuous $\tau(p)$ / $T(\Delta)$ sub-branches.
    pub branches: Vec<PhaseCurveBranch>,
}

#[inline]
fn raw_into_seismic_arrival(raw: RawArrival) -> SeismicArrival {
    SeismicArrival {
        name: raw.name,
        purist_name: raw.purist_name,
        time_s: raw.time_s,
        ray_param_s_per_rad: raw.ray_param_s_per_rad,
        ray_param_s_per_deg: raw.ray_param_s_per_rad * (PI / 180.0),
        takeoff_angle_deg: raw.takeoff_angle_deg,
        incident_angle_deg: raw.incident_angle_deg,
        distance_deg: raw.distance_deg,
        purist_distance_deg: raw.purist_dist_rad.to_degrees(),
        source_depth_km: raw.source_depth_km,
        receiver_depth_km: raw.receiver_depth_km,
        azimuth_deg: None,
        back_azimuth_deg: None,
        ellipticity_corr_s: None,
    }
}

#[inline]
fn raw_to_seismic_arrival(raw: &RawArrival) -> SeismicArrival {
    raw_into_seismic_arrival(raw.clone())
}

#[inline]
fn arrival_cmp(a: &SeismicArrival, b: &SeismicArrival) -> core::cmp::Ordering {
    a.time_s
        .total_cmp(&b.time_s)
        .then_with(|| a.name.cmp(&b.name))
        .then_with(|| b.ray_param_s_per_rad.total_cmp(&a.ray_param_s_per_rad))
}

#[inline]
fn wrap_deg_360(deg: f64) -> f64 {
    let r = math::rem_euclid(deg, 360.0);
    if r == 360.0 {
        0.0
    } else {
        r
    }
}

#[inline]
fn wrap_lon_180(lon_deg: f64) -> f64 {
    let shifted = math::rem_euclid(lon_deg + 180.0, 360.0) - 180.0;
    if shifted == -180.0 && lon_deg > 0.0 {
        180.0
    } else {
        shifted
    }
}

/// Pole-safe, multi-wrap-exact spherical great-circle direct position starting from
/// `(lat1_deg, lon1_deg)` along `azimuth_deg` clockwise from North for `dist_rad` radians.
#[inline]
pub fn spherical_great_circle_point(
    lat1_deg: f64,
    lon1_deg: f64,
    azimuth_deg: f64,
    dist_rad: f64,
) -> (f64, f64) {
    if dist_rad == 0.0 {
        return (lat1_deg.clamp(-90.0, 90.0), wrap_lon_180(lon1_deg));
    }
    let phi1 = lat1_deg.clamp(-90.0, 90.0).to_radians();
    let lam1 = lon1_deg.to_radians();
    let az = azimuth_deg.to_radians();

    let (sphi1, cphi1) = math::sin_cos(phi1);
    let (slam1, clam1) = math::sin_cos(lam1);
    let (saz, caz) = math::sin_cos(az);
    let (sd, cd) = math::sin_cos(dist_rad);

    // Orthonormal frame (u0, e_N, e_E) valid everywhere including exact +/-90 deg poles.
    let u0 = [cphi1 * clam1, cphi1 * slam1, sphi1];
    let e_n = [-sphi1 * clam1, -sphi1 * slam1, cphi1];
    let e_e = [-slam1, clam1, 0.0];
    let t0 = [
        caz * e_n[0] + saz * e_e[0],
        caz * e_n[1] + saz * e_e[1],
        caz * e_n[2],
    ];

    let px = cd * u0[0] + sd * t0[0];
    let py = cd * u0[1] + sd * t0[1];
    let pz = cd * u0[2] + sd * t0[2];

    let lat_deg = math::atan2(pz, math::hypot(px, py)).to_degrees();
    let lon_deg = wrap_lon_180(math::atan2(py, px).to_degrees());
    (lat_deg, lon_deg)
}

#[inline]
fn major_arc_departure_azimuth(minor_azimuth_deg: f64, purist_distance_deg: f64) -> f64 {
    let rem = math::rem_euclid(purist_distance_deg, 360.0);
    if rem > 180.0 {
        wrap_deg_360(minor_azimuth_deg + 180.0)
    } else {
        wrap_deg_360(minor_azimuth_deg)
    }
}

#[inline]
fn major_arc_back_azimuth(minor_back_azimuth_deg: f64, purist_distance_deg: f64) -> f64 {
    let rem = math::rem_euclid(purist_distance_deg, 360.0);
    if rem > 180.0 {
        wrap_deg_360(minor_back_azimuth_deg + 180.0)
    } else {
        wrap_deg_360(minor_back_azimuth_deg)
    }
}

impl TauModel {
    fn prepare_depth_model(
        &self,
        source_depth_km: f64,
        receiver_depth_km: f64,
    ) -> Result<Cow<'_, TauModel>> {
        if !source_depth_km.is_finite()
            || source_depth_km < 0.0
            || source_depth_km > self.radius_of_planet
        {
            return Err(EarthSnailError::InvalidParameter(alloc::format!(
                "source depth {source_depth_km} km outside [0, {}]",
                self.radius_of_planet
            )));
        }
        if !receiver_depth_km.is_finite()
            || receiver_depth_km < 0.0
            || receiver_depth_km > self.radius_of_planet
        {
            return Err(EarthSnailError::InvalidParameter(alloc::format!(
                "receiver depth {receiver_depth_km} km outside [0, {}]",
                self.radius_of_planet
            )));
        }

        let mut model: Cow<'_, TauModel> = if (self.source_depth - source_depth_km).abs() == 0.0 {
            Cow::Borrowed(self)
        } else if self.source_depth == 0.0 {
            Cow::Owned(self.with_source_depth(source_depth_km)?)
        } else {
            return Err(EarthSnailError::ModelError(alloc::format!(
                "TauModel already depth-corrected to {} km; query {} km on a surface-source model",
                self.source_depth,
                source_depth_km
            )));
        };

        if receiver_depth_km > 0.0 {
            let has_rec_boundary = model.tau_branches[0]
                .iter()
                .any(|tb| tb.top_depth == receiver_depth_km || tb.bot_depth == receiver_depth_km);
            if !has_rec_boundary {
                let mut split = model.split_branch(receiver_depth_km)?;
                split.source_branch = split.find_branch(source_depth_km)?;
                model = Cow::Owned(split);
            }
        }

        Ok(model)
    }

    /// Computes multi-phase spherical travel times sorted by arrival `time_s` (`receiver_depth = 0`).
    pub fn get_travel_times<S: AsRef<str>>(
        &self,
        depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<SeismicArrival>> {
        self.get_travel_times_with_receiver_depth(depth_km, 0.0, distance_deg, phases)
    }

    /// Computes multi-phase spherical travel times for an arbitrary `receiver_depth_km` (`>= 0`).
    pub fn get_travel_times_with_receiver_depth<S: AsRef<str>>(
        &self,
        depth_km: f64,
        receiver_depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<SeismicArrival>> {
        if !distance_deg.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite epicentral distance_deg".into(),
            ));
        }
        let dm = self.prepare_depth_model(depth_km, receiver_depth_km)?;
        let phase_names = parse_phase_list(phases);
        let mut out = Vec::new();
        for ph_name in &phase_names {
            let sp = SeismicPhase::with_receiver_depth(ph_name, &dm, receiver_depth_km)?;
            for raw in sp.calc_time(&dm, distance_deg, DEFAULT_RAY_PARAM_TOL)? {
                out.push(raw_to_seismic_arrival(&raw));
            }
        }
        out.sort_by(arrival_cmp);
        Ok(out)
    }

    /// Computes geographic source-to-station travel times, azimuth/back-azimuth, and optional
    /// Dziewonski & Gilbert (1976) WGS84 ellipticity corrections (`apply_ellipticity = true`).
    #[allow(clippy::too_many_arguments)]
    pub fn get_travel_times_geo<S: AsRef<str>>(
        &self,
        ev_lat_deg: f64,
        ev_lon_deg: f64,
        ev_depth_km: f64,
        st_lat_deg: f64,
        st_lon_deg: f64,
        phases: &[S],
        apply_ellipticity: bool,
    ) -> Result<Vec<SeismicArrival>> {
        let wgs_geo = try_gps2dist_azimuth(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg)?;
        let (lookup_dist_deg, base_az_deg, base_baz_deg) = if apply_ellipticity {
            (
                geocentric_central_angle_deg(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg),
                wgs_geo.azimuth_deg,
                wgs_geo.back_azimuth_deg,
            )
        } else {
            let sph = gps2dist_azimuth_ellipsoid(
                ev_lat_deg,
                ev_lon_deg,
                st_lat_deg,
                st_lon_deg,
                Ellipsoid {
                    a_m: self.radius_of_planet * 1_000.0,
                    f: 0.0,
                },
            );
            (
                sph.distance_deg_spherical,
                sph.azimuth_deg,
                sph.back_azimuth_deg,
            )
        };

        let dm = self.prepare_depth_model(ev_depth_km, 0.0)?;
        let phase_names = parse_phase_list(phases);
        let mut out = Vec::new();
        for ph_name in &phase_names {
            let sp = SeismicPhase::new(ph_name, &dm)?;
            for raw in sp.calc_time(&dm, lookup_dist_deg, DEFAULT_RAY_PARAM_TOL)? {
                let mut arr = raw_to_seismic_arrival(&raw);
                let ray_az = major_arc_departure_azimuth(base_az_deg, arr.purist_distance_deg);
                let ray_baz = major_arc_back_azimuth(base_baz_deg, arr.purist_distance_deg);
                arr.azimuth_deg = Some(ray_az);
                arr.back_azimuth_deg = Some(ray_baz);
                if apply_ellipticity {
                    let dt_ellip =
                        arrival_ellipticity_correction_s(&sp, &dm, &raw, ev_lat_deg, ray_az)?;
                    arr.time_s += dt_ellip;
                    arr.ellipticity_corr_s = Some(dt_ellip);
                }
                out.push(arr);
            }
        }
        out.sort_by(arrival_cmp);
        Ok(out)
    }

    /// Computes discontinuity and turning-point pierce sequences for all matching arrivals.
    pub fn get_pierce_points<S: AsRef<str>>(
        &self,
        depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<ArrivalPierce>> {
        if !distance_deg.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite epicentral distance_deg".into(),
            ));
        }
        let dm = self.prepare_depth_model(depth_km, 0.0)?;
        let phase_names = parse_phase_list(phases);
        let mut out = Vec::new();
        for ph_name in &phase_names {
            let sp = SeismicPhase::new(ph_name, &dm)?;
            for raw in sp.calc_time(&dm, distance_deg, DEFAULT_PATH_RAY_PARAM_TOL)? {
                let td_pts = sp.calc_pierce_for_arrival(&dm, &raw)?;
                let pierce_points = td_pts
                    .into_iter()
                    .map(|td: TimeDist| PiercePoint {
                        p_s_per_rad: td.p,
                        time_s: td.time,
                        dist_rad: td.dist,
                        dist_deg: td.dist.to_degrees(),
                        depth_km: td.depth,
                        lat_deg: None,
                        lon_deg: None,
                    })
                    .collect();
                out.push(ArrivalPierce {
                    arrival: raw_to_seismic_arrival(&raw),
                    pierce_points,
                });
            }
        }
        out.sort_by(|a, b| arrival_cmp(&a.arrival, &b.arrival));
        Ok(out)
    }

    /// Computes discontinuity and turning-point pierce sequences decorated with geographic
    /// `(lat_deg, lon_deg)` waypoints along the minor/major great-circle azimuth.
    #[allow(clippy::too_many_arguments)]
    pub fn get_pierce_points_geo<S: AsRef<str>>(
        &self,
        ev_lat_deg: f64,
        ev_lon_deg: f64,
        ev_depth_km: f64,
        st_lat_deg: f64,
        st_lon_deg: f64,
        phases: &[S],
        apply_ellipticity: bool,
    ) -> Result<Vec<ArrivalPierce>> {
        let wgs_geo = try_gps2dist_azimuth(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg)?;
        let sph = gps2dist_azimuth_ellipsoid(
            ev_lat_deg,
            ev_lon_deg,
            st_lat_deg,
            st_lon_deg,
            Ellipsoid {
                a_m: self.radius_of_planet * 1_000.0,
                f: 0.0,
            },
        );
        let lookup_dist_deg = if apply_ellipticity {
            geocentric_central_angle_deg(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg)
        } else {
            sph.distance_deg_spherical
        };
        let base_az = if apply_ellipticity {
            wgs_geo.azimuth_deg
        } else {
            sph.azimuth_deg
        };
        let base_baz = if apply_ellipticity {
            wgs_geo.back_azimuth_deg
        } else {
            sph.back_azimuth_deg
        };

        let dm = self.prepare_depth_model(ev_depth_km, 0.0)?;
        let phase_names = parse_phase_list(phases);
        let mut out = Vec::new();
        for ph_name in &phase_names {
            let sp = SeismicPhase::new(ph_name, &dm)?;
            for raw in sp.calc_time(&dm, lookup_dist_deg, DEFAULT_RAY_PARAM_TOL)? {
                let mut arr = raw_to_seismic_arrival(&raw);
                let ray_az = major_arc_departure_azimuth(base_az, arr.purist_distance_deg);
                let ray_baz = major_arc_back_azimuth(base_baz, arr.purist_distance_deg);
                arr.azimuth_deg = Some(ray_az);
                arr.back_azimuth_deg = Some(ray_baz);
                if apply_ellipticity {
                    let dt_ellip =
                        arrival_ellipticity_correction_s(&sp, &dm, &raw, ev_lat_deg, ray_az)?;
                    arr.time_s += dt_ellip;
                    arr.ellipticity_corr_s = Some(dt_ellip);
                }

                // Use spherical departure azimuth so the terminal waypoint lands cleanly at station!
                let wp_az = major_arc_departure_azimuth(sph.azimuth_deg, arr.purist_distance_deg);
                let dist_scale = if lookup_dist_deg > 0.0 && apply_ellipticity {
                    sph.distance_deg_spherical / lookup_dist_deg
                } else {
                    1.0
                };

                let td_pts = sp.calc_pierce_for_arrival(&dm, &raw)?;
                let pierce_points = td_pts
                    .into_iter()
                    .map(|td| {
                        let (lat, lon) = spherical_great_circle_point(
                            ev_lat_deg,
                            ev_lon_deg,
                            wp_az,
                            td.dist * dist_scale,
                        );
                        PiercePoint {
                            p_s_per_rad: td.p,
                            time_s: td.time,
                            dist_rad: td.dist,
                            dist_deg: td.dist.to_degrees(),
                            depth_km: td.depth,
                            lat_deg: Some(lat),
                            lon_deg: Some(lon),
                        }
                    })
                    .collect();

                out.push(ArrivalPierce {
                    arrival: arr,
                    pierce_points,
                });
            }
        }
        out.sort_by(|a, b| arrival_cmp(&a.arrival, &b.arrival));
        Ok(out)
    }

    /// Computes dense spherical `(p, time, dist_rad, dist_deg, depth_km, radius_km)` ray trajectories.
    pub fn get_ray_paths<S: AsRef<str>>(
        &self,
        depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<ArrivalPath>> {
        if !distance_deg.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite epicentral distance_deg".into(),
            ));
        }
        let dm = self.prepare_depth_model(depth_km, 0.0)?;
        let r_planet = dm.radius_of_planet;
        let phase_names = parse_phase_list(phases);
        let mut out = Vec::new();
        for ph_name in &phase_names {
            let sp = SeismicPhase::new(ph_name, &dm)?;
            for raw in sp.calc_time(&dm, distance_deg, DEFAULT_PATH_RAY_PARAM_TOL)? {
                let td_pts = sp.calc_path_for_arrival(&dm, &raw)?;
                let path_points = td_pts
                    .into_iter()
                    .map(|td| RayPathPoint {
                        p_s_per_rad: td.p,
                        time_s: td.time,
                        dist_rad: td.dist,
                        dist_deg: td.dist.to_degrees(),
                        depth_km: td.depth,
                        radius_km: r_planet - td.depth,
                        lat_deg: None,
                        lon_deg: None,
                    })
                    .collect();
                out.push(ArrivalPath {
                    arrival: raw_to_seismic_arrival(&raw),
                    path_points,
                });
            }
        }
        out.sort_by(|a, b| arrival_cmp(&a.arrival, &b.arrival));
        Ok(out)
    }

    /// Computes dense 3D spherical ray trajectories decorated with geographic `(lat_deg, lon_deg)`
    /// coordinates along the great circle.
    #[allow(clippy::too_many_arguments)]
    pub fn get_ray_paths_geo<S: AsRef<str>>(
        &self,
        ev_lat_deg: f64,
        ev_lon_deg: f64,
        ev_depth_km: f64,
        st_lat_deg: f64,
        st_lon_deg: f64,
        phases: &[S],
        apply_ellipticity: bool,
    ) -> Result<Vec<ArrivalPath>> {
        let wgs_geo = try_gps2dist_azimuth(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg)?;
        let sph = gps2dist_azimuth_ellipsoid(
            ev_lat_deg,
            ev_lon_deg,
            st_lat_deg,
            st_lon_deg,
            Ellipsoid {
                a_m: self.radius_of_planet * 1_000.0,
                f: 0.0,
            },
        );
        let lookup_dist_deg = if apply_ellipticity {
            geocentric_central_angle_deg(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg)
        } else {
            sph.distance_deg_spherical
        };
        let base_az = if apply_ellipticity {
            wgs_geo.azimuth_deg
        } else {
            sph.azimuth_deg
        };
        let base_baz = if apply_ellipticity {
            wgs_geo.back_azimuth_deg
        } else {
            sph.back_azimuth_deg
        };

        let dm = self.prepare_depth_model(ev_depth_km, 0.0)?;
        let r_planet = dm.radius_of_planet;
        let phase_names = parse_phase_list(phases);
        let mut out = Vec::new();
        for ph_name in &phase_names {
            let sp = SeismicPhase::new(ph_name, &dm)?;
            for raw in sp.calc_time(&dm, lookup_dist_deg, DEFAULT_RAY_PARAM_TOL)? {
                let mut arr = raw_to_seismic_arrival(&raw);
                let ray_az = major_arc_departure_azimuth(base_az, arr.purist_distance_deg);
                let ray_baz = major_arc_back_azimuth(base_baz, arr.purist_distance_deg);
                arr.azimuth_deg = Some(ray_az);
                arr.back_azimuth_deg = Some(ray_baz);
                if apply_ellipticity {
                    let dt_ellip =
                        arrival_ellipticity_correction_s(&sp, &dm, &raw, ev_lat_deg, ray_az)?;
                    arr.time_s += dt_ellip;
                    arr.ellipticity_corr_s = Some(dt_ellip);
                }

                let wp_az = major_arc_departure_azimuth(sph.azimuth_deg, arr.purist_distance_deg);
                let dist_scale = if lookup_dist_deg > 0.0 && apply_ellipticity {
                    sph.distance_deg_spherical / lookup_dist_deg
                } else {
                    1.0
                };

                let td_pts = sp.calc_path_for_arrival(&dm, &raw)?;
                let path_points = td_pts
                    .into_iter()
                    .map(|td| {
                        let (lat, lon) = spherical_great_circle_point(
                            ev_lat_deg,
                            ev_lon_deg,
                            wp_az,
                            td.dist * dist_scale,
                        );
                        RayPathPoint {
                            p_s_per_rad: td.p,
                            time_s: td.time,
                            dist_rad: td.dist,
                            dist_deg: td.dist.to_degrees(),
                            depth_km: td.depth,
                            radius_km: r_planet - td.depth,
                            lat_deg: Some(lat),
                            lon_deg: Some(lon),
                        }
                    })
                    .collect();

                out.push(ArrivalPath {
                    arrival: arr,
                    path_points,
                });
            }
        }
        out.sort_by(|a, b| arrival_cmp(&a.arrival, &b.arrival));
        Ok(out)
    }

    /// Computes $T(\Delta)$ and $\tau(p)$ branch curves for `phases` at `depth_km`.
    ///
    /// - When `n_points == 0`, returns the exact internal slowness-grid knots of each continuous
    ///   sub-branch.
    /// - When `n_points >= 2`, evaluates `n_points` uniformly spaced samples across each continuous
    ///   sub-branch via exact analytical Bullen slowness integration (`SeismicPhase::shoot_ray_td`),
    ///   preserving $T(p) = \tau(p) + p\Delta(p)$ and $d\tau/dp = -\Delta(p)$ to machine precision.
    pub fn get_travel_time_curves<S: AsRef<str>>(
        &self,
        depth_km: f64,
        phases: &[S],
        n_points: usize,
    ) -> Result<Vec<PhaseTravelTimeCurve>> {
        let dm = self.prepare_depth_model(depth_km, 0.0)?;
        let phase_names = parse_phase_list(phases);
        let mut curves = Vec::with_capacity(phase_names.len());

        for ph_name in &phase_names {
            let sp = SeismicPhase::new(ph_name, &dm)?;
            if sp.ray_param.is_empty() || sp.max_distance < 0.0 {
                continue;
            }
            let splits = sp.shadow_zone_splits();
            let mut branches = Vec::with_capacity(splits.len());

            for (b_idx, range) in splits.into_iter().enumerate() {
                if range.is_empty() {
                    continue;
                }
                let p_first = sp.ray_param[range.start];
                let p_last = sp.ray_param[range.end - 1];
                let is_const_p =
                    (p_first - p_last).abs() == 0.0 || !sp.head_or_diffract_seq.is_empty();

                let mut pts = Vec::new();
                if n_points == 0 {
                    pts.reserve(range.len());
                    for k in range {
                        let p = sp.ray_param[k];
                        let t = sp.time[k];
                        let d = sp.dist[k];
                        pts.push(TauPCurvePoint {
                            ray_param_s_per_rad: p,
                            tau_s: t - p * d,
                            time_s: t,
                            dist_rad: d,
                            dist_deg: d.to_degrees(),
                        });
                    }
                } else if n_points == 1 {
                    let k = range.start;
                    let p = sp.ray_param[k];
                    let t = sp.time[k];
                    let d = sp.dist[k];
                    pts.push(TauPCurvePoint {
                        ray_param_s_per_rad: p,
                        tau_s: t - p * d,
                        time_s: t,
                        dist_rad: d,
                        dist_deg: d.to_degrees(),
                    });
                } else if is_const_p {
                    // Constant ray-parameter head wave, diffracted wave, or surface `kmps` phase:
                    let d0 = sp.dist[range.start];
                    let d1 = sp.dist[range.end - 1];
                    let t0 = sp.time[range.start];
                    pts.reserve(n_points);
                    let denom = (n_points - 1) as f64;
                    for j in 0..n_points {
                        let frac = (j as f64) / denom;
                        let d = d0 + frac * (d1 - d0);
                        let t = t0 + p_first * (d - d0);
                        pts.push(TauPCurvePoint {
                            ray_param_s_per_rad: p_first,
                            tau_s: t - p_first * d,
                            time_s: t,
                            dist_rad: d,
                            dist_deg: d.to_degrees(),
                        });
                    }
                } else {
                    pts.reserve(n_points);
                    let denom = (n_points - 1) as f64;
                    for j in 0..n_points {
                        let frac = (j as f64) / denom;
                        let p = p_first + frac * (p_last - p_first);
                        let (t, d) = sp.shoot_ray_td(&dm, p)?;
                        pts.push(TauPCurvePoint {
                            ray_param_s_per_rad: p,
                            tau_s: t - p * d,
                            time_s: t,
                            dist_rad: d,
                            dist_deg: d.to_degrees(),
                        });
                    }
                }

                branches.push(PhaseCurveBranch {
                    branch_index: b_idx,
                    points: pts,
                });
            }

            curves.push(PhaseTravelTimeCurve {
                name: sp.name,
                purist_name: sp.purist_name,
                source_depth_km: sp.source_depth,
                min_ray_param_s_per_rad: sp.min_ray_param,
                max_ray_param_s_per_rad: sp.max_ray_param,
                min_distance_deg: sp.min_distance.to_degrees(),
                max_distance_deg: sp.max_distance.to_degrees(),
                branches,
            });
        }

        Ok(curves)
    }
}

type PhaseCacheEntry = (f64, Vec<String>, Vec<String>, Vec<SeismicPhase>);

/// High-throughput TauP query engine caching depth-split [`TauModel`]s and compiled
/// [`SeismicPhase`]s so repeated or grid queries execute in microseconds per call.
#[derive(Debug, Clone)]
pub struct TauEngine {
    base_model: TauModel,
    cache_capacity: usize,
    depth_cache: Vec<(f64, TauModel)>,
    phase_cache: Vec<PhaseCacheEntry>,
}

impl TauEngine {
    /// Creates a `TauEngine` wrapping a surface-source `TauModel` (`source_depth == 0.0`).
    pub fn new(base_model: TauModel) -> Result<Self> {
        if base_model.source_depth != 0.0 {
            return Err(EarthSnailError::ModelError(
                "TauEngine::new requires a surface-source TauModel (source_depth == 0.0)".into(),
            ));
        }
        Ok(Self {
            base_model,
            cache_capacity: 16,
            depth_cache: Vec::with_capacity(16),
            phase_cache: Vec::with_capacity(16),
        })
    }

    /// Creates a `TauEngine` for a named built-in Earth model (`"iasp91"`, `"ak135"`, `"prem"`).
    pub fn by_name(model_name: &str) -> Result<Self> {
        Self::new(TauModel::by_name(model_name)?)
    }

    /// Borrows the underlying surface-source [`TauModel`].
    #[inline]
    pub fn base_model(&self) -> &TauModel {
        &self.base_model
    }

    fn ensure_depth_and_phases<S: AsRef<str>>(
        &mut self,
        depth_km: f64,
        phases: &[S],
    ) -> Result<(&TauModel, &[SeismicPhase])> {
        if !depth_km.is_finite() || depth_km < 0.0 || depth_km > self.base_model.radius_of_planet {
            return Err(EarthSnailError::InvalidParameter(alloc::format!(
                "source depth {depth_km} km outside [0, {}]",
                self.base_model.radius_of_planet
            )));
        }

        let d_idx = if let Some(pos) = self
            .depth_cache
            .iter()
            .position(|(d, _)| (*d - depth_km).abs() == 0.0)
        {
            pos
        } else {
            let dm = if depth_km == 0.0 {
                self.base_model.clone()
            } else {
                self.base_model.with_source_depth(depth_km)?
            };
            if self.depth_cache.len() >= self.cache_capacity {
                self.depth_cache.remove(0);
            }
            self.depth_cache.push((depth_km, dm));
            self.depth_cache.len() - 1
        };

        // Zero-allocation fast path when caller passes the identical `phases` slice:
        if let Some(pos) = self.phase_cache.iter().position(|(d, raw_in, _, _)| {
            (*d - depth_km).abs() == 0.0
                && raw_in.len() == phases.len()
                && raw_in
                    .iter()
                    .zip(phases.iter())
                    .all(|(a, b)| a == b.as_ref())
        }) {
            return Ok((&self.depth_cache[d_idx].1, &self.phase_cache[pos].3));
        }

        let raw_input: Vec<String> = phases
            .iter()
            .map(|s| alloc::string::ToString::to_string(s.as_ref()))
            .collect();
        let phase_names = parse_phase_list(phases);
        let p_idx = if let Some(pos) = self
            .phase_cache
            .iter()
            .position(|(d, _, names, _)| (*d - depth_km).abs() == 0.0 && *names == phase_names)
        {
            pos
        } else {
            let dm_ref = &self.depth_cache[d_idx].1;
            let mut compiled = Vec::with_capacity(phase_names.len());
            for ph in &phase_names {
                compiled.push(SeismicPhase::new(ph, dm_ref)?);
            }
            if self.phase_cache.len() >= self.cache_capacity {
                self.phase_cache.remove(0);
            }
            self.phase_cache
                .push((depth_km, raw_input, phase_names, compiled));
            self.phase_cache.len() - 1
        };

        Ok((&self.depth_cache[d_idx].1, &self.phase_cache[p_idx].3))
    }

    /// Cached high-speed `get_travel_times` query.
    pub fn get_travel_times<S: AsRef<str>>(
        &mut self,
        depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<SeismicArrival>> {
        if !distance_deg.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite epicentral distance_deg".into(),
            ));
        }
        let (dm, compiled) = self.ensure_depth_and_phases(depth_km, phases)?;
        let mut out = Vec::with_capacity(8);
        for sp in compiled {
            for raw in sp.calc_time(dm, distance_deg, DEFAULT_RAY_PARAM_TOL)? {
                out.push(raw_into_seismic_arrival(raw));
            }
        }
        out.sort_by(arrival_cmp);
        Ok(out)
    }

    /// Cached high-speed `get_travel_times_geo` query.
    #[allow(clippy::too_many_arguments)]
    pub fn get_travel_times_geo<S: AsRef<str>>(
        &mut self,
        ev_lat_deg: f64,
        ev_lon_deg: f64,
        ev_depth_km: f64,
        st_lat_deg: f64,
        st_lon_deg: f64,
        phases: &[S],
        apply_ellipticity: bool,
    ) -> Result<Vec<SeismicArrival>> {
        let wgs_geo = try_gps2dist_azimuth(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg)?;
        let (lookup_dist_deg, base_az_deg, base_baz_deg) = if apply_ellipticity {
            (
                geocentric_central_angle_deg(ev_lat_deg, ev_lon_deg, st_lat_deg, st_lon_deg),
                wgs_geo.azimuth_deg,
                wgs_geo.back_azimuth_deg,
            )
        } else {
            let sph = gps2dist_azimuth_ellipsoid(
                ev_lat_deg,
                ev_lon_deg,
                st_lat_deg,
                st_lon_deg,
                Ellipsoid {
                    a_m: self.base_model.radius_of_planet * 1_000.0,
                    f: 0.0,
                },
            );
            (
                sph.distance_deg_spherical,
                sph.azimuth_deg,
                sph.back_azimuth_deg,
            )
        };

        let (dm, compiled) = self.ensure_depth_and_phases(ev_depth_km, phases)?;
        let mut out = Vec::new();
        for sp in compiled {
            for raw in sp.calc_time(dm, lookup_dist_deg, DEFAULT_RAY_PARAM_TOL)? {
                let mut arr = raw_to_seismic_arrival(&raw);
                let ray_az = major_arc_departure_azimuth(base_az_deg, arr.purist_distance_deg);
                let ray_baz = major_arc_back_azimuth(base_baz_deg, arr.purist_distance_deg);
                arr.azimuth_deg = Some(ray_az);
                arr.back_azimuth_deg = Some(ray_baz);
                if apply_ellipticity {
                    let dt_ellip =
                        arrival_ellipticity_correction_s(sp, dm, &raw, ev_lat_deg, ray_az)?;
                    arr.time_s += dt_ellip;
                    arr.ellipticity_corr_s = Some(dt_ellip);
                }
                out.push(arr);
            }
        }
        out.sort_by(arrival_cmp);
        Ok(out)
    }

    /// Cached `get_pierce_points` query.
    pub fn get_pierce_points<S: AsRef<str>>(
        &mut self,
        depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<ArrivalPierce>> {
        if !distance_deg.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite epicentral distance_deg".into(),
            ));
        }
        let (dm, compiled) = self.ensure_depth_and_phases(depth_km, phases)?;
        let mut out = Vec::new();
        for sp in compiled {
            for raw in sp.calc_time(dm, distance_deg, DEFAULT_PATH_RAY_PARAM_TOL)? {
                let td_pts = sp.calc_pierce_for_arrival(dm, &raw)?;
                let pierce_points = td_pts
                    .into_iter()
                    .map(|td| PiercePoint {
                        p_s_per_rad: td.p,
                        time_s: td.time,
                        dist_rad: td.dist,
                        dist_deg: td.dist.to_degrees(),
                        depth_km: td.depth,
                        lat_deg: None,
                        lon_deg: None,
                    })
                    .collect();
                out.push(ArrivalPierce {
                    arrival: raw_to_seismic_arrival(&raw),
                    pierce_points,
                });
            }
        }
        out.sort_by(|a, b| arrival_cmp(&a.arrival, &b.arrival));
        Ok(out)
    }

    /// Cached `get_ray_paths` query.
    pub fn get_ray_paths<S: AsRef<str>>(
        &mut self,
        depth_km: f64,
        distance_deg: f64,
        phases: &[S],
    ) -> Result<Vec<ArrivalPath>> {
        if !distance_deg.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite epicentral distance_deg".into(),
            ));
        }
        let (dm, compiled) = self.ensure_depth_and_phases(depth_km, phases)?;
        let r_planet = dm.radius_of_planet;
        let mut out = Vec::new();
        for sp in compiled {
            for raw in sp.calc_time(dm, distance_deg, DEFAULT_PATH_RAY_PARAM_TOL)? {
                let td_pts = sp.calc_path_for_arrival(dm, &raw)?;
                let path_points = td_pts
                    .into_iter()
                    .map(|td| RayPathPoint {
                        p_s_per_rad: td.p,
                        time_s: td.time,
                        dist_rad: td.dist,
                        dist_deg: td.dist.to_degrees(),
                        depth_km: td.depth,
                        radius_km: r_planet - td.depth,
                        lat_deg: None,
                        lon_deg: None,
                    })
                    .collect();
                out.push(ArrivalPath {
                    arrival: raw_to_seismic_arrival(&raw),
                    path_points,
                });
            }
        }
        out.sort_by(|a, b| arrival_cmp(&a.arrival, &b.arrival));
        Ok(out)
    }

    /// Delegates `get_travel_time_curves` to the base `TauModel`.
    pub fn get_travel_time_curves<S: AsRef<str>>(
        &self,
        depth_km: f64,
        phases: &[S],
        n_points: usize,
    ) -> Result<Vec<PhaseTravelTimeCurve>> {
        self.base_model
            .get_travel_time_curves(depth_km, phases, n_points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geo_travel_times_pierce_and_curves_are_consistent() {
        let model = crate::tau_model::test_iasp91_model();
        let mut engine = TauEngine::new(model.clone()).unwrap();
        let arrs_plain = engine
            .get_travel_times_geo(38.297, 142.373, 29.0, 34.946, -106.457, &["P", "S"], false)
            .unwrap();
        let arrs_ellip = engine
            .get_travel_times_geo(38.297, 142.373, 29.0, 34.946, -106.457, &["P", "S"], true)
            .unwrap();
        assert_eq!(arrs_plain.len(), 2);
        assert_eq!(arrs_ellip.len(), 2);
        assert!(arrs_ellip[0].ellipticity_corr_s.is_some());
        assert!((arrs_plain[0].time_s - arrs_ellip[0].time_s).abs() < 2.0);

        let pierces = model
            .get_pierce_points_geo(38.297, 142.373, 29.0, 34.946, -106.457, &["P"], false)
            .unwrap();
        assert_eq!(pierces.len(), 1);
        let first_p = pierces[0].pierce_points.first().unwrap();
        let last_p = pierces[0].pierce_points.last().unwrap();
        assert!((first_p.lat_deg.unwrap() - 38.297).abs() < 1e-9);
        assert!((last_p.lat_deg.unwrap() - 34.946).abs() < 1e-4);
        assert!((last_p.lon_deg.unwrap() - (-106.457)).abs() < 1e-4);

        let n_curve = if cfg!(miri) { 5 } else { 25 };
        let curves = engine
            .get_travel_time_curves(35.0, &["P", "PKP"], n_curve)
            .unwrap();
        assert_eq!(curves.len(), 2);
        for c in &curves {
            assert!(!c.branches.is_empty());
            for b in &c.branches {
                assert_eq!(b.points.len(), n_curve);
            }
        }
    }
}
