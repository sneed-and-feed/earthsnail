//! Dziewonski & Gilbert (1976) / Clairaut hydrostatic WGS84 travel-time ellipticity corrections.
//!
//! For a hydrostatic oblate Earth with radial flattening profile $\epsilon(r)$ anchored to the
//! WGS84 surface flattening $f_{\text{WGS84}} = 1 / 298.257223563$, each surface of constant
//! material velocity at spherical radius $r$ is mapped to physical geocentric radius
//! $$r_{\text{phys}}(r, \vartheta) = r\left[1 - \frac{2}{3}\epsilon(r) P_2^0(\cos\vartheta)\right].$$
//! By the degree-2 spherical harmonic addition theorem along a great-circle ray leaving geocentric
//! source colatitude $\theta_0$ at azimuth $\zeta$,
//! $$P_2^0(\cos\vartheta(\Delta')) = \sum_{m=0}^{2} w_m(\theta_0, \zeta)\,g_m(\Delta'),$$
//! where $w_m$ are [`earthsnail_core::geodesics::dziewonski_gilbert_weights`] and
//! $g_0(\Delta') = \frac{1}{4}(1 + 3\cos 2\Delta')$, $g_1(\Delta') = \frac{\sqrt{3}}{2}\sin 2\Delta'$,
//! $g_2(\Delta') = \frac{\sqrt{3}}{2}\sin^2\Delta'$. Applying Fermat's variational principle
//! along any compiled 1D ray path yields the three distance-and-depth coefficients
//! $[\tau_0(\Delta, h), \tau_1(\Delta, h), \tau_2(\Delta, h)]$ in seconds and total travel-time
//! correction $\delta t_{\text{ellip}} = \sum_{m=0}^2 \tau_m w_m$.

use crate::phase::{RawArrival, SeismicPhase};
use crate::tau_model::{TauModel, TimeDist};
use earthsnail_core::geodesics::{dziewonski_gilbert_weights, Ellipsoid};
use earthsnail_core::{math, Result};

const HALF_SQRT3: f64 = 0.866_025_403_784_438_6;

/// Normalized radius knots $x = r / R_E \in [0, 1]$ and Clairaut hydrostatic flattening $\epsilon(x)$
/// computed for PREM/ak135 density structure and anchored to WGS84 surface flattening at $x = 1$.
const CLAIRAUT_X_KNOTS: [f64; 10] = [
    0.0, 0.191_7, // ICB (~1221.5 km)
    0.35, 0.546_2, // CMB (~3480.0 km)
    0.68, 0.80, 0.896_4, // 660 km discontinuity
    0.935_6, // 410 km discontinuity
    0.994_5, // Moho (~35 km)
    1.0,     // Free surface
];

const CLAIRAUT_EPS_KNOTS: [f64; 10] = [
    0.002_395_0,
    0.002_432_0,
    0.002_478_0,
    0.002_547_0,
    0.002_695_0,
    0.002_845_0,
    0.002_975_0,
    0.003_095_0,
    0.003_328_0,
    Ellipsoid::WGS84.f,
];

/// Evaluates Clairaut's hydrostatic radial flattening $\epsilon(r)$ at spherical radius
/// `radius_km` inside a planet of mean radius `planet_radius_km` (anchored to `WGS84.f` at $r = R_E$).
#[inline]
pub fn hydrostatic_ellipticity(radius_km: f64, planet_radius_km: f64) -> f64 {
    if planet_radius_km <= 0.0 {
        return Ellipsoid::WGS84.f;
    }
    let x = (radius_km / planet_radius_km).clamp(0.0, 1.0);
    // Smooth quadratic in x^2 across the inner core [0, x_icb] so d(eps)/dr = 0 at r = 0.
    if x <= CLAIRAUT_X_KNOTS[1] {
        let u = x / CLAIRAUT_X_KNOTS[1];
        return CLAIRAUT_EPS_KNOTS[0] + (CLAIRAUT_EPS_KNOTS[1] - CLAIRAUT_EPS_KNOTS[0]) * (u * u);
    }
    for i in 1..(CLAIRAUT_X_KNOTS.len() - 1) {
        let x0 = CLAIRAUT_X_KNOTS[i];
        let x1 = CLAIRAUT_X_KNOTS[i + 1];
        if x <= x1 {
            let t = (x - x0) / (x1 - x0);
            return CLAIRAUT_EPS_KNOTS[i] + t * (CLAIRAUT_EPS_KNOTS[i + 1] - CLAIRAUT_EPS_KNOTS[i]);
        }
    }
    Ellipsoid::WGS84.f
}

/// Evaluates the three angular basis functions `[g0(delta), g1(delta), g2(delta)]` at ray-path
/// angular distance `delta_rad` from the source.
#[inline]
pub fn ellipticity_distance_basis(delta_rad: f64) -> [f64; 3] {
    let sin_d = math::sin(delta_rad);
    let (sin_2d, cos_2d) = math::sin_cos(2.0 * delta_rad);
    [
        0.25 * (1.0 + 3.0 * cos_2d),
        HALF_SQRT3 * sin_2d,
        HALF_SQRT3 * (sin_d * sin_d),
    ]
}

/// Integrates the Dziewonski & Gilbert (1976) degree-2 ellipticity travel-time coefficients
/// `[\tau_0, \tau_1, \tau_2]` (in seconds) along a cumulative 1D spherical ray `path`
/// (`TimeDist` sequence produced by [`SeismicPhase::calc_path_for_arrival`]).
pub fn integrate_ellipticity_tau_coeffs(path: &[TimeDist], planet_radius_km: f64) -> [f64; 3] {
    if path.len() < 2 || planet_radius_km <= 0.0 {
        return [0.0; 3];
    }

    let mut tau = [0.0; 3];
    let r0 = (planet_radius_km - path[0].depth).max(0.0);
    let eps0 = hydrostatic_ellipticity(r0, planet_radius_km);
    let g0 = ellipticity_distance_basis(path[0].dist);
    let mut prev_r = r0;
    let mut prev_t = path[0].time;
    let mut prev_d = path[0].dist;
    let mut prev_r_sigma = [
        -2.0 / 3.0 * r0 * eps0 * g0[0],
        -2.0 / 3.0 * r0 * eps0 * g0[1],
        -2.0 / 3.0 * r0 * eps0 * g0[2],
    ];

    for pt in &path[1..] {
        let r1 = (planet_radius_km - pt.depth).max(0.0);
        let dt = pt.time - prev_t;
        let dd = pt.dist - prev_d;
        let eps1 = hydrostatic_ellipticity(r1, planet_radius_km);
        let g1 = ellipticity_distance_basis(pt.dist);
        let next_r_sigma = [
            -2.0 / 3.0 * r1 * eps1 * g1[0],
            -2.0 / 3.0 * r1 * eps1 * g1[1],
            -2.0 / 3.0 * r1 * eps1 * g1[2],
        ];

        if dt > 0.0 {
            let dr = r1 - prev_r;
            let r_mid = 0.5 * (prev_r + r1);
            let ds2 = dr * dr + (r_mid * dd) * (r_mid * dd);
            if ds2 > 0.0 {
                let d_mid = 0.5 * (prev_d + pt.dist);
                let eps_mid = hydrostatic_ellipticity(r_mid, planet_radius_km);
                let g_mid = ellipticity_distance_basis(d_mid);
                let r2_dd2 = (r_mid * dd) * (r_mid * dd);
                for m in 0..3 {
                    let dr_hat = next_r_sigma[m] - prev_r_sigma[m];
                    let sigma_mid = -2.0 / 3.0 * eps_mid * g_mid[m];
                    let stretch = (dr * dr_hat + r2_dd2 * sigma_mid) / ds2;
                    tau[m] += dt * stretch;
                }
            } else {
                // Zero-chord fallback (should not occur when dt > 0).
                let eps_mid = hydrostatic_ellipticity(r_mid, planet_radius_km);
                let g_mid = ellipticity_distance_basis(prev_d);
                for m in 0..3 {
                    tau[m] += dt * (-2.0 / 3.0 * eps_mid * g_mid[m]);
                }
            }
        }

        prev_r = r1;
        prev_t = pt.time;
        prev_d = pt.dist;
        prev_r_sigma = next_r_sigma;
    }

    tau
}

/// Computes the Dziewonski & Gilbert (1976) coefficients `[\tau_0, \tau_1, \tau_2]` (seconds)
/// for a specific compiled `phase` and `arrival` on `depth_corrected_model`.
pub fn phase_ellipticity_tau_coeffs(
    phase: &SeismicPhase,
    depth_corrected_model: &TauModel,
    arrival: &RawArrival,
) -> Result<[f64; 3]> {
    let path = phase.calc_path_for_arrival(depth_corrected_model, arrival)?;
    Ok(integrate_ellipticity_tau_coeffs(
        &path,
        depth_corrected_model.radius_of_planet,
    ))
}

/// Evaluates the WGS84 Dziewonski & Gilbert (1976) travel-time ellipticity correction
/// $\delta t_{\text{ellip}} = \sum_{m=0}^2 \tau_m w_m(\theta_0, \zeta)$ in seconds given
/// precomputed `tau_coeffs`, geographic `source_lat_deg`, and departure `azimuth_deg` along the ray.
#[inline]
pub fn evaluate_ellipticity_correction(
    tau_coeffs: [f64; 3],
    source_lat_deg: f64,
    azimuth_deg: f64,
) -> f64 {
    let w = dziewonski_gilbert_weights(source_lat_deg, azimuth_deg);
    tau_coeffs[0] * w[0] + tau_coeffs[1] * w[1] + tau_coeffs[2] * w[2]
}

/// Computes the WGS84 Dziewonski & Gilbert (1976) travel-time ellipticity correction (seconds)
/// directly for a compiled `phase`, `arrival`, geographic `source_lat_deg`, and ray departure `azimuth_deg`.
pub fn arrival_ellipticity_correction_s(
    phase: &SeismicPhase,
    depth_corrected_model: &TauModel,
    arrival: &RawArrival,
    source_lat_deg: f64,
    azimuth_deg: f64,
) -> Result<f64> {
    let coeffs = phase_ellipticity_tau_coeffs(phase, depth_corrected_model, arrival)?;
    Ok(evaluate_ellipticity_correction(
        coeffs,
        source_lat_deg,
        azimuth_deg,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_theorem_identity_holds_across_sphere() {
        // Verify P20(cos theta(Delta')) == sum_m w_m(theta0, az) * g_m(Delta')
        for &(src_lat, az, delta_deg) in &[
            (90.0_f64, 0.0_f64, 0.0_f64),
            (90.0, 123.0, 90.0),
            (0.0, 90.0, 67.5),
            (37.5, 42.0, 55.0),
            (-52.3, 215.0, 132.0),
        ] {
            let geoc_lat = earthsnail_core::geodesics::geographic_to_geocentric_lat(src_lat);
            let th0 = (90.0 - geoc_lat).to_radians();
            let az_r = az.to_radians();
            let del = delta_deg.to_radians();
            let cos_th =
                math::cos(th0) * math::cos(del) + math::sin(th0) * math::cos(az_r) * math::sin(del);
            let p20_exact = 1.5 * cos_th * cos_th - 0.5;

            let w = dziewonski_gilbert_weights(src_lat, az);
            let g = ellipticity_distance_basis(del);
            let p20_sum = w[0] * g[0] + w[1] * g[1] + w[2] * g[2];
            assert!(
                (p20_exact - p20_sum).abs() < 1e-12,
                "mismatch: {p20_exact} vs {p20_sum}"
            );
        }
    }

    #[test]
    fn polar_pcp_arrives_early_and_equatorial_p_arrives_late() {
        let model = if cfg!(miri) {
            crate::tau_model::test_iasp91_model()
        } else {
            TauModel::ak135().unwrap().with_source_depth(0.0).unwrap()
        };

        // 1. Vertical PcP at the North Pole (0 deg epicentral distance):
        // Polar flattening shortens the mantle column by ~8.3 km each way -> ~ -1.46 s!
        let pcp = SeismicPhase::new("PcP", &model).unwrap();
        let pcp_arr = &pcp.calc_time(&model, 0.0, 1e-6).unwrap()[0];
        let dt_pole_pcp =
            arrival_ellipticity_correction_s(&pcp, &model, pcp_arr, 90.0, 0.0).unwrap();
        assert!(
            dt_pole_pcp < -1.2 && dt_pole_pcp > -1.7,
            "unexpected polar PcP ellipticity correction: {dt_pole_pcp} s"
        );

        // 2. Teleseismic P at 60 deg along the equator vs from the North Pole:
        let p_phase = SeismicPhase::new("P", &model).unwrap();
        let p_arr = &p_phase.calc_time(&model, 60.0, 1e-6).unwrap()[0];
        let dt_equator =
            arrival_ellipticity_correction_s(&p_phase, &model, p_arr, 0.0, 90.0).unwrap();
        let dt_polar =
            arrival_ellipticity_correction_s(&p_phase, &model, p_arr, 90.0, 180.0).unwrap();
        assert!(
            dt_equator > 0.3 && dt_equator < 1.0,
            "equatorial P should arrive later (+0.3..+1.0 s), got {dt_equator}"
        );
        assert!(
            dt_polar < -0.3 && dt_polar > -1.2,
            "polar P should arrive earlier (-1.2..-0.3 s), got {dt_polar}"
        );
    }
}
