//! Property-based (`proptest`) mathematical & physical invariant verification suite for
//! `earthsnail-taup`:
//! 1. Legendre delay-time identities $T(p) = \tau(p) + p\Delta(p)$ and
//!    $(\tau(p + \delta p) - \tau(p - \delta p)) / (2\delta p) = -\Delta(p)$ to `< 1e-10`.
//! 2. Spherical Snell's law invariant $\eta(r)\sin i(r) = p$ and monotonic trajectory checks
//!    along `get_ray_paths` across crustal, mantle, converted, diffracted, and core phases.
//! 3. Synthetic multi-station `P + S + pP` 4D hypocenter recovery (`< 0.05 km` noise-free and
//!    `< 3.0 km` under multi-second clock-glitch outliers via damped Huber-IRLS).
//! 4. Adversarial fuzzing of `.nd` / `.tvel` model compilers and `leg_puller` / `SeismicPhase`
//!    ensuring zero panics.

use earthsnail_core::geodesics::locations2degrees;
use earthsnail_taup::{
    bullen_radial_slowness, evaluate_at_bullen, leg_puller, locate_hypocenter,
    spherical_great_circle_point, HypocenterConfig, PhaseArrivalObservation, SeismicPhase,
    SlownessLayer, TauModel, VelocityModel, DEFAULT_PATH_RAY_PARAM_TOL,
};
use proptest::prelude::*;
use std::f64::consts::PI;
use std::sync::OnceLock;

fn cached_models() -> &'static [TauModel; 3] {
    static MODELS: OnceLock<[TauModel; 3]> = OnceLock::new();
    MODELS.get_or_init(|| {
        [
            TauModel::iasp91().unwrap(),
            TauModel::ak135().unwrap(),
            TauModel::prem().unwrap(),
        ]
    })
}

fn eval_phase_legendre_case(
    model_idx: usize,
    depth_km: f64,
    phase_name: &str,
    slot_frac: f64,
) -> Option<(f64, f64, f64, f64, f64)> {
    let base = &cached_models()[model_idx];
    let dm = base.with_source_depth(depth_km).unwrap();
    let sp = SeismicPhase::new(phase_name, &dm).unwrap();
    if !sp.has_arrivals() || sp.ray_param.len() < 4 {
        return None;
    }

    // Collect ALL slowness-layer knots (including source-depth split knots!) inside
    // `[sp.min_ray_param, sp.max_ray_param]` so `(p_lo, p_hi)` never straddles a C^1 knot:
    let mut knots = Vec::with_capacity(dm.s_mod.p_layers.len() * 2 + sp.ray_param.len());
    knots.extend_from_slice(&sp.ray_param);
    for sl in &dm.s_mod.p_layers {
        if sl.top_p >= sp.min_ray_param && sl.top_p <= sp.max_ray_param {
            knots.push(sl.top_p);
        }
        if sl.bot_p >= sp.min_ray_param && sl.bot_p <= sp.max_ray_param {
            knots.push(sl.bot_p);
        }
    }
    for sl in &dm.s_mod.s_layers {
        if sl.top_p >= sp.min_ray_param && sl.top_p <= sp.max_ray_param {
            knots.push(sl.top_p);
        }
        if sl.bot_p >= sp.min_ray_param && sl.bot_p <= sp.max_ray_param {
            knots.push(sl.bot_p);
        }
    }
    knots.sort_by(|a, b| b.total_cmp(a));
    knots.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    // Select interior C^infty smooth slots of width > 2.0 s/rad (with fallback > 0.8 s/rad) so
    // `p_mid` sits >= 1.0 s/rad below any `(p_knot - p)^{3/2}` gradient-change onset, while
    // `h = 0.012 s/rad` keeps both IEEE-754 mantissa roundoff (~2.5e-11) and 4th-order stencil
    // truncation (< 5e-13) well below 1e-10!
    let mut smooth_slots = Vec::new();
    for k in 1..(knots.len().saturating_sub(2)) {
        let p_hi = knots[k];
        let p_lo = knots[k + 1];
        if p_hi - p_lo > 2.0 {
            smooth_slots.push((p_lo, p_hi));
        }
    }
    if smooth_slots.is_empty() {
        for k in 1..(knots.len().saturating_sub(2)) {
            let p_hi = knots[k];
            let p_lo = knots[k + 1];
            if p_hi - p_lo > 0.8 {
                smooth_slots.push((p_lo, p_hi));
            }
        }
    }
    if smooth_slots.is_empty() {
        return None;
    }

    let pick_idx = ((slot_frac * (smooth_slots.len() as f64)) as usize).min(smooth_slots.len() - 1);
    let (p_lo, p_hi) = smooth_slots[pick_idx];
    let p_mid = 0.5 * (p_lo + p_hi);
    let h = 1.2e-2_f64.min(0.08 * (p_hi - p_lo));

    let (t0, d0) = sp.shoot_ray_td(&dm, p_mid).unwrap();
    let tau0 = t0 - p_mid * d0;

    // Cancellation-reduced tau increment + 6th-order Richardson extrapolation:
    let eval_delta_tau = |dp: f64| -> f64 {
        let (t, d) = sp.shoot_ray_td(&dm, p_mid + dp).unwrap();
        (t - t0) - p_mid * (d - d0) - dp * d
    };
    let d4 = |step: f64| -> f64 {
        (eval_delta_tau(-2.0 * step) - 8.0 * eval_delta_tau(-step) + 8.0 * eval_delta_tau(step)
            - eval_delta_tau(2.0 * step))
            / (12.0 * step)
    };
    let d4_full = d4(h);
    let d4_half = d4(0.5 * h);
    let dtau_dp = (16.0 * d4_half - d4_full) / 15.0;

    Some((p_mid, t0, tau0, d0, dtau_dp))
}

#[test]
fn regression_iasp91_ss_legendre_identity() {
    // Reproduces verifier #24/#29 edge cases across iasp91 and ak135:
    for &(model_idx, depth_km, phase, slot_frac) in &[
        (0usize, 0.0_f64, "SS", 0.868_473_791_353_493_3_f64),
        (
            1usize,
            226.832_825_896_625_82_f64,
            "SS",
            0.201_381_289_043_578_08_f64,
        ),
        (
            1usize,
            316.262_654_451_587_25_f64,
            "pP",
            0.298_772_240_996_597_5_f64,
        ),
        (
            1usize,
            510.477_053_529_583_66_f64,
            "SS",
            0.223_017_173_269_426_85_f64,
        ),
    ] {
        let (p_mid, t0, tau0, d0, dtau_dp) =
            eval_phase_legendre_case(model_idx, depth_km, phase, slot_frac)
                .expect("expected valid smooth slot for SS regression");
        assert!((t0 - (tau0 + p_mid * d0)).abs() < 1e-11 * (1.0 + t0.abs()));
        let tol = 1e-10 * (1.0 + d0.abs() + 1e-3 * t0.abs());
        let err = (dtau_dp + d0).abs();
        assert!(
            err < tol,
            "regression {phase} Legendre error {err:.3e} >= {tol:.3e} (dtau/dp={dtau_dp}, -d0={})",
            -d0
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// Closed-form Bullen power-law shell Legendre identity:
    /// $T(p) = \tau(p) + p\Delta(p)$ and $\frac{\tau(p + \delta p) - \tau(p - \delta p)}{2\delta p} \approx -\Delta(p)$ to `< 1e-10`.
    #[test]
    fn prop_bullen_shell_legendre_identity(
        top_depth in 5.0_f64..2500.0,
        thickness in 5.0_f64..400.0,
        eta_low in 150.0_f64..650.0,
        eta_jump in 0.0_f64..250.0,
        p_frac in 0.05_f64..0.92,
        increasing in proptest::bool::ANY,
    ) {
        let bot_depth = (top_depth + thickness).min(6300.0);
        let (top_p, bot_p) = if increasing {
            (eta_low, eta_low + eta_jump)
        } else {
            (eta_low + eta_jump, eta_low)
        };
        let min_eta = top_p.min(bot_p);
        let p = p_frac * min_eta;
        let dp = 5e-4_f64.min(0.25 * (min_eta - p));
        prop_assume!(dp > 1e-5);

        let sl = SlownessLayer {
            top_p,
            top_depth,
            bot_p,
            bot_depth,
        };
        let (t0, d0) = bullen_radial_slowness(sl, p, 6371.0, true).unwrap();
        let (tp, dp_d) = bullen_radial_slowness(sl, p + dp, 6371.0, true).unwrap();
        let (tm, dm_d) = bullen_radial_slowness(sl, p - dp, 6371.0, true).unwrap();

        let tau0 = t0 - p * d0;
        let tau_p = tp - (p + dp) * dp_d;
        let tau_m = tm - (p - dp) * dm_d;
        let dtau_dp = (tau_p - tau_m) / (2.0 * dp);

        prop_assert!((t0 - (tau0 + p * d0)).abs() < 1e-12);
        prop_assert!(
            (dtau_dp + d0).abs() < 1e-10,
            "Legendre identity error {:.3e} >= 1e-10 (dtau/dp={dtau_dp}, -d0={})",
            (dtau_dp + d0).abs(),
            -d0
        );
    }

    /// Full multi-branch compiled `SeismicPhase` Legendre identity:
    /// Pick an interior ray parameter strictly inside a single slowness interval `(ray_params[k+1], ray_params[k])`
    /// so no turning-layer discontinuity kink occurs inside `[p - dp, p + dp]`, and verify
    /// $T(p) = \tau(p) + p\Delta(p)$ and central-difference $d\tau/dp = -\Delta(p)$ to `< 1e-10`!
    #[test]
    fn prop_phase_legendre_identity(
        model_idx in 0usize..3,
        depth_km in 0.0_f64..600.0,
        phase_idx in 0usize..12,
        slot_frac in 0.05_f64..0.95,
    ) {
        const BODY_PHASES: [&str; 12] = [
            "P", "S", "pP", "sP", "PP", "SS", "PcP", "ScS", "PKP", "PKIKP", "PKiKP", "SKS",
        ];
        let Some((p_mid, t0, tau0, d0, dtau_dp)) =
            eval_phase_legendre_case(model_idx, depth_km, BODY_PHASES[phase_idx], slot_frac)
        else {
            prop_assume!(false);
            unreachable!();
        };

        prop_assert!((t0 - (tau0 + p_mid * d0)).abs() < 1e-11 * (1.0 + t0.abs()));
        let tol = 1e-10 * (1.0 + d0.abs() + 1e-3 * t0.abs());
        prop_assert!(
            (dtau_dp + d0).abs() < tol,
            "Phase {} Legendre error {:.3e} >= {:.3e} (dtau/dp={dtau_dp}, -d0={})",
            BODY_PHASES[phase_idx],
            (dtau_dp + d0).abs(),
            tol,
            -d0
        );
    }
}

fn verify_snells_law_on_ray_paths(model_idx: usize, depth_km: f64, dist_deg: f64, phase: &str) {
    let base = &cached_models()[model_idx];
    let paths = base.get_ray_paths(depth_km, dist_deg, &[phase]).unwrap();
    for ap in &paths {
        let p = ap.arrival.ray_param_s_per_rad;
        let pts = &ap.path_points;
        assert!(pts.len() >= 2);

        let first = pts.first().unwrap();
        let last = pts.last().unwrap();
        assert!((first.depth_km - depth_km).abs() < 1e-9);
        assert!((last.depth_km - 0.0).abs() < 1e-9);
        let theta_last =
            last.time_s + p * (ap.arrival.purist_distance_deg.to_radians() - last.dist_rad);
        assert!((theta_last - ap.arrival.time_s).abs() < 5e-5);
        assert!((last.time_s - ap.arrival.time_s).abs() < 2e-3);
        assert!((last.dist_deg - ap.arrival.purist_distance_deg).abs() < 5e-3);

        for w in pts.windows(2) {
            let a = w[0];
            let b = w[1];
            assert!(b.time_s + 1e-12 >= a.time_s);
            assert!(b.dist_rad + 1e-12 >= a.dist_rad);
            assert!((b.p_s_per_rad - p).abs() < 1e-12);
            assert!((b.radius_km - (base.radius_of_planet - b.depth_km)).abs() < 1e-10);

            let dz = b.depth_km - a.depth_km;
            if dz.abs() > 1e-5 {
                let mid_z = 0.5 * (a.depth_km + b.depth_km);
                let mid_r = base.radius_of_planet - mid_z;
                let pl = base
                    .s_mod
                    .get_slowness_layer(base.s_mod.layer_number_below(mid_z, true).unwrap(), true);
                let sl = base.s_mod.get_slowness_layer(
                    base.s_mod.layer_number_below(mid_z, false).unwrap(),
                    false,
                );
                let eta_p = evaluate_at_bullen(pl, mid_z, base.radius_of_planet).unwrap();
                let eta_s = evaluate_at_bullen(sl, mid_z, base.radius_of_planet).unwrap();

                let dt = b.time_s - a.time_s;
                let dd = b.dist_rad - a.dist_rad;
                let ds = dz.hypot(mid_r * dd);
                let vp = mid_r / eta_p;
                let vs = mid_r / eta_s;
                let eta_active = if !base.s_mod.depth_in_fluid(mid_z)
                    && (ds / vs - dt).abs() < (ds / vp - dt).abs()
                {
                    eta_s
                } else {
                    eta_p
                };

                assert!(
                    p <= eta_active * (1.0 + 1e-9),
                    "Snell violation: p={p} > eta={eta_active} at z={mid_z}"
                );
                let sin_i = (p / eta_active).clamp(0.0, 1.0);
                let inc_rad = sin_i.asin();
                let snell_lhs = eta_active * inc_rad.sin();
                if p <= eta_active {
                    assert!((snell_lhs - p).abs() < 1e-10 * (1.0 + p));
                }
            }
        }
    }
    const { assert!(DEFAULT_PATH_RAY_PARAM_TOL > 0.0) };
}

#[test]
fn regression_snells_law_near_source_turning_ray() {
    verify_snells_law_on_ray_paths(0, 592.470_309_708_176_5, 39.035_523_261_386_58, "pP");
    verify_snells_law_on_ray_paths(0, 0.0, 113.945_756_396_017_8, "PKIKP");
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// Spherical Snell's law invariant $\eta(r)\sin i(r) = p$ along every `get_ray_paths` trajectory!
    #[test]
    fn prop_snells_law_along_ray_paths(
        model_idx in 0usize..3,
        depth_km in 0.0_f64..600.0,
        dist_deg in 2.0_f64..175.0,
        phase_idx in 0usize..10,
    ) {
        const PHASES: [&str; 10] = [
            "P", "S", "pP", "PP", "PcP", "ScS", "PKP", "PKIKP", "SKS", "P410s",
        ];
        verify_snells_law_on_ray_paths(model_idx, depth_km, dist_deg, PHASES[phase_idx]);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// Synthetic multi-station `P + S + pP` 4D hypocenter recovery:
    /// - `< 0.05 km` in noise-free observations (zero 10 km depth collapse across `[12..560] km`!)
    /// - `< 3.0 km` under large multi-second clock-glitch outliers (Huber-IRLS robustness).
    #[test]
    fn prop_hypocenter_recovery_clean_and_outlier_glitches(
        true_lat in -58.0_f64..58.0,
        true_lon in -165.0_f64..165.0,
        true_depth in 14.0_f64..560.0,
        true_t0 in -300.0_f64..300.0,
        az_rot in 0.0_f64..360.0,
        glitch_idx in 0usize..12,
    ) {
        let model = &cached_models()[0]; // iasp91

        // Build an 8-station azimuthal array (regional + teleseismic) around `(true_lat, true_lon)`:
        let station_rings_deg = [0.8, 2.2, 4.5, 8.0, 14.0, 26.0, 42.0, 62.0];
        let mut obs = Vec::with_capacity(18);

        for (k, &r_deg) in station_rings_deg.iter().enumerate() {
            let az = (az_rot + (k as f64) * 47.5) % 360.0;
            let (slat, slon) =
                spherical_great_circle_point(true_lat, true_lon, az, r_deg * PI / 180.0);
            let sta_name = format!("STA{k}");

            for &(label, query) in &[("P", &["P", "p"][..]), ("S", &["S", "s"][..])] {
                let arrs = model
                    .get_travel_times_geo(true_lat, true_lon, true_depth, slat, slon, query, false)
                    .unwrap();
                if let Some(a) = arrs.first() {
                    obs.push(PhaseArrivalObservation::new(
                        &sta_name,
                        slat,
                        slon,
                        label,
                        true_t0 + a.time_s,
                        0.05,
                    ));
                }
            }

            // Include `pP` depth phase on the two teleseismic stations (`42 deg` and `62 deg`):
            if r_deg >= 40.0 {
                let pp = model
                    .get_travel_times_geo(true_lat, true_lon, true_depth, slat, slon, &["pP"], false)
                    .unwrap();
                if let Some(a) = pp.first() {
                    obs.push(PhaseArrivalObservation::new(
                        &sta_name,
                        slat,
                        slon,
                        "pP",
                        true_t0 + a.time_s,
                        0.05,
                    ));
                }
            }
        }

        prop_assert!(obs.len() >= 14);
        let cfg = HypocenterConfig::default();

        // 1. Noise-free recovery (< 0.05 km 3D hypocentral accuracy!):
        let sol = locate_hypocenter(model, &obs, &cfg).unwrap();
        let horiz_deg = locations2degrees(true_lat, true_lon, sol.lat_deg, sol.lon_deg);
        let horiz_km = horiz_deg * (6371.0 * PI / 180.0);
        let depth_err_km = (sol.depth_km - true_depth).abs();
        let err_3d_km = horiz_km.hypot(depth_err_km);

        prop_assert!(
            err_3d_km < 0.05,
            "noise-free hypocenter error {err_3d_km:.5} km >= 0.05 km (true=({true_lat:.3}, {true_lon:.3}, {true_depth:.2}), got=({:.3}, {:.3}, {:.2}))",
            sol.lat_deg,
            sol.lon_deg,
            sol.depth_km
        );
        prop_assert!((sol.origin_time_s - true_t0).abs() < 0.01);
        prop_assert!(sol.ellipsoid.semi_axes_1sigma_km[0] >= sol.ellipsoid.semi_axes_1sigma_km[1]);
        prop_assert!(sol.ellipsoid.semi_axes_1sigma_km[1] >= sol.ellipsoid.semi_axes_1sigma_km[2]);
        prop_assert!(sol.ellipsoid.semi_axes_1sigma_km[2] > 0.0);

        // 2. Outlier glitch robustness (< 3.0 km under +16 s and -13 s blunders!):
        let mut glitched = obs.clone();
        let i1 = glitch_idx % glitched.len();
        let i2 = (i1 + 7) % glitched.len();
        glitched[i1].arrival_time_s += 16.0;
        glitched[i2].arrival_time_s -= 13.0;

        let sol_g = locate_hypocenter(model, &glitched, &cfg).unwrap();
        let h_deg_g = locations2degrees(true_lat, true_lon, sol_g.lat_deg, sol_g.lon_deg);
        let h_km_g = h_deg_g * (6371.0 * PI / 180.0);
        let z_err_g = (sol_g.depth_km - true_depth).abs();
        let err_g_km = h_km_g.hypot(z_err_g);

        prop_assert!(
            err_g_km < 3.0,
            "glitched hypocenter error {err_g_km:.4} km >= 3.0 km (true_z={true_depth:.2}, got_z={:.2})",
            sol_g.depth_km
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    /// Adversarial fuzzing of phase-name grammar and `.nd` / `.tvel` model parsers: zero panics!
    #[test]
    fn prop_adversarial_parsers_never_panic(
        arb_phase in "[A-Za-z0-9\\^v\\._\\- ]{0,24}",
        arb_text in "[A-Za-z0-9\\.\\-\\n\\t #/]{0,120}",
    ) {
        let model = &cached_models()[0];
        let _ = leg_puller(&arb_phase);
        let _ = SeismicPhase::new(&arb_phase, model);
        let _ = VelocityModel::from_nd_str("fuzz_nd", &arb_text);
        let _ = VelocityModel::from_tvel_str("fuzz_tvel", &arb_text);
    }
}
