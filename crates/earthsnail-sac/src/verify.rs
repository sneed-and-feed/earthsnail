//! Built-in fast self-verification suite (`earthsnail verify [--json]`) auditing core codecs,
//! WGS84 geodesics, Bluestein/Cooley–Tukey FFT Parseval invariants, SOS filter stability,
//! `SACPZ` spectral deconvolution, `TauP` Legendre identities, and Morse spike immunity.

use core::f64::consts::{PI, TAU};
use earthsnail_core::{
    direct_geodesic, gps2dist_azimuth, math, ChannelId, ChannelResponse, Complex64,
    EarthSnailError, KinematicUnit, Result, SacEndian, Trace, TraceStats, UtcTimestamp,
};
use earthsnail_dsp::{
    design_iir_sos, fft, ifft, morse_pick, remove_response, sac_transfer, sosfilt_zerophase_obspy,
    FilterBand, FilterFamily, MorsePickConfig, ResponseRemovalConfig,
};
use earthsnail_io::{
    read_mseed3_bytes, read_mseed_bytes, read_sac_bytes, write_mseed3_trace_bytes,
    write_mseed_trace_bytes, write_sac_bytes, Mseed3WriteOptions, MseedEncoding, MseedWriteOptions,
};
use earthsnail_taup::TauModel;
#[cfg(miri)]
use earthsnail_taup::{SlownessModel, SlownessModelConfig, VelocityModel};

use crate::json::{JsonArr, JsonObj};

/// Result of an individual self-verification check.
#[derive(Debug, Clone)]
pub struct VerifyCheckReceipt {
    /// Short check identifier.
    pub name: &'static str,
    /// Pass/fail status.
    pub passed: bool,
    /// Maximum observed numerical residual or error metric.
    pub metric: f64,
    /// Allowed numerical threshold.
    pub tolerance: f64,
    /// Human-readable check summary.
    pub detail: String,
    /// Elapsed wall-clock milliseconds (`0.0` under Miri isolation).
    pub elapsed_ms: f64,
}

/// Complete report returned by [`run_self_verify`].
#[derive(Debug, Clone)]
pub struct SelfVerifyReport {
    /// Whether every check passed.
    pub all_passed: bool,
    /// Total elapsed wall-clock milliseconds.
    pub total_elapsed_ms: f64,
    /// Individual check receipts.
    pub checks: Vec<VerifyCheckReceipt>,
}

impl SelfVerifyReport {
    /// Serializes this verification report as RFC-8259 JSON.
    pub fn to_json(&self) -> String {
        let mut arr = JsonArr::new();
        for c in &self.checks {
            let mut o = JsonObj::new();
            o.str("name", c.name)
                .bool("passed", c.passed)
                .num("metric", c.metric)
                .num("tolerance", c.tolerance)
                .num("elapsed_ms", c.elapsed_ms)
                .str("detail", &c.detail);
            arr.push_raw(&o.finish());
        }
        let mut root = JsonObj::new();
        root.bool("passed", self.all_passed)
            .uint("total_checks", self.checks.len())
            .uint(
                "passed_checks",
                self.checks.iter().filter(|c| c.passed).count(),
            )
            .num("total_elapsed_ms", self.total_elapsed_ms)
            .raw("checks", &arr.finish());
        root.finish()
    }

    /// Formats a crisp human-readable table.
    pub fn to_pretty_table(&self) -> String {
        let mut out = String::new();
        out.push_str("EarthSnail Built-In Self-Verification Suite\n");
        out.push_str(
            "----------------------------------------------------------------------------\n",
        );
        for c in &self.checks {
            let badge = if c.passed { "[PASS]" } else { "[FAIL]" };
            out.push_str(&format!(
                "{badge} {:<24} ({:>6.2} ms)  metric={:.3e} <= {:.1e}  | {}\n",
                c.name, c.elapsed_ms, c.metric, c.tolerance, c.detail
            ));
        }
        out.push_str(
            "----------------------------------------------------------------------------\n",
        );
        out.push_str(&format!(
            "Summary: {}/{} checks passed in {:.2} ms\n",
            self.checks.iter().filter(|c| c.passed).count(),
            self.checks.len(),
            self.total_elapsed_ms
        ));
        out
    }
}

#[cfg(not(miri))]
#[inline]
fn now_timer() -> Option<std::time::Instant> {
    Some(std::time::Instant::now())
}

#[cfg(miri)]
#[inline]
fn now_timer() -> Option<()> {
    None
}

#[cfg(not(miri))]
#[inline]
fn elapsed_ms(t: Option<std::time::Instant>) -> f64 {
    t.map(|i| i.elapsed().as_secs_f64() * 1_000.0)
        .unwrap_or(0.0)
}

#[cfg(miri)]
#[inline]
fn elapsed_ms(_t: Option<()>) -> f64 {
    0.0
}

/// Runs the built-in self-verification suite across codecs, geodesics, FFT, SOS filters,
/// instrument deconvolution, TauP ray tracing, and Morse phase picking.
pub fn run_self_verify() -> Result<SelfVerifyReport> {
    let t_total = now_timer();
    let mut checks = Vec::with_capacity(7);

    checks.push(check_codecs()?);
    checks.push(check_geodesics()?);
    checks.push(check_fft_parseval()?);
    checks.push(check_sos_filter()?);
    checks.push(check_sacpz_transfer()?);
    if !cfg!(miri) {
        checks.push(check_taup_invariants()?);
    }
    checks.push(check_morse_picker()?);

    let all_passed = checks.iter().all(|c| c.passed);
    let total_elapsed_ms = elapsed_ms(t_total);
    Ok(SelfVerifyReport {
        all_passed,
        total_elapsed_ms,
        checks,
    })
}

fn check_codecs() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    let ts: UtcTimestamp = "2024-07-04T12:00:00.125000Z".parse()?;
    let ch = ChannelId::new("IU", "ANMO", "00", "BHZ");
    let samples: Vec<f64> = (0..64_i32)
        .map(|i| (i * i * 7 - 113 * i + 500) as f64)
        .collect();
    let tr = Trace::new(TraceStats::new(ch, ts, 40.0), samples.clone());

    let sac_le = write_sac_bytes(&tr, SacEndian::Little)?;
    let sac_be = write_sac_bytes(&tr, SacEndian::Big)?;
    let rt_le = read_sac_bytes(&sac_le)?;
    let rt_be = read_sac_bytes(&sac_be)?;

    let ms2 = write_mseed_trace_bytes(
        &tr,
        MseedWriteOptions {
            encoding: MseedEncoding::Steim2,
            ..MseedWriteOptions::default()
        },
    )?;
    let rt_ms2 = read_mseed_bytes(&ms2)?;

    let ms3 = write_mseed3_trace_bytes(
        &tr,
        &Mseed3WriteOptions {
            encoding: MseedEncoding::Steim2,
            ..Mseed3WriteOptions::default()
        },
    )?;
    let rt_ms3 = read_mseed3_bytes(&ms3)?;

    let mut max_diff = 0.0_f64;
    for (i, &ref_s) in samples.iter().enumerate() {
        max_diff = max_diff
            .max((rt_le.data[i] - ref_s).abs())
            .max((rt_be.data[i] - ref_s).abs())
            .max((rt_ms2[0].data[i] - ref_s).abs())
            .max((rt_ms3[0].data[i] - ref_s).abs());
    }
    Ok(VerifyCheckReceipt {
        name: "codecs_sac_mseed_steim2",
        passed: max_diff == 0.0,
        metric: max_diff,
        tolerance: 0.0,
        detail: String::from("SAC LE/BE + miniSEED v2/v3 Steim-2 & CRC-32C 0-bit round-trip"),
        elapsed_ms: elapsed_ms(t0),
    })
}

fn check_geodesics() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    // Tohoku-Oki epicenter -> Canberra (G.CAN) trans-equatorial WGS84 geodesic:
    let inv = gps2dist_azimuth(38.297, 142.373, -35.3187, 148.9963);
    let fwd = direct_geodesic(38.297, 142.373, inv.azimuth_deg, inv.distance_m);
    let back = gps2dist_azimuth(38.297, 142.373, fwd.lat_deg, fwd.lon_deg);
    let dist_err_m = (back.distance_m - inv.distance_m).abs();
    let lat_err_deg = (fwd.lat_deg - (-35.3187)).abs();
    let metric = dist_err_m.max(lat_err_deg * 111_000.0);
    let tol = 1e-6;
    Ok(VerifyCheckReceipt {
        name: "wgs84_karney_geodesics",
        passed: metric <= tol,
        metric,
        tolerance: tol,
        detail: format!(
            "dist={:.3} km, az={:.4}°",
            inv.distance_m * 1e-3,
            inv.azimuth_deg
        ),
        elapsed_ms: elapsed_ms(t0),
    })
}

fn check_fft_parseval() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    let mut max_rel_err = 0.0_f64;
    let sizes = if cfg!(miri) {
        &[64usize, 67usize][..]
    } else {
        &[257usize, 512usize][..]
    };
    for &n in sizes {
        let orig: Vec<Complex64> = (0..n)
            .map(|k| {
                let x = (k as f64) * 0.173;
                Complex64::new(
                    math::sin(x) + 0.4 * math::cos(2.3 * x),
                    -0.25 * math::sin(1.7 * x),
                )
            })
            .collect();
        let e_time: f64 = orig.iter().map(|z| z.norm_sqr()).sum();
        let spec = fft(&orig)?;
        let e_freq: f64 = spec.iter().map(|z| z.norm_sqr()).sum::<f64>() / (n as f64);
        let parseval_err = (e_time - e_freq).abs() / e_time.max(1e-30);
        let rec = ifft(&spec)?;
        let rt_err = orig
            .iter()
            .zip(rec.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_rel_err = max_rel_err.max(parseval_err).max(rt_err);
    }
    let tol = 1e-11;
    Ok(VerifyCheckReceipt {
        name: "fft_bluestein_parseval",
        passed: max_rel_err <= tol,
        metric: max_rel_err,
        tolerance: tol,
        detail: String::from("Composite & prime-length FFT Parseval + IFFT identity"),
        elapsed_ms: elapsed_ms(t0),
    })
}

fn check_sos_filter() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    let fs = 100.0;
    let sos = design_iir_sos(
        4,
        FilterBand::Bandpass(1.5, 18.0),
        fs,
        FilterFamily::Butterworth,
    )?;
    if !sos.is_stable() {
        return Err(EarthSnailError::Numerical(
            "synthesized Butterworth SOS filter has unstable pole".into(),
        ));
    }
    let n = if cfg!(miri) { 200 } else { 800 };
    let taper_len = 24usize;
    let f0 = 5.0; // strictly centered inside [1.5, 18.0] Hz passband (20 samples/cycle)
    let sig: Vec<f64> = (0..n)
        .map(|i| {
            let env = if i < taper_len {
                0.5 * (1.0 - math::cos(PI * (i as f64) / (taper_len as f64)))
            } else if i + taper_len >= n {
                0.5 * (1.0 - math::cos(PI * ((n - 1 - i) as f64) / (taper_len as f64)))
            } else {
                1.0
            };
            env * math::cos(TAU * f0 * (i as f64) / fs)
        })
        .collect();
    let y = sosfilt_zerophase_obspy(&sos, &sig);
    // Peak at midpoint n/2 (100 or 400 -> cos(2*pi*5.0*(n/2)/100) = +1.0) has zero phase lag:
    let mid = n / 2;
    let gain_err = (y[mid] - sig[mid]).abs();
    let tol = 5e-3;
    Ok(VerifyCheckReceipt {
        name: "sos_butter_zerophase",
        passed: gain_err <= tol,
        metric: gain_err,
        tolerance: tol,
        detail: format!(
            "{} biquad sections, all |p_i| < 1, zero group delay",
            sos.sections.len()
        ),
        elapsed_ms: elapsed_ms(t0),
    })
}

fn check_sacpz_transfer() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    let fs = if cfg!(miri) { 10.0 } else { 40.0 };
    let npts = if cfg!(miri) { 128 } else { 512 };
    let p1 = Complex64::new(-0.037_004, 0.037_016);
    let resp = ChannelResponse::from_sacpz(
        vec![Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
        vec![p1, p1.conj()],
        1.5e9,
    );
    let mid_t = 0.5 * (npts as f64) / fs;
    let vel_true: Vec<f64> = (0..npts)
        .map(|i| {
            let t = (i as f64) / fs;
            let u = PI * 1.0 * (t - mid_t);
            (1.0 - 2.0 * u * u) * math::exp(-u * u) * 1e-4
        })
        .collect();
    let counts = sac_transfer(
        &vel_true,
        fs,
        None,
        Some(&resp),
        KinematicUnit::Velocity,
        None,
        None,
    )?;
    let flim = if cfg!(miri) {
        [0.02, 0.05, 3.5, 4.8]
    } else {
        [0.01, 0.03, 12.0, 18.0]
    };
    let rec = remove_response(
        &counts,
        fs,
        &resp,
        &ResponseRemovalConfig {
            output: KinematicUnit::Velocity,
            pre_filt_hz: Some(flim),
            water_level_db: Some(60.0),
            ..ResponseRemovalConfig::default()
        },
    )?;
    let mut err_sq = 0.0;
    let mut ref_sq = 0.0;
    for i in (npts / 4)..(3 * npts / 4) {
        let d = rec[i] - vel_true[i];
        err_sq += d * d;
        ref_sq += vel_true[i] * vel_true[i];
    }
    let rel_rmse = math::sqrt(err_sq / ref_sq.max(1e-30));
    let tol = 5e-3;
    Ok(VerifyCheckReceipt {
        name: "sacpz_deconvolution",
        passed: rel_rmse <= tol,
        metric: rel_rmse,
        tolerance: tol,
        detail: String::from("120s STS-2 SACPZ forward/inverse spectral recovery"),
        elapsed_ms: elapsed_ms(t0),
    })
}

fn check_taup_invariants() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    #[cfg(miri)]
    let (tm, p_err): (TauModel, f64) = {
        const COMPACT_EARTH_ND: &str = "\
0.0    5.80  3.36  2.72
20.0   5.80  3.36  2.72
conrad
20.0   6.50  3.75  2.92
35.0   6.50  3.75  2.92
moho
35.0   8.04  4.47  3.32
410.0  9.03  4.87  3.54
410.0  9.36  5.07  3.72
660.0  10.20 5.60  3.99
660.0  10.79 5.95  4.38
2889.0 13.69 7.30  5.56
outer-core
2889.0 8.00  0.00  9.90
5153.9 10.26 0.00  12.14
inner-core
5153.9 11.12 3.50  12.70
6371.0 11.24 3.56  13.01
";
        let vm = VelocityModel::from_nd_str("compact_iasp91", COMPACT_EARTH_ND)?;
        let cfg = SlownessModelConfig {
            min_delta_p: 5.0,
            max_delta_p: 150.0,
            max_depth_interval: 1200.0,
            max_range_interval_rad: 45.0 * PI / 180.0,
            max_interp_error: 2.0,
            allow_inner_core_s: true,
            slowness_tolerance: 1e-16,
        };
        let tm = TauModel::from_slowness_model(SlownessModel::with_config(vm, cfg)?)?;
        let arrivals =
            tm.get_travel_times(29.0, 73.870_787_765_677_22, &["P", "S", "PcP", "SKS"])?;
        let p_arr = arrivals
            .iter()
            .find(|a| a.name == "P")
            .ok_or_else(|| EarthSnailError::ModelError("missing P arrival".into()))?;
        let s_arr = arrivals
            .iter()
            .find(|a| a.name == "S")
            .ok_or_else(|| EarthSnailError::ModelError("missing S arrival".into()))?;
        let ord_err = if p_arr.time_s > 600.0 && p_arr.time_s < 800.0 && s_arr.time_s > p_arr.time_s
        {
            0.0
        } else {
            1.0
        };
        (tm, ord_err)
    };
    #[cfg(not(miri))]
    let (tm, p_err) = {
        let tm = TauModel::iasp91()?;
        let arrivals =
            tm.get_travel_times(29.0, 73.870_787_765_677_22, &["P", "S", "PcP", "SKS"])?;
        let p_arr = arrivals
            .iter()
            .find(|a| a.name == "P")
            .ok_or_else(|| EarthSnailError::ModelError("missing P arrival on iasp91".into()))?;
        // Golden Tohoku-Oki @ G.CAN P travel time is 692.08444774 s:
        let err = (p_arr.time_s - 692.084_447_743_431_7).abs();
        (tm, err)
    };

    let p_arr_time = tm
        .get_travel_times(29.0, 73.870_787_765_677_22, &["P"])?
        .first()
        .map(|a| a.time_s)
        .unwrap_or(0.0);
    let curves = tm.get_travel_time_curves(29.0, &["P"], if cfg!(miri) { 6 } else { 16 })?;
    let mut legendre_err = 0.0_f64;
    for br in &curves[0].branches {
        for pt in &br.points {
            let lhs = pt.time_s;
            let rhs = pt.tau_s + pt.ray_param_s_per_rad * pt.dist_rad;
            legendre_err = legendre_err.max((lhs - rhs).abs());
        }
    }
    let metric = p_err.max(legendre_err);
    let tol = 5e-3;
    Ok(VerifyCheckReceipt {
        name: "taup_iasp91_legendre",
        passed: metric <= tol,
        metric,
        tolerance: tol,
        detail: format!("iasp91 P={p_arr_time:.3}s, T(p)=tau(p)+p*Delta(p) err={legendre_err:.2e}"),
        elapsed_ms: elapsed_ms(t0),
    })
}

fn check_morse_picker() -> Result<VerifyCheckReceipt> {
    let t0 = now_timer();
    let n = if cfg!(miri) { 180 } else { 400 };
    let cfg = MorsePickConfig {
        nsta: if cfg!(miri) { 8 } else { 15 },
        nlta: if cfg!(miri) { 50 } else { 100 },
        thr_on: 4.5,
        prominence_sigma: 5.5,
        confirm_samples: 4,
    };
    let mut spiky = vec![0.0_f64; n];
    for (i, slot) in spiky.iter_mut().enumerate() {
        *slot = 0.01 * math::sin((i as f64) * 0.41);
    }
    spiky[25] = 1.0e9;
    spiky[90] = -4.0e8;
    let false_pick = morse_pick(&spiky, &cfg)?.pick_sample;

    let onset = if cfg!(miri) { 110 } else { 220 };
    for (i, slot) in spiky.iter_mut().enumerate().skip(onset) {
        let dt = ((i - onset) as f64) * 0.01;
        *slot += 4.0 * math::exp(-1.5 * dt) * math::sin(22.0 * dt + 0.3);
    }
    let true_pick = morse_pick(&spiky, &cfg)?.pick_sample;
    let (passed, err_samples) = match (false_pick, true_pick) {
        (None, Some(idx)) => {
            let diff = (idx as isize - onset as isize).unsigned_abs() as f64;
            (diff <= 8.0, diff)
        }
        _ => (false, 999.0),
    };
    Ok(VerifyCheckReceipt {
        name: "morse_spike_immunity",
        passed,
        metric: err_samples,
        tolerance: 8.0,
        detail: String::from("0 false triggers on 1e9 spike; accurate onset recovery"),
        elapsed_ms: elapsed_ms(t0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_verify_passes_all_checks() {
        let rep = run_self_verify().unwrap();
        assert!(rep.all_passed, "{}", rep.to_pretty_table());
        let j = rep.to_json();
        assert!(j.contains("\"passed\":true"));
    }
}
