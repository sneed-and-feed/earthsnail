//! Spectral instrument response deconvolution (`remove_response`) and forward/inverse instrument
//! transfer simulation (`sac_transfer`) supporting multi-stage `StationXML` (`ChannelResponse`)
//! and continuous-time `SACPZ` across `Displacement`, `Velocity`, and `Acceleration`.

use crate::fft::{irfft_with_scratch, next_pow2, npts2nfft, rfft_with_scratch};
use crate::preprocess::{cosine_taper, demean_in_place};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
use earthsnail_core::inventory::{
    CoefficientsStage, FirStage, FirSymmetry, PolesZerosStage, PzTransferFunctionType,
    ResponseStage,
};
use earthsnail_core::{
    math, ChannelResponse, Complex64, EarthSnailError, KinematicUnit, Result, ScratchWorkspace,
    Trace,
};

/// Configuration parameters for spectral instrument response removal (`ObsPy Trace.remove_response`
/// and `SAC transfer` compatible).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResponseRemovalConfig {
    /// Target ground-motion physical quantity (`Displacement`, `Velocity`, or `Acceleration`).
    pub output: KinematicUnit,
    /// Optional 4-corner frequency-domain SAC cosine bandpass taper `[f1, f2, f3, f4]` in Hz.
    pub pre_filt_hz: Option<[f64; 4]>,
    /// Optional water-level regularization floor in dB below `max(|H(f)|)` (`Some(60.0)` default).
    pub water_level_db: Option<f64>,
    /// Subtract mean in the time domain prior to tapering and FFT (`true` default).
    pub zero_mean: bool,
    /// Apply time-domain SAC quarter-cosine taper prior to FFT (`true` default).
    pub taper: bool,
    /// Total tapered fraction passed to `cosine_taper(npts, taper_fraction, false, true)` (`0.05` default,
    /// matching `Trace.remove_response(taper_fraction=0.05)` and `simulate_seismometer(..., sacsim=True)`).
    pub taper_fraction: f64,
    /// Zero-pad FFT to `next_pow2(2 * npts)` instead of `npts2nfft(npts)` (`false` default).
    pub nfft_pow2: bool,
    /// When `true` and `ChannelResponse::instrument_sensitivity` is `Some(sens)`, rescales `H(f)` so
    /// `|H(sens.frequency_hz)|` in native `input_units` equals `sens.value`. Default `false` matches
    /// FDSN `evalresp` stage-cascade evaluation (`< 1e-15` relative error on `IU.ANMO.10.BHZ`).
    pub rescale_to_overall_sensitivity: bool,
}

impl Default for ResponseRemovalConfig {
    fn default() -> Self {
        Self {
            output: KinematicUnit::Velocity,
            pre_filt_hz: None,
            water_level_db: Some(60.0),
            zero_mean: true,
            taper: true,
            taper_fraction: 0.05,
            nfft_pow2: false,
            rescale_to_overall_sensitivity: false,
        }
    }
}

#[inline]
fn validate_pre_filt(flimit: [f64; 4]) -> Result<()> {
    let [f1, f2, f3, f4] = flimit;
    if !f1.is_finite()
        || !f2.is_finite()
        || !f3.is_finite()
        || !f4.is_finite()
        || f1 < 0.0
        || f1 > f2
        || f2 > f3
        || f3 > f4
    {
        return Err(EarthSnailError::InvalidParameter(format!(
            "invalid pre_filt corner frequencies [{f1}, {f2}, {f3}, {f4}]; expected 0 <= f1 <= f2 <= f3 <= f4"
        )));
    }
    Ok(())
}

#[inline]
fn eval_sac_taper_point(f: f64, [f1, f2, f3, f4]: [f64; 4]) -> f64 {
    if f > f2 && f < f3 {
        1.0
    } else if f >= f1 && f <= f2 {
        if f2 > f1 {
            0.5 * (1.0 - math::cos(PI * (f - f1) / (f2 - f1)))
        } else {
            1.0
        }
    } else if f >= f3 && f <= f4 {
        if f4 > f3 {
            0.5 * (1.0 + math::cos(PI * (f - f3) / (f4 - f3)))
        } else {
            1.0
        }
    } else {
        0.0
    }
}

/// Frequency-domain 4-corner SAC cosine bandpass taper (`obspy.signal.invsim.cosine_sac_taper`):
/// - `0.0` for `f < f1` or `f > f4`,
/// - `0.5 * (1 - cos(pi * (f - f1) / (f2 - f1)))` on `[f1, f2]`,
/// - `1.0` on `(f2, f3)`,
/// - `0.5 * (1 + cos(pi * (f - f3) / (f4 - f3)))` on `[f3, f4]`.
pub fn cosine_sac_taper(freqs_hz: &[f64], flimit: [f64; 4]) -> Result<Vec<f64>> {
    validate_pre_filt(flimit)?;
    Ok(freqs_hz
        .iter()
        .map(|&f| eval_sac_taper_point(f, flimit))
        .collect())
}

/// In-place water-level spectral inversion matching `obspy.signal.invsim.invert_spectrum`.
/// Returns the number of non-zero bins clamped to the water-level floor.
pub fn invert_spectrum_waterlevel(
    spec: &mut [Complex64],
    water_level_db: Option<f64>,
) -> Result<usize> {
    if spec.is_empty() {
        return Ok(0);
    }
    match water_level_db {
        Some(wlev) => {
            if !wlev.is_finite() || wlev < 0.0 {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "water_level_db={wlev} must be finite and >= 0"
                )));
            }
            let max_amp = spec.iter().map(|z| z.abs()).fold(0.0_f64, f64::max);
            let swamp = max_amp * math::powf(10.0, -wlev / 20.0);
            let mut clamped_count = 0_usize;
            for z in spec.iter_mut() {
                let len = z.abs();
                if len == 0.0 {
                    *z = Complex64::ZERO;
                } else if len < swamp {
                    // `z_clamped = z * (swamp / len)`, and `(z_clamped)^{-1} = conj(z) / (len * swamp)`.
                    let scale = 1.0 / (len * swamp);
                    *z = Complex64::new(z.re * scale, -z.im * scale);
                    clamped_count += 1;
                } else {
                    *z = z.inv();
                }
            }
            Ok(clamped_count)
        }
        None => {
            spec[0] = Complex64::ZERO;
            for z in spec.iter_mut().skip(1) {
                if z.re == 0.0 && z.im == 0.0 {
                    *z = Complex64::ZERO;
                } else {
                    *z = z.inv();
                }
            }
            Ok(0)
        }
    }
}

// ============================================================================
// Vectorized / Horner / Chebyshev Uniform-Grid Stage Response Evaluator
// ============================================================================

struct FactoredLaplacePoly {
    origin_count: i32,
    neg_reals: Vec<f64>,
    /// `(re^2 + im^2, -2.0 * re)` for conjugate pairs `(re + i*im, re - i*im)`.
    conj_quads: Vec<(f64, f64)>,
    unpaired: Vec<Complex64>,
}

impl FactoredLaplacePoly {
    fn from_roots(roots: &[Complex64]) -> Self {
        let mut origin_count = 0_i32;
        let mut neg_reals = Vec::new();
        let mut pos_im = Vec::new();
        let mut neg_im = Vec::new();

        for &r in roots {
            if r.re == 0.0 && r.im == 0.0 {
                origin_count += 1;
            } else if r.im == 0.0 {
                neg_reals.push(-r.re);
            } else if r.im > 0.0 {
                pos_im.push(r);
            } else {
                neg_im.push(r);
            }
        }

        let cmp_root = |a: &Complex64, b: &Complex64| {
            if (a.re - b.re).abs() > 1e-12 * (1.0 + a.abs()) {
                a.re.total_cmp(&b.re)
            } else {
                a.im.abs().total_cmp(&b.im.abs())
            }
        };
        pos_im.sort_by(cmp_root);
        neg_im.sort_by(cmp_root);

        let mut conj_quads = Vec::new();
        let mut unpaired = Vec::new();
        let mut ip = 0_usize;
        let mut ineg = 0_usize;
        while ip < pos_im.len() && ineg < neg_im.len() {
            let a = pos_im[ip];
            let b = neg_im[ineg];
            let tol = 1e-12 * (1.0 + a.abs());
            if (a.re - b.re).abs() <= tol && (a.im + b.im).abs() <= tol {
                let re = 0.5 * (a.re + b.re);
                let im = 0.5 * (a.im - b.im);
                conj_quads.push((re * re + im * im, -2.0 * re));
                ip += 1;
                ineg += 1;
            } else if cmp_root(&a, &b).is_lt() {
                unpaired.push(a);
                ip += 1;
            } else {
                unpaired.push(b);
                ineg += 1;
            }
        }
        unpaired.extend_from_slice(&pos_im[ip..]);
        unpaired.extend_from_slice(&neg_im[ineg..]);

        Self {
            origin_count,
            neg_reals,
            conj_quads,
            unpaired,
        }
    }

    #[inline]
    fn eval_non_origin(&self, u: f64) -> Complex64 {
        let u2 = u * u;
        let mut acc = Complex64::ONE;
        for &(c0, c1) in &self.conj_quads {
            let q_re = c0 - u2;
            let q_im = c1 * u;
            acc = Complex64::new(acc.re * q_re - acc.im * q_im, acc.re * q_im + acc.im * q_re);
        }
        for &nr in &self.neg_reals {
            acc = Complex64::new(acc.re * nr - acc.im * u, acc.re * u + acc.im * nr);
        }
        for &r in &self.unpaired {
            let f_re = -r.re;
            let f_im = u - r.im;
            acc = Complex64::new(acc.re * f_re - acc.im * f_im, acc.re * f_im + acc.im * f_re);
        }
        acc
    }
}

enum FastStage<'a> {
    LaplacePz {
        orig: &'a PolesZerosStage,
        u_scale: f64,
        zeros: FactoredLaplacePoly,
        poles: FactoredLaplacePoly,
        can_batch_product: bool,
    },
    DigitalPz {
        orig: &'a PolesZerosStage,
        fs_hz: f64,
    },
    HornerDigital {
        b_rev: Vec<f64>,
        a_rev: Vec<f64>,
        fs_hz: f64,
    },
    SymmetricFirOdd {
        c_center: f64,
        c_sym_2x: Vec<f64>,
        fs_hz: f64,
    },
    SymmetricFirEven {
        c_sym_2x: Vec<f64>,
        fs_hz: f64,
    },
}

#[inline]
fn horner_eval_rev(taps_rev: &[f64], z_inv: Complex64) -> Complex64 {
    let Some((&first, rest)) = taps_rev.split_first() else {
        return Complex64::ONE;
    };
    let mut acc = Complex64::new(first, 0.0);
    for &c in rest {
        acc = Complex64::new(
            acc.re * z_inv.re - acc.im * z_inv.im + c,
            acc.re * z_inv.im + acc.im * z_inv.re,
        );
    }
    acc
}

#[inline]
fn mul_iw_pow(z: Complex64, iw_mag: f64, exp: i32) -> Complex64 {
    match exp {
        0 => z,
        1 => Complex64::new(-z.im * iw_mag, z.re * iw_mag),
        -1 => {
            let inv = 1.0 / iw_mag;
            Complex64::new(z.im * inv, -z.re * inv)
        }
        2 => {
            let s = -(iw_mag * iw_mag);
            Complex64::new(z.re * s, z.im * s)
        }
        -2 => {
            let s = -1.0 / (iw_mag * iw_mag);
            Complex64::new(z.re * s, z.im * s)
        }
        other => z * Complex64::new(0.0, iw_mag).powi(other),
    }
}

fn compile_fast_plan<'a>(
    response: &'a ChannelResponse,
    target: KinematicUnit,
    gain_scale: f64,
) -> (f64, f64, i32, Vec<FastStage<'a>>) {
    let mut combined_gain = gain_scale;
    let mut total_correction_s = 0.0_f64;
    let mut stages = Vec::new();

    if response.stages.is_empty() {
        if let Some(sens) = response.instrument_sensitivity {
            combined_gain *= sens.value;
        }
    }

    for st in &response.stages {
        match st {
            ResponseStage::GainOnly(g) => {
                combined_gain *= g.value;
            }
            ResponseStage::Polynomial(poly) => {
                let c1 = poly.coefficients.get(1).copied().unwrap_or(1.0);
                if c1 != 0.0 {
                    combined_gain /= c1;
                }
            }
            ResponseStage::PolesZeros(pz) => {
                if pz.zeros.is_empty() && pz.poles.is_empty() {
                    combined_gain *= pz.a0 * pz.stage_gain;
                } else {
                    match pz.pz_transfer_function_type {
                        PzTransferFunctionType::RadPerSec | PzTransferFunctionType::Hz => {
                            combined_gain *= pz.a0 * pz.stage_gain;
                            let u_scale = match pz.pz_transfer_function_type {
                                PzTransferFunctionType::RadPerSec => TAU,
                                PzTransferFunctionType::Hz => 1.0,
                                PzTransferFunctionType::DigitalZTransform => 1.0,
                            };
                            let can_batch = pz.zeros.len() <= 36 && pz.poles.len() <= 36;
                            stages.push(FastStage::LaplacePz {
                                orig: pz,
                                u_scale,
                                zeros: FactoredLaplacePoly::from_roots(&pz.zeros),
                                poles: FactoredLaplacePoly::from_roots(&pz.poles),
                                can_batch_product: can_batch,
                            });
                        }
                        PzTransferFunctionType::DigitalZTransform => {
                            combined_gain *= pz.a0 * pz.stage_gain;
                            let fs_hz = pz.input_sample_rate_hz.unwrap_or(1.0).max(1e-30);
                            stages.push(FastStage::DigitalPz { orig: pz, fs_hz });
                        }
                    }
                }
            }
            ResponseStage::Coefficients(CoefficientsStage {
                numerator,
                denominator,
                input_sample_rate_hz,
                correction_s,
                stage_gain,
                gain_frequency_hz,
            }) => {
                if *input_sample_rate_hz <= 0.0 || (numerator.is_empty() && denominator.is_empty())
                {
                    combined_gain *= *stage_gain;
                } else {
                    let norm = st.eval_freq(*gain_frequency_hz).abs();
                    // Note `st.eval_freq(gain_freq)` already equals `stage_gain` in magnitude when normalized,
                    // whereas raw normalization divides by `|eval_raw(gain_freq)|`:
                    let z0 = Complex64::cis(-TAU * (*gain_frequency_hz) / (*input_sample_rate_hz));
                    let b_rev: Vec<f64> = numerator.iter().copied().rev().collect();
                    let a_rev: Vec<f64> = denominator.iter().copied().rev().collect();
                    let raw0 = horner_eval_rev(&b_rev, z0) / horner_eval_rev(&a_rev, z0);
                    let raw_norm = raw0.abs();
                    let _ = norm;
                    if raw_norm > 0.0 && raw_norm.is_finite() {
                        combined_gain *= *stage_gain / raw_norm;
                    } else {
                        combined_gain *= *stage_gain;
                    }
                    total_correction_s += *correction_s;
                    stages.push(FastStage::HornerDigital {
                        b_rev,
                        a_rev,
                        fs_hz: *input_sample_rate_hz,
                    });
                }
            }
            ResponseStage::Fir(FirStage {
                symmetry,
                coefficients,
                input_sample_rate_hz,
                correction_s,
                stage_gain,
                gain_frequency_hz,
                ..
            }) => {
                if coefficients.is_empty() || *input_sample_rate_hz <= 0.0 {
                    combined_gain *= *stage_gain;
                } else {
                    let dt = 1.0 / (*input_sample_rate_hz);
                    match symmetry {
                        FirSymmetry::None => {
                            let c_rev: Vec<f64> = coefficients.iter().copied().rev().collect();
                            let z0 = Complex64::cis(-TAU * (*gain_frequency_hz) * dt);
                            let raw_norm = horner_eval_rev(&c_rev, z0).abs();
                            if raw_norm > 0.0 && raw_norm.is_finite() {
                                combined_gain *= *stage_gain / raw_norm;
                            } else {
                                combined_gain *= *stage_gain;
                            }
                            total_correction_s += *correction_s;
                            stages.push(FastStage::HornerDigital {
                                b_rev: c_rev,
                                a_rev: Vec::new(),
                                fs_hz: *input_sample_rate_hz,
                            });
                        }
                        FirSymmetry::Odd => {
                            let l = coefficients.len() - 1;
                            let c_center = coefficients[l];
                            let mut c_sym_2x = Vec::with_capacity(l);
                            for d in 1..=l {
                                c_sym_2x.push(2.0 * coefficients[l - d]);
                            }
                            let cos0 = math::cos(TAU * (*gain_frequency_hz) * dt);
                            let raw_norm = eval_sym_odd_amp(c_center, &c_sym_2x, cos0).abs();
                            if raw_norm > 0.0 && raw_norm.is_finite() {
                                combined_gain *= *stage_gain / raw_norm;
                            } else {
                                combined_gain *= *stage_gain;
                            }
                            total_correction_s += *correction_s - (l as f64) * dt;
                            stages.push(FastStage::SymmetricFirOdd {
                                c_center,
                                c_sym_2x,
                                fs_hz: *input_sample_rate_hz,
                            });
                        }
                        FirSymmetry::Even => {
                            let l = coefficients.len();
                            let center = ((2 * l - 1) as f64) * 0.5;
                            let mut c_sym_2x = Vec::with_capacity(l);
                            for d in 0..l {
                                c_sym_2x.push(2.0 * coefficients[l - 1 - d]);
                            }
                            let cos_half0 = math::cos(0.5 * TAU * (*gain_frequency_hz) * dt);
                            let raw_norm = eval_sym_even_amp(&c_sym_2x, cos_half0).abs();
                            if raw_norm > 0.0 && raw_norm.is_finite() {
                                combined_gain *= *stage_gain / raw_norm;
                            } else {
                                combined_gain *= *stage_gain;
                            }
                            total_correction_s += *correction_s - center * dt;
                            stages.push(FastStage::SymmetricFirEven {
                                c_sym_2x,
                                fs_hz: *input_sample_rate_hz,
                            });
                        }
                    }
                }
            }
        }
    }

    let kin_diff = match (
        response.input_units.derivative_order(),
        target.derivative_order(),
    ) {
        (Some(ord_in), Some(ord_out)) => ord_in - ord_out,
        _ => 0,
    };

    (combined_gain, total_correction_s, kin_diff, stages)
}

#[inline]
fn eval_sym_odd_amp(c_center: f64, c_sym_2x: &[f64], cos_theta: f64) -> f64 {
    let Some((&first, rest)) = c_sym_2x.split_first() else {
        return c_center;
    };
    let two_c = 2.0 * cos_theta;
    let mut t_prev2 = 1.0_f64;
    let mut t_prev1 = cos_theta;
    let mut amp = c_center + first * t_prev1;
    for &c2 in rest {
        let t_cur = two_c * t_prev1 - t_prev2;
        amp += c2 * t_cur;
        t_prev2 = t_prev1;
        t_prev1 = t_cur;
    }
    amp
}

#[inline]
fn eval_sym_even_amp(c_sym_2x: &[f64], cos_half: f64) -> f64 {
    let Some((&first, rest)) = c_sym_2x.split_first() else {
        return 0.0;
    };
    let cos_theta = 2.0 * cos_half * cos_half - 1.0;
    let two_c = 2.0 * cos_theta;
    let mut u_prev2 = cos_half;
    let mut u_prev1 = cos_half;
    let mut amp = first * u_prev1;
    for &c2 in rest {
        let u_cur = two_c * u_prev1 - u_prev2;
        amp += c2 * u_cur;
        u_prev2 = u_prev1;
        u_prev1 = u_cur;
    }
    amp
}

/// Evaluate `response.eval_freq_kinematic(k * df_hz, target) * gain_scale` across uniform bins
/// `k in k_start..=k_end` (`k >= 1`) into `out[0..n_bins]` (`out[0] = 0` for DC).
#[allow(clippy::too_many_arguments)]
fn eval_response_uniform_grid(
    response: &ChannelResponse,
    target: KinematicUnit,
    gain_scale: f64,
    n_bins: usize,
    df_hz: f64,
    k_start: usize,
    k_end: usize,
    out: &mut Vec<Complex64>,
) {
    out.clear();
    out.resize(n_bins, Complex64::ZERO);
    if n_bins <= 1 || k_start > k_end || k_end >= n_bins {
        return;
    }
    let k_lo = k_start.max(1);
    if k_lo > k_end {
        return;
    }

    let (combined_gain, total_correction_s, mut net_tau_pow, stages) =
        compile_fast_plan(response, target, gain_scale);

    for slot in &mut out[k_lo..=k_end] {
        *slot = Complex64::new(combined_gain, 0.0);
    }

    for st in &stages {
        match st {
            FastStage::LaplacePz {
                orig,
                u_scale,
                zeros,
                poles,
                can_batch_product,
            } => {
                if *can_batch_product && (*u_scale - TAU).abs() == 0.0 {
                    // Fold `(i * TAU * f)^(origin_zeros - origin_poles)` directly into `net_tau_pow`!
                    net_tau_pow += zeros.origin_count - poles.origin_count;
                    let du = TAU * df_hz;
                    for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                        let k = k_lo + idx;
                        let u = (k as f64) * du;
                        let num = zeros.eval_non_origin(u);
                        let den = poles.eval_non_origin(u);
                        let den_n2 = den.norm_sqr();
                        if den_n2 > 1e-250 && den_n2 < 1e250 && num.is_finite() && den.is_finite() {
                            *slot *= num / den;
                        } else {
                            let f = (k as f64) * df_hz;
                            let exact_no_gain = orig.eval_freq(f) / (orig.a0 * orig.stage_gain);
                            // Undo `origin_count` since `net_tau_pow` will apply it at the end:
                            let orig_diff = zeros.origin_count - poles.origin_count;
                            *slot *= mul_iw_pow(exact_no_gain, u, -orig_diff);
                        }
                    }
                } else if *can_batch_product {
                    let orig_diff = zeros.origin_count - poles.origin_count;
                    let du = (*u_scale) * df_hz;
                    for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                        let k = k_lo + idx;
                        let u = (k as f64) * du;
                        let num = zeros.eval_non_origin(u);
                        let den = poles.eval_non_origin(u);
                        let den_n2 = den.norm_sqr();
                        if den_n2 > 1e-250 && den_n2 < 1e250 && num.is_finite() && den.is_finite() {
                            *slot *= mul_iw_pow(num / den, u, orig_diff);
                        } else {
                            let f = (k as f64) * df_hz;
                            *slot *= orig.eval_freq(f) / (orig.a0 * orig.stage_gain);
                        }
                    }
                } else {
                    let inv_ag = 1.0 / (orig.a0 * orig.stage_gain);
                    for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                        let f = ((k_lo + idx) as f64) * df_hz;
                        *slot *= orig.eval_freq(f) * inv_ag;
                    }
                }
            }
            FastStage::DigitalPz { orig, fs_hz } => {
                let dtheta = TAU * df_hz / (*fs_hz);
                let z_step = Complex64::cis(dtheta);
                let mut z_var = Complex64::ONE;
                for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                    let k = k_lo + idx;
                    if (idx & 63) == 0 {
                        z_var = Complex64::cis(dtheta * (k as f64));
                    }
                    let mut acc = Complex64::ONE;
                    let shared = orig.zeros.len().min(orig.poles.len());
                    let mut singular = false;
                    for i in 0..shared {
                        let den = z_var - orig.poles[i];
                        if den.re == 0.0 && den.im == 0.0 {
                            singular = true;
                            break;
                        }
                        acc *= (z_var - orig.zeros[i]) / den;
                    }
                    if singular {
                        *slot = Complex64::ZERO;
                    } else {
                        for &zv in &orig.zeros[shared..] {
                            acc *= z_var - zv;
                        }
                        for &pv in &orig.poles[shared..] {
                            let den = z_var - pv;
                            if den.re == 0.0 && den.im == 0.0 {
                                acc = Complex64::ZERO;
                                break;
                            }
                            acc /= den;
                        }
                        *slot *= acc;
                    }
                    z_var *= z_step;
                }
            }
            FastStage::HornerDigital {
                b_rev,
                a_rev,
                fs_hz,
            } => {
                let dtheta = -TAU * df_hz / (*fs_hz);
                let w_step = Complex64::cis(dtheta);
                let mut z_inv = Complex64::ONE;
                if a_rev.is_empty() {
                    for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                        let k = k_lo + idx;
                        if (idx & 63) == 0 {
                            z_inv = Complex64::cis(dtheta * (k as f64));
                        }
                        *slot *= horner_eval_rev(b_rev, z_inv);
                        z_inv *= w_step;
                    }
                } else {
                    for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                        let k = k_lo + idx;
                        if (idx & 63) == 0 {
                            z_inv = Complex64::cis(dtheta * (k as f64));
                        }
                        let num = horner_eval_rev(b_rev, z_inv);
                        let den = horner_eval_rev(a_rev, z_inv);
                        *slot *= num / den;
                        z_inv *= w_step;
                    }
                }
            }
            FastStage::SymmetricFirOdd {
                c_center,
                c_sym_2x,
                fs_hz,
            } => {
                let dtheta = TAU * df_hz / (*fs_hz);
                let w_step = Complex64::cis(dtheta);
                let mut z_pos = Complex64::ONE;
                for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                    let k = k_lo + idx;
                    if (idx & 63) == 0 {
                        z_pos = Complex64::cis(dtheta * (k as f64));
                    }
                    let amp = eval_sym_odd_amp(*c_center, c_sym_2x, z_pos.re);
                    *slot = slot.scale(amp);
                    z_pos *= w_step;
                }
            }
            FastStage::SymmetricFirEven { c_sym_2x, fs_hz } => {
                let dhalf = 0.5 * TAU * df_hz / (*fs_hz);
                let w_step = Complex64::cis(dhalf);
                let mut z_half = Complex64::ONE;
                for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
                    let k = k_lo + idx;
                    if (idx & 63) == 0 {
                        z_half = Complex64::cis(dhalf * (k as f64));
                    }
                    let amp = eval_sym_even_amp(c_sym_2x, z_half.re);
                    *slot = slot.scale(amp);
                    z_half *= w_step;
                }
            }
        }
    }

    // Final fused pass for net `(i * TAU * f)^net_tau_pow` + `exp(i * TAU * f * total_correction_s)`:
    let has_corr = total_correction_s != 0.0;
    if has_corr || net_tau_pow != 0 {
        let dcorr = TAU * df_hz * total_correction_s;
        let corr_step = if has_corr {
            Complex64::cis(dcorr)
        } else {
            Complex64::ONE
        };
        let mut corr = Complex64::ONE;
        let dw = TAU * df_hz;
        for (idx, slot) in out[k_lo..=k_end].iter_mut().enumerate() {
            let k = k_lo + idx;
            let mut val = *slot;
            if has_corr {
                if (idx & 63) == 0 {
                    corr = Complex64::cis(dcorr * (k as f64));
                }
                val *= corr;
                corr *= corr_step;
            }
            if net_tau_pow != 0 {
                val = mul_iw_pow(val, (k as f64) * dw, net_tau_pow);
            }
            *slot = val;
        }
    }
}

/// Removes an instrument response (`StationXML` `ChannelResponse` or `SACPZ`) from `data` sampled
/// at `sampling_rate_hz`, returning physical ground motion in `cfg.output` units (`m`, `m/s`, or `m/s^2`).
pub fn remove_response(
    data: &[f64],
    sampling_rate_hz: f64,
    response: &ChannelResponse,
    cfg: &ResponseRemovalConfig,
) -> Result<Vec<f64>> {
    let mut scratch = ScratchWorkspace::new();
    remove_response_with_scratch(data, sampling_rate_hz, response, cfg, &mut scratch)
}

/// Scratch-reusing spectral instrument deconvolution matching `ObsPy Trace.remove_response`.
pub fn remove_response_with_scratch(
    data: &[f64],
    sampling_rate_hz: f64,
    response: &ChannelResponse,
    cfg: &ResponseRemovalConfig,
    scratch: &mut ScratchWorkspace,
) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "remove_response input data is empty".into(),
        ));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "remove_response requires positive finite sampling_rate_hz".into(),
        ));
    }
    if let Some(flimit) = cfg.pre_filt_hz {
        validate_pre_filt(flimit)?;
    }

    let npts = data.len();
    let dt = 1.0 / sampling_rate_hz;

    // 1. Time-domain preprocessing (demean + SAC quarter-cosine taper):
    let mut work = data.to_vec();
    if cfg.zero_mean {
        demean_in_place(&mut work);
    }
    if cfg.taper && cfg.taper_fraction > 0.0 {
        let win = cosine_taper(npts, cfg.taper_fraction, false, true)?;
        for (x, &w) in work.iter_mut().zip(win.iter()) {
            *x *= w;
        }
    }

    // 2. Choose zero-padded FFT length >= 2 * npts to eliminate circular wrap-around:
    let nfft = if cfg.nfft_pow2 {
        next_pow2(2 * npts)
    } else {
        npts2nfft(npts)
    };
    let n_bins = nfft / 2 + 1;
    let df_hz = 1.0 / ((nfft as f64) * dt);

    let mut data_spec = Vec::with_capacity(n_bins);
    rfft_with_scratch(&work, nfft, &mut data_spec, scratch)?;
    drop(work);

    // 3. Evaluate instrument transfer function H(f) in target kinematic units:
    let gain_scale = if cfg.rescale_to_overall_sensitivity {
        if let Some(sens) = response.instrument_sensitivity {
            if sens.frequency_hz > 0.0 && sens.value > 0.0 {
                let h_ref = response.eval_freq(sens.frequency_hz).abs();
                if h_ref > 0.0 && h_ref.is_finite() {
                    sens.value / h_ref
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        }
    } else {
        1.0
    };

    // When `water_level_db` is `None` and `pre_filt_hz` is active, bins outside `[f1, f4]` are zeroed
    // unconditionally and do not contribute to a global water-level maximum, so skip evaluating them!
    let (k_start, k_end) = match (cfg.water_level_db, cfg.pre_filt_hz) {
        (None, Some([f1, _, _, f4])) if df_hz > 0.0 => {
            let ks = (math::floor(f1 / df_hz).max(1.0) as usize).min(n_bins.saturating_sub(1));
            let ke = (math::ceil(f4 / df_hz).max(0.0) as usize).min(n_bins.saturating_sub(1));
            (ks, ke)
        }
        _ => (1, n_bins.saturating_sub(1)),
    };

    let mut resp_spec = Vec::with_capacity(n_bins);
    eval_response_uniform_grid(
        response,
        cfg.output,
        gain_scale,
        n_bins,
        df_hz,
        k_start,
        k_end,
        &mut resp_spec,
    );

    // 4 & 5. Apply water-level inversion and optional 4-corner SAC cosine bandpass pre-filter in-place:
    invert_spectrum_waterlevel(&mut resp_spec, cfg.water_level_db)?;
    if let Some(flimit) = cfg.pre_filt_hz {
        for (k, (x_bin, &h_inv)) in data_spec.iter_mut().zip(resp_spec.iter()).enumerate() {
            let w = eval_sac_taper_point((k as f64) * df_hz, flimit);
            *x_bin = (*x_bin * w) * h_inv;
        }
    } else {
        for (x_bin, &h_inv) in data_spec.iter_mut().zip(resp_spec.iter()) {
            *x_bin *= h_inv;
        }
    }
    drop(resp_spec);

    // Force Nyquist bin pure real (`abs(data[-1]) + 0j`) matching ObsPy `remove_response` & `simulate_seismometer`:
    if let Some(last) = data_spec.last_mut() {
        *last = Complex64::new(last.abs(), 0.0);
    }

    // 6. Inverse RFFT and truncate to original `npts`:
    let mut time_out = Vec::with_capacity(nfft);
    irfft_with_scratch(&data_spec, nfft, &mut time_out, scratch)?;
    time_out.truncate(npts);
    Ok(time_out)
}

/// Removes instrument response from a `Trace`, updating optional `SacHeader` (`idep` & amplitude stats).
pub fn remove_response_trace(
    trace: &Trace,
    response: &ChannelResponse,
    cfg: &ResponseRemovalConfig,
) -> Result<Trace> {
    let deconv = remove_response(&trace.data, trace.stats.sampling_rate, response, cfg)?;
    let mut new_stats = trace.stats.clone();
    new_stats.calib = 1.0;
    if new_stats.sac.is_some() {
        new_stats.sync_to_sac_header(&deconv);
    }
    Ok(Trace::new(new_stats, deconv))
}

/// General SAC `transfer` operator deconvolving `from_response` (if `Some`) to physical ground motion
/// and convolving with `to_response` (if `Some`, e.g. Wood–Anderson torsion seismometer or target sensor).
pub fn sac_transfer(
    data: &[f64],
    sampling_rate_hz: f64,
    from_response: Option<&ChannelResponse>,
    to_response: Option<&ChannelResponse>,
    output_unit: KinematicUnit,
    pre_filt_hz: Option<[f64; 4]>,
    water_level_db: Option<f64>,
) -> Result<Vec<f64>> {
    if from_response.is_none() && to_response.is_none() {
        return Err(EarthSnailError::InvalidParameter(
            "sac_transfer requires at least one of from_response or to_response".into(),
        ));
    }
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "sac_transfer input data is empty".into(),
        ));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "sac_transfer requires positive finite sampling_rate_hz".into(),
        ));
    }
    if let Some(flimit) = pre_filt_hz {
        validate_pre_filt(flimit)?;
    }

    // Fast path when only deconvolving:
    if let (Some(from_r), None) = (from_response, to_response) {
        let cfg = ResponseRemovalConfig {
            output: output_unit,
            pre_filt_hz,
            water_level_db,
            ..ResponseRemovalConfig::default()
        };
        return remove_response(data, sampling_rate_hz, from_r, &cfg);
    }

    let npts = data.len();
    let dt = 1.0 / sampling_rate_hz;
    let mut work = data.to_vec();
    demean_in_place(&mut work);
    let win = cosine_taper(npts, 0.05, false, true)?;
    for (x, &w) in work.iter_mut().zip(win.iter()) {
        *x *= w;
    }

    let nfft = npts2nfft(npts);
    let n_bins = nfft / 2 + 1;
    let df_hz = 1.0 / ((nfft as f64) * dt);
    let mut scratch = ScratchWorkspace::new();
    let mut spec = Vec::with_capacity(n_bins);
    rfft_with_scratch(&work, nfft, &mut spec, &mut scratch)?;
    drop(work);

    if let Some(flimit) = pre_filt_hz {
        for (k, bin) in spec.iter_mut().enumerate() {
            *bin *= eval_sac_taper_point((k as f64) * df_hz, flimit);
        }
    }

    let mut h_buf = Vec::with_capacity(n_bins);
    if let Some(from_r) = from_response {
        eval_response_uniform_grid(
            from_r,
            output_unit,
            1.0,
            n_bins,
            df_hz,
            1,
            n_bins.saturating_sub(1),
            &mut h_buf,
        );
        invert_spectrum_waterlevel(&mut h_buf, water_level_db)?;
        for (bin, &h_inv) in spec.iter_mut().zip(h_buf.iter()) {
            *bin *= h_inv;
        }
    }

    if let Some(to_r) = to_response {
        eval_response_uniform_grid(
            to_r,
            output_unit,
            1.0,
            n_bins,
            df_hz,
            1,
            n_bins.saturating_sub(1),
            &mut h_buf,
        );
        spec[0] = Complex64::ZERO;
        for (bin, &h_to) in spec.iter_mut().zip(h_buf.iter()).skip(1) {
            *bin *= h_to;
        }
    }

    if let Some(last) = spec.last_mut() {
        *last = Complex64::new(last.abs(), 0.0);
    }

    let mut out = Vec::with_capacity(nfft);
    irfft_with_scratch(&spec, nfft, &mut out, &mut scratch)?;
    out.truncate(npts);
    Ok(out)
}

/// IASPEI standard Wood–Anderson horizontal torsion seismometer response (`T0 = 0.8 s`, `h = 0.8`,
/// magnification `2080` per IASPEI NMSOP; maps ground displacement in `m` to Wood–Anderson trace in `m`,
/// multiply by `1000.0` for `mm`).
#[allow(clippy::approx_constant)]
pub fn wood_anderson_response_iaspei() -> ChannelResponse {
    ChannelResponse::from_sacpz(
        vec![Complex64::ZERO, Complex64::ZERO],
        vec![
            Complex64::new(-6.283_185_3, 4.712_389_0),
            Complex64::new(-6.283_185_3, -4.712_389_0),
        ],
        2_080.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use earthsnail_core::inventory::StageGain;

    #[test]
    fn sac_transfer_forward_then_inverse_recovers_inband_wavepacket() {
        let fs = if cfg!(miri) { 10.0 } else { 40.0 };
        let npts = if cfg!(miri) { 256 } else { 1024 };
        // 2-pole 120s broadband velocity sensor in SACPZ (3 zeros at origin for DISP -> COUNTS):
        let p1 = Complex64::new(-0.037_004, 0.037_016);
        let p2 = p1.conj();
        let resp = ChannelResponse::from_sacpz(
            vec![Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            vec![p1, p2],
            1.5e9,
        );

        // Smooth 1.0 Hz Ricker-modulated wavepacket well inside [0.05, 4.0] Hz passband:
        let mid_t = 0.5 * (npts as f64) / fs;
        let ground_vel: Vec<f64> = (0..npts)
            .map(|i| {
                let t = (i as f64) / fs;
                let u = PI * 1.0 * (t - mid_t);
                (1.0 - 2.0 * u * u) * math::exp(-u * u) * 1e-4
            })
            .collect();

        let counts = sac_transfer(
            &ground_vel,
            fs,
            None,
            Some(&resp),
            KinematicUnit::Velocity,
            None,
            None,
        )
        .unwrap();

        let f_hi = if cfg!(miri) {
            [0.01, 0.02, 3.5, 4.8]
        } else {
            [0.01, 0.02, 12.0, 18.0]
        };
        let recovered = remove_response(
            &counts,
            fs,
            &resp,
            &ResponseRemovalConfig {
                output: KinematicUnit::Velocity,
                pre_filt_hz: Some(f_hi),
                water_level_db: Some(60.0),
                zero_mean: true,
                taper: true,
                taper_fraction: 0.05,
                ..ResponseRemovalConfig::default()
            },
        )
        .unwrap();

        // Compare central window [npts/4 .. 3*npts/4] away from 5% edge tapers:
        let mut err_sq = 0.0;
        let mut ref_sq = 0.0;
        for i in (npts / 4)..(3 * npts / 4) {
            let d = recovered[i] - ground_vel[i];
            err_sq += d * d;
            ref_sq += ground_vel[i] * ground_vel[i];
        }
        let rel_rmse = math::sqrt(err_sq / ref_sq);
        assert!(rel_rmse < 2e-3, "roundtrip rel_rmse={rel_rmse}");
    }

    #[test]
    fn fast_vector_stage_evaluator_matches_pointwise_across_all_stage_types() {
        let multi = ChannelResponse {
            input_units: KinematicUnit::Velocity,
            output_units: "COUNTS".into(),
            instrument_sensitivity: Some(StageGain {
                value: 2.0e9,
                frequency_hz: 0.05,
            }),
            stages: vec![
                ResponseStage::PolesZeros(PolesZerosStage {
                    pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
                    a0: 8.46e17,
                    normalization_freq: 0.02,
                    zeros: vec![
                        Complex64::ZERO,
                        Complex64::ZERO,
                        Complex64::new(-31.63, 0.0),
                        Complex64::new(-160.0, 0.0),
                    ],
                    poles: vec![
                        Complex64::new(-0.0368, -0.0362),
                        Complex64::new(-0.0368, 0.0362),
                        Complex64::new(-32.55, 0.0),
                        Complex64::new(-364.0, -404.0),
                        Complex64::new(-364.0, 404.0),
                    ],
                    stage_gain: 1177.0,
                    input_sample_rate_hz: Some(40.0),
                }),
                ResponseStage::Coefficients(CoefficientsStage {
                    numerator: vec![0.01, -0.05, 0.25, 0.60, 0.22, -0.04, 0.01],
                    denominator: vec![1.0, -0.15, 0.04],
                    input_sample_rate_hz: 80.0,
                    correction_s: 0.0375,
                    stage_gain: 1_677_720.0,
                    gain_frequency_hz: 0.02,
                }),
                ResponseStage::Fir(FirStage {
                    symmetry: FirSymmetry::Odd,
                    coefficients: vec![-0.02, 0.08, 0.24, 0.40],
                    input_sample_rate_hz: 80.0,
                    decimation_factor: 2,
                    delay_s: 0.0375,
                    correction_s: 0.0375,
                    stage_gain: 1.0,
                    gain_frequency_hz: 0.0,
                }),
                ResponseStage::Fir(FirStage {
                    symmetry: FirSymmetry::Even,
                    coefficients: vec![-0.03, 0.15, 0.38],
                    input_sample_rate_hz: 40.0,
                    decimation_factor: 1,
                    delay_s: 0.0625,
                    correction_s: 0.05,
                    stage_gain: 1.0,
                    gain_frequency_hz: 0.02,
                }),
            ],
        };

        let n_bins = if cfg!(miri) { 65 } else { 600 };
        let df = 20.0 / ((n_bins - 1) as f64);
        let mut fast = Vec::new();
        for unit in [
            KinematicUnit::Displacement,
            KinematicUnit::Velocity,
            KinematicUnit::Acceleration,
        ] {
            eval_response_uniform_grid(&multi, unit, 1.25, n_bins, df, 1, n_bins - 1, &mut fast);
            assert_eq!(fast[0], Complex64::ZERO);
            let exps: Vec<Complex64> = (0..n_bins)
                .map(|k| {
                    if k == 0 {
                        Complex64::ZERO
                    } else {
                        multi.eval_freq_kinematic((k as f64) * df, unit) * 1.25
                    }
                })
                .collect();
            let peak_mag = exps.iter().map(|c| c.abs()).fold(1.0_f64, f64::max);
            for (k, (&got, &exp)) in fast.iter().zip(exps.iter()).enumerate().skip(1) {
                let rel = (got - exp).abs() / exp.abs().max(peak_mag * 1e-3);
                let tol = if cfg!(miri) { 1e-10 } else { 1e-12 };
                assert!(
                    rel < tol,
                    "unit={unit:?} k={k}: got={got:?} exp={exp:?} rel={rel:.3e}"
                );
            }
        }
    }
}
