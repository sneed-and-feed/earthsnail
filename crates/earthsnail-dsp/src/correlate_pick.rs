//! Cross-correlation, STA/LTA detectors, Baer–Kradolfer, Maeda AIC, and
//! 3-Point Discrete Morse Causal-Deferral glitch-immune phase picking.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use earthsnail_core::math::{hypot, ln, powf, sqrt};
use earthsnail_core::{Complex64, EarthSnailError, Result, ScratchWorkspace};

use crate::fft::{fft_with_scratch, ifft_with_scratch, next_pow2};
use crate::spectral::analytic_signal_with_scratch;

#[inline]
fn abs(x: f64) -> f64 {
    x.abs()
}

/// Normalization strategy for cross-correlation routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcorrNormalize {
    /// Unnormalized raw inner products.
    None,
    /// Normalize by global `sqrt(sum(a^2) * sum(b^2))` (ObsPy `'naive'`).
    Naive,
    /// Sliding-window zero-normalized / local energy normalization (ObsPy `'full'`).
    Full,
}

/// Output length mode for [`correlate_template`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcorrMode {
    /// Only shifts where `template` fits completely inside `data` (`len(data) - len(template) + 1`).
    Valid,
    /// Centered output with the same length as `data`.
    Same,
    /// Full linear cross-correlation (`len(data) + len(template) - 1`).
    Full,
}

/// Compute linear cross-correlation `full[k] = sum_n a[n] * b[n + (len_b - 1) - k]`
/// matching `scipy.signal.correlate(a, b, mode='full')`.
fn xcorr_full_raw(a: &[f64], b: &[f64], scratch: &mut ScratchWorkspace) -> Result<Vec<f64>> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return Err(EarthSnailError::EmptyData(
            "cross-correlation input slice is empty".into(),
        ));
    }
    let out_len = na + nb - 1;
    if na.min(nb) <= 32 || out_len <= 128 {
        let mut out = vec![0.0_f64; out_len];
        for (ia, &va) in a.iter().enumerate() {
            for (ib, &vb) in b.iter().enumerate() {
                out[ia + nb - 1 - ib] += va * vb;
            }
        }
        return Ok(out);
    }

    let nfft = next_pow2(out_len);
    let mut fa = vec![Complex64::ZERO; nfft];
    let mut fb = vec![Complex64::ZERO; nfft];
    for (i, &v) in a.iter().enumerate() {
        fa[i] = Complex64::new(v, 0.0);
    }
    for (i, &v) in b.iter().enumerate() {
        fb[nb - 1 - i] = Complex64::new(v, 0.0);
    }
    fft_with_scratch(&mut fa, scratch)?;
    fft_with_scratch(&mut fb, scratch)?;
    for i in 0..nfft {
        fa[i] *= fb[i];
    }
    ifft_with_scratch(&mut fa, scratch)?;
    Ok(fa[..out_len].iter().map(|c| c.re).collect())
}

/// Cross-correlate `a` and `b` up to `max_shift` samples using caller-provided scratch space.
///
/// Matches `obspy.signal.cross_correlation.correlate(a, b, shift, demean, normalize)`:
/// if `b` is delayed by `+k` samples relative to `a` (`b[n] = a[n - k]`), the correlation peak
/// occurs at lag `-k` samples (`xcorr_max` returns `-k`).
pub fn correlate_with_scratch(
    a: &[f64],
    b: &[f64],
    max_shift: usize,
    demean: bool,
    normalize: XcorrNormalize,
    scratch: &mut ScratchWorkspace,
) -> Result<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "correlate requires non-empty input slices".into(),
        ));
    }
    if normalize == XcorrNormalize::Full {
        return Err(EarthSnailError::InvalidParameter(
            "correlate supports None or Naive normalization; use correlate_template for Full"
                .into(),
        ));
    }

    let a_work: Vec<f64> = if demean {
        let mean_a = a.iter().copied().sum::<f64>() / (a.len() as f64);
        a.iter().map(|&v| v - mean_a).collect()
    } else {
        a.to_vec()
    };
    let b_work: Vec<f64> = if demean {
        let mean_b = b.iter().copied().sum::<f64>() / (b.len() as f64);
        b.iter().map(|&v| v - mean_b).collect()
    } else {
        b.to_vec()
    };

    let full = xcorr_full_raw(&a_work, &b_work, scratch)?;
    let full_len = full.len();
    let mid = full_len / 2;
    let extra = full_len % 2;
    let out_len = 2 * max_shift + extra;
    let mut cc = vec![0.0_f64; out_len];

    for (idx, slot) in cc.iter_mut().enumerate() {
        let full_idx = (mid as isize) - (max_shift as isize) + (idx as isize);
        if full_idx >= 0 && (full_idx as usize) < full_len {
            *slot = full[full_idx as usize];
        }
    }

    if normalize == XcorrNormalize::Naive {
        let ea: f64 = a_work.iter().map(|&v| v * v).sum();
        let eb: f64 = b_work.iter().map(|&v| v * v).sum();
        let norm = sqrt(ea * eb);
        if norm <= f64::EPSILON {
            cc.fill(0.0);
        } else {
            for v in &mut cc {
                *v /= norm;
            }
        }
    }

    Ok(cc)
}

/// Cross-correlate `a` and `b` up to `max_shift` samples (ObsPy sign convention).
pub fn correlate(
    a: &[f64],
    b: &[f64],
    max_shift: usize,
    demean: bool,
    normalize: XcorrNormalize,
) -> Result<Vec<f64>> {
    let mut scratch = ScratchWorkspace::new();
    correlate_with_scratch(a, b, max_shift, demean, normalize, &mut scratch)
}

/// Return `(shift_samples, value)` of the peak of a cross-correlation curve.
///
/// Matches `obspy.signal.cross_correlation.xcorr_max(cc, abs_max)`.
pub fn xcorr_max(cc: &[f64], abs_max: bool) -> Result<(f64, f64)> {
    if cc.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "xcorr_max requires non-empty cross-correlation slice".into(),
        ));
    }
    let mut best_idx = 0usize;
    let mut best_metric = if abs_max { abs(cc[0]) } else { cc[0] };
    for (i, &v) in cc.iter().enumerate().skip(1) {
        let m = if abs_max { abs(v) } else { v };
        if m > best_metric {
            best_metric = m;
            best_idx = i;
        }
    }
    let mid = ((cc.len() - 1) as f64) * 0.5;
    Ok(((best_idx as f64) - mid, cc[best_idx]))
}

/// Sub-sample 3-point parabolic peak interpolation on a cross-correlation curve.
pub fn xcorr_max_subsample(cc: &[f64], abs_max: bool) -> Result<(f64, f64)> {
    if cc.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "xcorr_max_subsample requires non-empty cross-correlation slice".into(),
        ));
    }
    let mut best_idx = 0usize;
    let mut best_metric = if abs_max { abs(cc[0]) } else { cc[0] };
    for (i, &v) in cc.iter().enumerate().skip(1) {
        let m = if abs_max { abs(v) } else { v };
        if m > best_metric {
            best_metric = m;
            best_idx = i;
        }
    }
    let mid = ((cc.len() - 1) as f64) * 0.5;
    if best_idx == 0 || best_idx + 1 >= cc.len() {
        return Ok(((best_idx as f64) - mid, cc[best_idx]));
    }

    let sgn = if abs_max && cc[best_idx] < 0.0 {
        -1.0
    } else {
        1.0
    };
    let ym = sgn * cc[best_idx - 1];
    let y0 = sgn * cc[best_idx];
    let yp = sgn * cc[best_idx + 1];
    let denom = ym - 2.0 * y0 + yp;
    if denom >= -1e-15 {
        return Ok(((best_idx as f64) - mid, cc[best_idx]));
    }
    let delta = (0.5 * (ym - yp) / denom).clamp(-0.5, 0.5);
    let peak = sgn * (y0 - 0.25 * (ym - yp) * delta);
    Ok(((best_idx as f64) + delta - mid, peak))
}

fn rolling_window_sum(padded: &[f64], win_len: usize) -> Vec<f64> {
    if padded.len() <= win_len {
        return Vec::new();
    }
    let mut csum = Vec::with_capacity(padded.len());
    let mut acc = 0.0_f64;
    for &v in padded {
        acc += v;
        csum.push(acc);
    }
    let out_len = padded.len() - win_len;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        out.push(csum[i + win_len] - csum[i]);
    }
    out
}

/// Sliding template cross-correlation (including zero-normalized cross-correlation `'full'`).
///
/// Matches `obspy.signal.cross_correlation.correlate_template`.
pub fn correlate_template(
    data: &[f64],
    template: &[f64],
    mode: XcorrMode,
    normalize: XcorrNormalize,
    demean: bool,
) -> Result<Vec<f64>> {
    let mut scratch = ScratchWorkspace::new();
    correlate_template_with_scratch(data, template, mode, normalize, demean, &mut scratch)
}

/// Scratch-backed [`correlate_template`].
pub fn correlate_template_with_scratch(
    data: &[f64],
    template: &[f64],
    mode: XcorrMode,
    normalize: XcorrNormalize,
    demean: bool,
    scratch: &mut ScratchWorkspace,
) -> Result<Vec<f64>> {
    let nd = data.len();
    let nt = template.len();
    if nd == 0 || nt == 0 {
        return Err(EarthSnailError::EmptyData(
            "correlate_template requires non-empty data and template".into(),
        ));
    }
    if nd < nt {
        return Err(EarthSnailError::InvalidParameter(
            "data must not be shorter than template".into(),
        ));
    }

    let tpl: Vec<f64> = if demean {
        let m = template.iter().copied().sum::<f64>() / (nt as f64);
        template.iter().map(|&v| v - m).collect()
    } else {
        template.to_vec()
    };

    let dat: Vec<f64> = if demean && normalize != XcorrNormalize::Full {
        let m = data.iter().copied().sum::<f64>() / (nd as f64);
        data.iter().map(|&v| v - m).collect()
    } else {
        data.to_vec()
    };

    let full = xcorr_full_raw(&dat, &tpl, scratch)?;
    let mut cc: Vec<f64> = match mode {
        XcorrMode::Full => full,
        XcorrMode::Valid => full[(nt - 1)..nd].to_vec(),
        XcorrMode::Same => {
            let start = (nt - 1) / 2;
            full[start..(start + nd)].to_vec()
        }
    };

    match normalize {
        XcorrNormalize::None => {}
        XcorrNormalize::Naive => {
            let tnorm: f64 = tpl.iter().map(|&v| v * v).sum();
            let dnorm: f64 = dat.iter().map(|&v| v * v).sum();
            let norm = sqrt(tnorm * dnorm);
            if norm <= f64::EPSILON {
                cc.fill(0.0);
            } else {
                for v in &mut cc {
                    *v /= norm;
                }
            }
        }
        XcorrNormalize::Full => {
            let tnorm: f64 = tpl.iter().map(|&v| v * v).sum();
            let pad = cc.len() + nt - nd;
            let (pad1, pad2) = if mode == XcorrMode::Same {
                ((pad + 2) / 2, pad.saturating_sub(1) / 2)
            } else {
                (pad.div_ceil(2), pad / 2)
            };
            let mut padded = Vec::with_capacity(pad1 + nd + pad2);
            padded.resize(pad1, 0.0);
            padded.extend_from_slice(&dat);
            padded.resize(pad1 + nd + pad2, 0.0);
            let padded_sq: Vec<f64> = padded.iter().map(|&v| v * v).collect();
            let mut var_win = rolling_window_sum(&padded_sq, nt);
            if demean {
                let win_s = rolling_window_sum(&padded, nt);
                for (vw, ws) in var_win.iter_mut().zip(win_s.iter()) {
                    *vw = (*vw - (ws * ws) / (nt as f64)).max(0.0);
                }
            }
            for (c, &vw) in cc.iter_mut().zip(var_win.iter()) {
                let norm = sqrt((vw * tnorm).max(0.0));
                if norm <= f64::EPSILON {
                    *c = 0.0;
                } else {
                    *c /= norm;
                }
            }
        }
    }

    Ok(cc)
}

/// Schimmel (1999) amplitude-unbiased Phase Cross-Correlation (`PCC`).
pub fn pcc(a: &[f64], b: &[f64], max_shift: usize, power_nu: f64) -> Result<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "pcc requires non-empty input slices".into(),
        ));
    }
    if !power_nu.is_finite() || power_nu <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "pcc power exponent nu must be finite and positive".into(),
        ));
    }

    let mut scratch = ScratchWorkspace::new();
    let za = analytic_signal_with_scratch(a, &mut scratch)?;
    let zb = analytic_signal_with_scratch(b, &mut scratch)?;

    let to_unit = |z: Complex64| -> Complex64 {
        let mag = hypot(z.re, z.im);
        if mag > 1e-30 {
            Complex64::new(z.re / mag, z.im / mag)
        } else {
            Complex64::ZERO
        }
    };
    let ua: Vec<Complex64> = za.into_iter().map(to_unit).collect();
    let ub: Vec<Complex64> = zb.into_iter().map(to_unit).collect();

    let na = ua.len();
    let nb = ub.len();
    let full_len = na + nb - 1;
    let mid = full_len / 2;
    let extra = full_len % 2;
    let out_len = 2 * max_shift + extra;
    let scale = 1.0 / powf(2.0, power_nu);

    let mut out = vec![0.0_f64; out_len];
    for (idx, slot) in out.iter_mut().enumerate() {
        let full_k = (mid as isize) - (max_shift as isize) + (idx as isize);
        let delta = (nb as isize) - 1 - full_k;
        let ia_start = (-delta).max(0) as usize;
        let ia_end_excl = ((nb as isize) - delta).min(na as isize).max(0) as usize;
        if ia_end_excl <= ia_start {
            continue;
        }
        let mut acc = 0.0_f64;
        let mut count = 0usize;
        for (ia, &va) in ua.iter().enumerate().take(ia_end_excl).skip(ia_start) {
            let ib = ((ia as isize) + delta) as usize;
            let vb = ub[ib];
            let sum_mag = hypot(va.re + vb.re, va.im + vb.im);
            let diff_mag = hypot(va.re - vb.re, va.im - vb.im);
            acc += powf(sum_mag, power_nu) - powf(diff_mag, power_nu);
            count += 1;
        }
        if count > 0 {
            *slot = (acc * scale) / (count as f64);
        }
    }

    Ok(out)
}

/// Classic sliding-window energy ratio STA/LTA characteristic function (`obspy.signal.trigger.classic_sta_lta`).
pub fn classic_sta_lta(data: &[f64], nsta: usize, nlta: usize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "classic_sta_lta input is empty".into(),
        ));
    }
    if nsta == 0 || nlta == 0 || nsta > nlta || data.len() < nlta {
        return Err(EarthSnailError::InvalidParameter(
            "require 0 < nsta <= nlta <= data.len()".into(),
        ));
    }

    let n = data.len();
    let mut csum = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for (i, &x) in data.iter().enumerate() {
        acc += x * x;
        csum[i] = acc;
    }

    let inv_sta = 1.0 / (nsta as f64);
    let inv_lta = 1.0 / (nlta as f64);
    let dtiny = f64::MIN_POSITIVE;
    let mut cft = vec![0.0_f64; n];

    for i in (nlta - 1)..n {
        let sta_sum = if i >= nsta {
            csum[i] - csum[i - nsta]
        } else {
            csum[i]
        };
        let lta_sum = if i >= nlta {
            csum[i] - csum[i - nlta]
        } else {
            csum[i]
        };
        let sta = sta_sum * inv_sta;
        let lta = (lta_sum * inv_lta).max(dtiny);
        cft[i] = sta / lta;
    }

    Ok(cft)
}

/// Recursive exponential-moving-average STA/LTA characteristic function (`obspy.signal.trigger.recursive_sta_lta`).
pub fn recursive_sta_lta(data: &[f64], nsta: usize, nlta: usize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "recursive_sta_lta input is empty".into(),
        ));
    }
    if nsta == 0 || nlta == 0 || nsta > nlta || data.len() < nlta {
        return Err(EarthSnailError::InvalidParameter(
            "require 0 < nsta <= nlta <= data.len()".into(),
        ));
    }

    let n = data.len();
    let csta = 1.0 / (nsta as f64);
    let clta = 1.0 / (nlta as f64);
    let icsta = 1.0 - csta;
    let iclta = 1.0 - clta;

    let mut sta = 0.0_f64;
    let mut lta = f64::MIN_POSITIVE;
    let mut cft = vec![0.0_f64; n];

    for i in 1..n {
        let sq = data[i] * data[i];
        sta = csta * sq + icsta * sta;
        lta = clta * sq + iclta * lta;
        if i >= nlta {
            cft[i] = sta / lta;
        }
    }

    Ok(cft)
}

/// Earthworm `carl_sta_trig` characteristic function (`obspy.signal.trigger.carl_sta_trig`).
pub fn carl_sta_trig(
    data: &[f64],
    nsta: usize,
    nlta: usize,
    ratio: f64,
    quiet: f64,
) -> Result<Vec<f64>> {
    let m = data.len();
    if m == 0 {
        return Err(EarthSnailError::EmptyData(
            "carl_sta_trig input is empty".into(),
        ));
    }
    if nsta == 0 || nlta == 0 || m <= nlta {
        return Err(EarthSnailError::InvalidParameter(
            "require 0 < nsta and 0 < nlta < data.len()".into(),
        ));
    }

    let lagged_boxcar = |src: &[f64], w: usize| -> Vec<f64> {
        let mut out = vec![0.0_f64; src.len()];
        if src.len() <= w {
            return out;
        }
        let mut csum = Vec::with_capacity(src.len() + 1);
        csum.push(0.0);
        let mut acc = 0.0_f64;
        for &v in src {
            acc += v;
            csum.push(acc);
        }
        let inv_w = 1.0 / (w as f64);
        for (k, slot) in out.iter_mut().enumerate().skip(w) {
            *slot = (csum[k] - csum[k - w]) * inv_w;
        }
        out
    };

    let sta = lagged_boxcar(data, nsta);
    let lta_raw = lagged_boxcar(&sta, nlta);
    let mut lta = vec![0.0_f64; m];
    lta[1..].copy_from_slice(&lta_raw[..(m - 1)]);

    let abs_diff: Vec<f64> = data
        .iter()
        .zip(lta.iter())
        .map(|(&a, &l)| abs(a - l))
        .collect();
    let star = lagged_boxcar(&abs_diff, nsta);
    let ltar = lagged_boxcar(&star, nlta);

    let mut eta = vec![-1.0_f64; m];
    for i in nlta..m {
        eta[i] = star[i] - ratio * ltar[i] - abs(sta[i] - lta[i]) - quiet;
    }
    Ok(eta)
}

/// Swindell & Snell (1977) Z-detector standardized energy characteristic function (`obspy.signal.trigger.z_detect`).
pub fn z_detect(data: &[f64], nsta: usize) -> Result<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return Err(EarthSnailError::EmptyData("z_detect input is empty".into()));
    }
    if nsta == 0 || n <= nsta {
        return Err(EarthSnailError::InvalidParameter(
            "require 0 < nsta < data.len()".into(),
        ));
    }

    let mut csum = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for (i, &v) in data.iter().enumerate() {
        acc += v * v;
        csum[i] = acc;
    }

    let mut sta = vec![0.0_f64; n];
    sta[nsta] = csum[nsta - 1];
    for i in (nsta + 1)..n {
        sta[i] = csum[i - 1] - csum[i - 1 - nsta];
    }

    let mean = sta.iter().copied().sum::<f64>() / (n as f64);
    let var = sta
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64);
    let std = sqrt(var);
    if std <= 1e-30 {
        return Ok(vec![0.0; n]);
    }
    Ok(sta.into_iter().map(|v| (v - mean) / std).collect())
}

/// Hysteresis trigger `[on_sample, off_sample]` detector (`obspy.signal.trigger.trigger_onset`).
pub fn trigger_onset(charfct: &[f64], thr_on: f64, thr_off: f64) -> Vec<[usize; 2]> {
    trigger_onset_with_options(charfct, thr_on, thr_off, usize::MAX / 2, false)
}

/// Hysteresis trigger detector with `max_len` and `max_len_delete` options matching ObsPy.
pub fn trigger_onset_with_options(
    charfct: &[f64],
    thr_on: f64,
    thr_off: f64,
    max_len: usize,
    max_len_delete: bool,
) -> Vec<[usize; 2]> {
    let mut ind1 = Vec::new();
    let mut ind2 = Vec::new();
    for (i, &v) in charfct.iter().enumerate() {
        if v >= thr_on {
            ind1.push(i);
        }
        if v >= thr_off {
            ind2.push(i);
        }
    }
    if ind1.is_empty() || ind2.is_empty() {
        return Vec::new();
    }

    let mut on: Vec<isize> = Vec::new();
    on.push(ind1[0] as isize);
    for w in ind1.windows(2) {
        if w[1] - w[0] > 1 {
            on.push(w[1] as isize);
        }
    }

    let mut of: Vec<isize> = Vec::new();
    of.push(-1);
    for w in ind2.windows(2) {
        if w[1] - w[0] > 1 {
            of.push(w[0] as isize);
        }
    }
    of.push(*ind2.last().unwrap_or(&0) as isize);

    if max_len_delete {
        of.push(isize::MAX / 2);
        let last_on = *on.last().unwrap_or(&0);
        on.push(last_on);
    } else {
        of.push(*ind2.last().unwrap_or(&0) as isize);
    }
    of.push(charfct.len() as isize);

    let mut on_head = 0usize;
    let mut of_head = 0usize;
    let mut picks = Vec::new();
    let max_len_i = (max_len.min((isize::MAX / 4) as usize)) as isize;

    while on.last().copied().unwrap_or(-1) > of[of_head] {
        while on_head < on.len() && on[on_head] <= of[of_head] {
            on_head += 1;
        }
        if on_head >= on.len() {
            break;
        }
        while of_head < of.len() && of[of_head] < on[on_head] {
            of_head += 1;
        }
        if of_head >= of.len() {
            break;
        }
        if of[of_head] - on[on_head] > max_len_i {
            if max_len_delete {
                on_head += 1;
                if on_head >= on.len() {
                    break;
                }
                continue;
            }
            let split_off = on[on_head] + max_len_i;
            picks.push([on[on_head] as usize, split_off as usize]);
            of_head = of_head.saturating_sub(1);
            of[of_head] = split_off;
            continue;
        }
        picks.push([on[on_head] as usize, of[of_head] as usize]);
    }

    picks
}

/// Configuration parameters for [`pk_baer`] (Baer & Kradolfer 1987 automatic P-picker).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaerPickParams {
    /// Sampling rate in samples per second (`samp_int`).
    pub samp_int: usize,
    /// Maximum allowed consecutive samples below `thr1` before evaluating trigger validity.
    pub tdownmax: usize,
    /// Minimum total active samples required to accept a candidate onset.
    pub tupevent: usize,
    /// Characteristic function onset trigger threshold (`thr1`, typically `7.0`–`10.0`).
    pub thr1: f64,
    /// Characteristic function threshold (`thr2`, typically `12.0`–`20.0`) above which variance updates pause.
    pub thr2: f64,
    /// Initial preset sample count for background `sigma` initialization (`preset_len`).
    pub preset_len: usize,
    /// Post-pick sample window for amplitude / HYPO71 quality grading (`p_dur`).
    pub p_dur: usize,
}

impl Default for BaerPickParams {
    fn default() -> Self {
        Self {
            samp_int: 100,
            tdownmax: 20,
            tupevent: 60,
            thr1: 7.0,
            thr2: 12.0,
            preset_len: 100,
            p_dur: 100,
        }
    }
}

/// Result of Baer & Kradolfer (1987) automatic P-phase picking.
#[derive(Debug, Clone, PartialEq)]
pub struct BaerPickResult {
    /// Accepted 0-based pick index matching `obspy.signal.trigger.pk_baer` (`None` if no valid pick).
    pub pick_sample: Option<usize>,
    /// Raw 0-based integer returned by `obspy.signal.trigger.pk_baer` (`1` when unpicked, else `pick + 1`).
    pub obspy_pick_sample: usize,
    /// HYPO71/SED 4-character onset descriptor (e.g. `"EPU3"`, `"IPU0"`, `"EPD2"`, or `""` if unpicked).
    pub first_motion: String,
    /// Characteristic function array of length `data.len() - 1` matching ObsPy `return_cf=True`.
    pub characteristic_function: Vec<f64>,
}

/// Baer & Kradolfer (1987) envelope-squared / fourth-power CF phase picker (`pk_baer`).
pub fn pk_baer(data: &[f64], params: &BaerPickParams) -> Result<BaerPickResult> {
    if data.len() < 4 {
        return Err(EarthSnailError::InvalidParameter(
            "pk_baer requires at least 4 samples".into(),
        ));
    }
    let n = data.len() - 1;
    if params.preset_len == 0 || params.preset_len >= n {
        return Err(EarthSnailError::InvalidParameter(
            "preset_len must satisfy 1 <= preset_len < data.len() - 1".into(),
        ));
    }

    let y: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let m = params.preset_len;

    let mut s_y = 0.0_f64;
    let mut s_y2 = 0.0_f64;
    let mut sum_dy2 = 0.0_f64;
    for i in 1..=m {
        let yi = y[i] as f64;
        s_y += yi;
        if i >= 2 {
            s_y2 += yi * yi;
            let dy = (y[i] - y[i - 1]) as f64;
            sum_dy2 += dy * dy;
        }
    }
    let mut sum_y2 = s_y2;
    if sum_dy2 <= 0.0 {
        sum_dy2 = 1e-30;
    }
    let arg = (m as f64) * s_y2 - s_y * s_y;
    let mut sigma = if arg <= 0.0 {
        1.0
    } else {
        sqrt(arg) / ((m * m) as f64)
    };

    let mut mean_sf = 0.0_f64;
    let mut sum_sf = 0.0_f64;
    let mut sum_sf2 = 0.0_f64;
    let mut cnt = 0usize;

    let mut cf = vec![0.0_f64; n];
    let mut itrm = 0usize;
    let mut uptime = 0usize;
    let mut dtime = 0usize;
    let mut accepted_itrm = 0usize;

    for i in 1..=n {
        let yi = y[i] as f64;
        let dyi = if i == 1 {
            0.0
        } else {
            (y[i] - y[i - 1]) as f64
        };
        let env2 = yi * yi + (sum_y2 / sum_dy2) * dyi * dyi;
        let sf = env2 * env2;
        sum_y2 += yi * yi;
        sum_dy2 = (sum_dy2 + dyi * dyi).max(1e-30);

        let cf_val = ((sf - mean_sf) / sigma) as f32 as f64;
        if i == 1 {
            cf[0] = cf_val;
        } else if i >= 3 {
            cf[i - 2] = cf_val;
        }

        if i > 2 * m {
            if itrm == 0 {
                if cf_val >= params.thr1 {
                    itrm = i;
                    uptime = 1;
                    dtime = 0;
                }
            } else if cf_val >= params.thr1 {
                uptime += 1;
                dtime = 0;
            } else {
                dtime += 1;
                uptime += 1;
                if dtime > params.tdownmax {
                    if uptime.saturating_sub(dtime) >= params.tupevent {
                        accepted_itrm = itrm;
                        break;
                    }
                    itrm = 0;
                    uptime = 0;
                    dtime = 0;
                }
            }
        }

        if cf_val <= params.thr2 || i <= 2 * m {
            sum_sf += sf;
            sum_sf2 += sf * sf;
            cnt += 1;
            if cnt >= 2 {
                mean_sf = sum_sf / (cnt as f64);
                let var_sf = (sum_sf2 / (cnt as f64) - mean_sf * mean_sf).max(1e-30);
                sigma = sqrt(var_sf);
            }
        }
    }

    if accepted_itrm == 0 && itrm != 0 && uptime >= params.tupevent {
        accepted_itrm = itrm;
    }

    if accepted_itrm == 0 {
        return Ok(BaerPickResult {
            pick_sample: None,
            obspy_pick_sample: 1,
            first_motion: String::new(),
            characteristic_function: cf,
        });
    }

    let base_val = y[accepted_itrm - 1] as f64;
    let mut pol_char = 'U';
    for &yk in y.iter().take(n + 1).skip(accepted_itrm) {
        let diff = (yk as f64) - base_val;
        if abs(diff) > 0.0 {
            pol_char = if diff >= 0.0 { 'U' } else { 'D' };
            break;
        }
    }

    let cf_start = accepted_itrm.saturating_sub(2);
    let cf_end = (cf_start + params.p_dur).min(cf.len());
    let mut cf_peak = 0.0_f64;
    for &c in &cf[cf_start..cf_end] {
        if c > cf_peak {
            cf_peak = c;
        }
    }

    let weight: u8 = if cf_peak >= 1.0e5 {
        0
    } else if cf_peak >= 6.5e4 {
        1
    } else if cf_peak >= 3.0e4 {
        2
    } else if cf_peak >= 2.0 * params.thr1 {
        3
    } else {
        4
    };
    let onset_char = if weight <= 1 { 'I' } else { 'E' };
    let mut pfm = String::with_capacity(4);
    pfm.push(onset_char);
    pfm.push('P');
    pfm.push(pol_char);
    pfm.push((b'0' + weight) as char);

    let obspy_idx = accepted_itrm + 1;
    Ok(BaerPickResult {
        pick_sample: Some(obspy_idx),
        obspy_pick_sample: obspy_idx,
        first_motion: pfm,
        characteristic_function: cf,
    })
}

/// Maeda (1985) Akaike Information Criterion (`AIC`) curve (`O(N)` single-pass prefix sums).
///
/// Matches `obspy.signal.trigger.aic_simple` (`aic[n - 1] == aic[n - 2]`).
pub fn aic_simple(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n <= 2 {
        return vec![0.0; n];
    }

    let mut pref_s = Vec::with_capacity(n + 1);
    let mut pref_s2 = Vec::with_capacity(n + 1);
    pref_s.push(0.0_f64);
    pref_s2.push(0.0_f64);
    let mut s = 0.0_f64;
    let mut s2 = 0.0_f64;
    for &x in data {
        s += x;
        s2 += x * x;
        pref_s.push(s);
        pref_s2.push(s2);
    }

    let slice_var = |lo: usize, hi: usize| -> f64 {
        let len = (hi - lo) as f64;
        let sum = pref_s[hi] - pref_s[lo];
        let sum2 = pref_s2[hi] - pref_s2[lo];
        let mean = sum / len;
        (sum2 / len - mean * mean).max(0.0)
    };

    let mut aic = vec![0.0_f64; n];
    for (idx, slot) in aic.iter_mut().enumerate().take(n - 1) {
        let m = idx + 1;
        let t1 = if m > 1 {
            let v1 = slice_var(0, m);
            if v1 > 0.0 {
                (m as f64) * ln(v1)
            } else {
                0.0
            }
        } else {
            0.0
        };
        let rem = n - m;
        let t2 = if rem > 1 {
            let v2 = slice_var(m, n);
            if v2 > 0.0 {
                ((rem - 1) as f64) * ln(v2)
            } else {
                0.0
            }
        } else {
            0.0
        };
        *slot = t1 + t2;
    }
    aic[n - 1] = aic[n - 2];
    aic
}

/// Pick the P-wave onset index via the global interior minimum of Maeda's [`aic_simple`].
pub fn aic_pick(data: &[f64]) -> Option<usize> {
    if data.len() < 4 {
        return None;
    }
    let aic = aic_simple(data);
    let mut best_idx = 1usize;
    let mut best_val = aic[1];
    for (i, &v) in aic.iter().enumerate().take(data.len() - 2).skip(2) {
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    if best_val.is_finite() {
        Some(best_idx)
    } else {
        None
    }
}

/// Configuration for [`morse_pick`] (3-Point Discrete Morse Causal-Deferral Picker).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorsePickConfig {
    /// Short-term persistence energy window in samples.
    pub nsta: usize,
    /// Long-term background persistence window in samples.
    pub nlta: usize,
    /// Minimum STA/LTA persistence ratio required to trigger.
    pub thr_on: f64,
    /// Minimum normalized Morse prominence `(e_tilde - med_bg) / sigma_bg` required at onset.
    pub prominence_sigma: f64,
    /// Number of consecutive post-onset samples whose 3-point Morse persistence must remain elevated.
    pub confirm_samples: usize,
}

impl Default for MorsePickConfig {
    fn default() -> Self {
        Self {
            nsta: 20,
            nlta: 120,
            thr_on: 4.5,
            prominence_sigma: 5.5,
            confirm_samples: 6,
        }
    }
}

/// Result of [`morse_pick`] including the picked onset index and spike-excised Morse persistence curve.
#[derive(Debug, Clone, PartialEq)]
pub struct MorsePickResult {
    /// Picked 0-based onset sample (`None` if no sustained seismic arrival is present).
    pub pick_sample: Option<usize>,
    /// Peak STA/LTA ratio of the 3-point Morse persistence envelope at the detected arrival.
    pub peak_ratio: f64,
    /// 3-point causal-deferred Discrete Morse persistence envelope `\tilde{e}_k`.
    pub persistence_envelope: Vec<f64>,
}

#[inline]
fn median3(a: f64, b: f64, c: f64) -> f64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    if c <= lo {
        lo
    } else if c >= hi {
        hi
    } else {
        c
    }
}

fn exact_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut buf = values.to_vec();
    buf.sort_by(|x, y| x.total_cmp(y));
    let mid = buf.len() / 2;
    if buf.len() % 2 == 1 {
        buf[mid]
    } else {
        0.5 * (buf[mid - 1] + buf[mid])
    }
}

/// 3-Point Discrete Morse Causal-Deferral Glitch-Immune Phase Picker (`morse_pick`).
///
/// Rejects arbitrary-amplitude 1-sample telemetry spikes (`0`-persistence critical pairs under
/// 1-sample causal deferral) with zero false triggers while accurately picking sustained seismic arrivals.
pub fn morse_pick(data: &[f64], config: &MorsePickConfig) -> Result<MorsePickResult> {
    let n = data.len();
    if n == 0 {
        return Err(EarthSnailError::EmptyData(
            "morse_pick input is empty".into(),
        ));
    }
    if config.nsta == 0 || config.nlta == 0 || config.nsta > config.nlta || n <= config.nlta + 4 {
        return Err(EarthSnailError::InvalidParameter(
            "require 0 < nsta <= nlta and data.len() > nlta + 4".into(),
        ));
    }

    let preset = config.nlta.min(n);
    let mu = exact_median(&data[..preset]);

    let dev: Vec<f64> = data.iter().map(|&v| abs(v - mu)).collect();
    let mut e_tilde = vec![0.0_f64; n];
    e_tilde[0] = dev[0].min(dev[1]);
    for k in 1..(n - 1) {
        e_tilde[k] = median3(dev[k - 1], dev[k], dev[k + 1]);
    }
    e_tilde[n - 1] = dev[n - 2].min(dev[n - 1]);

    let med_e = exact_median(&e_tilde[..preset]);
    let abs_e_dev: Vec<f64> = e_tilde[..preset].iter().map(|&v| abs(v - med_e)).collect();
    let mad_scale = 1.4826 * exact_median(&abs_e_dev);
    let rms_e = sqrt(e_tilde[..preset].iter().map(|&v| v * v).sum::<f64>() / (preset as f64));
    let sigma_bg = mad_scale.max(0.5 * rms_e).max(1e-12);
    let noise_floor_sq = (sigma_bg * sigma_bg).max(1e-24);

    let mut pref_e2 = vec![0.0_f64; n + 1];
    for k in 0..n {
        pref_e2[k + 1] = pref_e2[k] + e_tilde[k] * e_tilde[k];
    }
    let inv_sta = 1.0 / (config.nsta as f64);
    let inv_lta = 1.0 / (config.nlta as f64);

    let confirm = config.confirm_samples.max(2);
    let prom_thresh = med_e + config.prominence_sigma * sigma_bg;

    let mut pick_idx: Option<usize> = None;
    let mut peak_ratio = 0.0_f64;

    for k in config.nlta..(n.saturating_sub(confirm)) {
        let sta = (pref_e2[k + 1] - pref_e2[k + 1 - config.nsta]) * inv_sta;
        let lta = ((pref_e2[k + 1 - config.nsta]
            - pref_e2[(k + 1 - config.nsta).saturating_sub(config.nlta)])
            * inv_lta)
            .max(noise_floor_sq);
        let ratio = sta / lta;
        if ratio > peak_ratio {
            peak_ratio = ratio;
        }

        if pick_idx.is_none() && ratio >= config.thr_on && e_tilde[k] >= prom_thresh {
            let sustained = e_tilde[k..(k + confirm)]
                .iter()
                .filter(|&&v| v >= med_e + 0.65 * config.prominence_sigma * sigma_bg)
                .count();
            if sustained * 3 >= confirm * 2 {
                let onset_floor = med_e + 2.5 * sigma_bg;
                let min_back = k.saturating_sub(config.nsta);
                let mut onset = k;
                while onset > min_back && e_tilde[onset - 1] > onset_floor {
                    onset -= 1;
                }
                pick_idx = Some(onset);
            }
        }
    }

    Ok(MorsePickResult {
        pick_sample: pick_idx,
        peak_ratio,
        persistence_envelope: e_tilde,
    })
}

/// Convenience wrapper around [`morse_pick`] with [`MorsePickConfig::default`].
pub fn morse_pick_default(data: &[f64]) -> Result<Option<usize>> {
    Ok(morse_pick(data, &MorsePickConfig::default())?.pick_sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use earthsnail_core::math::{cos, sin};

    #[test]
    fn test_xcorr_shift_sign_and_subsample() {
        let n = if cfg!(miri) { 64 } else { 256 };
        let shift = 5usize;
        let mut a = vec![0.0_f64; n];
        let mut b = vec![0.0_f64; n];
        for (i, slot) in a.iter_mut().enumerate() {
            let t = (i as f64) * 0.05;
            let u = (t - 1.5) * 4.0;
            *slot = (1.0 - 2.0 * u * u) * earthsnail_core::math::exp(-u * u);
        }
        for i in shift..n {
            b[i] = 1.8 * a[i - shift];
        }
        let cc = correlate(&a, &b, 20, true, XcorrNormalize::Naive).unwrap();
        let (lag, val) = xcorr_max(&cc, false).unwrap();
        assert_eq!(lag, -(shift as f64));
        assert!(val > 0.99);
        let (sub_lag, sub_val) = xcorr_max_subsample(&cc, false).unwrap();
        assert!(abs(sub_lag + (shift as f64)) < 0.05);
        assert!(sub_val >= val - 1e-12);
    }

    #[test]
    fn test_morse_picker_spike_immunity_and_true_onset() {
        let n = if cfg!(miri) { 180 } else { 600 };
        let onset = if cfg!(miri) { 100 } else { 280 };
        let cfg = if cfg!(miri) {
            MorsePickConfig {
                nsta: 8,
                nlta: 50,
                thr_on: 4.5,
                prominence_sigma: 5.5,
                confirm_samples: 4,
            }
        } else {
            MorsePickConfig::default()
        };

        let mut spiky_noise = vec![0.0_f64; n];
        for (i, slot) in spiky_noise.iter_mut().enumerate() {
            *slot = 0.01 * sin((i as f64) * 0.37) + 0.005 * cos((i as f64) * 1.13);
        }
        spiky_noise[20] = 1.0e9;
        spiky_noise[75] = -5.0e8;
        spiky_noise[onset] = 2.5e10;
        let res_noise = morse_pick(&spiky_noise, &cfg).unwrap();
        assert_eq!(res_noise.pick_sample, None);

        let mut seismic = spiky_noise;
        seismic[onset] = 0.0;
        for (i, slot) in seismic.iter_mut().enumerate().skip(onset) {
            let dt = (i - onset) as f64 * 0.01;
            *slot += 6.0
                * (dt / 0.05).min(1.0)
                * earthsnail_core::math::exp(-dt * 2.0)
                * sin(25.0 * dt + 0.4);
        }
        let res_seis = morse_pick(&seismic, &cfg).unwrap();
        let picked = res_seis.pick_sample.expect("expected valid seismic pick");
        assert!(
            picked >= onset.saturating_sub(2) && picked <= onset + 8,
            "picked={picked}, expected near {onset}"
        );
    }

    #[test]
    fn test_aic_simple_matches_maeda() {
        let a = [1.0, -2.0, 3.5, -0.5, 4.0, -1.5, 2.0, 0.5, -3.0, 1.2];
        let aic = aic_simple(&a);
        assert_eq!(aic.len(), a.len());
        assert!(abs(aic[0] - 13.304671939241985) < 1e-11);
        assert!(abs(aic[1] - 12.99817799) < 1e-7);
        assert_eq!(aic[8], aic[9]);
    }
}
