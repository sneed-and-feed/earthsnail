//! Waveform preprocessing, detrending (`demean`/`rmean`, `detrend_linear`/`rtrend`, Householder-QR
//! `detrend_polynomial`), tapering (`Hann`, `Hamming`, `Cosine`, `SAC-Cosine`, `Kaiser`, `Blackman`,
//! `Tukey`), `Stream` gap/overlap `merge`, nanosecond-exact `trim`/`cut`, Lanczos & sinc resampling,
//! integer decimation, and time/spectral calculus.

use crate::fft::{irfft, rfft, rfftfreq};
use crate::filter::{cheby2, sosfilt_zerophase_obspy, FilterBand};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{FRAC_PI_2, PI, TAU};
use earthsnail_core::{
    math, Complex64, EarthSnailError, Result, Stream, Trace, TraceStats, UtcTimestamp,
};

// ============================================================================
// 1. Detrending (`demean`, `detrend_linear`, `detrend_polynomial`)
// ============================================================================

/// Subtracts the arithmetic mean in-place using compensated Kahan summation (`SAC rmean` / `ObsPy demean`).
pub fn demean_in_place(data: &mut [f64]) {
    if data.is_empty() {
        return;
    }
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &x in data.iter() {
        let y = x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    let mean = sum / (data.len() as f64);
    for x in data.iter_mut() {
        *x -= mean;
    }
}

/// Alias for [`demean_in_place`] matching classic SAC `rmean`.
#[inline]
pub fn rmean_in_place(data: &mut [f64]) {
    demean_in_place(data);
}

/// Returns a mean-removed copy of `data`.
pub fn demean(data: &[f64]) -> Vec<f64> {
    let mut out = data.to_vec();
    demean_in_place(&mut out);
    out
}

/// Removes the least-squares linear trend in-place (`scipy.signal.detrend(..., type='linear')` / `SAC rtrend`).
pub fn detrend_linear_in_place(data: &mut [f64]) {
    let n = data.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        data[0] = 0.0;
        return;
    }
    let nf = n as f64;
    let mid = 0.5 * ((n - 1) as f64);

    // Kahan sums for y and (i - mid) * y (exact orthogonal basis {1, i - mid}):
    let mut sum_y = 0.0_f64;
    let mut comp_y = 0.0_f64;
    let mut sum_ty = 0.0_f64;
    let mut comp_ty = 0.0_f64;

    for (i, &y_val) in data.iter().enumerate() {
        let t = (i as f64) - mid;
        let dy = y_val - comp_y;
        let sy = sum_y + dy;
        comp_y = (sy - sum_y) - dy;
        sum_y = sy;

        let ty = t * y_val;
        let dty = ty - comp_ty;
        let sty = sum_ty + dty;
        comp_ty = (sty - sum_ty) - dty;
        sum_ty = sty;
    }

    // Exact sum_{i=0}^{n-1} (i - mid)^2 = n * (n^2 - 1) / 12
    let denom = nf * (nf * nf - 1.0) / 12.0;
    let mean_y = sum_y / nf;
    let slope = if denom > 0.0 { sum_ty / denom } else { 0.0 };

    for (i, y_slot) in data.iter_mut().enumerate() {
        let t = (i as f64) - mid;
        *y_slot -= mean_y + slope * t;
    }
}

/// Alias for [`detrend_linear_in_place`] matching classic SAC `rtrend`.
#[inline]
pub fn rtrend_in_place(data: &mut [f64]) {
    detrend_linear_in_place(data);
}

/// Returns a linearly detrended copy of `data`.
pub fn detrend_linear(data: &[f64]) -> Vec<f64> {
    let mut out = data.to_vec();
    detrend_linear_in_place(&mut out);
    out
}

/// Subtracts the line connecting the first and last samples (`obspy.signal.detrend.simple`).
pub fn detrend_simple_in_place(data: &mut [f64]) {
    let n = data.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        data[0] = 0.0;
        return;
    }
    let y0 = data[0];
    let yn = data[n - 1];
    let slope = (yn - y0) / ((n - 1) as f64);
    for (i, slot) in data.iter_mut().enumerate() {
        *slot -= y0 + slope * (i as f64);
    }
}

/// Removes a least-squares polynomial trend of `order` (`0..=20`) in-place using Householder QR
/// factorization on the scaled domain $u_i \in [-1, 1]$ (`obspy.signal.detrend.polynomial`).
pub fn detrend_polynomial_in_place(data: &mut [f64], order: usize) -> Result<()> {
    if order > 20 {
        return Err(EarthSnailError::InvalidParameter(format!(
            "polynomial detrend order {order} exceeds maximum supported order 20"
        )));
    }
    let n = data.len();
    if n == 0 {
        return Ok(());
    }
    if order == 0 {
        demean_in_place(data);
        return Ok(());
    }
    if order == 1 {
        detrend_linear_in_place(data);
        return Ok(());
    }
    let p = order + 1;
    if n <= order {
        data.fill(0.0);
        return Ok(());
    }

    // Build column-major Vandermonde matrix A (n rows x p cols) on u_i = -1 + 2*i/(n-1):
    let inv_span = 2.0 / ((n - 1) as f64);
    let mut cols: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; p];
    for i in 0..n {
        let u = -1.0 + (i as f64) * inv_span;
        let mut upow = 1.0_f64;
        for col in cols.iter_mut() {
            col[i] = upow;
            upow *= u;
        }
    }

    // Copy rhs = data and apply in-place Householder reflections to A and rhs:
    let mut rhs = data.to_vec();
    let mut r_diag = vec![0.0_f64; p];

    for k in 0..p {
        let mut norm_sq = 0.0_f64;
        for &v in &cols[k][k..n] {
            norm_sq += v * v;
        }
        let norm = math::sqrt(norm_sq);
        if norm < 1e-15 {
            return Err(EarthSnailError::SingularMatrix(
                "rank-deficient Vandermonde system in detrend_polynomial".into(),
            ));
        }
        let x0 = cols[k][k];
        let alpha = -norm.copysign(if x0 == 0.0 { 1.0 } else { x0 });
        r_diag[k] = alpha;

        // Householder vector v stored in cols[k][k..n] with v[k] = x0 - alpha:
        let vk0 = x0 - alpha;
        cols[k][k] = vk0;
        let v_norm_sq = norm_sq - x0 * x0 + vk0 * vk0;
        if v_norm_sq <= 0.0 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // Reflect remaining columns j = (k+1)..p:
        let (left, right) = cols.split_at_mut(k + 1);
        let v_col = &left[k];
        for col_j in right.iter_mut() {
            let mut dot = 0.0_f64;
            for i in k..n {
                dot += v_col[i] * col_j[i];
            }
            let scale = beta * dot;
            for i in k..n {
                col_j[i] -= scale * v_col[i];
            }
        }

        // Reflect rhs:
        let mut dot_rhs = 0.0_f64;
        for i in k..n {
            dot_rhs += v_col[i] * rhs[i];
        }
        let scale_rhs = beta * dot_rhs;
        for i in k..n {
            rhs[i] -= scale_rhs * v_col[i];
        }
    }

    // Back-substitution on upper-triangular R * coeffs = rhs[0..p]:
    let mut coeffs = vec![0.0_f64; p];
    for k in (0..p).rev() {
        let mut acc = rhs[k];
        for j in (k + 1)..p {
            acc -= cols[j][k] * coeffs[j];
        }
        coeffs[k] = acc / r_diag[k];
    }

    // Evaluate fitted polynomial via Horner's rule and subtract:
    for (i, slot) in data.iter_mut().enumerate() {
        let u = -1.0 + (i as f64) * inv_span;
        let mut poly_val = 0.0_f64;
        for &c in coeffs.iter().rev() {
            poly_val = poly_val * u + c;
        }
        *slot -= poly_val;
    }
    Ok(())
}

/// Returns a copy of `data` with a polynomial trend of `order` subtracted.
pub fn detrend_polynomial(data: &[f64], order: usize) -> Result<Vec<f64>> {
    let mut out = data.to_vec();
    detrend_polynomial_in_place(&mut out, order)?;
    Ok(out)
}

// ============================================================================
// 2. Tapering (`hann`, `hamming`, `cosine`, `sac_cosine`, `kaiser`, `blackman`, `tukey`)
// ============================================================================

/// Which trace edge(s) to taper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaperSide {
    /// Symmetric taper at both start and end (`side='both'`).
    #[default]
    Both,
    /// Taper only the leading edge (`side='left'`).
    Left,
    /// Taper only the trailing edge (`side='right'`).
    Right,
}

/// Supported taper window shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaperType {
    /// Raised-cosine Hann window (`0.5 - 0.5 * cos(...)`).
    Hann,
    /// Hamming window (`0.54 - 0.46 * cos(...)`).
    Hamming,
    /// ObsPy half-cosine taper (`cosine_taper(..., halfcosine=true, sactaper=false)`).
    Cosine,
    /// Classic SAC quarter-cosine taper (`cosine_taper(..., halfcosine=false, sactaper=true)`).
    SacCosine,
    /// Kaiser–Bessel window parameterized by shape $\beta \ge 0$.
    Kaiser {
        /// Kaiser shape parameter $\beta$ (e.g. `8.6` for ~90 dB sidelobe attenuation).
        beta: f64,
    },
    /// 3-term Blackman window.
    Blackman,
    /// Cosine-tapered Tukey window (`alpha` in `[0.0, 1.0]`).
    Tukey {
        /// Tapered fraction $\alpha \in [0, 1]$ (`0` = rectangular, `1` = Hann).
        alpha: f64,
    },
}

/// Modified Bessel function of the first kind of order zero $I_0(x)$ accurate to `< 2 ULP` for all $x$.
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if !ax.is_finite() {
        return f64::INFINITY;
    }
    if ax <= 40.0 {
        // Strictly positive Maclaurin series sum_{k=0}^inf ((x/2)^2)^k / (k!)^2 — zero cancellation!
        let quarter_x2 = 0.25 * ax * ax;
        let mut sum = 1.0_f64;
        let mut term = 1.0_f64;
        for k in 1..120 {
            let kf = k as f64;
            term *= quarter_x2 / (kf * kf);
            sum += term;
            if term <= 1e-17 * sum {
                break;
            }
        }
        sum
    } else {
        // Hankel asymptotic expansion for large |x| > 40:
        // I_0(x) ~ exp(x) / sqrt(2*pi*x) * sum_{m=0} ((2m-1)!!)^2 / (m! * (8x)^m)
        let inv_8x = 1.0 / (8.0 * ax);
        let mut sum = 1.0_f64;
        let mut term = 1.0_f64;
        for m in 1..25 {
            let odd = (2 * m - 1) as f64;
            term *= (odd * odd) * inv_8x / (m as f64);
            if term.abs() <= 1e-17 * sum {
                sum += term;
                break;
            }
            sum += term;
        }
        (math::exp(ax) / math::sqrt(TAU * ax)) * sum
    }
}

/// Symmetric $N$-point Hann window (`scipy.signal.windows.hann(n, sym=True)`).
pub fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 0.5 - 0.5 * math::cos(TAU * (k as f64) / denom))
        .collect()
}

/// Symmetric $N$-point Hamming window (`scipy.signal.windows.hamming(n, sym=True)`).
pub fn hamming_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 0.54 - 0.46 * math::cos(TAU * (k as f64) / denom))
        .collect()
}

/// Symmetric $N$-point 3-term Blackman window (`scipy.signal.windows.blackman(n, sym=True)`).
pub fn blackman_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| {
            let phase = TAU * (k as f64) / denom;
            (0.42 - 0.5 * math::cos(phase) + 0.08 * math::cos(2.0 * phase)).max(0.0)
        })
        .collect()
}

/// Symmetric $N$-point Kaiser window (`scipy.signal.windows.kaiser(n, beta, sym=True)`).
pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom_i0 = bessel_i0(beta);
    let half = 0.5 * ((n - 1) as f64);
    (0..n)
        .map(|k| {
            let ratio = ((k as f64) - half) / half;
            let rad = (1.0 - ratio * ratio).max(0.0);
            bessel_i0(beta * math::sqrt(rad)) / denom_i0
        })
        .collect()
}

/// Symmetric $N$-point Tukey (tapered cosine) window with shape `alpha` in `[0.0, 1.0]`.
pub fn tukey_window(n: usize, alpha: f64) -> Result<Vec<f64>> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(EarthSnailError::InvalidParameter(
            "tukey alpha must be in [0.0, 1.0]".into(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 || alpha == 0.0 {
        return Ok(vec![1.0; n]);
    }
    if (alpha - 1.0).abs() == 0.0 {
        return Ok(hann_window(n));
    }
    let width = math::floor(alpha * ((n - 1) as f64) * 0.5) as usize;
    let denom = alpha * ((n - 1) as f64);
    let mut win = vec![1.0_f64; n];
    for (k, slot) in win.iter_mut().enumerate().take(width + 1) {
        *slot = 0.5 * (1.0 + math::cos(PI * (-1.0 + 2.0 * (k as f64) / denom)));
    }
    for (k, slot) in win.iter_mut().enumerate().skip(n - width - 1) {
        *slot = 0.5 * (1.0 + math::cos(PI * (-2.0 / alpha + 1.0 + 2.0 * (k as f64) / denom)));
    }
    Ok(win)
}

/// Exact replica of `obspy.signal.invsim.cosine_taper(npts, p, halfcosine, sactaper)` where `p`
/// in `[0.0, 1.0]` is the **total** tapered decimal fraction (`p = 0.10` tapers 5% on each end).
pub fn cosine_taper(npts: usize, p: f64, halfcosine: bool, sactaper: bool) -> Result<Vec<f64>> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(EarthSnailError::InvalidParameter(format!(
            "cosine_taper decimal percentage p={p} must lie in [0.0, 1.0]"
        )));
    }
    if npts == 0 {
        return Ok(Vec::new());
    }
    if npts == 1 {
        return Ok(vec![0.0]);
    }

    let frac: isize = if p == 0.0 || p == 1.0 {
        ((npts as f64) * p * 0.5) as isize
    } else {
        ((npts as f64) * p * 0.5 + 0.5) as isize
    };

    let idx1: isize = 0;
    let mut idx2: isize = frac - 1;
    let mut idx3: isize = (npts as isize) - frac;
    let idx4: isize = (npts as isize) - 1;

    if sactaper {
        idx2 += 1;
        idx3 -= 1;
    }
    if idx1 == idx2 {
        idx2 += 1;
    }
    if idx3 == idx4 {
        idx3 -= 1;
    }

    if idx2 < idx1 && idx3 > idx4 {
        return Ok(vec![1.0; npts]);
    }

    let mut cos_win = vec![0.0_f64; npts];
    let n_isize = npts as isize;

    if idx2 >= idx1 && idx2 < n_isize {
        let span_l = (idx2 - idx1) as f64;
        if span_l > 0.0 {
            for i in idx1..=idx2 {
                let iu = i as usize;
                cos_win[iu] = if halfcosine {
                    0.5 * (1.0 - math::cos(PI * ((i - idx1) as f64) / span_l))
                } else {
                    math::cos(-FRAC_PI_2 * ((idx2 - i) as f64) / span_l)
                };
            }
        }
    }

    let mid_start = (idx2 + 1).max(0) as usize;
    let mid_end = idx3.clamp(0, n_isize) as usize;
    for slot in cos_win
        .iter_mut()
        .take(mid_end.min(npts))
        .skip(mid_start.min(npts))
    {
        *slot = 1.0;
    }

    if idx4 >= idx3 && idx3 >= 0 {
        let span_r = (idx4 - idx3) as f64;
        if span_r > 0.0 {
            for i in idx3..=idx4.min(n_isize - 1) {
                let iu = i as usize;
                cos_win[iu] = if halfcosine {
                    0.5 * (1.0 + math::cos(PI * ((idx3 - i) as f64) / span_r))
                } else {
                    math::cos(FRAC_PI_2 * ((idx3 - i) as f64) / span_r)
                };
            }
        }
    }

    if idx1 == idx2 && (idx1 as usize) < npts {
        cos_win[idx1 as usize] = 0.0;
    }
    if idx3 == idx4 && idx3 >= 0 && (idx3 as usize) < npts {
        cos_win[idx3 as usize] = 0.0;
    }

    Ok(cos_win)
}

/// Classic SAC quarter-cosine taper where `half_fraction` in `[0.0, 0.5]` is the fraction tapered
/// on **each** side (`half_fraction = 0.05` -> `p_total = 0.10`, `sactaper = true`, `halfcosine = false`).
pub fn sac_cosine_taper(npts: usize, half_fraction: f64) -> Result<Vec<f64>> {
    if !half_fraction.is_finite() || !(0.0..=0.5).contains(&half_fraction) {
        return Err(EarthSnailError::InvalidParameter(format!(
            "sac_cosine_taper half_fraction={half_fraction} must lie in [0.0, 0.5]"
        )));
    }
    cosine_taper(npts, 2.0 * half_fraction, false, true)
}

/// Builds a taper window of length `npts` matching `obspy.core.trace.Trace.taper(max_percentage, type, side)`
/// where `max_percentage` in `[0.0, 0.5]` specifies the half-window fraction at **each** end.
pub fn taper_window(
    npts: usize,
    max_percentage: f64,
    taper_type: TaperType,
    side: TaperSide,
) -> Result<Vec<f64>> {
    if !max_percentage.is_finite() || !(0.0..=0.5).contains(&max_percentage) {
        return Err(EarthSnailError::InvalidParameter(format!(
            "max_percentage={max_percentage} must lie in [0.0, 0.5]"
        )));
    }
    if npts == 0 {
        return Ok(Vec::new());
    }
    if let TaperType::SacCosine = taper_type {
        let base = sac_cosine_taper(npts, max_percentage)?;
        return Ok(match side {
            TaperSide::Both => base,
            TaperSide::Left => {
                let mut w = base;
                w[npts / 2..].fill(1.0);
                w
            }
            TaperSide::Right => {
                let mut w = base;
                w[..npts / 2].fill(1.0);
                w
            }
        });
    }

    let wlen = ((max_percentage * (npts as f64)) as usize).min(npts / 2);
    if wlen == 0 {
        return Ok(vec![1.0; npts]);
    }
    let side_n = if 2 * wlen == npts {
        2 * wlen
    } else {
        2 * wlen + 1
    };

    let taper_sides = match taper_type {
        TaperType::Hann => hann_window(side_n),
        TaperType::Hamming => hamming_window(side_n),
        TaperType::Cosine => cosine_taper(side_n, 1.0, true, false)?,
        TaperType::SacCosine => unreachable!(),
        TaperType::Kaiser { beta } => {
            if !beta.is_finite() || beta < 0.0 {
                return Err(EarthSnailError::InvalidParameter(
                    "Kaiser beta must be finite and >= 0".into(),
                ));
            }
            kaiser_window(side_n, beta)
        }
        TaperType::Blackman => blackman_window(side_n),
        TaperType::Tukey { alpha } => tukey_window(side_n, alpha)?,
    };

    let mut out = vec![1.0_f64; npts];
    match side {
        TaperSide::Left => {
            out[..wlen].copy_from_slice(&taper_sides[..wlen]);
        }
        TaperSide::Right => {
            out[npts - wlen..].copy_from_slice(&taper_sides[side_n - wlen..]);
        }
        TaperSide::Both => {
            out[..wlen].copy_from_slice(&taper_sides[..wlen]);
            out[npts - wlen..].copy_from_slice(&taper_sides[side_n - wlen..]);
        }
    }
    Ok(out)
}

/// Applies `taper_window` in-place to `data`.
pub fn taper_in_place(
    data: &mut [f64],
    max_percentage: f64,
    taper_type: TaperType,
    side: TaperSide,
) -> Result<()> {
    let win = taper_window(data.len(), max_percentage, taper_type, side)?;
    for (x, w) in data.iter_mut().zip(win.iter()) {
        *x *= *w;
    }
    Ok(())
}

// ============================================================================
// 3. Stream `merge` & Nanosecond-Exact `trim` / `cut`
// ============================================================================

/// Gap and overlap resolution policy when merging segmented traces of the same channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Fill missing gap samples with `0.0` and resolve overlaps using the latest segment.
    #[default]
    FillZero,
    /// Linearly interpolate across gaps between adjacent segment endpoints and cross-fade overlaps.
    Interpolate,
    /// Overwrite overlapping samples with the later segment; fill gaps with `0.0` (or `fill_value`).
    Latest,
}

/// Merges traces in `stream` sharing `(ChannelId, sampling_rate)` into contiguous `Trace`s
/// according to `strategy`, using exact nanosecond sample-index arithmetic.
pub fn merge_stream(stream: &Stream, strategy: MergeStrategy) -> Result<Stream> {
    let mut non_empty: Vec<Trace> = stream
        .traces
        .iter()
        .filter(|t| !t.is_empty())
        .cloned()
        .collect();
    if non_empty.is_empty() {
        return Ok(Stream::new());
    }

    non_empty.sort_by(|a, b| {
        a.stats
            .channel
            .cmp(&b.stats.channel)
            .then_with(|| a.stats.sampling_rate.total_cmp(&b.stats.sampling_rate))
            .then_with(|| a.stats.starttime.cmp(&b.stats.starttime))
    });

    let mut merged_traces = Vec::new();
    let mut i = 0;
    while i < non_empty.len() {
        let first = &non_empty[i];
        let fs = first.stats.sampling_rate;
        if !fs.is_finite() || fs <= 0.0 {
            return Err(EarthSnailError::InvalidParameter(format!(
                "invalid sampling_rate {fs} in merge_stream"
            )));
        }
        let mut j = i + 1;
        while j < non_empty.len()
            && non_empty[j].stats.channel == first.stats.channel
            && (non_empty[j].stats.sampling_rate - fs).abs() <= 1e-6 * fs
        {
            j += 1;
        }

        let group = &non_empty[i..j];
        if group.len() == 1 {
            merged_traces.push(group[0].clone());
        } else {
            merged_traces.push(merge_compatible_group(group, strategy)?);
        }
        i = j;
    }

    Ok(Stream::from_traces(merged_traces))
}

fn merge_compatible_group(group: &[Trace], strategy: MergeStrategy) -> Result<Trace> {
    let base = &group[0];
    let fs = base.stats.sampling_rate;
    let t0 = base.stats.starttime;

    // Determine total span in samples relative to t0:
    let mut max_end_idx: i64 = (base.npts() as i64) - 1;
    for seg in &group[1..] {
        let seg_start_idx = t0.sample_index_nearest(seg.stats.starttime, fs)?;
        let seg_end_idx = seg_start_idx + (seg.npts() as i64) - 1;
        if seg_end_idx > max_end_idx {
            max_end_idx = seg_end_idx;
        }
    }
    let total_len = usize::try_from(max_end_idx + 1)
        .map_err(|_| EarthSnailError::CapacityExceeded("merged trace exceeds usize".into()))?;
    if total_len > 200_000_000 {
        return Err(EarthSnailError::CapacityExceeded(format!(
            "merged trace length {total_len} exceeds 200M sample safety bound"
        )));
    }

    let mut out = vec![0.0_f64; total_len];
    let mut covered = vec![false; total_len];
    out[..base.npts()].copy_from_slice(&base.data);
    covered[..base.npts()].fill(true);
    let mut prev_end_idx = base.npts() - 1;

    for seg in &group[1..] {
        let s_idx_i64 = t0.sample_index_nearest(seg.stats.starttime, fs)?;
        let s_idx = s_idx_i64.max(0) as usize;
        let e_idx = s_idx + seg.npts() - 1;

        if s_idx > prev_end_idx + 1 {
            // Gap [prev_end_idx + 1 .. s_idx)
            if strategy == MergeStrategy::Interpolate {
                let left_val = out[prev_end_idx];
                let right_val = seg.data[0];
                let span = (s_idx - prev_end_idx) as f64;
                for (k, slot) in out
                    .iter_mut()
                    .enumerate()
                    .take(s_idx)
                    .skip(prev_end_idx + 1)
                {
                    let frac = ((k - prev_end_idx) as f64) / span;
                    *slot = left_val + frac * (right_val - left_val);
                }
            }
        }

        // Write segment samples, handling overlap according to strategy:
        for (k, &val) in seg.data.iter().enumerate() {
            let dst = s_idx + k;
            if covered[dst] && strategy == MergeStrategy::Interpolate && s_idx <= prev_end_idx {
                let ov_len = (prev_end_idx - s_idx + 1).max(1) as f64;
                let w_new = ((k as f64) + 0.5) / ov_len;
                let w_clamped = w_new.clamp(0.0, 1.0);
                out[dst] = (1.0 - w_clamped) * out[dst] + w_clamped * val;
            } else {
                out[dst] = val;
            }
            covered[dst] = true;
        }

        if e_idx > prev_end_idx {
            prev_end_idx = e_idx;
        }
    }

    let mut new_stats: TraceStats = base.stats.clone();
    if new_stats.sac.is_some() {
        new_stats.sync_to_sac_header(&out);
    }
    Ok(Trace::new(new_stats, out))
}

/// Nanosecond-exact `Trace` window trim/cut onto `[start, end]`. When `pad_fill` is `Some(val)`,
/// samples requested outside the trace's time span are padded with `val` (`ObsPy trim(..., pad=True, fill_value=val)`);
/// when `None`, the window is clamped to available samples (`SAC cut`).
pub fn trim_trace(
    trace: &Trace,
    start: Option<UtcTimestamp>,
    end: Option<UtcTimestamp>,
    pad_fill: Option<f64>,
) -> Result<Trace> {
    if trace.is_empty() {
        return Ok(trace.clone());
    }
    let fs = trace.stats.sampling_rate;
    let t_start = start.unwrap_or(trace.stats.starttime);
    let t_end = end.unwrap_or_else(|| trace.endtime());
    if t_end < t_start {
        return Err(EarthSnailError::InvalidParameter(
            "trim_trace: end timestamp precedes start timestamp".into(),
        ));
    }

    if pad_fill.is_none() {
        return trace.slice_time(t_start, t_end);
    }

    let fill = pad_fill.unwrap_or(0.0);
    let i0 = trace.stats.starttime.sample_index_nearest(t_start, fs)?;
    let i1 = trace.stats.starttime.sample_index_nearest(t_end, fs)?;
    if i1 < i0 {
        let mut empty_stats = trace.stats.clone();
        empty_stats.starttime = trace.stats.starttime.sample_timestamp(i0, fs)?;
        return Ok(Trace::new(empty_stats, Vec::new()));
    }
    let out_len = usize::try_from(i1 - i0 + 1)
        .map_err(|_| EarthSnailError::CapacityExceeded("trim_trace length overflow".into()))?;
    if out_len > 200_000_000 {
        return Err(EarthSnailError::CapacityExceeded(
            "trim_trace padded length exceeds 200M sample bound".into(),
        ));
    }

    let mut out = vec![fill; out_len];
    let n_orig = trace.npts() as i64;
    let copy_start_orig = i0.max(0);
    let copy_end_orig = i1.min(n_orig - 1);
    if copy_start_orig <= copy_end_orig {
        let dst_offset = (copy_start_orig - i0) as usize;
        let src_start = copy_start_orig as usize;
        let count = (copy_end_orig - copy_start_orig + 1) as usize;
        out[dst_offset..dst_offset + count]
            .copy_from_slice(&trace.data[src_start..src_start + count]);
    }

    let new_starttime = trace.stats.starttime.sample_timestamp(i0, fs)?;
    let mut new_stats = trace.stats.clone();
    new_stats.starttime = new_starttime;
    if new_stats.sac.is_some() {
        new_stats.sync_to_sac_header(&out);
    }
    Ok(Trace::new(new_stats, out))
}

/// Applies [`trim_trace`] to every trace in `stream`.
pub fn trim_stream(
    stream: &Stream,
    start: Option<UtcTimestamp>,
    end: Option<UtcTimestamp>,
    pad_fill: Option<f64>,
) -> Result<Stream> {
    let mut out = Vec::with_capacity(stream.len());
    for tr in &stream.traces {
        out.push(trim_trace(tr, start, end, pad_fill)?);
    }
    Ok(Stream::from_traces(out))
}

// ============================================================================
// 4. Resampling (`decimate`, `lanczos_resample`, `sinc_resample`) & Calculus
// ============================================================================

/// Downsamples `data` by an integer `factor >= 1`, optionally applying a zero-phase Chebyshev Type II
/// anti-aliasing lowpass filter (`rs_db = 60 dB` at `0.8 * nyquist_new`) prior to decimation.
pub fn decimate(
    data: &[f64],
    sampling_rate_hz: f64,
    factor: usize,
    anti_alias_zerophase: bool,
) -> Result<Vec<f64>> {
    if factor == 0 {
        return Err(EarthSnailError::InvalidParameter(
            "decimate factor must be >= 1".into(),
        ));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "decimate requires positive sampling_rate_hz".into(),
        ));
    }
    if factor == 1 || data.is_empty() {
        return Ok(data.to_vec());
    }
    if anti_alias_zerophase {
        if factor > 16 {
            return Err(EarthSnailError::InvalidParameter(format!(
                "single-stage decimate factor {factor} > 16; cascade smaller factors"
            )));
        }
        let fc = 0.8 * (0.5 * sampling_rate_hz / (factor as f64));
        let sos = cheby2(8, 60.0, FilterBand::Lowpass(fc), sampling_rate_hz)?;
        let filtered = sosfilt_zerophase_obspy(&sos, data);
        Ok(filtered.into_iter().step_by(factor).collect())
    } else {
        Ok(data.iter().copied().step_by(factor).collect())
    }
}

/// Decimates a `Trace` by integer `factor`, updating `sampling_rate` and optional `SacHeader`.
pub fn decimate_trace(trace: &Trace, factor: usize, anti_alias_zerophase: bool) -> Result<Trace> {
    let new_data = decimate(
        &trace.data,
        trace.stats.sampling_rate,
        factor,
        anti_alias_zerophase,
    )?;
    let mut new_stats = trace.stats.clone();
    new_stats.sampling_rate /= factor as f64;
    if new_stats.sac.is_some() {
        new_stats.sync_to_sac_header(&new_data);
    }
    Ok(Trace::new(new_stats, new_data))
}

/// Window taper applied to the $\operatorname{sinc}$ reconstruction kernel in [`lanczos_resample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LanczosWindow {
    /// Classic Lanczos $\operatorname{sinc}(t/a)$ taper (`window="lanczos"`).
    #[default]
    Lanczos,
    /// Raised-cosine Hann taper $\frac{1}{2}(1 + \cos(\pi t / a))$ (`window="hanning"`).
    Hanning,
    /// Blackman taper $\frac{21}{50} + \frac{1}{2}\cos(\pi t / a) + \frac{2}{25}\cos(2\pi t / a)$ (`window="blackman"`).
    Blackman,
}

#[inline]
fn normalized_sinc(t: f64) -> f64 {
    if t.abs() < 1e-12 {
        let pt = PI * t;
        1.0 - (pt * pt) / 6.0
    } else {
        let pt = PI * t;
        math::sin(pt) / pt
    }
}

/// Evaluates the finite-support Lanczos kernel $L(t)$ at normalized sample offset `t`.
#[inline]
pub fn lanczos_kernel(t: f64, a: usize, window: LanczosWindow) -> f64 {
    let af = a as f64;
    if t == 0.0 {
        return 1.0;
    }
    if t.abs() >= af {
        return 0.0;
    }
    let s = normalized_sinc(t);
    let u = t / af;
    let tap = match window {
        LanczosWindow::Lanczos => normalized_sinc(u),
        LanczosWindow::Hanning => 0.5 * (1.0 + math::cos(PI * u)),
        LanczosWindow::Blackman => 0.42 + 0.5 * math::cos(PI * u) + 0.08 * math::cos(TAU * u),
    };
    s * tap
}

/// High-accuracy time-domain Lanczos interpolation matching `obspy.signal.interpolation.lanczos_interpolation`:
/// $$S(t_j) = \sum_{i = \lfloor \tau_j \rfloor - a + 1}^{\lfloor \tau_j \rfloor + a} s_i\,L(\tau_j - i), \qquad \tau_j = \frac{t_0^{\text{new}} - t_0^{\text{old}}}{\Delta t_{\text{old}}} + j \frac{\Delta t_{\text{new}}}{\Delta t_{\text{old}}}$$
#[allow(clippy::too_many_arguments)]
pub fn lanczos_resample(
    data: &[f64],
    old_start: f64,
    old_dt: f64,
    new_start: f64,
    new_dt: f64,
    new_npts: usize,
    a: usize,
    window: LanczosWindow,
) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "lanczos_resample input data is empty".into(),
        ));
    }
    if !old_dt.is_finite() || old_dt <= 0.0 || !new_dt.is_finite() || new_dt <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "lanczos_resample requires positive finite old_dt and new_dt".into(),
        ));
    }
    if a == 0 {
        return Err(EarthSnailError::InvalidParameter(
            "lanczos_resample half-width a must be >= 1".into(),
        ));
    }
    if new_npts == 0 {
        return Ok(Vec::new());
    }

    let old_end = old_start + old_dt * ((data.len() - 1) as f64);
    let new_end = new_start + new_dt * ((new_npts - 1) as f64);
    let eps_t = 1e-12 * old_dt;
    if new_start < old_start - eps_t || new_end > old_end + eps_t {
        return Err(EarthSnailError::InvalidParameter(format!(
            "lanczos_resample target span [{new_start}, {new_end}] exceeds input span [{old_start}, {old_end}]"
        )));
    }

    let dt_factor = new_dt / old_dt;
    let offset = ((new_start - old_start) / old_dt).max(0.0);
    let n_in = data.len() as isize;
    let a_isize = a as isize;

    let mut out = vec![0.0_f64; new_npts];
    for (j, out_slot) in out.iter_mut().enumerate() {
        let tau = offset + (j as f64) * dt_factor;
        let center = math::floor(tau) as isize;
        // Exact match to ObsPy's `lanczos.c`: when `tau` is an exact integer sample index, return `data[center]` directly
        if (tau - (center as f64)).abs() == 0.0 && center >= 0 && center < n_in {
            *out_slot = data[center as usize];
            continue;
        }
        let i_min = (center - a_isize + 1).max(0);
        let i_max = (center + a_isize).min(n_in - 1);
        let mut sum = 0.0_f64;
        for idx in i_min..=i_max {
            sum += data[idx as usize] * lanczos_kernel(tau - (idx as f64), a, window);
        }
        *out_slot = sum;
    }
    Ok(out)
}

/// Fourier-domain bandlimited sinc resampling of `data` from `data.len()` to `new_npts` samples
/// (`scipy.signal.resample` convention).
pub fn sinc_resample(data: &[f64], new_npts: usize) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "sinc_resample input data is empty".into(),
        ));
    }
    if new_npts == 0 {
        return Ok(Vec::new());
    }
    let old_npts = data.len();
    if new_npts == old_npts {
        return Ok(data.to_vec());
    }
    let spec_old = rfft(data, Some(old_npts))?;
    let new_bins = new_npts / 2 + 1;
    let mut spec_new = vec![Complex64::ZERO; new_bins];
    let copy_bins = spec_old.len().min(new_bins);
    spec_new[..copy_bins].copy_from_slice(&spec_old[..copy_bins]);

    // Handle unmatched even Nyquist bin factor of 2 / 0.5 matching `scipy.signal.resample`:
    if new_npts > old_npts && old_npts % 2 == 0 {
        spec_new[old_npts / 2] *= 0.5;
    } else if new_npts < old_npts && new_npts % 2 == 0 {
        spec_new[new_npts / 2] *= 2.0;
        spec_new[new_npts / 2].im = 0.0;
    }

    let mut rec = irfft(&spec_new, new_npts)?;
    let scale = (new_npts as f64) / (old_npts as f64);
    for v in &mut rec {
        *v *= scale;
    }
    Ok(rec)
}

/// Time-domain 2nd-order central-difference derivative (1st-order forward/backward at boundaries),
/// matching `numpy.gradient(data, dt)` and `Trace.differentiate(method='gradient')`.
pub fn differentiate_gradient(data: &[f64], dt_s: f64) -> Result<Vec<f64>> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "differentiate_gradient requires positive finite dt_s".into(),
        ));
    }
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![0.0]);
    }
    let mut out = vec![0.0_f64; n];
    out[0] = (data[1] - data[0]) / dt_s;
    let inv_2dt = 0.5 / dt_s;
    for i in 1..(n - 1) {
        out[i] = (data[i + 1] - data[i - 1]) * inv_2dt;
    }
    out[n - 1] = (data[n - 1] - data[n - 2]) / dt_s;
    Ok(out)
}

/// Cumulative trapezoidal time-domain integration (`out[0] = 0.0`, `out[i] = out[i-1] + 0.5*dt*(x[i-1]+x[i])`),
/// matching `scipy.integrate.cumulative_trapezoid(..., initial=0.0)`.
pub fn integrate_trapezoidal(data: &[f64], dt_s: f64) -> Result<Vec<f64>> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "integrate_trapezoidal requires positive finite dt_s".into(),
        ));
    }
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut out = vec![0.0_f64; n];
    let half_dt = 0.5 * dt_s;
    let mut acc = 0.0_f64;
    for i in 1..n {
        acc += half_dt * (data[i - 1] + data[i]);
        out[i] = acc;
    }
    Ok(out)
}

/// Spectral Fourier derivative multiplying $X(f)$ by $+i 2\pi f$ (zeroing the Nyquist bin when `n` is even).
pub fn differentiate_spectral(data: &[f64], dt_s: f64) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let n = data.len();
    let mut spec = rfft(data, Some(n))?;
    let freqs = rfftfreq(n, dt_s)?;
    for (k, bin) in spec.iter_mut().enumerate() {
        if n % 2 == 0 && k == n / 2 {
            *bin = Complex64::ZERO;
        } else {
            let omega = TAU * freqs[k];
            *bin = Complex64::new(-omega * bin.im, omega * bin.re);
        }
    }
    irfft(&spec, n)
}

/// Spectral Fourier integral dividing $X(f)$ by $+i 2\pi f$ for $f > 0$ (setting DC bin `f = 0` to `0.0`).
pub fn integrate_spectral(data: &[f64], dt_s: f64) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let n = data.len();
    let mut spec = rfft(data, Some(n))?;
    let freqs = rfftfreq(n, dt_s)?;
    spec[0] = Complex64::ZERO;
    for (k, bin) in spec.iter_mut().enumerate().skip(1) {
        let omega = TAU * freqs[k];
        // bin / (i * omega) = (bin.im - i * bin.re) / omega
        *bin = Complex64::new(bin.im / omega, -bin.re / omega);
    }
    if n % 2 == 0 {
        let nyq = spec.len() - 1;
        spec[nyq].im = 0.0;
    }
    irfft(&spec, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use earthsnail_core::ChannelId;

    #[test]
    fn polynomial_detrend_annihilates_exact_cubic() {
        let n = 120;
        let mut x: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) * 0.05 - 3.0;
                4.2 - 1.7 * t + 0.9 * t * t - 0.35 * t * t * t
            })
            .collect();
        detrend_polynomial_in_place(&mut x, 3).unwrap();
        for &v in &x {
            assert!(v.abs() < 1e-12, "residual {v} != 0");
        }
    }

    #[test]
    fn stream_merge_and_nanosecond_trim_handle_gaps_and_overlaps() {
        let t0: UtcTimestamp = "2025-06-01T00:00:00Z".parse().unwrap();
        let ch = ChannelId::new("IU", "ANMO", "00", "BHZ");
        // Segment 1: samples 0..10 (value 2.0) at 10 Hz (dt = 0.1 s -> 0.0..0.9 s)
        let tr1 = Trace::new(TraceStats::new(ch.clone(), t0, 10.0), vec![2.0; 10]);
        // Segment 2: starts at 1.4 s (sample index 14 -> 4-sample gap at 10, 11, 12, 13!), value 6.0
        let t2 = t0.sample_timestamp(14, 10.0).unwrap();
        let tr2 = Trace::new(TraceStats::new(ch.clone(), t2, 10.0), vec![6.0; 6]);

        let st = Stream::from_traces(vec![tr2, tr1]);
        let merged_zero = merge_stream(&st, MergeStrategy::FillZero).unwrap();
        assert_eq!(merged_zero.len(), 1);
        assert_eq!(merged_zero[0].npts(), 20);
        assert_eq!(merged_zero[0].data[9], 2.0);
        assert_eq!(merged_zero[0].data[10], 0.0);
        assert_eq!(merged_zero[0].data[13], 0.0);
        assert_eq!(merged_zero[0].data[14], 6.0);

        let merged_interp = merge_stream(&st, MergeStrategy::Interpolate).unwrap();
        // Gap indices 10, 11, 12, 13 between 2.0 (at 9) and 6.0 (at 14) -> span = 5 -> 2.8, 3.6, 4.4, 5.2!
        assert!((merged_interp[0].data[10] - 2.8).abs() < 1e-14);
        assert!((merged_interp[0].data[11] - 3.6).abs() < 1e-14);
        assert!((merged_interp[0].data[12] - 4.4).abs() < 1e-14);
        assert!((merged_interp[0].data[13] - 5.2).abs() < 1e-14);

        // Padded trim extending 2 samples before t0:
        let t_pre = t0.sample_timestamp(-2, 10.0).unwrap();
        let t_sub_end = t0.sample_timestamp(4, 10.0).unwrap();
        let trimmed =
            trim_trace(&merged_zero[0], Some(t_pre), Some(t_sub_end), Some(-1.0)).unwrap();
        assert_eq!(trimmed.npts(), 7);
        assert_eq!(&trimmed.data[..3], &[-1.0, -1.0, 2.0]);
        assert_eq!(trimmed.stats.starttime, t_pre);
    }

    #[test]
    fn spectral_and_time_calculus_consistency() {
        let n = 128;
        let dt = 1.0 / (n as f64);
        // Periodic smooth wave sin(2*pi*3*t):
        let x: Vec<f64> = (0..n)
            .map(|i| math::sin(TAU * 3.0 * (i as f64) * dt))
            .collect();
        let dx = differentiate_spectral(&x, dt).unwrap();
        let ix = integrate_spectral(&dx, dt).unwrap();
        for i in 0..n {
            let expected_dx = TAU * 3.0 * math::cos(TAU * 3.0 * (i as f64) * dt);
            assert!((dx[i] - expected_dx).abs() < 1e-11);
            assert!((ix[i] - x[i]).abs() < 1e-12);
        }
    }
}
