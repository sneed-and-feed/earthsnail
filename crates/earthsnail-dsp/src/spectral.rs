//! Spectral estimation (`welch_psd`), Peterson (1993) Global Noise Models (`get_nlnm`, `get_nhnm`),
//! McNamara & Buland (2004) Probabilistic Power Spectral Density (`Ppsd`), Morlet Continuous Wavelet
//! Transform (`cwt_morlet`), Konno–Ohmachi logarithmic smoothing (`konno_ohmachi_smooth`), 3-component
//! Nakamura $H/V$ Spectral Ratio (`hvsr`), and Hilbert analytic-signal attributes (`analytic_signal`,
//! `envelope`, `instantaneous_phase`, `instantaneous_frequency`).

use crate::fft::{
    fft_with_scratch, fftfreq, ifft_with_scratch, next_pow2, prev_pow2, rfft, rfftfreq,
};
use crate::preprocess::{
    blackman_window, cosine_taper, demean_in_place, detrend_linear_in_place,
    differentiate_gradient, hamming_window, hann_window, merge_stream, MergeStrategy,
};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
use earthsnail_core::{
    math, ChannelResponse, Complex64, EarthSnailError, KinematicUnit, Result, ScratchWorkspace,
    Stream, Trace,
};

// ============================================================================
// 1. Welch Power Spectral Density (`welch_psd`)
// ============================================================================

/// Segment detrending mode prior to windowing in [`welch_psd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WelchDetrend {
    /// Do not detrend segments.
    None,
    /// Subtract segment mean (`detrend='constant'`).
    #[default]
    Demean,
    /// Subtract segment least-squares line (`detrend='linear'`, used in McNamara `PPSD`).
    Linear,
}

/// Segment taper window shape in [`welch_psd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WelchWindow {
    /// Standard symmetric Hann window.
    #[default]
    Hann,
    /// 10%-each-end (`p = 0.20` total) cosine taper used by McNamara & Buland (2004) `PPSD`.
    Cosine20Pct,
    /// Symmetric Hamming window.
    Hamming,
    /// Symmetric 3-term Blackman window.
    Blackman,
}

/// Configuration for Welch's overlapped-segment periodogram [`welch_psd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WelchConfig {
    /// Samples per segment (`>= 2`).
    pub nperseg: usize,
    /// Overlapping samples between consecutive segments (`< nperseg`).
    pub noverlap: usize,
    /// Optional zero-padded FFT length (`>= nperseg`; defaults to `nperseg`).
    pub nfft: Option<usize>,
    /// Per-segment detrending mode.
    pub detrend: WelchDetrend,
    /// Per-segment window function.
    pub window: WelchWindow,
}

impl Default for WelchConfig {
    fn default() -> Self {
        Self {
            nperseg: 256,
            noverlap: 128,
            nfft: None,
            detrend: WelchDetrend::Demean,
            window: WelchWindow::Hann,
        }
    }
}

/// One-sided Power Spectral Density estimate in $(\text{units})^2 / \text{Hz}$.
#[derive(Debug, Clone, PartialEq)]
pub struct PsdEstimate {
    /// Frequency bins `[0, df, ..., fs/2]` in Hz.
    pub freqs_hz: Vec<f64>,
    /// One-sided PSD values in $(\text{units})^2 / \text{Hz}$.
    pub psd: Vec<f64>,
    /// Number of averaged segments.
    pub num_segments: usize,
}

/// Computes a one-sided Welch Power Spectral Density estimate (`scipy.signal.welch(..., scaling='density')`).
pub fn welch_psd(data: &[f64], sampling_rate_hz: f64, cfg: &WelchConfig) -> Result<PsdEstimate> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "welch_psd input is empty".into(),
        ));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "welch_psd requires positive finite sampling_rate_hz".into(),
        ));
    }
    let nperseg = cfg.nperseg.min(data.len());
    if nperseg < 2 {
        return Err(EarthSnailError::InvalidParameter(
            "welch_psd requires at least 2 samples per segment".into(),
        ));
    }
    let noverlap = cfg.noverlap.min(nperseg - 1);
    let step = nperseg - noverlap;
    let nfft = cfg.nfft.unwrap_or(nperseg).max(nperseg);

    let win = match cfg.window {
        WelchWindow::Hann => hann_window(nperseg),
        WelchWindow::Cosine20Pct => cosine_taper(nperseg, 0.20, true, false)?,
        WelchWindow::Hamming => hamming_window(nperseg),
        WelchWindow::Blackman => blackman_window(nperseg),
    };
    let win_energy: f64 = win.iter().map(|&w| w * w).sum();
    if win_energy <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "welch_psd window has zero energy".into(),
        ));
    }

    let n_bins = nfft / 2 + 1;
    let mut accum_psd = vec![0.0_f64; n_bins];
    let mut seg_buf = vec![0.0_f64; nperseg];
    let mut spec_buf = Vec::with_capacity(n_bins);
    let mut scratch = ScratchWorkspace::new();
    let mut num_segments = 0_usize;

    let mut offset = 0_usize;
    while offset + nperseg <= data.len() {
        seg_buf.copy_from_slice(&data[offset..offset + nperseg]);
        match cfg.detrend {
            WelchDetrend::None => {}
            WelchDetrend::Demean => demean_in_place(&mut seg_buf),
            WelchDetrend::Linear => detrend_linear_in_place(&mut seg_buf),
        }
        for (s, &w) in seg_buf.iter_mut().zip(win.iter()) {
            *s *= w;
        }
        crate::fft::rfft_with_scratch(&seg_buf, nfft, &mut spec_buf, &mut scratch)?;
        for (k, bin) in spec_buf.iter().enumerate() {
            accum_psd[k] += bin.norm_sqr();
        }
        num_segments += 1;
        offset += step;
    }

    let base_scale = 1.0 / (sampling_rate_hz * win_energy * (num_segments as f64));
    let even_nfft = nfft % 2 == 0;
    for (k, p) in accum_psd.iter_mut().enumerate() {
        let two_sided_factor = if k == 0 || (even_nfft && k == n_bins - 1) {
            1.0
        } else {
            2.0
        };
        *p *= two_sided_factor * base_scale;
    }

    let freqs_hz = rfftfreq(nfft, 1.0 / sampling_rate_hz)?;
    Ok(PsdEstimate {
        freqs_hz,
        psd: accum_psd,
        num_segments,
    })
}

// ============================================================================
// 2. Peterson (1993) Global New Low & High Noise Models (`NLNM` & `NHNM`)
// ============================================================================

/// USGS Open-File Report 93-322 (Peterson, 1993) Table 1: `NLNM` piecewise `(min_period_s, A, B)`
/// where $P_{\text{dB}}(T) = A + B \log_{10}(T)$ in $\text{dB rel. } 1\,(\text{m/s}^2)^2/\text{Hz}$.
const NLNM_SEGMENTS: [(f64, f64, f64); 21] = [
    (0.10, -162.36, 5.64),
    (0.17, -166.70, 0.00),
    (0.40, -170.00, -8.30),
    (0.80, -166.40, 28.90),
    (1.24, -168.60, 52.48),
    (2.40, -159.98, 29.81),
    (4.30, -141.10, 0.00),
    (5.00, -71.36, -99.77),
    (6.00, -97.26, -66.49),
    (10.00, -132.18, -31.57),
    (12.00, -205.27, 36.16),
    (15.60, -37.65, -104.33),
    (21.90, -114.37, -47.10),
    (31.60, -160.58, -16.28),
    (45.00, -187.50, 0.00),
    (70.00, -216.47, 15.70),
    (101.00, -185.00, 0.00),
    (154.00, -168.34, -7.61),
    (328.00, -217.43, 11.90),
    (600.00, -258.28, 26.60),
    (10_000.00, -346.88, 48.75),
];

/// USGS Open-File Report 93-322 (Peterson, 1993) Table 2: `NHNM` piecewise `(min_period_s, A, B)`
/// where $P_{\text{dB}}(T) = A + B \log_{10}(T)$ in $\text{dB rel. } 1\,(\text{m/s}^2)^2/\text{Hz}$.
const NHNM_SEGMENTS: [(f64, f64, f64); 11] = [
    (0.10, -108.73, -17.23),
    (0.22, -150.34, -80.50),
    (0.32, -122.31, -23.87),
    (0.80, -116.85, 32.51),
    (3.80, -108.48, 18.08),
    (4.60, -74.66, -32.95),
    (6.30, 0.66, -127.18),
    (7.90, -93.37, -22.42),
    (15.40, 73.54, -162.98),
    (20.00, -151.52, 10.01),
    (354.80, -206.66, 31.63),
];

#[inline]
fn eval_peterson_table(period_s: f64, table: &[(f64, f64, f64)]) -> f64 {
    let mut idx = 0_usize;
    for (i, &(p_min, _, _)) in table.iter().enumerate() {
        if period_s >= p_min - 1e-9 {
            idx = i;
        }
    }
    let (_, a, b) = table[idx];
    a + b * math::log10(period_s)
}

/// Evaluates Peterson (1993) New Low Noise Model (`NLNM`) at arbitrary `period_s` in
/// $\text{dB rel. } 1\,(\text{m/s}^2)^2/\text{Hz}$.
pub fn eval_nlnm_db(period_s: f64) -> Result<f64> {
    if !period_s.is_finite() || period_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "eval_nlnm_db requires positive finite period_s".into(),
        ));
    }
    Ok(eval_peterson_table(period_s, &NLNM_SEGMENTS))
}

/// Evaluates Peterson (1993) New High Noise Model (`NHNM`) at arbitrary `period_s` in
/// $\text{dB rel. } 1\,(\text{m/s}^2)^2/\text{Hz}$.
pub fn eval_nhnm_db(period_s: f64) -> Result<f64> {
    if !period_s.is_finite() || period_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "eval_nhnm_db requires positive finite period_s".into(),
        ));
    }
    Ok(eval_peterson_table(period_s, &NHNM_SEGMENTS))
}

#[inline]
fn round_sig6(val: f64) -> f64 {
    if val == 0.0 || !val.is_finite() {
        return val;
    }
    let exp10 = math::floor(math::log10(val.abs())) as i32;
    let digits = 5 - exp10;
    if digits >= 0 {
        let scale = math::powi(10.0, digits);
        math::round(val * scale) / scale
    } else {
        let scale = math::powi(10.0, -digits);
        math::round(val / scale) * scale
    }
}

/// Returns the 1001-point `(periods_s, power_db)` Peterson (1993) `NLNM` curve descending from
/// $10^5\,\text{s}$ down to $0.1\,\text{s}$ (`0.006` decade spacing), matching
/// `obspy.signal.spectral_estimation.get_nlnm()`.
pub fn get_nlnm() -> (Vec<f64>, Vec<f64>) {
    let mut periods = Vec::with_capacity(1001);
    let mut power_db = Vec::with_capacity(1001);
    for k in 0..=1000 {
        let log_p = 5.0 - 0.006 * (k as f64);
        let p = round_sig6(math::powf(10.0, log_p));
        periods.push(p);
        power_db.push(eval_peterson_table(p, &NLNM_SEGMENTS));
    }
    (periods, power_db)
}

/// Returns the 1001-point `(periods_s, power_db)` Peterson (1993) `NHNM` curve descending from
/// $10^5\,\text{s}$ down to $0.1\,\text{s}$ (`0.006` decade spacing), matching
/// `obspy.signal.spectral_estimation.get_nhnm()`.
pub fn get_nhnm() -> (Vec<f64>, Vec<f64>) {
    let mut periods = Vec::with_capacity(1001);
    let mut power_db = Vec::with_capacity(1001);
    for k in 0..=1000 {
        let log_p = 5.0 - 0.006 * (k as f64);
        let p = round_sig6(math::powf(10.0, log_p));
        periods.push(p);
        power_db.push(eval_peterson_table(p, &NHNM_SEGMENTS));
    }
    (periods, power_db)
}

// ============================================================================
// 3. McNamara & Buland (2004) Probabilistic Power Spectral Density (`Ppsd`)
// ============================================================================

/// Configuration for [`Ppsd`] accumulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpsdConfig {
    /// Window length per PPSD segment in seconds (`3600.0` default in McNamara & Buland 2004).
    pub ppsd_length_s: f64,
    /// Fractional overlap between consecutive PPSD segments in `[0.0, 0.95]` (`0.5` default).
    pub overlap: f64,
    /// Minimum histogram power bin edge in dB (`-200.0` default).
    pub db_min: f64,
    /// Maximum histogram power bin edge in dB (`-50.0` default).
    pub db_max: f64,
    /// Histogram bin width in dB (`1.0` default).
    pub db_step: f64,
    /// Logarithmic period smoothing width in octaves (`1.0` default).
    pub period_smoothing_width_octaves: f64,
    /// Center-period step size in octaves (`0.125` default = 1/8 octave).
    pub period_step_octaves: f64,
}

impl Default for PpsdConfig {
    fn default() -> Self {
        Self {
            ppsd_length_s: 3600.0,
            overlap: 0.5,
            db_min: -200.0,
            db_max: -50.0,
            db_step: 1.0,
            period_smoothing_width_octaves: 1.0,
            period_step_octaves: 0.125,
        }
    }
}

/// McNamara & Buland (2004) Probabilistic Power Spectral Density (`PPSD`) accumulator converting
/// raw sensor counts into acceleration PSD $(\text{m/s}^2)^2/\text{Hz}$ in dB across octave-smoothed
/// period bins and 2D power-density histograms.
#[derive(Debug, Clone)]
pub struct Ppsd {
    sampling_rate_hz: f64,
    seg_len: usize,
    seg_step: usize,
    sub_nfft: usize,
    sub_nlap: usize,
    psd_periods_asc: Vec<f64>,
    acc_resp_mag_sq_asc: Vec<f64>,
    period_left_edges: Vec<f64>,
    period_centers: Vec<f64>,
    period_right_edges: Vec<f64>,
    db_bin_edges: Vec<f64>,
    db_step: f64,
    binned_segments_db: Vec<Vec<f64>>,
    histogram_counts: Vec<Vec<u32>>,
}

impl Ppsd {
    /// Creates a new `Ppsd` accumulator for `sampling_rate_hz` and instrument `response`.
    pub fn new(sampling_rate_hz: f64, response: &ChannelResponse, cfg: PpsdConfig) -> Result<Self> {
        if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
            return Err(EarthSnailError::InvalidParameter(
                "Ppsd requires positive finite sampling_rate_hz".into(),
            ));
        }
        if !cfg.ppsd_length_s.is_finite() || cfg.ppsd_length_s <= 0.0 {
            return Err(EarthSnailError::InvalidParameter(
                "Ppsd ppsd_length_s must be > 0".into(),
            ));
        }
        if !(0.0..=0.95).contains(&cfg.overlap) {
            return Err(EarthSnailError::InvalidParameter(
                "Ppsd overlap must lie in [0.0, 0.95]".into(),
            ));
        }
        if cfg.db_max <= cfg.db_min || cfg.db_step <= 0.0 {
            return Err(EarthSnailError::InvalidParameter(
                "Ppsd invalid dB histogram bounds".into(),
            ));
        }

        let seg_len = math::round(cfg.ppsd_length_s * sampling_rate_hz) as usize;
        if seg_len < 16 {
            return Err(EarthSnailError::InvalidParameter(
                "Ppsd segment length must be at least 16 samples".into(),
            ));
        }
        let seg_step = ((math::round((1.0 - cfg.overlap) * (seg_len as f64))) as usize).max(1);

        // McNamara & Buland (2004): subsegment FFT length = prev_pow2(seg_len / 4), 75% overlap (13 subsegments)
        let sub_nfft = prev_pow2(seg_len / 4).max(8);
        let sub_nlap = (3 * sub_nfft) / 4;

        let freqs = rfftfreq(sub_nfft, 1.0 / sampling_rate_hz)?;
        // Drop DC (index 0) and reverse so periods are ascending:
        let mut psd_periods_asc = Vec::with_capacity(freqs.len() - 1);
        let mut acc_resp_mag_sq_asc = Vec::with_capacity(freqs.len() - 1);
        for &f in freqs[1..].iter().rev() {
            psd_periods_asc.push(1.0 / f);
            let h_acc = response.eval_freq_kinematic(f, KinematicUnit::Acceleration);
            acc_resp_mag_sq_asc.push(h_acc.norm_sqr());
        }

        // Construct logarithmic octave period bins spanning [psd_periods_asc[0], psd_periods_asc[last]]:
        let width_fac = math::powf(2.0, cfg.period_smoothing_width_octaves);
        let step_fac = math::powf(2.0, cfg.period_step_octaves);
        let mut per_left = (2.0 / sampling_rate_hz) / math::powf(step_fac, 8.0);
        let p_min_fft = psd_periods_asc[0];
        let p_max_fft = *psd_periods_asc.last().unwrap_or(&p_min_fft);

        let mut period_left_edges = Vec::new();
        let mut period_centers = Vec::new();
        let mut period_right_edges = Vec::new();

        while per_left < p_max_fft {
            let per_right = per_left * width_fac;
            let per_center = math::sqrt(per_left * per_right);
            if per_right > p_min_fft && per_left < p_max_fft {
                // Ensure at least one FFT period falls in [per_left, per_right], or nearest neighbor:
                period_left_edges.push(per_left);
                period_centers.push(per_center);
                period_right_edges.push(per_right);
            }
            per_left *= step_fac;
        }

        let n_db_bins = math::round((cfg.db_max - cfg.db_min) / cfg.db_step) as usize;
        let db_bin_edges: Vec<f64> = (0..=n_db_bins)
            .map(|i| cfg.db_min + (i as f64) * cfg.db_step)
            .collect();

        let n_per = period_centers.len();
        Ok(Self {
            sampling_rate_hz,
            seg_len,
            seg_step,
            sub_nfft,
            sub_nlap,
            psd_periods_asc,
            acc_resp_mag_sq_asc,
            period_left_edges,
            period_centers,
            period_right_edges,
            db_bin_edges,
            db_step: cfg.db_step,
            binned_segments_db: Vec::new(),
            histogram_counts: vec![vec![0_u32; n_db_bins]; n_per],
        })
    }

    /// Feeds contiguous waveform samples into the `Ppsd` accumulator, returning the number of new segments processed.
    pub fn add_samples(&mut self, data: &[f64]) -> Result<usize> {
        if data.len() < self.seg_len {
            return Ok(0);
        }
        let welch_cfg = WelchConfig {
            nperseg: self.sub_nfft,
            noverlap: self.sub_nlap,
            nfft: Some(self.sub_nfft),
            detrend: WelchDetrend::Linear,
            window: WelchWindow::Cosine20Pct,
        };
        let n_db = self.db_bin_edges.len() - 1;
        let db_min = self.db_bin_edges[0];
        let mut added = 0_usize;
        let mut pos = 0_usize;

        while pos + self.seg_len <= data.len() {
            let chunk = &data[pos..pos + self.seg_len];
            let est = welch_psd(chunk, self.sampling_rate_hz, &welch_cfg)?;

            // Convert counts^2/Hz -> acceleration (m/s^2)^2/Hz in dB in ascending period order:
            let mut spec_db_asc = Vec::with_capacity(self.psd_periods_asc.len());
            for (k, &p_raw) in est.psd[1..].iter().rev().enumerate() {
                let h_sq = self.acc_resp_mag_sq_asc[k];
                let p_acc = if h_sq > 0.0 {
                    (p_raw / h_sq).max(f64::MIN_POSITIVE)
                } else {
                    f64::MIN_POSITIVE
                };
                spec_db_asc.push(10.0 * math::log10(p_acc));
            }

            // Full-octave period bin averaging:
            let mut smoothed = Vec::with_capacity(self.period_centers.len());
            for b in 0..self.period_centers.len() {
                let pl = self.period_left_edges[b];
                let pr = self.period_right_edges[b];
                let mut sum = 0.0_f64;
                let mut cnt = 0_usize;
                for (k, &per) in self.psd_periods_asc.iter().enumerate() {
                    if per >= pl && per <= pr {
                        sum += spec_db_asc[k];
                        cnt += 1;
                    }
                }
                let val_db = if cnt > 0 {
                    sum / (cnt as f64)
                } else {
                    // Nearest period fallback for narrow coarse bins near Nyquist:
                    let pc = self.period_centers[b];
                    let mut best_k = 0;
                    let mut best_d = f64::INFINITY;
                    for (k, &per) in self.psd_periods_asc.iter().enumerate() {
                        let d = (per - pc).abs();
                        if d < best_d {
                            best_d = d;
                            best_k = k;
                        }
                    }
                    spec_db_asc[best_k]
                };
                smoothed.push(val_db);

                if n_db > 0 {
                    let raw_bin = math::floor((val_db - db_min) / self.db_step) as isize;
                    let bin_idx = raw_bin.clamp(0, (n_db as isize) - 1) as usize;
                    self.histogram_counts[b][bin_idx] += 1;
                }
            }

            self.binned_segments_db.push(smoothed);
            added += 1;
            pos += self.seg_step;
        }
        Ok(added)
    }

    /// Feeds a `Trace` into the `Ppsd` accumulator.
    pub fn add_trace(&mut self, trace: &Trace) -> Result<usize> {
        if (trace.stats.sampling_rate - self.sampling_rate_hz).abs() > 1e-4 * self.sampling_rate_hz
        {
            return Err(EarthSnailError::IncompatibleTraces(format!(
                "trace sampling_rate {} != Ppsd sampling_rate {}",
                trace.stats.sampling_rate, self.sampling_rate_hz
            )));
        }
        self.add_samples(&trace.data)
    }

    /// Merges gaps with zeros and feeds matching traces from `stream` into the `Ppsd`.
    pub fn add_stream(&mut self, stream: &Stream) -> Result<usize> {
        let merged = merge_stream(stream, MergeStrategy::FillZero)?;
        let mut total = 0;
        for tr in &merged.traces {
            if (tr.stats.sampling_rate - self.sampling_rate_hz).abs()
                <= 1e-4 * self.sampling_rate_hz
            {
                total += self.add_samples(&tr.data)?;
            }
        }
        Ok(total)
    }

    /// Total number of processed PPSD time segments.
    #[inline]
    pub fn num_segments(&self) -> usize {
        self.binned_segments_db.len()
    }

    /// Octave-smoothed center periods in seconds (ascending).
    #[inline]
    pub fn periods_s(&self) -> &[f64] {
        &self.period_centers
    }

    /// Power histogram bin edges in dB (`len == n_db_bins + 1`).
    #[inline]
    pub fn db_bin_edges(&self) -> &[f64] {
        &self.db_bin_edges
    }

    /// Raw `[n_periods][n_db_bins]` integer segment counts.
    #[inline]
    pub fn histogram_counts(&self) -> &[Vec<u32>] {
        &self.histogram_counts
    }

    /// Normalized `[n_periods][n_db_bins]` probability percentages (`0..=100%`).
    pub fn histogram_percent(&self) -> Vec<Vec<f64>> {
        let total = self.binned_segments_db.len().max(1) as f64;
        self.histogram_counts
            .iter()
            .map(|row| row.iter().map(|&c| (c as f64) * 100.0 / total).collect())
            .collect()
    }

    /// Mean dB curve across all accumulated segments at each `periods_s()` bin.
    pub fn mean_curve_db(&self) -> Result<Vec<f64>> {
        if self.binned_segments_db.is_empty() {
            return Err(EarthSnailError::EmptyData("Ppsd has 0 segments".into()));
        }
        let n_seg = self.binned_segments_db.len() as f64;
        let n_per = self.period_centers.len();
        let mut out = vec![0.0_f64; n_per];
        for seg in &self.binned_segments_db {
            for (b, &v) in seg.iter().enumerate() {
                out[b] += v;
            }
        }
        for v in &mut out {
            *v /= n_seg;
        }
        Ok(out)
    }

    /// Histogram mode dB curve (bin center with peak count) at each `periods_s()` bin.
    pub fn mode_curve_db(&self) -> Result<Vec<f64>> {
        if self.binned_segments_db.is_empty() {
            return Err(EarthSnailError::EmptyData("Ppsd has 0 segments".into()));
        }
        let db_min = self.db_bin_edges[0];
        let mut out = Vec::with_capacity(self.period_centers.len());
        for row in &self.histogram_counts {
            let mut best_idx = 0_usize;
            let mut best_cnt = 0_u32;
            for (idx, &c) in row.iter().enumerate() {
                if c > best_cnt {
                    best_cnt = c;
                    best_idx = idx;
                }
            }
            out.push(db_min + ((best_idx as f64) + 0.5) * self.db_step);
        }
        Ok(out)
    }

    /// Exact empirical `percentile` curve (`0.0..=100.0`) across accumulated segments.
    pub fn percentile_curve_db(&self, percentile: f64) -> Result<Vec<f64>> {
        if self.binned_segments_db.is_empty() {
            return Err(EarthSnailError::EmptyData("Ppsd has 0 segments".into()));
        }
        if !percentile.is_finite() || !(0.0..=100.0).contains(&percentile) {
            return Err(EarthSnailError::InvalidParameter(
                "percentile must lie in [0.0, 100.0]".into(),
            ));
        }
        let n_seg = self.binned_segments_db.len();
        let n_per = self.period_centers.len();
        let mut col = vec![0.0_f64; n_seg];
        let mut out = Vec::with_capacity(n_per);
        let rank = (percentile / 100.0) * ((n_seg - 1) as f64);
        let lo = math::floor(rank) as usize;
        let hi = (lo + 1).min(n_seg - 1);
        let frac = rank - (lo as f64);

        for b in 0..n_per {
            for (s, seg) in self.binned_segments_db.iter().enumerate() {
                col[s] = seg[b];
            }
            col.sort_by(|a, b_val| a.total_cmp(b_val));
            out.push(col[lo] + frac * (col[hi] - col[lo]));
        }
        Ok(out)
    }
}

// ============================================================================
// 4. Morlet Continuous Wavelet Transform (`cwt_morlet`)
// ============================================================================

/// Computes the Continuous Wavelet Transform (`CWT`) scalogram `[freqs_hz.len()][data.len()]`
/// using the complex Morlet wavelet $\psi_0(\eta) = \pi^{-1/4} e^{i\omega_0 \eta} e^{-\eta^2 / 2}$
/// (`omega0 = 6.0` standard; Torrence & Compo 1998 Fourier convolution).
pub fn cwt_morlet(
    data: &[f64],
    sampling_rate_hz: f64,
    freqs_hz: &[f64],
    omega0: f64,
) -> Result<Vec<Vec<Complex64>>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "cwt_morlet data is empty".into(),
        ));
    }
    if freqs_hz.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "cwt_morlet freqs_hz is empty".into(),
        ));
    }
    if !sampling_rate_hz.is_finite()
        || sampling_rate_hz <= 0.0
        || !omega0.is_finite()
        || omega0 <= 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "cwt_morlet requires positive sampling_rate_hz and omega0".into(),
        ));
    }
    let npts = data.len();
    let dt = 1.0 / sampling_rate_hz;
    let nfft = next_pow2(2 * npts);

    let mut scratch = ScratchWorkspace::new();
    let mut x_pad = vec![Complex64::ZERO; nfft];
    let mut mean = 0.0_f64;
    for &v in data {
        mean += v;
    }
    mean /= npts as f64;
    for (i, &v) in data.iter().enumerate() {
        x_pad[i] = Complex64::new(v - mean, 0.0);
    }
    fft_with_scratch(&mut x_pad, &mut scratch)?;

    let omega_bins: Vec<f64> = fftfreq(nfft, dt)?.into_iter().map(|f| TAU * f).collect();
    let pi_neg_quarter = math::powf(PI, -0.25);
    let scale_numer = omega0 + math::sqrt(2.0 + omega0 * omega0);

    let mut out = Vec::with_capacity(freqs_hz.len());
    let mut work = vec![Complex64::ZERO; nfft];

    for &f_target in freqs_hz {
        if !f_target.is_finite() || f_target <= 0.0 {
            return Err(EarthSnailError::InvalidParameter(format!(
                "cwt_morlet target frequency {f_target} must be > 0"
            )));
        }
        let scale = scale_numer / (2.0 * TAU * f_target);
        let norm = math::sqrt(TAU * scale / dt) * pi_neg_quarter;

        work.fill(Complex64::ZERO);
        // Daughter Morlet spectrum has support strictly on omega > 0 (analytic wavelet):
        for k in 1..=(nfft / 2) {
            let sw = scale * omega_bins[k];
            let diff = sw - omega0;
            let psi_hat = norm * math::exp(-0.5 * diff * diff);
            work[k] = x_pad[k] * psi_hat;
        }
        ifft_with_scratch(&mut work, &mut scratch)?;
        out.push(work[..npts].to_vec());
    }

    Ok(out)
}

// ============================================================================
// 5. Konno–Ohmachi Smoothing (`konno_ohmachi_smooth`) & 3C `hvsr`
// ============================================================================

/// Evaluates the Konno & Ohmachi (1998) logarithmic spectral smoothing window around `center_freq_hz`
/// (`obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing_window`):
/// $$W(f, f_c) = \left[\frac{\sin(b \log_{10}(f / f_c))}{b \log_{10}(f / f_c)}\right]^4$$
pub fn konno_ohmachi_window(
    freqs_hz: &[f64],
    center_freq_hz: f64,
    bandwidth: f64,
    normalize: bool,
) -> Result<Vec<f64>> {
    if !bandwidth.is_finite()
        || bandwidth <= 0.0
        || !center_freq_hz.is_finite()
        || center_freq_hz < 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "konno_ohmachi_window requires bandwidth > 0 and center_freq_hz >= 0".into(),
        ));
    }
    let n = freqs_hz.len();
    let mut win = vec![0.0_f64; n];
    if n == 0 {
        return Ok(win);
    }
    if center_freq_hz == 0.0 {
        for (w, &f) in win.iter_mut().zip(freqs_hz.iter()) {
            if f == 0.0 {
                *w = 1.0;
            }
        }
        if normalize {
            let sum: f64 = win.iter().sum();
            if sum > 0.0 {
                for w in &mut win {
                    *w /= sum;
                }
            }
        }
        return Ok(win);
    }

    for (w, &f) in win.iter_mut().zip(freqs_hz.iter()) {
        if f <= 0.0 {
            *w = 0.0;
        } else if (f - center_freq_hz).abs() <= 1e-15 * center_freq_hz {
            *w = 1.0;
        } else {
            let z = bandwidth * math::log10(f / center_freq_hz);
            if z.abs() < 1e-8 {
                let z2 = z * z;
                let s = 1.0 - z2 / 6.0;
                *w = (s * s) * (s * s);
            } else {
                let ratio = math::sin(z) / z;
                let r2 = ratio * ratio;
                *w = r2 * r2;
            }
        }
    }

    if normalize {
        let sum: f64 = win.iter().sum();
        if sum > 0.0 {
            for w in &mut win {
                *w /= sum;
            }
        }
    }
    Ok(win)
}

/// Smooths a 1D amplitude or power spectrum using the Konno–Ohmachi logarithmic window (`bandwidth = 40.0` standard).
pub fn konno_ohmachi_smooth(
    spectrum: &[f64],
    freqs_hz: &[f64],
    bandwidth: f64,
    normalize: bool,
) -> Result<Vec<f64>> {
    if spectrum.len() != freqs_hz.len() {
        return Err(EarthSnailError::InvalidParameter(
            "konno_ohmachi_smooth: spectrum and freqs_hz lengths must match".into(),
        ));
    }
    let n = spectrum.len();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let win = konno_ohmachi_window(freqs_hz, freqs_hz[i], bandwidth, normalize)?;
        let mut acc = 0.0_f64;
        for (s, w) in spectrum.iter().zip(win.iter()) {
            acc += s * w;
        }
        out[i] = acc;
    }
    Ok(out)
}

/// Configuration for Nakamura (1989) / SESAME 3-Component Horizontal-to-Vertical Spectral Ratio [`hvsr`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HvsrConfig {
    /// Window length in seconds (`25.0` default).
    pub window_len_s: f64,
    /// Fractional window overlap in `[0.0, 0.9]` (`0.5` default).
    pub overlap: f64,
    /// Konno–Ohmachi smoothing parameter $b$ (`40.0` default).
    pub ko_bandwidth: f64,
    /// Minimum search frequency in Hz for fundamental resonance peak $f_0$ (`0.2` default).
    pub min_peak_freq_hz: f64,
    /// Maximum search frequency in Hz for fundamental resonance peak $f_0$ (`20.0` default).
    pub max_peak_freq_hz: f64,
}

impl Default for HvsrConfig {
    fn default() -> Self {
        Self {
            window_len_s: 25.0,
            overlap: 0.5,
            ko_bandwidth: 40.0,
            min_peak_freq_hz: 0.2,
            max_peak_freq_hz: 20.0,
        }
    }
}

/// Output of 3-component Horizontal-to-Vertical Spectral Ratio [`hvsr`].
#[derive(Debug, Clone, PartialEq)]
pub struct HvsrResult {
    /// Positive frequency axis in Hz.
    pub freqs_hz: Vec<f64>,
    /// Geometric-mean $H/V$ spectral ratio curve across windows.
    pub hv_curve: Vec<f64>,
    /// Multiplicative $1\sigma$ geometric standard deviation factor $\exp(\sigma_{\ln(H/V)})$.
    pub hv_geom_std: Vec<f64>,
    /// Dominant site resonance frequency $f_0$ in Hz within `[min_peak_freq_hz, max_peak_freq_hz]`.
    pub peak_freq_hz: f64,
    /// Peak $H/V$ amplitude $A_0 = \max H/V(f_0)$.
    pub peak_amplitude: f64,
    /// Number of windows averaged.
    pub windows_used: usize,
}

/// Computes the 3-component Nakamura (1989) / SESAME Horizontal-to-Vertical Spectral Ratio (`HVSR`):
/// $$H/V(f) = \frac{\sqrt{\frac{1}{2}\left(S_N^2(f) + S_E^2(f)\right)}}{S_Z(f)}$$
/// where $S_Z, S_N, S_E$ are Konno–Ohmachi smoothed Fourier amplitude spectra per window, aggregated
/// via the geometric mean across windows.
pub fn hvsr(
    z: &[f64],
    n: &[f64],
    e: &[f64],
    sampling_rate_hz: f64,
    cfg: &HvsrConfig,
) -> Result<HvsrResult> {
    if z.is_empty() || z.len() != n.len() || z.len() != e.len() {
        return Err(EarthSnailError::IncompatibleTraces(
            "hvsr requires non-empty equal-length Z, N, E slices".into(),
        ));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "hvsr requires positive sampling_rate_hz".into(),
        ));
    }
    let win_len = ((cfg.window_len_s * sampling_rate_hz) as usize)
        .min(z.len())
        .max(8);
    if z.len() < win_len {
        return Err(EarthSnailError::InvalidParameter(
            "hvsr input shorter than window length".into(),
        ));
    }
    let step = (((1.0 - cfg.overlap.clamp(0.0, 0.9)) * (win_len as f64)) as usize).max(1);
    let nfft = next_pow2(win_len);
    let taper = cosine_taper(win_len, 0.10, true, false)?;
    let all_freqs = rfftfreq(nfft, 1.0 / sampling_rate_hz)?;
    let freqs_pos = all_freqs[1..].to_vec();
    let n_pos = freqs_pos.len();

    // Precompute normalized Konno-Ohmachi rows once and reuse across all windows and all 3 components!
    let mut ko_rows: Vec<Vec<f64>> = Vec::with_capacity(n_pos);
    for &fc in &freqs_pos {
        ko_rows.push(konno_ohmachi_window(
            &freqs_pos,
            fc,
            cfg.ko_bandwidth,
            true,
        )?);
    }

    let smooth_with_ko = |amp: &[f64]| -> Vec<f64> {
        ko_rows
            .iter()
            .map(|row| {
                row.iter()
                    .zip(amp.iter())
                    .map(|(&w, &a)| w * a)
                    .sum::<f64>()
            })
            .collect()
    };

    let mut log_hv_sum = vec![0.0_f64; n_pos];
    let mut log_hv_sq_sum = vec![0.0_f64; n_pos];
    let mut windows_used = 0_usize;

    let mut wz = vec![0.0_f64; win_len];
    let mut wn = vec![0.0_f64; win_len];
    let mut we = vec![0.0_f64; win_len];

    let mut pos = 0_usize;
    while pos + win_len <= z.len() {
        wz.copy_from_slice(&z[pos..pos + win_len]);
        wn.copy_from_slice(&n[pos..pos + win_len]);
        we.copy_from_slice(&e[pos..pos + win_len]);
        demean_in_place(&mut wz);
        demean_in_place(&mut wn);
        demean_in_place(&mut we);
        for i in 0..win_len {
            wz[i] *= taper[i];
            wn[i] *= taper[i];
            we[i] *= taper[i];
        }

        let sz_raw: Vec<f64> = rfft(&wz, Some(nfft))?[1..]
            .iter()
            .map(|c| c.abs())
            .collect();
        let sn_raw: Vec<f64> = rfft(&wn, Some(nfft))?[1..]
            .iter()
            .map(|c| c.abs())
            .collect();
        let se_raw: Vec<f64> = rfft(&we, Some(nfft))?[1..]
            .iter()
            .map(|c| c.abs())
            .collect();

        let sz = smooth_with_ko(&sz_raw);
        let sn = smooth_with_ko(&sn_raw);
        let se = smooth_with_ko(&se_raw);

        for k in 0..n_pos {
            let horiz = math::sqrt(0.5 * (sn[k] * sn[k] + se[k] * se[k]));
            let vert = sz[k].max(1e-30);
            let ratio = (horiz / vert).max(1e-30);
            let lr = math::ln(ratio);
            log_hv_sum[k] += lr;
            log_hv_sq_sum[k] += lr * lr;
        }
        windows_used += 1;
        pos += step;
    }

    let inv_w = 1.0 / (windows_used as f64);
    let mut hv_curve = Vec::with_capacity(n_pos);
    let mut hv_geom_std = Vec::with_capacity(n_pos);
    let mut peak_freq_hz = freqs_pos[0];
    let mut peak_amplitude = 0.0_f64;

    for k in 0..n_pos {
        let mean_ln = log_hv_sum[k] * inv_w;
        let var_ln = (log_hv_sq_sum[k] * inv_w - mean_ln * mean_ln).max(0.0);
        let hv = math::exp(mean_ln);
        hv_curve.push(hv);
        hv_geom_std.push(math::exp(math::sqrt(var_ln)));

        let f = freqs_pos[k];
        if f >= cfg.min_peak_freq_hz && f <= cfg.max_peak_freq_hz && hv > peak_amplitude {
            peak_amplitude = hv;
            peak_freq_hz = f;
        }
    }

    Ok(HvsrResult {
        freqs_hz: freqs_pos,
        hv_curve,
        hv_geom_std,
        peak_freq_hz,
        peak_amplitude,
        windows_used,
    })
}

// ============================================================================
// 6. Hilbert Transform (`analytic_signal`, `envelope`, `instantaneous_phase`, `instantaneous_frequency`)
// ============================================================================

/// Computes the causal discrete-time analytic signal $z[n] = x[n] + i\,\mathcal{H}\{x\}[n]$
/// via Marple's (1999) spectral algorithm (`scipy.signal.hilbert`).
pub fn analytic_signal(data: &[f64]) -> Result<Vec<Complex64>> {
    let mut scratch = ScratchWorkspace::new();
    analytic_signal_with_scratch(data, &mut scratch)
}

/// Scratch-backed [`analytic_signal`].
pub fn analytic_signal_with_scratch(
    data: &[f64],
    scratch: &mut ScratchWorkspace,
) -> Result<Vec<Complex64>> {
    if data.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "analytic_signal input is empty".into(),
        ));
    }
    let n = data.len();
    if n == 1 {
        return Ok(vec![Complex64::new(data[0], 0.0)]);
    }
    let mut buf: Vec<Complex64> = data.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    fft_with_scratch(&mut buf, scratch)?;

    // Marple spectral weights: H[0] = 1, H[1..(n+1)/2] = 2, H[n/2] = 1 (if even), negative freqs = 0:
    let pos_double_end = n.div_ceil(2);
    for z in buf.iter_mut().take(pos_double_end).skip(1) {
        *z *= 2.0;
    }
    let zero_start = n / 2 + 1;
    if zero_start < n {
        buf[zero_start..].fill(Complex64::ZERO);
    }

    ifft_with_scratch(&mut buf, scratch)?;
    Ok(buf)
}

/// Instantaneous amplitude envelope $E[n] = \sqrt{x[n]^2 + \mathcal{H}\{x\}[n]^2}$
/// (`obspy.signal.filter.envelope`).
pub fn envelope(data: &[f64]) -> Result<Vec<f64>> {
    let z = analytic_signal(data)?;
    Ok(z.into_iter().map(Complex64::abs).collect())
}

/// Principal instantaneous phase $\phi[n] = \operatorname{atan2}(\operatorname{Im} z[n], \operatorname{Re} z[n]) \in [-\pi, \pi]$.
pub fn instantaneous_phase(data: &[f64]) -> Result<Vec<f64>> {
    let z = analytic_signal(data)?;
    Ok(z.into_iter().map(Complex64::arg).collect())
}

/// Instantaneous frequency in Hz computed from the analytic signal and its central-difference derivative:
/// $$f_{\text{inst}}(t) = \frac{x(t)\dot{y}(t) - y(t)\dot{x}(t)}{2\pi (x(t)^2 + y(t)^2)}$$
/// matching `obspy.signal.polarization.instantaneous_frequency`.
pub fn instantaneous_frequency(data: &[f64], sampling_rate_hz: f64) -> Result<Vec<f64>> {
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "instantaneous_frequency requires positive finite sampling_rate_hz".into(),
        ));
    }
    let z = analytic_signal(data)?;
    let re: Vec<f64> = z.iter().map(|c| c.re).collect();
    let im: Vec<f64> = z.iter().map(|c| c.im).collect();
    let dt = 1.0 / sampling_rate_hz;
    let dre = differentiate_gradient(&re, dt)?;
    let dim = differentiate_gradient(&im, dt)?;

    let mut out = Vec::with_capacity(z.len());
    for i in 0..z.len() {
        let mag_sq = z[i].norm_sqr();
        if mag_sq > 1e-30 {
            out.push((re[i] * dim[i] - im[i] * dre[i]) / (TAU * mag_sq));
        } else {
            out.push(0.0);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cwt_morlet_and_hvsr_identify_exact_resonance_frequency() {
        let fs = 50.0;
        let npts = if cfg!(miri) { 128 } else { 512 };
        // Broadband ambient microtremor comb (0.5..16 Hz) with a 5x horizontal site resonance peak at 4.0 Hz:
        let mut z = vec![0.0_f64; npts];
        let mut n = vec![0.0_f64; npts];
        for (i, (zv, nv)) in z.iter_mut().zip(n.iter_mut()).enumerate() {
            let t = (i as f64) / fs;
            let mut bg = 0.0_f64;
            let mut h_amp = 0.0_f64;
            for m in 1..=30 {
                let f = 0.5 * (m as f64);
                let phase = (m as f64) * 1.618_033_988_75;
                let comp = math::sin(TAU * f * t + phase);
                bg += 0.15 * comp;
                // Lorentzian site transfer function centered at f0 = 4.0 Hz (peak amplification = 5.0):
                let df_norm = (f - 4.0) / 0.6;
                let site_gain = 1.0 + 4.0 / (1.0 + df_norm * df_norm);
                h_amp += 0.15 * site_gain * comp;
            }
            *zv = bg;
            *nv = h_amp;
        }
        let e = n.clone();

        let hv = hvsr(
            &z,
            &n,
            &e,
            fs,
            &HvsrConfig {
                window_len_s: (npts as f64) / fs,
                overlap: 0.0,
                ko_bandwidth: 40.0,
                min_peak_freq_hz: 0.5,
                max_peak_freq_hz: 15.0,
            },
        )
        .unwrap();
        assert!(
            (hv.peak_freq_hz - 4.0).abs() < 0.6,
            "expected HVSR peak ~4.0 Hz, got {} Hz",
            hv.peak_freq_hz
        );
        assert!(hv.peak_amplitude > 2.5);

        // Morlet CWT peaks at 4.0 Hz over 1.5 Hz and 10.0 Hz:
        let freqs = [1.5, 4.0, 10.0];
        let scalogram = cwt_morlet(&n, fs, &freqs, 6.0).unwrap();
        let mid = npts / 2;
        assert!(scalogram[1][mid].abs() > scalogram[0][mid].abs() * 2.0);
        assert!(scalogram[1][mid].abs() > scalogram[2][mid].abs() * 2.0);
    }

    #[test]
    fn ppsd_accumulates_broadband_segments_and_percentiles() {
        let fs = 20.0;
        let resp = ChannelResponse::from_sacpz(
            vec![Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            vec![
                Complex64::new(-0.037_004, 0.037_016),
                Complex64::new(-0.037_004, -0.037_016),
            ],
            1.5e9,
        );
        let seg_s = if cfg!(miri) { 4.0 } else { 32.0 };
        let mut ppsd = Ppsd::new(
            fs,
            &resp,
            PpsdConfig {
                ppsd_length_s: seg_s,
                overlap: 0.5,
                ..PpsdConfig::default()
            },
        )
        .unwrap();

        let total_n = ((seg_s * 2.5 * fs) as usize).max(160);
        let data: Vec<f64> = (0..total_n)
            .map(|i| {
                let t = (i as f64) / fs;
                150.0 * math::sin(TAU * 0.5 * t) + 40.0 * math::cos(TAU * 2.0 * t)
            })
            .collect();
        let added = ppsd.add_samples(&data).unwrap();
        assert!(added >= 3);
        let mean_db = ppsd.mean_curve_db().unwrap();
        let p50 = ppsd.percentile_curve_db(50.0).unwrap();
        let mode_db = ppsd.mode_curve_db().unwrap();
        assert_eq!(mean_db.len(), ppsd.periods_s().len());
        assert_eq!(p50.len(), mean_db.len());
        assert_eq!(mode_db.len(), mean_db.len());
    }
}
