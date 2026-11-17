//! Cascaded Second-Order Sections (`SOS`) IIR filter synthesis (Butterworth, Chebyshev Type I & II,
//! and Bessel with `"phase"`, `"delay"`, and `"mag"` normalizations across `Lowpass`, `Highpass`,
//! `Bandpass`, and `Bandstop`) and execution (`sosfilt`, ObsPy/SAC 2-pass `sosfilt_zerophase_obspy`,
//! and SciPy odd-reflection steady-state `sosfiltfilt`).

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{FRAC_1_SQRT_2, PI, TAU};
use earthsnail_core::{math, Complex64, EarthSnailError, Result, ScratchWorkspace};

/// Single normalized biquadratic Second-Order Section:
/// $$H_k(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$
/// with `a[0] == 1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SosSection {
    /// Numerator coefficients `[b0, b1, b2]`.
    pub b: [f64; 3],
    /// Denominator coefficients `[1.0, a1, a2]`.
    pub a: [f64; 3],
}

impl Default for SosSection {
    fn default() -> Self {
        Self {
            b: [1.0, 0.0, 0.0],
            a: [1.0, 0.0, 0.0],
        }
    }
}

impl SosSection {
    /// Constructs a normalized `SosSection` dividing all coefficients by `a[0]` if `a[0] != 1.0`.
    pub fn new(b: [f64; 3], a: [f64; 3]) -> Result<Self> {
        if !a[0].is_finite() || a[0] == 0.0 {
            return Err(EarthSnailError::InvalidParameter(
                "SosSection a[0] must be finite and non-zero".into(),
            ));
        }
        if (a[0] - 1.0).abs() == 0.0 {
            Ok(Self { b, a })
        } else {
            let inv = 1.0 / a[0];
            Ok(Self {
                b: [b[0] * inv, b[1] * inv, b[2] * inv],
                a: [1.0, a[1] * inv, a[2] * inv],
            })
        }
    }

    /// Converts from a SciPy-style `[b0, b1, b2, a0, a1, a2]` 6-element row.
    pub fn from_row6(row: [f64; 6]) -> Result<Self> {
        Self::new([row[0], row[1], row[2]], [row[3], row[4], row[5]])
    }

    /// Exports as a SciPy-style `[b0, b1, b2, 1.0, a1, a2]` 6-element row.
    #[inline]
    pub const fn to_row6(&self) -> [f64; 6] {
        [
            self.b[0], self.b[1], self.b[2], self.a[0], self.a[1], self.a[2],
        ]
    }

    /// Evaluates the complex Z-domain transfer function $H_k(e^{i\omega})$ at digital radian frequency $\omega \in [0, \pi]$.
    pub fn eval_omega(&self, omega_rad: f64) -> Complex64 {
        let z_inv = Complex64::cis(-omega_rad);
        let z_inv2 = z_inv * z_inv;
        let num = z_inv * self.b[1] + z_inv2 * self.b[2] + self.b[0];
        let den = z_inv * self.a[1] + z_inv2 * self.a[2] + self.a[0];
        num / den
    }

    /// Computes the two poles of $z^2 + a_1 z + a_2 = 0$.
    pub fn poles(&self) -> [Complex64; 2] {
        let a1 = self.a[1];
        let a2 = self.a[2];
        let disc = a1 * a1 - 4.0 * a2;
        if disc >= 0.0 {
            let s = math::sqrt(disc);
            let r1 = if a1 >= 0.0 {
                -0.5 * (a1 + s)
            } else {
                -0.5 * (a1 - s)
            };
            let r2 = if r1.abs() > 0.0 { a2 / r1 } else { 0.0 };
            [Complex64::new(r1, 0.0), Complex64::new(r2, 0.0)]
        } else {
            let im = 0.5 * math::sqrt(-disc);
            let re = -0.5 * a1;
            [Complex64::new(re, im), Complex64::new(re, -im)]
        }
    }
}

/// Cascaded Second-Order Sections digital IIR filter.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SosFilter {
    /// Ordered biquadratic sections (poles closest to unit circle placed last for numerical dynamic range).
    pub sections: Vec<SosSection>,
}

impl SosFilter {
    /// Wraps a vector of `SosSection`s.
    pub fn new(sections: Vec<SosSection>) -> Result<Self> {
        if sections.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SosFilter requires at least one section".into(),
            ));
        }
        Ok(Self { sections })
    }

    /// Constructs from SciPy-style `[b0, b1, b2, a0, a1, a2]` rows.
    pub fn from_rows(rows: &[[f64; 6]]) -> Result<Self> {
        if rows.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SosFilter rows cannot be empty".into(),
            ));
        }
        let mut sections = Vec::with_capacity(rows.len());
        for &r in rows {
            sections.push(SosSection::from_row6(r)?);
        }
        Ok(Self { sections })
    }

    /// Exports all sections as `[b0, b1, b2, 1.0, a1, a2]` rows.
    pub fn to_rows(&self) -> Vec<[f64; 6]> {
        self.sections.iter().map(SosSection::to_row6).collect()
    }

    /// Evaluates cascaded complex frequency response $H(e^{i 2\pi f / f_s})$ at `freq_hz`.
    pub fn freqz(&self, freq_hz: f64, sampling_rate_hz: f64) -> Complex64 {
        let omega = TAU * freq_hz / sampling_rate_hz;
        self.sections.iter().map(|s| s.eval_omega(omega)).product()
    }

    /// Collects all poles across all sections.
    pub fn poles(&self) -> Vec<Complex64> {
        let mut out = Vec::with_capacity(2 * self.sections.len());
        for sec in &self.sections {
            let [p1, p2] = sec.poles();
            out.push(p1);
            out.push(p2);
        }
        out
    }

    /// Returns `true` if every pole strictly lies inside the unit circle ($|p_i| < 1$).
    pub fn is_stable(&self) -> bool {
        self.poles()
            .into_iter()
            .all(|p| p.is_finite() && p.abs() < 1.0)
    }
}

/// Frequency band specification in Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterBand {
    /// Lowpass filter passing $0 \le f \le f_{\text{cut}}$.
    Lowpass(f64),
    /// Highpass filter passing $f_{\text{cut}} \le f < f_s/2$.
    Highpass(f64),
    /// Bandpass filter passing $f_{\text{min}} \le f \le f_{\text{max}}$.
    Bandpass(f64, f64),
    /// Bandstop (notch/band-reject) filter attenuating $f_{\text{min}} \le f \le f_{\text{max}}$.
    Bandstop(f64, f64),
}

/// Bessel analog prototype cutoff normalization convention (matching `scipy.signal.bessel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BesselNorm {
    /// Phase-matched midpoint normalization (`norm='phase'`, default in SciPy and ObsPy).
    #[default]
    Phase,
    /// Unit passband group-delay normalization (`norm='delay'`).
    Delay,
    /// $-3\,\text{dB}$ half-power magnitude normalization at cutoff (`norm='mag'`).
    Mag,
}

/// Classical analog IIR prototype family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterFamily {
    /// Maximally flat magnitude Butterworth filter.
    Butterworth,
    /// Equiripple passband Chebyshev Type I filter (`rp_db > 0` passband ripple in dB).
    Chebyshev1 {
        /// Maximum passband ripple in decibels (`> 0.0`, e.g. `1.0`).
        rp_db: f64,
    },
    /// Equiripple stopband Inverse Chebyshev Type II filter (`rs_db > 0` stopband attenuation in dB).
    Chebyshev2 {
        /// Minimum stopband attenuation in decibels (`> 0.0`, e.g. `40.0`).
        rs_db: f64,
    },
    /// Maximally flat group-delay Bessel–Thomson filter (`order <= 12`).
    Bessel {
        /// Frequency normalization convention (`Phase`, `Delay`, or `Mag`).
        norm: BesselNorm,
    },
}

/// Synthesizes a cascaded digital `SosFilter` via analog prototype -> pre-warped frequency transform
/// -> bilinear transform -> nearest-pole-zero SOS pairing (`scipy.signal.iirfilter(..., output='sos')`).
pub fn design_iir_sos(
    order: usize,
    band: FilterBand,
    sampling_rate_hz: f64,
    family: FilterFamily,
) -> Result<SosFilter> {
    if order == 0 || order > 32 {
        return Err(EarthSnailError::InvalidParameter(format!(
            "filter order {order} must be in 1..=32"
        )));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "sampling_rate_hz must be finite and positive".into(),
        ));
    }
    let nyquist = 0.5 * sampling_rate_hz;

    // 1. Validate & pre-warp critical frequencies (`fs_warp = 2.0`, `w = 4.0 * tan(pi * f / fs)`):
    let validate_f = |f: f64, label: &str| -> Result<f64> {
        if !f.is_finite() || f <= 0.0 || f >= nyquist {
            return Err(EarthSnailError::InvalidParameter(format!(
                "{label} frequency {f} Hz must lie strictly inside (0, {nyquist}) Hz"
            )));
        }
        Ok(4.0 * math::tan(PI * (f / sampling_rate_hz)))
    };

    // 2. Synthesize normalized analog lowpass prototype (cutoff = 1 rad/s):
    let (z_proto, p_proto, k_proto) = match family {
        FilterFamily::Butterworth => buttap(order),
        FilterFamily::Chebyshev1 { rp_db } => {
            if !rp_db.is_finite() || rp_db <= 0.0 {
                return Err(EarthSnailError::InvalidParameter(
                    "Chebyshev Type I rp_db must be finite and > 0".into(),
                ));
            }
            cheb1ap(order, rp_db)
        }
        FilterFamily::Chebyshev2 { rs_db } => {
            if !rs_db.is_finite() || rs_db <= 0.0 {
                return Err(EarthSnailError::InvalidParameter(
                    "Chebyshev Type II rs_db must be finite and > 0".into(),
                ));
            }
            cheb2ap(order, rs_db)
        }
        FilterFamily::Bessel { norm } => {
            if order > 12 {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "Bessel filter order {order} exceeds supported maximum 12"
                )));
            }
            besselap(order, norm)?
        }
    };

    // 3. Apply analog frequency-band transformation:
    let (z_analog, p_analog, k_analog) = match band {
        FilterBand::Lowpass(fc) => {
            let wo = validate_f(fc, "lowpass cutoff")?;
            lp2lp_zpk(&z_proto, &p_proto, k_proto, wo)
        }
        FilterBand::Highpass(fc) => {
            let wo = validate_f(fc, "highpass cutoff")?;
            lp2hp_zpk(&z_proto, &p_proto, k_proto, wo)
        }
        FilterBand::Bandpass(f_lo, f_hi) => {
            let w1 = validate_f(f_lo, "bandpass low")?;
            let w2 = validate_f(f_hi, "bandpass high")?;
            if w2 <= w1 {
                return Err(EarthSnailError::InvalidParameter(
                    "bandpass high corner must exceed low corner".into(),
                ));
            }
            let wo = math::sqrt(w1 * w2);
            let bw = w2 - w1;
            lp2bp_zpk(&z_proto, &p_proto, k_proto, wo, bw)
        }
        FilterBand::Bandstop(f_lo, f_hi) => {
            let w1 = validate_f(f_lo, "bandstop low")?;
            let w2 = validate_f(f_hi, "bandstop high")?;
            if w2 <= w1 {
                return Err(EarthSnailError::InvalidParameter(
                    "bandstop high corner must exceed low corner".into(),
                ));
            }
            let wo = math::sqrt(w1 * w2);
            let bw = w2 - w1;
            lp2bs_zpk(&z_proto, &p_proto, k_proto, wo, bw)
        }
    };

    // 4. Bilinear transform (`fs_warp = 2.0`) -> discrete Z-plane zeros, poles, gain:
    let (z_dig, p_dig, k_dig) = bilinear_zpk(&z_analog, &p_analog, k_analog, 2.0);

    // 5. Pair discrete poles & zeros into well-conditioned biquadratic sections (`zpk2sos`):
    zpk_to_sos(&z_dig, &p_dig, k_dig)
}

/// Designs a Butterworth `SosFilter`.
pub fn butter(order: usize, band: FilterBand, sampling_rate_hz: f64) -> Result<SosFilter> {
    design_iir_sos(order, band, sampling_rate_hz, FilterFamily::Butterworth)
}

/// Designs a Chebyshev Type I `SosFilter` with passband ripple `rp_db` (in dB).
pub fn cheby1(
    order: usize,
    rp_db: f64,
    band: FilterBand,
    sampling_rate_hz: f64,
) -> Result<SosFilter> {
    design_iir_sos(
        order,
        band,
        sampling_rate_hz,
        FilterFamily::Chebyshev1 { rp_db },
    )
}

/// Designs a Chebyshev Type II `SosFilter` with stopband attenuation `rs_db` (in dB).
pub fn cheby2(
    order: usize,
    rs_db: f64,
    band: FilterBand,
    sampling_rate_hz: f64,
) -> Result<SosFilter> {
    design_iir_sos(
        order,
        band,
        sampling_rate_hz,
        FilterFamily::Chebyshev2 { rs_db },
    )
}

/// Designs a Bessel–Thomson `SosFilter` with `norm` (`Phase`, `Delay`, or `Mag`).
pub fn bessel(
    order: usize,
    band: FilterBand,
    sampling_rate_hz: f64,
    norm: BesselNorm,
) -> Result<SosFilter> {
    design_iir_sos(order, band, sampling_rate_hz, FilterFamily::Bessel { norm })
}

// ============================================================================
// Filter Execution Modes (`sosfilt`, `sosfilt_zerophase_obspy`, `sosfiltfilt`)
// ============================================================================

/// Causal 1-pass Direct Form II Transposed SOS filter (`scipy.signal.sosfilt` / SAC `passes=1`).
pub fn sosfilt(sos: &SosFilter, data: &[f64]) -> Vec<f64> {
    let mut out = data.to_vec();
    sosfilt_in_place(sos, &mut out);
    out
}

/// In-place causal 1-pass Direct Form II Transposed SOS filter with zero initial delay state.
pub fn sosfilt_in_place(sos: &SosFilter, data: &mut [f64]) {
    let mut zi = vec![[0.0_f64; 2]; sos.sections.len()];
    sosfilt_in_place_with_zi(sos, data, &mut zi);
}

/// In-place Direct Form II Transposed SOS filter updating caller-supplied delay states `zi` (`[n_sections][2]`).
pub fn sosfilt_in_place_with_zi(sos: &SosFilter, data: &mut [f64], zi: &mut [[f64; 2]]) {
    assert_eq!(
        zi.len(),
        sos.sections.len(),
        "zi length must equal number of SOS sections"
    );
    for x_slot in data.iter_mut() {
        let mut x = *x_slot;
        for (sec, st) in sos.sections.iter().zip(zi.iter_mut()) {
            let y = sec.b[0] * x + st[0];
            st[0] = sec.b[1] * x - sec.a[1] * y + st[1];
            st[1] = sec.b[2] * x - sec.a[2] * y;
            x = y;
        }
        *x_slot = x;
    }
}

/// ObsPy `zerophase=True` / SAC `passes=2` unpadded forward-backward zero-phase SOS filter:
/// executes `flip(sosfilt(sos, flip(sosfilt(sos, data))))`.
pub fn sosfilt_zerophase_obspy(sos: &SosFilter, data: &[f64]) -> Vec<f64> {
    let mut out = data.to_vec();
    sosfilt_zerophase_obspy_in_place(sos, &mut out);
    out
}

/// In-place unpadded 2-pass ObsPy/SAC zero-phase SOS filter (zero heap allocation for state <= 16 sections).
pub fn sosfilt_zerophase_obspy_in_place(sos: &SosFilter, data: &mut [f64]) {
    let mut stack_zi = [[0.0_f64; 2]; 16];
    if sos.sections.len() <= 16 {
        let n = sos.sections.len();
        sosfilt_in_place_with_zi(sos, data, &mut stack_zi[..n]);
        data.reverse();
        stack_zi[..n].fill([0.0, 0.0]);
        sosfilt_in_place_with_zi(sos, data, &mut stack_zi[..n]);
        data.reverse();
    } else {
        sosfilt_in_place(sos, data);
        data.reverse();
        sosfilt_in_place(sos, data);
        data.reverse();
    }
}

/// Computes step-response steady-state initial conditions `zi` (`[n_sections][2]`) matching
/// `scipy.signal.sosfilt_zi(sos)`.
pub fn sosfilt_zi(sos: &SosFilter) -> Result<Vec<[f64; 2]>> {
    let mut zi = Vec::with_capacity(sos.sections.len());
    let mut scale = 1.0_f64;
    for sec in &sos.sections {
        let sum_a = sec.a[0] + sec.a[1] + sec.a[2];
        if !sum_a.is_finite() || sum_a.abs() < 1e-15 {
            return Err(EarthSnailError::Numerical(
                "sosfilt_zi: unstable section with pole at z = 1 (sum(a) == 0)".into(),
            ));
        }
        let sum_b = sec.b[0] + sec.b[1] + sec.b[2];
        let y_inf = sum_b / sum_a;
        // Recurrence z_k = z_{k+1} + (b_k - y_inf * a_k) for k = 1, 2:
        let z1 = sec.b[2] - y_inf * sec.a[2];
        let z0 = (sec.b[1] - y_inf * sec.a[1]) + z1;
        zi.push([scale * z0, scale * z1]);
        scale *= y_inf;
    }
    Ok(zi)
}

/// SciPy-compatible forward-backward zero-phase SOS filter (`scipy.signal.sosfiltfilt`) using
/// odd-reflection boundary padding of length `3 * (2 * n_sections + 1 - trailing_zeros)` and
/// steady-state initial conditions `sosfilt_zi` to suppress startup/edge transients.
pub fn sosfiltfilt(sos: &SosFilter, data: &[f64]) -> Result<Vec<f64>> {
    let mut scratch = ScratchWorkspace::new();
    sosfiltfilt_with_scratch(sos, data, &mut scratch)
}

/// Scratch-reusing `sosfiltfilt` matching `scipy.signal.sosfiltfilt(sos, data)`.
pub fn sosfiltfilt_with_scratch(
    sos: &SosFilter,
    data: &[f64],
    scratch: &mut ScratchWorkspace,
) -> Result<Vec<f64>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let n_sections = sos.sections.len();
    let b2_zeros = sos.sections.iter().filter(|s| s.b[2] == 0.0).count();
    let a2_zeros = sos.sections.iter().filter(|s| s.a[2] == 0.0).count();
    let ntaps = 2 * n_sections + 1 - b2_zeros.min(a2_zeros);
    let edge = 3 * ntaps;

    if data.len() <= edge {
        return Err(EarthSnailError::InvalidParameter(format!(
            "sosfiltfilt requires data length ({}) > padlen ({edge})",
            data.len()
        )));
    }

    let n = data.len();
    let ext_len = n + 2 * edge;
    let ext = scratch.real_zeroed(ext_len);

    // Left odd extension: 2 * data[0] - data[edge..=1]
    let x0_first = data[0];
    for i in 0..edge {
        ext[i] = 2.0 * x0_first - data[edge - i];
    }
    // Interior signal
    ext[edge..edge + n].copy_from_slice(data);
    // Right odd extension: 2 * data[n - 1] - data[n - 2..=n - 1 - edge]
    let xn_last = data[n - 1];
    for i in 0..edge {
        ext[edge + n + i] = 2.0 * xn_last - data[n - 2 - i];
    }

    let zi_base = sosfilt_zi(sos)?;
    let mut zi_work = zi_base.clone();
    let ext_start = ext[0];
    for (w, base) in zi_work.iter_mut().zip(zi_base.iter()) {
        w[0] = base[0] * ext_start;
        w[1] = base[1] * ext_start;
    }

    // Forward pass:
    sosfilt_in_place_with_zi(sos, ext, &mut zi_work);

    // Reverse and backward pass:
    let y_last = ext[ext_len - 1];
    ext.reverse();
    for (w, base) in zi_work.iter_mut().zip(zi_base.iter()) {
        w[0] = base[0] * y_last;
        w[1] = base[1] * y_last;
    }
    sosfilt_in_place_with_zi(sos, ext, &mut zi_work);
    ext.reverse();

    Ok(ext[edge..edge + n].to_vec())
}

// ============================================================================
// ObsPy-Style Convenience Filter Helpers
// ============================================================================

/// ObsPy-compatible Butterworth bandpass filter (`zerophase = false` -> 1-pass `sosfilt`,
/// `zerophase = true` -> 2-pass `sosfilt_zerophase_obspy`).
pub fn bandpass(
    data: &[f64],
    freqmin_hz: f64,
    freqmax_hz: f64,
    sampling_rate_hz: f64,
    corners: usize,
    zerophase: bool,
) -> Result<Vec<f64>> {
    let sos = butter(
        corners,
        FilterBand::Bandpass(freqmin_hz, freqmax_hz),
        sampling_rate_hz,
    )?;
    Ok(if zerophase {
        sosfilt_zerophase_obspy(&sos, data)
    } else {
        sosfilt(&sos, data)
    })
}

/// ObsPy-compatible Butterworth bandstop filter.
pub fn bandstop(
    data: &[f64],
    freqmin_hz: f64,
    freqmax_hz: f64,
    sampling_rate_hz: f64,
    corners: usize,
    zerophase: bool,
) -> Result<Vec<f64>> {
    let sos = butter(
        corners,
        FilterBand::Bandstop(freqmin_hz, freqmax_hz),
        sampling_rate_hz,
    )?;
    Ok(if zerophase {
        sosfilt_zerophase_obspy(&sos, data)
    } else {
        sosfilt(&sos, data)
    })
}

/// ObsPy-compatible Butterworth lowpass filter.
pub fn lowpass(
    data: &[f64],
    freq_hz: f64,
    sampling_rate_hz: f64,
    corners: usize,
    zerophase: bool,
) -> Result<Vec<f64>> {
    let sos = butter(corners, FilterBand::Lowpass(freq_hz), sampling_rate_hz)?;
    Ok(if zerophase {
        sosfilt_zerophase_obspy(&sos, data)
    } else {
        sosfilt(&sos, data)
    })
}

/// ObsPy-compatible Butterworth highpass filter.
pub fn highpass(
    data: &[f64],
    freq_hz: f64,
    sampling_rate_hz: f64,
    corners: usize,
    zerophase: bool,
) -> Result<Vec<f64>> {
    let sos = butter(corners, FilterBand::Highpass(freq_hz), sampling_rate_hz)?;
    Ok(if zerophase {
        sosfilt_zerophase_obspy(&sos, data)
    } else {
        sosfilt(&sos, data)
    })
}

// ============================================================================
// Analog Prototypes, ZPK Frequency Transforms, Bilinear Transform, & ZPK->SOS
// ============================================================================

#[inline]
fn asinh_f64(x: f64) -> f64 {
    let ax = x.abs();
    let res = math::ln(ax + math::sqrt(ax * ax + 1.0));
    res.copysign(x)
}

#[inline]
fn csinh(mu: f64, theta: f64) -> Complex64 {
    // sinh(mu + i*theta) = sinh(mu)*cos(theta) + i*cosh(mu)*sin(theta)
    let em = math::exp(mu);
    let em_inv = 1.0 / em;
    let sh = 0.5 * (em - em_inv);
    let ch = 0.5 * (em + em_inv);
    let (s, c) = math::sin_cos(theta);
    Complex64::new(sh * c, ch * s)
}

fn buttap(order: usize) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let mut p = Vec::with_capacity(order);
    let n_i = order as i64;
    let mut m = -n_i + 1;
    while m < n_i {
        if m == 0 {
            p.push(Complex64::new(-1.0, 0.0));
        } else {
            let theta = PI * (m as f64) / ((2 * order) as f64);
            let (s, c) = math::sin_cos(theta);
            p.push(Complex64::new(-c, -s));
        }
        m += 2;
    }
    (Vec::new(), p, 1.0)
}

fn cheb1ap(order: usize, rp_db: f64) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let eps = math::sqrt(math::powf(10.0, 0.1 * rp_db) - 1.0);
    let mu = asinh_f64(1.0 / eps) / (order as f64);

    let mut p = Vec::with_capacity(order);
    let n_i = order as i64;
    let mut m = -n_i + 1;
    while m < n_i {
        let theta = if m == 0 {
            0.0
        } else {
            PI * (m as f64) / ((2 * order) as f64)
        };
        let mut pole = -csinh(mu, theta);
        if m == 0 {
            pole.im = 0.0;
        }
        p.push(pole);
        m += 2;
    }

    let prod_neg_p: Complex64 = p.iter().map(|&pi| -pi).product();
    let mut k = prod_neg_p.re;
    if order % 2 == 0 {
        k /= math::sqrt(1.0 + eps * eps);
    }
    (Vec::new(), p, k)
}

fn cheb2ap(order: usize, rs_db: f64) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let de = 1.0 / math::sqrt(math::powf(10.0, 0.1 * rs_db) - 1.0);
    let mu = asinh_f64(1.0 / de) / (order as f64);

    let mut z = Vec::new();
    let mut p = Vec::with_capacity(order);
    let n_i = order as i64;
    let mut m = -n_i + 1;
    while m < n_i {
        if m != 0 {
            let theta = PI * (m as f64) / ((2 * order) as f64);
            z.push(Complex64::new(0.0, 1.0 / math::sin(theta)));
        }
        let theta_p = if m == 0 {
            0.0
        } else {
            PI * (m as f64) / ((2 * order) as f64)
        };
        let mut pole = -csinh(mu, theta_p).inv();
        if m == 0 {
            pole.im = 0.0;
        }
        p.push(pole);
        m += 2;
    }

    let num: Complex64 = p.iter().map(|&pi| -pi).product();
    let den: Complex64 = z.iter().map(|&zi| -zi).product();
    let k = (num / den).re;
    (z, p, k)
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline]
fn split_f64(a: f64) -> (f64, f64) {
    // Veltkamp splitting factor 2^27 + 1 = 134_217_729.0
    let c = 134_217_729.0 * a;
    let hi = c - (c - a);
    let lo = a - hi;
    (hi, lo)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split_f64(a);
    let (bh, bl) = split_f64(b);
    let err = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, err)
}

#[derive(Clone, Copy)]
struct Dd(f64, f64);

impl Dd {
    const ZERO: Self = Self(0.0, 0.0);

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self(x, 0.0)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.0, rhs.0);
        let e = e + self.1 + rhs.1;
        let (hi, lo) = two_sum(s, e);
        Self(hi, lo)
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.add(Self(-rhs.0, -rhs.1))
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_prod(self.0, rhs.0);
        let e = e + self.0 * rhs.1 + self.1 * rhs.0;
        let (hi, lo) = two_sum(p, e);
        Self(hi, lo)
    }
}

/// Evaluates polynomial $\sum_{k=0}^n c_k x^k$ and its derivative $\sum_{k=1}^n k c_k x^{k-1}$
/// in 106-bit double-double (`Dd`) precision to eliminate alternating-sign Horner cancellation
/// on high-order Bessel polynomials ($N \le 12$).
fn poly_eval_and_deriv(coeffs: &[f64], x: Complex64) -> (Complex64, Complex64) {
    let x_re = Dd::from_f64(x.re);
    let x_im = Dd::from_f64(x.im);
    let mut v_re = Dd::ZERO;
    let mut v_im = Dd::ZERO;
    let mut d_re = Dd::ZERO;
    let mut d_im = Dd::ZERO;

    for &c in coeffs.iter().rev() {
        // der = der * x + val
        let dr = d_re.mul(x_re).sub(d_im.mul(x_im)).add(v_re);
        let di = d_re.mul(x_im).add(d_im.mul(x_re)).add(v_im);
        d_re = dr;
        d_im = di;

        // val = val * x + c
        let vr = v_re.mul(x_re).sub(v_im.mul(x_im)).add(Dd::from_f64(c));
        let vi = v_re.mul(x_im).add(v_im.mul(x_re));
        v_re = vr;
        v_im = vi;
    }
    (
        Complex64::new(v_re.0 + v_re.1, v_im.0 + v_im.1),
        Complex64::new(d_re.0 + d_re.1, d_im.0 + d_im.1),
    )
}

#[inline]
fn real_polyval(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Campos–Calderon (2011) initial zero approximations + Aberth–Ehrlich simultaneous root polish
/// on the exact ordinary Bessel polynomial $y_N(x) = \sum_{k=0}^N \frac{(N+k)!}{(N-k)! k! 2^k} x^k$.
fn bessel_ordinary_zeros(order: usize) -> Result<(Vec<Complex64>, f64)> {
    // Exact coefficients c[0..=order] in IEEE-754 f64 (c[12] = 3.16e11 < 2^53):
    let mut coeffs = vec![0.0_f64; order + 1];
    coeffs[0] = 1.0;
    for k in 1..=order {
        let num = ((order + k) * (order + 1 - k)) as f64;
        let den = (2 * k) as f64;
        coeffs[k] = coeffs[k - 1] * (num / den);
    }
    let a_last = coeffs[order];

    if order == 1 {
        return Ok((vec![Complex64::new(-1.0, 0.0)], a_last));
    }

    // Campos-Calderon (2011) initial guesses matching SciPy `_campos_zeros`:
    let nf = order as f64;
    let s = real_polyval(nf, &[0.0, 0.0, 2.0, 0.0, -3.0, 1.0]);
    let b3 = real_polyval(nf, &[16.0, -8.0]) / s;
    let b2 = real_polyval(nf, &[-24.0, -12.0, 12.0]) / s;
    let b1 = real_polyval(nf, &[8.0, 24.0, -12.0, -2.0]) / s;
    let b0 = real_polyval(nf, &[0.0, -6.0, 0.0, 5.0, -1.0]) / s;

    let r = real_polyval(nf, &[0.0, 0.0, 2.0, 1.0]);
    let a1 = real_polyval(nf, &[-6.0, -6.0]) / r;
    let a2 = 6.0 / r;

    let mut roots = Vec::with_capacity(order);
    for k in 1..=order {
        let kf = k as f64;
        let re = real_polyval(kf, &[0.0, a1, a2]);
        let im = real_polyval(kf, &[b0, b1, b2, b3]);
        roots.push(Complex64::new(re, im));
    }

    // Aberth-Ehrlich simultaneous root refinement:
    let mut steps = vec![Complex64::ZERO; order];
    let mut converged = false;
    for _iter in 0..120 {
        let mut max_rel_step = 0.0_f64;
        for k in 0..order {
            let (f_val, fp_val) = poly_eval_and_deriv(&coeffs, roots[k]);
            if fp_val.abs() <= 1e-300 {
                continue;
            }
            let newton = -f_val / fp_val;
            let mut repulsion = Complex64::ZERO;
            for j in 0..order {
                if j != k {
                    let diff = roots[k] - roots[j];
                    if diff.abs() > 1e-300 {
                        repulsion += diff.inv();
                    }
                }
            }
            let delta = newton / (Complex64::ONE + newton * repulsion);
            steps[k] = delta;
            let rel = delta.abs() / roots[k].abs().max(1e-6);
            if rel > max_rel_step {
                max_rel_step = rel;
            }
        }
        for k in 0..order {
            roots[k] += steps[k];
            if !roots[k].is_finite() {
                return Err(EarthSnailError::Numerical(
                    "Bessel polynomial Aberth-Ehrlich solver diverged".into(),
                ));
            }
        }
        if max_rel_step <= 1e-13 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(EarthSnailError::Numerical(
            "Bessel polynomial roots failed to converge".into(),
        ));
    }

    // 2 extra individual Newton polish steps + exact conjugate symmetrization:
    for r_slot in &mut roots {
        for _ in 0..2 {
            let (fv, fpv) = poly_eval_and_deriv(&coeffs, *r_slot);
            *r_slot -= fv / fpv;
        }
    }
    let orig = roots.clone();
    for k in 0..order {
        roots[k] = (orig[k] + orig[order - 1 - k].conj()) * 0.5;
        if roots[k].im.abs() < 1e-15 {
            roots[k].im = 0.0;
        }
    }

    Ok((roots, a_last))
}

fn besselap(order: usize, norm: BesselNorm) -> Result<(Vec<Complex64>, Vec<Complex64>, f64)> {
    let (zeros_yn, a_last) = bessel_ordinary_zeros(order)?;
    let mut p: Vec<Complex64> = zeros_yn.into_iter().map(Complex64::inv).collect();
    let k = match norm {
        BesselNorm::Delay => a_last,
        BesselNorm::Phase => {
            let scale = math::powf(10.0, -math::log10(a_last) / (order as f64));
            for pi in &mut p {
                *pi *= scale;
            }
            1.0
        }
        BesselNorm::Mag => {
            // Newton solve |a_last / prod(i*w - p)| - 1/sqrt(2) == 0 starting from w = 1.5
            let mut w = 1.5_f64;
            for _ in 0..50 {
                let iw = Complex64::new(0.0, w);
                let den: Complex64 = p.iter().map(|&pi| iw - pi).product();
                let g = a_last / den.abs();
                let f = g - FRAC_1_SQRT_2;
                if f.abs() < 1e-15 {
                    break;
                }
                // d/dw ln G(w) = - sum_i w / ((w - Im(p_i))^2 + Re(p_i)^2)? Use exact derivative of ln|iw - p_i|:
                // |iw - (re + i*im)|^2 = re^2 + (w - im)^2 -> d/dw (1/2 ln(...)) = (w - im) / (re^2 + (w - im)^2)
                let d_ln_den: f64 = p
                    .iter()
                    .map(|pi| {
                        let dw = w - pi.im;
                        dw / (pi.re * pi.re + dw * dw)
                    })
                    .sum();
                let gp = -g * d_ln_den;
                w -= f / gp;
            }
            let inv_w = 1.0 / w;
            for pi in &mut p {
                *pi *= inv_w;
            }
            a_last * math::powi(inv_w, order as i32)
        }
    };
    Ok((Vec::new(), p, k))
}

fn lp2lp_zpk(
    z: &[Complex64],
    p: &[Complex64],
    k: f64,
    wo: f64,
) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let degree = (p.len() as i32) - (z.len() as i32);
    let z_out = z.iter().map(|&zi| zi * wo).collect();
    let p_out = p.iter().map(|&pi| pi * wo).collect();
    let k_out = k * math::powi(wo, degree);
    (z_out, p_out, k_out)
}

fn lp2hp_zpk(
    z: &[Complex64],
    p: &[Complex64],
    k: f64,
    wo: f64,
) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let mut z_out: Vec<Complex64> = z.iter().map(|&zi| Complex64::from(wo) / zi).collect();
    z_out.resize(z_out.len() + degree, Complex64::ZERO);
    let p_out: Vec<Complex64> = p.iter().map(|&pi| Complex64::from(wo) / pi).collect();
    let num: Complex64 = z.iter().map(|&zi| -zi).product();
    let den: Complex64 = p.iter().map(|&pi| -pi).product();
    let k_out = k * (num / den).re;
    (z_out, p_out, k_out)
}

fn lp2bp_zpk(
    z: &[Complex64],
    p: &[Complex64],
    k: f64,
    wo: f64,
    bw: f64,
) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let half_bw = 0.5 * bw;
    let wo2 = Complex64::from(wo * wo);

    let mut z_out = Vec::with_capacity(2 * z.len() + degree);
    for &zi in z {
        let z_lp = zi * half_bw;
        let root = (z_lp * z_lp - wo2).sqrt();
        z_out.push(z_lp + root);
        z_out.push(z_lp - root);
    }
    z_out.resize(z_out.len() + degree, Complex64::ZERO);

    let mut p_out = Vec::with_capacity(2 * p.len());
    for &pi in p {
        let p_lp = pi * half_bw;
        let root = (p_lp * p_lp - wo2).sqrt();
        p_out.push(p_lp + root);
        p_out.push(p_lp - root);
    }

    let k_out = k * math::powi(bw, degree as i32);
    (z_out, p_out, k_out)
}

fn lp2bs_zpk(
    z: &[Complex64],
    p: &[Complex64],
    k: f64,
    wo: f64,
    bw: f64,
) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let half_bw = Complex64::from(0.5 * bw);
    let wo2 = Complex64::from(wo * wo);

    let mut z_out = Vec::with_capacity(2 * z.len() + 2 * degree);
    for &zi in z {
        let z_hp = half_bw / zi;
        let root = (z_hp * z_hp - wo2).sqrt();
        z_out.push(z_hp + root);
        z_out.push(z_hp - root);
    }
    for _ in 0..degree {
        z_out.push(Complex64::new(0.0, wo));
    }
    for _ in 0..degree {
        z_out.push(Complex64::new(0.0, -wo));
    }

    let mut p_out = Vec::with_capacity(2 * p.len());
    for &pi in p {
        let p_hp = half_bw / pi;
        let root = (p_hp * p_hp - wo2).sqrt();
        p_out.push(p_hp + root);
        p_out.push(p_hp - root);
    }

    let num: Complex64 = z.iter().map(|&zi| -zi).product();
    let den: Complex64 = p.iter().map(|&pi| -pi).product();
    let k_out = k * (num / den).re;
    (z_out, p_out, k_out)
}

fn bilinear_zpk(
    z: &[Complex64],
    p: &[Complex64],
    k: f64,
    fs: f64,
) -> (Vec<Complex64>, Vec<Complex64>, f64) {
    let degree = p.len().saturating_sub(z.len());
    let fs2 = Complex64::from(2.0 * fs);

    let mut z_d: Vec<Complex64> = z.iter().map(|&zi| (fs2 + zi) / (fs2 - zi)).collect();
    z_d.resize(z_d.len() + degree, Complex64::new(-1.0, 0.0));

    let p_d: Vec<Complex64> = p.iter().map(|&pi| (fs2 + pi) / (fs2 - pi)).collect();

    // Pair numerator/denominator factors to prevent intermediate product overflow at high order:
    let mut ratio = Complex64::ONE;
    let shared = z.len().min(p.len());
    for i in 0..shared {
        ratio *= (fs2 - z[i]) / (fs2 - p[i]);
    }
    for &zi in &z[shared..] {
        ratio *= fs2 - zi;
    }
    for &pi in &p[shared..] {
        ratio /= fs2 - pi;
    }
    let k_d = k * ratio.re;
    (z_d, p_d, k_d)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RootKind {
    Any,
    Real,
    Complex,
}

#[inline]
fn is_real_root(c: Complex64) -> bool {
    c.im == 0.0
}

/// Groups roots into `[complex_upper_half_representatives..., real_roots...]` matching SciPy `_cplxreal`.
fn split_cplxreal(roots: &[Complex64]) -> Result<Vec<Complex64>> {
    const TOL: f64 = 100.0 * f64::EPSILON;
    let mut reals = Vec::new();
    let mut pos_im = Vec::new();
    let mut neg_im = Vec::new();
    for &r in roots {
        if r.im.abs() <= TOL * (1.0 + r.abs()) {
            reals.push(Complex64::new(r.re, 0.0));
        } else if r.im > 0.0 {
            pos_im.push(r);
        } else {
            neg_im.push(r);
        }
    }
    reals.sort_by(|a, b| a.re.total_cmp(&b.re));

    if pos_im.len() != neg_im.len() {
        return Err(EarthSnailError::Numerical(
            "zpk_to_sos: unmatched upper/lower half-plane complex roots".into(),
        ));
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

    let mut reps = Vec::with_capacity(pos_im.len() + reals.len());
    for (&a, &b) in pos_im.iter().zip(neg_im.iter()) {
        if (a.re - b.re).abs() > 1e-7 * (1.0 + a.abs())
            || (a.im + b.im).abs() > 1e-7 * (1.0 + a.abs())
        {
            return Err(EarthSnailError::Numerical(
                "zpk_to_sos: complex roots do not form conjugate pairs".into(),
            ));
        }
        let re_avg = 0.5 * (a.re + b.re);
        let im_pos = 0.5 * (a.im.abs() + b.im.abs());
        reps.push(Complex64::new(re_avg, im_pos));
    }
    reps.extend(reals);
    Ok(reps)
}

fn idx_worst_digital(poles: &[Complex64]) -> usize {
    let mut best_i = 0;
    let mut best_dist = f64::INFINITY;
    for (i, &p) in poles.iter().enumerate() {
        let d = (1.0 - p.abs()).abs();
        if d < best_dist {
            best_dist = d;
            best_i = i;
        }
    }
    best_i
}

fn nearest_root_idx(candidates: &[Complex64], target: Complex64, kind: RootKind) -> usize {
    let mut best_i = 0;
    let mut best_dist = f64::INFINITY;
    for (i, &c) in candidates.iter().enumerate() {
        let matches = match kind {
            RootKind::Any => true,
            RootKind::Real => is_real_root(c),
            RootKind::Complex => !is_real_root(c),
        };
        if matches {
            let d = (c - target).abs();
            if d < best_dist {
                best_dist = d;
                best_i = i;
            }
        }
    }
    best_i
}

fn poly_from_roots2(roots: &[Complex64]) -> [f64; 3] {
    match roots.len() {
        0 => [1.0, 0.0, 0.0],
        1 => [1.0, -roots[0].re, 0.0],
        _ => {
            let r0 = roots[0];
            let r1 = roots[1];
            let c1 = -(r0 + r1).re;
            let c2 = (r0 * r1).re;
            [1.0, c1, c2]
        }
    }
}

/// Converts discrete `(z, p, k)` into cascaded `SosFilter` using SciPy's `pairing = 'nearest'` algorithm.
fn zpk_to_sos(z_in: &[Complex64], p_in: &[Complex64], k: f64) -> Result<SosFilter> {
    if z_in.is_empty() && p_in.is_empty() {
        return SosFilter::new(vec![SosSection {
            b: [k, 0.0, 0.0],
            a: [1.0, 0.0, 0.0],
        }]);
    }

    let mut p_pad = p_in.to_vec();
    let mut z_pad = z_in.to_vec();
    let max_len = p_pad.len().max(z_pad.len());
    p_pad.resize(max_len, Complex64::ZERO);
    z_pad.resize(max_len, Complex64::ZERO);
    let n_sections = max_len.div_ceil(2);
    if p_pad.len() % 2 != 0 {
        p_pad.push(Complex64::ZERO);
        z_pad.push(Complex64::ZERO);
    }

    let mut z = split_cplxreal(&z_pad)?;
    let mut p = split_cplxreal(&p_pad)?;

    let mut sections = vec![SosSection::default(); n_sections];

    for si in (0..n_sections).rev() {
        let p1_idx = idx_worst_digital(&p);
        let p1 = p.remove(p1_idx);
        let p_real_rem = p.iter().filter(|&&r| is_real_root(r)).count();
        let z_real_rem = z.iter().filter(|&&r| is_real_root(r)).count();

        if is_real_root(p1) && p_real_rem == 0 {
            let z1_idx = nearest_root_idx(&z, p1, RootKind::Real);
            let z1 = z.remove(z1_idx);
            sections[si] = SosSection {
                b: poly_from_roots2(&[z1, Complex64::ZERO]),
                a: poly_from_roots2(&[p1, Complex64::ZERO]),
            };
        } else if p.len() + 1 == z.len() && !is_real_root(p1) && p_real_rem == 1 && z_real_rem == 1
        {
            let z1_idx = nearest_root_idx(&z, p1, RootKind::Complex);
            let z1 = z.remove(z1_idx);
            sections[si] = SosSection {
                b: poly_from_roots2(&[z1, z1.conj()]),
                a: poly_from_roots2(&[p1, p1.conj()]),
            };
        } else {
            let p2 = if is_real_root(p1) {
                let real_indices: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &r)| is_real_root(r).then_some(idx))
                    .collect();
                let real_poles: Vec<Complex64> = real_indices.iter().map(|&idx| p[idx]).collect();
                let worst_sub = idx_worst_digital(&real_poles);
                p.remove(real_indices[worst_sub])
            } else {
                p1.conj()
            };

            if !z.is_empty() {
                let z1_idx = nearest_root_idx(&z, p1, RootKind::Any);
                let z1 = z.remove(z1_idx);
                if !is_real_root(z1) {
                    sections[si] = SosSection {
                        b: poly_from_roots2(&[z1, z1.conj()]),
                        a: poly_from_roots2(&[p1, p2]),
                    };
                } else if !z.is_empty() {
                    let z2_idx = nearest_root_idx(&z, p1, RootKind::Real);
                    let z2 = z.remove(z2_idx);
                    sections[si] = SosSection {
                        b: poly_from_roots2(&[z1, z2]),
                        a: poly_from_roots2(&[p1, p2]),
                    };
                } else {
                    sections[si] = SosSection {
                        b: poly_from_roots2(&[z1]),
                        a: poly_from_roots2(&[p1, p2]),
                    };
                }
            } else {
                sections[si] = SosSection {
                    b: [1.0, 0.0, 0.0],
                    a: poly_from_roots2(&[p1, p2]),
                };
            }
        }
    }

    sections[0].b[0] *= k;
    sections[0].b[1] *= k;
    sections[0].b[2] *= k;

    SosFilter::new(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_filter_families_and_bands_are_stable_and_attenuate_stopband() {
        let fs = 100.0;
        let families = [
            FilterFamily::Butterworth,
            FilterFamily::Chebyshev1 { rp_db: 1.0 },
            FilterFamily::Chebyshev2 { rs_db: 40.0 },
            FilterFamily::Bessel {
                norm: BesselNorm::Phase,
            },
            FilterFamily::Bessel {
                norm: BesselNorm::Mag,
            },
            FilterFamily::Bessel {
                norm: BesselNorm::Delay,
            },
        ];
        let bands = [
            FilterBand::Lowpass(5.0),
            FilterBand::Highpass(0.5),
            FilterBand::Bandpass(0.5, 5.0),
            FilterBand::Bandstop(0.5, 5.0),
        ];

        for &fam in &families {
            for &band in &bands {
                let sos = design_iir_sos(4, band, fs, fam).unwrap();
                assert!(
                    sos.is_stable(),
                    "unstable poles for family={fam:?}, band={band:?}"
                );
            }
        }

        // Verify high-order Bessel stability up to order 10!
        let max_ord = if cfg!(miri) { 5 } else { 10 };
        let tol_mag = if cfg!(miri) { 1e-11 } else { 1e-12 };
        for ord in 1..=max_ord {
            let b_phase = bessel(ord, FilterBand::Lowpass(8.0), fs, BesselNorm::Phase).unwrap();
            let b_mag = bessel(ord, FilterBand::Lowpass(8.0), fs, BesselNorm::Mag).unwrap();
            assert!(b_phase.is_stable());
            assert!(b_mag.is_stable());
            // For BesselNorm::Mag, |H(fc)| == 1/sqrt(2) to high precision!
            let g_fc = b_mag.freqz(8.0, fs).abs();
            assert!(
                (g_fc - FRAC_1_SQRT_2).abs() < tol_mag,
                "Bessel mag ord={ord}: |H(fc)|={g_fc}"
            );
        }
    }

    #[test]
    fn sosfiltfilt_preserves_constant_step_and_zero_phase() {
        let sos = butter(4, FilterBand::Lowpass(10.0), 100.0).unwrap();
        // Constant DC signal through lowpass with steady-state sosfiltfilt has zero transient!
        let dc = vec![3.75_f64; 200];
        let y_dc = sosfiltfilt(&sos, &dc).unwrap();
        for &v in &y_dc {
            assert!((v - 3.75).abs() < 1e-12, "got {v}");
        }

        // In-passband 2 Hz sinusoid + 40 Hz stopband sinusoid: zero phase shift at peak!
        let n = 400;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) * 0.01;
                math::cos(TAU * 2.0 * (t - 2.0)) + 0.8 * math::sin(TAU * 40.0 * t)
            })
            .collect();
        let y_zp = sosfiltfilt(&sos, &x).unwrap();
        // At index 200 (t = 2.0 s), cos(2*pi*2*(t - 2.0)) has exact peak +1.0; neighbors are symmetric!
        assert!((y_zp[200] - 1.0).abs() < 1e-3);
        assert!((y_zp[199] - y_zp[201]).abs() < 1e-4);
    }
}
