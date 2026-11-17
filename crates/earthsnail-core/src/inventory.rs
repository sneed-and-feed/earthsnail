//! Station and channel metadata (`StationInventory`, `ChannelMetadata`) and multi-stage
//! instrument frequency-response evaluation (`ChannelResponse`, `PolesZerosStage`, `FirStage`).

pub use crate::complex::Complex64;

use crate::math;
use crate::time::UtcTimestamp;
use crate::trace::ChannelId;
use alloc::string::String;
use alloc::vec::Vec;

/// Physical ground-motion unit of an instrument response input or `remove_response` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KinematicUnit {
    /// Ground displacement (`m`, `"DISP"`).
    Displacement,
    /// Ground velocity (`m/s`, `"VEL"` — standard FDSN broadband StationXML input).
    #[default]
    Velocity,
    /// Ground acceleration (`m/s^2`, `"ACC"`).
    Acceleration,
    /// Non-kinematic physical unit (e.g. pressure `Pa`, temperature `K`, volts `V`).
    Other,
}

impl KinematicUnit {
    /// Parses FDSN / StationXML / ObsPy unit strings (`"M"`, `"DISP"`, `"M/S"`, `"VEL"`, `"M/S**2"`, `"ACC"`).
    pub fn from_unit_str(s: &str) -> Self {
        let u = s.trim();
        if u.eq_ignore_ascii_case("M")
            || u.eq_ignore_ascii_case("DISP")
            || u.eq_ignore_ascii_case("DIS")
        {
            Self::Displacement
        } else if u.eq_ignore_ascii_case("M/S")
            || u.eq_ignore_ascii_case("VEL")
            || u.eq_ignore_ascii_case("M/SEC")
        {
            Self::Velocity
        } else if u.eq_ignore_ascii_case("M/S**2")
            || u.eq_ignore_ascii_case("M/S^2")
            || u.eq_ignore_ascii_case("ACC")
        {
            Self::Acceleration
        } else {
            Self::Other
        }
    }

    /// Integer derivative order relative to displacement (`DISP = 0`, `VEL = 1`, `ACC = 2`).
    pub const fn derivative_order(self) -> Option<i32> {
        match self {
            Self::Displacement => Some(0),
            Self::Velocity => Some(1),
            Self::Acceleration => Some(2),
            Self::Other => None,
        }
    }
}

/// Transfer-function domain convention for a pole-zero stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PzTransferFunctionType {
    /// Analog Laplace transform in radians/second ($s = i 2\pi f$, SEED Type A & SACPZ).
    #[default]
    RadPerSec,
    /// Analog Laplace transform in Cycles/Hertz ($s = i f$, SEED Type B).
    Hz,
    /// Digital Z-transform ($z = e^{i 2\pi f / f_s}$, SEED Type D).
    DigitalZTransform,
}

/// Gain scalar evaluated at a reference normalization frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageGain {
    /// Scalar stage or overall sensitivity factor.
    pub value: f64,
    /// Reference frequency in Hz at which `value` was measured.
    pub frequency_hz: f64,
}

impl Default for StageGain {
    fn default() -> Self {
        Self {
            value: 1.0,
            frequency_hz: 1.0,
        }
    }
}

/// Analog Laplace or digital Z-transform pole-zero filter stage (`StationXML` `PolesZeros` or `SACPZ`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolesZerosStage {
    /// Domain of poles and zeros (`RadPerSec`, `Hz`, or `DigitalZTransform`).
    pub pz_transfer_function_type: PzTransferFunctionType,
    /// Normalization factor $A_0$ (or `CONSTANT = A0 * Sd` for single-stage `SACPZ` when `stage_gain == 1.0`).
    pub a0: f64,
    /// Normalization frequency $f_0$ in Hz.
    pub normalization_freq: f64,
    /// Complex zeros $z_k$.
    pub zeros: Vec<Complex64>,
    /// Complex poles $p_m$.
    pub poles: Vec<Complex64>,
    /// Stage gain $G_j$ (`1.0` when already absorbed into `a0`, as in `SACPZ`).
    pub stage_gain: f64,
    /// Input sampling rate in Hz (used only when `pz_transfer_function_type == DigitalZTransform`).
    pub input_sample_rate_hz: Option<f64>,
}

impl Default for PolesZerosStage {
    fn default() -> Self {
        Self {
            pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
            a0: 1.0,
            normalization_freq: 1.0,
            zeros: Vec::new(),
            poles: Vec::new(),
            stage_gain: 1.0,
            input_sample_rate_hz: None,
        }
    }
}

impl PolesZerosStage {
    /// Complex Laplace/Z variable $s(f)$ for `freq_hz`.
    #[inline]
    pub fn complex_var(&self, freq_hz: f64) -> Complex64 {
        match self.pz_transfer_function_type {
            PzTransferFunctionType::RadPerSec => {
                Complex64::new(0.0, core::f64::consts::TAU * freq_hz)
            }
            PzTransferFunctionType::Hz => Complex64::new(0.0, freq_hz),
            PzTransferFunctionType::DigitalZTransform => {
                let fs = self.input_sample_rate_hz.unwrap_or(1.0).max(1e-30);
                Complex64::cis(core::f64::consts::TAU * freq_hz / fs)
            }
        }
    }

    /// Recomputes `self.a0` so that $|\prod(s_0 - z_k) / \prod(s_0 - p_m)| \cdot A_0 = 1$ at `normalization_freq`.
    pub fn recompute_a0(&mut self) {
        let s0 = self.complex_var(self.normalization_freq);
        let mut ratio = Complex64::ONE;
        let shared = self.zeros.len().min(self.poles.len());
        for i in 0..shared {
            ratio *= (s0 - self.zeros[i]) / (s0 - self.poles[i]);
        }
        for z in &self.zeros[shared..] {
            ratio *= s0 - *z;
        }
        for p in &self.poles[shared..] {
            ratio /= s0 - *p;
        }
        let mag = ratio.abs();
        if mag > 0.0 && mag.is_finite() {
            self.a0 = 1.0 / mag;
        }
    }

    /// Evaluates `stage_gain * a0 * prod(s - z_k) / prod(s - p_m)` at `freq_hz` with paired products to prevent overflow.
    pub fn eval_freq(&self, freq_hz: f64) -> Complex64 {
        let s = self.complex_var(freq_hz);
        let mut acc = Complex64::from(self.a0 * self.stage_gain);
        let shared = self.zeros.len().min(self.poles.len());
        for i in 0..shared {
            let denom = s - self.poles[i];
            if denom.re == 0.0 && denom.im == 0.0 {
                return Complex64::ZERO;
            }
            acc *= (s - self.zeros[i]) / denom;
        }
        for z in &self.zeros[shared..] {
            acc *= s - *z;
        }
        for p in &self.poles[shared..] {
            let denom = s - *p;
            if denom.re == 0.0 && denom.im == 0.0 {
                return Complex64::ZERO;
            }
            acc /= denom;
        }
        acc
    }
}

/// Symmetry mode for a `StationXML` `FIR` coefficient list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FirSymmetry {
    /// Complete list of $M$ arbitrary numerator taps (`NONE`).
    #[default]
    None,
    /// Odd-length symmetric filter: `coefficients` stores $L+1$ taps `[c_0, ..., c_L]` expanding to length $2L+1$ (`ODD`).
    Odd,
    /// Even-length symmetric filter: `coefficients` stores $L$ taps `[c_0, ..., c_{L-1}]` expanding to length $2L$ (`EVEN`).
    Even,
}

/// Digital Finite Impulse Response (`FIR`) filter stage with decimation & delay-correction metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct FirStage {
    /// Coefficient symmetry specification (`None`, `Odd`, `Even`).
    pub symmetry: FirSymmetry,
    /// Stored numerator taps (half-list if `Odd`/`Even`, full list if `None`).
    pub coefficients: Vec<f64>,
    /// Stage input sampling rate in Hz (`> 0.0`).
    pub input_sample_rate_hz: f64,
    /// Integer decimation factor (`>= 1`).
    pub decimation_factor: u32,
    /// Estimated filter delay in seconds (`StationXML` `Decimation/Delay`).
    pub delay_s: f64,
    /// Applied time correction subtracted by digitizer in seconds (`StationXML` `Decimation/Correction`).
    pub correction_s: f64,
    /// Multiplicative stage gain (`1.0` default).
    pub stage_gain: f64,
    /// Reference frequency in Hz at which `stage_gain` is defined (`0.0` default).
    pub gain_frequency_hz: f64,
}

impl Default for FirStage {
    fn default() -> Self {
        Self {
            symmetry: FirSymmetry::None,
            coefficients: Vec::new(),
            input_sample_rate_hz: 1.0,
            decimation_factor: 1,
            delay_s: 0.0,
            correction_s: 0.0,
            stage_gain: 1.0,
            gain_frequency_hz: 0.0,
        }
    }
}

impl FirStage {
    /// Expands `self.coefficients` according to `self.symmetry` into the full tap sequence $h[0..M]$.
    pub fn expanded_taps(&self) -> Vec<f64> {
        match self.symmetry {
            FirSymmetry::None => self.coefficients.clone(),
            FirSymmetry::Odd => {
                if self.coefficients.is_empty() {
                    return Vec::new();
                }
                let l = self.coefficients.len() - 1;
                let mut taps = Vec::with_capacity(2 * l + 1);
                taps.extend_from_slice(&self.coefficients);
                for i in (0..l).rev() {
                    taps.push(self.coefficients[i]);
                }
                taps
            }
            FirSymmetry::Even => {
                let l = self.coefficients.len();
                let mut taps = Vec::with_capacity(2 * l);
                taps.extend_from_slice(&self.coefficients);
                for i in (0..l).rev() {
                    taps.push(self.coefficients[i]);
                }
                taps
            }
        }
    }

    fn eval_raw(&self, freq_hz: f64) -> Complex64 {
        let omega = core::f64::consts::TAU * freq_hz;
        let dt = 1.0 / self.input_sample_rate_hz;
        match self.symmetry {
            FirSymmetry::None => self
                .coefficients
                .iter()
                .enumerate()
                .map(|(k, &c)| Complex64::from_polar(c, -omega * (k as f64) * dt))
                .sum(),
            FirSymmetry::Odd => {
                let l = self.coefficients.len() - 1;
                let center_phase = -omega * (l as f64) * dt;
                let mut real_amp = self.coefficients[l];
                for (k, &coeff) in self.coefficients[..l].iter().enumerate() {
                    let dist = (l - k) as f64;
                    real_amp += 2.0 * coeff * math::cos(omega * dist * dt);
                }
                Complex64::from_polar(real_amp, center_phase)
            }
            FirSymmetry::Even => {
                let l = self.coefficients.len();
                let center = ((2 * l - 1) as f64) * 0.5;
                let center_phase = -omega * center * dt;
                let mut real_amp = 0.0;
                for (k, &coeff) in self.coefficients[..l].iter().enumerate() {
                    let dist = center - (k as f64);
                    real_amp += 2.0 * coeff * math::cos(omega * dist * dt);
                }
                Complex64::from_polar(real_amp, center_phase)
            }
        }
    }

    /// Evaluates the digital FIR transfer function $G \cdot e^{+i\omega\,\text{correction}} \frac{H_{\text{raw}}(f)}{|H_{\text{raw}}(f_{\text{gain}})|}$.
    pub fn eval_freq(&self, freq_hz: f64) -> Complex64 {
        if self.coefficients.is_empty() || self.input_sample_rate_hz <= 0.0 {
            return Complex64::from(self.stage_gain);
        }
        let raw = self.eval_raw(freq_hz);
        let norm = self.eval_raw(self.gain_frequency_hz).abs();
        let normalized = if norm > 0.0 && norm.is_finite() {
            raw / norm
        } else {
            raw
        };
        let omega = core::f64::consts::TAU * freq_hz;
        let phase_corr = if self.correction_s != 0.0 {
            Complex64::cis(omega * self.correction_s)
        } else {
            Complex64::ONE
        };
        normalized * phase_corr * self.stage_gain
    }
}

/// Rational IIR/FIR `StationXML` `Coefficients` stage (`DIGITAL` numerator/denominator taps).
#[derive(Debug, Clone, PartialEq)]
pub struct CoefficientsStage {
    /// Feedforward numerator taps $b_k$.
    pub numerator: Vec<f64>,
    /// Feedback denominator taps $a_m$ (empty or `[1.0]` for pure FIR).
    pub denominator: Vec<f64>,
    /// Stage input sampling rate in Hz.
    pub input_sample_rate_hz: f64,
    /// Applied time correction in seconds.
    pub correction_s: f64,
    /// Multiplicative stage gain.
    pub stage_gain: f64,
    /// Reference frequency in Hz at which `stage_gain` is defined (`0.0` default; `evalresp` normalizes
    /// digital taps so `|H_raw(gain_frequency_hz)| == 1.0`).
    pub gain_frequency_hz: f64,
}

impl Default for CoefficientsStage {
    fn default() -> Self {
        Self {
            numerator: Vec::new(),
            denominator: Vec::new(),
            input_sample_rate_hz: 1.0,
            correction_s: 0.0,
            stage_gain: 1.0,
            gain_frequency_hz: 0.0,
        }
    }
}

impl CoefficientsStage {
    fn eval_raw(&self, freq_hz: f64) -> Complex64 {
        let omega = core::f64::consts::TAU * freq_hz;
        let dt = 1.0 / self.input_sample_rate_hz;
        let num: Complex64 = if self.numerator.is_empty() {
            Complex64::ONE
        } else {
            self.numerator
                .iter()
                .enumerate()
                .map(|(k, &b)| Complex64::from_polar(b, -omega * (k as f64) * dt))
                .sum()
        };
        let den: Complex64 = if self.denominator.is_empty() {
            Complex64::ONE
        } else {
            self.denominator
                .iter()
                .enumerate()
                .map(|(m, &a)| Complex64::from_polar(a, -omega * (m as f64) * dt))
                .sum()
        };
        num / den
    }

    /// Evaluates rational Z-domain response at `freq_hz`, normalizing by `|H_raw(gain_frequency_hz)|`
    /// matching FDSN `evalresp` convention.
    pub fn eval_freq(&self, freq_hz: f64) -> Complex64 {
        if self.input_sample_rate_hz <= 0.0 {
            return Complex64::from(self.stage_gain);
        }
        if self.numerator.is_empty() && self.denominator.is_empty() {
            return Complex64::from(self.stage_gain);
        }
        let raw = self.eval_raw(freq_hz);
        let norm = self.eval_raw(self.gain_frequency_hz).abs();
        let normalized = if norm > 0.0 && norm.is_finite() {
            raw / norm
        } else {
            raw
        };
        let omega = core::f64::consts::TAU * freq_hz;
        let corr = if self.correction_s != 0.0 {
            Complex64::cis(omega * self.correction_s)
        } else {
            Complex64::ONE
        };
        normalized * corr * self.stage_gain
    }
}

/// Polynomial response stage (used in environmental/pressure/infrasound sensors; linear term defines small-signal gain).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PolynomialStage {
    /// Maclaurin polynomial coefficients $[c_0, c_1, c_2, \dots]$ mapping output to input physical units.
    pub coefficients: Vec<f64>,
}

/// Single stage inside a multi-stage `ChannelResponse`.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseStage {
    /// Analog or digital pole-zero stage.
    PolesZeros(PolesZerosStage),
    /// Symmetric or asymmetric digital FIR stage.
    Fir(FirStage),
    /// Digital rational `Coefficients` stage.
    Coefficients(CoefficientsStage),
    /// Scalar sensitivity/preamp gain stage.
    GainOnly(StageGain),
    /// Polynomial calibration stage.
    Polynomial(PolynomialStage),
}

impl ResponseStage {
    /// Evaluates the complex transfer function of this single stage at `freq_hz`.
    pub fn eval_freq(&self, freq_hz: f64) -> Complex64 {
        match self {
            Self::PolesZeros(pz) => pz.eval_freq(freq_hz),
            Self::Fir(fir) => fir.eval_freq(freq_hz),
            Self::Coefficients(c) => c.eval_freq(freq_hz),
            Self::GainOnly(g) => Complex64::from(g.value),
            Self::Polynomial(poly) => {
                // Small-signal linear sensitivity from input -> counts is 1 / c_1 when c_1 != 0.
                let c1 = poly.coefficients.get(1).copied().unwrap_or(1.0);
                if c1 != 0.0 {
                    Complex64::from(1.0 / c1)
                } else {
                    Complex64::ONE
                }
            }
        }
    }
}

/// Complete instrument response cascade for one seismic channel (`StationXML` `Response` or `SACPZ`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChannelResponse {
    /// Physical ground-motion unit entering stage 1 (`Velocity` for standard StationXML, `Displacement` for `SACPZ`).
    pub input_units: KinematicUnit,
    /// Output unit label (typically `"COUNTS"`).
    pub output_units: String,
    /// Optional overall `InstrumentSensitivity` declared in `StationXML`.
    pub instrument_sensitivity: Option<StageGain>,
    /// Ordered cascade of response stages (`stage 1..N`).
    pub stages: Vec<ResponseStage>,
}

impl ChannelResponse {
    /// Constructs a single-stage `ChannelResponse` from a `SACPZ` (`ZEROS`, `POLES` in `RadPerSec`,
    /// `CONSTANT`, native input `KinematicUnit::Displacement`).
    pub fn from_sacpz(zeros: Vec<Complex64>, poles: Vec<Complex64>, constant: f64) -> Self {
        Self {
            input_units: KinematicUnit::Displacement,
            output_units: String::from("COUNTS"),
            instrument_sensitivity: None,
            stages: alloc::vec![ResponseStage::PolesZeros(PolesZerosStage {
                pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
                a0: constant,
                normalization_freq: 1.0,
                zeros,
                poles,
                stage_gain: 1.0,
                input_sample_rate_hz: None,
            })],
        }
    }

    /// Evaluates the cascaded complex frequency response $H(f) = \prod_j H_j(f)$ in the channel's
    /// native `input_units` at `freq_hz` Hz.
    pub fn eval_freq(&self, freq_hz: f64) -> Complex64 {
        if self.stages.is_empty() {
            return match self.instrument_sensitivity {
                Some(sens) => Complex64::from(sens.value),
                None => Complex64::ONE,
            };
        }
        self.stages.iter().map(|st| st.eval_freq(freq_hz)).product()
    }

    /// Evaluates the complex frequency response converted to `target` kinematic ground-motion units
    /// (`Displacement`, `Velocity`, or `Acceleration`) using $V(\omega) = i\omega\,D(\omega)$:
    ///
    /// - Converting native `Velocity` (`Counts / (m/s)`) -> target `Displacement` (`Counts / m`) multiplies $H(f)$ by $+i 2\pi f$.
    /// - Converting native `Displacement` (`SACPZ`, `Counts / m`) -> target `Velocity` (`Counts / (m/s)`) divides $H(f)$ by $+i 2\pi f$.
    pub fn eval_freq_kinematic(&self, freq_hz: f64, target: KinematicUnit) -> Complex64 {
        let h_native = self.eval_freq(freq_hz);
        let (Some(ord_in), Some(ord_target)) = (
            self.input_units.derivative_order(),
            target.derivative_order(),
        ) else {
            return h_native;
        };
        let diff = ord_in - ord_target;
        if diff == 0 {
            return h_native;
        }
        if freq_hz == 0.0 && diff < 0 {
            return Complex64::ZERO;
        }
        let iw = Complex64::new(0.0, core::f64::consts::TAU * freq_hz);
        h_native * iw.powi(diff)
    }
}

/// Complete geographic, orientation, epoch, and response metadata for a single channel epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelMetadata {
    /// 4-part channel identifier (`NET.STA.LOC.CHA`).
    pub channel: ChannelId,
    /// Station/sensor geographic latitude in degrees (`[-90, 90]`).
    pub latitude: f64,
    /// Station/sensor geographic longitude in degrees (`[-180, 180]`).
    pub longitude: f64,
    /// Surface elevation above sea level in meters.
    pub elevation_m: f64,
    /// Local burial depth below surface in meters.
    pub depth_m: f64,
    /// Sensor component azimuth in degrees clockwise from North (`[0, 360)`).
    pub azimuth_deg: Option<f64>,
    /// Sensor component dip in degrees from horizontal (`-90` = up in SEED, `+90` = down).
    pub dip_deg: Option<f64>,
    /// Nominal channel sampling rate in Hz.
    pub sample_rate_hz: Option<f64>,
    /// Epoch start timestamp (`None` if open-ended).
    pub start_date: Option<UtcTimestamp>,
    /// Epoch end timestamp (`None` if currently active).
    pub end_date: Option<UtcTimestamp>,
    /// Optional instrument response cascade.
    pub response: Option<ChannelResponse>,
}

impl ChannelMetadata {
    /// Returns `true` if `ts` falls within `[start_date, end_date]`.
    pub fn active_at(&self, ts: UtcTimestamp) -> bool {
        if let Some(start) = self.start_date {
            if ts < start {
                return false;
            }
        }
        if let Some(end) = self.end_date {
            if ts > end {
                return false;
            }
        }
        true
    }

    /// Converts SEED dip (`[-90..+90]` from horizontal, `-90` = vertical up) to SAC `CMPINC`
    /// (`[0..180]` from vertical up: `cmpinc = dip + 90`).
    pub fn sac_cmpinc_deg(&self) -> Option<f64> {
        self.dip_deg.map(|dip| dip + 90.0)
    }
}

/// Station-level metadata grouping channels at a site.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StationMetadata {
    /// Station code (e.g. `"ANMO"`).
    pub code: String,
    /// Station latitude in degrees.
    pub latitude: f64,
    /// Station longitude in degrees.
    pub longitude: f64,
    /// Station elevation in meters.
    pub elevation_m: f64,
    /// Human-readable site name.
    pub site_name: Option<String>,
    /// Epoch start timestamp.
    pub start_date: Option<UtcTimestamp>,
    /// Epoch end timestamp.
    pub end_date: Option<UtcTimestamp>,
    /// Channels belonging to this station epoch.
    pub channels: Vec<ChannelMetadata>,
}

/// Network-level metadata grouping stations (`FDSN` `StationXML` `Network`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NetworkMetadata {
    /// Network code (e.g. `"IU"`).
    pub code: String,
    /// Optional network description.
    pub description: Option<String>,
    /// Epoch start timestamp.
    pub start_date: Option<UtcTimestamp>,
    /// Epoch end timestamp.
    pub end_date: Option<UtcTimestamp>,
    /// Stations belonging to this network.
    pub stations: Vec<StationMetadata>,
}

/// Multi-network FDSN station & response inventory (`StationXML` / `SACPZ` container).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StationInventory {
    /// Source string (e.g. `"IRIS-DMC"` or `"EarthSnail"`).
    pub source: Option<String>,
    /// Hierarchical network -> station -> channel tree.
    pub networks: Vec<NetworkMetadata>,
}

impl StationInventory {
    /// Creates an empty `StationInventory`.
    pub const fn new() -> Self {
        Self {
            source: None,
            networks: Vec::new(),
        }
    }

    /// Inserts a `ChannelMetadata` into the matching `(network, station)` node, creating nodes if needed.
    pub fn add_channel(&mut self, meta: ChannelMetadata) {
        let net_idx = if let Some(pos) = self
            .networks
            .iter()
            .position(|n| n.code.eq_ignore_ascii_case(&meta.channel.network))
        {
            pos
        } else {
            self.networks.push(NetworkMetadata {
                code: meta.channel.network.clone(),
                ..NetworkMetadata::default()
            });
            self.networks.len() - 1
        };

        let net = &mut self.networks[net_idx];
        let sta_idx = if let Some(pos) = net
            .stations
            .iter()
            .position(|s| s.code.eq_ignore_ascii_case(&meta.channel.station))
        {
            pos
        } else {
            net.stations.push(StationMetadata {
                code: meta.channel.station.clone(),
                latitude: meta.latitude,
                longitude: meta.longitude,
                elevation_m: meta.elevation_m,
                ..StationMetadata::default()
            });
            net.stations.len() - 1
        };

        net.stations[sta_idx].channels.push(meta);
    }

    /// Iterates over every `ChannelMetadata` across all networks and stations.
    pub fn channels(&self) -> impl Iterator<Item = &ChannelMetadata> {
        self.networks
            .iter()
            .flat_map(|n| n.stations.iter())
            .flat_map(|s| s.channels.iter())
    }

    /// Finds the matching `ChannelMetadata` for `id` (and optional UTC epoch timestamp `at_time`).
    pub fn find_channel(
        &self,
        id: &ChannelId,
        at_time: Option<UtcTimestamp>,
    ) -> Option<&ChannelMetadata> {
        self.channels().find(|ch| {
            ch.channel.network.eq_ignore_ascii_case(&id.network)
                && ch.channel.station.eq_ignore_ascii_case(&id.station)
                && ch.channel.location.eq_ignore_ascii_case(&id.location)
                && ch.channel.channel.eq_ignore_ascii_case(&id.channel)
                && at_time.is_none_or(|ts| ch.active_at(ts))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadband_polezero_and_sacpz_kinematic_parity() {
        // Classic 120-s STS-2 velocity sensor model:
        // 2 zeros at origin (velocity response), corner poles at -0.037004 +- 0.037016i rad/s, Sd = 1.5e9 counts/(m/s).
        let p1 = Complex64::new(-0.037_004, 0.037_016);
        let p2 = p1.conj();
        let mut pz = PolesZerosStage {
            pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
            a0: 1.0,
            normalization_freq: 1.0,
            zeros: alloc::vec![Complex64::ZERO, Complex64::ZERO],
            poles: alloc::vec![p1, p2],
            stage_gain: 1.5e9,
            input_sample_rate_hz: None,
        };
        pz.recompute_a0();

        let resp_vel = ChannelResponse {
            input_units: KinematicUnit::Velocity,
            output_units: String::from("COUNTS"),
            instrument_sensitivity: Some(StageGain {
                value: 1.5e9,
                frequency_hz: 1.0,
            }),
            stages: alloc::vec![ResponseStage::PolesZeros(pz.clone())],
        };

        // At normalization frequency 1.0 Hz, |H_vel(1.0 Hz)| == stage_gain == 1.5e9!
        let h_1hz = resp_vel.eval_freq(1.0);
        assert!(((h_1hz.abs() - 1.5e9) / 1.5e9).abs() < 1e-14);

        // Equivalent SACPZ displacement file has 3 zeros at origin and CONSTANT = a0 * 1.5e9!
        let resp_sacpz = ChannelResponse::from_sacpz(
            alloc::vec![Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            alloc::vec![p1, p2],
            pz.a0 * 1.5e9,
        );

        for &f in &[0.01, 0.1, 1.0, 5.0, 20.0] {
            let v1 = resp_vel.eval_freq_kinematic(f, KinematicUnit::Velocity);
            let v2 = resp_sacpz.eval_freq_kinematic(f, KinematicUnit::Velocity);
            assert!((v1 - v2).abs() / v1.abs() < 1e-14);

            let d1 = resp_vel.eval_freq_kinematic(f, KinematicUnit::Displacement);
            let d2 = resp_sacpz.eval_freq_kinematic(f, KinematicUnit::Displacement);
            assert!((d1 - d2).abs() / d1.abs() < 1e-14);
        }

        // Odd symmetric FIR stage with delay == correction has pure real response!
        let fir = FirStage {
            symmetry: FirSymmetry::Odd,
            coefficients: alloc::vec![0.1, 0.25, 0.3], // expands to [0.1, 0.25, 0.3, 0.25, 0.1], sum = 1.0, delay = 2 samples
            input_sample_rate_hz: 100.0,
            decimation_factor: 2,
            delay_s: 0.02,
            correction_s: 0.02,
            stage_gain: 1.0,
            gain_frequency_hz: 0.0,
        };
        assert!((fir.eval_freq(0.0).re - 1.0).abs() < 1e-13);
        assert!(fir.eval_freq(12.5).im.abs() < 1e-13);
    }

    #[test]
    fn iu_anmo_bhz_sacpz_and_stationxml_evalresp_golden_parity() {
        // Exact IU.ANMO.10.BHZ SACPZ & 3-stage StationXML reference from ObsPy/evalresp:
        let poles = alloc::vec![
            Complex64::new(-0.036_805_6, -0.036_264_9),
            Complex64::new(-0.036_805_6, 0.036_264_9),
            Complex64::new(-32.55, 0.0),
            Complex64::new(-142.0, 0.0),
            Complex64::new(-364.0, -404.0),
            Complex64::new(-364.0, 404.0),
            Complex64::new(-1260.0, 0.0),
            Complex64::new(-4900.0, -5200.0),
            Complex64::new(-4900.0, 5200.0),
            Complex64::new(-7100.0, -1700.0),
            Complex64::new(-7100.0, 1700.0),
        ];
        let zeros_sacpz = alloc::vec![
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(-31.63, 0.0),
            Complex64::new(-160.0, 0.0),
            Complex64::new(-350.0, 0.0),
            Complex64::new(-3177.0, 0.0),
            Complex64::ZERO,
        ];
        let sacpz = ChannelResponse::from_sacpz(zeros_sacpz, poles.clone(), 1.671_734e27);
        let freqs = [0.01, 0.1, 1.0, 5.0, 15.0];
        let exp_sacpz_vel = [
            Complex64::new(439_962_993.531_782_5, 1_592_012_196.381_528_6),
            Complex64::new(1_993_585_381.290_493_5, 234_590_415.900_936_25),
            Complex64::new(2_009_526_414.275_789_3, 17_575_194.378_130_317),
            Complex64::new(2_033_487_059.718_200_2, -49_910_580.474_672_064),
            Complex64::new(2_051_992_424.448_812_5, -214_313_483.470_316_2),
        ];
        for (i, &f) in freqs.iter().enumerate() {
            let got = sacpz.eval_freq_kinematic(f, KinematicUnit::Velocity);
            let rel_err = (got - exp_sacpz_vel[i]).abs() / exp_sacpz_vel[i].abs();
            assert!(
                rel_err < 1e-12,
                "sacpz VEL mismatch at {f} Hz: rel_err={rel_err}"
            );
        }

        // 3-stage StationXML (Stage 1 PZ + Stage 2 Gain + Stage 3 39-tap Q330 Coefficients FIR)
        let mut pz_stage = PolesZerosStage {
            pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
            a0: 8.465_85e17,
            normalization_freq: 0.02,
            zeros: alloc::vec![
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(-31.63, 0.0),
                Complex64::new(-160.0, 0.0),
                Complex64::new(-350.0, 0.0),
                Complex64::new(-3177.0, 0.0),
            ],
            poles,
            stage_gain: 1177.0,
            input_sample_rate_hz: Some(40.0),
        };
        pz_stage.recompute_a0();
        let stage2 = CoefficientsStage {
            numerator: Vec::new(),
            denominator: Vec::new(),
            input_sample_rate_hz: 40.0,
            correction_s: 0.0,
            stage_gain: 1_677_720.0,
            gain_frequency_hz: 0.0,
        };
        let stage3 = CoefficientsStage {
            numerator: alloc::vec![
                4.18952e-13,
                0.000330318,
                0.00102921,
                -0.00314123,
                0.000205709,
                0.00152521,
                -0.00623193,
                0.0104801,
                -0.0131202,
                0.0107821,
                -0.00144455,
                -0.0158729,
                0.0395074,
                -0.0651036,
                0.0853716,
                -0.0891913,
                0.0500619,
                0.837233,
                0.266723,
                -0.166693,
                0.095284,
                -0.0509218,
                0.0161458,
                0.00706362,
                -0.0183877,
                0.0199414,
                -0.0154895,
                0.00852735,
                -0.00255789,
                -0.00181103,
                0.00242649,
                -0.00375769,
                0.000467293,
                0.000633072,
                -1.56874e-6,
                -1.2548e-5,
                3.21041e-7,
                -2.63324e-8,
                -5.09997e-8,
            ],
            denominator: Vec::new(),
            input_sample_rate_hz: 40.0,
            correction_s: 0.43046,
            stage_gain: 1.0,
            gain_frequency_hz: 0.0,
        };
        let sxml_resp = ChannelResponse {
            input_units: KinematicUnit::Velocity,
            output_units: String::from("counts"),
            instrument_sensitivity: Some(StageGain {
                value: 1_974_680_000.0,
                frequency_hz: 0.02,
            }),
            stages: alloc::vec![
                ResponseStage::PolesZeros(pz_stage),
                ResponseStage::Coefficients(stage2),
                ResponseStage::Coefficients(stage3),
            ],
        };
        let exp_sxml_vel = [
            Complex64::new(439_665_477.279_939_06, 1_592_093_316.811_509_1),
            Complex64::new(1_993_319_473.703_265_7, 238_332_824.186_742_57),
            Complex64::new(2_022_248_410.412_974, 55_232_947.151_010_04),
            Complex64::new(2_064_730_905.494_899_7, 127_479_581.726_010_78),
            Complex64::new(2_049_684_414.460_646_9, 239_056_391.900_329_38),
        ];
        for (i, &f) in freqs.iter().enumerate() {
            let got = sxml_resp.eval_freq_kinematic(f, KinematicUnit::Velocity);
            let rel_err = (got - exp_sxml_vel[i]).abs() / exp_sxml_vel[i].abs();
            // Note: Stage 1*2 gain (1177*1677720 = 1974676440) vs overall_sensitivity (1974680000) is 1.8e-6 if unscaled,
            // or exact to <1e-5 with individual stage gains!
            assert!(
                rel_err < 5e-6,
                "StationXML evalresp VEL mismatch at {f} Hz: got {got} vs exp {}, rel_err={rel_err}",
                exp_sxml_vel[i]
            );
        }
    }
}
