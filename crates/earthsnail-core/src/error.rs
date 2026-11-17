//! Unified error and `Result` types across all EarthSnail crates.

use alloc::string::String;
use core::fmt;

/// Typed, panic-free error enumeration shared across EarthSnail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EarthSnailError {
    /// Underlying OS, stream, or file I/O failure.
    Io(String),
    /// Malformed binary or text waveform/metadata format (SAC, miniSEED, XML, SACPZ, GeoCSV).
    Format(String),
    /// Invalid or inconsistent header field (negative `NPTS`, non-positive `DELTA`, `NaN`/`Inf`).
    Header(String),
    /// Waveform compression/decompression error (Steim-1/2 frame nibble, difference, or CRC mismatch).
    Compression(String),
    /// Out-of-domain parameter supplied to a mathematical, DSP, geodesic, or query routine.
    InvalidParameter(String),
    /// Invalid date, time, or ISO-8601/FDSN timestamp string.
    TimeParse(String),
    /// Numerical non-convergence or instability.
    Numerical(String),
    /// Singular or ill-conditioned linear system (covariance, beamforming, or moment tensor inversion).
    SingularMatrix(String),
    /// Invalid seismic phase name or TauP phase grammar expression.
    PhaseParse(String),
    /// Invalid planetary velocity model layer, discontinuity, or slowness sampling.
    ModelError(String),
    /// Invalid or incomplete instrument response stage (pole-zero, FIR, polynomial, or sensitivity).
    ResponseError(String),
    /// Operation requires non-empty waveform or catalog input.
    EmptyData(String),
    /// Traces are incompatible for a multi-trace operation (rate, channel, or length mismatch).
    IncompatibleTraces(String),
    /// FDSNWS or SeedLink network/protocol failure.
    Network(String),
    /// Corrupted or untrusted input exceeded safe memory/sample capacity bounds.
    CapacityExceeded(String),
    /// Feature or format variant is recognized but unsupported.
    Unsupported(String),
}

/// Alias for `core::result::Result<T, EarthSnailError>`.
pub type Result<T> = core::result::Result<T, EarthSnailError>;

impl fmt::Display for EarthSnailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::Header(msg) => write!(f, "header error: {msg}"),
            Self::Compression(msg) => write!(f, "compression error: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::TimeParse(msg) => write!(f, "timestamp parse error: {msg}"),
            Self::Numerical(msg) => write!(f, "numerical error: {msg}"),
            Self::SingularMatrix(msg) => write!(f, "singular matrix: {msg}"),
            Self::PhaseParse(msg) => write!(f, "phase parse error: {msg}"),
            Self::ModelError(msg) => write!(f, "velocity model error: {msg}"),
            Self::ResponseError(msg) => write!(f, "instrument response error: {msg}"),
            Self::EmptyData(msg) => write!(f, "empty data: {msg}"),
            Self::IncompatibleTraces(msg) => write!(f, "incompatible traces: {msg}"),
            Self::Network(msg) => write!(f, "network error: {msg}"),
            Self::CapacityExceeded(msg) => write!(f, "capacity exceeded: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported feature: {msg}"),
        }
    }
}

impl core::error::Error for EarthSnailError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for EarthSnailError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(alloc::string::ToString::to_string(&err))
    }
}
