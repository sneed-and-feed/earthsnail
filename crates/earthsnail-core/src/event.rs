//! Seismic event catalog, hypocentral origins, magnitudes, phase picks, focal mechanisms,
//! and 6-component moment tensor types (`QuakeML` 1.2 & GCMT compatible).

use crate::math;
use crate::time::UtcTimestamp;
use crate::trace::ChannelId;
use alloc::string::String;
use alloc::vec::Vec;

/// First-motion polarity of a seismic phase pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickPolarity {
    /// Upward / compressional first motion (`+`, `c`, `u`).
    Positive,
    /// Downward / dilatational first motion (`-`, `d`).
    Negative,
    /// Ambiguous polarity.
    Undecidable,
}

/// Sharpness of a seismic arrival onset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickOnset {
    /// Sharp impulsive arrival (`i`).
    Impulsive,
    /// Gradual emergent arrival (`e`).
    Emergent,
    /// Questionable onset.
    Questionable,
}

/// Human vs automated pick/origin review mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationMode {
    /// Analyst-reviewed.
    Manual,
    /// Automated picker/locator output.
    Automatic,
}

/// Phase arrival pick observed on a single `ChannelId`.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    /// Optional QuakeML `publicID`.
    pub public_id: Option<String>,
    /// Observed arrival UTC timestamp.
    pub time: UtcTimestamp,
    /// Recording channel (`NET.STA.LOC.CHA`).
    pub channel: ChannelId,
    /// Optional phase hint (e.g. `"P"`, `"S"`, `"Pn"`, `"PKIKP"`).
    pub phase_hint: Option<String>,
    /// Optional first-motion polarity.
    pub polarity: Option<PickPolarity>,
    /// Optional arrival onset character.
    pub onset: Option<PickOnset>,
    /// Symmetric $1\sigma$ pick uncertainty in seconds.
    pub time_uncertainty_s: Option<f64>,
    /// Manual vs automatic pick provenance.
    pub evaluation_mode: Option<EvaluationMode>,
}

/// Association linking an observed `Pick` to a located `Origin` with residual & weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Arrival {
    /// Referenced `Pick::public_id`.
    pub pick_id: Option<String>,
    /// Assigned seismic phase leg name (e.g. `"P"`, `"S"`).
    pub phase: String,
    /// Source-to-station azimuth in degrees.
    pub azimuth_deg: Option<f64>,
    /// Source-to-station epicentral distance in degrees.
    pub distance_deg: Option<f64>,
    /// Takeoff angle from downward vertical at source in degrees.
    pub takeoff_angle_deg: Option<f64>,
    /// Travel-time residual $t_{\text{obs}} - t_{\text{pred}}$ in seconds.
    pub time_residual_s: Option<f64>,
    /// Normalized weight used in the hypocenter inversion.
    pub time_weight: Option<f64>,
}

/// Hypocenter inversion quality summary (`QuakeML` `OriginQuality`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OriginQuality {
    /// Total associated arrival count.
    pub associated_phase_count: Option<u32>,
    /// Defining arrival count with positive weight.
    pub used_phase_count: Option<u32>,
    /// Weighted RMS travel-time residual in seconds.
    pub standard_error_s: Option<f64>,
    /// Largest azimuthal gap between defining stations in degrees.
    pub azimuthal_gap_deg: Option<f64>,
    /// Distance to closest station in degrees.
    pub minimum_distance_deg: Option<f64>,
    /// Distance to farthest station in degrees.
    pub maximum_distance_deg: Option<f64>,
}

/// Hypocentral earthquake origin (`time`, `latitude`, `longitude`, `depth_m`).
#[derive(Debug, Clone, PartialEq)]
pub struct Origin {
    /// Optional QuakeML `publicID`.
    pub public_id: Option<String>,
    /// Origin rupture UTC timestamp.
    pub time: UtcTimestamp,
    /// Hypocenter geographic latitude in degrees (`[-90, 90]`).
    pub latitude: f64,
    /// Hypocenter geographic longitude in degrees (`[-180, 180]`).
    pub longitude: f64,
    /// Hypocenter focal depth below sea level in **meters** (`QuakeML` 1.2 standard).
    pub depth_m: f64,
    /// Origin time $1\sigma$ uncertainty in seconds.
    pub time_uncertainty_s: Option<f64>,
    /// Latitude $1\sigma$ uncertainty in kilometers.
    pub latitude_uncertainty_km: Option<f64>,
    /// Longitude $1\sigma$ uncertainty in kilometers.
    pub longitude_uncertainty_km: Option<f64>,
    /// Depth $1\sigma$ uncertainty in meters.
    pub depth_uncertainty_m: Option<f64>,
    /// Inversion quality metrics.
    pub quality: Option<OriginQuality>,
    /// Associated phase arrivals.
    pub arrivals: Vec<Arrival>,
}

impl Origin {
    /// Constructs a minimal `Origin` specifying focal depth in kilometers (`depth_km * 1000.0`).
    pub fn from_depth_km(time: UtcTimestamp, latitude: f64, longitude: f64, depth_km: f64) -> Self {
        Self {
            public_id: None,
            time,
            latitude,
            longitude,
            depth_m: depth_km * 1_000.0,
            time_uncertainty_s: None,
            latitude_uncertainty_km: None,
            longitude_uncertainty_km: None,
            depth_uncertainty_m: None,
            quality: None,
            arrivals: Vec::new(),
        }
    }

    /// Focal depth in kilometers (`depth_m * 1e-3`).
    #[inline]
    pub fn depth_km(&self) -> f64 {
        self.depth_m * 1e-3
    }
}

/// Earthquake magnitude estimate (`Mw`, `mb`, `MS`, `ML`, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct Magnitude {
    /// Optional QuakeML `publicID`.
    pub public_id: Option<String>,
    /// Dimensionless magnitude value.
    pub mag: f64,
    /// Magnitude scale label (e.g. `"Mw"`, `"mb"`, `"ML"`).
    pub mag_type: String,
    /// Associated `Origin::public_id`.
    pub origin_id: Option<String>,
    /// Number of stations contributing to this magnitude.
    pub station_count: Option<u32>,
    /// $1\sigma$ magnitude uncertainty.
    pub uncertainty: Option<f64>,
}

/// Symmetric 3x3 seismic moment tensor in spherical Harvard/GCMT/QuakeML `(r, theta, phi)` =
/// `(Up, South, East)` coordinates, in Newton-meters ($\text{N}\cdot\text{m}$).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MomentTensor6 {
    /// Radial-radial (`Up-Up`) component $M_{rr}$ in $\text{N}\cdot\text{m}$.
    pub m_rr: f64,
    /// Colatitude-colatitude (`South-South`) component $M_{\theta\theta}$ in $\text{N}\cdot\text{m}$.
    pub m_tt: f64,
    /// Longitude-longitude (`East-East`) component $M_{\phi\phi}$ in $\text{N}\cdot\text{m}$.
    pub m_pp: f64,
    /// Radial-colatitude (`Up-South`) component $M_{r\theta}$ in $\text{N}\cdot\text{m}$.
    pub m_rt: f64,
    /// Radial-longitude (`Up-East`) component $M_{r\phi}$ in $\text{N}\cdot\text{m}$.
    pub m_rp: f64,
    /// Colatitude-longitude (`South-East`) component $M_{\theta\phi}$ in $\text{N}\cdot\text{m}$.
    pub m_tp: f64,
}

impl MomentTensor6 {
    /// Creates a `MomentTensor6` from `(m_rr, m_tt, m_pp, m_rt, m_rp, m_tp)` in $\text{N}\cdot\text{m}$.
    pub const fn new(m_rr: f64, m_tt: f64, m_pp: f64, m_rt: f64, m_rp: f64, m_tp: f64) -> Self {
        Self {
            m_rr,
            m_tt,
            m_pp,
            m_rt,
            m_rp,
            m_tp,
        }
    }

    /// Converts to Aki & Richards `(North, East, Down)` symmetric 3x3 matrix:
    /// $M_{NN} = M_{\theta\theta}$, $M_{EE} = M_{\phi\phi}$, $M_{DD} = M_{rr}$,
    /// $M_{NE} = -M_{\theta\phi}$, $M_{ND} = M_{r\theta}$, $M_{ED} = -M_{r\phi}$.
    pub const fn to_ned_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.m_tt, -self.m_tp, self.m_rt],
            [-self.m_tp, self.m_pp, -self.m_rp],
            [self.m_rt, -self.m_rp, self.m_rr],
        ]
    }

    /// Constructs from Aki & Richards `(North, East, Down)` components `(m_nn, m_ee, m_dd, m_ne, m_nd, m_ed)`.
    pub const fn from_ned(
        m_nn: f64,
        m_ee: f64,
        m_dd: f64,
        m_ne: f64,
        m_nd: f64,
        m_ed: f64,
    ) -> Self {
        Self {
            m_rr: m_dd,
            m_tt: m_nn,
            m_pp: m_ee,
            m_rt: m_nd,
            m_rp: -m_ed,
            m_tp: -m_ne,
        }
    }

    /// Isotropic trace $\operatorname{tr}(M) = M_{rr} + M_{\theta\theta} + M_{\phi\phi}$.
    #[inline]
    pub fn trace(&self) -> f64 {
        self.m_rr + self.m_tt + self.m_pp
    }

    /// Silver & Jordan (1982) scalar seismic moment $M_0 = \sqrt{\frac{1}{2}\sum_{i,j} M_{ij}^2}$ in $\text{N}\cdot\text{m}$.
    pub fn scalar_moment_nm(&self) -> f64 {
        let diag_sq = self.m_rr * self.m_rr + self.m_tt * self.m_tt + self.m_pp * self.m_pp;
        let off_sq = 2.0 * (self.m_rt * self.m_rt + self.m_rp * self.m_rp + self.m_tp * self.m_tp);
        math::sqrt(0.5 * (diag_sq + off_sq))
    }

    /// Hanks & Kanamori (1979) moment magnitude $M_w = \frac{2}{3}(\log_{10} M_0 - 9.1)$ with $M_0$ in $\text{N}\cdot\text{m}$.
    pub fn moment_magnitude_mw(&self) -> f64 {
        let m0 = self.scalar_moment_nm();
        if m0 <= 0.0 {
            f64::NEG_INFINITY
        } else {
            (2.0 / 3.0) * (math::log10(m0) - 9.1)
        }
    }
}

/// Fault nodal plane parameterized by Aki & Richards `(strike, dip, rake)` in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodalPlane {
    /// Fault strike clockwise from North in `[0, 360)` degrees.
    pub strike_deg: f64,
    /// Fault dip downward from horizontal in `[0, 90]` degrees.
    pub dip_deg: f64,
    /// Slip rake angle in `[-180, 180]` degrees.
    pub rake_deg: f64,
}

/// Principal eigenvector axis (`T`, `P`, or `N`/`B`) of a focal mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalAxis {
    /// Trend/azimuth clockwise from North in `[0, 360)` degrees.
    pub azimuth_deg: f64,
    /// Plunge downward from horizontal in `[0, 90]` degrees.
    pub plunge_deg: f64,
    /// Eigenvalue in $\text{N}\cdot\text{m}$.
    pub length: f64,
}

/// Tension (`T`), Pressure (`P`), and Null (`N`) principal axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalAxes {
    /// Tension axis (largest eigenvalue).
    pub t_axis: PrincipalAxis,
    /// Pressure axis (smallest/most negative eigenvalue).
    pub p_axis: PrincipalAxis,
    /// Null / intermediate (`B`) axis.
    pub n_axis: Option<PrincipalAxis>,
}

/// Complete earthquake focal mechanism (`NodalPlane` pair and/or `MomentTensor6`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FocalMechanism {
    /// Optional QuakeML `publicID`.
    pub public_id: Option<String>,
    /// Primary nodal plane.
    pub nodal_plane_1: Option<NodalPlane>,
    /// Auxiliary nodal plane.
    pub nodal_plane_2: Option<NodalPlane>,
    /// Full 6-component seismic moment tensor.
    pub moment_tensor: Option<MomentTensor6>,
    /// Principal `T`/`P`/`N` axes.
    pub principal_axes: Option<PrincipalAxes>,
}

/// Seismic event aggregating origins, magnitudes, phase picks, and focal mechanisms.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Event {
    /// Event identifier (`QuakeML` `publicID`).
    pub public_id: String,
    /// Optional event type (e.g. `"earthquake"`, `"quarry blast"`).
    pub event_type: Option<String>,
    /// Optional Flinn-Engdahl or geographic region description.
    pub description: Option<String>,
    /// Preferred `Origin::public_id`.
    pub preferred_origin_id: Option<String>,
    /// Preferred `Magnitude::public_id`.
    pub preferred_magnitude_id: Option<String>,
    /// Preferred `FocalMechanism::public_id`.
    pub preferred_focal_mechanism_id: Option<String>,
    /// Candidate hypocentral origins.
    pub origins: Vec<Origin>,
    /// Candidate magnitudes.
    pub magnitudes: Vec<Magnitude>,
    /// Phase arrival picks.
    pub picks: Vec<Pick>,
    /// Focal mechanisms / moment tensors.
    pub focal_mechanisms: Vec<FocalMechanism>,
}

impl Event {
    /// Returns the preferred `Origin` (matching `preferred_origin_id`, falling back to `origins.first()`).
    pub fn preferred_origin(&self) -> Option<&Origin> {
        if let Some(ref pid) = self.preferred_origin_id {
            if let Some(o) = self
                .origins
                .iter()
                .find(|o| o.public_id.as_deref() == Some(pid.as_str()))
            {
                return Some(o);
            }
        }
        self.origins.first()
    }

    /// Returns the preferred `Magnitude` (matching `preferred_magnitude_id`, falling back to `magnitudes.first()`).
    pub fn preferred_magnitude(&self) -> Option<&Magnitude> {
        if let Some(ref pid) = self.preferred_magnitude_id {
            if let Some(m) = self
                .magnitudes
                .iter()
                .find(|m| m.public_id.as_deref() == Some(pid.as_str()))
            {
                return Some(m);
            }
        }
        self.magnitudes.first()
    }

    /// Returns the preferred `FocalMechanism` (matching `preferred_focal_mechanism_id`, falling back to `first()`).
    pub fn preferred_focal_mechanism(&self) -> Option<&FocalMechanism> {
        if let Some(ref pid) = self.preferred_focal_mechanism_id {
            if let Some(fm) = self
                .focal_mechanisms
                .iter()
                .find(|fm| fm.public_id.as_deref() == Some(pid.as_str()))
            {
                return Some(fm);
            }
        }
        self.focal_mechanisms.first()
    }
}

/// Collection of seismic `Event` entries (`QuakeML` `EventParameters` / catalog).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EventCatalog {
    /// Optional catalog description or agency header.
    pub description: Option<String>,
    /// Events in the catalog.
    pub events: Vec<Event>,
}

impl EventCatalog {
    /// Creates an empty `EventCatalog`.
    pub const fn new() -> Self {
        Self {
            description: None,
            events: Vec::new(),
        }
    }

    /// Number of events in the catalog.
    #[inline]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the catalog has no events.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Appends an `Event` to the catalog.
    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Returns a filtered catalog containing events whose preferred magnitude is `>= min_mag`.
    pub fn filter_min_magnitude(&self, min_mag: f64) -> Self {
        let events = self
            .events
            .iter()
            .filter(|ev| ev.preferred_magnitude().is_some_and(|m| m.mag >= min_mag))
            .cloned()
            .collect();
        Self {
            description: self.description.clone(),
            events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moment_tensor_scalar_moment_and_mw() {
        // Pure double-couple strike-slip M_tp = 1e19 N*m -> M0 = 1e19 N*m -> Mw = 2/3*(19 - 9.1) = 6.6
        let mt = MomentTensor6::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0e19);
        assert!((mt.scalar_moment_nm() - 1.0e19).abs() < 1e4);
        assert!((mt.moment_magnitude_mw() - 6.6).abs() < 1e-12);

        let ned = mt.to_ned_matrix();
        assert_eq!(ned[0][1], -1.0e19);
        assert_eq!(ned[1][0], -1.0e19);
    }
}
