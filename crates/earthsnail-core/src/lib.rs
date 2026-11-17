//! `earthsnail-core`: Zero-dependency pure-Rust foundation crate for the EarthSnail
//! seismological engine (`#![no_std] + alloc` compatible, `#![forbid(unsafe_code)]`).
//!
//! Provides exact nanosecond UTC timestamps (`UtcTimestamp(i128)`), the complete 158-word
//! `SacHeader`, `Trace` & `Stream` containers, `Complex64`, `StationInventory` & frequency-response
//! evaluators, `EventCatalog` & `MomentTensor6`, antipodal-safe WGS84/GRS80 ellipsoidal geodesics,
//! reusable `ScratchBuffer` workspaces, and zero-dependency SVG graphics primitives.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod complex;
pub mod error;
pub mod event;
pub mod geodesics;
pub mod inventory;
pub mod math;
pub mod sac_header;
pub mod scratch;
pub mod svg;
pub mod time;
pub mod trace;

pub use complex::Complex64;
pub use error::{EarthSnailError, Result};
pub use event::{
    Arrival, EvaluationMode, Event, EventCatalog, FocalMechanism, Magnitude, MomentTensor6,
    NodalPlane, Origin, OriginQuality, Pick, PickOnset, PickPolarity, PrincipalAxes, PrincipalAxis,
};
pub use geodesics::{
    degrees2kilometers, direct_geodesic, direct_geodesic_ellipsoid, dziewonski_gilbert_weights,
    geocentric_central_angle_deg, geocentric_to_geographic_lat, geographic_to_geocentric_lat,
    gps2dist_azimuth, gps2dist_azimuth_ellipsoid, kilometer2degrees, locations2degrees,
    spherical_interpolate, spherical_midpoint, spherical_waypoints, try_gps2dist_azimuth,
    DirectGeodesicResult, Ellipsoid, GeodesicResult, EARTH_RADIUS_KM, EARTH_RADIUS_M, KM_PER_DEG,
};
pub use inventory::{
    ChannelMetadata, ChannelResponse, CoefficientsStage, FirStage, FirSymmetry, KinematicUnit,
    NetworkMetadata, PolesZerosStage, PolynomialStage, PzTransferFunctionType, ResponseStage,
    StageGain, StationInventory, StationMetadata,
};
pub use sac_header::{
    is_sac_float_undef, is_sac_int_undef, is_sac_str_undef, SacDepType, SacEndian, SacFileType,
    SacHeader, SacZeroTime, SAC_FLOAT_UNDEF, SAC_FLOAT_UNDEF_F32, SAC_HEADER_BYTES,
    SAC_HEADER_WORDS, SAC_INT_UNDEF, SAC_STR_UNDEF,
};
pub use scratch::{ScratchBuffer, ScratchWorkspace};
pub use svg::{nice_ticks, xml_escape, PlotViewport, SvgCanvas, SvgPathBuilder, SvgStyle};
pub use time::UtcTimestamp;
pub use trace::{ChannelId, GapOverlap, Stream, Trace, TraceStats};
