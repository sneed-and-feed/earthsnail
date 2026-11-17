//! Global spherical & 3D-perturbed seismic travel-time, pierce, ray-path, ellipticity, and
//! damped Huber-IRLS 4D hypocenter engine for EarthSnail (`iasp91`, `ak135`, `prem`, custom
//! `.nd` / `.tvel` compilers, full IASPEI/TauP phase-name grammar, and zero external runtime crates).
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub use earthsnail_core as core_types;

pub mod calculators;
pub mod ellipticity;
pub mod locate;
mod model_tables;
pub mod models;
pub mod phase;
pub mod tau_model;

pub use calculators::{
    spherical_great_circle_point, ArrivalPath, ArrivalPierce, PhaseCurveBranch,
    PhaseTravelTimeCurve, PiercePoint, RayPathPoint, SeismicArrival, TauEngine, TauPCurvePoint,
};
pub use ellipticity::{
    arrival_ellipticity_correction_s, ellipticity_distance_basis, evaluate_ellipticity_correction,
    hydrostatic_ellipticity, integrate_ellipticity_tau_coeffs, phase_ellipticity_tau_coeffs,
};
pub use locate::{
    locate_hypocenter, perturb_ray_path_3d, ConfidenceEllipsoid, HypocenterConfig,
    HypocenterSolution, PerturbedRayPath3D, PhaseArrivalObservation, StationResidual,
    SubductionPlumeModel3D, VelocityPerturbation3D,
};
pub use models::{MaterialProperty, VelocityLayer, VelocityModel, WaveType};
pub use phase::{
    leg_puller, parse_phase_list, RawArrival, SeismicPhase, DEFAULT_PATH_RAY_PARAM_TOL,
    DEFAULT_RAY_PARAM_TOL,
};
pub use tau_model::{
    bullen_depth_for, bullen_radial_slowness, evaluate_at_bullen, CriticalDepth, DepthRange,
    SlownessLayer, SlownessModel, SlownessModelConfig, TauBranch, TauModel, TimeDist,
};
