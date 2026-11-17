//! `earthsnail-sac`: Modern structured CLI (`earthsnail`), interactive `SAC>` shell (`es-sac`),
//! `.m` macro script engine, built-in `taup` SAC header stamper, and zero-dependency W3C SVG 1.1
//! publication plotting engine.

#![forbid(unsafe_code)]

pub use earthsnail_core as core_types;

pub mod cli;
pub mod json;
pub mod plot;
pub mod sac_shell;
pub mod verify;

pub use cli::{run_cli_args, run_es_sac_args, EARTHSNAIL_VERSION};
pub use plot::{
    plot_beachball_svg, plot_ppsd_svg, plot_record_section_svg, plot_spectrogram_svg,
    plot_taup_paths_svg, plot_waveforms_svg, BeachballPlotOptions, DistanceUnit,
    FocalMechanismInput, FrequencyScale, PpsdPlotOptions, RecordSectionOptions, SpectrogramMethod,
    SpectrogramOptions, TaupPathsPlotOptions, TimeAxisMode, WaveformLayout, WaveformPlotOptions,
};
pub use sac_shell::{stamp_taup_into_sac_header, CutAnchor, CutSpec, SacSession, SavedSpectrum};
pub use verify::{run_self_verify, SelfVerifyReport, VerifyCheckReceipt};
