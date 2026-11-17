//! Seismological digital signal processing, spectral analysis, filtering, instrument
//! deconvolution, phase picking, 3C/array processing, and moment tensor algebra for EarthSnail.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub use earthsnail_core as core_types;

pub mod array_source;
pub mod correlate_pick;
pub mod fft;
pub mod filter;
pub mod moment_tensor;
pub mod preprocess;
pub mod response;
pub mod spectral;

pub use array_source::{
    arias_intensity, fk_analysis, flinn_polarization, jma_instrumental_shindo,
    jurkevics_polarization, nigam_jennings_response_spectra, peak_ground_motion,
    polarization_analysis_flinn, receiver_function_iter, receiver_function_waterlevel,
    rotate_lqt_zne, rotate_ne_rt, rotate_rt_ne, rotate_to_zne, rotate_uvw_zne, rotate_zne_lqt,
    rotate_zne_uvw, silver_chan_splitting, slant_stack, summarize_ground_motion, vespagram,
    worden_mmi, ArrayStation, FkAnalysisResult, FkMethod, GroundMotionSummary, IterativeRfResult,
    PolarizationMetrics, ResponseSpectrum, ShearWaveSplittingResult, SlidingPolarizationResult,
    StackMethod, VespagramResult,
};
pub use correlate_pick::{
    aic_pick, aic_simple, carl_sta_trig, classic_sta_lta, correlate, correlate_template,
    correlate_template_with_scratch, correlate_with_scratch, morse_pick, morse_pick_default, pcc,
    pk_baer, recursive_sta_lta, trigger_onset, trigger_onset_with_options, xcorr_max,
    xcorr_max_subsample, z_detect, BaerPickParams, BaerPickResult, MorsePickConfig,
    MorsePickResult, XcorrMode, XcorrNormalize,
};
pub use fft::{
    fft, fft_in_place, fft_with_scratch, fftfreq, ifft, ifft_in_place, ifft_with_scratch, irfft,
    irfft_with_scratch, largest_prime_factor, next_pow2, npts2nfft, prev_pow2, rfft,
    rfft_with_scratch, rfftfreq,
};
pub use filter::{
    bandpass, bandstop, bessel, butter, cheby1, cheby2, design_iir_sos, highpass, lowpass, sosfilt,
    sosfilt_zerophase_obspy, sosfilt_zi, sosfiltfilt, sosfiltfilt_with_scratch, BesselNorm,
    FilterBand, FilterFamily, SosFilter, SosSection,
};
pub use moment_tensor::{
    auxiliary_plane, beachball_p_radiation_grid, decompose_moment_tensor, jacobi_eigen_3x3,
    mt_to_nodal_planes, mt_to_principal_axes, normal_slip_to_nodal_plane, p_radiation_amplitude,
    strike_dip_rake_to_mt, BeachballGrid, BeachballProjection, MomentTensorDecomposition,
};
pub use preprocess::{
    bessel_i0, blackman_window, cosine_taper, decimate, decimate_trace, demean, demean_in_place,
    detrend_linear, detrend_linear_in_place, detrend_polynomial, detrend_polynomial_in_place,
    detrend_simple_in_place, differentiate_gradient, differentiate_spectral, hamming_window,
    hann_window, integrate_spectral, integrate_trapezoidal, kaiser_window, lanczos_resample,
    merge_stream, rmean_in_place, rtrend_in_place, sac_cosine_taper, sinc_resample, taper_in_place,
    taper_window, trim_stream, trim_trace, tukey_window, LanczosWindow, MergeStrategy, TaperSide,
    TaperType,
};
pub use response::{
    cosine_sac_taper, invert_spectrum_waterlevel, remove_response, remove_response_trace,
    remove_response_with_scratch, sac_transfer, wood_anderson_response_iaspei,
    ResponseRemovalConfig,
};
pub use spectral::{
    analytic_signal, analytic_signal_with_scratch, cwt_morlet, envelope, eval_nhnm_db,
    eval_nlnm_db, get_nhnm, get_nlnm, hvsr, instantaneous_frequency, instantaneous_phase,
    konno_ohmachi_smooth, konno_ohmachi_window, welch_psd, HvsrConfig, HvsrResult, Ppsd,
    PpsdConfig, PsdEstimate, WelchConfig, WelchDetrend, WelchWindow,
};
