//! Miri-verified C-ABI (`cdylib`) and `wasm32-unknown-unknown` (`+simd128`) interface for
//! EarthSnail Studio (`studio/earthsnail_portable_studio.html`) and zero-bloat Python `ctypes`
//! bindings (`python/earthsnail/`).
//!
//! `earthsnail-wasm` is the **only** crate in the EarthSnail workspace permitted to use `unsafe`,
//! strictly quarantined to C-ABI pointer validation, alignment checks, and linear-memory buffer
//! handoffs. Every FFI entrypoint validates null pointers, alignments, and allocation caps, and
//! preserves Stacked/Tree Borrows soundness even when input and output buffers alias.

pub use earthsnail_core as core_types;

pub mod wire;

#[cfg(test)]
mod ffi_tests;

use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::fmt::Write as _;

use earthsnail_core::{
    try_gps2dist_azimuth, ChannelId, ChannelMetadata, ChannelResponse, EarthSnailError,
    KinematicUnit, MomentTensor6, NodalPlane, SacEndian, ScratchWorkspace, StationInventory, Trace,
    TraceStats, UtcTimestamp,
};
use earthsnail_dsp::{
    aic_pick, aic_simple, beachball_p_radiation_grid, classic_sta_lta, decompose_moment_tensor,
    demean_in_place, design_iir_sos, detrend_linear_in_place, envelope, eval_nhnm_db, eval_nlnm_db,
    flinn_polarization, hann_window, jacobi_eigen_3x3, morse_pick, nigam_jennings_response_spectra,
    pk_baer, recursive_sta_lta, remove_response, rfft_with_scratch, rfftfreq, rotate_lqt_zne,
    rotate_ne_rt, rotate_rt_ne, rotate_zne_lqt, rotate_zne_uvw, sosfilt, sosfilt_zerophase_obspy,
    sosfiltfilt, strike_dip_rake_to_mt, taper_in_place, trigger_onset, welch_psd, BaerPickParams,
    BeachballProjection, BesselNorm, FilterBand, FilterFamily, MorsePickConfig,
    ResponseRemovalConfig, TaperSide, TaperType, WelchConfig, WelchDetrend, WelchWindow,
};
use earthsnail_io::{
    decode_steim2, detect_waveform_format, encode_steim2, read_sacpz_bytes, read_stationxml_bytes,
    read_waveforms_bytes, write_mseed3_trace_bytes, write_mseed_trace_bytes, write_sac_alpha_bytes,
    write_sac_bytes, Mseed3WriteOptions, MseedByteOrder, MseedEncoding, MseedWriteOptions,
};
use earthsnail_taup::{
    locate_hypocenter, HypocenterConfig, PhaseArrivalObservation, SlownessModel,
    SlownessModelConfig, TauEngine, TauModel, VelocityModel,
};

/// Success return code (`0`).
pub const ES_OK: i32 = 0;
/// Null or misaligned pointer passed across C-ABI boundary (`-1`).
pub const ES_ERR_NULL_PTR: i32 = -1;
/// Invalid parameter, length, or enum discriminant (`-2`).
pub const ES_ERR_INVALID_ARG: i32 = -2;
/// Binary/XML/ASCII format or compression error (`-3`).
pub const ES_ERR_PARSE: i32 = -3;
/// Numerical/matrix/convergence error (`-4`).
pub const ES_ERR_NUMERICAL: i32 = -4;
/// Allocation failure or capacity cap exceeded (`-5`).
pub const ES_ERR_ALLOC: i32 = -5;
/// Invalid UTF-8 string slice (`-6`).
pub const ES_ERR_UTF8: i32 = -6;

/// ABI version (`100` = v0.1.0).
pub const ES_ABI_VERSION: u32 = 100;

/// Guaranteed alignment (`8` bytes) for all `es_alloc` buffers so JS `Float64Array` / `BigInt64Array`
/// views and Rust `f64` slices are always properly aligned.
const FFI_ALIGN: usize = 8;

/// Defensive maximum single FFI buffer size (`512 MiB`) preventing bogus-length OOM aborts.
const MAX_FFI_BYTES: usize = 512 * 1024 * 1024;

const ERROR_BUF_CAP: usize = 512;

struct ErrorSlot {
    buf: [u8; ERROR_BUF_CAP],
    len: usize,
}

impl ErrorSlot {
    const fn new() -> Self {
        Self {
            buf: [0u8; ERROR_BUF_CAP],
            len: 0,
        }
    }
}

thread_local! {
    static LAST_ERROR: UnsafeCell<ErrorSlot> = const { UnsafeCell::new(ErrorSlot::new()) };
}

struct FixedWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl core::fmt::Write for FixedWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let rem = self.buf.len().saturating_sub(self.pos);
        let take = rem.min(s.len());
        self.buf[self.pos..self.pos + take].copy_from_slice(&s.as_bytes()[..take]);
        self.pos += take;
        Ok(())
    }
}

#[inline]
fn clear_error() {
    LAST_ERROR.with(|cell| unsafe {
        let slot = cell.get();
        (*slot).len = 0;
    });
}

fn set_error_str(code: i32, msg: &str) -> i32 {
    LAST_ERROR.with(|cell| unsafe {
        let slot = cell.get();
        let take = msg.len().min(ERROR_BUF_CAP);
        core::ptr::copy_nonoverlapping(msg.as_ptr(), (*slot).buf.as_mut_ptr(), take);
        (*slot).len = take;
    });
    code
}

fn set_earthsnail_error(err: EarthSnailError) -> i32 {
    let code = match &err {
        EarthSnailError::InvalidParameter(_)
        | EarthSnailError::EmptyData(_)
        | EarthSnailError::IncompatibleTraces(_)
        | EarthSnailError::PhaseParse(_)
        | EarthSnailError::ModelError(_)
        | EarthSnailError::ResponseError(_) => ES_ERR_INVALID_ARG,
        EarthSnailError::Format(_)
        | EarthSnailError::Header(_)
        | EarthSnailError::Compression(_)
        | EarthSnailError::TimeParse(_)
        | EarthSnailError::Io(_) => ES_ERR_PARSE,
        EarthSnailError::Numerical(_) | EarthSnailError::SingularMatrix(_) => ES_ERR_NUMERICAL,
        EarthSnailError::CapacityExceeded(_) => ES_ERR_ALLOC,
        EarthSnailError::Network(_) | EarthSnailError::Unsupported(_) => ES_ERR_INVALID_ARG,
    };
    LAST_ERROR.with(|cell| unsafe {
        let slot = cell.get();
        let buf_slice =
            core::slice::from_raw_parts_mut(core::ptr::addr_of_mut!((*slot).buf[0]), ERROR_BUF_CAP);
        let mut writer = FixedWriter {
            buf: buf_slice,
            pos: 0,
        };
        let _ = write!(&mut writer, "{err}");
        (*slot).len = writer.pos;
    });
    code
}

/// Validates a non-null, properly aligned shared slice of `len` elements (`len == 0` allowed when
/// `allow_empty` is true, returning `&[]` without dereferencing `ptr`).
#[inline]
unsafe fn check_slice<'a, T>(
    ptr: *const T,
    len: usize,
    allow_empty: bool,
    label: &str,
) -> Result<&'a [T], i32> {
    if len == 0 {
        if !allow_empty {
            return Err(set_error_str(ES_ERR_INVALID_ARG, label));
        }
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(set_error_str(ES_ERR_NULL_PTR, label));
    }
    let align = core::mem::align_of::<T>();
    if ((ptr as usize) & (align - 1)) != 0 {
        return Err(set_error_str(
            ES_ERR_NULL_PTR,
            "misaligned input pointer across FFI boundary",
        ));
    }
    let byte_len = len
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| set_error_str(ES_ERR_ALLOC, "FFI slice length overflow"))?;
    if byte_len > MAX_FFI_BYTES {
        return Err(set_error_str(
            ES_ERR_ALLOC,
            "FFI slice exceeds 512 MiB safety cap",
        ));
    }
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) })
}

/// Validates that `ptr` is non-null, aligned for `T`, and `len` fits within `MAX_FFI_BYTES`,
/// without yet creating `&mut [T]` (allowing safe alias-free read-then-write sequences).
#[inline]
fn validate_mut_ptr<T>(ptr: *mut T, len: usize, allow_empty: bool, label: &str) -> Result<(), i32> {
    if len == 0 {
        if !allow_empty {
            return Err(set_error_str(ES_ERR_INVALID_ARG, label));
        }
        return Ok(());
    }
    if ptr.is_null() {
        return Err(set_error_str(ES_ERR_NULL_PTR, label));
    }
    let align = core::mem::align_of::<T>();
    if ((ptr as usize) & (align - 1)) != 0 {
        return Err(set_error_str(
            ES_ERR_NULL_PTR,
            "misaligned output pointer across FFI boundary",
        ));
    }
    let byte_len = len
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| set_error_str(ES_ERR_ALLOC, "FFI output slice length overflow"))?;
    if byte_len > MAX_FFI_BYTES {
        return Err(set_error_str(
            ES_ERR_ALLOC,
            "FFI output slice exceeds 512 MiB safety cap",
        ));
    }
    Ok(())
}

#[inline]
unsafe fn check_slice_mut<'a, T>(
    ptr: *mut T,
    len: usize,
    allow_empty: bool,
    label: &str,
) -> Result<&'a mut [T], i32> {
    validate_mut_ptr(ptr, len, allow_empty, label)?;
    if len == 0 {
        return Ok(&mut []);
    }
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}

#[inline]
unsafe fn check_utf8<'a>(
    ptr: *const u8,
    len: usize,
    allow_empty: bool,
    label: &str,
) -> Result<&'a str, i32> {
    let bytes = unsafe { check_slice(ptr, len, allow_empty, label)? };
    core::str::from_utf8(bytes).map_err(|_| set_error_str(ES_ERR_UTF8, "invalid UTF-8 FFI string"))
}

/// Allocates an 8-byte-aligned owned buffer via `es_alloc`, copies `bytes` into it, and writes
/// `(ptr, len)` to `(out_buf_ptr, out_buf_len)`.
unsafe fn emit_owned_bytes(
    bytes: &[u8],
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    if let Err(c) = validate_mut_ptr(out_buf_ptr, 1, false, "null out_buf_ptr") {
        return c;
    }
    if let Err(c) = validate_mut_ptr(out_buf_len, 1, false, "null out_buf_len") {
        return c;
    }
    if bytes.is_empty() {
        unsafe {
            out_buf_ptr.write(core::ptr::null_mut());
            out_buf_len.write(0);
        }
        return ES_OK;
    }
    let raw = es_alloc(bytes.len());
    if raw.is_null() {
        return ES_ERR_ALLOC;
    }
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), raw, bytes.len());
        out_buf_ptr.write(raw);
        out_buf_len.write(bytes.len());
    }
    ES_OK
}

fn parse_kinematic_unit(code: u32) -> Result<KinematicUnit, i32> {
    match code {
        0 => Ok(KinematicUnit::Displacement),
        1 => Ok(KinematicUnit::Velocity),
        2 => Ok(KinematicUnit::Acceleration),
        _ => Err(set_error_str(
            ES_ERR_INVALID_ARG,
            "target_unit must be 0 (DISP), 1 (VEL), or 2 (ACC)",
        )),
    }
}

fn parse_channel_or_default(s: &str) -> Result<ChannelId, i32> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(ChannelId::new("XX", "STA", "00", "BHZ"));
    }
    ChannelId::parse_id(trimmed).map_err(set_earthsnail_error)
}

fn find_stationxml_channel<'a>(
    inv: &'a StationInventory,
    chan_filter: &str,
    epoch_s: i64,
) -> Option<&'a ChannelMetadata> {
    let ts_opt = (epoch_s != 0).then(|| UtcTimestamp((epoch_s as i128) * 1_000_000_000));
    let pat = chan_filter.trim();

    if !pat.is_empty() {
        if let Ok(exact_id) = ChannelId::parse_id(pat) {
            if let Some(ch) = inv.find_channel(&exact_id, ts_opt) {
                return Some(ch);
            }
            if ts_opt.is_some() {
                if let Some(ch) = inv.find_channel(&exact_id, None) {
                    return Some(ch);
                }
            }
        }
        // Fallback: match by glob pattern or trailing 3-char channel code (e.g. "BHZ"):
        let mut fallback: Option<&ChannelMetadata> = None;
        for ch in inv.channels() {
            let matches =
                ch.channel.matches_pattern(pat) || ch.channel.channel.eq_ignore_ascii_case(pat);
            if matches {
                if ts_opt.is_none_or(|ts| ch.active_at(ts)) {
                    return Some(ch);
                }
                fallback = Some(ch);
            }
        }
        if fallback.is_some() {
            return fallback;
        }
    }

    if let Some(ts) = ts_opt {
        if let Some(ch) = inv.channels().find(|c| c.active_at(ts)) {
            return Some(ch);
        }
    }
    inv.channels().next()
}

// ============================================================================
// 1. Memory & Error Management
// ============================================================================

/// Returns the EarthSnail FFI ABI version (`100`).
#[no_mangle]
pub extern "C" fn es_abi_version() -> u32 {
    ES_ABI_VERSION
}

/// Allocates `len` zero-initialized, 8-byte-aligned bytes in Rust linear/heap memory.
///
/// Returns `null_mut()` and records a diagnostic in `es_last_error_*` if `len == 0` or
/// `len > 512 MiB`. Every non-null pointer returned by `es_alloc(len)` **must** be released via
/// `es_free(ptr, len)` with the exact same `len`.
#[no_mangle]
pub extern "C" fn es_alloc(len: usize) -> *mut u8 {
    if len == 0 || len > MAX_FFI_BYTES {
        set_error_str(
            ES_ERR_ALLOC,
            "es_alloc length must lie in 1..=536870912 bytes",
        );
        return core::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(len, FFI_ALIGN) else {
        set_error_str(ES_ERR_ALLOC, "invalid allocation layout");
        return core::ptr::null_mut();
    };
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        set_error_str(ES_ERR_ALLOC, "heap allocator returned null");
    }
    ptr
}

/// Frees an 8-byte-aligned buffer previously allocated by `es_alloc(len)` or returned as an owned
/// output buffer by `es_decode_*` / `es_encode_*` / `es_taup_*`.
///
/// Passing a null pointer or `len == 0` is a safe no-op.
///
/// # Safety
/// When `ptr` is non-null and `len > 0`, `(ptr, len)` must have been allocated by `es_alloc(len)`
/// (or an EarthSnail FFI owned-output function) in the same module instance and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn es_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if ((ptr as usize) & (FFI_ALIGN - 1)) != 0 || len > MAX_FFI_BYTES {
        set_error_str(
            ES_ERR_INVALID_ARG,
            "es_free called with misaligned pointer or invalid length",
        );
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, FFI_ALIGN) {
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

/// Returns a pointer to the thread-local UTF-8 error message buffer (`es_last_error_len()` bytes).
#[no_mangle]
pub extern "C" fn es_last_error_ptr() -> *const u8 {
    LAST_ERROR.with(|cell| unsafe { core::ptr::addr_of!((*cell.get()).buf[0]) })
}

/// Returns the byte length of the last UTF-8 error message (`0` if the last call succeeded).
#[no_mangle]
pub extern "C" fn es_last_error_len() -> usize {
    LAST_ERROR.with(|cell| unsafe { (*cell.get()).len })
}

/// Clears the thread-local error buffer.
#[no_mangle]
pub extern "C" fn es_clear_error() {
    clear_error();
}

// ============================================================================
// 2. Waveform Auto-Sniffing Decode & Lossless Encode
// ============================================================================

/// Auto-detects and decodes any supported waveform payload (`SAC` LE/BE, `SAC Alpha`, `miniSEED v2`
/// Steim-1/2/Int16/Int32/Float32/Float64, `miniSEED v3`, `GeoCSV`, `SLIST`, `TSPAIR`), writing an
/// owned `ESWF` v1 binary buffer to `(*out_buf_ptr, *out_buf_len)`.
///
/// # Safety
/// `input_ptr` must reference `input_len` valid bytes, and `out_buf_ptr` / `out_buf_len` must be
/// valid aligned writable pointers. Free `*out_buf_ptr` with `es_free(*out_buf_ptr, *out_buf_len)`.
#[no_mangle]
pub unsafe extern "C" fn es_decode_waveforms(
    input_ptr: *const u8,
    input_len: usize,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_buf_ptr, 1, false, "null out_buf_ptr") {
        return c;
    }
    if let Err(c) = validate_mut_ptr(out_buf_len, 1, false, "null out_buf_len") {
        return c;
    }
    let bytes = match unsafe { check_slice(input_ptr, input_len, false, "empty waveform input") } {
        Ok(b) => b,
        Err(c) => return c,
    };
    let fmt = detect_waveform_format(bytes);
    let fmt_code = wire::format_code(fmt);
    let stream = match read_waveforms_bytes(bytes) {
        Ok(s) => s,
        Err(e) => return set_earthsnail_error(e),
    };
    let encoded = wire::encode_stream_eswf(&stream, fmt_code);
    unsafe { emit_owned_bytes(&encoded, out_buf_ptr, out_buf_len) }
}

/// Decodes raw 64-byte big-endian Steim-2 frames directly into `n_samples` exact `i32` integers.
///
/// # Safety
/// `frames_ptr` must reference `frames_len` valid bytes (`frames_len % 64 == 0`), and `out_i32_ptr`
/// must reference `n_samples` aligned writable `i32` slots.
#[no_mangle]
pub unsafe extern "C" fn es_decode_steim2_frames(
    frames_ptr: *const u8,
    frames_len: usize,
    n_samples: usize,
    out_i32_ptr: *mut i32,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_i32_ptr, n_samples, false, "null out_i32_ptr") {
        return c;
    }
    let decoded = {
        let frames =
            match unsafe { check_slice(frames_ptr, frames_len, false, "empty Steim-2 input") } {
                Ok(s) => s,
                Err(c) => return c,
            };
        match decode_steim2(frames, n_samples) {
            Ok(v) => v,
            Err(e) => return set_earthsnail_error(e),
        }
    };
    let out = match unsafe { check_slice_mut(out_i32_ptr, n_samples, false, "null out_i32_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out.copy_from_slice(&decoded);
    ES_OK
}

/// Encodes `n_samples` `i32` integers into raw 64-byte big-endian Steim-2 frames (`X0`/`Xn` verified).
///
/// # Safety
/// `samples_i32_ptr` must reference `n_samples` valid aligned `i32` values; `out_buf_ptr` and
/// `out_buf_len` must be valid writable pointers.
#[no_mangle]
pub unsafe extern "C" fn es_encode_steim2_frames(
    samples_i32_ptr: *const i32,
    n_samples: usize,
    d0: i32,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let samples =
        match unsafe { check_slice(samples_i32_ptr, n_samples, false, "empty Steim-2 samples") } {
            Ok(s) => s,
            Err(c) => return c,
        };
    let frames = match encode_steim2(samples, d0) {
        Ok(f) => f,
        Err(e) => return set_earthsnail_error(e),
    };
    unsafe { emit_owned_bytes(&frames, out_buf_ptr, out_buf_len) }
}

/// Encodes a single `f64` waveform trace into SAC format (`format_mode`: `0 = Little-Endian Binary`,
/// `1 = Big-Endian Binary`, `2 = SAC Alphanumeric`).
///
/// # Safety
/// `chan_ptr`, `samples_ptr`, `out_buf_ptr`, and `out_buf_len` must be valid aligned pointers.
#[no_mangle]
pub unsafe extern "C" fn es_encode_sac(
    chan_ptr: *const u8,
    chan_len: usize,
    start_epoch_s: i64,
    start_subsec_ns: i32,
    sampling_rate_hz: f64,
    samples_ptr: *const f64,
    npts: usize,
    format_mode: u32,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let chan_str = match unsafe { check_utf8(chan_ptr, chan_len, true, "chan_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let ch = match parse_channel_or_default(chan_str) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let samples = match unsafe { check_slice(samples_ptr, npts, false, "empty samples") } {
        Ok(s) => s.to_vec(),
        Err(c) => return c,
    };
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return set_error_str(
            ES_ERR_INVALID_ARG,
            "sampling_rate_hz must be finite and > 0",
        );
    }
    let ns = (start_epoch_s as i128) * 1_000_000_000 + (start_subsec_ns as i128);
    let tr = Trace::new(
        TraceStats::new(ch, UtcTimestamp(ns), sampling_rate_hz),
        samples,
    );

    let bytes = match format_mode {
        0 => write_sac_bytes(&tr, SacEndian::Little),
        1 => write_sac_bytes(&tr, SacEndian::Big),
        2 => write_sac_alpha_bytes(&tr),
        _ => {
            return set_error_str(
                ES_ERR_INVALID_ARG,
                "format_mode must be 0 (LE), 1 (BE), or 2 (Alpha)",
            )
        }
    };
    match bytes {
        Ok(b) => unsafe { emit_owned_bytes(&b, out_buf_ptr, out_buf_len) },
        Err(e) => set_earthsnail_error(e),
    }
}

/// Encodes a single `f64` integer-valued trace into SEED v2.4 miniSEED with lossless **Steim-2**
/// compression (`record_len = 512` default when `0`).
///
/// # Safety
/// `chan_ptr`, `samples_ptr`, `out_buf_ptr`, and `out_buf_len` must be valid aligned pointers.
#[no_mangle]
pub unsafe extern "C" fn es_encode_mseed2_steim2(
    chan_ptr: *const u8,
    chan_len: usize,
    start_epoch_s: i64,
    start_subsec_ns: i32,
    sampling_rate_hz: f64,
    samples_ptr: *const f64,
    npts: usize,
    record_len: u32,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let chan_str = match unsafe { check_utf8(chan_ptr, chan_len, true, "chan_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let ch = match parse_channel_or_default(chan_str) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let samples = match unsafe { check_slice(samples_ptr, npts, false, "empty samples") } {
        Ok(s) => s.to_vec(),
        Err(c) => return c,
    };
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return set_error_str(
            ES_ERR_INVALID_ARG,
            "sampling_rate_hz must be finite and > 0",
        );
    }
    let rec_len = if record_len == 0 {
        512
    } else {
        record_len as usize
    };
    let ns = (start_epoch_s as i128) * 1_000_000_000 + (start_subsec_ns as i128);
    let tr = Trace::new(
        TraceStats::new(ch, UtcTimestamp(ns), sampling_rate_hz),
        samples,
    );
    let opts = MseedWriteOptions {
        record_length: rec_len,
        encoding: MseedEncoding::Steim2,
        byte_order: MseedByteOrder::Big,
        quality_indicator: 'D',
    };
    match write_mseed_trace_bytes(&tr, opts) {
        Ok(b) => unsafe { emit_owned_bytes(&b, out_buf_ptr, out_buf_len) },
        Err(e) => set_earthsnail_error(e),
    }
}

/// Encodes a single trace into FDSN **miniSEED v3** with Castagnoli CRC-32C (`encoding_code`:
/// `1 = Int16`, `3 = Int32`, `4 = Float32`, `5 = Float64`, `10 = Steim1`, `11 = Steim2`).
///
/// # Safety
/// `chan_ptr`, `samples_ptr`, `out_buf_ptr`, and `out_buf_len` must be valid aligned pointers.
#[no_mangle]
pub unsafe extern "C" fn es_encode_mseed3(
    chan_ptr: *const u8,
    chan_len: usize,
    start_epoch_s: i64,
    start_subsec_ns: i32,
    sampling_rate_hz: f64,
    samples_ptr: *const f64,
    npts: usize,
    encoding_code: u32,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let chan_str = match unsafe { check_utf8(chan_ptr, chan_len, true, "chan_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let ch = match parse_channel_or_default(chan_str) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let samples = match unsafe { check_slice(samples_ptr, npts, false, "empty samples") } {
        Ok(s) => s.to_vec(),
        Err(c) => return c,
    };
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return set_error_str(
            ES_ERR_INVALID_ARG,
            "sampling_rate_hz must be finite and > 0",
        );
    }
    let enc = match encoding_code {
        1 => MseedEncoding::Int16,
        3 => MseedEncoding::Int32,
        4 => MseedEncoding::Float32,
        0 | 5 => MseedEncoding::Float64,
        10 => MseedEncoding::Steim1,
        11 => MseedEncoding::Steim2,
        _ => {
            return set_error_str(
                ES_ERR_INVALID_ARG,
                "unsupported miniSEED v3 encoding_code (expected 1, 3, 4, 5, 10, or 11)",
            )
        }
    };
    let ns = (start_epoch_s as i128) * 1_000_000_000 + (start_subsec_ns as i128);
    let tr = Trace::new(
        TraceStats::new(ch, UtcTimestamp(ns), sampling_rate_hz),
        samples,
    );
    let opts = Mseed3WriteOptions {
        encoding: enc,
        ..Mseed3WriteOptions::default()
    };
    match write_mseed3_trace_bytes(&tr, &opts) {
        Ok(b) => unsafe { emit_owned_bytes(&b, out_buf_ptr, out_buf_len) },
        Err(e) => set_earthsnail_error(e),
    }
}

// ============================================================================
// 3. Metadata & WGS84 Geodesics
// ============================================================================

/// Solves the exact Karney WGS84 inverse geodesic between `(lat1_deg, lon1_deg)` and
/// `(lat2_deg, lon2_deg)`, writing 5 `f64` values to `out_5_f64`:
/// `[distance_m, gcarc_deg, distance_deg_spherical, azimuth_deg, back_azimuth_deg]`.
///
/// # Safety
/// `out_5_f64` must be a non-null, 8-byte-aligned pointer to at least `5` writable `f64` elements.
#[no_mangle]
pub unsafe extern "C" fn es_wgs84_geodesic(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    out_5_f64: *mut f64,
) -> i32 {
    clear_error();
    let out = match unsafe { check_slice_mut(out_5_f64, 5, false, "null out_5_f64") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let res = match try_gps2dist_azimuth(lat1_deg, lon1_deg, lat2_deg, lon2_deg) {
        Ok(r) => r,
        Err(e) => return set_earthsnail_error(e),
    };
    out[0] = res.distance_m;
    out[1] = res.gcarc_deg;
    out[2] = res.distance_deg_spherical;
    out[3] = res.azimuth_deg;
    out[4] = res.back_azimuth_deg;
    ES_OK
}

/// Parses an FDSN `StationXML` document, selects the channel matching `(chan_ptr, chan_len)` at
/// optional UTC `epoch_s` (`0` = any epoch), writes optional 7-element metadata
/// `[latitude, longitude, elevation_m, depth_m, azimuth_deg, dip_deg, sample_rate_hz]` into
/// `out_meta_7_f64` (when non-null), and evaluates complex `H(f)` in `target_unit`
/// (`0 = DISP`, `1 = VEL`, `2 = ACC`) at `freqs[0..n_freqs]` into interleaved `[re_0, im_0, ...]`
/// in `out_re_im_ptr` (`2 * n_freqs` `f64`s).
///
/// # Safety
/// All non-empty slice pointers must be valid and properly aligned.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_parse_stationxml_eval_response(
    xml_ptr: *const u8,
    xml_len: usize,
    chan_ptr: *const u8,
    chan_len: usize,
    epoch_s: i64,
    target_unit: u32,
    freqs_ptr: *const f64,
    n_freqs: usize,
    out_re_im_ptr: *mut f64,
    out_meta_7_f64: *mut f64,
) -> i32 {
    clear_error();
    let unit = match parse_kinematic_unit(target_unit) {
        Ok(u) => u,
        Err(c) => return c,
    };
    if n_freqs > 0 {
        if let Err(c) = validate_mut_ptr(out_re_im_ptr, 2 * n_freqs, false, "null out_re_im_ptr") {
            return c;
        }
    }
    if !out_meta_7_f64.is_null() {
        if let Err(c) = validate_mut_ptr(out_meta_7_f64, 7, false, "misaligned out_meta_7_f64") {
            return c;
        }
    }

    let inv = {
        let xml_bytes = match unsafe { check_slice(xml_ptr, xml_len, false, "empty StationXML") } {
            Ok(b) => b,
            Err(c) => return c,
        };
        match read_stationxml_bytes(xml_bytes) {
            Ok(i) => i,
            Err(e) => return set_earthsnail_error(e),
        }
    };
    let chan_str = match unsafe { check_utf8(chan_ptr, chan_len, true, "chan_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let Some(ch_meta) = find_stationxml_channel(&inv, chan_str, epoch_s) else {
        return set_error_str(ES_ERR_INVALID_ARG, "no matching channel in StationXML");
    };

    let freqs_vec = if n_freqs > 0 {
        match unsafe { check_slice(freqs_ptr, n_freqs, false, "null freqs_ptr") } {
            Ok(f) => f.to_vec(),
            Err(c) => return c,
        }
    } else {
        Vec::new()
    };

    if !out_meta_7_f64.is_null() {
        let meta_out = match unsafe { check_slice_mut(out_meta_7_f64, 7, false, "out_meta_7_f64") }
        {
            Ok(s) => s,
            Err(c) => return c,
        };
        meta_out[0] = ch_meta.latitude;
        meta_out[1] = ch_meta.longitude;
        meta_out[2] = ch_meta.elevation_m;
        meta_out[3] = ch_meta.depth_m;
        meta_out[4] = ch_meta.azimuth_deg.unwrap_or(f64::NAN);
        meta_out[5] = ch_meta.dip_deg.unwrap_or(f64::NAN);
        meta_out[6] = ch_meta.sample_rate_hz.unwrap_or(f64::NAN);
    }

    if n_freqs > 0 {
        let Some(resp) = &ch_meta.response else {
            return set_error_str(ES_ERR_INVALID_ARG, "selected channel has no Response stage");
        };
        let re_im =
            match unsafe { check_slice_mut(out_re_im_ptr, 2 * n_freqs, false, "out_re_im_ptr") } {
                Ok(s) => s,
                Err(c) => return c,
            };
        for (i, &f) in freqs_vec.iter().enumerate() {
            let h = resp.eval_freq_kinematic(f, unit);
            re_im[2 * i] = h.re;
            re_im[2 * i + 1] = h.im;
        }
    }
    ES_OK
}

/// Parses a `SACPZ` pole-zero file, writes `CONSTANT` to `*out_constant_f64` (when non-null), and
/// evaluates complex `H(f)` in `target_unit` (`0 = DISP`, `1 = VEL`, `2 = ACC`) at
/// `freqs[0..n_freqs]` into interleaved `[re_0, im_0, ...]` in `out_re_im_ptr`.
///
/// # Safety
/// All non-empty slice pointers must be valid and properly aligned.
#[no_mangle]
pub unsafe extern "C" fn es_parse_sacpz(
    pz_ptr: *const u8,
    pz_len: usize,
    target_unit: u32,
    freqs_ptr: *const f64,
    n_freqs: usize,
    out_re_im_ptr: *mut f64,
    out_constant_f64: *mut f64,
) -> i32 {
    clear_error();
    let unit = match parse_kinematic_unit(target_unit) {
        Ok(u) => u,
        Err(c) => return c,
    };
    if n_freqs > 0 {
        if let Err(c) = validate_mut_ptr(out_re_im_ptr, 2 * n_freqs, false, "null out_re_im_ptr") {
            return c;
        }
    }
    if !out_constant_f64.is_null() {
        if let Err(c) = validate_mut_ptr(out_constant_f64, 1, false, "misaligned out_constant_f64")
        {
            return c;
        }
    }

    let pz = {
        let bytes = match unsafe { check_slice(pz_ptr, pz_len, false, "empty SACPZ input") } {
            Ok(b) => b,
            Err(c) => return c,
        };
        match read_sacpz_bytes(bytes) {
            Ok(rec) => rec,
            Err(e) => return set_earthsnail_error(e),
        }
    };
    let freqs_vec = if n_freqs > 0 {
        match unsafe { check_slice(freqs_ptr, n_freqs, false, "null freqs_ptr") } {
            Ok(f) => f.to_vec(),
            Err(c) => return c,
        }
    } else {
        Vec::new()
    };

    if !out_constant_f64.is_null() {
        unsafe { out_constant_f64.write(pz.constant) };
    }
    if n_freqs > 0 {
        let re_im =
            match unsafe { check_slice_mut(out_re_im_ptr, 2 * n_freqs, false, "out_re_im_ptr") } {
                Ok(s) => s,
                Err(c) => return c,
            };
        for (i, &f) in freqs_vec.iter().enumerate() {
            let h = pz.response.eval_freq_kinematic(f, unit);
            re_im[2 * i] = h.re;
            re_im[2 * i + 1] = h.im;
        }
    }
    ES_OK
}

// ============================================================================
// 4. Real-Time Digital Signal Processing (DSP)
// ============================================================================

/// In-place waveform preprocessing (`mode_flags` bitmask: `1 = demean/rmean`,
/// `2 = detrend_linear/rtrend`, `4 = taper` with `taper_type`: `0 = Hann`, `1 = Hamming`,
/// `2 = Cosine`, `3 = SacCosine`, `4 = Blackman`, `5 = Kaiser(8.6)` and half-window fraction
/// `taper_max_pct` in `[0.0, 0.5]`).
///
/// # Safety
/// `data_ptr` must reference `npts` valid, 8-byte-aligned writable `f64` samples.
#[no_mangle]
pub unsafe extern "C" fn es_demean_detrend_taper(
    data_ptr: *mut f64,
    npts: usize,
    mode_flags: u32,
    taper_type: u32,
    taper_max_pct: f64,
) -> i32 {
    clear_error();
    let data = match unsafe { check_slice_mut(data_ptr, npts, false, "empty data_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    if (mode_flags & 1) != 0 {
        demean_in_place(data);
    }
    if (mode_flags & 2) != 0 {
        detrend_linear_in_place(data);
    }
    if (mode_flags & 4) != 0 {
        let tt =
            match taper_type {
                0 => TaperType::Hann,
                1 => TaperType::Hamming,
                2 => TaperType::Cosine,
                3 => TaperType::SacCosine,
                4 => TaperType::Blackman,
                5 => TaperType::Kaiser { beta: 8.6 },
                _ => return set_error_str(
                    ES_ERR_INVALID_ARG,
                    "taper_type must be 0..=5 (Hann, Hamming, Cosine, SacCosine, Blackman, Kaiser)",
                ),
            };
        if let Err(e) = taper_in_place(data, taper_max_pct, tt, TaperSide::Both) {
            return set_earthsnail_error(e);
        }
    }
    ES_OK
}

/// Synthesizes and executes a cascaded Second-Order Sections (`SOS`) IIR filter:
/// - `family`: `0 = Butterworth`, `1 = Chebyshev1(ripple_db)`, `2 = Chebyshev2(ripple_db)`,
///   `3 = Bessel(Phase)`, `4 = Bessel(Delay)`, `5 = Bessel(Mag)`
/// - `band_type`: `0 = Bandpass(f_low, f_high)`, `1 = Lowpass(f_high_or_low)`,
///   `2 = Highpass(f_low)`, `3 = Bandstop(f_low, f_high)`
/// - `zero_phase_mode`: `0 = Causal 1-pass (sosfilt)`, `1 = ObsPy/SAC 2-pass zero-phase`,
///   `2 = SciPy odd-reflection steady-state zero-phase (sosfiltfilt)`.
///
/// Safe even when `input_ptr == out_ptr` (alias-safe read-compute-write discipline).
///
/// # Safety
/// `input_ptr` and `out_ptr` must reference `npts` valid, 8-byte-aligned `f64` elements.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_sos_filter(
    input_ptr: *const f64,
    npts: usize,
    sampling_rate_hz: f64,
    family: u32,
    band_type: u32,
    order: u32,
    f_low_hz: f64,
    f_high_hz: f64,
    ripple_db: f64,
    zero_phase_mode: u32,
    out_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_ptr, npts, false, "null out_ptr") {
        return c;
    }
    let fam = match family {
        0 => FilterFamily::Butterworth,
        1 => FilterFamily::Chebyshev1 {
            rp_db: if ripple_db > 0.0 { ripple_db } else { 1.0 },
        },
        2 => FilterFamily::Chebyshev2 {
            rs_db: if ripple_db > 0.0 { ripple_db } else { 40.0 },
        },
        3 => FilterFamily::Bessel {
            norm: BesselNorm::Phase,
        },
        4 => FilterFamily::Bessel {
            norm: BesselNorm::Delay,
        },
        5 => FilterFamily::Bessel {
            norm: BesselNorm::Mag,
        },
        _ => return set_error_str(ES_ERR_INVALID_ARG, "filter family must be 0..=5"),
    };
    let band = match band_type {
        0 => FilterBand::Bandpass(f_low_hz, f_high_hz),
        1 => FilterBand::Lowpass(if f_high_hz > 0.0 && f_low_hz <= 0.0 {
            f_high_hz
        } else {
            f_low_hz
        }),
        2 => FilterBand::Highpass(f_low_hz),
        3 => FilterBand::Bandstop(f_low_hz, f_high_hz),
        _ => return set_error_str(ES_ERR_INVALID_ARG, "band_type must be 0..=3"),
    };
    let sos = match design_iir_sos(order as usize, band, sampling_rate_hz, fam) {
        Ok(s) => s,
        Err(e) => return set_earthsnail_error(e),
    };

    let filtered = {
        let input = match unsafe { check_slice(input_ptr, npts, false, "empty input_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        match zero_phase_mode {
            0 => sosfilt(&sos, input),
            1 => sosfilt_zerophase_obspy(&sos, input),
            2 => match sosfiltfilt(&sos, input) {
                Ok(v) => v,
                Err(e) => return set_earthsnail_error(e),
            },
            _ => return set_error_str(ES_ERR_INVALID_ARG, "zero_phase_mode must be 0, 1, or 2"),
        }
    };

    let out = match unsafe { check_slice_mut(out_ptr, npts, false, "null out_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out.copy_from_slice(&filtered);
    ES_OK
}

/// Spectral instrument response removal (`StationXML` or `SACPZ`) converting raw digital counts to
/// physical ground motion (`target_unit`: `0 = DISP (m)`, `1 = VEL (m/s)`, `2 = ACC (m/s^2)`).
///
/// - `meta_format`: `0 = StationXML`, `1 = SACPZ`, `2 = Auto-sniff ('<' -> StationXML else SACPZ)`.
/// - `pre_filt_4_f64`: optional pointer to `[f1, f2, f3, f4]` Hz 4-corner cosine taper (`null` = none).
/// - `water_level_db`: water-level floor in dB (`< 0.0` disables water-level clamping).
///
/// # Safety
/// All non-null pointers must be valid and 8-byte aligned where required.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_remove_response(
    input_ptr: *const f64,
    npts: usize,
    sampling_rate_hz: f64,
    meta_ptr: *const u8,
    meta_len: usize,
    meta_format: u32,
    chan_ptr: *const u8,
    chan_len: usize,
    epoch_s: i64,
    target_unit: u32,
    pre_filt_4_f64: *const f64,
    water_level_db: f64,
    out_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_ptr, npts, false, "null out_ptr") {
        return c;
    }
    let unit = match parse_kinematic_unit(target_unit) {
        Ok(u) => u,
        Err(c) => return c,
    };
    let pre_filt = if pre_filt_4_f64.is_null() {
        None
    } else {
        let pf = match unsafe { check_slice(pre_filt_4_f64, 4, false, "pre_filt_4_f64") } {
            Ok(s) => [s[0], s[1], s[2], s[3]],
            Err(c) => return c,
        };
        Some(pf)
    };

    let response: ChannelResponse = {
        let meta_bytes = match unsafe { check_slice(meta_ptr, meta_len, false, "empty meta_ptr") } {
            Ok(b) => b,
            Err(c) => return c,
        };
        let chan_str = match unsafe { check_utf8(chan_ptr, chan_len, true, "chan_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let is_xml = match meta_format {
            0 => true,
            1 => false,
            2 => meta_bytes
                .iter()
                .find(|&&b| !b.is_ascii_whitespace())
                .is_some_and(|&b| b == b'<'),
            _ => return set_error_str(ES_ERR_INVALID_ARG, "meta_format must be 0, 1, or 2"),
        };
        if is_xml {
            let inv = match read_stationxml_bytes(meta_bytes) {
                Ok(i) => i,
                Err(e) => return set_earthsnail_error(e),
            };
            let Some(ch) = find_stationxml_channel(&inv, chan_str, epoch_s) else {
                return set_error_str(ES_ERR_INVALID_ARG, "channel not found in StationXML");
            };
            let Some(resp) = ch.response.clone() else {
                return set_error_str(ES_ERR_INVALID_ARG, "StationXML channel has no Response");
            };
            resp
        } else {
            match read_sacpz_bytes(meta_bytes) {
                Ok(pz) => pz.response,
                Err(e) => return set_earthsnail_error(e),
            }
        }
    };

    let cfg = ResponseRemovalConfig {
        output: unit,
        pre_filt_hz: pre_filt,
        water_level_db: (water_level_db >= 0.0 && water_level_db.is_finite())
            .then_some(water_level_db),
        zero_mean: true,
        taper: true,
        taper_fraction: 0.05,
        nfft_pow2: false,
        rescale_to_overall_sensitivity: false,
    };

    let deconvolved = {
        let input = match unsafe { check_slice(input_ptr, npts, false, "empty input_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        match remove_response(input, sampling_rate_hz, &response, &cfg) {
            Ok(v) => v,
            Err(e) => return set_earthsnail_error(e),
        }
    };

    let out = match unsafe { check_slice_mut(out_ptr, npts, false, "null out_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out.copy_from_slice(&deconvolved);
    ES_OK
}

/// Computes Welch one-sided Power Spectral Density (`welch_psd`, Hann window, linear segment
/// detrend), writing up to `max_bins` frequency/PSD pairs and storing total bin count in
/// `*out_bins_written`.
///
/// If `max_bins == 0` and `out_freqs_ptr` / `out_psd_ptr` are null, writes the exact output bin
/// count (`nperseg / 2 + 1`) to `*out_bins_written` and returns `ES_OK`.
///
/// # Safety
/// All non-null pointers must be valid and 8-byte aligned.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_rfft_psd(
    input_ptr: *const f64,
    npts: usize,
    sampling_rate_hz: f64,
    nperseg: usize,
    noverlap: usize,
    out_freqs_ptr: *mut f64,
    out_psd_ptr: *mut f64,
    max_bins: usize,
    out_bins_written: *mut usize,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_bins_written, 1, false, "null out_bins_written") {
        return c;
    }
    let seg = if nperseg == 0 {
        npts.min(1024)
    } else {
        nperseg.min(npts)
    };
    if max_bins > 0 {
        if let Err(c) = validate_mut_ptr(out_freqs_ptr, max_bins, false, "null out_freqs_ptr") {
            return c;
        }
        if let Err(c) = validate_mut_ptr(out_psd_ptr, max_bins, false, "null out_psd_ptr") {
            return c;
        }
    }

    let est = {
        let input = match unsafe { check_slice(input_ptr, npts, false, "empty input_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let ov = if noverlap >= seg { seg / 2 } else { noverlap };
        let cfg = WelchConfig {
            nperseg: seg,
            noverlap: ov,
            nfft: Some(seg),
            detrend: WelchDetrend::Linear,
            window: WelchWindow::Hann,
        };
        match welch_psd(input, sampling_rate_hz, &cfg) {
            Ok(p) => p,
            Err(e) => return set_earthsnail_error(e),
        }
    };

    let total_bins = est.freqs_hz.len();
    if max_bins == 0 {
        unsafe { out_bins_written.write(total_bins) };
        return ES_OK;
    }
    let write_n = total_bins.min(max_bins);
    let out_f = match unsafe { check_slice_mut(out_freqs_ptr, write_n, true, "out_freqs_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out_f.copy_from_slice(&est.freqs_hz[..write_n]);

    let out_p = match unsafe { check_slice_mut(out_psd_ptr, write_n, true, "out_psd_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out_p.copy_from_slice(&est.psd[..write_n]);
    unsafe { out_bins_written.write(write_n) };
    ES_OK
}

/// Computes a Hann-windowed STFT power spectrogram in dB, returning an owned 8-byte-aligned binary
/// buffer `b"ESSP"` (`version=1u32`, `n_frames: u32`, `n_bins: u32`, `dt_frame_s: f64`, `df_hz: f64`,
/// followed by `n_frames * n_bins` `f64` values in dB).
///
/// # Safety
/// `input_ptr`, `out_buf_ptr`, and `out_buf_len` must be valid aligned pointers.
#[no_mangle]
pub unsafe extern "C" fn es_stft_spectrogram(
    input_ptr: *const f64,
    npts: usize,
    sampling_rate_hz: f64,
    win_len: usize,
    hop_len: usize,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return set_error_str(ES_ERR_INVALID_ARG, "sampling_rate_hz must be > 0");
    }
    let input = match unsafe { check_slice(input_ptr, npts, false, "empty input_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let wlen = win_len.clamp(8, npts);
    let hop = hop_len.max(1);
    if npts < wlen {
        return set_error_str(ES_ERR_INVALID_ARG, "npts must be >= 8 for spectrogram");
    }
    let win = hann_window(wlen);
    let win_energy: f64 = win.iter().map(|&w| w * w).sum::<f64>().max(1e-30);
    let scale = 2.0 / (sampling_rate_hz * win_energy);
    let n_bins = wlen / 2 + 1;
    let n_frames = (npts - wlen) / hop + 1;
    let freqs = match rfftfreq(wlen, 1.0 / sampling_rate_hz) {
        Ok(f) => f,
        Err(e) => return set_earthsnail_error(e),
    };
    let df_hz = if freqs.len() >= 2 {
        freqs[1] - freqs[0]
    } else {
        sampling_rate_hz * 0.5
    };
    let dt_frame_s = (hop as f64) / sampling_rate_hz;

    let mut out = Vec::with_capacity(32 + n_frames * n_bins * 8);
    out.extend_from_slice(b"ESSP");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(n_frames as u32).to_le_bytes());
    out.extend_from_slice(&(n_bins as u32).to_le_bytes());
    out.extend_from_slice(&dt_frame_s.to_le_bytes());
    out.extend_from_slice(&df_hz.to_le_bytes());

    let mut seg = vec![0.0_f64; wlen];
    let mut spec = Vec::with_capacity(n_bins);
    let mut scratch = ScratchWorkspace::new();

    for f_idx in 0..n_frames {
        let start = f_idx * hop;
        seg.copy_from_slice(&input[start..start + wlen]);
        demean_in_place(&mut seg);
        for (s, &w) in seg.iter_mut().zip(win.iter()) {
            *s *= w;
        }
        if let Err(e) = rfft_with_scratch(&seg, wlen, &mut spec, &mut scratch) {
            return set_earthsnail_error(e);
        }
        for bin in &spec {
            let pwr = (bin.norm_sqr() * scale).max(1e-38);
            let db = 10.0 * earthsnail_core::math::log10(pwr);
            out.extend_from_slice(&db.to_le_bytes());
        }
    }

    unsafe { emit_owned_bytes(&out, out_buf_ptr, out_buf_len) }
}

/// Evaluates Peterson (1993) Global New Low & High Noise Models (`NLNM` & `NHNM`) in
/// `dB rel. 1 (m/s^2)^2/Hz` at `periods_ptr[0..n_periods]`.
///
/// # Safety
/// `periods_ptr`, `out_nlnm_db_ptr`, and `out_nhnm_db_ptr` must reference `n_periods` valid aligned `f64`s.
#[no_mangle]
pub unsafe extern "C" fn es_nlnm_nhnm(
    periods_ptr: *const f64,
    n_periods: usize,
    out_nlnm_db_ptr: *mut f64,
    out_nhnm_db_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_nlnm_db_ptr, n_periods, false, "null out_nlnm_db_ptr") {
        return c;
    }
    if let Err(c) = validate_mut_ptr(out_nhnm_db_ptr, n_periods, false, "null out_nhnm_db_ptr") {
        return c;
    }
    let periods = match unsafe { check_slice(periods_ptr, n_periods, false, "empty periods_ptr") } {
        Ok(p) => p.to_vec(),
        Err(c) => return c,
    };
    let mut nlnm = Vec::with_capacity(n_periods);
    let mut nhnm = Vec::with_capacity(n_periods);
    for &p in &periods {
        match (eval_nlnm_db(p), eval_nhnm_db(p)) {
            (Ok(lo), Ok(hi)) => {
                nlnm.push(lo);
                nhnm.push(hi);
            }
            (Err(e), _) | (_, Err(e)) => return set_earthsnail_error(e),
        }
    }
    let out_lo =
        match unsafe { check_slice_mut(out_nlnm_db_ptr, n_periods, false, "out_nlnm_db_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
    out_lo.copy_from_slice(&nlnm);
    let out_hi =
        match unsafe { check_slice_mut(out_nhnm_db_ptr, n_periods, false, "out_nhnm_db_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
    out_hi.copy_from_slice(&nhnm);
    ES_OK
}

/// Computes the Marple FFT Hilbert analytic-signal amplitude envelope (`|z(t)|`).
///
/// # Safety
/// `input_ptr` and `out_ptr` must reference `npts` valid, 8-byte-aligned `f64` elements.
#[no_mangle]
pub unsafe extern "C" fn es_hilbert_envelope(
    input_ptr: *const f64,
    npts: usize,
    out_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_ptr, npts, false, "null out_ptr") {
        return c;
    }
    let env = {
        let input = match unsafe { check_slice(input_ptr, npts, false, "empty input_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        match envelope(input) {
            Ok(v) => v,
            Err(e) => return set_earthsnail_error(e),
        }
    };
    let out = match unsafe { check_slice_mut(out_ptr, npts, false, "null out_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out.copy_from_slice(&env);
    ES_OK
}

/// Automatic seismic phase arrival picker supporting:
/// - `algorithm = 0`: Classic `STA/LTA` (`param1 = nsta_samples`, `param2 = nlta_samples`, `param3 = thr_on`)
/// - `algorithm = 1`: Recursive `STA/LTA` (`param1 = nsta_samples`, `param2 = nlta_samples`, `param3 = thr_on`)
/// - `algorithm = 2`: Baer & Kradolfer `pk_baer` (`param1 = thr1`, `param2 = thr2`, `param3 = tupevent`)
/// - `algorithm = 3`: Maeda `AIC` global minimum picker
/// - `algorithm = 4`: 3-Point Discrete `Morse` Causal-Deferral Glitch-Immune Picker
///   (`param1 = nsta`, `param2 = nlta`, `param3 = thr_on`; pass `param1 <= 0` for defaults).
///
/// Writes 0-based onset sample (`-1` if unpicked) to `*out_pick_index`, peak CF/ratio metric to
/// `*out_peak_metric`, and optional length-`npts` characteristic/persistence curve to `out_cf_ptr`
/// (when non-null).
///
/// # Safety
/// `input_ptr`, `out_pick_index`, `out_peak_metric`, and (if non-null) `out_cf_ptr` must be valid aligned pointers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_pick_arrivals(
    input_ptr: *const f64,
    npts: usize,
    sampling_rate_hz: f64,
    algorithm: u32,
    param1: f64,
    param2: f64,
    param3: f64,
    out_pick_index: *mut i64,
    out_peak_metric: *mut f64,
    out_cf_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_pick_index, 1, false, "null out_pick_index") {
        return c;
    }
    if let Err(c) = validate_mut_ptr(out_peak_metric, 1, false, "null out_peak_metric") {
        return c;
    }
    if !out_cf_ptr.is_null() {
        if let Err(c) = validate_mut_ptr(out_cf_ptr, npts, false, "misaligned out_cf_ptr") {
            return c;
        }
    }

    let (pick_idx, peak_val, cf_vec): (Option<usize>, f64, Vec<f64>) = {
        let data = match unsafe { check_slice(input_ptr, npts, false, "empty input_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let fs = if sampling_rate_hz.is_finite() && sampling_rate_hz > 0.0 {
            sampling_rate_hz
        } else {
            100.0
        };
        match algorithm {
            0 | 1 => {
                let nsta = if param1 > 0.0 {
                    param1 as usize
                } else {
                    (fs * 1.0).round().max(1.0) as usize
                };
                let nlta = if param2 > 0.0 {
                    param2 as usize
                } else {
                    (fs * 10.0).round().max((nsta + 1) as f64) as usize
                };
                let thr_on = if param3 > 0.0 { param3 } else { 4.0 };
                let cf = if algorithm == 0 {
                    match classic_sta_lta(data, nsta, nlta) {
                        Ok(v) => v,
                        Err(e) => return set_earthsnail_error(e),
                    }
                } else {
                    match recursive_sta_lta(data, nsta, nlta) {
                        Ok(v) => v,
                        Err(e) => return set_earthsnail_error(e),
                    }
                };
                let peak = cf.iter().copied().fold(0.0_f64, f64::max);
                let trigs = trigger_onset(&cf, thr_on, (0.5 * thr_on).min(1.5));
                let first = trigs.first().map(|t| t[0]);
                (first, peak, cf)
            }
            2 => {
                let samp_int = (fs.round() as usize).max(1);
                let preset_len = (npts / 10).clamp(4, npts.saturating_sub(2).max(4));
                let params = BaerPickParams {
                    samp_int,
                    tdownmax: (samp_int / 5).max(2),
                    tupevent: if param3 > 0.0 {
                        param3 as usize
                    } else {
                        (samp_int / 2).max(4)
                    },
                    thr1: if param1 > 0.0 { param1 } else { 7.0 },
                    thr2: if param2 > 0.0 { param2 } else { 12.0 },
                    preset_len,
                    p_dur: samp_int,
                };
                let res = match pk_baer(data, &params) {
                    Ok(r) => r,
                    Err(e) => return set_earthsnail_error(e),
                };
                let mut cf = res.characteristic_function;
                if cf.len() < npts {
                    let tail = cf.last().copied().unwrap_or(0.0);
                    cf.resize(npts, tail);
                }
                let peak = cf.iter().copied().fold(0.0_f64, f64::max);
                (res.pick_sample, peak, cf)
            }
            3 => {
                let pick = aic_pick(data);
                let cf = aic_simple(data);
                let min_aic = cf.iter().copied().fold(f64::INFINITY, f64::min);
                (pick, min_aic, cf)
            }
            4 => {
                let mut cfg = MorsePickConfig::default();
                if param1 > 0.0 {
                    cfg.nsta = param1 as usize;
                }
                if param2 > 0.0 {
                    cfg.nlta = param2 as usize;
                }
                if param3 > 0.0 {
                    cfg.thr_on = param3;
                }
                let res = match morse_pick(data, &cfg) {
                    Ok(r) => r,
                    Err(e) => return set_earthsnail_error(e),
                };
                (res.pick_sample, res.peak_ratio, res.persistence_envelope)
            }
            _ => {
                return set_error_str(
                    ES_ERR_INVALID_ARG,
                    "picker algorithm must be 0 (STA/LTA), 1 (RecSTA/LTA), 2 (Baer), 3 (AIC), or 4 (Morse)",
                )
            }
        }
    };

    unsafe {
        out_pick_index.write(pick_idx.map_or(-1_i64, |idx| idx as i64));
        out_peak_metric.write(peak_val);
    }
    if !out_cf_ptr.is_null() {
        let out_cf = match unsafe { check_slice_mut(out_cf_ptr, npts, false, "out_cf_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        out_cf.copy_from_slice(&cf_vec[..npts]);
    }
    ES_OK
}

/// 3-component coordinate rotation (`mode`: `0 = ZNE -> ZRT`, `1 = ZRT -> ZNE`, `2 = ZNE -> LQT`,
/// `3 = LQT -> ZNE`, `4 = ZNE -> STS-2 UVW`).
///
/// # Safety
/// Input and output pointers must reference `npts` valid, 8-byte-aligned `f64` elements.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_rotate_3c(
    c1_ptr: *const f64,
    c2_ptr: *const f64,
    c3_ptr: *const f64,
    npts: usize,
    mode: u32,
    back_azimuth_deg: f64,
    incidence_deg: f64,
    out1_ptr: *mut f64,
    out2_ptr: *mut f64,
    out3_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out1_ptr, npts, false, "null out1_ptr") {
        return c;
    }
    if let Err(c) = validate_mut_ptr(out2_ptr, npts, false, "null out2_ptr") {
        return c;
    }
    if let Err(c) = validate_mut_ptr(out3_ptr, npts, false, "null out3_ptr") {
        return c;
    }

    let (r1, r2, r3) = {
        let c1 = match unsafe { check_slice(c1_ptr, npts, false, "empty c1_ptr") } {
            Ok(s) => s,
            Err(code) => return code,
        };
        let c2 = match unsafe { check_slice(c2_ptr, npts, false, "empty c2_ptr") } {
            Ok(s) => s,
            Err(code) => return code,
        };
        let c3 = match unsafe { check_slice(c3_ptr, npts, false, "empty c3_ptr") } {
            Ok(s) => s,
            Err(code) => return code,
        };
        match mode {
            0 => match rotate_ne_rt(c2, c3, back_azimuth_deg) {
                Ok((r, t)) => (c1.to_vec(), r, t),
                Err(e) => return set_earthsnail_error(e),
            },
            1 => match rotate_rt_ne(c2, c3, back_azimuth_deg) {
                Ok((n, e)) => (c1.to_vec(), n, e),
                Err(err) => return set_earthsnail_error(err),
            },
            2 => match rotate_zne_lqt(c1, c2, c3, back_azimuth_deg, incidence_deg) {
                Ok(t) => t,
                Err(e) => return set_earthsnail_error(e),
            },
            3 => match rotate_lqt_zne(c1, c2, c3, back_azimuth_deg, incidence_deg) {
                Ok(t) => t,
                Err(e) => return set_earthsnail_error(e),
            },
            4 => match rotate_zne_uvw(c1, c2, c3) {
                Ok(t) => t,
                Err(e) => return set_earthsnail_error(e),
            },
            _ => return set_error_str(ES_ERR_INVALID_ARG, "rotate mode must be 0..=4"),
        }
    };

    let o1 = match unsafe { check_slice_mut(out1_ptr, npts, false, "out1_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    o1.copy_from_slice(&r1);
    let o2 = match unsafe { check_slice_mut(out2_ptr, npts, false, "out2_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    o2.copy_from_slice(&r2);
    let o3 = match unsafe { check_slice_mut(out3_ptr, npts, false, "out3_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    o3.copy_from_slice(&r3);
    ES_OK
}

/// Computes Flinn (1965) 3C covariance polarization metrics, writing 7 `f64`s to `out_7_f64`:
/// `[azimuth_deg, incidence_deg, rectilinearity, planarity, eval0, eval1, eval2]`.
///
/// # Safety
/// `z_ptr`, `n_ptr`, `e_ptr` (`npts` `f64`s) and `out_7_f64` (`7` `f64`s) must be valid aligned pointers.
#[no_mangle]
pub unsafe extern "C" fn es_flinn_polarization(
    z_ptr: *const f64,
    n_ptr: *const f64,
    e_ptr: *const f64,
    npts: usize,
    noise_thres: f64,
    out_7_f64: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_7_f64, 7, false, "null out_7_f64") {
        return c;
    }
    let pol = {
        let z = match unsafe { check_slice(z_ptr, npts, false, "empty z_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let n = match unsafe { check_slice(n_ptr, npts, false, "empty n_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let e = match unsafe { check_slice(e_ptr, npts, false, "empty e_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        match flinn_polarization(z, n, e, noise_thres) {
            Ok(p) => p,
            Err(err) => return set_earthsnail_error(err),
        }
    };
    let out = match unsafe { check_slice_mut(out_7_f64, 7, false, "out_7_f64") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out[0] = pol.azimuth_deg;
    out[1] = pol.incidence_deg;
    out[2] = pol.rectilinearity;
    out[3] = pol.planarity;
    out[4] = pol.eigenvalues[0];
    out[5] = pol.eigenvalues[1];
    out[6] = pol.eigenvalues[2];
    ES_OK
}

/// Computes Nigam & Jennings (1969) damped SDOF response spectra (`PSA`, and optional `PSV` & `SD`)
/// for ground acceleration `acc_ptr[0..npts]` (`m/s^2`) at oscillator periods `periods_ptr[0..n_periods]`.
///
/// # Safety
/// Non-null pointers must be valid and 8-byte aligned.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_response_spectrum(
    acc_ptr: *const f64,
    npts: usize,
    dt_s: f64,
    periods_ptr: *const f64,
    n_periods: usize,
    damping_ratio: f64,
    out_psa_ptr: *mut f64,
    out_psv_ptr: *mut f64,
    out_sd_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_psa_ptr, n_periods, false, "null out_psa_ptr") {
        return c;
    }
    if !out_psv_ptr.is_null() {
        if let Err(c) = validate_mut_ptr(out_psv_ptr, n_periods, false, "misaligned out_psv_ptr") {
            return c;
        }
    }
    if !out_sd_ptr.is_null() {
        if let Err(c) = validate_mut_ptr(out_sd_ptr, n_periods, false, "misaligned out_sd_ptr") {
            return c;
        }
    }

    let spec = {
        let acc = match unsafe { check_slice(acc_ptr, npts, false, "empty acc_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let periods = match unsafe { check_slice(periods_ptr, n_periods, false, "empty periods") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        match nigam_jennings_response_spectra(acc, dt_s, periods, damping_ratio) {
            Ok(r) => r,
            Err(e) => return set_earthsnail_error(e),
        }
    };

    let out_psa = match unsafe { check_slice_mut(out_psa_ptr, n_periods, false, "out_psa_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out_psa.copy_from_slice(&spec.psa_m_s2);

    if !out_psv_ptr.is_null() {
        let out_psv = match unsafe { check_slice_mut(out_psv_ptr, n_periods, false, "out_psv_ptr") }
        {
            Ok(s) => s,
            Err(c) => return c,
        };
        out_psv.copy_from_slice(&spec.psv_m_s);
    }
    if !out_sd_ptr.is_null() {
        let out_sd = match unsafe { check_slice_mut(out_sd_ptr, n_periods, false, "out_sd_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        out_sd.copy_from_slice(&spec.sd_m);
    }
    ES_OK
}

/// Synthesizes or decomposes a seismic moment tensor into `ISO + DC + CLVD` + principal axes +
/// fault nodal planes + optional lower-hemisphere equal-area beachball radiation grid:
///
/// - `input_mode = 0`: `mt_or_sdr_6_f64` = `[m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]` in `N*m`.
/// - `input_mode = 1`: `mt_or_sdr_6_f64` = `[strike_deg, dip_deg, rake_deg, m0_nm, iso_pct, clvd_pct]`
///   (constructs exact moment tensor with prescribed strike/dip/rake orientation and `ISO`/`CLVD` mix).
/// - `out_20_f64`:
///   `[0] iso_pct`, `[1] dc_pct`, `[2] clvd_pct`, `[3] m0_nm`, `[4] mw`,
///   `[5..8] strike1, dip1, rake1`, `[8..11] strike2, dip2, rake2`,
///   `[11..14] t_az, t_plunge, t_val`, `[14..17] p_az, p_plunge, p_val`, `[17..20] n_az, n_plunge, n_val`.
/// - If `grid_size > 0` and `out_beachball_ptr` is non-null, writes `grid_size * grid_size` P-wave
///   radiation values (`f64::NAN` outside the focal circle, signed amplitude inside).
///
/// # Safety
/// `mt_or_sdr_6_f64`, `out_20_f64`, and (if `grid_size > 0`) `out_beachball_ptr` must be valid aligned pointers.
#[no_mangle]
pub unsafe extern "C" fn es_moment_tensor_decompose(
    mt_or_sdr_6_f64: *const f64,
    input_mode: u32,
    out_20_f64: *mut f64,
    grid_size: usize,
    out_beachball_ptr: *mut f64,
) -> i32 {
    clear_error();
    if let Err(c) = validate_mut_ptr(out_20_f64, 20, false, "null out_20_f64") {
        return c;
    }
    if grid_size > 0 {
        let grid_len = match grid_size.checked_mul(grid_size) {
            Some(n) => n,
            None => return set_error_str(ES_ERR_ALLOC, "grid_size overflow"),
        };
        if let Err(c) =
            validate_mut_ptr(out_beachball_ptr, grid_len, false, "null out_beachball_ptr")
        {
            return c;
        }
    }

    let inp = match unsafe { check_slice(mt_or_sdr_6_f64, 6, false, "null mt_or_sdr_6_f64") } {
        Ok(s) => [s[0], s[1], s[2], s[3], s[4], s[5]],
        Err(c) => return c,
    };

    let mt = match input_mode {
        0 => MomentTensor6::new(inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]),
        1 => {
            let plane = NodalPlane {
                strike_deg: inp[0],
                dip_deg: inp[1],
                rake_deg: inp[2],
            };
            let m0 = if inp[3] > 0.0 && inp[3].is_finite() {
                inp[3]
            } else {
                1.0e20
            };
            let base_dc = match strike_dip_rake_to_mt(plane, m0) {
                Ok(m) => m,
                Err(e) => return set_earthsnail_error(e),
            };
            let iso_f = (inp[4] / 100.0).clamp(-1.0, 1.0);
            let rem = 1.0 - iso_f.abs();
            let clvd_f = (inp[5] / 100.0).clamp(-rem, rem);
            let dc_f = (1.0 - iso_f.abs() - clvd_f.abs()).max(0.0);

            if iso_f == 0.0 && clvd_f == 0.0 {
                base_dc
            } else {
                // Reconstruct in the principal eigenframe of `base_dc` (`T, B, P` eigenvectors):
                let (_, evecs) = jacobi_eigen_3x3(base_dc.to_ned_matrix());
                let lam_iso = iso_f * m0;
                // Matching `decompose_moment_tensor` (`clvd_signed = -2.0 * d1`):
                //   clvd_f >= 0 (d1 <= 0): dev = dc_f * [1, 0, -1] + clvd_f * [1, -0.5, -0.5]
                //   clvd_f <  0 (d1 >  0): dev = dc_f * [1, 0, -1] + a      * [0.5, 0.5, -1] (a = -clvd_f)
                let (c0, c1, c2) = if clvd_f >= 0.0 {
                    (clvd_f, -0.5 * clvd_f, -0.5 * clvd_f)
                } else {
                    let a = -clvd_f;
                    (0.5 * a, 0.5 * a, -a)
                };
                let evals = [
                    lam_iso + m0 * (dc_f + c0),
                    lam_iso + m0 * c1,
                    lam_iso + m0 * (-dc_f + c2),
                ];
                let mut m_ned = [[0.0_f64; 3]; 3];
                for k in 0..3 {
                    for r in 0..3 {
                        for c in 0..3 {
                            m_ned[r][c] += evals[k] * evecs[k][r] * evecs[k][c];
                        }
                    }
                }
                MomentTensor6::from_ned(
                    m_ned[0][0],
                    m_ned[1][1],
                    m_ned[2][2],
                    m_ned[0][1],
                    m_ned[0][2],
                    m_ned[1][2],
                )
            }
        }
        _ => {
            return set_error_str(
                ES_ERR_INVALID_ARG,
                "input_mode must be 0 (MT6) or 1 (SDR+mix)",
            )
        }
    };

    let dec = decompose_moment_tensor(&mt);

    let grid_opt = if grid_size > 0 {
        match beachball_p_radiation_grid(&mt, grid_size, BeachballProjection::LambertEqualArea) {
            Ok(g) => Some(g),
            Err(e) => return set_earthsnail_error(e),
        }
    } else {
        None
    };

    let out = match unsafe { check_slice_mut(out_20_f64, 20, false, "out_20_f64") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    out[0] = dec.iso_percent;
    out[1] = dec.dc_percent;
    out[2] = dec.clvd_percent;
    out[3] = mt.scalar_moment_nm();
    out[4] = mt.moment_magnitude_mw();
    out[5] = dec.nodal_planes.0.strike_deg;
    out[6] = dec.nodal_planes.0.dip_deg;
    out[7] = dec.nodal_planes.0.rake_deg;
    out[8] = dec.nodal_planes.1.strike_deg;
    out[9] = dec.nodal_planes.1.dip_deg;
    out[10] = dec.nodal_planes.1.rake_deg;
    out[11] = dec.principal_axes.t_axis.azimuth_deg;
    out[12] = dec.principal_axes.t_axis.plunge_deg;
    out[13] = dec.principal_axes.t_axis.length;
    out[14] = dec.principal_axes.p_axis.azimuth_deg;
    out[15] = dec.principal_axes.p_axis.plunge_deg;
    out[16] = dec.principal_axes.p_axis.length;
    if let Some(n_ax) = dec.principal_axes.n_axis {
        out[17] = n_ax.azimuth_deg;
        out[18] = n_ax.plunge_deg;
        out[19] = n_ax.length;
    } else {
        out[17] = f64::NAN;
        out[18] = f64::NAN;
        out[19] = f64::NAN;
    }

    if let Some(grid) = grid_opt {
        let len = grid_size * grid_size;
        let out_g =
            match unsafe { check_slice_mut(out_beachball_ptr, len, false, "out_beachball_ptr") } {
                Ok(s) => s,
                Err(c) => return c,
            };
        for (i, slot) in out_g.iter_mut().enumerate().take(len) {
            *slot = if grid.inside_mask[i] {
                grid.amplitudes[i]
            } else {
                f64::NAN
            };
        }
    }
    ES_OK
}

// ============================================================================
// 5. Real-Time TauP Spherical Ray Engine & 4D Hypocenter Locator
// ============================================================================

/// Opaque handle wrapping a cached [`TauEngine`] (`iasp91`, `ak135`, or `prem`).
pub struct EsTauEngine {
    inner: TauEngine,
}

fn compile_tau_model_spec(spec: &str) -> Result<TauModel, EarthSnailError> {
    let trimmed = spec.trim();
    let name = if trimmed.is_empty() {
        "iasp91"
    } else {
        trimmed
    };
    if let Some(nd_body) = name
        .strip_prefix("nd:")
        .or_else(|| name.contains('\n').then_some(name))
    {
        let vm = VelocityModel::from_nd_str("custom_nd", nd_body)?;
        let sm = if cfg!(miri) {
            let cfg = SlownessModelConfig {
                min_delta_p: 5.0,
                max_delta_p: 150.0,
                max_depth_interval: 1200.0,
                max_range_interval_rad: 45.0 * core::f64::consts::PI / 180.0,
                max_interp_error: 2.0,
                ..SlownessModelConfig::default()
            };
            SlownessModel::with_config(vm, cfg)?
        } else {
            SlownessModel::from_velocity_model(vm)?
        };
        TauModel::from_slowness_model(sm)
    } else {
        TauModel::by_name(name)
    }
}

/// Creates a cached [`EsTauEngine`] for a named Earth model (`"iasp91"`, `"ak135"`, `"prem"`, or
/// custom `"nd:<named-discontinuity text>"`).
///
/// Returns `null_mut()` and sets `es_last_error_*` if the model name is invalid. Release with
/// `es_taup_engine_free(handle)` when finished.
///
/// # Safety
/// `model_ptr` must reference `model_len` valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn es_taup_engine_create(
    model_ptr: *const u8,
    model_len: usize,
) -> *mut EsTauEngine {
    clear_error();
    let name = match unsafe { check_utf8(model_ptr, model_len, false, "empty model name") } {
        Ok(s) => s,
        Err(_) => return core::ptr::null_mut(),
    };
    match compile_tau_model_spec(name).and_then(TauEngine::new) {
        Ok(engine) => Box::into_raw(Box::new(EsTauEngine { inner: engine })),
        Err(e) => {
            set_earthsnail_error(e);
            core::ptr::null_mut()
        }
    }
}

/// Destroys an [`EsTauEngine`] previously returned by `es_taup_engine_create`.
///
/// Passing `null_mut()` is a safe no-op.
///
/// # Safety
/// When non-null, `engine` must be a valid live pointer returned by `es_taup_engine_create`.
#[no_mangle]
pub unsafe extern "C" fn es_taup_engine_free(engine: *mut EsTauEngine) {
    if engine.is_null() {
        return;
    }
    let align = core::mem::align_of::<EsTauEngine>();
    if ((engine as usize) & (align - 1)) != 0 {
        set_error_str(ES_ERR_INVALID_ARG, "misaligned EsTauEngine handle");
        return;
    }
    unsafe { drop(Box::from_raw(engine)) };
}

fn split_phase_csv(phases_str: &str) -> Vec<&str> {
    let trimmed = phases_str.trim();
    if trimmed.is_empty() {
        vec!["ttbasic"]
    } else {
        trimmed
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }
}

unsafe fn with_tau_engine<R>(
    engine_or_null: *mut EsTauEngine,
    model_ptr: *const u8,
    model_len: usize,
    f: impl FnOnce(&mut TauEngine) -> Result<R, EarthSnailError>,
) -> Result<R, i32> {
    if !engine_or_null.is_null() {
        let align = core::mem::align_of::<EsTauEngine>();
        if ((engine_or_null as usize) & (align - 1)) != 0 {
            return Err(set_error_str(
                ES_ERR_NULL_PTR,
                "misaligned EsTauEngine handle",
            ));
        }
        let eng = unsafe { &mut (*engine_or_null).inner };
        f(eng).map_err(set_earthsnail_error)
    } else {
        let model_name = unsafe { check_utf8(model_ptr, model_len, true, "model_ptr")? };
        let tm = compile_tau_model_spec(model_name).map_err(set_earthsnail_error)?;
        let mut eng = TauEngine::new(tm).map_err(set_earthsnail_error)?;
        f(&mut eng).map_err(set_earthsnail_error)
    }
}

/// Computes `TauP` phase travel times across `iasp91`, `ak135`, or `prem`, returning an owned
/// `ESTT` v1 binary buffer (`b"ESTT"`).
///
/// Pass either a live `engine_or_null` handle (for sub-millisecond cached slider queries) OR
/// `null_mut()` + `(model_ptr, model_len)` (`"iasp91"`, `"ak135"`, `"prem"`).
///
/// # Safety
/// All non-null pointers must be valid and aligned. Free `*out_buf_ptr` with `es_free`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_taup_travel_times(
    engine_or_null: *mut EsTauEngine,
    model_ptr: *const u8,
    model_len: usize,
    depth_km: f64,
    distance_deg: f64,
    phases_ptr: *const u8,
    phases_len: usize,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let phases_str = match unsafe { check_utf8(phases_ptr, phases_len, true, "phases_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let phase_list = split_phase_csv(phases_str);
    let arrivals = match unsafe {
        with_tau_engine(engine_or_null, model_ptr, model_len, |eng| {
            eng.get_travel_times(depth_km, distance_deg, &phase_list)
        })
    } {
        Ok(a) => a,
        Err(c) => return c,
    };
    let encoded = wire::encode_arrivals_estt(&arrivals);
    unsafe { emit_owned_bytes(&encoded, out_buf_ptr, out_buf_len) }
}

/// Computes dense spherical `TauP` ray-path trajectories across `iasp91`, `ak135`, or `prem`,
/// returning an owned `ESRP` v1 binary buffer (`b"ESRP"`) including planetary Moho/CMB/ICB radii.
///
/// # Safety
/// All non-null pointers must be valid and aligned. Free `*out_buf_ptr` with `es_free`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_taup_ray_paths(
    engine_or_null: *mut EsTauEngine,
    model_ptr: *const u8,
    model_len: usize,
    depth_km: f64,
    distance_deg: f64,
    phases_ptr: *const u8,
    phases_len: usize,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let phases_str = match unsafe { check_utf8(phases_ptr, phases_len, true, "phases_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let phase_list = split_phase_csv(phases_str);
    let encoded = match unsafe {
        with_tau_engine(engine_or_null, model_ptr, model_len, |eng| {
            let paths = eng.get_ray_paths(depth_km, distance_deg, &phase_list)?;
            Ok(wire::encode_ray_paths_esrp(&paths, eng.base_model()))
        })
    } {
        Ok(b) => b,
        Err(c) => return c,
    };
    unsafe { emit_owned_bytes(&encoded, out_buf_ptr, out_buf_len) }
}

/// Computes `TauP` discontinuity & turning-point pierce waypoints across `iasp91`, `ak135`, or
/// `prem`, returning an owned `ESPP` v1 binary buffer (`b"ESPP"`).
///
/// # Safety
/// All non-null pointers must be valid and aligned. Free `*out_buf_ptr` with `es_free`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_taup_pierce_points(
    engine_or_null: *mut EsTauEngine,
    model_ptr: *const u8,
    model_len: usize,
    depth_km: f64,
    distance_deg: f64,
    phases_ptr: *const u8,
    phases_len: usize,
    out_buf_ptr: *mut *mut u8,
    out_buf_len: *mut usize,
) -> i32 {
    clear_error();
    let phases_str = match unsafe { check_utf8(phases_ptr, phases_len, true, "phases_ptr") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let phase_list = split_phase_csv(phases_str);
    let encoded = match unsafe {
        with_tau_engine(engine_or_null, model_ptr, model_len, |eng| {
            let pierces = eng.get_pierce_points(depth_km, distance_deg, &phase_list)?;
            Ok(wire::encode_pierce_points_espp(&pierces, eng.base_model()))
        })
    } {
        Ok(b) => b,
        Err(c) => return c,
    };
    unsafe { emit_owned_bytes(&encoded, out_buf_ptr, out_buf_len) }
}

/// Runs the Damped Huber-IRLS 4D Hypocenter Inverter (`locate_hypocenter`) on `n_obs >= 4`
/// station arrival observations:
///
/// - `obs_f64_ptr`: `n_obs * 5` `f64`s `[sta_lat_deg, sta_lon_deg, sta_elev_m, arrival_time_s, sigma_s]`.
/// - `obs_phases_ptr`: comma-separated phase labels of length `n_obs` (e.g. `"P,P,S,P,pP"`).
/// - `out_16_f64`:
///   `[0] lat_deg`, `[1] lon_deg`, `[2] depth_km`, `[3] origin_time_s`,
///   `[4] weighted_rms_s`, `[5] azimuthal_gap_deg`, `[6] iterations`,
///   `[7..10] semi_axes_95pct_km[0..3]`,
///   `[10..14] cov_nn, cov_ee, cov_dd, cov_tt`,
///   `[14] variance_scale_s2`, `[15] used_obs_count`.
///
/// # Safety
/// `obs_f64_ptr`, `obs_phases_ptr`, and `out_16_f64` must be valid aligned pointers.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn es_locate_hypocenter(
    model_ptr: *const u8,
    model_len: usize,
    obs_f64_ptr: *const f64,
    obs_phases_ptr: *const u8,
    obs_phases_len: usize,
    n_obs: usize,
    apply_ellipticity: u32,
    out_16_f64: *mut f64,
) -> i32 {
    clear_error();
    if n_obs < 4 {
        return set_error_str(
            ES_ERR_INVALID_ARG,
            "es_locate_hypocenter requires n_obs >= 4",
        );
    }
    if let Err(c) = validate_mut_ptr(out_16_f64, 16, false, "null out_16_f64") {
        return c;
    }
    let obs_len = match n_obs.checked_mul(5) {
        Some(k) => k,
        None => return set_error_str(ES_ERR_ALLOC, "n_obs overflow"),
    };

    let sol = {
        let model_str = match unsafe { check_utf8(model_ptr, model_len, true, "model_ptr") } {
            Ok(s) => s.trim(),
            Err(c) => return c,
        };
        let model = match compile_tau_model_spec(model_str) {
            Ok(m) => m,
            Err(e) => return set_earthsnail_error(e),
        };
        let obs_nums = match unsafe { check_slice(obs_f64_ptr, obs_len, false, "obs_f64_ptr") } {
            Ok(s) => s,
            Err(c) => return c,
        };
        let phases_csv =
            match unsafe { check_utf8(obs_phases_ptr, obs_phases_len, false, "obs_phases_ptr") } {
                Ok(s) => s,
                Err(c) => return c,
            };
        let phase_names: Vec<&str> = phases_csv.split(',').map(str::trim).collect();
        if phase_names.len() != n_obs {
            return set_error_str(
                ES_ERR_INVALID_ARG,
                "obs_phases_ptr comma-separated count must equal n_obs",
            );
        }

        let mut observations = Vec::with_capacity(n_obs);
        for (i, &ph_name) in phase_names.iter().enumerate().take(n_obs) {
            let base = i * 5;
            observations.push(PhaseArrivalObservation {
                station_id: format!("STA{:03}", i + 1),
                station_lat_deg: obs_nums[base],
                station_lon_deg: obs_nums[base + 1],
                station_elevation_m: obs_nums[base + 2],
                phase: ph_name.to_string(),
                arrival_time_s: obs_nums[base + 3],
                uncertainty_s: obs_nums[base + 4],
            });
        }
        let cfg = HypocenterConfig {
            apply_ellipticity: apply_ellipticity != 0,
            ..HypocenterConfig::default()
        };
        match locate_hypocenter(&model, &observations, &cfg) {
            Ok(s) => s,
            Err(e) => return set_earthsnail_error(e),
        }
    };

    let out = match unsafe { check_slice_mut(out_16_f64, 16, false, "out_16_f64") } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let used = sol
        .residuals
        .iter()
        .filter(|r| r.huber_weight > 0.0)
        .count();
    out[0] = sol.lat_deg;
    out[1] = sol.lon_deg;
    out[2] = sol.depth_km;
    out[3] = sol.origin_time_s;
    out[4] = sol.weighted_rms_s;
    out[5] = sol.azimuthal_gap_deg;
    out[6] = sol.iterations as f64;
    out[7] = sol.ellipsoid.semi_axes_95pct_km[0];
    out[8] = sol.ellipsoid.semi_axes_95pct_km[1];
    out[9] = sol.ellipsoid.semi_axes_95pct_km[2];
    out[10] = sol.covariance_nedt[0][0];
    out[11] = sol.covariance_nedt[1][1];
    out[12] = sol.covariance_nedt[2][2];
    out[13] = sol.covariance_nedt[3][3];
    out[14] = sol.variance_scale_s2;
    out[15] = used as f64;
    ES_OK
}
