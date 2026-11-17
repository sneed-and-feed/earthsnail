//! Binary wire-frame encoders for zero-parse, 8-byte-aligned FFI transfers between Rust
//! (`x86_64` / `wasm32-unknown-unknown`), JavaScript (`WebAssembly.Memory` `DataView` / `Float64Array`),
//! and Python (`ctypes` / `struct` / `memoryview`).

use earthsnail_core::{is_sac_float_undef, Stream};
use earthsnail_io::WaveformFormat;
use earthsnail_taup::{ArrivalPath, ArrivalPierce, SeismicArrival, TauModel};

/// Magic tag `"ESWF"` (`0x4657_5345` LE) for decoded waveform streams.
pub const MAGIC_ESWF: [u8; 4] = *b"ESWF";
/// Magic tag `"ESTT"` (`0x5454_5345` LE) for `TauP` travel-time arrivals.
pub const MAGIC_ESTT: [u8; 4] = *b"ESTT";
/// Magic tag `"ESRP"` (`0x5052_5345` LE) for `TauP` ray-path trajectories.
pub const MAGIC_ESRP: [u8; 4] = *b"ESRP";
/// Magic tag `"ESPP"` (`0x5050_5345` LE) for `TauP` discontinuity pierce points.
pub const MAGIC_ESPP: [u8; 4] = *b"ESPP";

/// Wire protocol version (`1`).
pub const WIRE_VERSION: u32 = 1;

#[inline]
fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_padded_ascii<const N: usize>(buf: &mut Vec<u8>, s: &str) {
    let mut slot = [0u8; N];
    let bytes = s.as_bytes();
    let n = bytes.len().min(N);
    slot[..n].copy_from_slice(&bytes[..n]);
    buf.extend_from_slice(&slot);
}

#[inline]
fn sac_f64_or_nan(val: f64) -> f64 {
    if is_sac_float_undef(val) {
        f64::NAN
    } else {
        val
    }
}

/// Maps [`WaveformFormat`] (plus SAC endianness check) into a compact integer code:
/// - `0`: Unknown
/// - `1`: SAC Binary Little-Endian
/// - `2`: SAC Binary Big-Endian
/// - `3`: SAC Alphanumeric
/// - `4`: miniSEED v2
/// - `5`: miniSEED v3
/// - `6`: GeoCSV
/// - `7`: SLIST
/// - `8`: TSPAIR
pub fn format_code(fmt: Option<WaveformFormat>) -> u32 {
    match fmt {
        None => 0,
        Some(WaveformFormat::SacBinaryLittle) => 1,
        Some(WaveformFormat::SacBinaryBig) => 2,
        Some(WaveformFormat::SacAlpha) => 3,
        Some(WaveformFormat::MiniSeed2) => 4,
        Some(WaveformFormat::MiniSeed3) => 5,
        Some(WaveformFormat::GeoCsv) => 6,
        Some(WaveformFormat::SList) => 7,
        Some(WaveformFormat::TsPair) => 8,
    }
}

/// Encodes a decoded [`Stream`] into an 8-byte-aligned `ESWF` v1 binary buffer:
///
/// - **Global Header (`16` bytes)**:
///   `magic = b"ESWF"` (`4B`), `version = 1u32` (`4B`), `trace_count: u32` (`4B`), `format_code: u32` (`4B`).
/// - **Per-Trace Header (`192` bytes, multiple of 8)**:
///   - `[0..8]`: `npts: u64`
///   - `[8..16]`: `sampling_rate_hz: f64`
///   - `[16..24]`: `start_epoch_s: i64`
///   - `[24..28]`: `start_subsec_ns: u32`
///   - `[28..32]`: `reserved: u32`
///   - `[32..40]`: `calib: f64`
///   - `[40..152]`: `14 * f64` SAC metadata (`stla, stlo, stel, stdp, evla, evlo, evdp, mag, dist, az, baz, gcarc, cmpaz, cmpinc`; `NaN` when undefined)
///   - `[152..184]`: `network[8], station[8], location[8], channel[8]` (null-padded UTF-8)
///   - `[184..192]`: `origin_offset_s: f64` (`o - b` when defined, else `NaN`)
/// - **Per-Trace Samples (`npts * 8` bytes, 8-byte aligned)**:
///   Little-endian IEEE-754 `f64` samples.
pub fn encode_stream_eswf(stream: &Stream, fmt_code: u32) -> Vec<u8> {
    let total_samples: usize = stream.traces.iter().map(|t| t.data.len()).sum();
    let cap = 16 + stream.traces.len() * 192 + total_samples * 8;
    let mut out = Vec::with_capacity(cap);

    out.extend_from_slice(&MAGIC_ESWF);
    push_u32(&mut out, WIRE_VERSION);
    push_u32(&mut out, stream.traces.len() as u32);
    push_u32(&mut out, fmt_code);

    for tr in &stream.traces {
        let npts = tr.data.len();
        let ns_total = tr.stats.starttime.0;
        let epoch_s = ns_total.div_euclid(1_000_000_000) as i64;
        let subsec_ns = ns_total.rem_euclid(1_000_000_000) as u32;

        push_u64(&mut out, npts as u64);
        push_f64(&mut out, tr.stats.sampling_rate);
        push_i64(&mut out, epoch_s);
        push_u32(&mut out, subsec_ns);
        push_u32(&mut out, 0);
        push_f64(&mut out, tr.stats.calib);

        if let Some(sac) = &tr.stats.sac {
            push_f64(&mut out, sac_f64_or_nan(sac.stla));
            push_f64(&mut out, sac_f64_or_nan(sac.stlo));
            push_f64(&mut out, sac_f64_or_nan(sac.stel));
            push_f64(&mut out, sac_f64_or_nan(sac.stdp));
            push_f64(&mut out, sac_f64_or_nan(sac.evla));
            push_f64(&mut out, sac_f64_or_nan(sac.evlo));
            push_f64(&mut out, sac_f64_or_nan(sac.evdp));
            push_f64(&mut out, sac_f64_or_nan(sac.mag));
            push_f64(&mut out, sac_f64_or_nan(sac.dist));
            push_f64(&mut out, sac_f64_or_nan(sac.az));
            push_f64(&mut out, sac_f64_or_nan(sac.baz));
            push_f64(&mut out, sac_f64_or_nan(sac.gcarc));
            push_f64(&mut out, sac_f64_or_nan(sac.cmpaz));
            push_f64(&mut out, sac_f64_or_nan(sac.cmpinc));
        } else {
            for _ in 0..14 {
                push_f64(&mut out, f64::NAN);
            }
        }

        push_padded_ascii::<8>(&mut out, &tr.stats.channel.network);
        push_padded_ascii::<8>(&mut out, &tr.stats.channel.station);
        push_padded_ascii::<8>(&mut out, &tr.stats.channel.location);
        push_padded_ascii::<8>(&mut out, &tr.stats.channel.channel);

        let origin_off = tr
            .stats
            .sac
            .as_ref()
            .and_then(|s| {
                if !is_sac_float_undef(s.o) && !is_sac_float_undef(s.b) {
                    Some(s.o - s.b)
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN);
        push_f64(&mut out, origin_off);

        debug_assert_eq!(out.len() % 8, 0);
        for &v in &tr.data {
            push_f64(&mut out, v);
        }
    }

    out
}

/// Encodes [`SeismicArrival`]s into an 8-byte-aligned `ESTT` v1 binary buffer:
///
/// - **Header (`16` bytes)**: `b"ESTT"`, `version = 1u32`, `count: u32`, `reserved = 0u32`.
/// - **Records (`count * 96` bytes)**:
///   `[time_s, ray_param_s_per_rad, ray_param_s_per_deg, takeoff_angle_deg, incident_angle_deg, distance_deg, purist_distance_deg, source_depth_km]` (`8 * f64 = 64B`),
///   `name[16]` (`16B`), `purist_name[16]` (`16B`).
pub fn encode_arrivals_estt(arrivals: &[SeismicArrival]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + arrivals.len() * 96);
    out.extend_from_slice(&MAGIC_ESTT);
    push_u32(&mut out, WIRE_VERSION);
    push_u32(&mut out, arrivals.len() as u32);
    push_u32(&mut out, 0);

    for arr in arrivals {
        push_f64(&mut out, arr.time_s);
        push_f64(&mut out, arr.ray_param_s_per_rad);
        push_f64(&mut out, arr.ray_param_s_per_deg);
        push_f64(&mut out, arr.takeoff_angle_deg);
        push_f64(&mut out, arr.incident_angle_deg);
        push_f64(&mut out, arr.distance_deg);
        push_f64(&mut out, arr.purist_distance_deg);
        push_f64(&mut out, arr.source_depth_km);
        push_padded_ascii::<16>(&mut out, &arr.name);
        push_padded_ascii::<16>(&mut out, &arr.purist_name);
    }
    out
}

/// Encodes [`ArrivalPath`]s into an 8-byte-aligned `ESRP` v1 binary buffer:
///
/// - **Header (`48` bytes)**:
///   `b"ESRP"`, `version = 1u32`, `count: u32`, `reserved = 0u32`,
///   `planet_radius_km: f64`, `moho_depth_km: f64`, `cmb_depth_km: f64`, `iocb_depth_km: f64`.
/// - **Per-Arrival Block (`80 + point_count * 32` bytes)**:
///   - Arrival Header (`80` bytes):
///     `time_s: f64`, `ray_param_s_per_rad: f64`, `ray_param_s_per_deg: f64`,
///     `takeoff_angle_deg: f64`, `incident_angle_deg: f64`, `purist_distance_deg: f64`,
///     `point_count: u64`, `name[16]`, `source_depth_km: f64`.
///   - Points (`point_count * 32` bytes):
///     `[dist_deg: f64, depth_km: f64, radius_km: f64, time_s: f64]`.
pub fn encode_ray_paths_esrp(paths: &[ArrivalPath], base_model: &TauModel) -> Vec<u8> {
    let total_pts: usize = paths.iter().map(|p| p.path_points.len()).sum();
    let mut out = Vec::with_capacity(48 + paths.len() * 80 + total_pts * 32);

    out.extend_from_slice(&MAGIC_ESRP);
    push_u32(&mut out, WIRE_VERSION);
    push_u32(&mut out, paths.len() as u32);
    push_u32(&mut out, 0);
    push_f64(&mut out, base_model.radius_of_planet);
    push_f64(&mut out, base_model.moho_depth);
    push_f64(&mut out, base_model.cmb_depth);
    push_f64(&mut out, base_model.iocb_depth);

    for ap in paths {
        let arr = &ap.arrival;
        push_f64(&mut out, arr.time_s);
        push_f64(&mut out, arr.ray_param_s_per_rad);
        push_f64(&mut out, arr.ray_param_s_per_deg);
        push_f64(&mut out, arr.takeoff_angle_deg);
        push_f64(&mut out, arr.incident_angle_deg);
        push_f64(&mut out, arr.purist_distance_deg);
        push_u64(&mut out, ap.path_points.len() as u64);
        push_padded_ascii::<16>(&mut out, &arr.name);
        push_f64(&mut out, arr.source_depth_km);

        for pt in &ap.path_points {
            push_f64(&mut out, pt.dist_deg);
            push_f64(&mut out, pt.depth_km);
            push_f64(&mut out, pt.radius_km);
            push_f64(&mut out, pt.time_s);
        }
    }
    out
}

/// Encodes [`ArrivalPierce`]s into an 8-byte-aligned `ESPP` v1 binary buffer:
///
/// - **Header (`48` bytes)**:
///   `b"ESPP"`, `version = 1u32`, `count: u32`, `reserved = 0u32`,
///   `planet_radius_km: f64`, `moho_depth_km: f64`, `cmb_depth_km: f64`, `iocb_depth_km: f64`.
/// - **Per-Arrival Block (`80 + point_count * 32` bytes)**:
///   - Arrival Header (`80` bytes): same layout as `ESRP`.
///   - Pierce Points (`point_count * 32` bytes):
///     `[dist_deg: f64, depth_km: f64, p_s_per_rad: f64, time_s: f64]`.
pub fn encode_pierce_points_espp(pierces: &[ArrivalPierce], base_model: &TauModel) -> Vec<u8> {
    let total_pts: usize = pierces.iter().map(|p| p.pierce_points.len()).sum();
    let mut out = Vec::with_capacity(48 + pierces.len() * 80 + total_pts * 32);

    out.extend_from_slice(&MAGIC_ESPP);
    push_u32(&mut out, WIRE_VERSION);
    push_u32(&mut out, pierces.len() as u32);
    push_u32(&mut out, 0);
    push_f64(&mut out, base_model.radius_of_planet);
    push_f64(&mut out, base_model.moho_depth);
    push_f64(&mut out, base_model.cmb_depth);
    push_f64(&mut out, base_model.iocb_depth);

    for ap in pierces {
        let arr = &ap.arrival;
        push_f64(&mut out, arr.time_s);
        push_f64(&mut out, arr.ray_param_s_per_rad);
        push_f64(&mut out, arr.ray_param_s_per_deg);
        push_f64(&mut out, arr.takeoff_angle_deg);
        push_f64(&mut out, arr.incident_angle_deg);
        push_f64(&mut out, arr.purist_distance_deg);
        push_u64(&mut out, ap.pierce_points.len() as u64);
        push_padded_ascii::<16>(&mut out, &arr.name);
        push_f64(&mut out, arr.source_depth_km);

        for pt in &ap.pierce_points {
            push_f64(&mut out, pt.dist_deg);
            push_f64(&mut out, pt.depth_km);
            push_f64(&mut out, pt.p_s_per_rad);
            push_f64(&mut out, pt.time_s);
        }
    }
    out
}
