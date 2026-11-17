//! FDSN miniSEED version 3 (`MS3`, Specification v3.0) multi-record reader and writer with
//! nanosecond UTC timestamps, `f64` sample rate/period, Castagnoli CRC-32C (`0x1EDC6F41`)
//! verification & computation, FDSN Source Identifiers (`"FDSN:NET_STA_LOC_B_S_SS"`), UTF-8
//! JSON extra-header pass-through, and encodings `0/1/3/4/5/10/11`.

use crate::mseed::{MseedEncoding, MAX_MSEED_STREAM_SAMPLES};
use crate::steim::{decode_steim1, decode_steim2, encode_steim1, encode_steim2};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use earthsnail_core::{
    math, ChannelId, EarthSnailError, Result, Stream, Trace, TraceStats, UtcTimestamp,
};

/// Fixed header length of an FDSN miniSEED v3 record (`40` bytes).
pub const MS3_FIXED_HEADER_BYTES: usize = 40;

/// Maximum payload bytes accepted in a single MS3 record (`64 MiB`).
pub const MAX_MS3_RECORD_PAYLOAD: usize = 64 * 1024 * 1024;

const CRC32C_TABLE: [u32; 256] = build_crc32c_table();

const fn build_crc32c_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut i = 0_usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0x82F6_3B78;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Computes the Castagnoli CRC-32C checksum (`0x1EDC6F41` / reflected `0x82F63B78`) of `bytes`.
pub fn crc32c(bytes: &[u8]) -> u32 {
    crc32c_update(0xFFFF_FFFF, bytes) ^ 0xFFFF_FFFF
}

#[inline]
fn crc32c_update(mut crc: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        let idx = ((crc as u8) ^ b) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[idx];
    }
    crc
}

/// Computes the FDSN miniSEED v3 record CRC-32C with bytes `28..32` treated as `[0, 0, 0, 0]`.
pub fn ms3_record_crc32c(record: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    crc = crc32c_update(crc, &record[..28]);
    crc = crc32c_update(crc, &[0, 0, 0, 0]);
    crc = crc32c_update(crc, &record[32..]);
    crc ^ 0xFFFF_FFFF
}

/// Returns `true` if `bytes` starts with the FDSN miniSEED v3 magic & format version (`[b'M', b'S', 3]`).
#[inline]
pub fn looks_like_mseed3(bytes: &[u8]) -> bool {
    bytes.len() >= MS3_FIXED_HEADER_BYTES && bytes[0] == b'M' && bytes[1] == b'S' && bytes[2] == 3
}

/// Parsed single FDSN miniSEED v3 record.
#[derive(Debug, Clone, PartialEq)]
pub struct Mseed3Record {
    /// Bit flags (`bit 0`: calibration, `bit 1`: time tag questionable, `bit 2`: clock locked).
    pub flags: u8,
    /// Exact nanosecond UTC start timestamp.
    pub starttime: UtcTimestamp,
    /// Payload encoding (`0`, `1`, `3`, `4`, `5`, `10`, `11`).
    pub encoding: MseedEncoding,
    /// Sampling rate in Hz (`> 0.0` for numeric waveforms).
    pub sampling_rate: f64,
    /// Publication version (`1` default).
    pub publication_version: u8,
    /// Raw FDSN Source Identifier string (e.g. `"FDSN:IU_ANMO_10_B_H_Z"`).
    pub source_id: String,
    /// Parsed `ChannelId` derived from `source_id`.
    pub channel: ChannelId,
    /// Optional UTF-8 JSON extra headers preserved verbatim.
    pub extra_headers_json: Option<String>,
    /// Verified CRC-32C checksum of the record.
    pub crc32c: u32,
    /// Decoded waveform samples (`f64`).
    pub samples: Vec<f64>,
    /// Exact `i32` samples for integer / Steim encodings.
    pub int_samples: Option<Vec<i32>>,
    /// Decoded UTF-8 text when `encoding == MseedEncoding::Ascii`.
    pub text_payload: Option<String>,
}

/// Options controlling FDSN miniSEED v3 serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mseed3WriteOptions {
    /// Sample payload encoding (`Steim2` default).
    pub encoding: MseedEncoding,
    /// Record header flags (`0x04` = clock locked, `0` default).
    pub flags: u8,
    /// Data publication version (`1` default).
    pub publication_version: u8,
    /// Optional UTF-8 JSON object string written into the extra headers section.
    pub extra_headers_json: Option<String>,
    /// Maximum samples per MS3 record (`0` = all samples of a trace in a single record).
    pub max_samples_per_record: usize,
}

impl Default for Mseed3WriteOptions {
    fn default() -> Self {
        Self {
            encoding: MseedEncoding::Steim2,
            flags: 0,
            publication_version: 1,
            extra_headers_json: None,
            max_samples_per_record: 0,
        }
    }
}

/// Parses all FDSN miniSEED v3 records in `bytes`, verifying Castagnoli CRC-32C on every record.
pub fn read_mseed3_records(bytes: &[u8]) -> Result<Vec<Mseed3Record>> {
    if bytes.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "empty miniSEED v3 buffer".into(),
        ));
    }
    let mut records = Vec::new();
    let mut pos = 0_usize;
    let mut total_samples: usize = 0;

    while pos < bytes.len() {
        let rem = &bytes[pos..];
        if rem.iter().all(|&b| b == 0 || b.is_ascii_whitespace()) {
            break;
        }
        let (rec, consumed) = parse_one_mseed3_record(rem, total_samples)?;
        total_samples = total_samples.saturating_add(rec.samples.len());
        if total_samples > MAX_MSEED_STREAM_SAMPLES {
            return Err(EarthSnailError::CapacityExceeded(format!(
                "miniSEED v3 stream exceeded {MAX_MSEED_STREAM_SAMPLES} total samples"
            )));
        }
        records.push(rec);
        pos += consumed;
    }

    if records.is_empty() {
        return Err(EarthSnailError::Format(
            "no miniSEED v3 records found".into(),
        ));
    }
    Ok(records)
}

/// Reads a multi-record FDSN miniSEED v3 byte buffer into a `Stream`, reassembling contiguous
/// segments per channel and preserving time gaps (`Stream::get_gaps` compatible).
pub fn read_mseed3_bytes(bytes: &[u8]) -> Result<Stream> {
    let records = read_mseed3_records(bytes)?;
    let mut open_traces: Vec<Trace> = Vec::new();
    let mut finished_traces: Vec<Trace> = Vec::new();

    for rec in records {
        if rec.encoding == MseedEncoding::Ascii || rec.samples.is_empty() {
            continue;
        }
        if let Some(pos) = open_traces
            .iter()
            .position(|tr| tr.stats.channel == rec.channel)
        {
            let same_rate = (open_traces[pos].stats.sampling_rate - rec.sampling_rate).abs()
                <= 1e-6 * open_traces[pos].stats.sampling_rate;
            let contiguous = if same_rate {
                let fs = open_traces[pos].stats.sampling_rate;
                let dt = 1.0 / fs;
                match open_traces[pos]
                    .stats
                    .starttime
                    .sample_timestamp(open_traces[pos].npts() as i64, fs)
                {
                    Ok(expected) => {
                        rec.starttime.duration_secs_f64_since(expected).abs() <= 0.5 * dt
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if contiguous {
                open_traces[pos].data.extend_from_slice(&rec.samples);
            } else {
                let prev = open_traces.remove(pos);
                finished_traces.push(prev);
                open_traces.push(Trace::new(
                    TraceStats::new(rec.channel, rec.starttime, rec.sampling_rate),
                    rec.samples,
                ));
            }
        } else {
            open_traces.push(Trace::new(
                TraceStats::new(rec.channel, rec.starttime, rec.sampling_rate),
                rec.samples,
            ));
        }
    }

    finished_traces.extend(open_traces);
    Ok(Stream::from_traces(finished_traces))
}

/// Encodes a single `Trace` into FDSN miniSEED v3 bytes with CRC-32C integrity protection.
pub fn write_mseed3_trace_bytes(trace: &Trace, options: &Mseed3WriteOptions) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    append_trace_mseed3_records(trace, options, &mut out)?;
    Ok(out)
}

/// Encodes a `Stream` into concatenated FDSN miniSEED v3 records with CRC-32C integrity protection.
pub fn write_mseed3_stream_bytes(stream: &Stream, options: &Mseed3WriteOptions) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for trace in &stream.traces {
        append_trace_mseed3_records(trace, options, &mut out)?;
    }
    Ok(out)
}

/// Reads an FDSN miniSEED v3 file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_mseed3_file(path: impl AsRef<std::path::Path>) -> Result<Stream> {
    let bytes = std::fs::read(path)?;
    read_mseed3_bytes(&bytes)
}

/// Writes a `Stream` as FDSN miniSEED v3 to disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_mseed3_file(
    path: impl AsRef<std::path::Path>,
    stream: &Stream,
    options: &Mseed3WriteOptions,
) -> Result<()> {
    let bytes = write_mseed3_stream_bytes(stream, options)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn parse_one_mseed3_record(rem: &[u8], so_far: usize) -> Result<(Mseed3Record, usize)> {
    if rem.len() < MS3_FIXED_HEADER_BYTES {
        return Err(EarthSnailError::Format(format!(
            "truncated miniSEED v3 fixed header ({} < 40 bytes)",
            rem.len()
        )));
    }
    if rem[0] != b'M' || rem[1] != b'S' {
        return Err(EarthSnailError::Format(
            "invalid miniSEED v3 magic indicator (expected 'MS')".into(),
        ));
    }
    let version = rem[2];
    if version != 3 {
        return Err(EarthSnailError::Unsupported(format!(
            "unsupported miniSEED format version {version} (expected 3)"
        )));
    }

    let flags = rem[3];
    let nanosecond = u32::from_le_bytes([rem[4], rem[5], rem[6], rem[7]]);
    let year = u16::from_le_bytes([rem[8], rem[9]]) as i32;
    let jday = u16::from_le_bytes([rem[10], rem[11]]);
    let hour = rem[12];
    let min = rem[13];
    let sec = rem[14];
    let enc_code = rem[15];
    let raw_rate = f64::from_le_bytes([
        rem[16], rem[17], rem[18], rem[19], rem[20], rem[21], rem[22], rem[23],
    ]);
    let num_samples = u32::from_le_bytes([rem[24], rem[25], rem[26], rem[27]]) as usize;
    let expected_crc = u32::from_le_bytes([rem[28], rem[29], rem[30], rem[31]]);
    let pub_ver = rem[32];
    let sid_len = rem[33] as usize;
    let extra_len = u16::from_le_bytes([rem[34], rem[35]]) as usize;
    let data_len = u32::from_le_bytes([rem[36], rem[37], rem[38], rem[39]]) as usize;

    if data_len > MAX_MS3_RECORD_PAYLOAD {
        return Err(EarthSnailError::CapacityExceeded(format!(
            "MS3 payload length {data_len} exceeds safety bound {MAX_MS3_RECORD_PAYLOAD}"
        )));
    }
    let total_len = MS3_FIXED_HEADER_BYTES
        .checked_add(sid_len)
        .and_then(|v| v.checked_add(extra_len))
        .and_then(|v| v.checked_add(data_len))
        .ok_or_else(|| EarthSnailError::CapacityExceeded("MS3 record length overflow".into()))?;

    if rem.len() < total_len {
        return Err(EarthSnailError::Format(format!(
            "truncated MS3 record: expected {total_len} bytes, got {}",
            rem.len()
        )));
    }

    // Verify Castagnoli CRC-32C BEFORE allocating or decoding samples!
    let record_slice = &rem[..total_len];
    let computed_crc = ms3_record_crc32c(record_slice);
    if computed_crc != expected_crc {
        return Err(EarthSnailError::Compression(format!(
            "MS3 CRC-32C mismatch: header 0x{expected_crc:08X} != computed 0x{computed_crc:08X}"
        )));
    }

    if nanosecond >= 1_000_000_000 {
        return Err(EarthSnailError::TimeParse(format!(
            "MS3 nanosecond field {nanosecond} >= 1_000_000_000"
        )));
    }
    let (base_sec, leap_s) = if sec == 60 {
        (59_u8, 1_i128)
    } else {
        (sec, 0_i128)
    };
    let starttime =
        UtcTimestamp::from_year_jday_hms_nano(year, jday, hour, min, base_sec, nanosecond)?
            .add_nanos(leap_s * UtcTimestamp::NANOS_PER_SEC);

    if !raw_rate.is_finite() {
        return Err(EarthSnailError::Header(
            "non-finite sample_rate in MS3 header".into(),
        ));
    }
    let sampling_rate = if raw_rate >= 0.0 {
        raw_rate
    } else {
        -1.0 / raw_rate
    };

    let encoding = MseedEncoding::from_code(enc_code)?;
    if encoding != MseedEncoding::Ascii && num_samples > 0 && sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(
            "MS3 waveform record has zero sampling rate".into(),
        ));
    }
    if so_far.saturating_add(num_samples) > MAX_MSEED_STREAM_SAMPLES {
        return Err(EarthSnailError::CapacityExceeded(
            "MS3 sample count exceeds stream safety bound".into(),
        ));
    }

    let sid_off = MS3_FIXED_HEADER_BYTES;
    let extra_off = sid_off + sid_len;
    let data_off = extra_off + extra_len;

    let source_id_str = core::str::from_utf8(&record_slice[sid_off..extra_off])
        .map_err(|_| EarthSnailError::Format("MS3 Source Identifier is not valid UTF-8".into()))?;
    let source_id = String::from(source_id_str);
    let channel = parse_ms3_source_id(source_id_str);

    let extra_headers_json = if extra_len > 0 {
        let s = core::str::from_utf8(&record_slice[extra_off..data_off]).map_err(|_| {
            EarthSnailError::Format("MS3 JSON extra headers are not valid UTF-8".into())
        })?;
        let trimmed = s.trim();
        if !trimmed.is_empty() && (!trimmed.starts_with('{') || !trimmed.ends_with('}')) {
            return Err(EarthSnailError::Format(
                "MS3 extra headers must be a JSON object '{...}'".into(),
            ));
        }
        Some(String::from(s))
    } else {
        None
    };

    let payload = &record_slice[data_off..total_len];
    let mut samples = Vec::new();
    let mut int_samples = None;
    let mut text_payload = None;

    match encoding {
        MseedEncoding::Ascii => {
            let txt = core::str::from_utf8(payload).map_err(|_| {
                EarthSnailError::Format("non-UTF8 bytes in MS3 text payload".into())
            })?;
            text_payload = Some(String::from(txt));
        }
        MseedEncoding::Int16 => {
            let needed = num_samples.checked_mul(2).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("MS3 INT16 length overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated MS3 INT16 payload".into(),
                ));
            }
            let mut iv = Vec::with_capacity(num_samples);
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(2) {
                let v = i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
                iv.push(v);
                samples.push(v as f64);
            }
            int_samples = Some(iv);
        }
        MseedEncoding::Int32 => {
            let needed = num_samples.checked_mul(4).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("MS3 INT32 length overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated MS3 INT32 payload".into(),
                ));
            }
            let mut iv = Vec::with_capacity(num_samples);
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(4) {
                let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                iv.push(v);
                samples.push(v as f64);
            }
            int_samples = Some(iv);
        }
        MseedEncoding::Float32 => {
            let needed = num_samples.checked_mul(4).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("MS3 FLOAT32 length overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated MS3 FLOAT32 payload".into(),
                ));
            }
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !v.is_finite() {
                    return Err(EarthSnailError::Format(
                        "non-finite sample in MS3 FLOAT32 payload".into(),
                    ));
                }
                samples.push(v as f64);
            }
        }
        MseedEncoding::Float64 => {
            let needed = num_samples.checked_mul(8).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("MS3 FLOAT64 length overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated MS3 FLOAT64 payload".into(),
                ));
            }
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(8) {
                let v = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                if !v.is_finite() {
                    return Err(EarthSnailError::Format(
                        "non-finite sample in MS3 FLOAT64 payload".into(),
                    ));
                }
                samples.push(v);
            }
        }
        MseedEncoding::Steim1 => {
            if num_samples > 0 {
                let iv = decode_steim1(payload, num_samples)?;
                samples = iv.iter().map(|&x| x as f64).collect();
                int_samples = Some(iv);
            }
        }
        MseedEncoding::Steim2 => {
            if num_samples > 0 {
                let iv = decode_steim2(payload, num_samples)?;
                samples = iv.iter().map(|&x| x as f64).collect();
                int_samples = Some(iv);
            }
        }
    }

    Ok((
        Mseed3Record {
            flags,
            starttime,
            encoding,
            sampling_rate,
            publication_version: pub_ver,
            source_id,
            channel,
            extra_headers_json,
            crc32c: expected_crc,
            samples,
            int_samples,
            text_payload,
        },
        total_len,
    ))
}

fn parse_ms3_source_id(sid: &str) -> ChannelId {
    if let Ok(ch) = ChannelId::parse_id(sid) {
        return ch;
    }
    let body = sid.strip_prefix("FDSN:").unwrap_or(sid);
    let parts: Vec<&str> = body.split('_').collect();
    match parts.len() {
        4 => ChannelId::new(parts[0], parts[1], parts[2], parts[3]),
        _ => ChannelId::new("", body, "", ""),
    }
}

fn append_trace_mseed3_records(
    trace: &Trace,
    options: &Mseed3WriteOptions,
    out: &mut Vec<u8>,
) -> Result<()> {
    if !trace.stats.sampling_rate.is_finite() || trace.stats.sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(format!(
            "invalid sampling_rate {} for MS3 export",
            trace.stats.sampling_rate
        )));
    }
    if trace.data.is_empty() {
        return Ok(());
    }

    let sid = trace.stats.channel.fdsn_source_id();
    let sid_bytes = sid.as_bytes();
    if sid_bytes.len() > 255 {
        return Err(EarthSnailError::Header(
            "MS3 FDSN Source Identifier exceeds 255 bytes".into(),
        ));
    }

    let extra_bytes: &[u8] = match &options.extra_headers_json {
        Some(json) => {
            let trimmed = json.trim();
            if !trimmed.is_empty() && (!trimmed.starts_with('{') || !trimmed.ends_with('}')) {
                return Err(EarthSnailError::InvalidParameter(
                    "MS3 extra_headers_json must be a JSON object '{...}'".into(),
                ));
            }
            if json.len() > (u16::MAX as usize) {
                return Err(EarthSnailError::CapacityExceeded(
                    "MS3 extra_headers_json exceeds 65535 bytes".into(),
                ));
            }
            json.as_bytes()
        }
        None => &[],
    };

    let chunk_size = if options.max_samples_per_record == 0 {
        trace.data.len()
    } else {
        options.max_samples_per_record
    };

    let mut pos = 0_usize;
    let mut prev_steim = 0_i32;

    while pos < trace.data.len() {
        let end = (pos + chunk_size).min(trace.data.len());
        let sub = &trace.data[pos..end];
        let rec_start = trace
            .stats
            .starttime
            .sample_timestamp(pos as i64, trace.stats.sampling_rate)?;
        let (year, jday, hour, min, sec, nano) = rec_start.to_year_jday_hms_nano();
        if !(0..=65535).contains(&year) {
            return Err(EarthSnailError::TimeParse(format!(
                "year {year} out of MS3 u16 range 0..=65535"
            )));
        }

        let payload = match options.encoding {
            MseedEncoding::Int16 => {
                let mut buf = Vec::with_capacity(sub.len() * 2);
                for &v in sub {
                    if !v.is_finite() || v < (i16::MIN as f64) || v > (i16::MAX as f64) {
                        return Err(EarthSnailError::Compression(
                            "sample out of INT16 range in MS3 export".into(),
                        ));
                    }
                    buf.extend_from_slice(&(math::round(v) as i16).to_le_bytes());
                }
                buf
            }
            MseedEncoding::Int32 => {
                let mut buf = Vec::with_capacity(sub.len() * 4);
                for &v in sub {
                    if !v.is_finite() || v < (i32::MIN as f64) || v > (i32::MAX as f64) {
                        return Err(EarthSnailError::Compression(
                            "sample out of INT32 range in MS3 export".into(),
                        ));
                    }
                    buf.extend_from_slice(&(math::round(v) as i32).to_le_bytes());
                }
                buf
            }
            MseedEncoding::Float32 => {
                let mut buf = Vec::with_capacity(sub.len() * 4);
                for &v in sub {
                    let f = v as f32;
                    if !f.is_finite() {
                        return Err(EarthSnailError::Compression(
                            "sample out of FLOAT32 range in MS3 export".into(),
                        ));
                    }
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                buf
            }
            MseedEncoding::Float64 => {
                let mut buf = Vec::with_capacity(sub.len() * 8);
                for &v in sub {
                    if !v.is_finite() {
                        return Err(EarthSnailError::Compression(
                            "non-finite sample in MS3 FLOAT64 export".into(),
                        ));
                    }
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
            MseedEncoding::Steim1 => {
                let mut iv = Vec::with_capacity(sub.len());
                for &v in sub {
                    if !v.is_finite() || v < (i32::MIN as f64) || v > (i32::MAX as f64) {
                        return Err(EarthSnailError::Compression(
                            "sample out of INT32 range for MS3 Steim-1".into(),
                        ));
                    }
                    iv.push(math::round(v) as i32);
                }
                let encoded = encode_steim1(&iv, prev_steim);
                prev_steim = *iv.last().unwrap_or(&0);
                encoded
            }
            MseedEncoding::Steim2 => {
                let mut iv = Vec::with_capacity(sub.len());
                for &v in sub {
                    if !v.is_finite() || v < (i32::MIN as f64) || v > (i32::MAX as f64) {
                        return Err(EarthSnailError::Compression(
                            "sample out of INT32 range for MS3 Steim-2".into(),
                        ));
                    }
                    iv.push(math::round(v) as i32);
                }
                let encoded = encode_steim2(&iv, prev_steim)?;
                prev_steim = *iv.last().unwrap_or(&0);
                encoded
            }
            MseedEncoding::Ascii => {
                return Err(EarthSnailError::Unsupported(
                    "ASCII encoding is not valid for numeric Trace MS3 export".into(),
                ));
            }
        };

        let rec_total =
            MS3_FIXED_HEADER_BYTES + sid_bytes.len() + extra_bytes.len() + payload.len();
        let mut record = Vec::with_capacity(rec_total);
        record.push(b'M');
        record.push(b'S');
        record.push(3_u8);
        record.push(options.flags);
        record.extend_from_slice(&nano.to_le_bytes());
        record.extend_from_slice(&(year as u16).to_le_bytes());
        record.extend_from_slice(&jday.to_le_bytes());
        record.push(hour);
        record.push(min);
        record.push(sec);
        record.push(options.encoding as u8);
        let rate_field = if trace.stats.sampling_rate >= 1.0 {
            trace.stats.sampling_rate
        } else {
            -1.0 / trace.stats.sampling_rate
        };
        record.extend_from_slice(&rate_field.to_le_bytes());
        record.extend_from_slice(&(sub.len() as u32).to_le_bytes());
        record.extend_from_slice(&[0_u8; 4]); // CRC-32C placeholder at 28..32
        record.push(options.publication_version);
        record.push(sid_bytes.len() as u8);
        record.extend_from_slice(&(extra_bytes.len() as u16).to_le_bytes());
        record.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        debug_assert_eq!(record.len(), MS3_FIXED_HEADER_BYTES);

        record.extend_from_slice(sid_bytes);
        record.extend_from_slice(extra_bytes);
        record.extend_from_slice(&payload);

        let crc = ms3_record_crc32c(&record);
        record[28..32].copy_from_slice(&crc.to_le_bytes());

        out.extend_from_slice(&record);
        pos = end;
    }

    Ok(())
}
