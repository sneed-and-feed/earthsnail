//! Pure-Rust SEED / miniSEED v2 (`2.4`) multi-record reader and writer supporting arbitrary
//! power-of-2 record lengths (`256B..=65536B`, `512B` default), 48-byte Fixed Section of Data
//! Header (FSDH), Blockette `100` (exact `f32` sample rate), Blockette `1000` (encoding, endian,
//! record length), Blockette `1001` (timing quality, signed microsecond offset, frame count),
//! encodings `0` (`ASCII`), `1` (`INT16`), `3` (`INT32`), `4` (`FLOAT32`), `5` (`FLOAT64`),
//! `10` (`STEIM1`), and `11` (`STEIM2`), with contiguous multi-trace `Stream` reassembly and
//! gap/overlap preservation (`Stream::get_gaps` compatible).

use crate::steim::{
    decode_steim1, decode_steim2, encode_steim1_bounded, encode_steim2_bounded, STEIM_FRAME_BYTES,
};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use earthsnail_core::{
    math, ChannelId, EarthSnailError, Result, Stream, Trace, TraceStats, UtcTimestamp,
};

/// Size of the SEED v2.4 Fixed Section of Data Header in bytes.
pub const FSDH_BYTES: usize = 48;

/// Default miniSEED v2 data record size (`512` bytes).
pub const DEFAULT_MSEED_RECORD_LEN: usize = 512;

/// Minimum legal miniSEED v2 power-of-2 record length (`2^8 = 256` bytes).
pub const MIN_MSEED_RECORD_LEN: usize = 256;

/// Maximum legal miniSEED v2 power-of-2 record length (`2^16 = 65536` bytes).
pub const MAX_MSEED_RECORD_LEN: usize = 65_536;

/// Upper bound on total decoded samples across one `read_mseed_bytes` call (`256M` samples).
pub const MAX_MSEED_STREAM_SAMPLES: usize = 256 * 1024 * 1024;

/// SEED Blockette 1000 payload encoding codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MseedEncoding {
    /// `0`: ASCII text / log channel.
    Ascii = 0,
    /// `1`: 16-bit two's-complement integer (`i16`).
    Int16 = 1,
    /// `3`: 32-bit two's-complement integer (`i32`).
    Int32 = 3,
    /// `4`: IEEE-754 single-precision float (`f32`).
    Float32 = 4,
    /// `5`: IEEE-754 double-precision float (`f64`).
    Float64 = 5,
    /// `10`: Steim-1 compressed differences (`i32`).
    Steim1 = 10,
    /// `11`: Steim-2 compressed differences (`i32`).
    Steim2 = 11,
}

impl MseedEncoding {
    /// Maps a raw SEED encoding code (`u8`) to `MseedEncoding`.
    pub fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(Self::Ascii),
            1 => Ok(Self::Int16),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Float32),
            5 => Ok(Self::Float64),
            10 => Ok(Self::Steim1),
            11 => Ok(Self::Steim2),
            other => Err(EarthSnailError::Unsupported(format!(
                "unsupported miniSEED encoding code {other}"
            ))),
        }
    }
}

/// Byte order declared in SEED headers / Blockette 1000 (`0` = Little-Endian, `1` = Big-Endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MseedByteOrder {
    /// `0`: VAX / 80x86 little-endian.
    Little,
    /// `1`: SPARC / Motorola / network big-endian (SEED standard default).
    Big,
}

/// Options controlling miniSEED v2 record serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MseedWriteOptions {
    /// Power-of-2 record length in bytes (`256..=65536`, default `512`).
    pub record_length: usize,
    /// Sample payload encoding (`Steim2` default for integer counts, `Float64` for general floats).
    pub encoding: MseedEncoding,
    /// Header & uncompressed payload byte order (`Big` default; Steim frames are always big-endian).
    pub byte_order: MseedByteOrder,
    /// SEED data quality indicator (`'D'`, `'R'`, `'Q'`, or `'M'`; default `'D'`).
    pub quality_indicator: char,
}

impl Default for MseedWriteOptions {
    fn default() -> Self {
        Self {
            record_length: DEFAULT_MSEED_RECORD_LEN,
            encoding: MseedEncoding::Steim2,
            byte_order: MseedByteOrder::Big,
            quality_indicator: 'D',
        }
    }
}

/// Decoded single miniSEED v2 data record.
#[derive(Debug, Clone, PartialEq)]
pub struct MseedRecord {
    /// 6-digit sequence number (`1..=999999`).
    pub sequence_number: u32,
    /// Data quality indicator (`'D'`, `'R'`, `'Q'`, `'M'`).
    pub quality_indicator: char,
    /// Channel identifier (`NET.STA.LOC.CHA`).
    pub channel: ChannelId,
    /// Exact UTC timestamp of the first sample (incorporating FSDH `time_correction` and B1001 `micro_sec`).
    pub starttime: UtcTimestamp,
    /// Sampling rate in Hz (`0.0` for pure ASCII log records).
    pub sampling_rate: f64,
    /// Payload encoding code.
    pub encoding: MseedEncoding,
    /// Declared payload byte order.
    pub byte_order: MseedByteOrder,
    /// Record length in bytes (`256..=65536`).
    pub record_length: usize,
    /// Optional timing quality percentage (`0..=100`) from Blockette 1001.
    pub timing_quality: Option<u8>,
    /// Decoded numeric samples as `f64` (lossless for all `i16`, `i32`, `f32`, `f64`, `Steim1`, `Steim2`).
    pub samples: Vec<f64>,
    /// Exact `i32` samples when `encoding` is `Int16`, `Int32`, `Steim1`, or `Steim2`.
    pub int_samples: Option<Vec<i32>>,
    /// Decoded ASCII text when `encoding == MseedEncoding::Ascii`.
    pub ascii_text: Option<String>,
}

/// Returns `true` if `bytes` begins with a plausible 48-byte miniSEED v2 Fixed Section of Data Header.
pub fn looks_like_mseed2(bytes: &[u8]) -> bool {
    if bytes.len() < 64 {
        return false;
    }
    if !bytes[0..6].iter().all(|&b| b.is_ascii_digit() || b == b' ') {
        return false;
    }
    if !matches!(bytes[6], b'D' | b'R' | b'Q' | b'M' | b' ') {
        return false;
    }
    if !bytes[8..20]
        .iter()
        .all(|&b| b.is_ascii_graphic() || b == b' ' || b == 0)
    {
        return false;
    }
    detect_fsdh_byte_order(&bytes[..FSDH_BYTES]).is_some()
}

/// Parses all miniSEED v2 records in `bytes`, returning the raw `Vec<MseedRecord>` sequence.
pub fn read_mseed_records(bytes: &[u8]) -> Result<Vec<MseedRecord>> {
    if bytes.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "empty miniSEED v2 input buffer".into(),
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
        let (rec, consumed) = parse_one_mseed_record(rem, total_samples)?;
        total_samples = total_samples.saturating_add(rec.samples.len());
        if total_samples > MAX_MSEED_STREAM_SAMPLES {
            return Err(EarthSnailError::CapacityExceeded(format!(
                "miniSEED stream exceeded {MAX_MSEED_STREAM_SAMPLES} total samples"
            )));
        }
        records.push(rec);
        pos += consumed;
    }

    if records.is_empty() {
        return Err(EarthSnailError::Format(
            "no valid miniSEED v2 records found".into(),
        ));
    }
    Ok(records)
}

/// Decodes a multi-record miniSEED v2 byte slice into a `Stream`, reassembling contiguous records
/// per channel while preserving gaps/overlaps (`|dt_gap| > 0.5 / fs`) as distinct `Trace` segments.
pub fn read_mseed_bytes(bytes: &[u8]) -> Result<Stream> {
    let records = read_mseed_records(bytes)?;
    Ok(reassemble_records_to_stream(records))
}

/// Encodes a single `Trace` into multi-record miniSEED v2 bytes using `options`.
pub fn write_mseed_trace_bytes(trace: &Trace, options: MseedWriteOptions) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut seq = 1_u32;
    append_trace_mseed_records(trace, options, &mut seq, &mut out)?;
    Ok(out)
}

/// Encodes a multi-trace `Stream` into concatenated miniSEED v2 records using `options`.
pub fn write_mseed_stream_bytes(stream: &Stream, options: MseedWriteOptions) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut seq = 1_u32;
    for trace in &stream.traces {
        append_trace_mseed_records(trace, options, &mut seq, &mut out)?;
    }
    Ok(out)
}

/// Convenience writer encoding a `Stream` into 512-byte big-endian miniSEED v2 (`Steim2` if all
/// samples are exact `i32` integers with `<= 30`-bit diffs, falling back to `Float64` otherwise).
pub fn write_mseed_bytes(stream: &Stream) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut seq = 1_u32;
    for trace in &stream.traces {
        let enc = choose_auto_mseed_encoding(&trace.data);
        let opts = MseedWriteOptions {
            encoding: enc,
            ..MseedWriteOptions::default()
        };
        append_trace_mseed_records(trace, opts, &mut seq, &mut out)?;
    }
    Ok(out)
}

/// Reads a miniSEED v2 file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_mseed_file(path: impl AsRef<std::path::Path>) -> Result<Stream> {
    let bytes = std::fs::read(path)?;
    read_mseed_bytes(&bytes)
}

/// Writes a `Stream` to a miniSEED v2 file on disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_mseed_file(
    path: impl AsRef<std::path::Path>,
    stream: &Stream,
    options: MseedWriteOptions,
) -> Result<()> {
    let bytes = write_mseed_stream_bytes(stream, options)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn choose_auto_mseed_encoding(data: &[f64]) -> MseedEncoding {
    let mut all_i32 = true;
    let mut steim2_ok = true;
    let mut prev: i64 = 0;
    for (idx, &v) in data.iter().enumerate() {
        if !v.is_finite()
            || v < (i32::MIN as f64)
            || v > (i32::MAX as f64)
            || (v - math::trunc(v)).abs() > 0.0
        {
            all_i32 = false;
            break;
        }
        let cur = v as i32 as i64;
        if idx > 0 {
            let diff = cur - prev;
            if !(-536_870_912..=536_870_911).contains(&diff) {
                steim2_ok = false;
            }
        }
        prev = cur;
    }
    if !all_i32 {
        MseedEncoding::Float64
    } else if steim2_ok {
        MseedEncoding::Steim2
    } else {
        MseedEncoding::Steim1
    }
}

fn detect_fsdh_byte_order(fsdh: &[u8]) -> Option<MseedByteOrder> {
    let year_be = u16::from_be_bytes([fsdh[20], fsdh[21]]);
    let jday_be = u16::from_be_bytes([fsdh[22], fsdh[23]]);
    let hour = fsdh[24];
    let min = fsdh[25];
    let sec = fsdh[26];
    let tms_be = u16::from_be_bytes([fsdh[28], fsdh[29]]);

    if hour <= 23 && min <= 59 && sec <= 60 {
        if (1900..=2999).contains(&year_be) && (1..=366).contains(&jday_be) && tms_be < 10_000 {
            return Some(MseedByteOrder::Big);
        }
        let year_le = u16::from_le_bytes([fsdh[20], fsdh[21]]);
        let jday_le = u16::from_le_bytes([fsdh[22], fsdh[23]]);
        let tms_le = u16::from_le_bytes([fsdh[28], fsdh[29]]);
        if (1900..=2999).contains(&year_le) && (1..=366).contains(&jday_le) && tms_le < 10_000 {
            return Some(MseedByteOrder::Little);
        }
    }
    None
}

fn parse_one_mseed_record(rem: &[u8], so_far_samples: usize) -> Result<(MseedRecord, usize)> {
    if rem.len() < MIN_MSEED_RECORD_LEN {
        return Err(EarthSnailError::Format(format!(
            "truncated miniSEED v2 record ({} < {MIN_MSEED_RECORD_LEN} bytes)",
            rem.len()
        )));
    }

    let seq_str = core::str::from_utf8(&rem[0..6])
        .map_err(|_| EarthSnailError::Format("non-ASCII miniSEED sequence number".into()))?
        .trim();
    let sequence_number: u32 = if seq_str.is_empty() {
        0
    } else {
        seq_str.parse().map_err(|_| {
            EarthSnailError::Format(format!("invalid miniSEED sequence number '{seq_str}'"))
        })?
    };

    let q_byte = rem[6];
    if !matches!(q_byte, b'D' | b'R' | b'Q' | b'M' | b' ') {
        return Err(EarthSnailError::Format(format!(
            "invalid miniSEED quality indicator byte 0x{q_byte:02X}"
        )));
    }
    let quality_indicator = q_byte as char;

    let station = clean_seed_ascii(&rem[8..13]);
    let location = clean_seed_ascii(&rem[13..15]);
    let channel_code = clean_seed_ascii(&rem[15..18]);
    let network = clean_seed_ascii(&rem[18..20]);
    let channel = ChannelId::new(network, station, location, channel_code);

    let hdr_order = detect_fsdh_byte_order(&rem[..FSDH_BYTES])
        .ok_or_else(|| EarthSnailError::Header("invalid SEED BTime in miniSEED v2 FSDH".into()))?;

    let year = read_u16(rem, 20, hdr_order) as i32;
    let jday = read_u16(rem, 22, hdr_order);
    let hour = rem[24];
    let min = rem[25];
    let sec = rem[26];
    let tenth_millis = read_u16(rem, 28, hdr_order);

    let num_samples = read_u16(rem, 30, hdr_order) as usize;
    let sr_factor = read_i16(rem, 32, hdr_order);
    let sr_mult = read_i16(rem, 34, hdr_order);
    let activity_flags = rem[36];
    let num_blockettes = rem[39] as usize;
    let time_correction = read_i32(rem, 40, hdr_order);
    let data_offset = read_u16(rem, 44, hdr_order) as usize;
    let first_blockette_offset = read_u16(rem, 46, hdr_order) as usize;

    let mut starttime = UtcTimestamp::from_seed_btime(year, jday, hour, min, sec, tenth_millis)?;
    // Apply FSDH 0.1ms `time_correction` if bit 1 (`0x02`) of `activity_flags` is NOT set:
    if (activity_flags & 0x02) == 0 && time_correction != 0 {
        starttime = starttime.add_nanos((time_correction as i128) * 100_000);
    }

    let mut sampling_rate = decode_seed_sample_rate(sr_factor, sr_mult);
    let mut encoding_opt: Option<MseedEncoding> = None;
    let mut payload_order = hdr_order;
    let mut record_len_opt: Option<usize> = None;
    let mut timing_quality: Option<u8> = None;
    let mut micro_sec_offset: i8 = 0;
    let mut b1001_frames: Option<usize> = None;

    // Walk blockette chain safely within `rem[..rem.len().min(MAX_MSEED_RECORD_LEN)]`
    let scan_cap = rem.len().min(MAX_MSEED_RECORD_LEN);
    let mut blk_off = first_blockette_offset;
    let mut blk_seen = 0_usize;

    while blk_off != 0 && blk_seen < num_blockettes.max(16) {
        if blk_off < FSDH_BYTES || blk_off + 4 > scan_cap {
            return Err(EarthSnailError::Format(format!(
                "out-of-bounds miniSEED blockette offset {blk_off}"
            )));
        }
        if let Some(rl) = record_len_opt {
            if blk_off + 4 > rl {
                return Err(EarthSnailError::Format(format!(
                    "blockette offset {blk_off} exceeds record length {rl}"
                )));
            }
        }
        let blk_type = read_u16(rem, blk_off, hdr_order);
        let next_off = read_u16(rem, blk_off + 2, hdr_order) as usize;
        if next_off != 0 && next_off <= blk_off {
            return Err(EarthSnailError::Format(format!(
                "corrupt non-advancing blockette pointer {blk_off} -> {next_off}"
            )));
        }

        match blk_type {
            100 => {
                if blk_off + 12 > scan_cap {
                    return Err(EarthSnailError::Format("truncated Blockette 100".into()));
                }
                let exact_sr = read_f32(rem, blk_off + 4, hdr_order) as f64;
                if !exact_sr.is_finite() || exact_sr < 0.0 {
                    return Err(EarthSnailError::Header(
                        "invalid sample rate in Blockette 100".into(),
                    ));
                }
                if exact_sr > 0.0 {
                    sampling_rate = exact_sr;
                }
            }
            1000 => {
                if blk_off + 8 > scan_cap {
                    return Err(EarthSnailError::Format("truncated Blockette 1000".into()));
                }
                let enc_code = rem[blk_off + 4];
                let bo_code = rem[blk_off + 5];
                let rec_len_exp = rem[blk_off + 6];
                if !(8..=16).contains(&rec_len_exp) {
                    return Err(EarthSnailError::Header(format!(
                        "invalid Blockette 1000 record length exponent {rec_len_exp} (expected 8..=16)"
                    )));
                }
                encoding_opt = Some(MseedEncoding::from_code(enc_code)?);
                payload_order = if bo_code == 0 {
                    MseedByteOrder::Little
                } else {
                    MseedByteOrder::Big
                };
                record_len_opt = Some(1_usize << (rec_len_exp as u32));
            }
            1001 => {
                if blk_off + 8 > scan_cap {
                    return Err(EarthSnailError::Format("truncated Blockette 1001".into()));
                }
                timing_quality = Some(rem[blk_off + 4]);
                micro_sec_offset = rem[blk_off + 5] as i8;
                let fc = rem[blk_off + 7] as usize;
                if fc > 0 {
                    b1001_frames = Some(fc);
                }
            }
            _ => {
                // Unrecognized optional SEED blockette (e.g. 200/201/300/500/2000): skip cleanly via `next_off`.
            }
        }

        blk_seen += 1;
        if blk_seen == num_blockettes && next_off == 0 {
            break;
        }
        blk_off = next_off;
    }

    if micro_sec_offset != 0 {
        starttime = starttime.add_nanos((micro_sec_offset as i128) * UtcTimestamp::NANOS_PER_MICRO);
    }

    let record_length = match record_len_opt {
        Some(rl) => rl,
        None => infer_fallback_record_len(rem, data_offset)?,
    };

    if rem.len() < record_length {
        return Err(EarthSnailError::Format(format!(
            "truncated miniSEED v2 record: expected {record_length} bytes, got {}",
            rem.len()
        )));
    }

    if so_far_samples.saturating_add(num_samples) > MAX_MSEED_STREAM_SAMPLES {
        return Err(EarthSnailError::CapacityExceeded(
            "miniSEED sample count exceeds safety limit".into(),
        ));
    }

    let encoding = encoding_opt.unwrap_or(MseedEncoding::Steim1);
    if num_samples > 0 && (data_offset < FSDH_BYTES || data_offset >= record_length) {
        return Err(EarthSnailError::Header(format!(
            "invalid miniSEED data_offset {data_offset} for record_length {record_length}"
        )));
    }
    if encoding != MseedEncoding::Ascii && num_samples > 0 && sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(
            "non-positive sampling rate on miniSEED waveform record".into(),
        ));
    }

    let payload = if data_offset <= record_length {
        &rem[data_offset..record_length]
    } else {
        &[][..]
    };

    let mut samples = Vec::new();
    let mut int_samples = None;
    let mut ascii_text = None;

    match encoding {
        MseedEncoding::Ascii => {
            let take = num_samples.min(payload.len());
            let txt = core::str::from_utf8(&payload[..take]).map_err(|_| {
                EarthSnailError::Format("non-UTF8 bytes in miniSEED ASCII payload".into())
            })?;
            ascii_text = Some(String::from(txt));
        }
        MseedEncoding::Int16 => {
            let needed = num_samples.checked_mul(2).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("INT16 byte count overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated miniSEED INT16 payload".into(),
                ));
            }
            let mut iv = Vec::with_capacity(num_samples);
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(2) {
                let v = match payload_order {
                    MseedByteOrder::Big => i16::from_be_bytes([chunk[0], chunk[1]]) as i32,
                    MseedByteOrder::Little => i16::from_le_bytes([chunk[0], chunk[1]]) as i32,
                };
                iv.push(v);
                samples.push(v as f64);
            }
            int_samples = Some(iv);
        }
        MseedEncoding::Int32 => {
            let needed = num_samples.checked_mul(4).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("INT32 byte count overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated miniSEED INT32 payload".into(),
                ));
            }
            let mut iv = Vec::with_capacity(num_samples);
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(4) {
                let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let v = match payload_order {
                    MseedByteOrder::Big => i32::from_be_bytes(raw),
                    MseedByteOrder::Little => i32::from_le_bytes(raw),
                };
                iv.push(v);
                samples.push(v as f64);
            }
            int_samples = Some(iv);
        }
        MseedEncoding::Float32 => {
            let needed = num_samples.checked_mul(4).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("FLOAT32 byte count overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated miniSEED FLOAT32 payload".into(),
                ));
            }
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(4) {
                let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let v = match payload_order {
                    MseedByteOrder::Big => f32::from_be_bytes(raw),
                    MseedByteOrder::Little => f32::from_le_bytes(raw),
                };
                if !v.is_finite() {
                    return Err(EarthSnailError::Format(
                        "non-finite sample in miniSEED FLOAT32 payload".into(),
                    ));
                }
                samples.push(v as f64);
            }
        }
        MseedEncoding::Float64 => {
            let needed = num_samples.checked_mul(8).ok_or_else(|| {
                EarthSnailError::CapacityExceeded("FLOAT64 byte count overflow".into())
            })?;
            if payload.len() < needed {
                return Err(EarthSnailError::Format(
                    "truncated miniSEED FLOAT64 payload".into(),
                ));
            }
            samples.reserve(num_samples);
            for chunk in payload[..needed].chunks_exact(8) {
                let raw = [
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ];
                let v = match payload_order {
                    MseedByteOrder::Big => f64::from_be_bytes(raw),
                    MseedByteOrder::Little => f64::from_le_bytes(raw),
                };
                if !v.is_finite() {
                    return Err(EarthSnailError::Format(
                        "non-finite sample in miniSEED FLOAT64 payload".into(),
                    ));
                }
                samples.push(v);
            }
        }
        MseedEncoding::Steim1 | MseedEncoding::Steim2 => {
            if num_samples > 0 {
                let max_avail_frames = payload.len() / STEIM_FRAME_BYTES;
                let frame_cnt = match b1001_frames {
                    Some(fc) if fc <= max_avail_frames => fc,
                    _ => max_avail_frames,
                };
                let frame_slice = &payload[..frame_cnt * STEIM_FRAME_BYTES];
                let iv = if encoding == MseedEncoding::Steim2 {
                    decode_steim2(frame_slice, num_samples)?
                } else {
                    decode_steim1(frame_slice, num_samples)?
                };
                samples = iv.iter().map(|&x| x as f64).collect();
                int_samples = Some(iv);
            }
        }
    }

    Ok((
        MseedRecord {
            sequence_number,
            quality_indicator,
            channel,
            starttime,
            sampling_rate,
            encoding,
            byte_order: payload_order,
            record_length,
            timing_quality,
            samples,
            int_samples,
            ascii_text,
        },
        record_length,
    ))
}

fn infer_fallback_record_len(rem: &[u8], data_offset: usize) -> Result<usize> {
    for &cand in &[512_usize, 4096, 256, 1024, 2048, 8192] {
        if rem.len() >= cand
            && data_offset <= cand
            && (rem.len() == cand || looks_like_mseed2(&rem[cand..]))
        {
            return Ok(cand);
        }
    }
    Err(EarthSnailError::Header(
        "miniSEED record missing Blockette 1000 and record length could not be inferred".into(),
    ))
}

/// Reassembles decoded `MseedRecord`s into a `Stream`, merging contiguous records per channel
/// while splitting on any time tear exceeding `0.5 * dt` (`Stream::get_gaps` compatible).
pub(crate) fn reassemble_records_to_stream(records: Vec<MseedRecord>) -> Stream {
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
                    Ok(expected_next) => {
                        rec.starttime.duration_secs_f64_since(expected_next).abs() <= 0.5 * dt
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
    Stream::from_traces(finished_traces)
}

fn append_trace_mseed_records(
    trace: &Trace,
    options: MseedWriteOptions,
    seq_counter: &mut u32,
    out: &mut Vec<u8>,
) -> Result<()> {
    let rec_len = options.record_length;
    if !rec_len.is_power_of_two()
        || !(MIN_MSEED_RECORD_LEN..=MAX_MSEED_RECORD_LEN).contains(&rec_len)
    {
        return Err(EarthSnailError::InvalidParameter(format!(
            "miniSEED record_length {rec_len} must be a power of 2 in {MIN_MSEED_RECORD_LEN}..={MAX_MSEED_RECORD_LEN}"
        )));
    }
    if !trace.stats.sampling_rate.is_finite() || trace.stats.sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(format!(
            "invalid sampling_rate {} for miniSEED export",
            trace.stats.sampling_rate
        )));
    }
    if trace.data.is_empty() {
        return Ok(());
    }

    let rec_len_exp = rec_len.trailing_zeros() as u8;
    let (sr_factor, sr_mult, need_b100) = encode_seed_sample_rate(trace.stats.sampling_rate);

    // Header layout:
    // - 48B FSDH (`0..48`)
    // - 8B Blockette 1000 (`48..56`)
    // - 8B Blockette 1001 (`56..64`)
    // - Optional 12B Blockette 100 (`64..76`), padded to 128B for Steim 64B frame alignment or 80B for raw!
    let is_steim = matches!(
        options.encoding,
        MseedEncoding::Steim1 | MseedEncoding::Steim2
    );
    let data_offset = if need_b100 {
        if is_steim {
            128_usize
        } else {
            80_usize
        }
    } else {
        64_usize
    };
    if data_offset >= rec_len {
        return Err(EarthSnailError::InvalidParameter(
            "miniSEED record_length too small for header blockettes".into(),
        ));
    }
    let payload_cap = rec_len - data_offset;

    // Pre-validate/convert integer samples if integer/Steim encoding requested:
    let int_buf: Vec<i32> = match options.encoding {
        MseedEncoding::Int16 => {
            let mut v = Vec::with_capacity(trace.data.len());
            for (idx, &s) in trace.data.iter().enumerate() {
                if !s.is_finite() || s < (i16::MIN as f64) || s > (i16::MAX as f64) {
                    return Err(EarthSnailError::Compression(format!(
                        "sample {s} at index {idx} out of INT16 range"
                    )));
                }
                v.push(math::round(s) as i32);
            }
            v
        }
        MseedEncoding::Int32 | MseedEncoding::Steim1 | MseedEncoding::Steim2 => {
            let mut v = Vec::with_capacity(trace.data.len());
            for (idx, &s) in trace.data.iter().enumerate() {
                if !s.is_finite() || s < (i32::MIN as f64) || s > (i32::MAX as f64) {
                    return Err(EarthSnailError::Compression(format!(
                        "sample {s} at index {idx} out of INT32/Steim range"
                    )));
                }
                v.push(math::round(s) as i32);
            }
            v
        }
        MseedEncoding::Float32 => {
            for (idx, &s) in trace.data.iter().enumerate() {
                if !s.is_finite() || !(s as f32).is_finite() {
                    return Err(EarthSnailError::Compression(format!(
                        "sample {s} at index {idx} out of FLOAT32 range"
                    )));
                }
            }
            Vec::new()
        }
        MseedEncoding::Float64 => {
            for (idx, &s) in trace.data.iter().enumerate() {
                if !s.is_finite() {
                    return Err(EarthSnailError::Compression(format!(
                        "non-finite sample at index {idx}"
                    )));
                }
            }
            Vec::new()
        }
        MseedEncoding::Ascii => {
            return Err(EarthSnailError::Unsupported(
                "ASCII encoding is for log text, not numeric waveform samples".into(),
            ));
        }
    };

    let mut pos = 0_usize;
    let n = trace.data.len();
    let mut prev_steim_sample = 0_i32;

    while pos < n {
        let rec_start_ts = trace
            .stats
            .starttime
            .sample_timestamp(pos as i64, trace.stats.sampling_rate)?;

        let mut record = alloc::vec![0_u8; rec_len];
        let (consumed, active_frames) = match options.encoding {
            MseedEncoding::Int16 => {
                let max_n = (payload_cap / 2).min(u16::MAX as usize);
                let cnt = (n - pos).min(max_n);
                for (i, &val) in int_buf[pos..pos + cnt].iter().enumerate() {
                    let b = match options.byte_order {
                        MseedByteOrder::Big => (val as i16).to_be_bytes(),
                        MseedByteOrder::Little => (val as i16).to_le_bytes(),
                    };
                    let off = data_offset + i * 2;
                    record[off..off + 2].copy_from_slice(&b);
                }
                (cnt, 0_u8)
            }
            MseedEncoding::Int32 => {
                let max_n = (payload_cap / 4).min(u16::MAX as usize);
                let cnt = (n - pos).min(max_n);
                for (i, &val) in int_buf[pos..pos + cnt].iter().enumerate() {
                    let b = match options.byte_order {
                        MseedByteOrder::Big => val.to_be_bytes(),
                        MseedByteOrder::Little => val.to_le_bytes(),
                    };
                    let off = data_offset + i * 4;
                    record[off..off + 4].copy_from_slice(&b);
                }
                (cnt, 0_u8)
            }
            MseedEncoding::Float32 => {
                let max_n = (payload_cap / 4).min(u16::MAX as usize);
                let cnt = (n - pos).min(max_n);
                for (i, &val) in trace.data[pos..pos + cnt].iter().enumerate() {
                    let f = val as f32;
                    let b = match options.byte_order {
                        MseedByteOrder::Big => f.to_be_bytes(),
                        MseedByteOrder::Little => f.to_le_bytes(),
                    };
                    let off = data_offset + i * 4;
                    record[off..off + 4].copy_from_slice(&b);
                }
                (cnt, 0_u8)
            }
            MseedEncoding::Float64 => {
                let max_n = (payload_cap / 8).min(u16::MAX as usize);
                let cnt = (n - pos).min(max_n);
                for (i, &val) in trace.data[pos..pos + cnt].iter().enumerate() {
                    let b = match options.byte_order {
                        MseedByteOrder::Big => val.to_be_bytes(),
                        MseedByteOrder::Little => val.to_le_bytes(),
                    };
                    let off = data_offset + i * 8;
                    record[off..off + 8].copy_from_slice(&b);
                }
                (cnt, 0_u8)
            }
            MseedEncoding::Steim1 => {
                let max_frames = payload_cap / STEIM_FRAME_BYTES;
                let slice_cap = (n - pos).min(u16::MAX as usize);
                let (frame_bytes, cnt, used_frames) = encode_steim1_bounded(
                    &int_buf[pos..pos + slice_cap],
                    prev_steim_sample,
                    max_frames,
                    max_frames,
                )?;
                record[data_offset..data_offset + frame_bytes.len()].copy_from_slice(&frame_bytes);
                prev_steim_sample = int_buf[pos + cnt - 1];
                (cnt, used_frames.min(255) as u8)
            }
            MseedEncoding::Steim2 => {
                let max_frames = payload_cap / STEIM_FRAME_BYTES;
                let slice_cap = (n - pos).min(u16::MAX as usize);
                let (frame_bytes, cnt, used_frames) = encode_steim2_bounded(
                    &int_buf[pos..pos + slice_cap],
                    prev_steim_sample,
                    max_frames,
                    max_frames,
                )?;
                record[data_offset..data_offset + frame_bytes.len()].copy_from_slice(&frame_bytes);
                prev_steim_sample = int_buf[pos + cnt - 1];
                (cnt, used_frames.min(255) as u8)
            }
            MseedEncoding::Ascii => unreachable!(),
        };

        // Write 48B FSDH + Blockettes
        let seq_str = format!("{:06}", (*seq_counter) % 1_000_000);
        *seq_counter = seq_counter.wrapping_add(1).max(1);
        record[0..6].copy_from_slice(seq_str.as_bytes());
        record[6] = if matches!(options.quality_indicator, 'D' | 'R' | 'Q' | 'M') {
            options.quality_indicator as u8
        } else {
            b'D'
        };
        record[7] = b' ';
        write_seed_ascii(&mut record[8..13], &trace.stats.channel.station);
        write_seed_ascii(&mut record[13..15], &trace.stats.channel.location);
        write_seed_ascii(&mut record[15..18], &trace.stats.channel.channel);
        write_seed_ascii(&mut record[18..20], &trace.stats.channel.network);

        // Decompose `rec_start_ts` into SEED BTime (`0.1 ms`) + Blockette 1001 `micro_sec` (`0..=99 us`)
        // rounding to nearest microsecond cleanly.
        let rounded_ts = round_ts_to_micros(rec_start_ts);
        let (year, jday, hour, min, sec, nano) = rounded_ts.to_year_jday_hms_nano();
        let total_us = nano / 1_000;
        let tenth_millis = (total_us / 100) as u16;
        let micro_sec = (total_us % 100) as i8;

        let hdr_bo = options.byte_order;
        write_u16(&mut record, 20, year.clamp(0, 65535) as u16, hdr_bo);
        write_u16(&mut record, 22, jday, hdr_bo);
        record[24] = hour;
        record[25] = min;
        record[26] = sec;
        record[27] = 0;
        write_u16(&mut record, 28, tenth_millis, hdr_bo);
        write_u16(&mut record, 30, consumed as u16, hdr_bo);
        write_i16(&mut record, 32, sr_factor, hdr_bo);
        write_i16(&mut record, 34, sr_mult, hdr_bo);
        record[36] = 0;
        record[37] = 0;
        record[38] = 0;
        record[39] = if need_b100 { 3 } else { 2 };
        write_i32(&mut record, 40, 0, hdr_bo);
        write_u16(&mut record, 44, data_offset as u16, hdr_bo);
        write_u16(&mut record, 46, 48, hdr_bo);

        // Blockette 1000 at `48..56`
        write_u16(&mut record, 48, 1000, hdr_bo);
        write_u16(&mut record, 50, 56, hdr_bo);
        record[52] = options.encoding as u8;
        let payload_bo_flag = if is_steim {
            1_u8 // Steim frames are always big-endian
        } else {
            match options.byte_order {
                MseedByteOrder::Big => 1_u8,
                MseedByteOrder::Little => 0_u8,
            }
        };
        record[53] = payload_bo_flag;
        record[54] = rec_len_exp;
        record[55] = 0;

        // Blockette 1001 at `56..64`
        write_u16(&mut record, 56, 1001, hdr_bo);
        write_u16(&mut record, 58, if need_b100 { 64 } else { 0 }, hdr_bo);
        record[60] = 100; // timing quality 100%
        record[61] = micro_sec as u8;
        record[62] = 0;
        record[63] = active_frames;

        // Optional Blockette 100 at `64..76`
        if need_b100 {
            write_u16(&mut record, 64, 100, hdr_bo);
            write_u16(&mut record, 66, 0, hdr_bo);
            write_f32(&mut record, 68, trace.stats.sampling_rate as f32, hdr_bo);
            record[72] = 0;
            record[73] = 0;
            record[74] = 0;
            record[75] = 0;
        }

        out.extend_from_slice(&record);
        pos += consumed;
    }

    Ok(())
}

#[inline]
fn round_ts_to_micros(ts: UtcTimestamp) -> UtcTimestamp {
    let ns = ts.as_epoch_nanos();
    let rem = ns.rem_euclid(1_000);
    if rem >= 500 {
        UtcTimestamp::from_epoch_nanos(ns + (1_000 - rem))
    } else {
        UtcTimestamp::from_epoch_nanos(ns - rem)
    }
}

/// Evaluates SEED v2.4 nominal sample rate from `(factor, multiplier)`.
pub fn decode_seed_sample_rate(factor: i16, multiplier: i16) -> f64 {
    let f = factor as f64;
    let m = multiplier as f64;
    if factor > 0 && multiplier > 0 {
        f * m
    } else if factor > 0 && multiplier < 0 {
        -f / m
    } else if factor < 0 && multiplier > 0 {
        -m / f
    } else if factor < 0 && multiplier < 0 {
        1.0 / (f * m)
    } else {
        0.0
    }
}

/// Encodes `fs` Hz into SEED `(factor, multiplier, needs_blockette_100)`.
pub fn encode_seed_sample_rate(fs: f64) -> (i16, i16, bool) {
    if fs >= 1.0 {
        let r = math::round(fs);
        if r <= 32_767.0 && (fs - r).abs() <= 1e-9 * fs {
            return (r as i16, 1, false);
        }
        for &den in &[10_i32, 100, 1_000] {
            let num = math::round(fs * (den as f64));
            if (1.0..=32_767.0).contains(&num) {
                let approx = num / (den as f64);
                let need_b100 = (approx - fs).abs() > 1e-7 * fs;
                return (num as i16, -(den as i16), need_b100);
            }
        }
        (math::round(fs).clamp(1.0, 32_767.0) as i16, 1, true)
    } else if fs > 0.0 {
        let period = 1.0 / fs;
        let r = math::round(period);
        if (1.0..=32_767.0).contains(&r) && (1.0 / r - fs).abs() <= 1e-9 * fs {
            return (-(r as i16), 1, false);
        }
        let num = math::round(fs * 10_000.0).clamp(1.0, 32_767.0);
        let approx = num / 10_000.0;
        let need_b100 = (approx - fs).abs() > 1e-7 * fs;
        (num as i16, -10_000, need_b100)
    } else {
        (0, 0, false)
    }
}

fn clean_seed_ascii(slot: &[u8]) -> String {
    let mut s = String::with_capacity(slot.len());
    for &b in slot {
        if b == 0 {
            break;
        }
        if b.is_ascii() && b != b' ' {
            s.push(b as char);
        }
    }
    s
}

fn write_seed_ascii(slot: &mut [u8], text: &str) {
    slot.fill(b' ');
    let trimmed = text.trim();
    let src = trimmed.as_bytes();
    let n = src.len().min(slot.len());
    slot[..n].copy_from_slice(&src[..n]);
}

#[inline]
fn read_u16(b: &[u8], off: usize, bo: MseedByteOrder) -> u16 {
    let arr = [b[off], b[off + 1]];
    match bo {
        MseedByteOrder::Big => u16::from_be_bytes(arr),
        MseedByteOrder::Little => u16::from_le_bytes(arr),
    }
}

#[inline]
fn read_i16(b: &[u8], off: usize, bo: MseedByteOrder) -> i16 {
    let arr = [b[off], b[off + 1]];
    match bo {
        MseedByteOrder::Big => i16::from_be_bytes(arr),
        MseedByteOrder::Little => i16::from_le_bytes(arr),
    }
}

#[inline]
fn read_i32(b: &[u8], off: usize, bo: MseedByteOrder) -> i32 {
    let arr = [b[off], b[off + 1], b[off + 2], b[off + 3]];
    match bo {
        MseedByteOrder::Big => i32::from_be_bytes(arr),
        MseedByteOrder::Little => i32::from_le_bytes(arr),
    }
}

#[inline]
fn read_f32(b: &[u8], off: usize, bo: MseedByteOrder) -> f32 {
    let arr = [b[off], b[off + 1], b[off + 2], b[off + 3]];
    match bo {
        MseedByteOrder::Big => f32::from_be_bytes(arr),
        MseedByteOrder::Little => f32::from_le_bytes(arr),
    }
}

#[inline]
fn write_u16(b: &mut [u8], off: usize, val: u16, bo: MseedByteOrder) {
    let arr = match bo {
        MseedByteOrder::Big => val.to_be_bytes(),
        MseedByteOrder::Little => val.to_le_bytes(),
    };
    b[off..off + 2].copy_from_slice(&arr);
}

#[inline]
fn write_i16(b: &mut [u8], off: usize, val: i16, bo: MseedByteOrder) {
    let arr = match bo {
        MseedByteOrder::Big => val.to_be_bytes(),
        MseedByteOrder::Little => val.to_le_bytes(),
    };
    b[off..off + 2].copy_from_slice(&arr);
}

#[inline]
fn write_i32(b: &mut [u8], off: usize, val: i32, bo: MseedByteOrder) {
    let arr = match bo {
        MseedByteOrder::Big => val.to_be_bytes(),
        MseedByteOrder::Little => val.to_le_bytes(),
    };
    b[off..off + 4].copy_from_slice(&arr);
}

#[inline]
fn write_f32(b: &mut [u8], off: usize, val: f32, bo: MseedByteOrder) {
    let arr = match bo {
        MseedByteOrder::Big => val.to_be_bytes(),
        MseedByteOrder::Little => val.to_le_bytes(),
    };
    b[off..off + 4].copy_from_slice(&arr);
}
