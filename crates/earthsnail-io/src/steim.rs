//! Lossless Steim-1 (SEED encoding `10`) and Steim-2 (SEED encoding `11`) waveform compression
//! and decompression codecs operating on 64-byte big-endian frames with `X0` (forward) and
//! `Xn` (reverse) integration constant verification.

use alloc::format;
use alloc::vec::Vec;
use earthsnail_core::{EarthSnailError, Result};

/// Size of a single Steim compression frame in bytes (`16` 32-bit words = `64` bytes).
pub const STEIM_FRAME_BYTES: usize = 64;

/// Maximum samples accepted per single Steim decode call to prevent untrusted allocation bombs.
pub const MAX_STEIM_SAMPLES: usize = 256 * 1024 * 1024;

/// Decodes `num_samples` signed 32-bit integers from `frames` encoded in **Steim-1** (code `10`).
///
/// Verifies both frame bounds and the reverse integration constant `Xn` stored in Frame 0 `W[2]`.
pub fn decode_steim1(frames: &[u8], num_samples: usize) -> Result<Vec<i32>> {
    decode_steim_inner(frames, num_samples, false)
}

/// Decodes `num_samples` signed 32-bit integers from `frames` encoded in **Steim-2** (code `11`).
///
/// Supports all 7 Steim-2 difference widths (`4`-bit x7, `5`-bit x6, `6`-bit x5, `8`-bit x4,
/// `10`-bit x3, `15`-bit x2, `30`-bit x1) and verifies `X0` and `Xn`.
pub fn decode_steim2(frames: &[u8], num_samples: usize) -> Result<Vec<i32>> {
    decode_steim_inner(frames, num_samples, true)
}

fn decode_steim_inner(frames: &[u8], num_samples: usize, is_steim2: bool) -> Result<Vec<i32>> {
    if num_samples == 0 {
        return Ok(Vec::new());
    }
    if num_samples > MAX_STEIM_SAMPLES {
        return Err(EarthSnailError::CapacityExceeded(format!(
            "Steim sample count {num_samples} exceeds safety bound {MAX_STEIM_SAMPLES}"
        )));
    }
    if frames.len() < STEIM_FRAME_BYTES {
        return Err(EarthSnailError::Compression(format!(
            "Steim payload shorter than one 64-byte frame ({} bytes)",
            frames.len()
        )));
    }
    if frames.len() % STEIM_FRAME_BYTES != 0 {
        return Err(EarthSnailError::Compression(format!(
            "Steim payload length {} is not a multiple of 64 bytes",
            frames.len()
        )));
    }

    // Upper bound on possible samples in `num_frames` (at most 7 samples/word * 15 words/frame):
    let num_frames = frames.len() / STEIM_FRAME_BYTES;
    let max_possible = num_frames.saturating_mul(15 * 7);
    if num_samples > max_possible {
        return Err(EarthSnailError::Compression(format!(
            "Steim header claims {num_samples} samples, exceeding capacity {max_possible} of {num_frames} frames"
        )));
    }

    let x0 = read_be_i32(frames, 4);
    let xn = read_be_i32(frames, 8);

    let mut samples = Vec::with_capacity(num_samples);
    let mut current: i32 = x0;
    let mut diff_index: usize = 0;

    let mut scratch_diffs = [0_i32; 7];

    'frame_loop: for (f_idx, frame) in frames.chunks_exact(STEIM_FRAME_BYTES).enumerate() {
        let w0 = read_be_u32(frame, 0);
        for word_idx in 1..16 {
            let c = (w0 >> (30 - 2 * word_idx)) & 0b11;
            if f_idx == 0 && (word_idx == 1 || word_idx == 2) {
                // Words 1 and 2 of Frame 0 hold X0 and Xn (`c` must be `00`, though some legacy
                // encoders ignore nibbles 1..2; always treat Frame 0 W1/W2 as X0/Xn).
                continue;
            }
            if c == 0b00 {
                continue;
            }
            let w = read_be_u32(frame, word_idx * 4);
            let count = if is_steim2 {
                unpack_steim2_word(c, w, &mut scratch_diffs)?
            } else {
                unpack_steim1_word(c, w, &mut scratch_diffs)
            };

            for &d in &scratch_diffs[..count] {
                if diff_index == 0 {
                    // `d[0]` is `X[0] - X[-1]`; `X0` gives the exact starting sample directly.
                    samples.push(x0);
                    current = x0;
                } else {
                    current = current.wrapping_add(d);
                    samples.push(current);
                }
                diff_index += 1;
                if samples.len() == num_samples {
                    break 'frame_loop;
                }
            }
        }
    }

    if samples.len() != num_samples {
        return Err(EarthSnailError::Compression(format!(
            "Steim truncated stream: decoded {} samples, expected {num_samples}",
            samples.len()
        )));
    }

    let last = *samples.last().unwrap_or(&x0);
    if last != xn {
        return Err(EarthSnailError::Compression(format!(
            "Steim-{} reverse integration constant mismatch: last sample {last} != Xn {xn}",
            if is_steim2 { 2 } else { 1 }
        )));
    }

    Ok(samples)
}

#[inline]
fn unpack_steim1_word(c: u32, w: u32, out: &mut [i32; 7]) -> usize {
    match c {
        0b01 => {
            out[0] = ((w >> 24) as u8 as i8) as i32;
            out[1] = ((w >> 16) as u8 as i8) as i32;
            out[2] = ((w >> 8) as u8 as i8) as i32;
            out[3] = (w as u8 as i8) as i32;
            4
        }
        0b10 => {
            out[0] = ((w >> 16) as u16 as i16) as i32;
            out[1] = (w as u16 as i16) as i32;
            2
        }
        0b11 => {
            out[0] = w as i32;
            1
        }
        _ => 0,
    }
}

#[inline]
fn unpack_steim2_word(c: u32, w: u32, out: &mut [i32; 7]) -> Result<usize> {
    match c {
        0b01 => {
            out[0] = ((w >> 24) as u8 as i8) as i32;
            out[1] = ((w >> 16) as u8 as i8) as i32;
            out[2] = ((w >> 8) as u8 as i8) as i32;
            out[3] = (w as u8 as i8) as i32;
            Ok(4)
        }
        0b10 => {
            let dnib = (w >> 30) & 0b11;
            match dnib {
                0b01 => {
                    // 1 x 30-bit signed difference
                    out[0] = sign_extend_u32(w & 0x3FFF_FFFF, 30);
                    Ok(1)
                }
                0b10 => {
                    // 2 x 15-bit signed differences
                    out[0] = sign_extend_u32((w >> 15) & 0x7FFF, 15);
                    out[1] = sign_extend_u32(w & 0x7FFF, 15);
                    Ok(2)
                }
                0b11 => {
                    // 3 x 10-bit signed differences
                    out[0] = sign_extend_u32((w >> 20) & 0x03FF, 10);
                    out[1] = sign_extend_u32((w >> 10) & 0x03FF, 10);
                    out[2] = sign_extend_u32(w & 0x03FF, 10);
                    Ok(3)
                }
                _ => Err(EarthSnailError::Compression(
                    "invalid Steim-2 nibble combination c=0b10, dnib=0b00".into(),
                )),
            }
        }
        0b11 => {
            let dnib = (w >> 30) & 0b11;
            match dnib {
                0b00 => {
                    // 5 x 6-bit signed differences
                    for (i, slot) in out[..5].iter_mut().enumerate() {
                        let shift = 24 - i * 6;
                        *slot = sign_extend_u32((w >> shift) & 0x3F, 6);
                    }
                    Ok(5)
                }
                0b01 => {
                    // 6 x 5-bit signed differences
                    for (i, slot) in out[..6].iter_mut().enumerate() {
                        let shift = 25 - i * 5;
                        *slot = sign_extend_u32((w >> shift) & 0x1F, 5);
                    }
                    Ok(6)
                }
                0b10 => {
                    // 7 x 4-bit signed differences in bits [27..0]
                    for (i, slot) in out[..7].iter_mut().enumerate() {
                        let shift = 24 - i * 4;
                        *slot = sign_extend_u32((w >> shift) & 0x0F, 4);
                    }
                    Ok(7)
                }
                _ => Err(EarthSnailError::Compression(
                    "invalid Steim-2 nibble combination c=0b11, dnib=0b11".into(),
                )),
            }
        }
        _ => Ok(0),
    }
}

/// Encodes an entire `samples` slice into 64-byte **Steim-1** frames (`prev_sample` defaults to `0`
/// for the first record of a trace).
pub fn encode_steim1(samples: &[i32], prev_sample: i32) -> Vec<u8> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut pos = 0;
    let mut prev = prev_sample;
    while pos < samples.len() {
        // Encode up to 64 frames at a time or remaining samples; for unbounded standalone encode,
        // pack all samples into a single contiguous frame sequence with one Frame 0 (`X0`, `Xn`).
        let (chunk_bytes, packed) = encode_steim1_record_frames(&samples[pos..], prev, usize::MAX);
        prev = samples[pos + packed - 1];
        pos += packed;
        out.extend_from_slice(&chunk_bytes);
    }
    out
}

/// Encodes up to `max_frames` 64-byte **Steim-1** frames from the prefix of `samples`, returning
/// `(frame_bytes, samples_encoded)`.
///
/// Pads out to `pad_to_frames` (with zeroed frames) if `pad_to_frames > used_frames`, as required
/// by fixed-length miniSEED v2 records.
pub fn encode_steim1_bounded(
    samples: &[i32],
    prev_sample: i32,
    max_frames: usize,
    pad_to_frames: usize,
) -> Result<(Vec<u8>, usize, usize)> {
    if samples.is_empty() || max_frames == 0 {
        return Ok((Vec::new(), 0, 0));
    }
    let (mut bytes, count) = encode_steim1_record_frames(samples, prev_sample, max_frames);
    let used_frames = bytes.len() / STEIM_FRAME_BYTES;
    if pad_to_frames > used_frames {
        bytes.resize(pad_to_frames * STEIM_FRAME_BYTES, 0);
    }
    Ok((bytes, count, used_frames))
}

fn encode_steim1_record_frames(
    samples: &[i32],
    prev_sample: i32,
    max_frames: usize,
) -> (Vec<u8>, usize) {
    let mut frames: Vec<[u32; 16]> = Vec::new();
    let mut pos = 0_usize;
    let n = samples.len();

    while pos < n && frames.len() < max_frames {
        let f_idx = frames.len();
        let mut words = [0_u32; 16];
        let mut w0 = 0_u32;
        let start_word = if f_idx == 0 { 3 } else { 1 };

        for (word_idx, slot) in words.iter_mut().enumerate().skip(start_word) {
            if pos >= n {
                break;
            }
            let rem = n - pos;
            let (c, packed_word, consumed) = pack_next_steim1_word(samples, pos, prev_sample, rem);
            w0 |= c << (30 - 2 * word_idx);
            *slot = packed_word;
            pos += consumed;
        }
        words[0] = w0;
        frames.push(words);
    }

    // Populate Frame 0 `X0` (`W[1]`) and `Xn` (`W[2]`) from the exact slice encoded in this record!
    frames[0][1] = samples[0] as u32;
    frames[0][2] = samples[pos - 1] as u32;

    let mut out = Vec::with_capacity(frames.len() * STEIM_FRAME_BYTES);
    for frame in &frames {
        for &w in frame {
            out.extend_from_slice(&w.to_be_bytes());
        }
    }
    (out, pos)
}

#[inline]
fn steim_diff_i64(samples: &[i32], idx: usize, prev_sample: i32) -> i64 {
    if idx == 0 {
        (samples[0] as i64) - (prev_sample as i64)
    } else {
        (samples[idx] as i64) - (samples[idx - 1] as i64)
    }
}

#[inline]
fn pack_next_steim1_word(
    samples: &[i32],
    pos: usize,
    prev_sample: i32,
    rem: usize,
) -> (u32, u32, usize) {
    if rem >= 4 {
        let d0 = steim_diff_i64(samples, pos, prev_sample);
        let d1 = steim_diff_i64(samples, pos + 1, prev_sample);
        let d2 = steim_diff_i64(samples, pos + 2, prev_sample);
        let d3 = steim_diff_i64(samples, pos + 3, prev_sample);
        if (-128..=127).contains(&d0)
            && (-128..=127).contains(&d1)
            && (-128..=127).contains(&d2)
            && (-128..=127).contains(&d3)
        {
            let w = (((d0 as i8 as u8) as u32) << 24)
                | (((d1 as i8 as u8) as u32) << 16)
                | (((d2 as i8 as u8) as u32) << 8)
                | ((d3 as i8 as u8) as u32);
            return (0b01, w, 4);
        }
    }
    if rem >= 2 {
        let d0 = steim_diff_i64(samples, pos, prev_sample);
        let d1 = steim_diff_i64(samples, pos + 1, prev_sample);
        if (-32_768..=32_767).contains(&d0) && (-32_768..=32_767).contains(&d1) {
            let w = (((d0 as i16 as u16) as u32) << 16) | ((d1 as i16 as u16) as u32);
            return (0b10, w, 2);
        }
    }
    // 1 x 32-bit difference (`wrapping_sub` preserves arbitrary jumps across full `i32` domain!)
    let d0_wrapping = if pos == 0 {
        samples[0].wrapping_sub(prev_sample)
    } else {
        samples[pos].wrapping_sub(samples[pos - 1])
    };
    (0b11, d0_wrapping as u32, 1)
}

/// Encodes an entire `samples` slice into 64-byte **Steim-2** frames (`prev_sample` defaults to `0`).
///
/// Returns `Err(EarthSnailError::Compression)` if any consecutive difference exceeds the 30-bit
/// signed Steim-2 range `[-536_870_912, +536_870_911]`. Note that for `idx == 0`, `prev_sample`
/// is set to `samples[0]` when `|samples[0] - prev_sample|` would exceed 30 bits, because `X0`
/// independently anchors `samples[0]` in every Steim record!
pub fn encode_steim2(samples: &[i32], prev_sample: i32) -> Result<Vec<u8>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let (bytes, packed, _) = encode_steim2_bounded(samples, prev_sample, usize::MAX, 0)?;
    debug_assert_eq!(packed, samples.len());
    Ok(bytes)
}

/// Encodes up to `max_frames` 64-byte **Steim-2** frames from the prefix of `samples`, returning
/// `(frame_bytes, samples_encoded, active_frame_count)`.
pub fn encode_steim2_bounded(
    samples: &[i32],
    prev_sample: i32,
    max_frames: usize,
    pad_to_frames: usize,
) -> Result<(Vec<u8>, usize, usize)> {
    if samples.is_empty() || max_frames == 0 {
        return Ok((Vec::new(), 0, 0));
    }

    // Anchor `d[0]`: if `samples[0] - prev_sample` exceeds 30-bit signed range (e.g. initial offset
    // `samples[0] = 1_500_000_000` with `prev_sample = 0`), `d[0] = 0` is 100% valid because
    // decoders always initialize sample 0 from `X0 = samples[0]`.
    let raw_d0 = (samples[0] as i64) - (prev_sample as i64);
    let eff_prev = if (-536_870_912..=536_870_911).contains(&raw_d0) {
        prev_sample
    } else {
        samples[0]
    };

    let mut frames: Vec<[u32; 16]> = Vec::new();
    let mut pos = 0_usize;
    let n = samples.len();

    while pos < n && frames.len() < max_frames {
        let f_idx = frames.len();
        let mut words = [0_u32; 16];
        let mut w0 = 0_u32;
        let start_word = if f_idx == 0 { 3 } else { 1 };

        for (word_idx, slot) in words.iter_mut().enumerate().skip(start_word) {
            if pos >= n {
                break;
            }
            let rem = n - pos;
            let (c, packed_word, consumed) = pack_next_steim2_word(samples, pos, eff_prev, rem)?;
            w0 |= c << (30 - 2 * word_idx);
            *slot = packed_word;
            pos += consumed;
        }
        words[0] = w0;
        frames.push(words);
    }

    frames[0][1] = samples[0] as u32;
    frames[0][2] = samples[pos - 1] as u32;

    let used_frames = frames.len();
    let total_frames = used_frames.max(pad_to_frames);
    let mut out = Vec::with_capacity(total_frames * STEIM_FRAME_BYTES);
    for frame in &frames {
        for &w in frame {
            out.extend_from_slice(&w.to_be_bytes());
        }
    }
    if total_frames > used_frames {
        out.resize(total_frames * STEIM_FRAME_BYTES, 0);
    }
    Ok((out, pos, used_frames))
}

fn pack_next_steim2_word(
    samples: &[i32],
    pos: usize,
    prev_sample: i32,
    rem: usize,
) -> Result<(u32, u32, usize)> {
    // 1. Try 7 x 4-bit signed [-8, 7] (`c = 0b11`, `dnib = 0b10`)
    if rem >= 7 && all_diffs_in_range(samples, pos, prev_sample, 7, -8, 7) {
        let mut w = 0b10_u32 << 30;
        for i in 0..7 {
            let d = steim_diff_i64(samples, pos + i, prev_sample) as i32;
            w |= ((d as u32) & 0x0F) << (24 - i * 4);
        }
        return Ok((0b11, w, 7));
    }

    // 2. Try 6 x 5-bit signed [-16, 15] (`c = 0b11`, `dnib = 0b01`)
    if rem >= 6 && all_diffs_in_range(samples, pos, prev_sample, 6, -16, 15) {
        let mut w = 0b01_u32 << 30;
        for i in 0..6 {
            let d = steim_diff_i64(samples, pos + i, prev_sample) as i32;
            w |= ((d as u32) & 0x1F) << (25 - i * 5);
        }
        return Ok((0b11, w, 6));
    }

    // 3. Try 5 x 6-bit signed [-32, 31] (`c = 0b11`, `dnib = 0b00`)
    if rem >= 5 && all_diffs_in_range(samples, pos, prev_sample, 5, -32, 31) {
        let mut w = 0b00_u32 << 30;
        for i in 0..5 {
            let d = steim_diff_i64(samples, pos + i, prev_sample) as i32;
            w |= ((d as u32) & 0x3F) << (24 - i * 6);
        }
        return Ok((0b11, w, 5));
    }

    // 4. Try 4 x 8-bit signed [-128, 127] (`c = 0b01`, no dnib)
    if rem >= 4 && all_diffs_in_range(samples, pos, prev_sample, 4, -128, 127) {
        let mut w = 0_u32;
        for i in 0..4 {
            let d = steim_diff_i64(samples, pos + i, prev_sample) as i8 as u8;
            w |= (d as u32) << (24 - i * 8);
        }
        return Ok((0b01, w, 4));
    }

    // 5. Try 3 x 10-bit signed [-512, 511] (`c = 0b10`, `dnib = 0b11`)
    if rem >= 3 && all_diffs_in_range(samples, pos, prev_sample, 3, -512, 511) {
        let mut w = 0b11_u32 << 30;
        for i in 0..3 {
            let d = steim_diff_i64(samples, pos + i, prev_sample) as i32;
            w |= ((d as u32) & 0x03FF) << (20 - i * 10);
        }
        return Ok((0b10, w, 3));
    }

    // 6. Try 2 x 15-bit signed [-16_384, 16_383] (`c = 0b10`, `dnib = 0b10`)
    if rem >= 2 && all_diffs_in_range(samples, pos, prev_sample, 2, -16_384, 16_383) {
        let d0 = steim_diff_i64(samples, pos, prev_sample) as i32;
        let d1 = steim_diff_i64(samples, pos + 1, prev_sample) as i32;
        let w = (0b10_u32 << 30) | (((d0 as u32) & 0x7FFF) << 15) | ((d1 as u32) & 0x7FFF);
        return Ok((0b10, w, 2));
    }

    // 7. 1 x 30-bit signed [-536_870_912, 536_870_911] (`c = 0b10`, `dnib = 0b01`)
    let d0 = steim_diff_i64(samples, pos, prev_sample);
    if !(-536_870_912..=536_870_911).contains(&d0) {
        return Err(EarthSnailError::Compression(format!(
            "first difference {d0} at sample {pos} exceeds Steim-2 30-bit signed range [-536870912, 536870911]"
        )));
    }
    let w = (0b01_u32 << 30) | ((d0 as i32 as u32) & 0x3FFF_FFFF);
    Ok((0b10, w, 1))
}

#[inline]
fn all_diffs_in_range(
    samples: &[i32],
    pos: usize,
    prev_sample: i32,
    count: usize,
    lo: i64,
    hi: i64,
) -> bool {
    for i in 0..count {
        let d = steim_diff_i64(samples, pos + i, prev_sample);
        if d < lo || d > hi {
            return false;
        }
    }
    true
}

#[inline]
fn sign_extend_u32(raw: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((raw << shift) as i32) >> shift
}

#[inline]
fn read_be_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn read_be_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}
