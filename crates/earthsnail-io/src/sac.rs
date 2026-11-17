//! SAC Binary (`v6` 632-byte header + `f32` waveform, and `v7` optional 22-`f64` precision footer)
//! and SAC Alphanumeric (`SACXY` / `alpha` text) readers and writers with automatic endianness
//! sensing, `-12345` sentinel preservation, `LCALDA` geodesic updates, and bounded allocations.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use earthsnail_core::sac_header::{SAC_FLOAT_COUNT, SAC_INT_COUNT, SAC_V7_FOOTER_DOUBLES};
use earthsnail_core::{
    is_sac_float_undef, is_sac_int_undef, math, ChannelId, EarthSnailError, Result, SacEndian,
    SacFileType, SacHeader, Trace, TraceStats, UtcTimestamp, SAC_FLOAT_UNDEF, SAC_FLOAT_UNDEF_F32,
    SAC_HEADER_BYTES,
};

/// Upper bound on samples accepted from an untrusted SAC header (`256M` samples = `1 GiB` of `f32`).
pub const MAX_SAC_SAMPLES: usize = 256 * 1024 * 1024;

/// Size in bytes of the optional SAC v7 (`NVHDR == 7`) 22-double precision footer (`176` bytes).
pub const SAC_V7_FOOTER_BYTES: usize = SAC_V7_FOOTER_DOUBLES * 8;

/// Decodes a SAC Binary (`v6` or `v7`, Little-Endian or Big-Endian auto-sensed) byte buffer
/// into a `(Trace, SacEndian)` pair.
pub fn read_sac_bytes_with_endian(bytes: &[u8]) -> Result<(Trace, SacEndian)> {
    if bytes.len() < SAC_HEADER_BYTES {
        return Err(EarthSnailError::Format(format!(
            "SAC binary buffer too short ({} < {} bytes)",
            bytes.len(),
            SAC_HEADER_BYTES
        )));
    }

    let endian = SacHeader::detect_endian(bytes).ok_or_else(|| {
        EarthSnailError::Format(
            "unrecognized SAC binary header: NVHDR not in 1..=7 in either endianness".into(),
        )
    })?;

    let mut hdr_arr = [0_u8; SAC_HEADER_BYTES];
    hdr_arr.copy_from_slice(&bytes[..SAC_HEADER_BYTES]);
    let mut hdr = SacHeader::from_bytes(&hdr_arr, endian)?;

    if is_sac_int_undef(hdr.npts) || hdr.npts < 0 {
        return Err(EarthSnailError::Header(format!(
            "invalid or negative SAC NPTS: {}",
            hdr.npts
        )));
    }
    let npts = hdr.npts as usize;
    if npts > MAX_SAC_SAMPLES {
        return Err(EarthSnailError::CapacityExceeded(format!(
            "SAC NPTS {npts} exceeds safety limit {MAX_SAC_SAMPLES}"
        )));
    }

    let primary_bytes = npts
        .checked_mul(4)
        .ok_or_else(|| EarthSnailError::CapacityExceeded("SAC data byte size overflow".into()))?;
    let data_end = SAC_HEADER_BYTES
        .checked_add(primary_bytes)
        .ok_or_else(|| EarthSnailError::CapacityExceeded("SAC offset overflow".into()))?;

    if bytes.len() < data_end {
        return Err(EarthSnailError::Format(format!(
            "truncated SAC binary data: expected at least {data_end} bytes for NPTS={npts}, got {}",
            bytes.len()
        )));
    }

    // Determine whether a second data section precedes an optional v7 footer
    // (uneven IXY or 2-component spectral IRLIM/IAMPH).
    let two_sections = hdr.leven == 0
        || matches!(
            SacFileType::from_i32(hdr.iftype),
            Some(
                SacFileType::SpectralRealImag
                    | SacFileType::SpectralAmpPhase
                    | SacFileType::GeneralXY
            )
        );

    let v7_footer_offset = if hdr.nvhdr == 7 {
        let two_sec_end = data_end.saturating_add(primary_bytes);
        if two_sections && bytes.len() >= two_sec_end.saturating_add(SAC_V7_FOOTER_BYTES) {
            Some(two_sec_end)
        } else if bytes.len() >= data_end.saturating_add(SAC_V7_FOOTER_BYTES) {
            Some(data_end)
        } else {
            return Err(EarthSnailError::Format(format!(
                "SAC header specifies NVHDR=7 but buffer ({} bytes) lacks 176-byte v7 footer",
                bytes.len()
            )));
        }
    } else {
        None
    };

    if let Some(foff) = v7_footer_offset {
        let mut footer = [0.0_f64; SAC_V7_FOOTER_DOUBLES];
        for (i, slot) in footer.iter_mut().enumerate() {
            let b_off = foff + i * 8;
            let mut raw = [0_u8; 8];
            raw.copy_from_slice(&bytes[b_off..b_off + 8]);
            let val = match endian {
                SacEndian::Little => f64::from_le_bytes(raw),
                SacEndian::Big => f64::from_be_bytes(raw),
            };
            *slot = val;
        }
        hdr.apply_v7_footer(&footer)?;
    }

    // Now safe to allocate `npts` samples because `bytes.len() >= SAC_HEADER_BYTES + 4 * npts` was proven.
    let mut data = Vec::with_capacity(npts);
    let data_slice = &bytes[SAC_HEADER_BYTES..data_end];
    for (idx, chunk) in data_slice.chunks_exact(4).enumerate() {
        let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let val = match endian {
            SacEndian::Little => f32::from_le_bytes(raw),
            SacEndian::Big => f32::from_be_bytes(raw),
        };
        if !val.is_finite() {
            return Err(EarthSnailError::Format(format!(
                "non-finite sample at index {idx} in SAC binary waveform"
            )));
        }
        data.push(val as f64);
    }

    finalize_sac_header_on_read(&mut hdr);
    let stats = build_trace_stats_from_sac(hdr)?;
    Ok((Trace::new(stats, data), endian))
}

/// Decodes a SAC Binary (`v6` or `v7`, auto-sensing endianness) byte buffer into a `Trace`.
pub fn read_sac_bytes(bytes: &[u8]) -> Result<Trace> {
    read_sac_bytes_with_endian(bytes).map(|(tr, _)| tr)
}

/// Encodes a `Trace` into SAC Binary format (`v6`, or `v7` with 22-`f64` footer when
/// `trace.stats.sac` has `nvhdr == 7`) in the requested `endian` byte order.
pub fn write_sac_bytes(trace: &Trace, endian: SacEndian) -> Result<Vec<u8>> {
    if trace.data.len() > (i32::MAX as usize) {
        return Err(EarthSnailError::CapacityExceeded(format!(
            "trace length {} exceeds SAC i32 NPTS maximum",
            trace.data.len()
        )));
    }
    for (idx, &sample) in trace.data.iter().enumerate() {
        if !sample.is_finite() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "non-finite sample at index {idx} cannot be written to SAC"
            )));
        }
        let as_f32 = sample as f32;
        if !as_f32.is_finite() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "sample {sample} at index {idx} overflows f32 in SAC binary"
            )));
        }
    }

    let hdr = prepare_sac_header_for_write(trace)?;
    let include_v7 = hdr.nvhdr == 7;
    let footer_len = if include_v7 { SAC_V7_FOOTER_BYTES } else { 0 };
    let total_len = SAC_HEADER_BYTES + trace.data.len() * 4 + footer_len;

    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&hdr.to_bytes(endian));

    for &sample in &trace.data {
        let f = sample as f32;
        let b = match endian {
            SacEndian::Little => f.to_le_bytes(),
            SacEndian::Big => f.to_be_bytes(),
        };
        out.extend_from_slice(&b);
    }

    if include_v7 {
        let footer = hdr.to_v7_footer();
        for &d in &footer {
            let b = match endian {
                SacEndian::Little => d.to_le_bytes(),
                SacEndian::Big => d.to_be_bytes(),
            };
            out.extend_from_slice(&b);
        }
    }

    Ok(out)
}

/// Parses a SAC Alphanumeric (`SACXY` / `alpha`) text string into a `Trace`.
pub fn read_sac_alpha_str(input: &str) -> Result<Trace> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut lines = input.lines();

    // 1. 14 lines of 5 floats (70 floats total)
    let mut floats = [SAC_FLOAT_UNDEF_F32; SAC_FLOAT_COUNT];
    for row in 0..14 {
        let line = lines.next().ok_or_else(|| {
            EarthSnailError::Format(format!(
                "truncated SAC alpha header at float row {}",
                row + 1
            ))
        })?;
        let vals = parse_fixed_or_ws_floats(line, 5, 15)?;
        for (col, &v) in vals.iter().enumerate().take(5) {
            if !v.is_finite() {
                return Err(EarthSnailError::Header(format!(
                    "non-finite float in SAC alpha header row {}, col {}",
                    row + 1,
                    col + 1
                )));
            }
            let idx = row * 5 + col;
            floats[idx] = if is_sac_float_undef(v) {
                SAC_FLOAT_UNDEF_F32
            } else {
                v as f32
            };
        }
    }

    // 2. 8 lines of 5 integers (40 integers total)
    let mut ints = [0_i32; SAC_INT_COUNT];
    for row in 0..8 {
        let line = lines.next().ok_or_else(|| {
            EarthSnailError::Format(format!(
                "truncated SAC alpha header at integer row {}",
                row + 1
            ))
        })?;
        let vals = parse_fixed_or_ws_ints(line, 5, 10)?;
        for col in 0..5 {
            ints[row * 5 + col] = vals[col];
        }
    }

    // 3. 8 lines of 24-character string blocks (192 bytes total)
    let mut str_bytes = [b' '; 192];
    for row in 0..8 {
        let line = lines.next().ok_or_else(|| {
            EarthSnailError::Format(format!(
                "truncated SAC alpha header at string row {}",
                row + 1
            ))
        })?;
        let line_clean = line.strip_suffix('\r').unwrap_or(line);
        let raw = line_clean.as_bytes();
        let copy_len = raw.len().min(24);
        let dst_off = row * 24;
        str_bytes[dst_off..dst_off + copy_len].copy_from_slice(&raw[..copy_len]);
    }

    let mut hdr = SacHeader::empty_raw();
    hdr.from_float_words(&floats);
    hdr.from_int_words(&ints);
    hdr.from_string_bytes(&str_bytes);
    // Preserve full f64 accuracy from alpha floats where defined:
    let mut lines_re = input.lines();
    for row in 0..14 {
        if let Some(line) = lines_re.next() {
            let vals = parse_fixed_or_ws_floats(line, 5, 15)?;
            for (col, &v) in vals.iter().enumerate().take(5) {
                *hdr.float_by_index_mut(row * 5 + col) = if is_sac_float_undef(v) {
                    SAC_FLOAT_UNDEF
                } else {
                    v
                };
            }
        }
    }

    if is_sac_int_undef(hdr.npts) || hdr.npts < 0 {
        return Err(EarthSnailError::Header(format!(
            "invalid or negative SAC alpha NPTS: {}",
            hdr.npts
        )));
    }
    let npts = hdr.npts as usize;
    if npts > MAX_SAC_SAMPLES {
        return Err(EarthSnailError::CapacityExceeded(format!(
            "SAC alpha NPTS {npts} exceeds safety limit {MAX_SAC_SAMPLES}"
        )));
    }
    // Bounded allocation check: each sample in ASCII requires >= 1 byte, so `input.len()`
    // strictly bounds maximum realizable samples!
    if npts > input.len() {
        return Err(EarthSnailError::Format(format!(
            "truncated SAC alpha data: NPTS={npts} exceeds input byte length {}",
            input.len()
        )));
    }

    let mut data = Vec::with_capacity(npts);
    for line in lines {
        if data.len() >= npts {
            break;
        }
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() {
            continue;
        }
        let needed = (npts - data.len()).min(5);
        let parsed = parse_alpha_data_line(line, needed)?;
        for v in parsed {
            if data.len() < npts {
                let vf = v as f32;
                if !vf.is_finite() {
                    return Err(EarthSnailError::Format(
                        "non-finite sample in SAC alpha data".into(),
                    ));
                }
                data.push(vf as f64);
            }
        }
    }

    if data.len() != npts {
        return Err(EarthSnailError::Format(format!(
            "truncated SAC alpha data: expected {npts} samples, decoded {}",
            data.len()
        )));
    }

    finalize_sac_header_on_read(&mut hdr);
    let stats = build_trace_stats_from_sac(hdr)?;
    Ok(Trace::new(stats, data))
}

/// Parses a SAC Alphanumeric (`SACXY` / `alpha`) byte slice into a `Trace`.
pub fn read_sac_alpha_bytes(bytes: &[u8]) -> Result<Trace> {
    let text = core::str::from_utf8(bytes)
        .map_err(|_| EarthSnailError::Format("SAC alpha input is not valid UTF-8/ASCII".into()))?;
    read_sac_alpha_str(text)
}

/// Formats a `Trace` as SAC Alphanumeric (`SACXY` / `alpha`) text.
pub fn write_sac_alpha_str(trace: &Trace) -> Result<String> {
    if trace.data.len() > (i32::MAX as usize) {
        return Err(EarthSnailError::CapacityExceeded(
            "trace length overflows SAC i32 NPTS".into(),
        ));
    }
    for (idx, &v) in trace.data.iter().enumerate() {
        if !v.is_finite() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "non-finite sample at index {idx} cannot be written to SAC alpha"
            )));
        }
    }

    let hdr = prepare_sac_header_for_write(trace)?;
    let data_lines = trace.data.len().div_ceil(5);
    let mut out = String::with_capacity(30 * 78 + data_lines * 76);

    // 14 lines of 5 floats (`%15.7g`)
    for row in 0..14 {
        for col in 0..5 {
            let val = hdr.float_by_index(row * 5 + col);
            push_sac_g15_7(&mut out, val, true);
        }
        out.push('\n');
    }

    // 8 lines of 5 integers (`%10d`)
    for row in 0..8 {
        for col in 0..5 {
            let val = hdr.int_by_index(row * 5 + col);
            let _ = core::fmt::Write::write_fmt(&mut out, format_args!("{val:10}"));
        }
        out.push('\n');
    }

    // 8 lines of 3x8-char strings (24 bytes per line)
    let str_bytes = hdr.to_string_bytes();
    for row in 0..8 {
        let slice = &str_bytes[row * 24..(row + 1) * 24];
        for &b in slice {
            out.push(if b.is_ascii() { b as char } else { ' ' });
        }
        out.push('\n');
    }

    // Data section: 5 floats per line (`%15.7g`)
    for chunk in trace.data.chunks(5) {
        for &sample in chunk {
            push_sac_g15_7(&mut out, sample, false);
        }
        out.push('\n');
    }

    Ok(out)
}

/// Encodes a `Trace` as SAC Alphanumeric (`SACXY` / `alpha`) UTF-8 bytes.
pub fn write_sac_alpha_bytes(trace: &Trace) -> Result<Vec<u8>> {
    Ok(write_sac_alpha_str(trace)?.into_bytes())
}

/// Reads a SAC Binary file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_sac_file(path: impl AsRef<std::path::Path>) -> Result<Trace> {
    let bytes = std::fs::read(path)?;
    read_sac_bytes(&bytes)
}

/// Writes a `Trace` as SAC Binary to disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_sac_file(
    path: impl AsRef<std::path::Path>,
    trace: &Trace,
    endian: SacEndian,
) -> Result<()> {
    let bytes = write_sac_bytes(trace, endian)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Reads a SAC Alphanumeric (`SACXY`) file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_sac_alpha_file(path: impl AsRef<std::path::Path>) -> Result<Trace> {
    let bytes = std::fs::read(path)?;
    read_sac_alpha_bytes(&bytes)
}

/// Writes a `Trace` as SAC Alphanumeric (`SACXY`) to disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_sac_alpha_file(path: impl AsRef<std::path::Path>, trace: &Trace) -> Result<()> {
    let bytes = write_sac_alpha_bytes(trace)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn finalize_sac_header_on_read(hdr: &mut SacHeader) {
    // If `LCALDA == 1` and coordinates are present while `dist`/`az`/`baz`/`gcarc` were left
    // `-12345.0`, compute them automatically using WGS84 geodesics.
    if hdr.lcalda == 1
        && (is_sac_float_undef(hdr.dist)
            || is_sac_float_undef(hdr.az)
            || is_sac_float_undef(hdr.baz)
            || is_sac_float_undef(hdr.gcarc))
    {
        hdr.update_dist_az_baz_gcarc();
    }
}

fn build_trace_stats_from_sac(hdr: SacHeader) -> Result<TraceStats> {
    if is_sac_float_undef(hdr.delta) || hdr.delta <= 0.0 || !hdr.delta.is_finite() {
        return Err(EarthSnailError::Header(format!(
            "invalid or non-positive SAC DELTA: {}",
            hdr.delta
        )));
    }
    let sampling_rate = 1.0 / hdr.delta;
    let starttime = hdr.starttime().unwrap_or(UtcTimestamp::UNIX_EPOCH);
    let channel = ChannelId::new(
        hdr.get_str("knetwk").unwrap_or(""),
        hdr.get_str("kstnm").unwrap_or(""),
        hdr.get_str("khole").unwrap_or(""),
        hdr.get_str("kcmpnm").unwrap_or(""),
    );
    let calib = hdr.get_float("scale").unwrap_or(1.0);
    Ok(TraceStats {
        channel,
        starttime,
        sampling_rate,
        calib,
        sac: Some(hdr),
    })
}

fn prepare_sac_header_for_write(trace: &Trace) -> Result<SacHeader> {
    if !trace.stats.sampling_rate.is_finite() || trace.stats.sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(format!(
            "invalid Trace sampling_rate: {}",
            trace.stats.sampling_rate
        )));
    }
    let delta_s = trace.stats.delta();
    let npts = trace.data.len();

    let mut hdr = trace.stats.sac.clone().unwrap_or_default();
    if is_sac_int_undef(hdr.nvhdr) || !(1..=7).contains(&hdr.nvhdr) {
        hdr.nvhdr = 6;
    }
    if is_sac_int_undef(hdr.iftype) {
        hdr.iftype = SacFileType::TimeSeries as i32;
    }
    if is_sac_int_undef(hdr.leven) {
        hdr.leven = 1;
    }
    if is_sac_int_undef(hdr.lovrok) {
        hdr.lovrok = 1;
    }

    // Sync channel strings when populated on `trace.stats.channel`.
    if !trace.stats.channel.network.is_empty() || hdr.get_str("knetwk").is_some() {
        let _ = hdr.set_str("knetwk", &trace.stats.channel.network);
    }
    if !trace.stats.channel.station.is_empty() || hdr.get_str("kstnm").is_some() {
        let _ = hdr.set_str("kstnm", &trace.stats.channel.station);
    }
    if !trace.stats.channel.location.is_empty() || hdr.get_str("khole").is_some() {
        let _ = hdr.set_str("khole", &trace.stats.channel.location);
    }
    if !trace.stats.channel.channel.is_empty() || hdr.get_str("kcmpnm").is_some() {
        let _ = hdr.set_str("kcmpnm", &trace.stats.channel.channel);
    }

    if (trace.stats.calib - 1.0).abs() > 1e-12 && hdr.get_float("scale").is_none() {
        hdr.scale = trace.stats.calib;
    }

    // Preserve existing reference time + non-zero `B` if `hdr.starttime()` still matches `trace.stats.starttime`
    // within 0.5 ms (so `IO`-referenced SAC files with `B != 0` round-trip cleanly!).
    let preserve_reftime = hdr
        .starttime()
        .is_some_and(|st| st.duration_nanos_since(trace.stats.starttime).abs() <= 500_000)
        && !is_sac_float_undef(hdr.delta)
        && ((hdr.delta - delta_s).abs() <= 1e-6 * delta_s);

    if preserve_reftime {
        hdr.npts = npts as i32;
        let b_val = if is_sac_float_undef(hdr.b) {
            0.0
        } else {
            hdr.b
        };
        hdr.b = b_val;
        hdr.e = if npts > 0 {
            b_val + ((npts - 1) as f64) * hdr.delta
        } else {
            b_val
        };
    } else {
        hdr.set_starttime_and_span(trace.stats.starttime, npts, delta_s);
    }

    hdr.update_dep_stats(&trace.data);

    if hdr.lcalda == 1
        && (is_sac_float_undef(hdr.dist)
            || is_sac_float_undef(hdr.az)
            || is_sac_float_undef(hdr.baz)
            || is_sac_float_undef(hdr.gcarc))
    {
        hdr.update_dist_az_baz_gcarc();
    }

    Ok(hdr)
}

fn parse_fixed_or_ws_floats(line: &str, count: usize, width: usize) -> Result<Vec<f64>> {
    let clean = line.strip_suffix('\r').unwrap_or(line);
    // Try whitespace split first when exact token count matches.
    let ws_tokens: Vec<&str> = clean.split_whitespace().collect();
    if ws_tokens.len() == count {
        let mut out = Vec::with_capacity(count);
        for tok in ws_tokens {
            let v: f64 = tok.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SAC alpha float token '{tok}'"))
            })?;
            out.push(v);
        }
        return Ok(out);
    }

    // Fallback to fixed-column Fortran field slicing (`width` bytes per column).
    if clean.is_ascii() && clean.len() >= count * width {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let field = clean[i * width..(i + 1) * width].trim();
            let v: f64 = field.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SAC alpha fixed float field '{field}'"))
            })?;
            out.push(v);
        }
        return Ok(out);
    }

    Err(EarthSnailError::Format(format!(
        "expected {count} floats in SAC alpha line, got '{}'",
        clean
    )))
}

fn parse_fixed_or_ws_ints(line: &str, count: usize, width: usize) -> Result<Vec<i32>> {
    let clean = line.strip_suffix('\r').unwrap_or(line);
    let ws_tokens: Vec<&str> = clean.split_whitespace().collect();
    if ws_tokens.len() == count {
        let mut out = Vec::with_capacity(count);
        for tok in ws_tokens {
            let v: i32 = tok.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SAC alpha integer token '{tok}'"))
            })?;
            out.push(v);
        }
        return Ok(out);
    }

    if clean.is_ascii() && clean.len() >= count * width {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let field = clean[i * width..(i + 1) * width].trim();
            let v: i32 = field.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SAC alpha fixed int field '{field}'"))
            })?;
            out.push(v);
        }
        return Ok(out);
    }

    Err(EarthSnailError::Format(format!(
        "expected {count} integers in SAC alpha line, got '{}'",
        clean
    )))
}

fn parse_alpha_data_line(line: &str, max_expected: usize) -> Result<Vec<f64>> {
    let ws_tokens: Vec<&str> = line.split_whitespace().collect();
    let mut parsed_ws = Vec::with_capacity(ws_tokens.len());
    let mut all_ok = !ws_tokens.is_empty();
    for tok in &ws_tokens {
        match tok.parse::<f64>() {
            Ok(v) => parsed_ws.push(v),
            Err(_) => {
                all_ok = false;
                break;
            }
        }
    }
    if all_ok {
        return Ok(parsed_ws);
    }

    // Fixed 15-column fallback for touching Fortran `%15.7g` columns.
    if line.is_ascii() && line.len() >= 15 {
        let cols = (line.len() / 15).min(max_expected.max(1));
        let mut out = Vec::with_capacity(cols);
        for i in 0..cols {
            let field = line[i * 15..(i + 1) * 15].trim();
            if field.is_empty() {
                continue;
            }
            let v: f64 = field.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SAC alpha sample field '{field}'"))
            })?;
            out.push(v);
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }

    Err(EarthSnailError::Format(format!(
        "invalid SAC alpha data line: '{line}'"
    )))
}

/// Formats a floating-point value into a 15-character SAC `%15.7g` column.
fn push_sac_g15_7(out: &mut String, val: f64, check_undef: bool) {
    use core::fmt::Write as _;
    if check_undef && is_sac_float_undef(val) {
        out.push_str("      -12345.00");
        return;
    }
    let v = (val as f32) as f64;
    if v == 0.0 {
        out.push_str("       0.000000");
        return;
    }
    let abs_v = v.abs();
    let s = if !(1e-4..1e7).contains(&abs_v) {
        format!("{v:15.6e}")
    } else {
        let exp10 = math::floor(math::log10(abs_v)) as i32;
        let dec = (6 - exp10).clamp(0, 10) as usize;
        format!("{v:15.* }", dec)
    };
    if s.len() == 15 && s.starts_with(' ') {
        out.push_str(&s);
    } else {
        let fallback = format!("{v:.6e}");
        let pad = 15_usize.saturating_sub(fallback.len()).max(1);
        for _ in 0..pad {
            out.push(' ');
        }
        let _ = write!(out, "{fallback}");
    }
}
