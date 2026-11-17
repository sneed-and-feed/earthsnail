//! ASCII waveform formats:
//! - EarthScope **GeoCSV 2.0** (`# dataset: GeoCSV 2.0`, `# SID`, `# start_time`,
//!   `# sample_rate_hz`, `# field_unit`, supporting both 1-column `Sample` and 2-column
//!   `Time,Sample` layouts plus multi-segment concatenated responses).
//! - ObsPy **SLIST** (`TIMESERIES NET_STA_LOC_CHA, N samples, fs sps, ISO, SLIST, ...`).
//! - ObsPy **TSPAIR** (`TIMESERIES NET_STA_LOC_CHA, N samples, fs sps, ISO, TSPAIR, ...`).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use earthsnail_core::{
    ChannelId, EarthSnailError, Result, Stream, Trace, TraceStats, UtcTimestamp,
};

/// Maximum samples accepted across an ASCII waveform parse (`64M` samples).
pub const MAX_ASCII_SAMPLES: usize = 64 * 1024 * 1024;

/// Parses an EarthScope **GeoCSV** (`1.0` / `2.0`) waveform string into a `Stream`.
pub fn read_geocsv_str(input: &str) -> Result<Stream> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut traces = Vec::new();
    let mut total_samples = 0_usize;

    let mut cur_delim = ',';
    let mut cur_channel = ChannelId::default();
    let mut cur_start: Option<UtcTimestamp> = None;
    let mut cur_rate: Option<f64> = None;
    let mut cur_calib = 1.0_f64;
    let mut in_data = false;
    let mut cur_samples: Vec<f64> = Vec::new();
    let mut cur_row_times: Vec<UtcTimestamp> = Vec::new();
    let mut saw_geocsv_header = false;

    let flush_segment = |traces: &mut Vec<Trace>,
                         channel: &ChannelId,
                         start_opt: Option<UtcTimestamp>,
                         rate_opt: Option<f64>,
                         calib: f64,
                         samples: &mut Vec<f64>,
                         row_times: &mut Vec<UtcTimestamp>|
     -> Result<()> {
        if samples.is_empty() {
            row_times.clear();
            return Ok(());
        }
        let starttime = match start_opt {
            Some(st) => st,
            None => row_times
                .first()
                .copied()
                .unwrap_or(UtcTimestamp::UNIX_EPOCH),
        };
        let sampling_rate = match rate_opt {
            Some(r) if r > 0.0 && r.is_finite() => r,
            _ if row_times.len() >= 2 => {
                let dt = row_times[1].duration_secs_f64_since(row_times[0]);
                if dt <= 0.0 || !dt.is_finite() {
                    return Err(EarthSnailError::Header(
                        "cannot infer positive sampling rate from GeoCSV timestamps".into(),
                    ));
                }
                1.0 / dt
            }
            _ => {
                return Err(EarthSnailError::Header(
                    "GeoCSV missing `# sample_rate_hz` header".into(),
                ));
            }
        };

        let mut stats = TraceStats::new(channel.clone(), starttime, sampling_rate);
        stats.calib = calib;
        traces.push(Trace::new(stats, core::mem::take(samples)));
        row_times.clear();
        Ok(())
    };

    for raw_line in input.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            if in_data && !cur_samples.is_empty() {
                flush_segment(
                    &mut traces,
                    &cur_channel,
                    cur_start,
                    cur_rate,
                    cur_calib,
                    &mut cur_samples,
                    &mut cur_row_times,
                )?;
                cur_start = None;
                in_data = false;
            }
            saw_geocsv_header = true;
            if let Some((k_raw, v_raw)) = comment.split_once(':') {
                let key = k_raw.trim();
                let val = v_raw.trim();
                if key.eq_ignore_ascii_case("delimiter") && !val.is_empty() {
                    if let Some(ch) = val.chars().next() {
                        cur_delim = ch;
                    }
                } else if key.eq_ignore_ascii_case("SID") {
                    cur_channel = parse_flexible_channel_id(val)?;
                } else if key.eq_ignore_ascii_case("start_time") && !val.is_empty() {
                    cur_start = Some(UtcTimestamp::parse_iso8601(val)?);
                } else if (key.eq_ignore_ascii_case("sample_rate_hz")
                    || key.eq_ignore_ascii_case("sample_rate"))
                    && !val.is_empty()
                {
                    let r: f64 = val.parse().map_err(|_| {
                        EarthSnailError::Header(format!("invalid GeoCSV sample_rate_hz '{val}'"))
                    })?;
                    if !r.is_finite() || r <= 0.0 {
                        return Err(EarthSnailError::Header(format!(
                            "non-positive GeoCSV sample_rate_hz {r}"
                        )));
                    }
                    cur_rate = Some(r);
                } else if key.eq_ignore_ascii_case("scale_factor") && !val.is_empty() {
                    if let Ok(sf) = val.parse::<f64>() {
                        if sf.is_finite() && sf != 0.0 {
                            cur_calib = sf;
                        }
                    }
                }
            }
            continue;
        }

        // Check for optional column header row (`Time,Sample` or `Sample`).
        if !in_data {
            let first_ch = line.chars().next().unwrap_or(' ');
            if first_ch.is_ascii_alphabetic() {
                in_data = true;
                continue;
            }
            in_data = true;
        }

        if let Some((col1, col2_rest)) = line.split_once(cur_delim) {
            // 2-column `Time, Sample` (ignore extra trailing columns if any, taking 2nd field)
            let sample_str = match col2_rest.split_once(cur_delim) {
                Some((c2, _)) => c2.trim(),
                None => col2_rest.trim(),
            };
            let ts = UtcTimestamp::parse_iso8601(col1.trim())?;
            let val: f64 = sample_str.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid GeoCSV sample value '{sample_str}'"))
            })?;
            if !val.is_finite() {
                return Err(EarthSnailError::Format(
                    "non-finite sample value in GeoCSV".into(),
                ));
            }
            // Check for in-table time tear if `cur_rate` is known and we already have samples!
            if let (Some(fs), false) = (cur_rate, cur_samples.is_empty()) {
                let seg_start = cur_start.unwrap_or(cur_row_times[0]);
                if let Ok(expected) = seg_start.sample_timestamp(cur_samples.len() as i64, fs) {
                    if ts.duration_secs_f64_since(expected).abs() > 0.5 / fs {
                        flush_segment(
                            &mut traces,
                            &cur_channel,
                            Some(seg_start),
                            Some(fs),
                            cur_calib,
                            &mut cur_samples,
                            &mut cur_row_times,
                        )?;
                        cur_start = Some(ts);
                    }
                }
            }
            if cur_row_times.len() < 2 {
                cur_row_times.push(ts);
            }
            if cur_start.is_none() {
                cur_start = Some(ts);
            }
            cur_samples.push(val);
        } else {
            // 1-column `Sample`
            let val: f64 = line.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid 1-column GeoCSV sample '{line}'"))
            })?;
            if !val.is_finite() {
                return Err(EarthSnailError::Format(
                    "non-finite sample value in GeoCSV".into(),
                ));
            }
            cur_samples.push(val);
        }

        total_samples += 1;
        if total_samples > MAX_ASCII_SAMPLES {
            return Err(EarthSnailError::CapacityExceeded(
                "GeoCSV sample count exceeds safety limit".into(),
            ));
        }
    }

    flush_segment(
        &mut traces,
        &cur_channel,
        cur_start,
        cur_rate,
        cur_calib,
        &mut cur_samples,
        &mut cur_row_times,
    )?;

    if !saw_geocsv_header || traces.is_empty() {
        return Err(EarthSnailError::Format(
            "no valid GeoCSV waveform segments found".into(),
        ));
    }
    Ok(Stream::from_traces(traces))
}

/// Parses GeoCSV UTF-8 bytes into a `Stream`.
pub fn read_geocsv_bytes(bytes: &[u8]) -> Result<Stream> {
    let s = core::str::from_utf8(bytes)
        .map_err(|_| EarthSnailError::Format("GeoCSV input is not valid UTF-8".into()))?;
    read_geocsv_str(s)
}

/// Formats a `Stream` as EarthScope **GeoCSV 2.0** (`two_column == false` writes compact 1-column
/// `Sample`; `two_column == true` writes 2-column `Time,Sample`).
pub fn write_geocsv_stream(stream: &Stream, two_column: bool) -> Result<String> {
    let mut out = String::new();
    for (idx, trace) in stream.traces.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        write_geocsv_trace_into(&mut out, trace, two_column)?;
    }
    Ok(out)
}

/// Formats a single `Trace` as EarthScope **GeoCSV 2.0**.
pub fn write_geocsv_trace(trace: &Trace, two_column: bool) -> Result<String> {
    let mut out = String::new();
    write_geocsv_trace_into(&mut out, trace, two_column)?;
    Ok(out)
}

fn write_geocsv_trace_into(out: &mut String, trace: &Trace, two_column: bool) -> Result<()> {
    if !trace.stats.sampling_rate.is_finite() || trace.stats.sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(
            "invalid sampling_rate for GeoCSV export".into(),
        ));
    }
    for &v in &trace.data {
        if !v.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite sample cannot be written to GeoCSV".into(),
            ));
        }
    }

    let _ = writeln!(out, "# dataset: GeoCSV 2.0");
    let _ = writeln!(out, "# delimiter: ,");
    let _ = writeln!(out, "# SID: {}", trace.stats.channel.fdsn_source_id());
    let _ = writeln!(out, "# sample_count: {}", trace.data.len());
    let _ = writeln!(out, "# sample_rate_hz: {}", trace.stats.sampling_rate);
    let _ = writeln!(out, "# start_time: {}", trace.stats.starttime.to_iso8601());
    if (trace.stats.calib - 1.0).abs() > 1e-12 {
        let _ = writeln!(out, "# scale_factor: {}", trace.stats.calib);
    }

    if two_column {
        let _ = writeln!(out, "# field_unit: UTC, COUNTS");
        let _ = writeln!(out, "# field_type: datetime, float");
        let _ = writeln!(out, "Time,Sample");
        for (i, &v) in trace.data.iter().enumerate() {
            let ts = trace.sample_time(i)?;
            let _ = writeln!(out, "{},{}", ts.to_iso8601(), v);
        }
    } else {
        let _ = writeln!(out, "# field_unit: COUNTS");
        let _ = writeln!(out, "# field_type: float");
        let _ = writeln!(out, "Sample");
        for &v in &trace.data {
            let _ = writeln!(out, "{v}");
        }
    }
    Ok(())
}

/// Parses ObsPy **SLIST** or **TSPAIR** (`TIMESERIES ...`) ASCII waveforms into a `Stream`.
pub fn read_obspy_ascii_str(input: &str) -> Result<Stream> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut lines = input.lines().peekable();
    let mut traces = Vec::new();
    let mut total_samples = 0_usize;

    while let Some(raw_line) = lines.next() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if !line.starts_with("TIMESERIES") {
            return Err(EarthSnailError::Format(format!(
                "expected ObsPy 'TIMESERIES' header line, got '{line}'"
            )));
        }

        let hdr = parse_timeseries_header(line)?;
        if hdr.npts > input.len() || total_samples.saturating_add(hdr.npts) > MAX_ASCII_SAMPLES {
            return Err(EarthSnailError::CapacityExceeded(format!(
                "TIMESERIES header claims excessive sample count {}",
                hdr.npts
            )));
        }

        let mut samples = Vec::with_capacity(hdr.npts);
        if hdr.is_tspair {
            while samples.len() < hdr.npts {
                let data_line = lines.next().ok_or_else(|| {
                    EarthSnailError::Format("truncated TSPAIR data section".into())
                })?;
                let dl = data_line.trim();
                if dl.is_empty() {
                    continue;
                }
                let mut parts = dl.split_whitespace();
                let ts_tok = parts.next().ok_or_else(|| {
                    EarthSnailError::Format("missing timestamp in TSPAIR row".into())
                })?;
                let val_tok = parts.next().ok_or_else(|| {
                    EarthSnailError::Format(format!("missing sample value in TSPAIR row '{dl}'"))
                })?;
                let _ = UtcTimestamp::parse_iso8601(ts_tok)?;
                let val: f64 = val_tok.parse().map_err(|_| {
                    EarthSnailError::Format(format!("invalid TSPAIR sample '{val_tok}'"))
                })?;
                if !val.is_finite() {
                    return Err(EarthSnailError::Format("non-finite TSPAIR sample".into()));
                }
                samples.push(val);
            }
        } else {
            while samples.len() < hdr.npts {
                let data_line = lines.next().ok_or_else(|| {
                    EarthSnailError::Format("truncated SLIST data section".into())
                })?;
                let dl = data_line.trim();
                if dl.is_empty() {
                    continue;
                }
                for tok in dl.split_whitespace() {
                    let val: f64 = tok.parse().map_err(|_| {
                        EarthSnailError::Format(format!("invalid SLIST sample '{tok}'"))
                    })?;
                    if !val.is_finite() {
                        return Err(EarthSnailError::Format("non-finite SLIST sample".into()));
                    }
                    samples.push(val);
                    if samples.len() == hdr.npts {
                        break;
                    }
                }
            }
        }

        total_samples += samples.len();
        traces.push(Trace::new(
            TraceStats::new(hdr.channel, hdr.starttime, hdr.sampling_rate),
            samples,
        ));
    }

    if traces.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "no TIMESERIES blocks found in ASCII input".into(),
        ));
    }
    Ok(Stream::from_traces(traces))
}

/// Parses ObsPy **SLIST** text into a `Stream`.
pub fn read_slist_str(input: &str) -> Result<Stream> {
    read_obspy_ascii_str(input)
}

/// Parses ObsPy **TSPAIR** text into a `Stream`.
pub fn read_tspair_str(input: &str) -> Result<Stream> {
    read_obspy_ascii_str(input)
}

/// Formats a `Stream` in ObsPy **SLIST** format (`6` samples per line).
pub fn write_slist_stream(stream: &Stream) -> Result<String> {
    let mut out = String::new();
    for tr in &stream.traces {
        write_timeseries_block(&mut out, tr, false)?;
    }
    Ok(out)
}

/// Formats a `Stream` in ObsPy **TSPAIR** format (`<ISO8601>  <sample>` per line).
pub fn write_tspair_stream(stream: &Stream) -> Result<String> {
    let mut out = String::new();
    for tr in &stream.traces {
        write_timeseries_block(&mut out, tr, true)?;
    }
    Ok(out)
}

struct TimeseriesHeader {
    channel: ChannelId,
    npts: usize,
    sampling_rate: f64,
    starttime: UtcTimestamp,
    is_tspair: bool,
}

fn parse_timeseries_header(line: &str) -> Result<TimeseriesHeader> {
    let rest = line
        .strip_prefix("TIMESERIES")
        .ok_or_else(|| EarthSnailError::Format("missing TIMESERIES prefix".into()))?
        .trim();
    let cols: Vec<&str> = rest.split(',').map(str::trim).collect();
    if cols.len() < 5 {
        return Err(EarthSnailError::Format(format!(
            "malformed TIMESERIES header '{line}'"
        )));
    }
    let channel = parse_flexible_channel_id(cols[0])?;
    let npts_str = cols[1]
        .strip_suffix("samples")
        .or_else(|| cols[1].strip_suffix("sample"))
        .unwrap_or(cols[1])
        .trim();
    let npts: usize = npts_str.parse().map_err(|_| {
        EarthSnailError::Header(format!("invalid TIMESERIES sample count '{}'", cols[1]))
    })?;

    let sps_str = cols[2].strip_suffix("sps").unwrap_or(cols[2]).trim();
    let sampling_rate: f64 = sps_str.parse().map_err(|_| {
        EarthSnailError::Header(format!("invalid TIMESERIES sampling rate '{}'", cols[2]))
    })?;
    if !sampling_rate.is_finite() || sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(format!(
            "non-positive TIMESERIES sampling rate {sampling_rate}"
        )));
    }

    let starttime = UtcTimestamp::parse_iso8601(cols[3])?;
    let fmt_kind = cols[4];
    let is_tspair = if fmt_kind.eq_ignore_ascii_case("TSPAIR") {
        true
    } else if fmt_kind.eq_ignore_ascii_case("SLIST") {
        false
    } else {
        return Err(EarthSnailError::Format(format!(
            "unsupported TIMESERIES format '{fmt_kind}' (expected SLIST or TSPAIR)"
        )));
    };

    Ok(TimeseriesHeader {
        channel,
        npts,
        sampling_rate,
        starttime,
        is_tspair,
    })
}

fn write_timeseries_block(out: &mut String, tr: &Trace, tspair: bool) -> Result<()> {
    if !tr.stats.sampling_rate.is_finite() || tr.stats.sampling_rate <= 0.0 {
        return Err(EarthSnailError::Header(
            "invalid sampling_rate for TIMESERIES export".into(),
        ));
    }
    for &v in &tr.data {
        if !v.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite sample in TIMESERIES export".into(),
            ));
        }
    }
    let ch = &tr.stats.channel;
    let kind = if tspair { "TSPAIR" } else { "SLIST" };
    let iso_no_z = tr.stats.starttime.to_fdsn_query();
    let _ = writeln!(
        out,
        "TIMESERIES {}_{}_{}_{}, {} samples, {} sps, {}, {}, FLOAT, Counts",
        ch.network,
        ch.station,
        ch.location,
        ch.channel,
        tr.data.len(),
        tr.stats.sampling_rate,
        iso_no_z,
        kind
    );
    if tspair {
        for (i, &v) in tr.data.iter().enumerate() {
            let ts = tr.sample_time(i)?;
            let _ = writeln!(out, "{}  {v}", ts.to_fdsn_query());
        }
    } else {
        for chunk in tr.data.chunks(6) {
            for (j, &v) in chunk.iter().enumerate() {
                if j > 0 {
                    out.push('\t');
                }
                let _ = write!(out, "{v}");
            }
            out.push('\n');
        }
    }
    Ok(())
}

fn parse_flexible_channel_id(raw: &str) -> Result<ChannelId> {
    let s = raw.trim();
    if s.starts_with("FDSN:") || s.contains('.') {
        return ChannelId::parse_id(s);
    }
    let parts: Vec<&str> = s.split('_').collect();
    if parts.len() == 4 {
        return Ok(ChannelId::new(parts[0], parts[1], parts[2], parts[3]));
    }
    if parts.len() == 6 {
        let cha = format!("{}{}{}", parts[3], parts[4], parts[5]);
        return Ok(ChannelId::new(parts[0], parts[1], parts[2], cha));
    }
    Err(EarthSnailError::Format(format!(
        "unrecognized ASCII channel identifier '{raw}'"
    )))
}
