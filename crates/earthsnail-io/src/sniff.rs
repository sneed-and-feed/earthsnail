//! Automatic waveform format detector (`detect_waveform_format`) and universal byte/file reader
//! (`read_waveforms_bytes`, `read_waveforms_file`) covering MS3, miniSEED v2, SAC Binary LE/BE,
//! SAC Alphanumeric (`SACXY`), EarthScope GeoCSV, ObsPy `TSPAIR`, and ObsPy `SLIST`.

use crate::ascii::{read_geocsv_str, read_obspy_ascii_str};
use crate::mseed::{looks_like_mseed2, read_mseed_bytes};
use crate::mseed3::{looks_like_mseed3, read_mseed3_bytes};
use crate::sac::{read_sac_alpha_str, read_sac_bytes};
use earthsnail_core::{EarthSnailError, Result, SacEndian, SacHeader, Stream, SAC_HEADER_BYTES};

/// Recognized waveform wire/file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformFormat {
    /// FDSN miniSEED version 3 (`"MS"` + `0x03`).
    MiniSeed3,
    /// SEED / miniSEED version 2 (`2.4`).
    MiniSeed2,
    /// Little-endian SAC Binary (`v6` / `v7`).
    SacBinaryLittle,
    /// Big-endian SAC Binary (`v6` / `v7`).
    SacBinaryBig,
    /// Alphanumeric SAC (`SACXY` / `alpha`).
    SacAlpha,
    /// EarthScope GeoCSV (`1.0` / `2.0`).
    GeoCsv,
    /// ObsPy `TSPAIR` (`TIMESERIES ..., TSPAIR, ...`).
    TsPair,
    /// ObsPy `SLIST` (`TIMESERIES ..., SLIST, ...`).
    SList,
}

/// Inspects `bytes` without unbounded allocation and identifies its `WaveformFormat`.
pub fn detect_waveform_format(bytes: &[u8]) -> Option<WaveformFormat> {
    if bytes.is_empty() {
        return None;
    }

    // 1. FDSN miniSEED v3 (`[b'M', b'S', 3]`)
    if looks_like_mseed3(bytes) {
        return Some(WaveformFormat::MiniSeed3);
    }

    // 2. SEED / miniSEED v2 (`48B` FSDH + valid BTime)
    if looks_like_mseed2(bytes) {
        return Some(WaveformFormat::MiniSeed2);
    }

    // 3. SAC Binary (LE or BE): check NVHDR in `1..=7` AND verify `NPTS >= 0` fits buffer!
    if bytes.len() >= SAC_HEADER_BYTES {
        if let Some(endian) = SacHeader::detect_endian(bytes) {
            let npts_raw = [bytes[316], bytes[317], bytes[318], bytes[319]];
            let npts = match endian {
                SacEndian::Little => i32::from_le_bytes(npts_raw),
                SacEndian::Big => i32::from_be_bytes(npts_raw),
            };
            if npts >= 0 {
                let expected = (npts as usize)
                    .checked_mul(4)
                    .and_then(|d| d.checked_add(SAC_HEADER_BYTES));
                if let Some(min_len) = expected {
                    if bytes.len() >= min_len {
                        return Some(match endian {
                            SacEndian::Little => WaveformFormat::SacBinaryLittle,
                            SacEndian::Big => WaveformFormat::SacBinaryBig,
                        });
                    }
                }
            }
        }
    }

    // 4. UTF-8 / ASCII text waveform formats
    let text = core::str::from_utf8(bytes).ok()?;
    let trimmed = text.strip_prefix('\u{feff}').unwrap_or(text).trim_start();
    if trimmed.starts_with('#') {
        let head = &trimmed[..trimmed.len().min(2048)];
        if head.contains("GeoCSV")
            || head.contains("sample_rate")
            || head.contains("start_time")
            || head.contains("SID:")
        {
            return Some(WaveformFormat::GeoCsv);
        }
    }

    if let Some(rest) = trimmed.strip_prefix("TIMESERIES") {
        let first_line = rest.lines().next().unwrap_or("");
        if first_line.contains("TSPAIR") || first_line.contains("tspair") {
            return Some(WaveformFormat::TsPair);
        }
        return Some(WaveformFormat::SList);
    }

    if looks_like_sac_alpha(trimmed) {
        return Some(WaveformFormat::SacAlpha);
    }

    None
}

fn looks_like_sac_alpha(text: &str) -> bool {
    let mut lines = text.lines();
    // Check first line has 5 parseable floats and line 16 (2nd int line) has NVHDR in 1..=7.
    let Some(first) = lines.next() else {
        return false;
    };
    if first.split_whitespace().take(6).count() != 5 {
        return false;
    }
    if !first
        .split_whitespace()
        .all(|tok| tok.parse::<f64>().is_ok())
    {
        return false;
    }
    // Advance through float lines 2..=14 (13 lines) and int line 15 (1 line) to reach line 16:
    for _ in 0..14 {
        if lines.next().is_none() {
            return false;
        }
    }
    let Some(int_line_2) = lines.next() else {
        return false;
    };
    let ints: alloc::vec::Vec<i32> = int_line_2
        .split_whitespace()
        .filter_map(|t| t.parse::<i32>().ok())
        .collect();
    // Word 6 of SAC ints (`row 1, col 1`) is `NVHDR` (`1..=7`).
    ints.len() == 5 && (1..=7).contains(&ints[1])
}

/// Auto-detects the waveform format of `bytes` and decodes it into a `Stream`.
pub fn read_waveforms_bytes(bytes: &[u8]) -> Result<Stream> {
    let fmt = detect_waveform_format(bytes).ok_or_else(|| {
        EarthSnailError::Format(
            "unrecognized or truncated waveform format (not MS3, miniSEED v2, SAC, GeoCSV, SLIST, or TSPAIR)"
                .into(),
        )
    })?;

    match fmt {
        WaveformFormat::MiniSeed3 => read_mseed3_bytes(bytes),
        WaveformFormat::MiniSeed2 => read_mseed_bytes(bytes),
        WaveformFormat::SacBinaryLittle | WaveformFormat::SacBinaryBig => {
            let tr = read_sac_bytes(bytes)?;
            Ok(Stream::from_traces(alloc::vec![tr]))
        }
        WaveformFormat::SacAlpha => {
            let s = core::str::from_utf8(bytes)
                .map_err(|_| EarthSnailError::Format("invalid UTF-8 in SAC alpha".into()))?;
            let tr = read_sac_alpha_str(s)?;
            Ok(Stream::from_traces(alloc::vec![tr]))
        }
        WaveformFormat::GeoCsv => {
            let s = core::str::from_utf8(bytes)
                .map_err(|_| EarthSnailError::Format("invalid UTF-8 in GeoCSV".into()))?;
            read_geocsv_str(s)
        }
        WaveformFormat::TsPair | WaveformFormat::SList => {
            let s = core::str::from_utf8(bytes)
                .map_err(|_| EarthSnailError::Format("invalid UTF-8 in ASCII waveform".into()))?;
            read_obspy_ascii_str(s)
        }
    }
}

/// Auto-detects and reads any supported waveform file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_waveforms_file(path: impl AsRef<std::path::Path>) -> Result<Stream> {
    let bytes = std::fs::read(path)?;
    read_waveforms_bytes(&bytes)
}
