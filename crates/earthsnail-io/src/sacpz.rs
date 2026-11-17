//! SAC PoleZero (`SACPZ`) parser and writer (`ZEROS`, `POLES`, `CONSTANT`, implicit origin zeros,
//! and `*` header comment metadata blocks such as `NETWORK`, `STATION`, `LOCATION`, `CHANNEL`,
//! `INPUT UNIT`, `INSTGAIN`, `SENSITIVITY`, and `A0`).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use earthsnail_core::{
    ChannelId, ChannelMetadata, ChannelResponse, Complex64, EarthSnailError, KinematicUnit,
    PzTransferFunctionType, ResponseStage, Result, StageGain, UtcTimestamp,
};

/// Upper bound on poles or zeros accepted from a SACPZ file.
pub const MAX_SACPZ_ROOTS: usize = 4_096;

/// Complete parsed SAC PoleZero file including both the `ChannelResponse` and any `*` header
/// comment metadata exported by IRIS/EarthScope/ObsPy (`rdseed` / `obspy` `SACPZ` conventions).
#[derive(Debug, Clone, PartialEq)]
pub struct SacPzRecord {
    /// Optional 4-part channel identifier parsed from `* NETWORK` / `STATION` / `LOCATION` / `CHANNEL`.
    pub channel: Option<ChannelId>,
    /// Optional complete `ChannelMetadata` when coordinates are present in `*` comments.
    pub metadata: Option<ChannelMetadata>,
    /// Raw `* INPUT UNIT` comment value (`"M"` default in SACPZ).
    pub input_unit_raw: Option<String>,
    /// Raw `* OUTPUT UNIT` comment value (`"COUNTS"` default).
    pub output_unit_raw: Option<String>,
    /// Optional `* A0` normalization factor from comment block.
    pub a0: Option<f64>,
    /// Optional `* INSTGAIN` sensor gain from comment block.
    pub instgain: Option<f64>,
    /// Optional `* SENSITIVITY` overall gain from comment block.
    pub sensitivity: Option<f64>,
    /// Complex zeros (`rad/s`; implicit omitted SAC zeros filled with `0 + 0i`).
    pub zeros: Vec<Complex64>,
    /// Complex poles (`rad/s`).
    pub poles: Vec<Complex64>,
    /// Overall `CONSTANT` (`A0 * sensitivity` in displacement `rad/s` units).
    pub constant: f64,
    /// Ready-to-evaluate `ChannelResponse` (`ChannelResponse::from_sacpz` compatible).
    pub response: ChannelResponse,
}

/// Parses a SAC PoleZero (`SACPZ`) text string into a `SacPzRecord`.
pub fn read_sacpz_str(input: &str) -> Result<SacPzRecord> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);

    let mut net: Option<String> = None;
    let mut sta: Option<String> = None;
    let mut loc: Option<String> = None;
    let mut cha: Option<String> = None;
    let mut start_date: Option<UtcTimestamp> = None;
    let mut end_date: Option<UtcTimestamp> = None;
    let mut lat: Option<f64> = None;
    let mut lon: Option<f64> = None;
    let mut elev: Option<f64> = None;
    let mut depth: Option<f64> = None;
    let mut dip: Option<f64> = None;
    let mut azim: Option<f64> = None;
    let mut srate: Option<f64> = None;
    let mut input_unit_raw: Option<String> = None;
    let mut output_unit_raw: Option<String> = None;
    let mut a0_comment: Option<f64> = None;
    let mut instgain_comment: Option<f64> = None;
    let mut sensitivity_comment: Option<f64> = None;

    // Pre-scan `*` comment lines for metadata key-values (`* KEY : VALUE`).
    let mut body_lines: Vec<&str> = Vec::new();
    for raw_line in input.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(after_star) = trimmed.strip_prefix('*') {
            if let Some((k_raw, v_raw)) = after_star.split_once(':') {
                let key = k_raw.trim();
                let val = v_raw.trim();
                if val.is_empty() {
                    continue;
                }
                if key.eq_ignore_ascii_case("NETWORK") || key.eq_ignore_ascii_case("KNETWK") {
                    net = Some(String::from(val));
                } else if key.eq_ignore_ascii_case("STATION") || key.eq_ignore_ascii_case("KSTNM") {
                    sta = Some(String::from(val));
                } else if key.eq_ignore_ascii_case("LOCATION") || key.eq_ignore_ascii_case("KHOLE")
                {
                    loc = Some(String::from(val));
                } else if key.eq_ignore_ascii_case("CHANNEL") || key.eq_ignore_ascii_case("KCMPNM")
                {
                    cha = Some(String::from(val));
                } else if key.eq_ignore_ascii_case("START") {
                    start_date = UtcTimestamp::parse_iso8601(val).ok();
                } else if key.eq_ignore_ascii_case("END") {
                    end_date = UtcTimestamp::parse_iso8601(val).ok();
                } else if key.eq_ignore_ascii_case("LATITUDE") {
                    lat = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("LONGITUDE") {
                    lon = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("ELEVATION") {
                    elev = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("DEPTH") {
                    depth = parse_leading_f64(val);
                } else if key.starts_with("DIP") || key.eq_ignore_ascii_case("DIP") {
                    dip = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("AZIMUTH") {
                    azim = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("SAMPLE RATE") {
                    srate = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("INPUT UNIT") {
                    input_unit_raw = Some(String::from(val));
                } else if key.eq_ignore_ascii_case("OUTPUT UNIT") {
                    output_unit_raw = Some(String::from(val));
                } else if key.eq_ignore_ascii_case("A0") {
                    a0_comment = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("INSTGAIN") {
                    instgain_comment = parse_leading_f64(val);
                } else if key.eq_ignore_ascii_case("SENSITIVITY") {
                    sensitivity_comment = parse_leading_f64(val);
                }
            }
            continue;
        }
        // Also strip inline `*` or `#` trailing comments if present on directive lines.
        let clean = trimmed.split(['*', '#']).next().unwrap_or("").trim();
        if !clean.is_empty() {
            body_lines.push(clean);
        }
    }

    let mut zeros: Option<Vec<Complex64>> = None;
    let mut poles: Option<Vec<Complex64>> = None;
    let mut constant: Option<f64> = None;

    let mut idx = 0_usize;
    while idx < body_lines.len() {
        let mut toks = body_lines[idx].split_whitespace();
        let kw = toks.next().unwrap_or("");
        if kw.eq_ignore_ascii_case("ZEROS") || kw.eq_ignore_ascii_case("ZEROES") {
            let count_str = toks.next().ok_or_else(|| {
                EarthSnailError::Format("SACPZ 'ZEROS' directive missing integer count".into())
            })?;
            let nz: usize = count_str.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SACPZ ZEROS count '{count_str}'"))
            })?;
            if nz > MAX_SACPZ_ROOTS {
                return Err(EarthSnailError::CapacityExceeded(format!(
                    "SACPZ ZEROS count {nz} exceeds safety limit {MAX_SACPZ_ROOTS}"
                )));
            }
            idx += 1;
            let mut z_vec = Vec::with_capacity(nz);
            while idx < body_lines.len() && z_vec.len() < nz {
                if is_sacpz_keyword_line(body_lines[idx]) {
                    break;
                }
                z_vec.push(parse_complex_pair(body_lines[idx], "ZERO")?);
                idx += 1;
            }
            // Classic SAC convention: omitted trailing zeros default to `(0.0, 0.0)` at the origin!
            while z_vec.len() < nz {
                z_vec.push(Complex64::ZERO);
            }
            zeros = Some(z_vec);
        } else if kw.eq_ignore_ascii_case("POLES") {
            let count_str = toks.next().ok_or_else(|| {
                EarthSnailError::Format("SACPZ 'POLES' directive missing integer count".into())
            })?;
            let np: usize = count_str.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SACPZ POLES count '{count_str}'"))
            })?;
            if np > MAX_SACPZ_ROOTS {
                return Err(EarthSnailError::CapacityExceeded(format!(
                    "SACPZ POLES count {np} exceeds safety limit {MAX_SACPZ_ROOTS}"
                )));
            }
            idx += 1;
            let mut p_vec = Vec::with_capacity(np);
            while idx < body_lines.len() && p_vec.len() < np {
                if is_sacpz_keyword_line(body_lines[idx]) {
                    break;
                }
                p_vec.push(parse_complex_pair(body_lines[idx], "POLE")?);
                idx += 1;
            }
            while p_vec.len() < np {
                p_vec.push(Complex64::ZERO);
            }
            poles = Some(p_vec);
        } else if kw.eq_ignore_ascii_case("CONSTANT") {
            let c_str = toks.next().ok_or_else(|| {
                EarthSnailError::Format("SACPZ 'CONSTANT' directive missing value".into())
            })?;
            let c_val: f64 = c_str.parse().map_err(|_| {
                EarthSnailError::Format(format!("invalid SACPZ CONSTANT '{c_str}'"))
            })?;
            if !c_val.is_finite() {
                return Err(EarthSnailError::ResponseError(
                    "non-finite CONSTANT in SACPZ file".into(),
                ));
            }
            constant = Some(c_val);
            idx += 1;
        } else {
            return Err(EarthSnailError::Format(format!(
                "unexpected SACPZ directive '{}' in line '{}'",
                kw, body_lines[idx]
            )));
        }
    }

    let zeros_vec = zeros.unwrap_or_default();
    let poles_vec = poles.unwrap_or_default();
    let constant_val = constant.unwrap_or(1.0);
    if zeros_vec.is_empty() && poles_vec.is_empty() && constant.is_none() {
        return Err(EarthSnailError::Format(
            "empty or invalid SACPZ file (missing ZEROS, POLES, and CONSTANT)".into(),
        ));
    }

    let input_units = match input_unit_raw.as_deref() {
        Some(u) => {
            let ku = KinematicUnit::from_unit_str(u);
            if ku == KinematicUnit::Other {
                KinematicUnit::Displacement
            } else {
                ku
            }
        }
        None => KinematicUnit::Displacement,
    };
    let output_units = output_unit_raw
        .clone()
        .unwrap_or_else(|| String::from("COUNTS"));

    let mut response =
        ChannelResponse::from_sacpz(zeros_vec.clone(), poles_vec.clone(), constant_val);
    response.input_units = input_units;
    response.output_units = output_units;
    if let Some(sens) = sensitivity_comment {
        response.instrument_sensitivity = Some(StageGain {
            value: sens,
            frequency_hz: 1.0,
        });
    }

    let channel = if net.is_some() || sta.is_some() || cha.is_some() {
        Some(ChannelId::new(
            net.unwrap_or_default(),
            sta.unwrap_or_default(),
            loc.unwrap_or_default(),
            cha.unwrap_or_default(),
        ))
    } else {
        None
    };

    let metadata = channel.clone().map(|cid| ChannelMetadata {
        channel: cid,
        latitude: lat.unwrap_or(0.0),
        longitude: lon.unwrap_or(0.0),
        elevation_m: elev.unwrap_or(0.0),
        depth_m: depth.unwrap_or(0.0),
        azimuth_deg: azim,
        dip_deg: dip,
        sample_rate_hz: srate,
        start_date,
        end_date,
        response: Some(response.clone()),
    });

    Ok(SacPzRecord {
        channel,
        metadata,
        input_unit_raw,
        output_unit_raw,
        a0: a0_comment,
        instgain: instgain_comment,
        sensitivity: sensitivity_comment,
        zeros: zeros_vec,
        poles: poles_vec,
        constant: constant_val,
        response,
    })
}

/// Parses a SAC PoleZero (`SACPZ`) UTF-8 byte slice into a `SacPzRecord`.
pub fn read_sacpz_bytes(bytes: &[u8]) -> Result<SacPzRecord> {
    let s = core::str::from_utf8(bytes)
        .map_err(|_| EarthSnailError::Format("SACPZ file is not valid UTF-8".into()))?;
    read_sacpz_str(s)
}

/// Parses a SACPZ text directly into a `ChannelResponse`.
pub fn parse_sacpz_response(input: &str) -> Result<ChannelResponse> {
    Ok(read_sacpz_str(input)?.response)
}

/// Reads a SAC PoleZero (`SACPZ`) file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_sacpz_file(path: impl AsRef<std::path::Path>) -> Result<SacPzRecord> {
    let bytes = std::fs::read(path)?;
    read_sacpz_bytes(&bytes)
}

/// Serializes a `ChannelResponse` (and optional `ChannelMetadata`) into a standard SAC PoleZero
/// (`SACPZ`) text file with `INPUT UNIT : M` (displacement in `rad/s`), automatically converting
/// Hz poles/zeros to `rad/s` and adding origin zeros when converting from `VEL` or `ACC`!
pub fn write_sacpz_str(
    response: &ChannelResponse,
    metadata: Option<&ChannelMetadata>,
) -> Result<String> {
    let pz_stage = response
        .stages
        .iter()
        .find_map(|st| match st {
            ResponseStage::PolesZeros(pz) => Some(pz),
            _ => None,
        })
        .ok_or_else(|| {
            EarthSnailError::ResponseError(
                "ChannelResponse contains no PolesZerosStage for SACPZ export".into(),
            )
        })?;

    if pz_stage.pz_transfer_function_type == PzTransferFunctionType::DigitalZTransform {
        return Err(EarthSnailError::Unsupported(
            "cannot export digital Z-transform PolesZerosStage as analog Laplace SACPZ".into(),
        ));
    }

    let scale_to_rad = match pz_stage.pz_transfer_function_type {
        PzTransferFunctionType::RadPerSec => 1.0,
        PzTransferFunctionType::Hz => core::f64::consts::TAU,
        PzTransferFunctionType::DigitalZTransform => unreachable!(),
    };

    let mut zeros_rad: Vec<Complex64> = pz_stage.zeros.iter().map(|z| *z * scale_to_rad).collect();
    let poles_rad: Vec<Complex64> = pz_stage.poles.iter().map(|p| *p * scale_to_rad).collect();

    // Scale `a0` when converting from Hz (`prod(i f - z_hz)` -> `prod(i 2pi f - z_rad)`):
    // multiplying each zero by `2pi` multiplies numerator by `(2pi)^nz`, and poles by `(2pi)^np`,
    // so `a0_rad = a0_hz * (2pi)^(np - nz)`.
    let mut a0_rad = pz_stage.a0;
    if pz_stage.pz_transfer_function_type == PzTransferFunctionType::Hz {
        let exp_diff = (poles_rad.len() as i32) - (zeros_rad.len() as i32);
        a0_rad *= earthsnail_core::math::powi(core::f64::consts::TAU, exp_diff);
    }

    // Total gain: prefer `instrument_sensitivity` if multi-stage, otherwise product of stage gains:
    let total_sens = match response.instrument_sensitivity {
        Some(s) if response.stages.len() > 1 => s.value,
        _ => response
            .stages
            .iter()
            .map(|st| match st {
                ResponseStage::PolesZeros(p) => p.stage_gain,
                ResponseStage::Fir(f) => f.stage_gain,
                ResponseStage::Coefficients(c) => c.stage_gain,
                ResponseStage::GainOnly(g) => g.value,
                ResponseStage::Polynomial(_) => 1.0,
            })
            .product::<f64>(),
    };

    // Convert input units to Displacement (`M`):
    // - Velocity (`m/s`) -> Displacement (`m`): add 1 zero at `(0, 0)`!
    // - Acceleration (`m/s^2`) -> Displacement (`m`): add 2 zeros at `(0, 0)`!
    let extra_origin_zeros = match response.input_units {
        KinematicUnit::Displacement | KinematicUnit::Other => 0,
        KinematicUnit::Velocity => 1,
        KinematicUnit::Acceleration => 2,
    };
    for _ in 0..extra_origin_zeros {
        zeros_rad.push(Complex64::ZERO);
    }

    // Note: in `RadPerSec`, adding `k` origin zeros multiplies `|H(f0)|` by `(2pi * f0)^k`?
    // Wait: `H_vel(s) = G * A0_vel * prod(s - z) / prod(s - p)`, and `H_disp(s) = s * H_vel(s)`!
    // Since `s = (s - 0)`, appending `(0, 0)` to `zeros_rad` multiplies `prod(s - z)` by `s` directly
    // while keeping `CONSTANT = A0_vel * total_sens` UNCHANGED! (Matches `iu_anmo_bhz.sacpz`:
    // `8.46585e17 * 1.97468e9 == 1.671734e27`!).
    let constant = a0_rad * total_sens;

    let mut out = String::new();
    if let Some(meta) = metadata {
        let _ = writeln!(out, "* **************************************************");
        let _ = writeln!(out, "* NETWORK     : {}", meta.channel.network);
        let _ = writeln!(out, "* STATION     : {}", meta.channel.station);
        let _ = writeln!(out, "* LOCATION    : {}", meta.channel.location);
        let _ = writeln!(out, "* CHANNEL     : {}", meta.channel.channel);
        if let Some(st) = meta.start_date {
            let _ = writeln!(out, "* START       : {}", st.to_iso8601());
        }
        if let Some(ed) = meta.end_date {
            let _ = writeln!(out, "* END         : {}", ed.to_iso8601());
        }
        let _ = writeln!(out, "* LATITUDE    : {}", meta.latitude);
        let _ = writeln!(out, "* LONGITUDE   : {}", meta.longitude);
        let _ = writeln!(out, "* ELEVATION   : {}", meta.elevation_m);
        let _ = writeln!(out, "* DEPTH       : {}", meta.depth_m);
        if let Some(dip) = meta.dip_deg {
            let _ = writeln!(out, "* DIP (SEED)  : {dip}");
        }
        if let Some(az) = meta.azimuth_deg {
            let _ = writeln!(out, "* AZIMUTH     : {az}");
        }
        if let Some(sr) = meta.sample_rate_hz {
            let _ = writeln!(out, "* SAMPLE RATE : {sr}");
        }
        let _ = writeln!(out, "* INPUT UNIT  : M");
        let _ = writeln!(out, "* OUTPUT UNIT : {}", response.output_units);
        let _ = writeln!(out, "* INSTGAIN    : {}", pz_stage.stage_gain);
        let _ = writeln!(out, "* SENSITIVITY : {total_sens}");
        let _ = writeln!(out, "* A0          : {a0_rad:.6e}");
        let _ = writeln!(out, "* **************************************************");
    } else {
        let _ = writeln!(out, "* INPUT UNIT  : M");
        let _ = writeln!(out, "* OUTPUT UNIT : {}", response.output_units);
    }

    let _ = writeln!(out, "ZEROS {}", zeros_rad.len());
    for z in &zeros_rad {
        let _ = writeln!(out, " {:+.6e} {:+.6e}", z.re, z.im);
    }
    let _ = writeln!(out, "POLES {}", poles_rad.len());
    for p in &poles_rad {
        let _ = writeln!(out, " {:+.6e} {:+.6e}", p.re, p.im);
    }
    let _ = writeln!(out, "CONSTANT {constant:.6e}");
    Ok(out)
}

/// Writes a SAC PoleZero (`SACPZ`) file to disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_sacpz_file(
    path: impl AsRef<std::path::Path>,
    response: &ChannelResponse,
    metadata: Option<&ChannelMetadata>,
) -> Result<()> {
    let s = write_sacpz_str(response, metadata)?;
    std::fs::write(path, s.as_bytes())?;
    Ok(())
}

fn is_sacpz_keyword_line(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or("");
    first.eq_ignore_ascii_case("ZEROS")
        || first.eq_ignore_ascii_case("ZEROES")
        || first.eq_ignore_ascii_case("POLES")
        || first.eq_ignore_ascii_case("CONSTANT")
}

fn parse_complex_pair(line: &str, label: &str) -> Result<Complex64> {
    let mut toks = line.split_whitespace();
    let re_str = toks.next().ok_or_else(|| {
        EarthSnailError::Format(format!("missing real part on SACPZ {label} line '{line}'"))
    })?;
    let im_str = toks.next().ok_or_else(|| {
        EarthSnailError::Format(format!(
            "missing imaginary part on SACPZ {label} line '{line}'"
        ))
    })?;
    let re: f64 = re_str.parse().map_err(|_| {
        EarthSnailError::Format(format!("invalid SACPZ {label} real float '{re_str}'"))
    })?;
    let im: f64 = im_str.parse().map_err(|_| {
        EarthSnailError::Format(format!("invalid SACPZ {label} imag float '{im_str}'"))
    })?;
    if !re.is_finite() || !im.is_finite() {
        return Err(EarthSnailError::ResponseError(format!(
            "non-finite SACPZ {label} value ({re}, {im})"
        )));
    }
    Ok(Complex64::new(re, im))
}

fn parse_leading_f64(val: &str) -> Option<f64> {
    let tok = val.split_whitespace().next()?;
    let v: f64 = tok.parse().ok()?;
    v.is_finite().then_some(v)
}
