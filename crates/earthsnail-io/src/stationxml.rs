//! FDSN `StationXML` (`1.0` – `1.2`) parser and serializer mapping `<FDSNStationXML>` documents
//! into `StationInventory` (`NetworkMetadata`, `StationMetadata`, `ChannelMetadata`, and full
//! multi-stage `ChannelResponse` cascades covering `PolesZeros` in `RADIANS/SECOND`, `HERTZ`,
//! and `DIGITAL (Z-TRANSFORM)`, symmetric/asymmetric `FIR`, rational `Coefficients`, `Polynomial`,
//! and `StageGain` stages, with automatic decimation sample-rate propagation).

use crate::xml::{XmlEvent, XmlReader};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use earthsnail_core::{
    xml_escape, ChannelId, ChannelMetadata, ChannelResponse, CoefficientsStage, Complex64,
    EarthSnailError, FirStage, FirSymmetry, KinematicUnit, NetworkMetadata, PolesZerosStage,
    PolynomialStage, PzTransferFunctionType, ResponseStage, Result, StageGain, StationInventory,
    StationMetadata, UtcTimestamp,
};

/// Parses an FDSN `StationXML` (`1.0` – `1.2`) UTF-8 string into a `StationInventory`.
#[inline]
pub fn read_stationxml_str(input: &str) -> Result<StationInventory> {
    parse_stationxml(input)
}

/// Parses an FDSN `StationXML` (`1.0` – `1.2`) document from UTF-8 text into `StationInventory`.
pub fn parse_stationxml(input: &str) -> Result<StationInventory> {
    let mut xml = XmlReader::new(input);
    let mut inv = StationInventory::new();
    let mut found_root = false;

    loop {
        let ev = xml.next_event()?;
        match ev {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ref attributes,
            } => {
                if !found_root {
                    if !local_name.eq_ignore_ascii_case("FDSNStationXML") {
                        return Err(EarthSnailError::Format(format!(
                            "expected <FDSNStationXML> root element, got <{raw_name}>"
                        )));
                    }
                    found_root = true;
                    continue;
                }
                match local_name {
                    "Source" => {
                        let s = xml.read_text_to_end(raw_name)?;
                        if !s.is_empty() {
                            inv.source = Some(s);
                        }
                    }
                    "Network" => {
                        let code = attr_val(attributes, "code").unwrap_or_default();
                        let start_date = attr_val(attributes, "startDate")
                            .and_then(|s| UtcTimestamp::parse_iso8601(&s).ok());
                        let end_date = attr_val(attributes, "endDate")
                            .and_then(|s| UtcTimestamp::parse_iso8601(&s).ok());
                        let net = parse_network(&mut xml, raw_name, code, start_date, end_date)?;
                        inv.networks.push(net);
                    }
                    _ => {
                        xml.skip_element(raw_name)?;
                    }
                }
            }
            XmlEvent::EndElement { .. } | XmlEvent::Text(_) => {}
            XmlEvent::Eof => break,
        }
    }

    if !found_root {
        return Err(EarthSnailError::Format("empty StationXML document".into()));
    }
    Ok(inv)
}

/// Alias for `parse_stationxml` operating on `&str`.
#[inline]
pub fn read_stationxml(input: &str) -> Result<StationInventory> {
    parse_stationxml(input)
}

/// Parses an FDSN `StationXML` UTF-8 byte buffer into `StationInventory`.
pub fn read_stationxml_bytes(bytes: &[u8]) -> Result<StationInventory> {
    let s = core::str::from_utf8(bytes)
        .map_err(|_| EarthSnailError::Format("StationXML input is not valid UTF-8".into()))?;
    parse_stationxml(s)
}

/// Reads an FDSN `StationXML` file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_stationxml_file(path: impl AsRef<std::path::Path>) -> Result<StationInventory> {
    let bytes = std::fs::read(path)?;
    read_stationxml_bytes(&bytes)
}

fn parse_network<'a>(
    xml: &mut XmlReader<'a>,
    net_raw_name: &str,
    code: String,
    start_date: Option<UtcTimestamp>,
    end_date: Option<UtcTimestamp>,
) -> Result<NetworkMetadata> {
    let mut net = NetworkMetadata {
        code,
        description: None,
        start_date,
        end_date,
        stations: Vec::new(),
    };
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        let ev = xml.next_event()?;
        match ev {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ref attributes,
            } => match local_name {
                "Description" => {
                    let d = xml.read_text_to_end(raw_name)?;
                    if !d.is_empty() {
                        net.description = Some(d);
                    }
                }
                "Station" => {
                    let sta_code = attr_val(attributes, "code").unwrap_or_default();
                    let st_start = attr_val(attributes, "startDate")
                        .and_then(|s| UtcTimestamp::parse_iso8601(&s).ok());
                    let st_end = attr_val(attributes, "endDate")
                        .and_then(|s| UtcTimestamp::parse_iso8601(&s).ok());
                    let sta = parse_station(xml, raw_name, &net.code, sta_code, st_start, st_end)?;
                    net.stations.push(sta);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == net_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Network>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(net)
}

fn parse_station<'a>(
    xml: &mut XmlReader<'a>,
    sta_raw_name: &str,
    net_code: &str,
    sta_code: String,
    start_date: Option<UtcTimestamp>,
    end_date: Option<UtcTimestamp>,
) -> Result<StationMetadata> {
    let mut sta = StationMetadata {
        code: sta_code,
        latitude: 0.0,
        longitude: 0.0,
        elevation_m: 0.0,
        site_name: None,
        start_date,
        end_date,
        channels: Vec::new(),
    };
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        let ev = xml.next_event()?;
        match ev {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ref attributes,
            } => match local_name {
                "Latitude" => sta.latitude = parse_xml_f64(xml, raw_name)?,
                "Longitude" => sta.longitude = parse_xml_f64(xml, raw_name)?,
                "Elevation" => sta.elevation_m = parse_xml_f64(xml, raw_name)?,
                "Site" => sta.site_name = parse_site_name(xml, raw_name)?,
                "Channel" => {
                    let cha_code = attr_val(attributes, "code").unwrap_or_default();
                    let loc_code = attr_val(attributes, "locationCode").unwrap_or_default();
                    let ch_start = attr_val(attributes, "startDate")
                        .and_then(|s| UtcTimestamp::parse_iso8601(&s).ok());
                    let ch_end = attr_val(attributes, "endDate")
                        .and_then(|s| UtcTimestamp::parse_iso8601(&s).ok());
                    let cid = ChannelId::new(net_code, &sta.code, loc_code, cha_code);
                    let ch = parse_channel(
                        xml,
                        raw_name,
                        cid,
                        sta.latitude,
                        sta.longitude,
                        sta.elevation_m,
                        ch_start,
                        ch_end,
                    )?;
                    sta.channels.push(ch);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == sta_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Station>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(sta)
}

fn parse_site_name<'a>(xml: &mut XmlReader<'a>, site_raw_name: &str) -> Result<Option<String>> {
    let mut name = None;
    let target_depth = xml.depth().saturating_sub(1);
    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => {
                if local_name == "Name" {
                    let t = xml.read_text_to_end(raw_name)?;
                    if !t.is_empty() {
                        name = Some(t);
                    }
                } else {
                    xml.skip_element(raw_name)?;
                }
            }
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == site_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Site>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(name)
}

#[allow(clippy::too_many_arguments)]
fn parse_channel<'a>(
    xml: &mut XmlReader<'a>,
    cha_raw_name: &str,
    channel: ChannelId,
    default_lat: f64,
    default_lon: f64,
    default_elev: f64,
    start_date: Option<UtcTimestamp>,
    end_date: Option<UtcTimestamp>,
) -> Result<ChannelMetadata> {
    let mut meta = ChannelMetadata {
        channel,
        latitude: default_lat,
        longitude: default_lon,
        elevation_m: default_elev,
        depth_m: 0.0,
        azimuth_deg: None,
        dip_deg: None,
        sample_rate_hz: None,
        start_date,
        end_date,
        response: None,
    };
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "Latitude" => meta.latitude = parse_xml_f64(xml, raw_name)?,
                "Longitude" => meta.longitude = parse_xml_f64(xml, raw_name)?,
                "Elevation" => meta.elevation_m = parse_xml_f64(xml, raw_name)?,
                "Depth" => meta.depth_m = parse_xml_f64(xml, raw_name)?,
                "Azimuth" => meta.azimuth_deg = Some(parse_xml_f64(xml, raw_name)?),
                "Dip" => meta.dip_deg = Some(parse_xml_f64(xml, raw_name)?),
                "SampleRate" => meta.sample_rate_hz = Some(parse_xml_f64(xml, raw_name)?),
                "Response" => {
                    let resp = parse_response(xml, raw_name, meta.sample_rate_hz)?;
                    meta.response = Some(resp);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == cha_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Channel>".into(),
                ));
            }
            _ => {}
        }
    }

    // If `<Response>` was parsed before `<SampleRate>` appeared in `<Channel>`, backfill missing
    // digital stage sample rates from `meta.sample_rate_hz`.
    if let (Some(sr), Some(ref mut resp)) = (meta.sample_rate_hz, &mut meta.response) {
        if sr > 0.0 {
            for st in &mut resp.stages {
                match st {
                    ResponseStage::Fir(f) if f.input_sample_rate_hz <= 0.0 => {
                        f.input_sample_rate_hz = sr;
                    }
                    ResponseStage::Coefficients(c) if c.input_sample_rate_hz <= 0.0 => {
                        c.input_sample_rate_hz = sr;
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(meta)
}

struct ParsedStageEntry {
    seq_number: u32,
    input_units_str: Option<String>,
    output_units_str: Option<String>,
    decim_input_sr: Option<f64>,
    decim_factor: Option<u32>,
    decim_delay_s: f64,
    decim_correction_s: f64,
    stage_gain: Option<StageGain>,
    stage_body: RawStageBody,
}

enum RawStageBody {
    None,
    PolesZeros {
        tf_type: PzTransferFunctionType,
        a0: f64,
        norm_freq: f64,
        zeros: Vec<Complex64>,
        poles: Vec<Complex64>,
    },
    Fir {
        symmetry: FirSymmetry,
        coeffs: Vec<f64>,
    },
    Coefficients {
        numerator: Vec<f64>,
        denominator: Vec<f64>,
    },
    Polynomial {
        coeffs: Vec<f64>,
    },
}

fn parse_response<'a>(
    xml: &mut XmlReader<'a>,
    resp_raw_name: &str,
    channel_sr: Option<f64>,
) -> Result<ChannelResponse> {
    let mut instrument_sensitivity: Option<StageGain> = None;
    let mut sens_in_units: Option<String> = None;
    let mut sens_out_units: Option<String> = None;
    let mut raw_stages: Vec<ParsedStageEntry> = Vec::new();

    let target_depth = xml.depth().saturating_sub(1);
    loop {
        let ev = xml.next_event()?;
        match ev {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ref attributes,
            } => match local_name {
                "InstrumentSensitivity" => {
                    let (sg, iu, ou) = parse_instrument_sensitivity(xml, raw_name)?;
                    instrument_sensitivity = sg;
                    sens_in_units = iu;
                    sens_out_units = ou;
                }
                "Stage" => {
                    let seq_num = attr_val(attributes, "number")
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or((raw_stages.len() + 1) as u32);
                    raw_stages.push(parse_stage(xml, raw_name, seq_num)?);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == resp_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Response>".into(),
                ));
            }
            _ => {}
        }
    }

    raw_stages.sort_by_key(|s| s.seq_number);

    // Propagate missing `decim_input_sr` across stages (ObsPy `get_sampling_rates` rules):
    // 1. Backfill leading analog/gain stages from the first stage that specifies `decim_input_sr`.
    if let Some(first_known_idx) = raw_stages.iter().position(|s| s.decim_input_sr.is_some()) {
        let first_sr = raw_stages[first_known_idx].decim_input_sr;
        for st in &mut raw_stages[..first_known_idx] {
            st.decim_input_sr = first_sr;
            if st.decim_factor.is_none() {
                st.decim_factor = Some(1);
            }
        }
    }
    // 2. Forward-propagate `output_sr = input_sr / factor` to any subsequent stage missing `decim_input_sr`.
    let mut running_sr = channel_sr;
    for st in &mut raw_stages {
        if let Some(isr) = st.decim_input_sr {
            let fac = st.decim_factor.unwrap_or(1).max(1) as f64;
            running_sr = Some(isr / fac);
        } else if let Some(rsr) = running_sr {
            st.decim_input_sr = Some(rsr);
            let fac = st.decim_factor.unwrap_or(1).max(1) as f64;
            running_sr = Some(rsr / fac);
        }
    }

    let first_stage_in_units = raw_stages
        .first()
        .and_then(|s| s.input_units_str.as_deref());
    let last_stage_out_units = raw_stages
        .last()
        .and_then(|s| s.output_units_str.as_deref());

    let input_units = sens_in_units
        .as_deref()
        .or(first_stage_in_units)
        .map(KinematicUnit::from_unit_str)
        .unwrap_or(KinematicUnit::Velocity);
    let output_units = sens_out_units
        .or_else(|| last_stage_out_units.map(String::from))
        .unwrap_or_else(|| String::from("COUNTS"));

    let mut stages = Vec::with_capacity(raw_stages.len());
    for st in raw_stages {
        let sg = st.stage_gain.unwrap_or(StageGain {
            value: 1.0,
            frequency_hz: 0.0,
        });
        let isr = st.decim_input_sr.or(channel_sr).unwrap_or(1.0);
        let built = match st.stage_body {
            RawStageBody::PolesZeros {
                tf_type,
                a0,
                norm_freq,
                zeros,
                poles,
            } => {
                let mut pz = PolesZerosStage {
                    pz_transfer_function_type: tf_type,
                    a0,
                    normalization_freq: norm_freq,
                    zeros,
                    poles,
                    stage_gain: sg.value,
                    input_sample_rate_hz: st.decim_input_sr.or(channel_sr),
                };
                // FDSN `evalresp` standard: recompute exact `A0` at `normalization_freq` (`> 0`) so
                // 5-digit XML rounding of `NormalizationFactor` does not bias passband amplitude!
                if norm_freq > 0.0 {
                    pz.recompute_a0();
                }
                ResponseStage::PolesZeros(pz)
            }
            RawStageBody::Fir { symmetry, coeffs } => ResponseStage::Fir(FirStage {
                symmetry,
                coefficients: coeffs,
                input_sample_rate_hz: isr,
                decimation_factor: st.decim_factor.unwrap_or(1).max(1),
                delay_s: st.decim_delay_s,
                correction_s: st.decim_correction_s,
                stage_gain: sg.value,
                gain_frequency_hz: sg.frequency_hz,
            }),
            RawStageBody::Coefficients {
                numerator,
                denominator,
            } => ResponseStage::Coefficients(CoefficientsStage {
                numerator,
                denominator,
                input_sample_rate_hz: isr,
                correction_s: st.decim_correction_s,
                stage_gain: sg.value,
                gain_frequency_hz: sg.frequency_hz,
            }),
            RawStageBody::Polynomial { coeffs } => ResponseStage::Polynomial(PolynomialStage {
                coefficients: coeffs,
            }),
            RawStageBody::None => ResponseStage::GainOnly(sg),
        };
        stages.push(built);
    }

    Ok(ChannelResponse {
        input_units,
        output_units,
        instrument_sensitivity,
        stages,
    })
}

fn parse_instrument_sensitivity<'a>(
    xml: &mut XmlReader<'a>,
    sens_raw_name: &str,
) -> Result<(Option<StageGain>, Option<String>, Option<String>)> {
    let mut val: Option<f64> = None;
    let mut freq: f64 = 1.0;
    let mut in_u: Option<String> = None;
    let mut out_u: Option<String> = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "Value" => val = Some(parse_xml_f64(xml, raw_name)?),
                "Frequency" => freq = parse_xml_f64(xml, raw_name)?,
                "InputUnits" => in_u = parse_units_name(xml, raw_name)?,
                "OutputUnits" => out_u = parse_units_name(xml, raw_name)?,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == sens_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <InstrumentSensitivity>".into(),
                ));
            }
            _ => {}
        }
    }

    let sg = val.map(|v| StageGain {
        value: v,
        frequency_hz: freq,
    });
    Ok((sg, in_u, out_u))
}

fn parse_stage<'a>(
    xml: &mut XmlReader<'a>,
    stage_raw_name: &str,
    seq_number: u32,
) -> Result<ParsedStageEntry> {
    let mut entry = ParsedStageEntry {
        seq_number,
        input_units_str: None,
        output_units_str: None,
        decim_input_sr: None,
        decim_factor: None,
        decim_delay_s: 0.0,
        decim_correction_s: 0.0,
        stage_gain: None,
        stage_body: RawStageBody::None,
    };
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "PolesZeros" => {
                    let (body, iu, ou) = parse_poles_zeros_element(xml, raw_name)?;
                    entry.stage_body = body;
                    if iu.is_some() {
                        entry.input_units_str = iu;
                    }
                    if ou.is_some() {
                        entry.output_units_str = ou;
                    }
                }
                "FIR" => {
                    let (body, iu, ou) = parse_fir_element(xml, raw_name)?;
                    entry.stage_body = body;
                    if iu.is_some() {
                        entry.input_units_str = iu;
                    }
                    if ou.is_some() {
                        entry.output_units_str = ou;
                    }
                }
                "Coefficients" => {
                    let (body, iu, ou) = parse_coefficients_element(xml, raw_name)?;
                    entry.stage_body = body;
                    if iu.is_some() {
                        entry.input_units_str = iu;
                    }
                    if ou.is_some() {
                        entry.output_units_str = ou;
                    }
                }
                "Polynomial" => {
                    let (body, iu, ou) = parse_polynomial_element(xml, raw_name)?;
                    entry.stage_body = body;
                    if iu.is_some() {
                        entry.input_units_str = iu;
                    }
                    if ou.is_some() {
                        entry.output_units_str = ou;
                    }
                }
                "Decimation" => {
                    let (isr, fac, dly, corr) = parse_decimation_element(xml, raw_name)?;
                    entry.decim_input_sr = isr;
                    entry.decim_factor = fac;
                    entry.decim_delay_s = dly;
                    entry.decim_correction_s = corr;
                }
                "StageGain" => {
                    entry.stage_gain = Some(parse_stage_gain_element(xml, raw_name)?);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == stage_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Stage>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(entry)
}

fn parse_poles_zeros_element<'a>(
    xml: &mut XmlReader<'a>,
    pz_raw_name: &str,
) -> Result<(RawStageBody, Option<String>, Option<String>)> {
    let mut tf_type = PzTransferFunctionType::RadPerSec;
    let mut a0 = 1.0_f64;
    let mut norm_freq = 1.0_f64;
    let mut zeros = Vec::new();
    let mut poles = Vec::new();
    let mut iu = None;
    let mut ou = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "InputUnits" => iu = parse_units_name(xml, raw_name)?,
                "OutputUnits" => ou = parse_units_name(xml, raw_name)?,
                "PzTransferFunctionType" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    let up = s.to_ascii_uppercase();
                    tf_type = if up.contains("HERTZ") {
                        PzTransferFunctionType::Hz
                    } else if up.contains("DIGITAL") || up.contains("Z-TRANSFORM") {
                        PzTransferFunctionType::DigitalZTransform
                    } else {
                        PzTransferFunctionType::RadPerSec
                    };
                }
                "NormalizationFactor" => a0 = parse_xml_f64(xml, raw_name)?,
                "NormalizationFrequency" => norm_freq = parse_xml_f64(xml, raw_name)?,
                "Zero" => zeros.push(parse_pole_or_zero(xml, raw_name)?),
                "Pole" => poles.push(parse_pole_or_zero(xml, raw_name)?),
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == pz_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <PolesZeros>".into(),
                ));
            }
            _ => {}
        }
    }

    Ok((
        RawStageBody::PolesZeros {
            tf_type,
            a0,
            norm_freq,
            zeros,
            poles,
        },
        iu,
        ou,
    ))
}

fn parse_pole_or_zero<'a>(xml: &mut XmlReader<'a>, pz_item_raw: &str) -> Result<Complex64> {
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    let target_depth = xml.depth().saturating_sub(1);
    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "Real" => re = parse_float_or_value_child(xml, raw_name)?,
                "Imaginary" => im = parse_float_or_value_child(xml, raw_name)?,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == pz_item_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Pole>/<Zero>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(Complex64::new(re, im))
}

fn parse_fir_element<'a>(
    xml: &mut XmlReader<'a>,
    fir_raw_name: &str,
) -> Result<(RawStageBody, Option<String>, Option<String>)> {
    let mut symmetry = FirSymmetry::None;
    let mut coeffs = Vec::new();
    let mut iu = None;
    let mut ou = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "InputUnits" => iu = parse_units_name(xml, raw_name)?,
                "OutputUnits" => ou = parse_units_name(xml, raw_name)?,
                "Symmetry" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    symmetry = if s.eq_ignore_ascii_case("ODD") {
                        FirSymmetry::Odd
                    } else if s.eq_ignore_ascii_case("EVEN") {
                        FirSymmetry::Even
                    } else {
                        FirSymmetry::None
                    };
                }
                "NumeratorCoefficient" => {
                    coeffs.push(parse_float_or_value_child(xml, raw_name)?);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == fir_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <FIR>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok((RawStageBody::Fir { symmetry, coeffs }, iu, ou))
}

fn parse_coefficients_element<'a>(
    xml: &mut XmlReader<'a>,
    cf_raw_name: &str,
) -> Result<(RawStageBody, Option<String>, Option<String>)> {
    let mut numerator = Vec::new();
    let mut denominator = Vec::new();
    let mut iu = None;
    let mut ou = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "InputUnits" => iu = parse_units_name(xml, raw_name)?,
                "OutputUnits" => ou = parse_units_name(xml, raw_name)?,
                "Numerator" => numerator.push(parse_float_or_value_child(xml, raw_name)?),
                "Denominator" => denominator.push(parse_float_or_value_child(xml, raw_name)?),
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == cf_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Coefficients>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok((
        RawStageBody::Coefficients {
            numerator,
            denominator,
        },
        iu,
        ou,
    ))
}

fn parse_polynomial_element<'a>(
    xml: &mut XmlReader<'a>,
    poly_raw_name: &str,
) -> Result<(RawStageBody, Option<String>, Option<String>)> {
    let mut coeffs = Vec::new();
    let mut iu = None;
    let mut ou = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "InputUnits" => iu = parse_units_name(xml, raw_name)?,
                "OutputUnits" => ou = parse_units_name(xml, raw_name)?,
                "Coefficient" => coeffs.push(parse_float_or_value_child(xml, raw_name)?),
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == poly_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Polynomial>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok((RawStageBody::Polynomial { coeffs }, iu, ou))
}

fn parse_decimation_element<'a>(
    xml: &mut XmlReader<'a>,
    dec_raw_name: &str,
) -> Result<(Option<f64>, Option<u32>, f64, f64)> {
    let mut isr = None;
    let mut factor = None;
    let mut delay = 0.0_f64;
    let mut corr = 0.0_f64;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "InputSampleRate" => isr = Some(parse_float_or_value_child(xml, raw_name)?),
                "Factor" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    let f: u32 = s.parse().map_err(|_| {
                        EarthSnailError::Format(format!("invalid Decimation/Factor '{s}'"))
                    })?;
                    factor = Some(f.max(1));
                }
                "Delay" => delay = parse_float_or_value_child(xml, raw_name)?,
                "Correction" => corr = parse_float_or_value_child(xml, raw_name)?,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == dec_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Decimation>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok((isr, factor, delay, corr))
}

fn parse_stage_gain_element<'a>(xml: &mut XmlReader<'a>, sg_raw_name: &str) -> Result<StageGain> {
    let mut value = 1.0_f64;
    let mut frequency_hz = 0.0_f64;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "Value" => value = parse_xml_f64(xml, raw_name)?,
                "Frequency" => frequency_hz = parse_xml_f64(xml, raw_name)?,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == sg_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <StageGain>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(StageGain {
        value,
        frequency_hz,
    })
}

fn parse_units_name<'a>(xml: &mut XmlReader<'a>, units_raw_name: &str) -> Result<Option<String>> {
    let mut name = None;
    let target_depth = xml.depth().saturating_sub(1);
    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => {
                if local_name == "Name" {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        name = Some(s);
                    }
                } else {
                    xml.skip_element(raw_name)?;
                }
            }
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == units_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <Units>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(name)
}

/// Handles both StationXML 1.1/1.2 direct float (`<Real>0.5</Real>`) and StationXML 1.0 `<Real><Value>0.5</Value></Real>`.
fn parse_float_or_value_child<'a>(xml: &mut XmlReader<'a>, elem_raw: &str) -> Result<f64> {
    let mut direct_text = String::new();
    let mut child_val: Option<f64> = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::Text(t) => {
                if xml.depth() == target_depth + 1 {
                    direct_text.push_str(&t);
                }
            }
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => {
                if local_name.eq_ignore_ascii_case("Value") {
                    child_val = Some(parse_xml_f64(xml, raw_name)?);
                } else {
                    xml.skip_element(raw_name)?;
                }
            }
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == elem_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(format!(
                    "unexpected EOF inside <{elem_raw}>"
                )));
            }
            _ => {}
        }
    }

    if let Some(v) = child_val {
        return Ok(v);
    }
    let s = direct_text.trim();
    let v: f64 = s
        .parse()
        .map_err(|_| EarthSnailError::Format(format!("invalid float '{s}' inside <{elem_raw}>")))?;
    if !v.is_finite() {
        return Err(EarthSnailError::Header(format!(
            "non-finite float '{s}' inside <{elem_raw}>"
        )));
    }
    Ok(v)
}

fn parse_xml_f64<'a>(xml: &mut XmlReader<'a>, elem_raw: &str) -> Result<f64> {
    parse_float_or_value_child(xml, elem_raw)
}

fn attr_val(attrs: &[crate::xml::XmlAttr<'_>], local: &str) -> Option<String> {
    attrs
        .iter()
        .find(|a| a.local_name == local)
        .map(|a| a.value.clone())
}

/// Serializes a `StationInventory` into valid FDSN `StationXML` `1.2` UTF-8 text.
pub fn write_stationxml_str(inv: &StationInventory) -> Result<String> {
    let mut out = String::new();
    let _ = writeln!(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let _ = writeln!(
        out,
        "<FDSNStationXML xmlns=\"http://www.fdsn.org/xml/station/1\" schemaVersion=\"1.2\">"
    );
    let source = inv.source.as_deref().unwrap_or("EarthSnail");
    let _ = writeln!(out, "  <Source>{}</Source>", xml_escape(source));
    let _ = writeln!(
        out,
        "  <Created>{}</Created>",
        UtcTimestamp::UNIX_EPOCH.to_iso8601()
    );

    for net in &inv.networks {
        let mut net_attrs = format!("code=\"{}\"", xml_escape(&net.code));
        if let Some(sd) = net.start_date {
            let _ = write!(net_attrs, " startDate=\"{}\"", sd.to_iso8601());
        }
        if let Some(ed) = net.end_date {
            let _ = write!(net_attrs, " endDate=\"{}\"", ed.to_iso8601());
        }
        let _ = writeln!(out, "  <Network {net_attrs}>");
        if let Some(ref desc) = net.description {
            let _ = writeln!(out, "    <Description>{}</Description>", xml_escape(desc));
        }

        for sta in &net.stations {
            let mut sta_attrs = format!("code=\"{}\"", xml_escape(&sta.code));
            if let Some(sd) = sta.start_date {
                let _ = write!(sta_attrs, " startDate=\"{}\"", sd.to_iso8601());
            }
            if let Some(ed) = sta.end_date {
                let _ = write!(sta_attrs, " endDate=\"{}\"", ed.to_iso8601());
            }
            let _ = writeln!(out, "    <Station {sta_attrs}>");
            let _ = writeln!(out, "      <Latitude>{}</Latitude>", sta.latitude);
            let _ = writeln!(out, "      <Longitude>{}</Longitude>", sta.longitude);
            let _ = writeln!(out, "      <Elevation>{}</Elevation>", sta.elevation_m);
            let _ = writeln!(out, "      <Site>");
            let site = sta.site_name.as_deref().unwrap_or(&sta.code);
            let _ = writeln!(out, "        <Name>{}</Name>", xml_escape(site));
            let _ = writeln!(out, "      </Site>");

            for ch in &sta.channels {
                let mut ch_attrs = format!(
                    "code=\"{}\" locationCode=\"{}\"",
                    xml_escape(&ch.channel.channel),
                    xml_escape(&ch.channel.location)
                );
                if let Some(sd) = ch.start_date {
                    let _ = write!(ch_attrs, " startDate=\"{}\"", sd.to_iso8601());
                }
                if let Some(ed) = ch.end_date {
                    let _ = write!(ch_attrs, " endDate=\"{}\"", ed.to_iso8601());
                }
                let _ = writeln!(out, "      <Channel {ch_attrs}>");
                let _ = writeln!(out, "        <Latitude>{}</Latitude>", ch.latitude);
                let _ = writeln!(out, "        <Longitude>{}</Longitude>", ch.longitude);
                let _ = writeln!(out, "        <Elevation>{}</Elevation>", ch.elevation_m);
                let _ = writeln!(out, "        <Depth>{}</Depth>", ch.depth_m);
                if let Some(az) = ch.azimuth_deg {
                    let _ = writeln!(out, "        <Azimuth>{az}</Azimuth>");
                }
                if let Some(dip) = ch.dip_deg {
                    let _ = writeln!(out, "        <Dip>{dip}</Dip>");
                }
                if let Some(sr) = ch.sample_rate_hz {
                    let _ = writeln!(out, "        <SampleRate>{sr}</SampleRate>");
                }
                if let Some(ref resp) = ch.response {
                    write_response_xml(&mut out, resp)?;
                }
                let _ = writeln!(out, "      </Channel>");
            }
            let _ = writeln!(out, "    </Station>");
        }
        let _ = writeln!(out, "  </Network>");
    }

    let _ = writeln!(out, "</FDSNStationXML>");
    Ok(out)
}

fn write_response_xml(out: &mut String, resp: &ChannelResponse) -> Result<()> {
    let _ = writeln!(out, "        <Response>");
    let in_u_str = kinematic_unit_to_fdsn(resp.input_units);
    let out_u_esc = xml_escape(&resp.output_units);

    if let Some(sens) = resp.instrument_sensitivity {
        let _ = writeln!(out, "          <InstrumentSensitivity>");
        let _ = writeln!(out, "            <Value>{}</Value>", sens.value);
        let _ = writeln!(
            out,
            "            <Frequency>{}</Frequency>",
            sens.frequency_hz
        );
        let _ = writeln!(
            out,
            "            <InputUnits><Name>{in_u_str}</Name></InputUnits>"
        );
        let _ = writeln!(
            out,
            "            <OutputUnits><Name>{out_u_esc}</Name></OutputUnits>"
        );
        let _ = writeln!(out, "          </InstrumentSensitivity>");
    }

    for (idx, st) in resp.stages.iter().enumerate() {
        let seq = idx + 1;
        let _ = writeln!(out, "          <Stage number=\"{seq}\">");
        match st {
            ResponseStage::PolesZeros(pz) => {
                let tf_str = match pz.pz_transfer_function_type {
                    PzTransferFunctionType::RadPerSec => "LAPLACE (RADIANS/SECOND)",
                    PzTransferFunctionType::Hz => "LAPLACE (HERTZ)",
                    PzTransferFunctionType::DigitalZTransform => "DIGITAL (Z-TRANSFORM)",
                };
                let _ = writeln!(out, "            <PolesZeros>");
                let st_in = if idx == 0 { in_u_str } else { "V" };
                let _ = writeln!(
                    out,
                    "              <InputUnits><Name>{st_in}</Name></InputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <OutputUnits><Name>V</Name></OutputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <PzTransferFunctionType>{tf_str}</PzTransferFunctionType>"
                );
                let _ = writeln!(
                    out,
                    "              <NormalizationFactor>{}</NormalizationFactor>",
                    pz.a0
                );
                let _ = writeln!(
                    out,
                    "              <NormalizationFrequency>{}</NormalizationFrequency>",
                    pz.normalization_freq
                );
                for (zi, z) in pz.zeros.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "              <Zero number=\"{zi}\"><Real>{}</Real><Imaginary>{}</Imaginary></Zero>",
                        z.re, z.im
                    );
                }
                for (pi, p) in pz.poles.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "              <Pole number=\"{pi}\"><Real>{}</Real><Imaginary>{}</Imaginary></Pole>",
                        p.re, p.im
                    );
                }
                let _ = writeln!(out, "            </PolesZeros>");
                if let Some(isr) = pz.input_sample_rate_hz {
                    let _ = writeln!(
                        out,
                        "            <Decimation><InputSampleRate>{isr}</InputSampleRate><Factor>1</Factor><Offset>0</Offset><Delay>0.0</Delay><Correction>0.0</Correction></Decimation>"
                    );
                }
                let _ = writeln!(
                    out,
                    "            <StageGain><Value>{}</Value><Frequency>{}</Frequency></StageGain>",
                    pz.stage_gain, pz.normalization_freq
                );
            }
            ResponseStage::Fir(fir) => {
                let sym = match fir.symmetry {
                    FirSymmetry::None => "NONE",
                    FirSymmetry::Odd => "ODD",
                    FirSymmetry::Even => "EVEN",
                };
                let _ = writeln!(out, "            <FIR>");
                let _ = writeln!(
                    out,
                    "              <InputUnits><Name>counts</Name></InputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <OutputUnits><Name>{out_u_esc}</Name></OutputUnits>"
                );
                let _ = writeln!(out, "              <Symmetry>{sym}</Symmetry>");
                for c in &fir.coefficients {
                    let _ = writeln!(
                        out,
                        "              <NumeratorCoefficient>{c}</NumeratorCoefficient>"
                    );
                }
                let _ = writeln!(out, "            </FIR>");
                let _ = writeln!(
                    out,
                    "            <Decimation><InputSampleRate>{}</InputSampleRate><Factor>{}</Factor><Offset>0</Offset><Delay>{}</Delay><Correction>{}</Correction></Decimation>",
                    fir.input_sample_rate_hz, fir.decimation_factor, fir.delay_s, fir.correction_s
                );
                let _ = writeln!(
                    out,
                    "            <StageGain><Value>{}</Value><Frequency>{}</Frequency></StageGain>",
                    fir.stage_gain, fir.gain_frequency_hz
                );
            }
            ResponseStage::Coefficients(cf) => {
                let _ = writeln!(out, "            <Coefficients>");
                let _ = writeln!(
                    out,
                    "              <InputUnits><Name>counts</Name></InputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <OutputUnits><Name>{out_u_esc}</Name></OutputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <CfTransferFunctionType>DIGITAL</CfTransferFunctionType>"
                );
                for (ni, n) in cf.numerator.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "              <Numerator number=\"{ni}\">{n}</Numerator>"
                    );
                }
                for (di, d) in cf.denominator.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "              <Denominator number=\"{di}\">{d}</Denominator>"
                    );
                }
                let _ = writeln!(out, "            </Coefficients>");
                let _ = writeln!(
                    out,
                    "            <Decimation><InputSampleRate>{}</InputSampleRate><Factor>1</Factor><Offset>0</Offset><Delay>{}</Delay><Correction>{}</Correction></Decimation>",
                    cf.input_sample_rate_hz, cf.correction_s, cf.correction_s
                );
                let _ = writeln!(
                    out,
                    "            <StageGain><Value>{}</Value><Frequency>{}</Frequency></StageGain>",
                    cf.stage_gain, cf.gain_frequency_hz
                );
            }
            ResponseStage::GainOnly(g) => {
                let _ = writeln!(
                    out,
                    "            <StageGain><Value>{}</Value><Frequency>{}</Frequency></StageGain>",
                    g.value, g.frequency_hz
                );
            }
            ResponseStage::Polynomial(poly) => {
                let _ = writeln!(out, "            <Polynomial>");
                let _ = writeln!(
                    out,
                    "              <InputUnits><Name>{in_u_str}</Name></InputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <OutputUnits><Name>{out_u_esc}</Name></OutputUnits>"
                );
                let _ = writeln!(
                    out,
                    "              <ApproximationType>MACLAURIN</ApproximationType>"
                );
                for (ci, c) in poly.coefficients.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "              <Coefficient number=\"{ci}\">{c}</Coefficient>"
                    );
                }
                let _ = writeln!(out, "            </Polynomial>");
            }
        }
        let _ = writeln!(out, "          </Stage>");
    }

    let _ = writeln!(out, "        </Response>");
    Ok(())
}

fn kinematic_unit_to_fdsn(u: KinematicUnit) -> &'static str {
    match u {
        KinematicUnit::Displacement => "m",
        KinematicUnit::Velocity => "m/s",
        KinematicUnit::Acceleration => "m/s**2",
        KinematicUnit::Other => "unknown",
    }
}

/// Writes a `StationInventory` as FDSN `StationXML` `1.2` to disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_stationxml_file(
    path: impl AsRef<std::path::Path>,
    inv: &StationInventory,
) -> Result<()> {
    let xml = write_stationxml_str(inv)?;
    std::fs::write(path, xml.as_bytes())?;
    Ok(())
}
