//! FDSN `QuakeML` (`1.2`) parser and serializer mapping `<quakeml>` / `<eventParameters>`
//! documents into `EventCatalog` (`Event`, `Origin`, `OriginQuality`, `Arrival`, `Magnitude`,
//! `Pick`, `FocalMechanism`, `NodalPlane`, `PrincipalAxes`, and 6-component spherical
//! `MomentTensor6` `Mrr..Mtp` + `scalarMoment`).

use crate::xml::{XmlAttr, XmlEvent, XmlReader};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use earthsnail_core::{
    xml_escape, Arrival, ChannelId, EarthSnailError, EvaluationMode, Event, EventCatalog,
    FocalMechanism, Magnitude, MomentTensor6, NodalPlane, Origin, OriginQuality, Pick, PickOnset,
    PickPolarity, PrincipalAxes, PrincipalAxis, Result, UtcTimestamp, KM_PER_DEG,
};

/// Parses an FDSN `QuakeML` (`1.2`) UTF-8 string into an `EventCatalog`.
pub fn read_quakeml_str(input: &str) -> Result<EventCatalog> {
    let mut xml = XmlReader::new(input);
    let mut catalog = EventCatalog::new();
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
                    if !local_name.eq_ignore_ascii_case("quakeml")
                        && !local_name.eq_ignore_ascii_case("eventParameters")
                    {
                        return Err(EarthSnailError::Format(format!(
                            "expected <quakeml> or <eventParameters> root element, got <{raw_name}>"
                        )));
                    }
                    found_root = true;
                    if local_name.eq_ignore_ascii_case("eventParameters") {
                        parse_event_parameters(&mut xml, raw_name, &mut catalog)?;
                    }
                    continue;
                }

                match local_name {
                    "eventParameters" => {
                        parse_event_parameters(&mut xml, raw_name, &mut catalog)?;
                    }
                    "event" => {
                        let pid = attr_str(attributes, "publicID").unwrap_or_default();
                        catalog.events.push(parse_event(&mut xml, raw_name, pid)?);
                    }
                    _ => xml.skip_element(raw_name)?,
                }
            }
            XmlEvent::EndElement { .. } | XmlEvent::Text(_) => {}
            XmlEvent::Eof => break,
        }
    }

    if !found_root {
        return Err(EarthSnailError::Format("empty QuakeML document".into()));
    }
    Ok(catalog)
}

/// Parses an FDSN `QuakeML` (`1.2`) byte buffer into an `EventCatalog`.
pub fn read_quakeml_bytes(bytes: &[u8]) -> Result<EventCatalog> {
    let s = core::str::from_utf8(bytes)
        .map_err(|_| EarthSnailError::Format("QuakeML input is not valid UTF-8".into()))?;
    read_quakeml_str(s)
}

/// Reads an FDSN `QuakeML` (`1.2`) file from disk (`std` feature).
#[cfg(feature = "std")]
pub fn read_quakeml_file(path: impl AsRef<std::path::Path>) -> Result<EventCatalog> {
    let bytes = std::fs::read(path)?;
    read_quakeml_bytes(&bytes)
}

fn parse_event_parameters<'a>(
    xml: &mut XmlReader<'a>,
    ep_raw_name: &str,
    catalog: &mut EventCatalog,
) -> Result<()> {
    let target_depth = xml.depth().saturating_sub(1);
    loop {
        let ev = xml.next_event()?;
        match ev {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ref attributes,
            } => match local_name {
                "description" => {
                    if let Some(desc) = parse_description_text(xml, raw_name)? {
                        catalog.description = Some(desc);
                    }
                }
                "event" => {
                    let pid = attr_str(attributes, "publicID").unwrap_or_default();
                    catalog.events.push(parse_event(xml, raw_name, pid)?);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == ep_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <eventParameters>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_event<'a>(xml: &mut XmlReader<'a>, ev_raw_name: &str, public_id: String) -> Result<Event> {
    let mut event = Event {
        public_id,
        ..Event::default()
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
                "preferredOriginID" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        event.preferred_origin_id = Some(s);
                    }
                }
                "preferredMagnitudeID" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        event.preferred_magnitude_id = Some(s);
                    }
                }
                "preferredFocalMechanismID" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        event.preferred_focal_mechanism_id = Some(s);
                    }
                }
                "type" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        event.event_type = Some(s);
                    }
                }
                "description" => {
                    if let Some(d) = parse_description_text(xml, raw_name)? {
                        if event.description.is_none() {
                            event.description = Some(d);
                        }
                    }
                }
                "origin" => {
                    let pid = attr_str(attributes, "publicID");
                    event.origins.push(parse_origin(xml, raw_name, pid)?);
                }
                "magnitude" => {
                    let pid = attr_str(attributes, "publicID");
                    event.magnitudes.push(parse_magnitude(xml, raw_name, pid)?);
                }
                "pick" => {
                    let pid = attr_str(attributes, "publicID");
                    event.picks.push(parse_pick(xml, raw_name, pid)?);
                }
                "focalMechanism" => {
                    let pid = attr_str(attributes, "publicID");
                    event
                        .focal_mechanisms
                        .push(parse_focal_mechanism(xml, raw_name, pid)?);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == ev_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <event>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(event)
}

fn parse_origin<'a>(
    xml: &mut XmlReader<'a>,
    org_raw_name: &str,
    public_id: Option<String>,
) -> Result<Origin> {
    let mut time: Option<UtcTimestamp> = None;
    let mut time_unc: Option<f64> = None;
    let mut lat: Option<f64> = None;
    let mut lat_unc_km: Option<f64> = None;
    let mut lon: Option<f64> = None;
    let mut lon_unc_km: Option<f64> = None;
    let mut depth_m = 0.0_f64;
    let mut depth_unc_m: Option<f64> = None;
    let mut quality: Option<OriginQuality> = None;
    let mut arrivals: Vec<Arrival> = Vec::new();
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "time" => {
                    let (ts, unc) = parse_time_quantity(xml, raw_name)?;
                    time = Some(ts);
                    time_unc = unc;
                }
                "latitude" => {
                    let (v, u_deg) = parse_real_quantity(xml, raw_name)?;
                    lat = Some(v);
                    lat_unc_km = u_deg.map(|d| d * KM_PER_DEG);
                }
                "longitude" => {
                    let (v, u_deg) = parse_real_quantity(xml, raw_name)?;
                    lon = Some(v);
                    lon_unc_km = u_deg.map(|d| d * KM_PER_DEG);
                }
                "depth" => {
                    let (v, u_m) = parse_real_quantity(xml, raw_name)?;
                    depth_m = v;
                    depth_unc_m = u_m;
                }
                "quality" => {
                    quality = Some(parse_origin_quality(xml, raw_name)?);
                }
                "arrival" => {
                    arrivals.push(parse_arrival(xml, raw_name)?);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == org_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <origin>".into(),
                ));
            }
            _ => {}
        }
    }

    let time = time.ok_or_else(|| {
        EarthSnailError::Format("QuakeML <origin> missing required <time>".into())
    })?;
    let latitude = lat.ok_or_else(|| {
        EarthSnailError::Format("QuakeML <origin> missing required <latitude>".into())
    })?;
    let longitude = lon.ok_or_else(|| {
        EarthSnailError::Format("QuakeML <origin> missing required <longitude>".into())
    })?;

    Ok(Origin {
        public_id,
        time,
        latitude,
        longitude,
        depth_m,
        time_uncertainty_s: time_unc,
        latitude_uncertainty_km: lat_unc_km,
        longitude_uncertainty_km: lon_unc_km,
        depth_uncertainty_m: depth_unc_m,
        quality,
        arrivals,
    })
}

fn parse_origin_quality<'a>(xml: &mut XmlReader<'a>, q_raw_name: &str) -> Result<OriginQuality> {
    let mut q = OriginQuality::default();
    let target_depth = xml.depth().saturating_sub(1);
    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "associatedPhaseCount" => {
                    q.associated_phase_count = parse_opt_u32(xml, raw_name)?;
                }
                "usedPhaseCount" => {
                    q.used_phase_count = parse_opt_u32(xml, raw_name)?;
                }
                "standardError" => {
                    q.standard_error_s = Some(parse_real_quantity(xml, raw_name)?.0);
                }
                "azimuthalGap" => {
                    q.azimuthal_gap_deg = Some(parse_real_quantity(xml, raw_name)?.0);
                }
                "minimumDistance" => {
                    q.minimum_distance_deg = Some(parse_real_quantity(xml, raw_name)?.0);
                }
                "maximumDistance" => {
                    q.maximum_distance_deg = Some(parse_real_quantity(xml, raw_name)?.0);
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == q_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <quality>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(q)
}

fn parse_arrival<'a>(xml: &mut XmlReader<'a>, arr_raw_name: &str) -> Result<Arrival> {
    let mut pick_id = None;
    let mut phase = String::new();
    let mut azimuth_deg = None;
    let mut distance_deg = None;
    let mut takeoff_angle_deg = None;
    let mut time_residual_s = None;
    let mut time_weight = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "pickID" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        pick_id = Some(s);
                    }
                }
                "phase" => phase = xml.read_text_to_end(raw_name)?,
                "azimuth" => azimuth_deg = Some(parse_real_quantity(xml, raw_name)?.0),
                "distance" => distance_deg = Some(parse_real_quantity(xml, raw_name)?.0),
                "takeoffAngle" => takeoff_angle_deg = Some(parse_real_quantity(xml, raw_name)?.0),
                "timeResidual" => time_residual_s = Some(parse_real_quantity(xml, raw_name)?.0),
                "timeWeight" => time_weight = Some(parse_real_quantity(xml, raw_name)?.0),
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == arr_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <arrival>".into(),
                ));
            }
            _ => {}
        }
    }

    Ok(Arrival {
        pick_id,
        phase,
        azimuth_deg,
        distance_deg,
        takeoff_angle_deg,
        time_residual_s,
        time_weight,
    })
}

fn parse_magnitude<'a>(
    xml: &mut XmlReader<'a>,
    mag_raw_name: &str,
    public_id: Option<String>,
) -> Result<Magnitude> {
    let mut mag_val: Option<f64> = None;
    let mut uncertainty: Option<f64> = None;
    let mut mag_type = String::new();
    let mut origin_id = None;
    let mut station_count = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "mag" => {
                    let (v, u) = parse_real_quantity(xml, raw_name)?;
                    mag_val = Some(v);
                    uncertainty = u;
                }
                "type" => mag_type = xml.read_text_to_end(raw_name)?,
                "originID" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        origin_id = Some(s);
                    }
                }
                "stationCount" => station_count = parse_opt_u32(xml, raw_name)?,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == mag_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <magnitude>".into(),
                ));
            }
            _ => {}
        }
    }

    let mag = mag_val.ok_or_else(|| {
        EarthSnailError::Format("QuakeML <magnitude> missing required <mag>".into())
    })?;
    Ok(Magnitude {
        public_id,
        mag,
        mag_type,
        origin_id,
        station_count,
        uncertainty,
    })
}

fn parse_pick<'a>(
    xml: &mut XmlReader<'a>,
    pk_raw_name: &str,
    public_id: Option<String>,
) -> Result<Pick> {
    let mut time: Option<UtcTimestamp> = None;
    let mut time_uncertainty_s = None;
    let mut channel = ChannelId::default();
    let mut phase_hint = None;
    let mut polarity = None;
    let mut onset = None;
    let mut evaluation_mode = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        let ev = xml.next_event()?;
        match ev {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ref attributes,
            } => match local_name {
                "time" => {
                    let (t, u) = parse_time_quantity(xml, raw_name)?;
                    time = Some(t);
                    time_uncertainty_s = u;
                }
                "waveformID" => {
                    let net = attr_str(attributes, "networkCode").unwrap_or_default();
                    let sta = attr_str(attributes, "stationCode").unwrap_or_default();
                    let loc = attr_str(attributes, "locationCode").unwrap_or_default();
                    let cha = attr_str(attributes, "channelCode").unwrap_or_default();
                    channel = ChannelId::new(net, sta, loc, cha);
                    xml.skip_element(raw_name)?;
                }
                "phaseHint" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        phase_hint = Some(s);
                    }
                }
                "polarity" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    polarity = match s.as_str() {
                        p if p.eq_ignore_ascii_case("positive") => Some(PickPolarity::Positive),
                        p if p.eq_ignore_ascii_case("negative") => Some(PickPolarity::Negative),
                        p if p.eq_ignore_ascii_case("undecidable") => {
                            Some(PickPolarity::Undecidable)
                        }
                        _ => None,
                    };
                }
                "onset" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    onset = match s.as_str() {
                        o if o.eq_ignore_ascii_case("impulsive") => Some(PickOnset::Impulsive),
                        o if o.eq_ignore_ascii_case("emergent") => Some(PickOnset::Emergent),
                        o if o.eq_ignore_ascii_case("questionable") => {
                            Some(PickOnset::Questionable)
                        }
                        _ => None,
                    };
                }
                "evaluationMode" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    evaluation_mode = match s.as_str() {
                        m if m.eq_ignore_ascii_case("manual") => Some(EvaluationMode::Manual),
                        m if m.eq_ignore_ascii_case("automatic") => Some(EvaluationMode::Automatic),
                        _ => None,
                    };
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == pk_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <pick>".into(),
                ));
            }
            _ => {}
        }
    }

    let time = time
        .ok_or_else(|| EarthSnailError::Format("QuakeML <pick> missing required <time>".into()))?;
    Ok(Pick {
        public_id,
        time,
        channel,
        phase_hint,
        polarity,
        onset,
        time_uncertainty_s,
        evaluation_mode,
    })
}

fn parse_focal_mechanism<'a>(
    xml: &mut XmlReader<'a>,
    fm_raw_name: &str,
    public_id: Option<String>,
) -> Result<FocalMechanism> {
    let mut fm = FocalMechanism {
        public_id,
        ..FocalMechanism::default()
    };
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "nodalPlanes" => {
                    let (np1, np2) = parse_nodal_planes(xml, raw_name)?;
                    fm.nodal_plane_1 = np1;
                    fm.nodal_plane_2 = np2;
                }
                "principalAxes" => {
                    fm.principal_axes = parse_principal_axes(xml, raw_name)?;
                }
                "momentTensor" => {
                    fm.moment_tensor = parse_moment_tensor(xml, raw_name)?;
                }
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == fm_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <focalMechanism>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(fm)
}

fn parse_nodal_planes<'a>(
    xml: &mut XmlReader<'a>,
    np_raw_name: &str,
) -> Result<(Option<NodalPlane>, Option<NodalPlane>)> {
    let mut np1 = None;
    let mut np2 = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "nodalPlane1" => np1 = Some(parse_one_nodal_plane(xml, raw_name)?),
                "nodalPlane2" => np2 = Some(parse_one_nodal_plane(xml, raw_name)?),
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == np_raw_name =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <nodalPlanes>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok((np1, np2))
}

fn parse_one_nodal_plane<'a>(xml: &mut XmlReader<'a>, plane_raw: &str) -> Result<NodalPlane> {
    let mut strike = 0.0_f64;
    let mut dip = 0.0_f64;
    let mut rake = 0.0_f64;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "strike" => strike = parse_real_quantity(xml, raw_name)?.0,
                "dip" => dip = parse_real_quantity(xml, raw_name)?.0,
                "rake" => rake = parse_real_quantity(xml, raw_name)?.0,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == plane_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <nodalPlane>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(NodalPlane {
        strike_deg: strike,
        dip_deg: dip,
        rake_deg: rake,
    })
}

fn parse_principal_axes<'a>(
    xml: &mut XmlReader<'a>,
    pa_raw: &str,
) -> Result<Option<PrincipalAxes>> {
    let mut t_ax = None;
    let mut p_ax = None;
    let mut n_ax = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "tAxis" => t_ax = Some(parse_one_axis(xml, raw_name)?),
                "pAxis" => p_ax = Some(parse_one_axis(xml, raw_name)?),
                "nAxis" => n_ax = Some(parse_one_axis(xml, raw_name)?),
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == pa_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <principalAxes>".into(),
                ));
            }
            _ => {}
        }
    }
    match (t_ax, p_ax) {
        (Some(t_axis), Some(p_axis)) => Ok(Some(PrincipalAxes {
            t_axis,
            p_axis,
            n_axis: n_ax,
        })),
        _ => Ok(None),
    }
}

fn parse_one_axis<'a>(xml: &mut XmlReader<'a>, ax_raw: &str) -> Result<PrincipalAxis> {
    let mut azimuth_deg = 0.0_f64;
    let mut plunge_deg = 0.0_f64;
    let mut length = 0.0_f64;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "azimuth" => azimuth_deg = parse_real_quantity(xml, raw_name)?.0,
                "plunge" => plunge_deg = parse_real_quantity(xml, raw_name)?.0,
                "length" => length = parse_real_quantity(xml, raw_name)?.0,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == ax_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format("unexpected EOF inside axis".into()));
            }
            _ => {}
        }
    }
    Ok(PrincipalAxis {
        azimuth_deg,
        plunge_deg,
        length,
    })
}

fn parse_moment_tensor<'a>(xml: &mut XmlReader<'a>, mt_raw: &str) -> Result<Option<MomentTensor6>> {
    let mut tensor_opt = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => {
                if local_name == "tensor" {
                    tensor_opt = Some(parse_tensor_6(xml, raw_name)?);
                } else if local_name == "scalarMoment" {
                    // Validate `<scalarMoment><value>...</value></scalarMoment>` cleanly.
                    let _ = parse_real_quantity(xml, raw_name)?;
                } else {
                    xml.skip_element(raw_name)?;
                }
            }
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == mt_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <momentTensor>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(tensor_opt)
}

fn parse_tensor_6<'a>(xml: &mut XmlReader<'a>, ten_raw: &str) -> Result<MomentTensor6> {
    let mut mt = MomentTensor6::default();
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => match local_name {
                "Mrr" => mt.m_rr = parse_real_quantity(xml, raw_name)?.0,
                "Mtt" => mt.m_tt = parse_real_quantity(xml, raw_name)?.0,
                "Mpp" => mt.m_pp = parse_real_quantity(xml, raw_name)?.0,
                "Mrt" => mt.m_rt = parse_real_quantity(xml, raw_name)?.0,
                "Mrp" => mt.m_rp = parse_real_quantity(xml, raw_name)?.0,
                "Mtp" => mt.m_tp = parse_real_quantity(xml, raw_name)?.0,
                _ => xml.skip_element(raw_name)?,
            },
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == ten_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <tensor>".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(mt)
}

fn parse_description_text<'a>(xml: &mut XmlReader<'a>, desc_raw: &str) -> Result<Option<String>> {
    let mut direct = String::new();
    let mut text_child = None;
    let target_depth = xml.depth().saturating_sub(1);

    loop {
        match xml.next_event()? {
            XmlEvent::Text(t) => {
                if xml.depth() == target_depth + 1 {
                    direct.push_str(&t);
                }
            }
            XmlEvent::StartElement {
                raw_name,
                local_name,
                ..
            } => {
                if local_name == "text" {
                    let s = xml.read_text_to_end(raw_name)?;
                    if !s.is_empty() {
                        text_child = Some(s);
                    }
                } else {
                    xml.skip_element(raw_name)?;
                }
            }
            XmlEvent::EndElement { raw_name, .. }
                if xml.depth() == target_depth && raw_name == desc_raw =>
            {
                break;
            }
            XmlEvent::Eof => {
                return Err(EarthSnailError::Format(
                    "unexpected EOF inside <description>".into(),
                ));
            }
            _ => {}
        }
    }
    if text_child.is_some() {
        return Ok(text_child);
    }
    let dt = direct.trim();
    if dt.is_empty() {
        Ok(None)
    } else {
        Ok(Some(String::from(dt)))
    }
}

/// Parses either a `QuakeML` `RealQuantity` (`<tag><value>1.23</value><uncertainty>0.1</uncertainty></tag>`)
/// OR a direct scalar float (`<tag>1.23</tag>`).
fn parse_real_quantity<'a>(xml: &mut XmlReader<'a>, elem_raw: &str) -> Result<(f64, Option<f64>)> {
    let mut direct_text = String::new();
    let mut val_child: Option<f64> = None;
    let mut unc_child: Option<f64> = None;
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
            } => match local_name {
                "value" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    let v: f64 = s.parse().map_err(|_| {
                        EarthSnailError::Format(format!(
                            "invalid float '{s}' in <{elem_raw}><value>"
                        ))
                    })?;
                    if !v.is_finite() {
                        return Err(EarthSnailError::Header(format!(
                            "non-finite float '{s}' in <{elem_raw}><value>"
                        )));
                    }
                    val_child = Some(v);
                }
                "uncertainty" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if let Ok(u) = s.parse::<f64>() {
                        if u.is_finite() {
                            unc_child = Some(u);
                        }
                    }
                }
                _ => xml.skip_element(raw_name)?,
            },
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

    if let Some(v) = val_child {
        return Ok((v, unc_child));
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
    Ok((v, None))
}

fn parse_time_quantity<'a>(
    xml: &mut XmlReader<'a>,
    elem_raw: &str,
) -> Result<(UtcTimestamp, Option<f64>)> {
    let mut direct_text = String::new();
    let mut val_child: Option<UtcTimestamp> = None;
    let mut unc_child: Option<f64> = None;
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
            } => match local_name {
                "value" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    val_child = Some(UtcTimestamp::parse_iso8601(&s)?);
                }
                "uncertainty" => {
                    let s = xml.read_text_to_end(raw_name)?;
                    if let Ok(u) = s.parse::<f64>() {
                        if u.is_finite() {
                            unc_child = Some(u);
                        }
                    }
                }
                _ => xml.skip_element(raw_name)?,
            },
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

    if let Some(ts) = val_child {
        return Ok((ts, unc_child));
    }
    let ts = UtcTimestamp::parse_iso8601(direct_text.trim())?;
    Ok((ts, None))
}

fn parse_opt_u32<'a>(xml: &mut XmlReader<'a>, elem_raw: &str) -> Result<Option<u32>> {
    let s = xml.read_text_to_end(elem_raw)?;
    if s.is_empty() {
        return Ok(None);
    }
    let v: u32 = s.parse().map_err(|_| {
        EarthSnailError::Format(format!("invalid integer '{s}' inside <{elem_raw}>"))
    })?;
    Ok(Some(v))
}

fn attr_str(attrs: &[XmlAttr<'_>], local: &str) -> Option<String> {
    attrs
        .iter()
        .find(|a| a.local_name == local)
        .map(|a| a.value.clone())
}

/// Serializes an `EventCatalog` into valid FDSN `QuakeML` (`1.2`) UTF-8 XML.
pub fn write_quakeml_str(catalog: &EventCatalog) -> Result<String> {
    let mut out = String::new();
    let _ = writeln!(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    let _ = writeln!(
        out,
        "<q:quakeml xmlns=\"http://quakeml.org/xmlns/bed/1.2\" xmlns:q=\"http://quakeml.org/xmlns/quakeml/1.2\">"
    );
    let _ = writeln!(
        out,
        "  <eventParameters publicID=\"smi:earthsnail/catalog/1\">"
    );
    if let Some(ref desc) = catalog.description {
        let _ = writeln!(
            out,
            "    <description><text>{}</text></description>",
            xml_escape(desc)
        );
    }

    for (ev_idx, ev) in catalog.events.iter().enumerate() {
        let ev_pid = if ev.public_id.is_empty() {
            format!("smi:earthsnail/event/{}", ev_idx + 1)
        } else {
            ev.public_id.clone()
        };
        let _ = writeln!(out, "    <event publicID=\"{}\">", xml_escape(&ev_pid));
        if let Some(ref poid) = ev.preferred_origin_id {
            let _ = writeln!(
                out,
                "      <preferredOriginID>{}</preferredOriginID>",
                xml_escape(poid)
            );
        }
        if let Some(ref pmid) = ev.preferred_magnitude_id {
            let _ = writeln!(
                out,
                "      <preferredMagnitudeID>{}</preferredMagnitudeID>",
                xml_escape(pmid)
            );
        }
        if let Some(ref pfmid) = ev.preferred_focal_mechanism_id {
            let _ = writeln!(
                out,
                "      <preferredFocalMechanismID>{}</preferredFocalMechanismID>",
                xml_escape(pfmid)
            );
        }
        if let Some(ref et) = ev.event_type {
            let _ = writeln!(out, "      <type>{}</type>", xml_escape(et));
        }
        if let Some(ref desc) = ev.description {
            let _ = writeln!(
                out,
                "      <description><text>{}</text></description>",
                xml_escape(desc)
            );
        }

        for (oi, org) in ev.origins.iter().enumerate() {
            let oid = org
                .public_id
                .clone()
                .unwrap_or_else(|| format!("{ev_pid}/origin/{}", oi + 1));
            let _ = writeln!(out, "      <origin publicID=\"{}\">", xml_escape(&oid));
            write_time_qty(
                &mut out,
                "        ",
                "time",
                org.time,
                org.time_uncertainty_s,
            );
            write_real_qty(
                &mut out,
                "        ",
                "latitude",
                org.latitude,
                org.latitude_uncertainty_km.map(|k| k / KM_PER_DEG),
            );
            write_real_qty(
                &mut out,
                "        ",
                "longitude",
                org.longitude,
                org.longitude_uncertainty_km.map(|k| k / KM_PER_DEG),
            );
            write_real_qty(
                &mut out,
                "        ",
                "depth",
                org.depth_m,
                org.depth_uncertainty_m,
            );
            if let Some(ref q) = org.quality {
                let _ = writeln!(out, "        <quality>");
                if let Some(c) = q.associated_phase_count {
                    let _ = writeln!(
                        out,
                        "          <associatedPhaseCount>{c}</associatedPhaseCount>"
                    );
                }
                if let Some(c) = q.used_phase_count {
                    let _ = writeln!(out, "          <usedPhaseCount>{c}</usedPhaseCount>");
                }
                if let Some(se) = q.standard_error_s {
                    let _ = writeln!(out, "          <standardError>{se}</standardError>");
                }
                if let Some(ag) = q.azimuthal_gap_deg {
                    let _ = writeln!(out, "          <azimuthalGap>{ag}</azimuthalGap>");
                }
                if let Some(dmin) = q.minimum_distance_deg {
                    let _ = writeln!(out, "          <minimumDistance>{dmin}</minimumDistance>");
                }
                if let Some(dmax) = q.maximum_distance_deg {
                    let _ = writeln!(out, "          <maximumDistance>{dmax}</maximumDistance>");
                }
                let _ = writeln!(out, "        </quality>");
            }
            for (ai, arr) in org.arrivals.iter().enumerate() {
                let aid = format!("{oid}/arrival/{}", ai + 1);
                let _ = writeln!(out, "        <arrival publicID=\"{}\">", xml_escape(&aid));
                if let Some(ref pid) = arr.pick_id {
                    let _ = writeln!(out, "          <pickID>{}</pickID>", xml_escape(pid));
                }
                let _ = writeln!(out, "          <phase>{}</phase>", xml_escape(&arr.phase));
                if let Some(az) = arr.azimuth_deg {
                    let _ = writeln!(out, "          <azimuth>{az}</azimuth>");
                }
                if let Some(dist) = arr.distance_deg {
                    let _ = writeln!(out, "          <distance>{dist}</distance>");
                }
                if let Some(toa) = arr.takeoff_angle_deg {
                    write_real_qty(&mut out, "          ", "takeoffAngle", toa, None);
                }
                if let Some(res) = arr.time_residual_s {
                    let _ = writeln!(out, "          <timeResidual>{res}</timeResidual>");
                }
                if let Some(w) = arr.time_weight {
                    let _ = writeln!(out, "          <timeWeight>{w}</timeWeight>");
                }
                let _ = writeln!(out, "        </arrival>");
            }
            let _ = writeln!(out, "      </origin>");
        }

        for (mi, mag) in ev.magnitudes.iter().enumerate() {
            let mid = mag
                .public_id
                .clone()
                .unwrap_or_else(|| format!("{ev_pid}/magnitude/{}", mi + 1));
            let _ = writeln!(out, "      <magnitude publicID=\"{}\">", xml_escape(&mid));
            write_real_qty(&mut out, "        ", "mag", mag.mag, mag.uncertainty);
            if !mag.mag_type.is_empty() {
                let _ = writeln!(out, "        <type>{}</type>", xml_escape(&mag.mag_type));
            }
            if let Some(ref oid) = mag.origin_id {
                let _ = writeln!(out, "        <originID>{}</originID>", xml_escape(oid));
            }
            if let Some(sc) = mag.station_count {
                let _ = writeln!(out, "        <stationCount>{sc}</stationCount>");
            }
            let _ = writeln!(out, "      </magnitude>");
        }

        for (pi, pk) in ev.picks.iter().enumerate() {
            let pkid = pk
                .public_id
                .clone()
                .unwrap_or_else(|| format!("{ev_pid}/pick/{}", pi + 1));
            let _ = writeln!(out, "      <pick publicID=\"{}\">", xml_escape(&pkid));
            write_time_qty(&mut out, "        ", "time", pk.time, pk.time_uncertainty_s);
            let _ = writeln!(
                out,
                "        <waveformID networkCode=\"{}\" stationCode=\"{}\" locationCode=\"{}\" channelCode=\"{}\"/>",
                xml_escape(&pk.channel.network),
                xml_escape(&pk.channel.station),
                xml_escape(&pk.channel.location),
                xml_escape(&pk.channel.channel)
            );
            if let Some(ref ph) = pk.phase_hint {
                let _ = writeln!(out, "        <phaseHint>{}</phaseHint>", xml_escape(ph));
            }
            if let Some(pol) = pk.polarity {
                let ps = match pol {
                    PickPolarity::Positive => "positive",
                    PickPolarity::Negative => "negative",
                    PickPolarity::Undecidable => "undecidable",
                };
                let _ = writeln!(out, "        <polarity>{ps}</polarity>");
            }
            if let Some(ons) = pk.onset {
                let os = match ons {
                    PickOnset::Impulsive => "impulsive",
                    PickOnset::Emergent => "emergent",
                    PickOnset::Questionable => "questionable",
                };
                let _ = writeln!(out, "        <onset>{os}</onset>");
            }
            if let Some(em) = pk.evaluation_mode {
                let es = match em {
                    EvaluationMode::Manual => "manual",
                    EvaluationMode::Automatic => "automatic",
                };
                let _ = writeln!(out, "        <evaluationMode>{es}</evaluationMode>");
            }
            let _ = writeln!(out, "      </pick>");
        }

        for (fmi, fm) in ev.focal_mechanisms.iter().enumerate() {
            let fmid = fm
                .public_id
                .clone()
                .unwrap_or_else(|| format!("{ev_pid}/focalMechanism/{}", fmi + 1));
            let _ = writeln!(
                out,
                "      <focalMechanism publicID=\"{}\">",
                xml_escape(&fmid)
            );
            if fm.nodal_plane_1.is_some() || fm.nodal_plane_2.is_some() {
                let _ = writeln!(out, "        <nodalPlanes>");
                if let Some(np1) = fm.nodal_plane_1 {
                    write_nodal_plane(&mut out, "nodalPlane1", np1);
                }
                if let Some(np2) = fm.nodal_plane_2 {
                    write_nodal_plane(&mut out, "nodalPlane2", np2);
                }
                let _ = writeln!(out, "        </nodalPlanes>");
            }
            if let Some(pa) = fm.principal_axes {
                let _ = writeln!(out, "        <principalAxes>");
                write_axis(&mut out, "tAxis", pa.t_axis);
                write_axis(&mut out, "pAxis", pa.p_axis);
                if let Some(na) = pa.n_axis {
                    write_axis(&mut out, "nAxis", na);
                }
                let _ = writeln!(out, "        </principalAxes>");
            }
            if let Some(mt) = fm.moment_tensor {
                let mtid = format!("{fmid}/momentTensor");
                let _ = writeln!(
                    out,
                    "        <momentTensor publicID=\"{}\">",
                    xml_escape(&mtid)
                );
                write_real_qty(
                    &mut out,
                    "          ",
                    "scalarMoment",
                    mt.scalar_moment_nm(),
                    None,
                );
                let _ = writeln!(out, "          <tensor>");
                write_real_qty(&mut out, "            ", "Mrr", mt.m_rr, None);
                write_real_qty(&mut out, "            ", "Mtt", mt.m_tt, None);
                write_real_qty(&mut out, "            ", "Mpp", mt.m_pp, None);
                write_real_qty(&mut out, "            ", "Mrt", mt.m_rt, None);
                write_real_qty(&mut out, "            ", "Mrp", mt.m_rp, None);
                write_real_qty(&mut out, "            ", "Mtp", mt.m_tp, None);
                let _ = writeln!(out, "          </tensor>");
                let _ = writeln!(out, "        </momentTensor>");
            }
            let _ = writeln!(out, "      </focalMechanism>");
        }

        let _ = writeln!(out, "    </event>");
    }

    let _ = writeln!(out, "  </eventParameters>");
    let _ = writeln!(out, "</q:quakeml>");
    Ok(out)
}

fn write_real_qty(out: &mut String, indent: &str, tag: &str, val: f64, unc: Option<f64>) {
    if let Some(u) = unc {
        let _ = writeln!(
            out,
            "{indent}<{tag}><value>{val}</value><uncertainty>{u}</uncertainty></{tag}>"
        );
    } else {
        let _ = writeln!(out, "{indent}<{tag}><value>{val}</value></{tag}>");
    }
}

fn write_time_qty(out: &mut String, indent: &str, tag: &str, ts: UtcTimestamp, unc: Option<f64>) {
    let iso = ts.to_iso8601();
    if let Some(u) = unc {
        let _ = writeln!(
            out,
            "{indent}<{tag}><value>{iso}</value><uncertainty>{u}</uncertainty></{tag}>"
        );
    } else {
        let _ = writeln!(out, "{indent}<{tag}><value>{iso}</value></{tag}>");
    }
}

fn write_nodal_plane(out: &mut String, tag: &str, np: NodalPlane) {
    let _ = writeln!(out, "          <{tag}>");
    write_real_qty(out, "            ", "strike", np.strike_deg, None);
    write_real_qty(out, "            ", "dip", np.dip_deg, None);
    write_real_qty(out, "            ", "rake", np.rake_deg, None);
    let _ = writeln!(out, "          </{tag}>");
}

fn write_axis(out: &mut String, tag: &str, ax: PrincipalAxis) {
    let _ = writeln!(out, "          <{tag}>");
    write_real_qty(out, "            ", "azimuth", ax.azimuth_deg, None);
    write_real_qty(out, "            ", "plunge", ax.plunge_deg, None);
    write_real_qty(out, "            ", "length", ax.length, None);
    let _ = writeln!(out, "          </{tag}>");
}

/// Writes an `EventCatalog` as FDSN `QuakeML` (`1.2`) to disk (`std` feature).
#[cfg(feature = "std")]
pub fn write_quakeml_file(path: impl AsRef<std::path::Path>, catalog: &EventCatalog) -> Result<()> {
    let xml = write_quakeml_str(catalog)?;
    std::fs::write(path, xml.as_bytes())?;
    Ok(())
}
