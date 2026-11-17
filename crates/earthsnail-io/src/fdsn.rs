//! Pure-Rust FDSN Web Services (`FDSNWS` `1.x`) client and query builders (`dataselect`,
//! `station`, `event`) with data-center presets (`EarthScope`/`IRIS`, `USGS`, `ORFEUS`, `GEOFON`,
//! `INGV`, `ETHZ`), GET URL & bulk POST formatting, pluggable `HttpTransport` trait, and a
//! zero-dependency HTTP/1.1 `TcpHttpTransport` over `std::net::TcpStream` (`Content-Length` +
//! `Transfer-Encoding: chunked` + clear `204 No Content` & `301/302` HTTPS redirect reporting).

use crate::quakeml::read_quakeml_bytes;
use crate::sniff::read_waveforms_bytes;
use crate::stationxml::read_stationxml_bytes;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use core::time::Duration;
use earthsnail_core::{
    ChannelId, EarthSnailError, EventCatalog, Result, StationInventory, Stream, UtcTimestamp,
};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};

/// Maximum HTTP response header block size (`64 KiB`).
pub const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;

/// Default maximum HTTP response body size (`256 MiB`).
pub const DEFAULT_MAX_HTTP_BODY_BYTES: usize = 256 * 1024 * 1024;

/// Well-known FDSN Web Services (`FDSNWS 1.x`) data center endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FdsnProvider {
    /// EarthScope Consortium (formerly IRIS DMC): `https://service.earthscope.org`.
    EarthScope,
    /// Legacy plain-HTTP IRIS alias (`http://service.iris.edu`, typically redirects to HTTPS).
    IrisHttp,
    /// U.S. Geological Survey NEIC: `https://earthquake.usgs.gov`.
    Usgs,
    /// European ORFEUS / EIDA Data Center: `https://www.orfeus-eu.org`.
    Orfeus,
    /// GFZ Potsdam GEOFON: `https://geofon.gfz-potsdam.de`.
    Geofon,
    /// Istituto Nazionale di Geofisica e Vulcanologia: `https://webservices.ingv.it`.
    Ingv,
    /// Swiss Seismological Service (ETH Zurich): `http://arclink.ethz.ch`.
    Ethz,
    /// Custom base URL (e.g. `"http://localhost:8080"` or `"https://fdsnws.raspberryshakedata.com"`).
    Custom(String),
}

impl FdsnProvider {
    /// Resolves a provider name (case-insensitive, ObsPy `Client("EARTHSCOPE")` compatible).
    pub fn from_name(name: &str) -> Self {
        let s = name.trim();
        if s.starts_with("http://") || s.starts_with("https://") {
            return Self::Custom(String::from(s.trim_end_matches('/')));
        }
        match s.to_ascii_uppercase().as_str() {
            "EARTHSCOPE" | "IRIS" => Self::EarthScope,
            "IRIS_HTTP" => Self::IrisHttp,
            "USGS" => Self::Usgs,
            "ORFEUS" => Self::Orfeus,
            "GEOFON" | "GFZ" => Self::Geofon,
            "INGV" => Self::Ingv,
            "ETH" | "ETHZ" => Self::Ethz,
            _ => Self::Custom(format!("https://{s}")),
        }
    }

    /// Canonical base URL without trailing slash.
    pub fn base_url(&self) -> &str {
        match self {
            Self::EarthScope => "https://service.earthscope.org",
            Self::IrisHttp => "http://service.iris.edu",
            Self::Usgs => "https://earthquake.usgs.gov",
            Self::Orfeus => "https://www.orfeus-eu.org",
            Self::Geofon => "https://geofon.gfz-potsdam.de",
            Self::Ingv => "https://webservices.ingv.it",
            Self::Ethz => "http://arclink.ethz.ch",
            Self::Custom(u) => u.trim_end_matches('/'),
        }
    }
}

/// A single channel-time window row in a bulk FDSNWS `POST` request (`NET STA LOC CHA START END`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BulkChannelSelection {
    pub network: String,
    pub station: String,
    pub location: String,
    pub channel: String,
    pub starttime: UtcTimestamp,
    pub endtime: UtcTimestamp,
}

impl BulkChannelSelection {
    /// Creates a bulk selection row from a `ChannelId` (or wildcard strings) and `[starttime, endtime]`.
    pub fn new(
        network: impl Into<String>,
        station: impl Into<String>,
        location: impl Into<String>,
        channel: impl Into<String>,
        starttime: UtcTimestamp,
        endtime: UtcTimestamp,
    ) -> Self {
        Self {
            network: network.into(),
            station: station.into(),
            location: location.into(),
            channel: channel.into(),
            starttime,
            endtime,
        }
    }

    /// Creates a bulk selection row from an existing `ChannelId`.
    pub fn from_channel_id(id: &ChannelId, starttime: UtcTimestamp, endtime: UtcTimestamp) -> Self {
        Self::new(
            &id.network,
            &id.station,
            if id.location.is_empty() {
                "--"
            } else {
                &id.location
            },
            &id.channel,
            starttime,
            endtime,
        )
    }

    fn write_post_line(&self, out: &mut String) {
        let net = if self.network.trim().is_empty() {
            "*"
        } else {
            self.network.trim()
        };
        let sta = if self.station.trim().is_empty() {
            "*"
        } else {
            self.station.trim()
        };
        let loc = if self.location.trim().is_empty() {
            "--"
        } else {
            self.location.trim()
        };
        let cha = if self.channel.trim().is_empty() {
            "*"
        } else {
            self.channel.trim()
        };
        let _ = writeln!(
            out,
            "{net} {sta} {loc} {cha} {} {}",
            self.starttime.to_fdsn_query(),
            self.endtime.to_fdsn_query()
        );
    }
}

/// Query builder for `fdsnws/dataselect/1/query`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WaveformQuery {
    pub network: Option<String>,
    pub station: Option<String>,
    pub location: Option<String>,
    pub channel: Option<String>,
    pub starttime: Option<UtcTimestamp>,
    pub endtime: Option<UtcTimestamp>,
    pub quality: Option<char>,
    pub minimumlength_s: Option<f64>,
    pub longestonly: Option<bool>,
    pub format: Option<String>,
    pub bulk: Vec<BulkChannelSelection>,
}

impl WaveformQuery {
    /// Creates a new `WaveformQuery` for `(network, station, location, channel, starttime, endtime)`.
    pub fn new(
        network: impl Into<String>,
        station: impl Into<String>,
        location: impl Into<String>,
        channel: impl Into<String>,
        starttime: UtcTimestamp,
        endtime: UtcTimestamp,
    ) -> Self {
        Self {
            network: Some(network.into()),
            station: Some(station.into()),
            location: Some(location.into()),
            channel: Some(channel.into()),
            starttime: Some(starttime),
            endtime: Some(endtime),
            ..Self::default()
        }
    }

    /// Creates an empty bulk `WaveformQuery`.
    pub fn bulk_new() -> Self {
        Self::default()
    }

    /// Sets SEED quality filter (`'D'`, `'R'`, `'Q'`, `'M'`, `'B'`).
    pub fn with_quality(mut self, quality: char) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Sets minimum contiguous segment length in seconds.
    pub fn with_minimum_length(mut self, secs: f64) -> Self {
        self.minimumlength_s = Some(secs);
        self
    }

    /// Restricts response to the single longest segment per channel.
    pub fn with_longest_only(mut self, longest_only: bool) -> Self {
        self.longestonly = Some(longest_only);
        self
    }

    /// Sets output format (`"miniseed"`, `"miniseed3"`, `"geocsv"`).
    pub fn with_format(mut self, fmt: impl Into<String>) -> Self {
        self.format = Some(fmt.into());
        self
    }

    /// Appends a channel-time selection row for bulk `POST` queries.
    pub fn add_bulk(mut self, sel: BulkChannelSelection) -> Self {
        self.bulk.push(sel);
        self
    }

    /// Formats the complete `GET` URL against `base_url`.
    pub fn build_url(&self, base_url: &str) -> String {
        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(ref net) = self.network {
            params.push(("network", net.clone()));
        }
        if let Some(ref sta) = self.station {
            params.push(("station", sta.clone()));
        }
        if let Some(ref loc) = self.location {
            let loc_norm = if loc.trim().is_empty() {
                "--"
            } else {
                loc.trim()
            };
            params.push(("location", String::from(loc_norm)));
        }
        if let Some(ref cha) = self.channel {
            params.push(("channel", cha.clone()));
        }
        if let Some(st) = self.starttime {
            params.push(("starttime", st.to_fdsn_query()));
        }
        if let Some(et) = self.endtime {
            params.push(("endtime", et.to_fdsn_query()));
        }
        if let Some(q) = self.quality {
            params.push(("quality", String::from(q)));
        }
        if let Some(ml) = self.minimumlength_s {
            params.push(("minimumlength", format!("{ml}")));
        }
        if let Some(lo) = self.longestonly {
            params.push((
                "longestonly",
                String::from(if lo { "true" } else { "false" }),
            ));
        }
        if let Some(ref fmt) = self.format {
            params.push(("format", fmt.clone()));
        }
        params.push(("nodata", String::from("204")));
        format_fdsn_url(base_url, "dataselect", &params)
    }

    /// Formats the `POST` body for `fdsnws/dataselect/1/query`.
    pub fn build_post_body(&self) -> Result<String> {
        let mut out = String::new();
        if let Some(q) = self.quality {
            let _ = writeln!(out, "quality={q}");
        }
        if let Some(ml) = self.minimumlength_s {
            let _ = writeln!(out, "minimumlength={ml}");
        }
        if let Some(lo) = self.longestonly {
            let _ = writeln!(out, "longestonly={}", if lo { "true" } else { "false" });
        }
        if let Some(ref fmt) = self.format {
            let _ = writeln!(out, "format={fmt}");
        }

        if !self.bulk.is_empty() {
            for row in &self.bulk {
                row.write_post_line(&mut out);
            }
        } else if let (Some(st), Some(et)) = (self.starttime, self.endtime) {
            let single = BulkChannelSelection::new(
                self.network.clone().unwrap_or_else(|| String::from("*")),
                self.station.clone().unwrap_or_else(|| String::from("*")),
                self.location.clone().unwrap_or_else(|| String::from("--")),
                self.channel.clone().unwrap_or_else(|| String::from("*")),
                st,
                et,
            );
            single.write_post_line(&mut out);
        } else {
            return Err(EarthSnailError::InvalidParameter(
                "WaveformQuery POST requires either `bulk` entries or `(starttime, endtime)`"
                    .into(),
            ));
        }
        Ok(out)
    }
}

/// Detail level requested from `fdsnws/station/1/query`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StationLevel {
    Network,
    #[default]
    Station,
    Channel,
    Response,
}

impl StationLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::Station => "station",
            Self::Channel => "channel",
            Self::Response => "response",
        }
    }
}

/// Query builder for `fdsnws/station/1/query`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StationQuery {
    pub network: Option<String>,
    pub station: Option<String>,
    pub location: Option<String>,
    pub channel: Option<String>,
    pub starttime: Option<UtcTimestamp>,
    pub endtime: Option<UtcTimestamp>,
    pub level: StationLevel,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub minradius_deg: Option<f64>,
    pub maxradius_deg: Option<f64>,
    pub minlatitude: Option<f64>,
    pub maxlatitude: Option<f64>,
    pub minlongitude: Option<f64>,
    pub maxlongitude: Option<f64>,
    pub includerestricted: Option<bool>,
    pub format: Option<String>,
    pub bulk: Vec<BulkChannelSelection>,
}

impl StationQuery {
    /// Creates a new `StationQuery` defaulting to `StationLevel::Response`.
    pub fn new() -> Self {
        Self {
            level: StationLevel::Response,
            ..Self::default()
        }
    }

    pub fn with_network(mut self, network: impl Into<String>) -> Self {
        self.network = Some(network.into());
        self
    }

    pub fn with_station(mut self, station: impl Into<String>) -> Self {
        self.station = Some(station.into());
        self
    }

    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    pub fn with_time_window(mut self, start: UtcTimestamp, end: UtcTimestamp) -> Self {
        self.starttime = Some(start);
        self.endtime = Some(end);
        self
    }

    pub fn with_level(mut self, level: StationLevel) -> Self {
        self.level = level;
        self
    }

    pub fn with_radial(
        mut self,
        lat_deg: f64,
        lon_deg: f64,
        min_radius_deg: f64,
        max_radius_deg: f64,
    ) -> Self {
        self.latitude = Some(lat_deg);
        self.longitude = Some(lon_deg);
        self.minradius_deg = Some(min_radius_deg);
        self.maxradius_deg = Some(max_radius_deg);
        self
    }

    pub fn with_bbox(mut self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Self {
        self.minlatitude = Some(min_lat);
        self.maxlatitude = Some(max_lat);
        self.minlongitude = Some(min_lon);
        self.maxlongitude = Some(max_lon);
        self
    }

    pub fn add_bulk(mut self, sel: BulkChannelSelection) -> Self {
        self.bulk.push(sel);
        self
    }

    /// Formats the complete `GET` URL against `base_url`.
    pub fn build_url(&self, base_url: &str) -> String {
        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(ref net) = self.network {
            params.push(("network", net.clone()));
        }
        if let Some(ref sta) = self.station {
            params.push(("station", sta.clone()));
        }
        if let Some(ref loc) = self.location {
            let loc_norm = if loc.trim().is_empty() {
                "--"
            } else {
                loc.trim()
            };
            params.push(("location", String::from(loc_norm)));
        }
        if let Some(ref cha) = self.channel {
            params.push(("channel", cha.clone()));
        }
        if let Some(st) = self.starttime {
            params.push(("starttime", st.to_fdsn_query()));
        }
        if let Some(et) = self.endtime {
            params.push(("endtime", et.to_fdsn_query()));
        }
        params.push(("level", String::from(self.level.as_str())));
        if let Some(v) = self.latitude {
            params.push(("latitude", format!("{v}")));
        }
        if let Some(v) = self.longitude {
            params.push(("longitude", format!("{v}")));
        }
        if let Some(v) = self.minradius_deg {
            params.push(("minradius", format!("{v}")));
        }
        if let Some(v) = self.maxradius_deg {
            params.push(("maxradius", format!("{v}")));
        }
        if let Some(v) = self.minlatitude {
            params.push(("minlatitude", format!("{v}")));
        }
        if let Some(v) = self.maxlatitude {
            params.push(("maxlatitude", format!("{v}")));
        }
        if let Some(v) = self.minlongitude {
            params.push(("minlongitude", format!("{v}")));
        }
        if let Some(v) = self.maxlongitude {
            params.push(("maxlongitude", format!("{v}")));
        }
        if let Some(ir) = self.includerestricted {
            params.push((
                "includerestricted",
                String::from(if ir { "true" } else { "false" }),
            ));
        }
        if let Some(ref fmt) = self.format {
            params.push(("format", fmt.clone()));
        }
        params.push(("nodata", String::from("204")));
        format_fdsn_url(base_url, "station", &params)
    }

    /// Formats the bulk `POST` body for `fdsnws/station/1/query`.
    pub fn build_post_body(&self) -> Result<String> {
        let mut out = String::new();
        let _ = writeln!(out, "level={}", self.level.as_str());
        if let Some(ref fmt) = self.format {
            let _ = writeln!(out, "format={fmt}");
        }
        if !self.bulk.is_empty() {
            for row in &self.bulk {
                row.write_post_line(&mut out);
            }
        } else if let (Some(st), Some(et)) = (self.starttime, self.endtime) {
            let single = BulkChannelSelection::new(
                self.network.clone().unwrap_or_else(|| String::from("*")),
                self.station.clone().unwrap_or_else(|| String::from("*")),
                self.location.clone().unwrap_or_else(|| String::from("--")),
                self.channel.clone().unwrap_or_else(|| String::from("*")),
                st,
                et,
            );
            single.write_post_line(&mut out);
        } else {
            return Err(EarthSnailError::InvalidParameter(
                "StationQuery POST requires `bulk` entries or `(starttime, endtime)`".into(),
            ));
        }
        Ok(out)
    }
}

/// Query builder for `fdsnws/event/1/query`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EventQuery {
    pub starttime: Option<UtcTimestamp>,
    pub endtime: Option<UtcTimestamp>,
    pub minmagnitude: Option<f64>,
    pub maxmagnitude: Option<f64>,
    pub magnitudetype: Option<String>,
    pub mindepth_km: Option<f64>,
    pub maxdepth_km: Option<f64>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub minradius_deg: Option<f64>,
    pub maxradius_deg: Option<f64>,
    pub minlatitude: Option<f64>,
    pub maxlatitude: Option<f64>,
    pub minlongitude: Option<f64>,
    pub maxlongitude: Option<f64>,
    pub eventid: Option<String>,
    pub includeallorigins: Option<bool>,
    pub includeallmagnitudes: Option<bool>,
    pub includearrivals: Option<bool>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub orderby: Option<String>,
    pub catalog: Option<String>,
    pub contributor: Option<String>,
    pub format: Option<String>,
}

impl EventQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_time_window(mut self, start: UtcTimestamp, end: UtcTimestamp) -> Self {
        self.starttime = Some(start);
        self.endtime = Some(end);
        self
    }

    pub fn with_min_magnitude(mut self, min_mag: f64) -> Self {
        self.minmagnitude = Some(min_mag);
        self
    }

    pub fn with_max_magnitude(mut self, max_mag: f64) -> Self {
        self.maxmagnitude = Some(max_mag);
        self
    }

    pub fn with_depth_range_km(mut self, min_km: f64, max_km: f64) -> Self {
        self.mindepth_km = Some(min_km);
        self.maxdepth_km = Some(max_km);
        self
    }

    pub fn with_radial(
        mut self,
        lat_deg: f64,
        lon_deg: f64,
        min_radius_deg: f64,
        max_radius_deg: f64,
    ) -> Self {
        self.latitude = Some(lat_deg);
        self.longitude = Some(lon_deg);
        self.minradius_deg = Some(min_radius_deg);
        self.maxradius_deg = Some(max_radius_deg);
        self
    }

    pub fn with_bbox(mut self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Self {
        self.minlatitude = Some(min_lat);
        self.maxlatitude = Some(max_lat);
        self.minlongitude = Some(min_lon);
        self.maxlongitude = Some(max_lon);
        self
    }

    pub fn with_event_id(mut self, event_id: impl Into<String>) -> Self {
        self.eventid = Some(event_id.into());
        self
    }

    pub fn with_include_arrivals(mut self, include: bool) -> Self {
        self.includearrivals = Some(include);
        self
    }

    pub fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Formats the complete `GET` URL against `base_url`.
    pub fn build_url(&self, base_url: &str) -> String {
        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(st) = self.starttime {
            params.push(("starttime", st.to_fdsn_query()));
        }
        if let Some(et) = self.endtime {
            params.push(("endtime", et.to_fdsn_query()));
        }
        if let Some(v) = self.minmagnitude {
            params.push(("minmagnitude", format!("{v}")));
        }
        if let Some(v) = self.maxmagnitude {
            params.push(("maxmagnitude", format!("{v}")));
        }
        if let Some(ref mt) = self.magnitudetype {
            params.push(("magnitudetype", mt.clone()));
        }
        if let Some(v) = self.mindepth_km {
            params.push(("mindepth", format!("{v}")));
        }
        if let Some(v) = self.maxdepth_km {
            params.push(("maxdepth", format!("{v}")));
        }
        if let Some(v) = self.latitude {
            params.push(("latitude", format!("{v}")));
        }
        if let Some(v) = self.longitude {
            params.push(("longitude", format!("{v}")));
        }
        if let Some(v) = self.minradius_deg {
            params.push(("minradius", format!("{v}")));
        }
        if let Some(v) = self.maxradius_deg {
            params.push(("maxradius", format!("{v}")));
        }
        if let Some(v) = self.minlatitude {
            params.push(("minlatitude", format!("{v}")));
        }
        if let Some(v) = self.maxlatitude {
            params.push(("maxlatitude", format!("{v}")));
        }
        if let Some(v) = self.minlongitude {
            params.push(("minlongitude", format!("{v}")));
        }
        if let Some(v) = self.maxlongitude {
            params.push(("maxlongitude", format!("{v}")));
        }
        if let Some(ref eid) = self.eventid {
            params.push(("eventid", eid.clone()));
        }
        if let Some(b) = self.includeallorigins {
            params.push((
                "includeallorigins",
                String::from(if b { "true" } else { "false" }),
            ));
        }
        if let Some(b) = self.includeallmagnitudes {
            params.push((
                "includeallmagnitudes",
                String::from(if b { "true" } else { "false" }),
            ));
        }
        if let Some(b) = self.includearrivals {
            params.push((
                "includearrivals",
                String::from(if b { "true" } else { "false" }),
            ));
        }
        if let Some(lim) = self.limit {
            params.push(("limit", format!("{lim}")));
        }
        if let Some(off) = self.offset {
            params.push(("offset", format!("{off}")));
        }
        if let Some(ref ob) = self.orderby {
            params.push(("orderby", ob.clone()));
        }
        if let Some(ref cat) = self.catalog {
            params.push(("catalog", cat.clone()));
        }
        if let Some(ref c) = self.contributor {
            params.push(("contributor", c.clone()));
        }
        if let Some(ref fmt) = self.format {
            params.push(("format", fmt.clone()));
        }
        params.push(("nodata", String::from("204")));
        format_fdsn_url(base_url, "event", &params)
    }
}

fn format_fdsn_url(base_url: &str, service: &str, params: &[(&str, String)]) -> String {
    let base = base_url.trim_end_matches('/');
    let mut url = format!("{base}/fdsnws/{service}/1/query");
    if !params.is_empty() {
        url.push('?');
        for (i, (k, v)) in params.iter().enumerate() {
            if i > 0 {
                url.push('&');
            }
            url.push_str(k);
            url.push('=');
            percent_encode_query_val(&mut url, v);
        }
    }
    url
}

/// RFC-3986 percent-encoder preserving unreserved ASCII (`A-Z`, `a-z`, `0-9`, `-`, `_`, `.`, `~`, `,`, `*`, `?`, `:`).
/// Note: `,`, `*`, `?`, `:` are preserved unescaped or safely encoded; encoding `?` as `%3F` and
/// `:` as `%3A` ensures strict HTTP proxy & FDSNWS query compatibility!
pub fn percent_encode_query_val(out: &mut String, val: &str) {
    for &b in val.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b',' | b'*') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
}

/// Raw HTTP response returned by an `HttpTransport`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// HTTP status code (e.g. `200`, `204`, `301`, `400`).
    pub status_code: u16,
    /// Lowercased header `(name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Decoded HTTP entity body (already de-chunked if `Transfer-Encoding: chunked`).
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Pluggable HTTP transport abstraction allowing `FdsnClient` to run over built-in plain-TCP
/// HTTP/1.1 (`TcpHttpTransport`), external TLS clients, or deterministic offline mock transports.
pub trait HttpTransport {
    /// Performs an HTTP `GET` request to `url`.
    fn get(&self, url: &str) -> Result<HttpResponse>;

    /// Performs an HTTP `POST` request to `url` with `content_type` and `body`.
    fn post(&self, url: &str, content_type: &str, body: &[u8]) -> Result<HttpResponse>;
}

/// Zero-dependency HTTP/1.1 client over `std::net::TcpStream` supporting both `Content-Length`
/// and `Transfer-Encoding: chunked`, automatic plain-HTTP redirect following, and clear diagnostic
/// errors when pointed at or redirected to an `https://` TLS-only endpoint.
#[derive(Debug, Clone)]
pub struct TcpHttpTransport {
    /// Socket connect & read/write timeout (`15s` default).
    pub timeout: Duration,
    /// Maximum response payload size (`256 MiB` default).
    pub max_body_bytes: usize,
    /// Maximum HTTP `301/302/307/308` redirects followed (`5` default).
    pub max_redirects: usize,
}

impl Default for TcpHttpTransport {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(15),
            max_body_bytes: DEFAULT_MAX_HTTP_BODY_BYTES,
            max_redirects: 5,
        }
    }
}

impl TcpHttpTransport {
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Self::default()
        }
    }

    fn execute(
        &self,
        method: &str,
        url: &str,
        content_type: Option<&str>,
        body: Option<&[u8]>,
    ) -> Result<HttpResponse> {
        let mut current_url = String::from(url);
        let mut cur_method = method;
        let mut cur_ct = content_type;
        let mut cur_body = body;

        for hop in 0..=self.max_redirects {
            let parsed = parse_http_url(&current_url)?;
            let addr_str = format!("{}:{}", parsed.host, parsed.port);
            let mut addrs = addr_str.to_socket_addrs().map_err(|e| {
                EarthSnailError::Network(format!("DNS resolution failed for '{addr_str}': {e}"))
            })?;
            let sock_addr = addrs.next().ok_or_else(|| {
                EarthSnailError::Network(format!("no socket addresses resolved for '{addr_str}'"))
            })?;

            let mut stream = TcpStream::connect_timeout(&sock_addr, self.timeout).map_err(|e| {
                EarthSnailError::Network(format!("TCP connect to {sock_addr} failed: {e}"))
            })?;
            let _ = stream.set_read_timeout(Some(self.timeout));
            let _ = stream.set_write_timeout(Some(self.timeout));

            let host_hdr = if parsed.port == 80 {
                parsed.host.clone()
            } else {
                format!("{}:{}", parsed.host, parsed.port)
            };

            let mut req = String::with_capacity(256 + parsed.path_and_query.len());
            let _ = write!(req, "{cur_method} {} HTTP/1.1\r\n", parsed.path_and_query);
            let _ = write!(req, "Host: {host_hdr}\r\n");
            req.push_str("User-Agent: EarthSnail/0.1 (Pure-Rust)\r\n");
            req.push_str("Accept: */*\r\n");
            req.push_str("Connection: close\r\n");
            if let Some(ct) = cur_ct {
                let _ = write!(req, "Content-Type: {ct}\r\n");
            }
            if let Some(b) = cur_body {
                let _ = write!(req, "Content-Length: {}\r\n", b.len());
            }
            req.push_str("\r\n");

            stream
                .write_all(req.as_bytes())
                .map_err(|e| EarthSnailError::Network(format!("HTTP write header failed: {e}")))?;
            if let Some(b) = cur_body {
                stream.write_all(b).map_err(|e| {
                    EarthSnailError::Network(format!("HTTP write body failed: {e}"))
                })?;
            }
            stream
                .flush()
                .map_err(|e| EarthSnailError::Network(format!("HTTP socket flush failed: {e}")))?;

            let resp = read_http_response(&mut stream, self.max_body_bytes)?;

            if matches!(resp.status_code, 301 | 302 | 303 | 307 | 308) {
                let Some(loc) = resp.header("location") else {
                    return Ok(resp);
                };
                if hop == self.max_redirects {
                    return Err(EarthSnailError::Network(format!(
                        "exceeded {} HTTP redirects (last Location: {loc})",
                        self.max_redirects
                    )));
                }
                if loc.starts_with("https://") {
                    return Err(EarthSnailError::Network(format!(
                        "HTTP {} redirect points to TLS-only HTTPS endpoint '{loc}'; supply a TLS-capable HttpTransport implementation",
                        resp.status_code
                    )));
                }
                current_url = if loc.starts_with("http://") {
                    String::from(loc)
                } else if loc.starts_with('/') {
                    format!("http://{host_hdr}{loc}")
                } else {
                    format!("http://{host_hdr}/{loc}")
                };
                if matches!(resp.status_code, 301..=303) {
                    cur_method = "GET";
                    cur_ct = None;
                    cur_body = None;
                }
                continue;
            }

            return Ok(resp);
        }

        Err(EarthSnailError::Network("redirect loop".into()))
    }
}

impl HttpTransport for TcpHttpTransport {
    fn get(&self, url: &str) -> Result<HttpResponse> {
        self.execute("GET", url, None, None)
    }

    fn post(&self, url: &str, content_type: &str, body: &[u8]) -> Result<HttpResponse> {
        self.execute("POST", url, Some(content_type), Some(body))
    }
}

struct ParsedHttpUrl {
    host: String,
    port: u16,
    path_and_query: String,
}

fn parse_http_url(url: &str) -> Result<ParsedHttpUrl> {
    let trimmed = url.trim();
    if trimmed.starts_with("https://") {
        return Err(EarthSnailError::Unsupported(format!(
            "HTTPS URL '{trimmed}' requires a TLS-capable HttpTransport; built-in zero-dep TcpHttpTransport supports plain HTTP ('http://')"
        )));
    }
    let after_scheme = trimmed.strip_prefix("http://").ok_or_else(|| {
        EarthSnailError::InvalidParameter(format!(
            "unsupported URL scheme in '{trimmed}' (expected 'http://')"
        ))
    })?;

    let (authority, path_part) = match after_scheme.find('/') {
        Some(idx) => (&after_scheme[..idx], &after_scheme[idx..]),
        None => (after_scheme, "/"),
    };
    if authority.is_empty() {
        return Err(EarthSnailError::InvalidParameter(format!(
            "empty authority in URL '{trimmed}'"
        )));
    }

    let (host, port) = if let Some(rest_v6) = authority.strip_prefix('[') {
        let end_b = rest_v6.find(']').ok_or_else(|| {
            EarthSnailError::InvalidParameter(format!("malformed IPv6 host in '{trimmed}'"))
        })?;
        let h = format!("[{}]", &rest_v6[..end_b]);
        let after_b = &rest_v6[end_b + 1..];
        let p = if let Some(p_str) = after_b.strip_prefix(':') {
            p_str.parse::<u16>().map_err(|_| {
                EarthSnailError::InvalidParameter(format!("invalid port in '{trimmed}'"))
            })?
        } else {
            80
        };
        (h, p)
    } else if let Some((h, p_str)) = authority.rsplit_once(':') {
        let p = p_str.parse::<u16>().map_err(|_| {
            EarthSnailError::InvalidParameter(format!("invalid port '{p_str}' in '{trimmed}'"))
        })?;
        (String::from(h), p)
    } else {
        (String::from(authority), 80_u16)
    };

    Ok(ParsedHttpUrl {
        host,
        port,
        path_and_query: String::from(path_part),
    })
}

/// Parses an HTTP/1.1 response (status line, headers, and `Content-Length` / `chunked` / EOF body)
/// from any `Read` stream.
pub fn read_http_response<R: Read>(reader: &mut R, max_body_bytes: usize) -> Result<HttpResponse> {
    let mut raw = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 2048];
    let header_end_pos: usize;

    loop {
        let n = reader.read(&mut chunk).map_err(|e| {
            EarthSnailError::Network(format!(
                "HTTP socket read failed while reading headers: {e}"
            ))
        })?;
        if n == 0 {
            return Err(EarthSnailError::Network(
                "connection closed before HTTP response headers completed".into(),
            ));
        }
        raw.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_double_crlf(&raw) {
            header_end_pos = pos;
            break;
        }
        if raw.len() > MAX_HTTP_HEADER_BYTES {
            return Err(EarthSnailError::CapacityExceeded(
                "HTTP response headers exceeded 64 KiB limit".into(),
            ));
        }
    }

    let hdr_text = core::str::from_utf8(&raw[..header_end_pos])
        .map_err(|_| EarthSnailError::Network("non-ASCII HTTP response headers".into()))?;
    let mut lines = hdr_text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| EarthSnailError::Network("missing HTTP status line".into()))?;
    let mut status_parts = status_line.split_whitespace();
    let _http_ver = status_parts.next().ok_or_else(|| {
        EarthSnailError::Network(format!("malformed HTTP status line '{status_line}'"))
    })?;
    let code_str = status_parts.next().ok_or_else(|| {
        EarthSnailError::Network(format!("missing status code in '{status_line}'"))
    })?;
    let status_code: u16 = code_str
        .parse()
        .map_err(|_| EarthSnailError::Network(format!("invalid HTTP status code '{code_str}'")))?;

    let mut headers = Vec::new();
    let mut content_length: Option<usize> = None;
    let mut is_chunked = false;

    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key_lc = k.trim().to_ascii_lowercase();
            let val_tr = String::from(v.trim());
            if key_lc == "content-length" {
                let cl: usize = val_tr.parse().map_err(|_| {
                    EarthSnailError::Network(format!("invalid Content-Length '{val_tr}'"))
                })?;
                content_length = Some(cl);
            } else if key_lc == "transfer-encoding"
                && val_tr.to_ascii_lowercase().contains("chunked")
            {
                is_chunked = true;
            }
            headers.push((key_lc, val_tr));
        }
    }

    let body_start = header_end_pos + 4;
    let mut buffered_body = raw[body_start..].to_vec();

    if status_code == 204 || status_code == 304 {
        return Ok(HttpResponse {
            status_code,
            headers,
            body: Vec::new(),
        });
    }

    let body = if is_chunked {
        decode_chunked_body(reader, &mut buffered_body, max_body_bytes)?
    } else if let Some(cl) = content_length {
        if cl > max_body_bytes {
            return Err(EarthSnailError::CapacityExceeded(format!(
                "HTTP Content-Length {cl} exceeds limit {max_body_bytes}"
            )));
        }
        while buffered_body.len() < cl {
            let need = (cl - buffered_body.len()).min(chunk.len());
            let n = reader
                .read(&mut chunk[..need])
                .map_err(|e| EarthSnailError::Network(format!("HTTP body read error: {e}")))?;
            if n == 0 {
                return Err(EarthSnailError::Network(format!(
                    "truncated HTTP body: expected {cl} bytes, got {}",
                    buffered_body.len()
                )));
            }
            buffered_body.extend_from_slice(&chunk[..n]);
        }
        buffered_body.truncate(cl);
        buffered_body
    } else {
        loop {
            let n = reader.read(&mut chunk).map_err(|e| {
                EarthSnailError::Network(format!("HTTP EOF-delimited body read error: {e}"))
            })?;
            if n == 0 {
                break;
            }
            if buffered_body.len() + n > max_body_bytes {
                return Err(EarthSnailError::CapacityExceeded(
                    "HTTP body exceeded maximum safety limit".into(),
                ));
            }
            buffered_body.extend_from_slice(&chunk[..n]);
        }
        buffered_body
    };

    Ok(HttpResponse {
        status_code,
        headers,
        body,
    })
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn find_crlf(buf: &[u8], start: usize) -> Option<usize> {
    buf[start..]
        .windows(2)
        .position(|w| w == b"\r\n")
        .map(|p| start + p)
}

fn decode_chunked_body<R: Read>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    max_body_bytes: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0_usize;
    let mut io_tmp = [0_u8; 4096];

    loop {
        // Ensure we have a full `<hex-size>[;ext]\r\n` line in `buf[pos..]`:
        let line_end = loop {
            if let Some(idx) = find_crlf(buf, pos) {
                break idx;
            }
            if buf.len() - pos > 1024 {
                return Err(EarthSnailError::Network(
                    "malformed HTTP chunk size line".into(),
                ));
            }
            let n = reader
                .read(&mut io_tmp)
                .map_err(|e| EarthSnailError::Network(format!("HTTP chunked read error: {e}")))?;
            if n == 0 {
                return Err(EarthSnailError::Network(
                    "unexpected EOF in HTTP chunked transfer".into(),
                ));
            }
            buf.extend_from_slice(&io_tmp[..n]);
        };

        let size_line = core::str::from_utf8(&buf[pos..line_end])
            .map_err(|_| EarthSnailError::Network("non-ASCII HTTP chunk header".into()))?;
        let hex_part = size_line.split(';').next().unwrap_or("").trim();
        let chunk_size = usize::from_str_radix(hex_part, 16).map_err(|_| {
            EarthSnailError::Network(format!("invalid hex HTTP chunk size '{hex_part}'"))
        })?;
        pos = line_end + 2;

        if chunk_size == 0 {
            return Ok(out);
        }
        if out.len().saturating_add(chunk_size) > max_body_bytes {
            return Err(EarthSnailError::CapacityExceeded(
                "HTTP chunked body exceeded maximum safety limit".into(),
            ));
        }

        // Read `chunk_size` payload bytes + trailing `\r\n` (`chunk_size + 2` bytes):
        let required = pos + chunk_size + 2;
        while buf.len() < required {
            let n = reader.read(&mut io_tmp).map_err(|e| {
                EarthSnailError::Network(format!("HTTP chunk payload read error: {e}"))
            })?;
            if n == 0 {
                return Err(EarthSnailError::Network(
                    "truncated HTTP chunk payload".into(),
                ));
            }
            buf.extend_from_slice(&io_tmp[..n]);
        }

        if &buf[pos + chunk_size..pos + chunk_size + 2] != b"\r\n" {
            return Err(EarthSnailError::Network(
                "missing CRLF terminator after HTTP chunk payload".into(),
            ));
        }
        out.extend_from_slice(&buf[pos..pos + chunk_size]);
        pos += chunk_size + 2;

        // Compact `buf` periodically to keep memory bounded:
        if pos > 65_536 {
            buf.drain(..pos);
            pos = 0;
        }
    }
}

/// High-level FDSN Web Services (`FDSNWS 1.x`) client generic over any `HttpTransport`.
#[derive(Debug, Clone)]
pub struct FdsnClient<T: HttpTransport = TcpHttpTransport> {
    pub provider: FdsnProvider,
    pub transport: T,
}

impl FdsnClient<TcpHttpTransport> {
    /// Creates an `FdsnClient` for `provider` using the built-in zero-dependency `TcpHttpTransport`.
    pub fn new(provider: FdsnProvider) -> Self {
        Self {
            provider,
            transport: TcpHttpTransport::default(),
        }
    }

    /// Creates an `FdsnClient` from a provider name or custom base URL (ObsPy `Client("EARTHSCOPE")` style).
    pub fn from_name(name: &str) -> Self {
        Self::new(FdsnProvider::from_name(name))
    }
}

impl<T: HttpTransport> FdsnClient<T> {
    /// Creates an `FdsnClient` with a custom `HttpTransport`.
    pub fn with_transport(provider: FdsnProvider, transport: T) -> Self {
        Self {
            provider,
            transport,
        }
    }

    /// Queries `fdsnws/dataselect/1/query` via `GET` and decodes the returned waveforms (miniSEED v2,
    /// miniSEED v3, or GeoCSV).
    pub fn get_waveforms(&self, query: &WaveformQuery) -> Result<Stream> {
        let url = query.build_url(self.provider.base_url());
        let resp = self.transport.get(&url)?;
        let body = check_fdsn_status(resp, "dataselect")?;
        read_waveforms_bytes(&body)
    }

    /// Queries `fdsnws/dataselect/1/query` via bulk `POST` and decodes the returned waveforms.
    pub fn post_waveforms(&self, query: &WaveformQuery) -> Result<Stream> {
        let url = format!("{}/fdsnws/dataselect/1/query", self.provider.base_url());
        let post_body = query.build_post_body()?;
        let resp = self
            .transport
            .post(&url, "text/plain; charset=utf-8", post_body.as_bytes())?;
        let body = check_fdsn_status(resp, "dataselect")?;
        read_waveforms_bytes(&body)
    }

    /// Queries `fdsnws/station/1/query` via `GET` and parses the returned `StationXML` inventory.
    pub fn get_stations(&self, query: &StationQuery) -> Result<StationInventory> {
        let url = query.build_url(self.provider.base_url());
        let resp = self.transport.get(&url)?;
        let body = check_fdsn_status(resp, "station")?;
        read_stationxml_bytes(&body)
    }

    /// Queries `fdsnws/station/1/query` via bulk `POST` and parses the returned `StationXML`.
    pub fn post_stations(&self, query: &StationQuery) -> Result<StationInventory> {
        let url = format!("{}/fdsnws/station/1/query", self.provider.base_url());
        let post_body = query.build_post_body()?;
        let resp = self
            .transport
            .post(&url, "text/plain; charset=utf-8", post_body.as_bytes())?;
        let body = check_fdsn_status(resp, "station")?;
        read_stationxml_bytes(&body)
    }

    /// Queries `fdsnws/event/1/query` via `GET` and parses the returned `QuakeML` `EventCatalog`.
    pub fn get_events(&self, query: &EventQuery) -> Result<EventCatalog> {
        let url = query.build_url(self.provider.base_url());
        let resp = self.transport.get(&url)?;
        let body = check_fdsn_status(resp, "event")?;
        read_quakeml_bytes(&body)
    }
}

fn check_fdsn_status(resp: HttpResponse, service: &str) -> Result<Vec<u8>> {
    match resp.status_code {
        200 => Ok(resp.body),
        204 | 404 => Err(EarthSnailError::EmptyData(format!(
            "FDSNWS {service} returned {} (no matching seismic data)",
            resp.status_code
        ))),
        code => {
            let preview_len = resp.body.len().min(240);
            let snippet = String::from_utf8_lossy(&resp.body[..preview_len]);
            Err(EarthSnailError::Network(format!(
                "FDSNWS {service} failed with HTTP {code}: {}",
                snippet.trim()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mseed::{write_mseed_trace_bytes, MseedWriteOptions};
    use earthsnail_core::{Trace, TraceStats};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn fdsn_url_and_bulk_post_formatting() {
        let t0: UtcTimestamp = "2011-03-11T05:46:00Z".parse().unwrap();
        let t1: UtcTimestamp = "2011-03-11T06:00:00Z".parse().unwrap();

        let wq = WaveformQuery::new("IU", "ANMO", "10", "BH?", t0, t1)
            .with_quality('B')
            .with_minimum_length(10.0);
        let url = wq.build_url(FdsnProvider::EarthScope.base_url());
        assert!(url.starts_with("https://service.earthscope.org/fdsnws/dataselect/1/query?"));
        assert!(url.contains("network=IU"));
        assert!(url.contains("station=ANMO"));
        assert!(url.contains("location=10"));
        assert!(url.contains("channel=BH%3F"));
        assert!(url.contains("starttime=2011-03-11T05%3A46%3A00"));
        assert!(url.contains("quality=B"));

        let post = wq.build_post_body().unwrap();
        assert!(post.contains("quality=B\n"));
        assert!(post.contains("IU ANMO 10 BH? 2011-03-11T05:46:00 2011-03-11T06:00:00\n"));

        let eq = EventQuery::new()
            .with_time_window(t0, t1)
            .with_min_magnitude(7.5)
            .with_limit(5);
        let ev_url = eq.build_url(FdsnProvider::Usgs.base_url());
        assert!(ev_url.starts_with("https://earthquake.usgs.gov/fdsnws/event/1/query?"));
        assert!(ev_url.contains("minmagnitude=7.5"));
        assert!(ev_url.contains("limit=5"));
    }

    #[test]
    fn http_response_parser_chunked_and_content_length_in_memory() {
        let raw_chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/plain\r\n\r\n5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\n\r\n";
        let mut cursor = std::io::Cursor::new(&raw_chunked[..]);
        let resp = read_http_response(&mut cursor, 4096).unwrap();
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.body, b"hello, world");

        let raw_204 = b"HTTP/1.1 204 No Content\r\nServer: EarthScope\r\n\r\n";
        let mut c204 = std::io::Cursor::new(&raw_204[..]);
        let r204 = read_http_response(&mut c204, 4096).unwrap();
        assert_eq!(r204.status_code, 204);
        assert!(r204.body.is_empty());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn tcp_http_chunked_204_and_https_redirect_handling() {
        // 1. Loopback server serving chunked miniSEED v2 waveform!
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let t0: UtcTimestamp = "2020-01-01T00:00:00Z".parse().unwrap();
        let ch = ChannelId::new("IU", "ANMO", "00", "BHZ");
        let tr = Trace::new(
            TraceStats::new(ch.clone(), t0, 20.0),
            (0..100).map(|i| (i * 7 - 50) as f64).collect(),
        );
        let mseed_payload = write_mseed_trace_bytes(&tr, MseedWriteOptions::default()).unwrap();
        let payload_copy = mseed_payload.clone();

        let handle = thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut req_buf = [0_u8; 1024];
            let _ = sock.read(&mut req_buf).unwrap();
            // Send response split into 2 HTTP chunks (`100` bytes + remainder)
            let c1 = &payload_copy[..100];
            let c2 = &payload_copy[100..];
            let hdr = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/vnd.fdsn.mseed\r\nTransfer-Encoding: chunked\r\n\r\n{:X}\r\n",
                c1.len()
            );
            sock.write_all(hdr.as_bytes()).unwrap();
            sock.write_all(c1).unwrap();
            sock.write_all(format!("\r\n{:X}\r\n", c2.len()).as_bytes())
                .unwrap();
            sock.write_all(c2).unwrap();
            sock.write_all(b"\r\n0\r\n\r\n").unwrap();
        });

        let client = FdsnClient::from_name(&format!("http://{addr}"));
        let q = WaveformQuery::new("IU", "ANMO", "00", "BHZ", t0, t0.add_nanos(5_000_000_000));
        let st = client.get_waveforms(&q).unwrap();
        handle.join().unwrap();

        assert_eq!(st.len(), 1);
        assert_eq!(st[0].stats.channel, ch);
        assert_eq!(st[0].data, tr.data);

        // 2. 204 No Content -> EmptyData error
        let listener_204 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr_204 = listener_204.local_addr().unwrap();
        let h204 = thread::spawn(move || {
            let (mut sock, _) = listener_204.accept().unwrap();
            let mut b = [0_u8; 512];
            let _ = sock.read(&mut b);
            sock.write_all(b"HTTP/1.1 204 No Content\r\n\r\n").unwrap();
        });
        let client_204 = FdsnClient::from_name(&format!("http://{addr_204}"));
        let err_204 = client_204.get_waveforms(&q).unwrap_err();
        h204.join().unwrap();
        assert!(matches!(err_204, EarthSnailError::EmptyData(_)));

        // 3. 301 Redirect to https:// -> clear Network diagnostic error
        let listener_301 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr_301 = listener_301.local_addr().unwrap();
        let h301 = thread::spawn(move || {
            let (mut sock, _) = listener_301.accept().unwrap();
            let mut b = [0_u8; 512];
            let _ = sock.read(&mut b);
            sock.write_all(
                b"HTTP/1.1 301 Moved Permanently\r\nLocation: https://service.earthscope.org/fdsnws/dataselect/1/query\r\nContent-Length: 0\r\n\r\n",
            )
            .unwrap();
        });
        let client_301 = FdsnClient::from_name(&format!("http://{addr_301}"));
        let err_301 = client_301.get_waveforms(&q).unwrap_err();
        h301.join().unwrap();
        assert!(matches!(err_301, EarthSnailError::Network(_)));
    }
}
