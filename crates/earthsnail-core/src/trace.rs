//! Waveform containers (`ChannelId`, `TraceStats`, `Trace`, `Stream`, `GapOverlap`) with
//! drift-free timing and wildcard `NET.STA.LOC.CHA` selection.

use crate::error::{EarthSnailError, Result};
use crate::math;
use crate::sac_header::{is_sac_float_undef, SacHeader};
use crate::time::UtcTimestamp;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Index, IndexMut};
use core::str::FromStr;

/// Four-part FDSN / SEED channel identifier (`Network.Station.Location.Channel`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ChannelId {
    /// FDSN network code (e.g. `"IU"`, `"II"`, `"US"`).
    pub network: String,
    /// Station code (e.g. `"ANMO"`, `"CCM"`).
    pub station: String,
    /// Location code (e.g. `"00"`, `"10"`, or `""` for blank location).
    pub location: String,
    /// 3-character SEED band/source/subsource channel code (e.g. `"BHZ"`, `"HHN"`).
    pub channel: String,
}

impl ChannelId {
    /// Creates a trimmed `ChannelId`.
    pub fn new(
        network: impl Into<String>,
        station: impl Into<String>,
        location: impl Into<String>,
        channel: impl Into<String>,
    ) -> Self {
        let loc_raw = location.into();
        let loc_trimmed = loc_raw.trim();
        let location = if loc_trimmed == "--" {
            String::new()
        } else {
            String::from(loc_trimmed)
        };
        Self {
            network: String::from(network.into().trim()),
            station: String::from(station.into().trim()),
            location,
            channel: String::from(channel.into().trim()),
        }
    }

    /// Formats as canonical SEED string `"NET.STA.LOC.CHA"`.
    pub fn seed_id(&self) -> String {
        format!(
            "{}.{}.{}.{}",
            self.network, self.station, self.location, self.channel
        )
    }

    /// Formats as miniSEED v3 FDSN Source Identifier (`"FDSN:NET_STA_LOC_B_S_SS"`).
    pub fn fdsn_source_id(&self) -> String {
        let mut chars = self.channel.chars();
        let b = chars.next().map(String::from).unwrap_or_default();
        let s = chars.next().map(String::from).unwrap_or_default();
        let ss: String = chars.collect();
        format!(
            "FDSN:{}_{}_{}_{}_{}_{}",
            self.network, self.station, self.location, b, s, ss
        )
    }

    /// Parses `"NET.STA.LOC.CHA"` or miniSEED v3 `"FDSN:NET_STA_LOC_B_S_SS"`.
    pub fn parse_id(input: &str) -> Result<Self> {
        let s = input.trim();
        if let Some(rest) = s.strip_prefix("FDSN:") {
            let parts: Vec<&str> = rest.splitn(6, '_').collect();
            if parts.len() != 6 {
                return Err(EarthSnailError::Format(format!(
                    "invalid FDSN Source Identifier '{input}'"
                )));
            }
            let cha = format!("{}{}{}", parts[3], parts[4], parts[5]);
            return Ok(Self::new(parts[0], parts[1], parts[2], cha));
        }
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return Err(EarthSnailError::Format(format!(
                "expected 4-part 'NET.STA.LOC.CHA' identifier, got '{input}'"
            )));
        }
        Ok(Self::new(parts[0], parts[1], parts[2], parts[3]))
    }

    /// Last character of `channel` (typically `'Z'`, `'N'`, `'E'`, `'1'`, `'2'`, `'R'`, `'T'`).
    pub fn component(&self) -> Option<char> {
        self.channel.chars().last()
    }

    /// Case-insensitive glob matching (`*` and `?`) against a `"NET.STA.LOC.CHA"` pattern.
    pub fn matches_pattern(&self, pattern: &str) -> bool {
        let pat = pattern.trim();
        if pat == "*" {
            return true;
        }
        let parts: Vec<&str> = pat.split('.').collect();
        if parts.len() != 4 {
            return glob_match_ci(pat, &self.seed_id());
        }
        self.matches_components(
            Some(parts[0]),
            Some(parts[1]),
            Some(parts[2]),
            Some(parts[3]),
        )
    }

    /// Case-insensitive glob matching against optional `network`, `station`, `location`, `channel` patterns.
    pub fn matches_components(
        &self,
        network: Option<&str>,
        station: Option<&str>,
        location: Option<&str>,
        channel: Option<&str>,
    ) -> bool {
        if let Some(np) = network {
            if !glob_match_ci(np, &self.network) {
                return false;
            }
        }
        if let Some(sp) = station {
            if !glob_match_ci(sp, &self.station) {
                return false;
            }
        }
        if let Some(lp) = location {
            let loc_norm = if lp == "--" { "" } else { lp };
            if !glob_match_ci(loc_norm, &self.location) {
                return false;
            }
        }
        if let Some(cp) = channel {
            if !glob_match_ci(cp, &self.channel) {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.network, self.station, self.location, self.channel
        )
    }
}

impl FromStr for ChannelId {
    type Err = EarthSnailError;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse_id(s)
    }
}

/// Per-trace timing, sampling, calibration, and optional SAC header metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceStats {
    /// Channel identifier (`NET.STA.LOC.CHA`).
    pub channel: ChannelId,
    /// UTC timestamp of sample index `0`.
    pub starttime: UtcTimestamp,
    /// Sampling rate in Hz (`> 0.0`).
    pub sampling_rate: f64,
    /// Scalar calibration factor (`1.0` default).
    pub calib: f64,
    /// Optional preserved or synthesized 158-word SAC header.
    pub sac: Option<SacHeader>,
}

impl Default for TraceStats {
    fn default() -> Self {
        Self {
            channel: ChannelId::default(),
            starttime: UtcTimestamp::UNIX_EPOCH,
            sampling_rate: 1.0,
            calib: 1.0,
            sac: None,
        }
    }
}

impl TraceStats {
    /// Creates `TraceStats` with `calib = 1.0` and `sac = None`.
    pub fn new(channel: ChannelId, starttime: UtcTimestamp, sampling_rate: f64) -> Self {
        Self {
            channel,
            starttime,
            sampling_rate,
            calib: 1.0,
            sac: None,
        }
    }

    /// Sample interval $\Delta t = 1 / f_s$ in seconds.
    #[inline]
    pub fn delta(&self) -> f64 {
        1.0 / self.sampling_rate
    }

    /// Constructs `TraceStats` from a `SacHeader`.
    pub fn from_sac_header(hdr: &SacHeader) -> Result<Self> {
        if is_sac_float_undef(hdr.delta) || hdr.delta <= 0.0 || !hdr.delta.is_finite() {
            return Err(EarthSnailError::Header(format!(
                "invalid or undefined SAC DELTA: {}",
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
        Ok(Self {
            channel,
            starttime,
            sampling_rate,
            calib,
            sac: Some(hdr.clone()),
        })
    }

    /// Synchronizes `channel`, `starttime`, `sampling_rate`, and `npts` into `self.sac` (creating one if `None`).
    pub fn sync_to_sac_header(&mut self, data: &[f64]) {
        let delta_s = self.delta();
        let hdr = self.sac.get_or_insert_with(SacHeader::new);
        let _ = hdr.set_str("knetwk", &self.channel.network);
        let _ = hdr.set_str("kstnm", &self.channel.station);
        let _ = hdr.set_str("khole", &self.channel.location);
        let _ = hdr.set_str("kcmpnm", &self.channel.channel);
        hdr.set_starttime_and_span(self.starttime, data.len(), delta_s);
        hdr.update_dep_stats(data);
    }
}

/// A single contiguous, evenly sampled seismogram (`stats` + `Vec<f64>` samples).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trace {
    /// Channel, timing, rate, and optional SAC metadata.
    pub stats: TraceStats,
    /// Double-precision waveform samples (unbounded length).
    pub data: Vec<f64>,
}

impl Trace {
    /// Creates a new `Trace`.
    pub fn new(stats: TraceStats, data: Vec<f64>) -> Self {
        Self { stats, data }
    }

    /// Number of waveform samples (`NPTS`).
    #[inline]
    pub fn npts(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if `data` has zero samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sample interval $\Delta t = 1 / f_s$ in seconds.
    #[inline]
    pub fn delta(&self) -> f64 {
        self.stats.delta()
    }

    /// Drift-free UTC timestamp of the last sample (`starttime` if `data.len() <= 1`, matching ObsPy & SAC `E`).
    pub fn endtime(&self) -> UtcTimestamp {
        if self.data.len() <= 1 {
            return self.stats.starttime;
        }
        let last_idx = (self.data.len() - 1) as i64;
        self.stats
            .starttime
            .sample_timestamp(last_idx, self.stats.sampling_rate)
            .unwrap_or_else(|_| {
                self.stats
                    .starttime
                    .add_secs_f64((last_idx as f64) / self.stats.sampling_rate)
                    .unwrap_or(self.stats.starttime)
            })
    }

    /// Waveform duration `(npts - 1) / sampling_rate` in seconds (`0.0` when `npts <= 1`, matching `endtime - starttime`).
    #[inline]
    pub fn duration_s(&self) -> f64 {
        if self.data.len() <= 1 || self.stats.sampling_rate <= 0.0 {
            0.0
        } else {
            ((self.data.len() - 1) as f64) / self.stats.sampling_rate
        }
    }

    /// Drift-free UTC timestamp of sample `index`.
    pub fn sample_time(&self, index: usize) -> Result<UtcTimestamp> {
        if index >= self.data.len() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "sample index {index} out of bounds for trace of length {}",
                self.data.len()
            )));
        }
        self.stats
            .starttime
            .sample_timestamp(index as i64, self.stats.sampling_rate)
    }

    /// Computes `(min, max, mean)` of `self.data` using Kahan summation (`None` if empty).
    pub fn min_max_mean(&self) -> Option<(f64, f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut comp = 0.0;
        for &x in &self.data {
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
            let y = x - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        Some((mn, mx, sum / (self.data.len() as f64)))
    }

    /// Extracts a non-destructive sub-trace covering `[start, end]` (inclusive nearest samples).
    pub fn slice_time(&self, start: UtcTimestamp, end: UtcTimestamp) -> Result<Self> {
        if end < start {
            return Err(EarthSnailError::InvalidParameter(
                "slice end timestamp precedes start timestamp".into(),
            ));
        }
        if self.data.is_empty() {
            return Ok(self.clone());
        }
        let i0_raw = self
            .stats
            .starttime
            .sample_index_nearest(start, self.stats.sampling_rate)?;
        let i1_raw = self
            .stats
            .starttime
            .sample_index_nearest(end, self.stats.sampling_rate)?;
        let n = self.data.len() as i64;
        if i1_raw < 0 || i0_raw >= n {
            let mut empty_stats = self.stats.clone();
            empty_stats.starttime = start;
            return Ok(Self::new(empty_stats, Vec::new()));
        }
        let i0 = i0_raw.clamp(0, n - 1) as usize;
        let i1 = i1_raw.clamp(0, n - 1) as usize;
        if i1 < i0 {
            let mut empty_stats = self.stats.clone();
            empty_stats.starttime = start;
            return Ok(Self::new(empty_stats, Vec::new()));
        }
        let new_start = self.sample_time(i0)?;
        let sub_data = self.data[i0..=i1].to_vec();
        let mut new_stats = self.stats.clone();
        new_stats.starttime = new_start;
        if new_stats.sac.is_some() {
            new_stats.sync_to_sac_header(&sub_data);
        }
        Ok(Self::new(new_stats, sub_data))
    }
}

/// Detected time tear (gap `difference_s > 0` or overlap `difference_s < 0`) between adjacent traces
/// of the same channel in a `Stream`.
#[derive(Debug, Clone, PartialEq)]
pub struct GapOverlap {
    /// Channel identifier.
    pub channel: ChannelId,
    /// Sampling rate in Hz.
    pub sampling_rate: f64,
    /// Endtime of the earlier segment.
    pub prev_endtime: UtcTimestamp,
    /// Starttime of the following segment.
    pub next_starttime: UtcTimestamp,
    /// Time offset `next_starttime - (prev_endtime + dt)` in seconds (`> 0` gap, `< 0` overlap).
    pub difference_s: f64,
    /// Approximate missing (`> 0`) or overlapping (`< 0`) sample count.
    pub samples: i64,
}

/// Collection of `Trace` objects (`ObsPy` `Stream` equivalent) with unlimited trace slots.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stream {
    /// Ordered list of traces.
    pub traces: Vec<Trace>,
}

impl Stream {
    /// Creates an empty `Stream`.
    pub const fn new() -> Self {
        Self { traces: Vec::new() }
    }

    /// Wraps a `Vec<Trace>` into a `Stream`.
    pub const fn from_traces(traces: Vec<Trace>) -> Self {
        Self { traces }
    }

    /// Number of traces in the stream.
    #[inline]
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Returns `true` if the stream contains no traces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Appends a `Trace` to the stream.
    pub fn push(&mut self, trace: Trace) {
        self.traces.push(trace);
    }

    /// Borrows trace slice.
    #[inline]
    pub fn as_slice(&self) -> &[Trace] {
        &self.traces
    }

    /// Selects traces matching a `"NET.STA.LOC.CHA"` wildcard pattern (supporting `*` and `?`).
    pub fn select(&self, pattern: &str) -> Self {
        let matched = self
            .traces
            .iter()
            .filter(|tr| tr.stats.channel.matches_pattern(pattern))
            .cloned()
            .collect();
        Self { traces: matched }
    }

    /// Selects traces matching individual wildcard filters (`network`, `station`, `location`, `channel`).
    pub fn select_components(
        &self,
        network: Option<&str>,
        station: Option<&str>,
        location: Option<&str>,
        channel: Option<&str>,
    ) -> Self {
        let matched = self
            .traces
            .iter()
            .filter(|tr| {
                tr.stats
                    .channel
                    .matches_components(network, station, location, channel)
            })
            .cloned()
            .collect();
        Self { traces: matched }
    }

    /// Sorts traces in-place by `(channel, starttime, endtime)`.
    pub fn sort(&mut self) {
        self.traces.sort_by(|a, b| {
            a.stats
                .channel
                .cmp(&b.stats.channel)
                .then_with(|| a.stats.starttime.cmp(&b.stats.starttime))
                .then_with(|| a.endtime().cmp(&b.endtime()))
        });
    }

    /// Inspects adjacent segments sharing `(ChannelId, sampling_rate)` for gaps or overlaps exceeding `0.5 * dt`.
    pub fn get_gaps(&self) -> Vec<GapOverlap> {
        let mut refs: Vec<&Trace> = self.traces.iter().filter(|t| !t.is_empty()).collect();
        refs.sort_by(|a, b| {
            a.stats
                .channel
                .cmp(&b.stats.channel)
                .then_with(|| a.stats.starttime.cmp(&b.stats.starttime))
        });
        let mut out = Vec::new();
        for pair in refs.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if a.stats.channel != b.stats.channel {
                continue;
            }
            if (a.stats.sampling_rate - b.stats.sampling_rate).abs() > 1e-6 * a.stats.sampling_rate
            {
                continue;
            }
            let fs = a.stats.sampling_rate;
            let expected_next = match a.stats.starttime.sample_timestamp(a.npts() as i64, fs) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let diff_s = b.stats.starttime.duration_secs_f64_since(expected_next);
            let dt = 1.0 / fs;
            if diff_s.abs() > 0.5 * dt {
                let samples = math::round(diff_s * fs) as i64;
                out.push(GapOverlap {
                    channel: a.stats.channel.clone(),
                    sampling_rate: fs,
                    prev_endtime: a.endtime(),
                    next_starttime: b.stats.starttime,
                    difference_s: diff_s,
                    samples,
                });
            }
        }
        out
    }
}

impl Index<usize> for Stream {
    type Output = Trace;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.traces[index]
    }
}

impl IndexMut<usize> for Stream {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.traces[index]
    }
}

impl IntoIterator for Stream {
    type Item = Trace;
    type IntoIter = alloc::vec::IntoIter<Trace>;
    fn into_iter(self) -> Self::IntoIter {
        self.traces.into_iter()
    }
}

impl<'a> IntoIterator for &'a Stream {
    type Item = &'a Trace;
    type IntoIter = core::slice::Iter<'a, Trace>;
    fn into_iter(self) -> Self::IntoIter {
        self.traces.iter()
    }
}

impl<'a> IntoIterator for &'a mut Stream {
    type Item = &'a mut Trace;
    type IntoIter = core::slice::IterMut<'a, Trace>;
    fn into_iter(self) -> Self::IntoIter {
        self.traces.iter_mut()
    }
}

/// Iterative linear-time glob matcher supporting `*` and `?` (ASCII case-insensitive).
fn glob_match_ci(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let (mut pi, mut ti) = (0_usize, 0_usize);
    let (mut star_pi, mut star_ti) = (None::<usize>, 0_usize);

    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi].eq_ignore_ascii_case(&t[ti])) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star_pi = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_endtime_stream_select_and_gaps() {
        let t0: UtcTimestamp = "2024-01-01T00:00:00Z".parse().unwrap();
        let ch_z = ChannelId::new("IU", "ANMO", "00", "BHZ");
        let ch_n = ChannelId::new("IU", "ANMO", "00", "BHN");
        let ch_e = ChannelId::new("IU", "ANMO", "10", "HHE");

        let tr1 = Trace::new(
            TraceStats::new(ch_z.clone(), t0, 40.0),
            alloc::vec![1.0; 401],
        );
        assert!((tr1.duration_s() - 10.0).abs() < 1e-15);
        assert_eq!(tr1.endtime().to_iso8601(), "2024-01-01T00:00:10Z");

        // Second BHZ segment starting 2.0 s after expected continuation (expected = 10.025s, actual = 12.025s)
        let t2 = t0.add_nanos(12_025_000_000);
        let tr2 = Trace::new(TraceStats::new(ch_z, t2, 40.0), alloc::vec![2.0; 200]);
        let tr_n = Trace::new(TraceStats::new(ch_n, t0, 40.0), alloc::vec![0.0; 100]);
        let tr_e = Trace::new(TraceStats::new(ch_e, t0, 100.0), alloc::vec![0.0; 100]);

        let st = Stream::from_traces(alloc::vec![tr1, tr2, tr_n, tr_e]);
        assert_eq!(st.select("IU.ANMO.00.BH?").len(), 3);
        assert_eq!(st.select("*.*.10.HH*").len(), 1);
        assert_eq!(st.select("IU.ANMO.*.??Z").len(), 2);

        let gaps = st.get_gaps();
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].difference_s - 2.0).abs() < 1e-12);
        assert_eq!(gaps[0].samples, 80);
    }
}
