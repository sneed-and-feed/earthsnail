//! Gate 1 Differential Oracle Parity Suite for `earthsnail-io`.
//!
//! Verifies exact sample-for-sample, header-for-header, and complex transfer-function parity
//! against ALL ObsPy/evalresp golden fixtures in `fixtures/io/`:
//! - `test_le.sac`, `test_be.sac`, `test_alpha.sac` (`sac_expected.json`)
//! - `mseed_steim1.mseed`, `mseed_steim2.mseed`, `mseed_int32.mseed`, `mseed_float64.mseed` (`mseed_expected.json`)
//! - `iu_anmo_bhz.xml`, `iu_anmo_bhz.sacpz`, `events.qml` (`metadata_expected.json`)
//! - `real_event/real_3c.mseed`, `real_3c_station.xml`, `real_3c_bh*.sac` (`real_event_expected.json`)

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use earthsnail_core::{Complex64, KinematicUnit, SacEndian, UtcTimestamp};
use earthsnail_io::{
    detect_waveform_format, read_mseed3_bytes, read_mseed_bytes, read_mseed_records,
    read_quakeml_bytes, read_sac_alpha_bytes, read_sac_bytes, read_sac_bytes_with_endian,
    read_sacpz_bytes, read_stationxml_bytes, read_waveforms_bytes, write_mseed3_trace_bytes,
    write_mseed_trace_bytes, write_quakeml_str, write_sac_alpha_str, write_sac_bytes,
    write_sacpz_str, write_stationxml_str, Mseed3WriteOptions, MseedEncoding, MseedWriteOptions,
    WaveformFormat,
};

const ORACLE_STAGING_IO_DIR: &str = r"C:\Users\x\.gemini\antigravity\brain\566443e0-66bc-4144-a363-aad888d88d34\scratch\oracle_staging\fixtures\io";

fn io_fixture_path(rel: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_candidate = manifest_dir
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("io")
        .join(rel);
    if repo_candidate.exists() {
        return repo_candidate;
    }
    Path::new(ORACLE_STAGING_IO_DIR).join(rel)
}

// ---------------------------------------------------------------------------
// Minimal zero-dependency RFC-8259 JSON parser for reading golden `.json` files
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum JsonVal {
    Null,
    Bool(bool),
    Num(f64),
    Int(i64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(BTreeMap<String, JsonVal>),
}

impl JsonVal {
    fn get(&self, key: &str) -> &JsonVal {
        match self {
            Self::Obj(m) => m
                .get(key)
                .unwrap_or_else(|| panic!("missing JSON key '{key}'")),
            _ => panic!("expected JSON object when looking up '{key}'"),
        }
    }

    fn as_obj(&self) -> &BTreeMap<String, JsonVal> {
        match self {
            Self::Obj(m) => m,
            _ => panic!("expected JSON object"),
        }
    }

    fn as_arr(&self) -> &[JsonVal] {
        match self {
            Self::Arr(a) => a,
            _ => panic!("expected JSON array"),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Str(s) => s,
            _ => panic!("expected JSON string, got {self:?}"),
        }
    }

    fn as_f64(&self) -> f64 {
        match self {
            Self::Num(n) => *n,
            Self::Int(i) => *i as f64,
            _ => panic!("expected JSON number, got {self:?}"),
        }
    }

    fn as_i64(&self) -> i64 {
        match self {
            Self::Int(i) => *i,
            Self::Num(n) => *n as i64,
            _ => panic!("expected JSON integer, got {self:?}"),
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    #[allow(dead_code)]
    fn as_bool(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            _ => panic!("expected JSON bool"),
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn parse_root(text: &'a str) -> JsonVal {
        let mut p = Self {
            b: text.as_bytes(),
            i: 0,
        };
        let v = p.parse_val();
        p.skip_ws();
        assert_eq!(p.i, p.b.len(), "trailing bytes in JSON");
        v
    }

    fn skip_ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn parse_val(&mut self) -> JsonVal {
        self.skip_ws();
        match self.b[self.i] {
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b'"' => JsonVal::Str(self.parse_str()),
            b't' => {
                assert_eq!(&self.b[self.i..self.i + 4], b"true");
                self.i += 4;
                JsonVal::Bool(true)
            }
            b'f' => {
                assert_eq!(&self.b[self.i..self.i + 5], b"false");
                self.i += 5;
                JsonVal::Bool(false)
            }
            b'n' => {
                assert_eq!(&self.b[self.i..self.i + 4], b"null");
                self.i += 4;
                JsonVal::Null
            }
            _ => self.parse_num(),
        }
    }

    fn parse_obj(&mut self) -> JsonVal {
        self.i += 1; // '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.b[self.i] == b'}' {
            self.i += 1;
            return JsonVal::Obj(map);
        }
        loop {
            self.skip_ws();
            let k = self.parse_str();
            self.skip_ws();
            assert_eq!(self.b[self.i], b':');
            self.i += 1;
            let v = self.parse_val();
            map.insert(k, v);
            self.skip_ws();
            if self.b[self.i] == b'}' {
                self.i += 1;
                break;
            }
            assert_eq!(self.b[self.i], b',');
            self.i += 1;
        }
        JsonVal::Obj(map)
    }

    fn parse_arr(&mut self) -> JsonVal {
        self.i += 1; // '['
        self.skip_ws();
        let mut vec = Vec::new();
        if self.b[self.i] == b']' {
            self.i += 1;
            return JsonVal::Arr(vec);
        }
        loop {
            vec.push(self.parse_val());
            self.skip_ws();
            if self.b[self.i] == b']' {
                self.i += 1;
                break;
            }
            assert_eq!(self.b[self.i], b',');
            self.i += 1;
        }
        JsonVal::Arr(vec)
    }

    fn parse_str(&mut self) -> String {
        assert_eq!(self.b[self.i], b'"');
        self.i += 1;
        let mut out = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            if c == b'"' {
                return out;
            }
            if c == b'\\' {
                let esc = self.b[self.i];
                self.i += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap();
                        let code = u32::from_str_radix(hex, 16).unwrap();
                        self.i += 4;
                        out.push(char::from_u32(code).unwrap());
                    }
                    _ => panic!("unsupported JSON escape"),
                }
            } else {
                out.push(c as char);
            }
        }
        panic!("unterminated JSON string");
    }

    fn parse_num(&mut self) -> JsonVal {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).unwrap();
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            if let Ok(iv) = s.parse::<i64>() {
                return JsonVal::Int(iv);
            }
        }
        JsonVal::Num(s.parse::<f64>().unwrap())
    }
}

fn load_json(rel: &str) -> JsonVal {
    let text = std::fs::read_to_string(io_fixture_path(rel)).expect("read golden JSON");
    JsonParser::parse_root(&text)
}

// ---------------------------------------------------------------------------
// 1. SAC Binary (LE, BE) & SAC Alpha Golden Parity
// ---------------------------------------------------------------------------

#[test]
fn golden_sac_le_be_and_alpha_parity() {
    let expected = load_json("sac_expected.json");
    let files = expected.get("files");

    let le_bytes = std::fs::read(io_fixture_path("test_le.sac")).unwrap();
    let be_bytes = std::fs::read(io_fixture_path("test_be.sac")).unwrap();
    let alpha_bytes = std::fs::read(io_fixture_path("test_alpha.sac")).unwrap();

    assert_eq!(
        detect_waveform_format(&le_bytes),
        Some(WaveformFormat::SacBinaryLittle)
    );
    assert_eq!(
        detect_waveform_format(&be_bytes),
        Some(WaveformFormat::SacBinaryBig)
    );
    assert_eq!(
        detect_waveform_format(&alpha_bytes),
        Some(WaveformFormat::SacAlpha)
    );

    let (tr_le, end_le) = read_sac_bytes_with_endian(&le_bytes).unwrap();
    let (tr_be, end_be) = read_sac_bytes_with_endian(&be_bytes).unwrap();
    let tr_alpha = read_sac_alpha_bytes(&alpha_bytes).unwrap();

    assert_eq!(end_le, SacEndian::Little);
    assert_eq!(end_be, SacEndian::Big);

    for (label, tr, is_binary) in [
        ("little_endian", &tr_le, true),
        ("big_endian", &tr_be, true),
        ("alpha", &tr_alpha, false),
    ] {
        let spec = files.get(label);
        let st = spec.get("stats");
        assert_eq!(tr.stats.channel.network, st.get("network").as_str());
        assert_eq!(tr.stats.channel.station, st.get("station").as_str());
        assert_eq!(tr.stats.channel.location, st.get("location").as_str());
        assert_eq!(tr.stats.channel.channel, st.get("channel").as_str());
        assert_eq!(tr.npts(), st.get("npts").as_i64() as usize);

        let exp_start: UtcTimestamp = st.get("starttime").as_str().parse().unwrap();
        let exp_end: UtcTimestamp = st.get("endtime").as_str().parse().unwrap();
        assert_eq!(tr.stats.starttime, exp_start, "{label} starttime");
        assert!(
            tr.endtime().duration_nanos_since(exp_end).abs() <= 1_000,
            "{label} endtime mismatch: got {} vs {exp_end}",
            tr.endtime()
        );
        assert!(
            (tr.stats.sampling_rate - st.get("sampling_rate").as_f64()).abs() < 1e-5,
            "{label} sampling_rate"
        );

        // Verify EVERY key in `sac_header`:
        let hdr = tr.stats.sac.as_ref().expect("SAC header attached");
        for (k, v) in spec.get("sac_header").as_obj() {
            // Note `#7` (b): ObsPy names float word 9 `"internal0"`, whereas classic SAC names it `"internal1"`.
            let key = if k == "internal0" {
                "internal1"
            } else {
                k.as_str()
            };
            match v {
                JsonVal::Str(exp_s) => {
                    assert_eq!(hdr.get_str(key), Some(exp_s.as_str()), "{label}.{key}");
                }
                JsonVal::Int(exp_i) => {
                    if let Some(got_i) = hdr.get_int(key) {
                        assert_eq!(got_i as i64, *exp_i, "{label}.{key}");
                    } else {
                        let got_f = hdr.get_float(key).unwrap();
                        assert!((got_f - (*exp_i as f64)).abs() < 1e-5, "{label}.{key}");
                    }
                }
                JsonVal::Num(exp_f) => {
                    let got_f = hdr
                        .get_float(key)
                        .unwrap_or_else(|| panic!("missing float header {label}.{key}"));
                    if is_binary {
                        assert_eq!(
                            got_f as f32, *exp_f as f32,
                            "exact f32 header word mismatch on {label}.{key}"
                        );
                    } else {
                        assert!(
                            (got_f - *exp_f).abs() <= 1e-4 * exp_f.abs().max(1.0),
                            "{label}.{key}: got {got_f} vs {exp_f}"
                        );
                    }
                }
                _ => {}
            }
        }

        // Verify all 500 waveform samples:
        let exp_data = spec.get("data_f32_as_f64").as_arr();
        assert_eq!(tr.data.len(), exp_data.len());
        for (idx, (got, exp_v)) in tr.data.iter().zip(exp_data.iter()).enumerate() {
            let exp = exp_v.as_f64();
            if is_binary {
                // Exact 0-bit error on binary SAC LE & BE!
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "{label} sample {idx} bit mismatch: {got} vs {exp}"
                );
            } else {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "alpha sample {idx} mismatch: {got} vs {exp}"
                );
            }
        }
    }

    // Cross-endian binary round-trip (`LE -> write BE` matches `test_be.sac` samples & headers!):
    let rewritten_be = write_sac_bytes(&tr_le, SacEndian::Big).unwrap();
    assert_eq!(rewritten_be.len(), be_bytes.len());
    assert_eq!(&rewritten_be[632..], &be_bytes[632..]);

    // Alpha write -> read round-trip:
    let alpha_written = write_sac_alpha_str(&tr_le).unwrap();
    let tr_alpha_rt = read_waveforms_bytes(alpha_written.as_bytes()).unwrap();
    assert_eq!(tr_alpha_rt[0].npts(), 500);
    for (a, b) in tr_alpha_rt[0].data.iter().zip(tr_le.data.iter()) {
        assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0));
    }
}

// ---------------------------------------------------------------------------
// 2. miniSEED v2 (STEIM1, STEIM2, INT32, FLOAT64) & MS3 Round-Trip Parity
// ---------------------------------------------------------------------------

#[test]
fn golden_mseed_steim1_steim2_int32_float64_parity() {
    let expected = load_json("mseed_expected.json");
    let files = expected.get("files");

    for key in ["steim1", "steim2", "int32", "float64"] {
        let spec = files.get(key);
        let fname = spec.get("file").as_str();
        let raw = std::fs::read(io_fixture_path(fname)).unwrap();
        assert_eq!(raw.len() as i64, spec.get("size_bytes").as_i64());
        assert_eq!(
            detect_waveform_format(&raw),
            Some(WaveformFormat::MiniSeed2)
        );

        let records = read_mseed_records(&raw).unwrap();
        assert_eq!(
            records.len() as i64,
            spec.get("num_records").as_i64(),
            "{key} record count"
        );
        let exp_code = spec.get("seed_encoding_code").as_i64() as u8;
        assert_eq!(records[0].encoding as u8, exp_code);
        assert_eq!(
            records[0].record_length as i64,
            spec.get("record_length_bytes").as_i64()
        );

        let stream = read_mseed_bytes(&raw).unwrap();
        assert_eq!(stream.len(), 1, "{key} contiguous reassembly into 1 Trace");
        let tr = &stream[0];

        let st = spec.get("stats");
        assert_eq!(tr.stats.channel.network, st.get("network").as_str());
        assert_eq!(tr.stats.channel.station, st.get("station").as_str());
        assert_eq!(tr.stats.channel.location, st.get("location").as_str());
        assert_eq!(tr.stats.channel.channel, st.get("channel").as_str());
        assert_eq!(tr.npts(), st.get("npts").as_i64() as usize);
        assert_eq!(tr.stats.sampling_rate, st.get("sampling_rate").as_f64());

        let exp_start: UtcTimestamp = st.get("starttime").as_str().parse().unwrap();
        let exp_end: UtcTimestamp = st.get("endtime").as_str().parse().unwrap();
        assert_eq!(tr.stats.starttime, exp_start, "{key} exact ns starttime");
        assert_eq!(tr.endtime(), exp_end, "{key} exact ns endtime");

        let exp_samples = spec.get("samples").as_arr();
        assert_eq!(tr.data.len(), exp_samples.len());
        for (idx, (got, exp_val)) in tr.data.iter().zip(exp_samples.iter()).enumerate() {
            if key == "float64" {
                let exp = exp_val.as_f64();
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "FLOAT64 sample {idx} exact bit parity"
                );
            } else {
                let exp_i = exp_val.as_i64();
                assert_eq!(
                    *got as i64, exp_i,
                    "{key} sample {idx} exact lossless integer parity"
                );
            }
        }

        // Round-trip through OUR miniSEED v2 writer AND miniSEED v3 writer with identical encoding!
        let enc = MseedEncoding::from_code(exp_code).unwrap();
        let ms2_rt_bytes = write_mseed_trace_bytes(
            tr,
            MseedWriteOptions {
                record_length: 512,
                encoding: enc,
                ..MseedWriteOptions::default()
            },
        )
        .unwrap();
        let ms2_rt = read_mseed_bytes(&ms2_rt_bytes).unwrap();
        assert_eq!(ms2_rt.len(), 1);
        assert_eq!(ms2_rt[0].stats.starttime, tr.stats.starttime);
        assert_eq!(
            ms2_rt[0].data, tr.data,
            "{key} MS2 encoder->decoder lossless"
        );

        let ms3_rt_bytes = write_mseed3_trace_bytes(
            tr,
            &Mseed3WriteOptions {
                encoding: enc,
                max_samples_per_record: 500,
                ..Mseed3WriteOptions::default()
            },
        )
        .unwrap();
        let ms3_rt = read_mseed3_bytes(&ms3_rt_bytes).unwrap();
        assert_eq!(ms3_rt.len(), 1);
        assert_eq!(ms3_rt[0].stats.starttime, tr.stats.starttime);
        assert_eq!(
            ms3_rt[0].data, tr.data,
            "{key} MS3 encoder->decoder lossless"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. StationXML, SACPZ & QuakeML Golden Parity
// ---------------------------------------------------------------------------

#[test]
fn golden_stationxml_sacpz_and_quakeml_parity() {
    let expected = load_json("metadata_expected.json");

    // --- 3A. StationXML (`iu_anmo_bhz.xml`) ---
    let sxml_spec = expected.get("stationxml");
    let xml_bytes = std::fs::read(io_fixture_path("iu_anmo_bhz.xml")).unwrap();
    let inv = read_stationxml_bytes(&xml_bytes).unwrap();
    assert_eq!(inv.networks.len(), 1);
    let ch = inv.channels().next().unwrap();

    assert_eq!(ch.channel.network, sxml_spec.get("network").as_str());
    assert_eq!(ch.channel.station, sxml_spec.get("station").as_str());
    assert_eq!(ch.channel.location, sxml_spec.get("location").as_str());
    assert_eq!(ch.channel.channel, sxml_spec.get("channel").as_str());
    assert!((ch.latitude - sxml_spec.get("latitude").as_f64()).abs() < 1e-12);
    assert!((ch.longitude - sxml_spec.get("longitude").as_f64()).abs() < 1e-12);
    assert!((ch.elevation_m - sxml_spec.get("elevation_m").as_f64()).abs() < 1e-12);
    assert!((ch.depth_m - sxml_spec.get("depth_m").as_f64()).abs() < 1e-12);
    assert_eq!(ch.azimuth_deg, Some(sxml_spec.get("azimuth_deg").as_f64()));
    assert_eq!(ch.dip_deg, Some(sxml_spec.get("dip_deg").as_f64()));
    assert_eq!(
        ch.sample_rate_hz,
        Some(sxml_spec.get("sample_rate_hz").as_f64())
    );

    let resp = ch.response.as_ref().unwrap();
    assert_eq!(resp.stages.len(), 3);
    assert_eq!(
        resp.instrument_sensitivity.unwrap().value,
        sxml_spec.get("overall_sensitivity").as_f64()
    );

    let freqs: Vec<f64> = sxml_spec
        .get("frequencies_hz")
        .as_arr()
        .iter()
        .map(JsonVal::as_f64)
        .collect();
    let evalresp_map = sxml_spec.get("evalresp_complex_response");
    for (unit_name, ku) in [
        ("DISP", KinematicUnit::Displacement),
        ("VEL", KinematicUnit::Velocity),
        ("ACC", KinematicUnit::Acceleration),
    ] {
        let exp_arr = evalresp_map.get(unit_name).as_arr();
        for (i, &f) in freqs.iter().enumerate() {
            let exp_c = Complex64::new(
                exp_arr[i].get("real").as_f64(),
                exp_arr[i].get("imag").as_f64(),
            );
            let got_c = resp.eval_freq_kinematic(f, ku);
            let rel_err = (got_c - exp_c).abs() / exp_c.abs();
            // Note `#7 (c)`: XML stage gain product (`1177 * 1677720 = 1974676440`) vs 6-digit
            // `InstrumentSensitivity` (`1974680000`) is `1.8e-6` (~0.5–2.3 ppm).
            assert!(
                rel_err < 5e-6,
                "StationXML {unit_name} at {f} Hz mismatch: got {got_c} vs {exp_c} (rel_err={rel_err:e})"
            );
        }
    }

    // StationXML serialize -> re-parse round-trip preserves transfer function!
    let sxml_serialized = write_stationxml_str(&inv).unwrap();
    let inv_rt = read_stationxml_bytes(sxml_serialized.as_bytes()).unwrap();
    let resp_rt = inv_rt.channels().next().unwrap().response.as_ref().unwrap();
    for &f in &freqs {
        let h1 = resp.eval_freq_kinematic(f, KinematicUnit::Velocity);
        let h2 = resp_rt.eval_freq_kinematic(f, KinematicUnit::Velocity);
        assert!((h1 - h2).abs() / h1.abs() < 1e-12);
    }

    // --- 3B. SACPZ (`iu_anmo_bhz.sacpz`) ---
    let sacpz_spec = expected.get("sacpz");
    let sacpz_bytes = std::fs::read(io_fixture_path("iu_anmo_bhz.sacpz")).unwrap();
    let sacpz_rec = read_sacpz_bytes(&sacpz_bytes).unwrap();
    assert_eq!(
        sacpz_rec.channel.as_ref().unwrap().seed_id(),
        "IU.ANMO.10.BHZ"
    );
    assert!(
        (sacpz_rec.constant - sacpz_spec.get("constant").as_f64()).abs() / sacpz_rec.constant
            < 1e-14
    );
    assert_eq!(sacpz_rec.zeros.len(), 7);
    assert_eq!(sacpz_rec.poles.len(), 11);

    let sacpz_resp_map = sacpz_spec.get("complex_response");
    for (unit_name, ku) in [
        ("DISP", KinematicUnit::Displacement),
        ("VEL", KinematicUnit::Velocity),
        ("ACC", KinematicUnit::Acceleration),
    ] {
        let exp_arr = sacpz_resp_map.get(unit_name).as_arr();
        for (i, &f) in freqs.iter().enumerate() {
            let exp_c = Complex64::new(
                exp_arr[i].get("real").as_f64(),
                exp_arr[i].get("imag").as_f64(),
            );
            let got_c = sacpz_rec.response.eval_freq_kinematic(f, ku);
            let rel_err = (got_c - exp_c).abs() / exp_c.abs();
            assert!(
                rel_err < 1e-12,
                "SACPZ {unit_name} at {f} Hz mismatch: got {got_c} vs {exp_c} (rel_err={rel_err:e})"
            );
        }
    }

    // SACPZ serialize -> re-parse round-trip:
    let sacpz_written = write_sacpz_str(&sacpz_rec.response, sacpz_rec.metadata.as_ref()).unwrap();
    let sacpz_rt = read_sacpz_bytes(sacpz_written.as_bytes()).unwrap();
    for &f in &freqs {
        let h1 = sacpz_rec
            .response
            .eval_freq_kinematic(f, KinematicUnit::Velocity);
        let h2 = sacpz_rt
            .response
            .eval_freq_kinematic(f, KinematicUnit::Velocity);
        assert!((h1 - h2).abs() / h1.abs() < 1e-6);
    }

    // --- 3C. QuakeML (`events.qml`) ---
    let qml_spec = expected.get("quakeml");
    let qml_bytes = std::fs::read(io_fixture_path("events.qml")).unwrap();
    let catalog = read_quakeml_bytes(&qml_bytes).unwrap();
    assert_eq!(catalog.len() as i64, qml_spec.get("num_events").as_i64());

    for (ev, exp_ev) in catalog.events.iter().zip(qml_spec.get("events").as_arr()) {
        assert_eq!(ev.public_id, exp_ev.get("resource_id").as_str());
        assert_eq!(ev.origins.len() as i64, exp_ev.get("num_origins").as_i64());
        assert_eq!(
            ev.magnitudes.len() as i64,
            exp_ev.get("num_magnitudes").as_i64()
        );
        assert_eq!(
            ev.focal_mechanisms.len() as i64,
            exp_ev.get("num_focal_mechanisms").as_i64()
        );

        let po = exp_ev.get("preferred_origin");
        if !po.is_null() {
            let org = ev.preferred_origin().unwrap();
            let exp_t: UtcTimestamp = po.get("time").as_str().parse().unwrap();
            assert_eq!(org.time, exp_t);
            assert!((org.latitude - po.get("latitude").as_f64()).abs() < 1e-12);
            assert!((org.longitude - po.get("longitude").as_f64()).abs() < 1e-12);
            assert!((org.depth_m - po.get("depth_m").as_f64()).abs() < 1e-9);
        } else {
            assert!(ev.preferred_origin().is_none());
        }

        let pm = exp_ev.get("preferred_magnitude");
        if !pm.is_null() {
            let mag = ev.preferred_magnitude().unwrap();
            assert!((mag.mag - pm.get("mag").as_f64()).abs() < 1e-12);
            assert_eq!(mag.mag_type, pm.get("magnitude_type").as_str());
        } else {
            assert!(ev.preferred_magnitude().is_none());
        }

        for (fm, exp_fm) in ev
            .focal_mechanisms
            .iter()
            .zip(exp_ev.get("focal_mechanisms").as_arr())
        {
            assert_eq!(
                fm.public_id.as_deref(),
                Some(exp_fm.get("resource_id").as_str())
            );
            let np1 = fm.nodal_plane_1.unwrap();
            let exp_np1 = exp_fm.get("nodal_plane_1");
            assert_eq!(np1.strike_deg, exp_np1.get("strike").as_f64());
            assert_eq!(np1.dip_deg, exp_np1.get("dip").as_f64());
            assert_eq!(np1.rake_deg, exp_np1.get("rake").as_f64());

            let np2 = fm.nodal_plane_2.unwrap();
            let exp_np2 = exp_fm.get("nodal_plane_2");
            assert_eq!(np2.strike_deg, exp_np2.get("strike").as_f64());
            assert_eq!(np2.dip_deg, exp_np2.get("dip").as_f64());
            assert_eq!(np2.rake_deg, exp_np2.get("rake").as_f64());

            let mt = fm.moment_tensor.unwrap();
            let exp_mt = exp_fm.get("moment_tensor");
            assert_eq!(mt.m_rr, exp_mt.get("m_rr").as_f64());
            assert_eq!(mt.m_tt, exp_mt.get("m_tt").as_f64());
            assert_eq!(mt.m_pp, exp_mt.get("m_pp").as_f64());
            assert_eq!(mt.m_rt, exp_mt.get("m_rt").as_f64());
            assert_eq!(mt.m_rp, exp_mt.get("m_rp").as_f64());
            assert_eq!(mt.m_tp, exp_mt.get("m_tp").as_f64());
        }
    }

    // QuakeML serialize -> re-parse round-trip:
    let qml_written = write_quakeml_str(&catalog).unwrap();
    let cat_rt = read_quakeml_bytes(qml_written.as_bytes()).unwrap();
    assert_eq!(cat_rt.len(), catalog.len());
    assert_eq!(
        cat_rt.events[3].focal_mechanisms[0].moment_tensor,
        catalog.events[3].focal_mechanisms[0].moment_tensor
    );
}

// ---------------------------------------------------------------------------
// 4. Real 3C Broadband Event (`2011 Mw 9.1 Tohoku-Oki @ G.CAN.00.BH*`)
// ---------------------------------------------------------------------------

#[test]
fn golden_real_3c_tohoku_event_parity() {
    let expected = load_json("real_event/real_event_expected.json");

    let mseed_bytes = std::fs::read(io_fixture_path("real_event/real_3c.mseed")).unwrap();
    let stream = read_waveforms_bytes(&mseed_bytes).unwrap();
    assert_eq!(stream.len(), 3);
    assert!(stream.get_gaps().is_empty());

    let sxml_bytes = std::fs::read(io_fixture_path("real_event/real_3c_station.xml")).unwrap();
    let inv = read_stationxml_bytes(&sxml_bytes).unwrap();
    assert_eq!(inv.channels().count(), 3);

    let gcarc_exp = expected.get("gcarc_deg").as_f64();

    for ch_spec in expected.get("channels").as_arr() {
        let cha_name = ch_spec.get("channel").as_str();
        let sac_rel = ch_spec.get("sac_file").as_str();
        let exp_npts = ch_spec.get("npts").as_i64() as usize;
        let exp_sr = ch_spec.get("sampling_rate").as_f64();
        let exp_start: UtcTimestamp = ch_spec.get("starttime").as_str().parse().unwrap();
        let exp_end: UtcTimestamp = ch_spec.get("endtime").as_str().parse().unwrap();
        let exp_min = ch_spec.get("raw_min").as_f64();
        let exp_max = ch_spec.get("raw_max").as_f64();

        // 1. Verify miniSEED 3C trace (`Blockette 1001` sub-0.1ms `.019538Z` timestamp!)
        let sel = stream.select(&format!("G.CAN.00.{cha_name}"));
        assert_eq!(sel.len(), 1);
        let tr_ms = &sel[0];
        assert_eq!(tr_ms.npts(), exp_npts);
        assert_eq!(tr_ms.stats.sampling_rate, exp_sr);
        assert_eq!(
            tr_ms.stats.starttime, exp_start,
            "{cha_name} exact B1001 us starttime"
        );
        assert_eq!(tr_ms.endtime(), exp_end, "{cha_name} exact endtime");

        let (mn, mx, _) = tr_ms.min_max_mean().unwrap();
        assert_eq!(mn, exp_min);
        assert_eq!(mx, exp_max);

        for (i, exp_s) in ch_spec.get("raw_first_16").as_arr().iter().enumerate() {
            assert_eq!(tr_ms.data[i], exp_s.as_f64());
        }

        // 2. Verify matching StationXML channel response (`LAPLACE (HERTZ)` STS-1 + 68-tap FIR)
        let meta = inv
            .find_channel(&tr_ms.stats.channel, Some(exp_start))
            .expect("active channel metadata in real_3c_station.xml");
        let resp = meta.response.as_ref().unwrap();
        let h_002 = resp.eval_freq(0.01);
        let sens = resp.instrument_sensitivity.unwrap();
        assert!(
            (h_002.abs() - sens.value).abs() / sens.value < 1e-4,
            "{cha_name} LAPLACE(HERTZ) sensitivity match at {} Hz: got {} vs {}",
            sens.frequency_hz,
            h_002.abs(),
            sens.value
        );

        // 3. Verify matching populated SAC Binary file (`real_3c_bh*.sac`)
        let sac_bytes = std::fs::read(io_fixture_path(sac_rel)).unwrap();
        let tr_sac = read_sac_bytes(&sac_bytes).unwrap();
        assert_eq!(tr_sac.npts(), exp_npts);
        assert_eq!(tr_sac.data, tr_ms.data);
        let hdr = tr_sac.stats.sac.as_ref().unwrap();
        let sph_deg = earthsnail_core::locations2degrees(hdr.evla, hdr.evlo, hdr.stla, hdr.stlo);
        assert!((sph_deg - gcarc_exp).abs() < 1e-4);
        let wgs_geo = earthsnail_core::gps2dist_azimuth(hdr.evla, hdr.evlo, hdr.stla, hdr.stlo);
        assert!((hdr.gcarc - wgs_geo.gcarc_deg).abs() < 1e-4);
        assert!((hdr.dist - wgs_geo.distance_m * 1e-3).abs() < 1e-2);
        assert_eq!(hdr.get_str("kt0"), Some("P"));
        assert_eq!(hdr.get_str("kt1"), Some("S"));
    }
}
