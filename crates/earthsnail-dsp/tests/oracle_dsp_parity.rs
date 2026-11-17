//! Exhaustive differential parity verification against every entry in
//! `oracle_staging/fixtures/dsp.json` (ObsPy 1.4 / SciPy 1.15 / NumPy 2.x ground truth).
#![cfg(not(miri))]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use earthsnail_core::inventory::{
    ChannelResponse, CoefficientsStage, KinematicUnit, PolesZerosStage, PzTransferFunctionType,
    ResponseStage, StageGain,
};
use earthsnail_core::{ChannelId, Complex64, Trace, TraceStats, UtcTimestamp};
use earthsnail_dsp::{
    analytic_signal, classic_sta_lta, correlate, demean, design_iir_sos, detrend_linear,
    detrend_polynomial, envelope, flinn_polarization, get_nhnm, get_nlnm, instantaneous_phase,
    lanczos_resample, pk_baer, polarization_analysis_flinn, recursive_sta_lta, remove_response,
    rotate_ne_rt, rotate_zne_lqt, sac_cosine_taper, sosfilt, sosfilt_zerophase_obspy, sosfiltfilt,
    taper_window, trigger_onset, xcorr_max, BaerPickParams, BesselNorm, FilterBand, FilterFamily,
    LanczosWindow, ResponseRemovalConfig, TaperSide, TaperType, XcorrNormalize,
};

// ============================================================================
// Minimal Zero-Dependency Recursive-Descent JSON Parser for Oracle Fixtures
// ============================================================================

#[derive(Debug, Clone)]
enum JsonVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(BTreeMap<String, JsonVal>),
}

impl JsonVal {
    fn get(&self, key: &str) -> &JsonVal {
        match self {
            JsonVal::Obj(m) => m
                .get(key)
                .unwrap_or_else(|| panic!("missing JSON key `{key}`")),
            _ => panic!("expected JSON object when indexing `{key}`"),
        }
    }

    fn as_f64(&self) -> f64 {
        match self {
            JsonVal::Num(v) => *v,
            _ => panic!("expected JSON number, got {self:?}"),
        }
    }

    fn as_usize(&self) -> usize {
        self.as_f64().round() as usize
    }

    fn as_isize(&self) -> isize {
        self.as_f64().round() as isize
    }

    fn as_bool(&self) -> bool {
        match self {
            JsonVal::Bool(b) => *b,
            _ => panic!("expected JSON bool, got {self:?}"),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            JsonVal::Str(s) => s.as_str(),
            _ => panic!("expected JSON string, got {self:?}"),
        }
    }

    fn as_arr(&self) -> &[JsonVal] {
        match self {
            JsonVal::Arr(a) => a.as_slice(),
            _ => panic!("expected JSON array, got {self:?}"),
        }
    }

    fn as_f64_vec(&self) -> Vec<f64> {
        self.as_arr().iter().map(JsonVal::as_f64).collect()
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> u8 {
        self.skip_ws();
        self.bytes[self.pos]
    }

    fn parse_value(&mut self) -> JsonVal {
        match self.peek() {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => JsonVal::Str(self.parse_string()),
            b't' => {
                assert_eq!(&self.bytes[self.pos..self.pos + 4], b"true");
                self.pos += 4;
                JsonVal::Bool(true)
            }
            b'f' => {
                assert_eq!(&self.bytes[self.pos..self.pos + 5], b"false");
                self.pos += 5;
                JsonVal::Bool(false)
            }
            b'n' => {
                assert_eq!(&self.bytes[self.pos..self.pos + 4], b"null");
                self.pos += 4;
                JsonVal::Null
            }
            _ => self.parse_number(),
        }
    }

    fn parse_string(&mut self) -> String {
        self.skip_ws();
        assert_eq!(self.bytes[self.pos], b'"');
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return out;
            }
            if b == b'\\' {
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4]).unwrap();
                        let code = u32::from_str_radix(hex, 16).unwrap();
                        self.pos += 4;
                        out.push(char::from_u32(code).unwrap_or('?'));
                    }
                    other => out.push(other as char),
                }
            } else {
                out.push(b as char);
            }
        }
        panic!("unterminated JSON string");
    }

    fn parse_number(&mut self) -> JsonVal {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        JsonVal::Num(
            s.parse::<f64>()
                .unwrap_or_else(|e| panic!("bad float `{s}`: {e}")),
        )
    }

    fn parse_array(&mut self) -> JsonVal {
        self.skip_ws();
        assert_eq!(self.bytes[self.pos], b'[');
        self.pos += 1;
        let mut arr = Vec::new();
        if self.peek() == b']' {
            self.pos += 1;
            return JsonVal::Arr(arr);
        }
        loop {
            arr.push(self.parse_value());
            match self.peek() {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    break;
                }
                other => panic!("unexpected byte `{}` in array", other as char),
            }
        }
        JsonVal::Arr(arr)
    }

    fn parse_object(&mut self) -> JsonVal {
        self.skip_ws();
        assert_eq!(self.bytes[self.pos], b'{');
        self.pos += 1;
        let mut map = BTreeMap::new();
        if self.peek() == b'}' {
            self.pos += 1;
            return JsonVal::Obj(map);
        }
        loop {
            let key = self.parse_string();
            assert_eq!(self.peek(), b':');
            self.pos += 1;
            let val = self.parse_value();
            map.insert(key, val);
            match self.peek() {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    break;
                }
                other => panic!("unexpected byte `{}` in object", other as char),
            }
        }
        JsonVal::Obj(map)
    }
}

fn oracle_fixtures_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("EARTHSNAIL_ORACLE_FIXTURES") {
        return PathBuf::from(custom);
    }
    let repo_fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    if repo_fixtures.join("dsp.json").exists()
        && repo_fixtures.join("io/metadata_expected.json").exists()
    {
        return repo_fixtures;
    }
    PathBuf::from(
        r"C:\Users\x\.gemini\antigravity\brain\566443e0-66bc-4144-a363-aad888d88d34\scratch\oracle_staging\fixtures",
    )
}

fn load_json(rel_path: &str) -> JsonVal {
    let full = oracle_fixtures_dir().join(rel_path);
    let raw = fs::read_to_string(&full)
        .unwrap_or_else(|e| panic!("failed to read oracle fixture {}: {e}", full.display()));
    let mut parser = JsonParser::new(&raw);
    parser.parse_value()
}

fn assert_slice_close(label: &str, actual: &[f64], expected: &[f64], abs_tol: f64, rel_tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    let max_ref = expected
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let mut worst_abs = 0.0_f64;
    let mut worst_idx = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > worst_abs {
            worst_abs = diff;
            worst_idx = i;
        }
        let lim = abs_tol + rel_tol * e.abs().max(max_ref * 1e-3);
        assert!(
            diff <= lim,
            "{label}[{i}]: actual={a:.15e}, expected={e:.15e}, diff={diff:.3e} > lim={lim:.3e}"
        );
    }
    let _ = (worst_abs, worst_idx);
}

#[test]
fn test_oracle_dsp_complete_parity() {
    let dsp = load_json("dsp.json");
    let meta = load_json("io/metadata_expected.json");

    let clean_arrival = dsp
        .get("reference_waveforms")
        .get("clean_arrival_100hz")
        .as_f64_vec();
    let df_100 = dsp
        .get("reference_waveforms")
        .get("sampling_rate_hz")
        .as_f64();
    assert_eq!(clean_arrival.len(), 1000);
    assert_eq!(df_100, 100.0);

    // ------------------------------------------------------------------
    // 1. Detrend (`demean`, `linear`, `polynomial_order_3`)
    // ------------------------------------------------------------------
    let det = dsp.get("detrend");
    let det_in = det.get("input").as_f64_vec();
    assert_slice_close(
        "detrend.demean",
        &demean(&det_in),
        &det.get("demean").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "detrend.linear",
        &detrend_linear(&det_in),
        &det.get("linear").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "detrend.polynomial_order_3",
        &detrend_polynomial(&det_in, 3).unwrap(),
        &det.get("polynomial_order_3").as_f64_vec(),
        1e-12,
        1e-12,
    );

    // ------------------------------------------------------------------
    // 2. Tapers (`hann`, `cosine`, `sac_cosine`, `kaiser`)
    // ------------------------------------------------------------------
    let tapers = dsp.get("tapers");
    let taper_cases: [(&str, TaperType, f64); 5] = [
        ("hann_full", TaperType::Hann, 0.5),
        ("hann_0.05", TaperType::Hann, 0.05),
        ("cosine_0.05", TaperType::Cosine, 0.05),
        ("kaiser_full_beta_8.6", TaperType::Kaiser { beta: 8.6 }, 0.5),
        (
            "kaiser_0.05_beta_8.6",
            TaperType::Kaiser { beta: 8.6 },
            0.05,
        ),
    ];
    for (name, ttype, max_pct) in taper_cases {
        let entry = tapers.get(name);
        let win = taper_window(clean_arrival.len(), max_pct, ttype, TaperSide::Both).unwrap();
        let tapered: Vec<f64> = clean_arrival
            .iter()
            .zip(win.iter())
            .map(|(&x, &w)| x * w)
            .collect();
        assert_slice_close(
            &format!("tapers.{name}.window"),
            &win,
            &entry.get("window").as_f64_vec(),
            1e-12,
            1e-12,
        );
        assert_slice_close(
            &format!("tapers.{name}.tapered"),
            &tapered,
            &entry.get("tapered").as_f64_vec(),
            1e-12,
            1e-12,
        );
    }

    let sac_entry = tapers.get("sac_cosine_0.05_each_side");
    let sac_win = sac_cosine_taper(clean_arrival.len(), 0.05).unwrap();
    let sac_tap: Vec<f64> = clean_arrival
        .iter()
        .zip(sac_win.iter())
        .map(|(&x, &w)| x * w)
        .collect();
    assert_slice_close(
        "tapers.sac_cosine_0.05_each_side.window",
        &sac_win,
        &sac_entry.get("window").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "tapers.sac_cosine_0.05_each_side.tapered",
        &sac_tap,
        &sac_entry.get("tapered").as_f64_vec(),
        1e-12,
        1e-12,
    );

    // ------------------------------------------------------------------
    // 3. IIR SOS Filters (`butter`, `cheby1`, `bessel` x `bp`, `lp`, `hp`, `bs`)
    // ------------------------------------------------------------------
    let filters = dsp.get("filters");
    let fam_specs = [
        ("butter", FilterFamily::Butterworth),
        ("cheby1", FilterFamily::Chebyshev1 { rp_db: 1.0 }),
        (
            "bessel",
            FilterFamily::Bessel {
                norm: BesselNorm::Phase,
            },
        ),
    ];
    let mode_specs = [
        ("bandpass", FilterBand::Bandpass(0.5, 5.0)),
        ("lowpass", FilterBand::Lowpass(5.0)),
        ("highpass", FilterBand::Highpass(0.5)),
        ("bandstop", FilterBand::Bandstop(0.5, 5.0)),
    ];

    for (fam_key, fam) in fam_specs {
        let fam_obj = filters.get(fam_key);
        for (mode_key, band) in mode_specs {
            let entry = fam_obj.get(mode_key);
            let corners = entry.get("corners").as_usize();
            let fs = entry.get("sampling_rate_hz").as_f64();
            let sos = design_iir_sos(corners, band, fs, fam).unwrap();

            let exp_rows = entry.get("sos_sections").as_arr();
            let act_rows = sos.to_rows();
            assert_eq!(
                act_rows.len(),
                exp_rows.len(),
                "SOS row count mismatch for {fam_key}.{mode_key}"
            );
            for (r_idx, (act_r, exp_r_val)) in act_rows.iter().zip(exp_rows.iter()).enumerate() {
                let exp_r = exp_r_val.as_f64_vec();
                assert_slice_close(
                    &format!("filters.{fam_key}.{mode_key}.sos[{r_idx}]"),
                    act_r,
                    &exp_r,
                    1e-11,
                    1e-11,
                );
            }

            let y_causal = sosfilt(&sos, &clean_arrival);
            let y_zp_obspy = sosfilt_zerophase_obspy(&sos, &clean_arrival);
            let y_zp_filtfilt = sosfiltfilt(&sos, &clean_arrival).unwrap();

            assert_slice_close(
                &format!("filters.{fam_key}.{mode_key}.zerophase_false"),
                &y_causal,
                &entry.get("zerophase_false").as_f64_vec(),
                1e-10,
                1e-10,
            );
            assert_slice_close(
                &format!("filters.{fam_key}.{mode_key}.zerophase_true"),
                &y_zp_obspy,
                &entry.get("zerophase_true").as_f64_vec(),
                1e-10,
                1e-10,
            );
            assert_slice_close(
                &format!("filters.{fam_key}.{mode_key}.zerophase_sosfiltfilt"),
                &y_zp_filtfilt,
                &entry.get("zerophase_sosfiltfilt").as_f64_vec(),
                1e-10,
                1e-10,
            );
        }
    }

    // ------------------------------------------------------------------
    // 4. Instrument Response Deconvolution (`StationXML` & `SACPZ` x `VEL`, `DISP`, `ACC`)
    // ------------------------------------------------------------------
    let rr = dsp.get("remove_response");
    let raw_counts = rr.get("input_counts").as_f64_vec();
    let df_resp = rr.get("sampling_rate_hz").as_f64();
    let pf_vec = rr.get("pre_filt_hz").as_f64_vec();
    let pre_filt = [pf_vec[0], pf_vec[1], pf_vec[2], pf_vec[3]];
    let water_level_db = rr.get("water_level_db").as_f64();
    let taper_fraction = rr.get("taper_fraction").as_f64();

    // Build exact StationXML ChannelResponse from `metadata_expected.json` + Stage 3 Decimation/Correction `0.43046` s
    let sxml = meta.get("stationxml");
    let stages_json = sxml.get("stages").as_arr();
    let mut xml_stages = Vec::new();
    for stg in stages_json {
        let seq = stg.get("stage_sequence_number").as_usize();
        let stype = stg.get("stage_type").as_str();
        let gain = stg.get("stage_gain").as_f64();
        let gain_freq = stg.get("stage_gain_frequency").as_f64();
        if stype == "PolesZerosResponseStage" {
            let zeros: Vec<Complex64> = stg
                .get("zeros")
                .as_arr()
                .iter()
                .map(|z| Complex64::new(z.get("real").as_f64(), z.get("imag").as_f64()))
                .collect();
            let poles: Vec<Complex64> = stg
                .get("poles")
                .as_arr()
                .iter()
                .map(|p| Complex64::new(p.get("real").as_f64(), p.get("imag").as_f64()))
                .collect();
            xml_stages.push(ResponseStage::PolesZeros(PolesZerosStage {
                pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
                a0: stg.get("normalization_factor").as_f64(),
                normalization_freq: stg.get("normalization_frequency").as_f64(),
                zeros,
                poles,
                stage_gain: gain,
                input_sample_rate_hz: Some(stg.get("decimation_input_sample_rate").as_f64()),
            }));
        } else if stype == "CoefficientsTypeResponseStage" {
            let num = stg.get("numerator").as_f64_vec();
            let den = stg.get("denominator").as_f64_vec();
            let corr_s = if seq == 3 && !num.is_empty() {
                0.43046
            } else {
                0.0
            };
            xml_stages.push(ResponseStage::Coefficients(CoefficientsStage {
                numerator: num,
                denominator: den,
                input_sample_rate_hz: stg.get("decimation_input_sample_rate").as_f64(),
                correction_s: corr_s,
                stage_gain: gain,
                gain_frequency_hz: gain_freq,
            }));
        }
    }
    let resp_stationxml = ChannelResponse {
        input_units: KinematicUnit::from_unit_str(sxml.get("input_units").as_str()),
        output_units: sxml.get("output_units").as_str().to_string(),
        instrument_sensitivity: Some(StageGain {
            value: sxml.get("overall_sensitivity").as_f64(),
            frequency_hz: sxml.get("sensitivity_frequency_hz").as_f64(),
        }),
        stages: xml_stages,
    };

    // Build SACPZ ChannelResponse from `metadata_expected.json`
    let sacpz = meta.get("sacpz");
    let sacpz_zeros: Vec<Complex64> = sacpz
        .get("zeros_disp_rad_s")
        .as_arr()
        .iter()
        .map(|z| Complex64::new(z.get("real").as_f64(), z.get("imag").as_f64()))
        .collect();
    let sacpz_poles: Vec<Complex64> = sacpz
        .get("poles_rad_s")
        .as_arr()
        .iter()
        .map(|p| Complex64::new(p.get("real").as_f64(), p.get("imag").as_f64()))
        .collect();
    let resp_sacpz =
        ChannelResponse::from_sacpz(sacpz_zeros, sacpz_poles, sacpz.get("constant").as_f64());

    for (unit_key, kin_unit) in [
        ("VEL", KinematicUnit::Velocity),
        ("DISP", KinematicUnit::Displacement),
        ("ACC", KinematicUnit::Acceleration),
    ] {
        let cfg = ResponseRemovalConfig {
            output: kin_unit,
            pre_filt_hz: Some(pre_filt),
            water_level_db: Some(water_level_db),
            zero_mean: rr.get("zero_mean").as_bool(),
            taper: rr.get("taper").as_bool(),
            taper_fraction,
            nfft_pow2: false,
            rescale_to_overall_sensitivity: false,
        };
        let out_xml = remove_response(&raw_counts, df_resp, &resp_stationxml, &cfg).unwrap();
        let exp_xml = rr.get("stationxml_outputs").get(unit_key).as_f64_vec();
        assert_slice_close(
            &format!("remove_response.stationxml.{unit_key}"),
            &out_xml,
            &exp_xml,
            1e-15,
            1e-9,
        );

        let out_pz = remove_response(&raw_counts, df_resp, &resp_sacpz, &cfg).unwrap();
        let exp_pz = rr.get("sacpz_outputs").get(unit_key).as_f64_vec();
        assert_slice_close(
            &format!("remove_response.sacpz.{unit_key}"),
            &out_pz,
            &exp_pz,
            1e-15,
            1e-9,
        );
    }

    // Also verify `rescale_to_overall_sensitivity: true` rescales deconvolution by `|H(f_s)| / S_d`
    let cfg_rescaled = ResponseRemovalConfig {
        output: KinematicUnit::Velocity,
        pre_filt_hz: Some(pre_filt),
        water_level_db: Some(water_level_db),
        zero_mean: true,
        taper: true,
        taper_fraction,
        nfft_pow2: false,
        rescale_to_overall_sensitivity: true,
    };
    let out_rescaled =
        remove_response(&raw_counts, df_resp, &resp_stationxml, &cfg_rescaled).unwrap();
    let exp_vel_xml = rr.get("stationxml_outputs").get("VEL").as_f64_vec();
    let expected_scale = resp_stationxml.eval_freq(0.02).abs() / 1_974_680_000.0;
    let rescaled_ref: Vec<f64> = exp_vel_xml.iter().map(|&v| v * expected_scale).collect();
    assert_slice_close(
        "remove_response.stationxml.VEL.rescaled_overall_sensitivity",
        &out_rescaled,
        &rescaled_ref,
        1e-15,
        1e-9,
    );

    // ------------------------------------------------------------------
    // 5. Hilbert Transform (`envelope`, `quadrature_imag`, `instantaneous_phase_rad`)
    // ------------------------------------------------------------------
    let hilb = dsp.get("hilbert");
    let z_an = analytic_signal(&clean_arrival).unwrap();
    let quad_im: Vec<f64> = z_an.iter().map(|c| c.im).collect();
    let env_act = envelope(&clean_arrival).unwrap();
    let phase_act = instantaneous_phase(&clean_arrival).unwrap();

    assert_slice_close(
        "hilbert.envelope",
        &env_act,
        &hilb.get("envelope").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "hilbert.quadrature_imag",
        &quad_im,
        &hilb.get("quadrature_imag").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "hilbert.instantaneous_phase_rad",
        &phase_act,
        &hilb.get("instantaneous_phase_rad").as_f64_vec(),
        1e-11,
        1e-11,
    );

    // ------------------------------------------------------------------
    // 6. Normalized Cross-Correlation (`correlate` & `xcorr_max`)
    // ------------------------------------------------------------------
    let xc = dsp.get("cross_correlation");
    let sig_a = xc.get("signal_a").as_f64_vec();
    let sig_b = xc.get("signal_b").as_f64_vec();
    let max_shift = xc.get("max_shift_samples").as_usize();
    let cc_act = correlate(&sig_a, &sig_b, max_shift, true, XcorrNormalize::Naive).unwrap();
    let (shift_act, val_act) = xcorr_max(&cc_act, false).unwrap();

    assert_eq!(
        shift_act.round() as isize,
        xc.get("expected_shift_samples").as_isize()
    );
    assert!((val_act - xc.get("expected_max_value").as_f64()).abs() < 1e-12);
    assert_slice_close(
        "cross_correlation.cc_curve",
        &cc_act,
        &xc.get("cc_curve").as_f64_vec(),
        1e-12,
        1e-12,
    );

    // ------------------------------------------------------------------
    // 7. Classic & Recursive STA/LTA + Hysteresis `trigger_onset`
    // ------------------------------------------------------------------
    let sl = dsp.get("sta_lta");
    let nsta = sl.get("nsta").as_usize();
    let nlta = sl.get("nlta").as_usize();
    let thr_on = sl.get("thr_on").as_f64();
    let thr_off = sl.get("thr_off").as_f64();

    let cft_classic = classic_sta_lta(&clean_arrival, nsta, nlta).unwrap();
    let cft_recursive = recursive_sta_lta(&clean_arrival, nsta, nlta).unwrap();
    assert_slice_close(
        "sta_lta.classic_cft",
        &cft_classic,
        &sl.get("classic_cft").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "sta_lta.recursive_cft",
        &cft_recursive,
        &sl.get("recursive_cft").as_f64_vec(),
        1e-12,
        1e-12,
    );

    let trig_classic = trigger_onset(&cft_classic, thr_on, thr_off);
    let exp_trig_classic: Vec<[usize; 2]> = sl
        .get("classic_triggers")
        .as_arr()
        .iter()
        .map(|p| {
            let v = p.as_arr();
            [v[0].as_usize(), v[1].as_usize()]
        })
        .collect();
    assert_eq!(trig_classic, exp_trig_classic);

    let trig_recursive = trigger_onset(&cft_recursive, thr_on, thr_off);
    let exp_trig_recursive: Vec<[usize; 2]> = sl
        .get("recursive_triggers")
        .as_arr()
        .iter()
        .map(|p| {
            let v = p.as_arr();
            [v[0].as_usize(), v[1].as_usize()]
        })
        .collect();
    assert_eq!(trig_recursive, exp_trig_recursive);

    // ------------------------------------------------------------------
    // 8. Baer–Kradolfer Picker (`pk_baer` on `rjob_ehz_1000` & `clean_arrival_100hz`)
    // ------------------------------------------------------------------
    let baer = dsp.get("baer_picker");
    let bp_json = baer.get("params");
    let bparams = BaerPickParams {
        samp_int: bp_json.get("samp_int").as_usize(),
        tdownmax: bp_json.get("tdownmax").as_usize(),
        tupevent: bp_json.get("tupevent").as_usize(),
        thr1: bp_json.get("thr1").as_f64(),
        thr2: bp_json.get("thr2").as_f64(),
        preset_len: bp_json.get("preset_len").as_usize(),
        p_dur: bp_json.get("p_dur").as_usize(),
    };

    let rjob_entry = baer.get("rjob_ehz_1000");
    let rjob_in = rjob_entry.get("input_f32").as_f64_vec();
    let res_rjob = pk_baer(&rjob_in, &bparams).unwrap();
    assert_eq!(
        res_rjob.obspy_pick_sample,
        rjob_entry.get("pick_sample").as_usize()
    );
    assert_eq!(
        res_rjob.first_motion,
        rjob_entry.get("first_motion").as_str()
    );
    assert_slice_close(
        "baer_picker.rjob_ehz_1000.cf",
        &res_rjob.characteristic_function,
        &rjob_entry.get("characteristic_function").as_f64_vec(),
        1e-4,
        2e-5,
    );

    let ca_baer_entry = baer.get("clean_arrival_100hz");
    let res_ca_baer = pk_baer(&clean_arrival, &bparams).unwrap();
    assert_eq!(
        res_ca_baer.obspy_pick_sample,
        ca_baer_entry.get("pick_sample").as_usize()
    );
    assert_eq!(
        res_ca_baer.first_motion,
        ca_baer_entry.get("first_motion").as_str()
    );
    assert_slice_close(
        "baer_picker.clean_arrival_100hz.cf",
        &res_ca_baer.characteristic_function,
        &ca_baer_entry.get("characteristic_function").as_f64_vec(),
        1e-4,
        2e-5,
    );

    // ------------------------------------------------------------------
    // 9. 3C Rotations & Flinn Polarization (Window & Sliding Window)
    // ------------------------------------------------------------------
    let rp = dsp.get("rotation_and_polarization");
    let baz = rp.get("back_azimuth_deg").as_f64();
    let inc = rp.get("incidence_deg").as_f64();
    let z_3c = rp.get("input_z").as_f64_vec();
    let n_3c = rp.get("input_n").as_f64_vec();
    let e_3c = rp.get("input_e").as_f64_vec();

    let (rot_r, rot_t) = rotate_ne_rt(&n_3c, &e_3c, baz).unwrap();
    assert_slice_close(
        "rotate_ne_rt.radial",
        &rot_r,
        &rp.get("rotate_ne_rt").get("radial").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "rotate_ne_rt.transverse",
        &rot_t,
        &rp.get("rotate_ne_rt").get("transverse").as_f64_vec(),
        1e-12,
        1e-12,
    );

    let (rot_l, rot_q, rot_lqt_t) = rotate_zne_lqt(&z_3c, &n_3c, &e_3c, baz, inc).unwrap();
    assert_slice_close(
        "rotate_zne_lqt.l",
        &rot_l,
        &rp.get("rotate_zne_lqt").get("l").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "rotate_zne_lqt.q",
        &rot_q,
        &rp.get("rotate_zne_lqt").get("q").as_f64_vec(),
        1e-12,
        1e-12,
    );
    assert_slice_close(
        "rotate_zne_lqt.t",
        &rot_lqt_t,
        &rp.get("rotate_zne_lqt").get("t").as_f64_vec(),
        1e-12,
        1e-12,
    );

    let fp = rp.get("flinn_p_window");
    let s0 = fp.get("start_sample").as_usize();
    let s1 = fp.get("end_sample_exclusive").as_usize();
    let pm = flinn_polarization(
        &z_3c[s0..s1],
        &n_3c[s0..s1],
        &e_3c[s0..s1],
        fp.get("noise_thres").as_f64(),
    )
    .unwrap();
    assert!((pm.azimuth_deg - fp.get("azimuth_deg").as_f64()).abs() < 1e-10);
    assert!((pm.incidence_deg - fp.get("incidence_deg").as_f64()).abs() < 1e-10);
    assert!((pm.rectilinearity - fp.get("rectilinearity").as_f64()).abs() < 1e-12);
    assert!((pm.planarity - fp.get("planarity").as_f64()).abs() < 1e-12);

    let pa = rp.get("polarization_analysis_flinn");
    let t0_pol = UtcTimestamp::from_ymd_hms_nano(2025, 1, 1, 0, 0, 0, 0).unwrap();
    let t0_epoch_s = t0_pol.as_epoch_nanos() as f64 * 1e-9;
    let tr_z = Trace::new(
        TraceStats::new(ChannelId::new("", "", "", "BHZ"), t0_pol, df_100),
        z_3c,
    );
    let tr_n = Trace::new(
        TraceStats::new(ChannelId::new("", "", "", "BHN"), t0_pol, df_100),
        n_3c,
    );
    let tr_e = Trace::new(
        TraceStats::new(ChannelId::new("", "", "", "BHE"), t0_pol, df_100),
        e_3c,
    );
    let slide = polarization_analysis_flinn(
        &tr_z,
        &tr_n,
        &tr_e,
        pa.get("win_len_s").as_f64(),
        pa.get("win_frac").as_f64(),
        t0_epoch_s + 0.5,
        t0_epoch_s + 9.0,
        0.0,
    )
    .unwrap();
    assert_slice_close(
        "polarization_analysis.timestamps",
        &slide.timestamps_epoch_s,
        &pa.get("timestamps_epoch_s").as_f64_vec(),
        1e-6,
        1e-12,
    );
    assert_slice_close(
        "polarization_analysis.azimuth_deg",
        &slide.azimuth_deg,
        &pa.get("azimuth_deg").as_f64_vec(),
        1e-9,
        1e-9,
    );
    assert_slice_close(
        "polarization_analysis.incidence_deg",
        &slide.incidence_deg,
        &pa.get("incidence_deg").as_f64_vec(),
        1e-9,
        1e-9,
    );
    assert_slice_close(
        "polarization_analysis.rectilinearity",
        &slide.rectilinearity,
        &pa.get("rectilinearity").as_f64_vec(),
        1e-11,
        1e-11,
    );
    assert_slice_close(
        "polarization_analysis.planarity",
        &slide.planarity,
        &pa.get("planarity").as_f64_vec(),
        1e-11,
        1e-11,
    );

    // ------------------------------------------------------------------
    // 10. Peterson (1993) NLNM & NHNM Global Noise Models
    // ------------------------------------------------------------------
    let pnm = dsp.get("peterson_noise_models");
    let (nlnm_p, nlnm_db) = get_nlnm();
    let (nhnm_p, nhnm_db) = get_nhnm();
    assert_slice_close(
        "peterson.nlnm.periods_s",
        &nlnm_p,
        &pnm.get("nlnm").get("periods_s").as_f64_vec(),
        1e-9,
        1e-5,
    );
    assert_slice_close(
        "peterson.nlnm.power_db",
        &nlnm_db,
        &pnm.get("nlnm").get("power_db").as_f64_vec(),
        1e-3,
        1e-5,
    );
    assert_slice_close(
        "peterson.nhnm.periods_s",
        &nhnm_p,
        &pnm.get("nhnm").get("periods_s").as_f64_vec(),
        1e-9,
        1e-5,
    );
    assert_slice_close(
        "peterson.nhnm.power_db",
        &nhnm_db,
        &pnm.get("nhnm").get("power_db").as_f64_vec(),
        1e-3,
        1e-5,
    );

    // ------------------------------------------------------------------
    // 11. Lanczos Resampling (`a = 20`, downsample 100->40 Hz & upsample 40->100 Hz)
    // ------------------------------------------------------------------
    let lanc = dsp.get("lanczos_resampling");
    let a_lanc = lanc.get("a").as_usize();

    let down_spec = lanc.get("downsample_100hz_to_40hz");
    let down_act = lanczos_resample(
        &down_spec.get("input").as_f64_vec(),
        down_spec.get("old_start").as_f64(),
        down_spec.get("old_dt").as_f64(),
        down_spec.get("new_start").as_f64(),
        down_spec.get("new_dt").as_f64(),
        down_spec.get("new_npts").as_usize(),
        a_lanc,
        LanczosWindow::Lanczos,
    )
    .unwrap();
    assert_slice_close(
        "lanczos.downsample_100hz_to_40hz",
        &down_act,
        &down_spec.get("output").as_f64_vec(),
        1e-12,
        1e-12,
    );

    let up_spec = lanc.get("upsample_40hz_to_100hz");
    let up_act = lanczos_resample(
        &up_spec.get("input").as_f64_vec(),
        up_spec.get("old_start").as_f64(),
        up_spec.get("old_dt").as_f64(),
        up_spec.get("new_start").as_f64(),
        up_spec.get("new_dt").as_f64(),
        up_spec.get("new_npts").as_usize(),
        a_lanc,
        LanczosWindow::Lanczos,
    )
    .unwrap();
    assert_slice_close(
        "lanczos.upsample_40hz_to_100hz",
        &up_act,
        &up_spec.get("output").as_f64_vec(),
        1e-12,
        1e-12,
    );
}
