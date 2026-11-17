//! End-to-end integration, real Tohoku-Oki 3C `SAC>` macro workflow, >200-trace ceiling test,
//! all-6-SVG-plotter audit, full structured CLI subcommand suite, subprocess binary checks,
//! and `proptest` adversarial fuzzing for `earthsnail-sac`.

#![forbid(unsafe_code)]
#![cfg(not(miri))]

use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use core::f64::consts::TAU;
use earthsnail_core::{
    direct_geodesic, math, ChannelId, MomentTensor6, NodalPlane, SacEndian, Stream, Trace,
    TraceStats, UtcTimestamp,
};
use earthsnail_dsp::{Ppsd, PpsdConfig};
use earthsnail_io::{read_quakeml_file, read_sac_file, read_stationxml_file, write_sac_file};
use earthsnail_sac::{
    plot_beachball_svg, plot_ppsd_svg, plot_record_section_svg, plot_spectrogram_svg,
    plot_taup_paths_svg, plot_waveforms_svg, run_cli_args, run_es_sac_args, BeachballPlotOptions,
    DistanceUnit, FrequencyScale, PpsdPlotOptions, RecordSectionOptions, SacSession,
    SpectrogramMethod, SpectrogramOptions, TaupPathsPlotOptions, TimeAxisMode, WaveformLayout,
    WaveformPlotOptions,
};
use earthsnail_taup::TauModel;
use proptest::prelude::*;

/// Portable 3-tier fixture directory resolver (`EARTHSNAIL_ORACLE_FIXTURES` -> `../../tests/fixtures`
/// -> `oracle_staging/fixtures`), matching `earthsnail-io`, `earthsnail-dsp`, and `earthsnail-taup`.
fn oracle_fixtures_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("EARTHSNAIL_ORACLE_FIXTURES") {
        let p = PathBuf::from(custom);
        if p.exists() {
            return p;
        }
    }
    let repo_frozen = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    if repo_frozen.exists() {
        return repo_frozen;
    }
    let staging = PathBuf::from(
        r"C:\Users\x\.gemini\antigravity\brain\566443e0-66bc-4144-a363-aad888d88d34\scratch\oracle_staging\fixtures",
    );
    assert!(
        staging.exists(),
        "could not locate oracle fixtures in EARTHSNAIL_ORACLE_FIXTURES, ../../tests/fixtures, or scratch staging"
    );
    staging
}

struct TempWorkDir {
    path: PathBuf,
}

impl TempWorkDir {
    fn new(tag: &str) -> Self {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("earthsnail_sac_test_{tag}_{pid}_{nanos}"));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self { path: dir }
    }
}

impl Drop for TempWorkDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn assert_well_formed_svg(svg: &str, label: &str) {
    assert!(
        svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""),
        "{label}: missing <svg> header"
    );
    assert!(
        svg.trim_end().ends_with("</svg>"),
        "{label}: missing closing </svg>"
    );
    assert!(
        svg.len() > 400,
        "{label}: suspiciously tiny SVG ({})",
        svg.len()
    );
    let open_g = svg.matches("<g").count();
    let close_g = svg.matches("</g>").count();
    assert_eq!(open_g, close_g, "{label}: unbalanced <g> / </g> groups");
}

fn cli_vec(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| String::from(*s)).collect()
}

// ============================================================================
// 1. Real 2011 Mw 9.1 Tohoku-Oki 3C `SAC>` End-to-End Macro Workflow
// ============================================================================

#[test]
fn sac_tohoku_3c_full_workflow_rotation_taup_apk_and_svg() {
    let fixtures = oracle_fixtures_dir();
    let real_dir = fixtures.join("io").join("real_event");
    let tmp = TempWorkDir::new("tohoku_3c");

    for fname in [
        "real_3c_bhe.sac",
        "real_3c_bhn.sac",
        "real_3c_bhz.sac",
        "real_3c_station.xml",
    ] {
        fs::copy(real_dir.join(fname), tmp.path.join(fname)).unwrap();
    }

    let mut sess = SacSession::with_working_dir(&tmp.path);

    // Execute exact required SAC> pipeline from #18:
    let script = "\
        * End-to-end 2011 Mw 9.1 Tohoku-Oki 3C workflow @ G.CAN.00.BH*\n\
        r real_3c_bh*.sac; rmean; rtrend; taper\n\
        transfer from stationxml real_3c_station.xml to vel freqlimits 0.005 0.01 8 9.5\n\
        bp co 0.02 2.0 n 4 p 2\n\
        rot to gcp\n\
        taup model iasp91 phases P S PP PcP SKS\n\
        apk morse\n\
        p1 out.svg\n";
    sess.execute_script(script, &[]).unwrap();

    // Verify 3 traces in memory rotated to BHR, BHT, BHZ:
    assert_eq!(sess.traces.len(), 3);
    let comps: Vec<String> = sess
        .traces
        .iter()
        .map(|t| t.stats.channel.channel.clone())
        .collect();
    assert_eq!(comps, vec!["BHR", "BHT", "BHZ"]);

    // Verify TauP stamped all 5 phases (`P`, `PcP`, `PP`, `SKS`, `S`) into T0..T4 & KT0..KT4:
    let tm_ref = TauModel::iasp91().unwrap();
    for tr in &sess.traces {
        let sac = tr.stats.sac.as_ref().expect("SAC header present");
        let o_val = sac.get_float("o").expect("origin marker O defined");
        let gcarc = sac.get_float("gcarc").expect("GCARC defined");
        let evdp = sac.get_float("evdp").expect("EVDP defined");
        let t0_val = sac.get_float("t0").expect("T0 stamped");
        assert_eq!(sac.get_str("kt0"), Some("P"));
        let expected_p = tm_ref.get_travel_times(evdp, gcarc, &["P"]).unwrap()[0].time_s;
        let p_tt = t0_val - o_val;
        assert!(
            (p_tt - expected_p).abs() < 1e-6,
            "unexpected P travel time {p_tt} vs {expected_p}"
        );

        let mut stamped_phases = Vec::new();
        for k in 0..5 {
            assert!(sac.get_float(&format!("t{k}")).is_some());
            stamped_phases.push(String::from(sac.get_str(&format!("kt{k}")).unwrap()));
        }
        assert!(stamped_phases.contains(&String::from("P")));
        assert!(stamped_phases.contains(&String::from("PcP")));
        assert!(stamped_phases.contains(&String::from("PP")));
        assert!(stamped_phases.contains(&String::from("SKS")));
        assert!(stamped_phases.contains(&String::from("S")));

        // Verify `apk morse` stamped pick into A & KA:
        assert!(
            sac.get_float("a").is_some(),
            "expected A pick on {}",
            tr.stats.channel
        );
        assert_eq!(sac.get_str("ka"), Some("MORSE"));
    }

    // Verify `p1 out.svg` wrote a valid publication SVG containing phase labels & channel IDs:
    let svg_txt = fs::read_to_string(tmp.path.join("out.svg")).unwrap();
    assert_well_formed_svg(&svg_txt, "SAC p1 out.svg");
    assert!(svg_txt.contains("G.CAN.00.BHR"));
    assert!(svg_txt.contains("G.CAN.00.BHT"));
    assert!(svg_txt.contains("G.CAN.00.BHZ"));
    assert!(svg_txt.contains("MORSE"));
}

// ============================================================================
// 2. Killing SAC's 200-File Ceiling + REPL, Macros, Blackboard, Undo & All 6 SVGs
// ============================================================================

#[test]
fn sac_exceeds_200_file_limit_and_renders_all_six_svg_types() {
    let fixtures = oracle_fixtures_dir();
    let tmp = TempWorkDir::new("over_200");

    // Generate 240 distinct SAC files on disk and read them all at once + `read more` 15 more = 255 traces!
    for idx in 0..240 {
        let t0 = UtcTimestamp::from_ymd_hms_nano(2025, 1, 1, 0, 0, 0, 0).unwrap();
        let sta = format!("S{idx:03}");
        let mut stats = TraceStats::new(ChannelId::new("XX", &sta, "00", "BHZ"), t0, 20.0);
        let samples: Vec<f64> = (0..64)
            .map(|k| math::sin(TAU * 0.5 * (k as f64) * 0.05 + (idx as f64) * 0.1))
            .collect();
        stats.sync_to_sac_header(&samples);
        if let Some(ref mut sac) = stats.sac {
            let _ = sac.set_float("gcarc", 10.0 + (idx as f64) * 0.5);
            let _ = sac.set_float("evdp", 25.0);
            let _ = sac.set_float("o", 0.0);
        }
        let tr = Trace::new(stats, samples);
        write_sac_file(
            tmp.path.join(format!("wave_{idx:03}.sac")),
            &tr,
            SacEndian::Little,
        )
        .unwrap();
    }

    let macro_path = tmp.path.join("batch.m");
    fs::write(
        &macro_path,
        "* Macro exercising $1, setbb, getbb, do-loop, and >200 traces\n\
         setbb GAIN $1\n\
         r wave_*.sac\n\
         r more wave_00*.sac\n\
         mul $GAIN\n\
         getbb GAIN\n",
    )
    .unwrap();

    let mut sess = SacSession::with_working_dir(&tmp.path);
    sess.execute_line("m batch.m 2.5").unwrap();
    // 240 + 10 (`wave_000..wave_009`) = 250 traces > 200!
    assert_eq!(sess.traces.len(), 250);
    assert!(sess.take_output().contains("GAIN = 2.5"));

    // Interactive `run_repl` check with multi-line `do ... enddo`, `lh`, `cut`, `interpolate`, and `quit`:
    let repl_input = b"fg sine 2.0 delta 0.02 npts 200\n\
                       do k = 1 to 3\n\
                       add 0.5\n\
                       enddo\n\
                       interpolate delta 0.01\n\
                       lh npts delta depmen\n\
                       quit\n";
    let mut repl_out = Vec::new();
    let mut repl_sess = SacSession::with_working_dir(&tmp.path);
    repl_sess
        .run_repl(Cursor::new(&repl_input[..]), &mut repl_out)
        .unwrap();
    let repl_txt = String::from_utf8(repl_out).unwrap();
    assert!(repl_txt.contains("SAC>"));
    assert!(repl_txt.contains("NPTS"));
    assert_eq!(repl_sess.traces[0].npts(), 399);

    // Exercise all 6 SVG plotters on real oracle fixtures:
    let tr_le = read_sac_file(fixtures.join("io/test_le.sac")).unwrap();
    let inv = read_stationxml_file(fixtures.join("io/iu_anmo_bhz.xml")).unwrap();
    let qml = read_quakeml_file(fixtures.join("io/events.qml")).unwrap();

    let st_single = Stream::from_traces(vec![tr_le.clone()]);
    assert_well_formed_svg(
        &plot_waveforms_svg(&st_single, &WaveformPlotOptions::default()),
        "plot_waveforms_svg stacked",
    );
    assert_well_formed_svg(
        &plot_waveforms_svg(
            &st_single,
            &WaveformPlotOptions {
                layout: WaveformLayout::Overlaid,
                time_axis: TimeAxisMode::UtcClock,
                ..WaveformPlotOptions::default()
            },
        ),
        "plot_waveforms_svg overlaid",
    );

    let tm = TauModel::iasp91().unwrap();
    let curves = tm
        .get_travel_time_curves(29.0, &["P", "S", "PP"], 48)
        .unwrap();
    assert_well_formed_svg(
        &plot_record_section_svg(
            &Stream::from_traces(sess.traces[..12].to_vec()),
            &curves,
            &RecordSectionOptions {
                distance_unit: DistanceUnit::Degrees,
                reduction_velocity_km_s: Some(8.0),
                ..RecordSectionOptions::default()
            },
        ),
        "plot_record_section_svg",
    );

    assert_well_formed_svg(
        &plot_spectrogram_svg(
            &tr_le,
            &SpectrogramOptions {
                method: SpectrogramMethod::MorletCwt { omega0: 6.0 },
                freq_scale: FrequencyScale::Log10,
                ..SpectrogramOptions::default()
            },
        ),
        "plot_spectrogram_svg CWT",
    );
    assert_well_formed_svg(
        &plot_spectrogram_svg(
            &tr_le,
            &SpectrogramOptions {
                method: SpectrogramMethod::Stft { win_samples: 64 },
                freq_scale: FrequencyScale::Linear,
                ..SpectrogramOptions::default()
            },
        ),
        "plot_spectrogram_svg STFT",
    );

    let resp = inv.channels().next().unwrap().response.as_ref().unwrap();
    let mut ppsd = Ppsd::new(
        tr_le.stats.sampling_rate,
        resp,
        PpsdConfig {
            ppsd_length_s: tr_le.duration_s() * 0.5,
            overlap: 0.5,
            ..PpsdConfig::default()
        },
    )
    .unwrap();
    ppsd.add_trace(&tr_le).unwrap();
    assert_well_formed_svg(
        &plot_ppsd_svg(&ppsd, &PpsdPlotOptions::default()),
        "plot_ppsd_svg",
    );

    let paths = tm
        .get_ray_paths(120.0, 85.0, &["P", "pP", "PP", "PcP", "S", "SKS", "PKiKP"])
        .unwrap();
    assert_well_formed_svg(
        &plot_taup_paths_svg(&paths, "iasp91", &TaupPathsPlotOptions::default()),
        "plot_taup_paths_svg",
    );

    let fm = qml
        .events
        .iter()
        .find_map(|ev| ev.preferred_focal_mechanism())
        .expect("focal mechanism in events.qml");
    assert_well_formed_svg(
        &plot_beachball_svg(fm, &BeachballPlotOptions::default()),
        "plot_beachball_svg QuakeML MT",
    );
    assert_well_formed_svg(
        &plot_beachball_svg(
            NodalPlane {
                strike_deg: 310.0,
                dip_deg: 42.0,
                rake_deg: -95.0,
            },
            &BeachballPlotOptions::default(),
        ),
        "plot_beachball_svg SDR",
    );
    assert_well_formed_svg(
        &plot_beachball_svg(
            MomentTensor6::new(1.4e19, -0.8e19, -0.6e19, 0.5e19, -0.3e19, 0.2e19),
            &BeachballPlotOptions::default(),
        ),
        "plot_beachball_svg MomentTensor6",
    );
}

// ============================================================================
// 3. Structured CLI Integration Tests (All Subcommands + Subprocess Binaries)
// ============================================================================

#[test]
fn cli_all_subcommands_and_binaries_end_to_end() {
    let fixtures = oracle_fixtures_dir();
    let tmp = TempWorkDir::new("cli_e2e");

    let sac_le = fixtures.join("io/test_le.sac");
    let ms_steim2 = fixtures.join("io/mseed_steim2.mseed");
    let real_bhz = fixtures.join("io/real_event/real_3c_bhz.sac");
    let real_xml = fixtures.join("io/real_event/real_3c_station.xml");
    let sacpz = fixtures.join("io/iu_anmo_bhz.sacpz");
    let qml = fixtures.join("io/events.qml");

    // 1. `inspect --json`:
    let insp = run_cli_args(&cli_vec(&[
        "inspect",
        sac_le.to_str().unwrap(),
        ms_steim2.to_str().unwrap(),
        "--json",
    ]))
    .unwrap();
    assert!(insp.contains("\"format\":\"SAC_BINARY_LE\""));
    assert!(insp.contains("\"format\":\"MINISEED_V2\""));

    // 2. `convert` (miniSEED v2 Steim-2 -> miniSEED v3 -> SAC Binary BE -> GeoCSV):
    let out_ms3 = tmp.path.join("conv.ms3");
    let out_sac_be = tmp.path.join("conv_be.sac");
    let out_csv = tmp.path.join("conv.geocsv");
    run_cli_args(&cli_vec(&[
        "convert",
        ms_steim2.to_str().unwrap(),
        out_ms3.to_str().unwrap(),
        "--format",
        "ms3",
        "--encoding",
        "steim2",
        "--json",
    ]))
    .unwrap();
    run_cli_args(&cli_vec(&[
        "convert",
        out_ms3.to_str().unwrap(),
        out_sac_be.to_str().unwrap(),
        "--format",
        "sac-be",
    ]))
    .unwrap();
    run_cli_args(&cli_vec(&[
        "convert",
        out_sac_be.to_str().unwrap(),
        out_csv.to_str().unwrap(),
        "--format",
        "geocsv",
    ]))
    .unwrap();
    let orig_ms = earthsnail_io::read_mseed_file(&ms_steim2).unwrap();
    let rt_ms3 = earthsnail_io::read_mseed3_file(&out_ms3).unwrap();
    assert_eq!(orig_ms[0].data, rt_ms3[0].data);

    // 3. `filter`:
    let filt_sac = tmp.path.join("filtered.sac");
    let filt_json = run_cli_args(&cli_vec(&[
        "filter",
        sac_le.to_str().unwrap(),
        filt_sac.to_str().unwrap(),
        "--family",
        "butter",
        "--band",
        "bandpass",
        "--freqs",
        "0.5,8.0",
        "--corners",
        "4",
        "--passes",
        "2",
        "--json",
    ]))
    .unwrap();
    assert!(filt_json.contains("\"passes\":2"));
    assert!(filt_sac.exists());

    // 4. `transfer` on real Tohoku-Oki BHZ (checking deconvolved PGV parity with ObsPy!):
    let deconv_sac = tmp.path.join("tohoku_vel.sac");
    let tr_json = run_cli_args(&cli_vec(&[
        "transfer",
        real_bhz.to_str().unwrap(),
        deconv_sac.to_str().unwrap(),
        "--stationxml",
        real_xml.to_str().unwrap(),
        "--output",
        "vel",
        "--pre-filt",
        "0.005,0.01,8.0,9.5",
        "--water-level",
        "60",
        "--json",
    ]))
    .unwrap();
    assert!(tr_json.contains("\"kinematic_unit\":\"VEL\""));
    let deconv_tr = read_sac_file(&deconv_sac).unwrap();
    let pgv = deconv_tr
        .data
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);
    // Golden `deconvolved_pgv_m_per_s` in `real_event_expected.json` is ~1.237e-4 m/s:
    assert!(
        (pgv - 1.237e-4).abs() < 2.5e-5,
        "unexpected deconvolved PGV {pgv}"
    );

    // 5. `pick --json`:
    let pick_json = run_cli_args(&cli_vec(&[
        "pick",
        deconv_sac.to_str().unwrap(),
        "--method",
        "morse",
        "--json",
    ]))
    .unwrap();
    assert!(pick_json.contains("\"picked\":true"));

    // 6. `taup time --json`, `taup setsac --json`, and `taup locate --json`:
    let tt_json = run_cli_args(&cli_vec(&[
        "taup",
        "time",
        "--model",
        "iasp91",
        "-h",
        "29.0",
        "-d",
        "73.87078776567722",
        "-p",
        "P,S,PcP,SKS",
        "--json",
    ]))
    .unwrap();
    assert!(tt_json.contains("692.084"));

    let setsac_copy = tmp.path.join("stamped.sac");
    fs::copy(&real_bhz, &setsac_copy).unwrap();
    let setsac_json = run_cli_args(&cli_vec(&[
        "taup",
        "setsac",
        setsac_copy.to_str().unwrap(),
        "-p",
        "P,S,PP,PcP",
        "--json",
    ]))
    .unwrap();
    assert!(setsac_json.contains("\"stamped_count\":4"));

    // Synthetic 6-station P+S network for `taup locate --json`:
    let tm = TauModel::iasp91().unwrap();
    let ev_lat = 36.5_f64;
    let ev_lon = 140.2_f64;
    let ev_dep = 42.0_f64;
    let t0_true = 12.5_f64;
    let mut obs_txt = String::new();
    for (idx, &(az, dist_deg)) in [
        (15.0, 2.5),
        (80.0, 4.0),
        (145.0, 3.2),
        (210.0, 5.5),
        (275.0, 2.8),
        (330.0, 6.1),
    ]
    .iter()
    .enumerate()
    {
        let fwd = direct_geodesic(ev_lat, ev_lon, az, dist_deg * 111_195.0);
        let arrs = tm
            .get_travel_times_geo(
                ev_lat,
                ev_lon,
                ev_dep,
                fwd.lat_deg,
                fwd.lon_deg,
                &["P", "S"],
                false,
            )
            .unwrap();
        for ph_want in ["P", "S"] {
            if let Some(a) = arrs.iter().find(|a| a.name == ph_want) {
                obs_txt.push_str(&format!(
                    "STA{idx} {:.6} {:.6} {} {:.5} 0.05\n",
                    fwd.lat_deg,
                    fwd.lon_deg,
                    ph_want,
                    t0_true + a.time_s
                ));
            }
        }
    }
    let obs_path = tmp.path.join("picks.obs");
    fs::write(&obs_path, obs_txt).unwrap();
    let loc_json = run_cli_args(&cli_vec(&[
        "taup",
        "locate",
        "--obs",
        obs_path.to_str().unwrap(),
        "--model",
        "iasp91",
        "--json",
    ]))
    .unwrap();
    assert!(loc_json.contains("\"weighted_rms_s\":"));

    // 7. `ppsd`, `rf`, `fk`, and `plot`:
    let ppsd_svg = tmp.path.join("ppsd.svg");
    let ppsd_json = run_cli_args(&cli_vec(&[
        "ppsd",
        sac_le.to_str().unwrap(),
        "--sacpz",
        sacpz.to_str().unwrap(),
        "--svg",
        ppsd_svg.to_str().unwrap(),
        "--json",
    ]))
    .unwrap();
    assert!(ppsd_json.contains("\"num_segments\":"));
    assert_well_formed_svg(&fs::read_to_string(&ppsd_svg).unwrap(), "CLI ppsd.svg");

    let rf_svg = tmp.path.join("rf.svg");
    let rf_json = run_cli_args(&cli_vec(&[
        "rf",
        sac_le.to_str().unwrap(),
        filt_sac.to_str().unwrap(),
        "--svg",
        rf_svg.to_str().unwrap(),
        "--method",
        "waterlevel",
        "--json",
    ]))
    .unwrap();
    assert!(rf_json.contains("\"method\":\"waterlevel\""));
    assert_well_formed_svg(&fs::read_to_string(&rf_svg).unwrap(), "CLI rf.svg");

    // 4-element synthetic array for `fk --capon --json`:
    let mut fk_files = Vec::new();
    let true_baz = 60.0_f64;
    let true_s = 0.12_f64; // s/km
    let (sb, cb) = math::sin_cos(true_baz.to_radians());
    let fs_arr = 40.0;
    let t_arr0 = UtcTimestamp::from_ymd_hms_nano(2025, 3, 1, 0, 0, 0, 0).unwrap();
    for (idx, &(xe_km, yn_km)) in [(0.0, 0.0), (2.5, 0.4), (-1.8, 2.2), (0.6, -2.4)]
        .iter()
        .enumerate()
    {
        // Plane wave arriving from `true_baz` has slowness vector pointing from source -> array = (-s*sb, -s*cb),
        // or in `fk_analysis` phase convention `t_delay = -(sx * x_e + sy * y_n)` with `(sx, sy) = (s*sb, s*cb)`:
        let delay_s = -(true_s * sb * xe_km + true_s * cb * yn_km);
        let sig: Vec<f64> = (0..256)
            .map(|k| {
                let t = (k as f64) / fs_arr - delay_s;
                math::sin(TAU * 2.0 * t) + 0.5 * math::cos(TAU * 3.1 * t)
            })
            .collect();
        let d_m = math::hypot(xe_km, yn_km) * 1_000.0;
        let az = math::atan2(xe_km, yn_km).to_degrees().rem_euclid(360.0);
        let pos = direct_geodesic(35.0, -106.0, az, d_m);
        let mut st_stats = TraceStats::new(
            ChannelId::new("AR", format!("A{idx}"), "00", "BHZ"),
            t_arr0,
            fs_arr,
        );
        st_stats.sync_to_sac_header(&sig);
        if let Some(ref mut sac) = st_stats.sac {
            sac.set_float("stla", pos.lat_deg).unwrap();
            sac.set_float("stlo", pos.lon_deg).unwrap();
        }
        let fpath = tmp.path.join(format!("arr_{idx}.sac"));
        write_sac_file(&fpath, &Trace::new(st_stats, sig), SacEndian::Little).unwrap();
        fk_files.push(fpath);
    }
    let fk_json = run_cli_args(&cli_vec(&[
        "fk",
        fk_files[0].to_str().unwrap(),
        fk_files[1].to_str().unwrap(),
        fk_files[2].to_str().unwrap(),
        fk_files[3].to_str().unwrap(),
        "--capon",
        "--fmin",
        "1.0",
        "--fmax",
        "4.0",
        "--json",
    ]))
    .unwrap();
    assert!(fk_json.contains("\"method\":\"capon_mvdr\""));

    // Plot CLI subcommands:
    let bb_svg = tmp.path.join("bb.svg");
    run_cli_args(&cli_vec(&[
        "plot",
        "beachball",
        "--quakeml",
        qml.to_str().unwrap(),
        "-o",
        bb_svg.to_str().unwrap(),
        "--json",
    ]))
    .unwrap();
    assert_well_formed_svg(&fs::read_to_string(&bb_svg).unwrap(), "CLI bb.svg");

    let tp_svg = tmp.path.join("taup_paths.svg");
    run_cli_args(&cli_vec(&[
        "plot",
        "taup",
        "--model",
        "ak135",
        "-h",
        "50",
        "-d",
        "70",
        "-p",
        "P,S,PcP",
        "-o",
        tp_svg.to_str().unwrap(),
    ]))
    .unwrap();
    assert_well_formed_svg(&fs::read_to_string(&tp_svg).unwrap(), "CLI taup_paths.svg");

    // 8. Subprocess execution of real `earthsnail` and `es-sac` binaries:
    let bin_earthsnail = env!("CARGO_BIN_EXE_earthsnail");
    let out_verify = Command::new(bin_earthsnail)
        .args(["verify", "--json"])
        .output()
        .expect("spawn earthsnail verify");
    assert!(out_verify.status.success());
    let verify_stdout = String::from_utf8(out_verify.stdout).unwrap();
    assert!(verify_stdout.contains("\"passed\":true"));

    let bin_es_sac = env!("CARGO_BIN_EXE_es-sac");
    let sac_svg = tmp.path.join("es_sac_out.svg");
    let out_sac_bin = Command::new(bin_es_sac)
        .args([
            "-c",
            &format!(
                "fg sine 2.0 delta 0.02 npts 200; rmean; bp co 0.5 5.0 n 4 p 2; p1 \"{}\"; quit",
                sac_svg.display()
            ),
        ])
        .output()
        .expect("spawn es-sac");
    assert!(out_sac_bin.status.success());
    assert_well_formed_svg(&fs::read_to_string(&sac_svg).unwrap(), "es-sac binary SVG");

    // Direct in-process check of `run_es_sac_args`:
    let ver_es = run_es_sac_args(&cli_vec(&["--version"])).unwrap();
    assert!(ver_es.contains("es-sac"));
}

#[test]
fn sac_exhaustive_command_matrix_coverage() {
    let fixtures = oracle_fixtures_dir();
    let tmp = TempWorkDir::new("sac_matrix");
    let sacpz = fixtures.join("io/iu_anmo_bhz.sacpz");
    let sac_le = fixtures.join("io/test_le.sac");

    fs::copy(&sacpz, tmp.path.join("anmo.sacpz")).unwrap();
    fs::copy(&sac_le, tmp.path.join("work.sac")).unwrap();

    let mut sess = SacSession::with_working_dir(&tmp.path);

    // 1. SACPZ deconvolution + filter families (bu, be, c1, c2 across lp, hp, br) + write formats:
    sess.execute_script(
        "\
        r work.sac\n\
        rmean; rtrend; taper type hamming width 0.05\n\
        transfer from polezero subtype anmo.sacpz to disp freqlimits 0.01 0.02 12 18 waterlevel 60\n\
        lp co 8.0 n 4 p 2 be\n\
        hp co 0.05 n 4 p 1 c1\n\
        br co 2.0 3.0 n 4 p 2 c2 attenuation 40\n\
        w sac out_le.sac\n\
        w be out_be.sac\n\
        w alpha out_alpha.sac\n\
        w mseed out.mseed\n\
        w ms3 out.ms3\n\
        ch kuser0 TESTED\n\
        wh\n\
        ",
        &[],
    )
    .unwrap();

    // Auto-sniff back `work.sac` (updated in-place by `wh`!) + all 5 written formats (`le`, `be`, `alpha`, `mseed`, `ms3`):
    sess.execute_line("r work.sac out_le.sac out_be.sac out_alpha.sac out.mseed out.ms3")
        .unwrap();
    assert_eq!(sess.traces.len(), 6);
    assert_eq!(
        sess.traces[0].stats.sac.as_ref().unwrap().get_str("kuser0"),
        Some("TESTED")
    );

    // 2. Pointwise & file-by-file math (`mul`, `div`, `add`, `sub`, `sqr`, `sqrt`, `abs`, `log10`, `exp`, `reverse`, `addf`, `subf`, `mulf`, `divf`, `chpf`):
    sess.execute_script(
        "\
        fg boxcar delta 0.02 npts 100\n\
        add 2.0; mul 3.0; div 2.0; sub 1.0\n\
        sqr; sqrt; abs; log10; exp; reverse\n\
        w op_ref.sac\n\
        addf op_ref.sac; subf op_ref.sac; mulf op_ref.sac; divf op_ref.sac; chpf\n\
        dif; int trapezoidal; decimate 2\n\
        ",
        &[],
    )
    .unwrap();
    assert_eq!(sess.traces[0].npts(), 50);

    // 3. `cut` on T0 + offsets, `merge`, `sort`, `hilbert`, `envelope`, `correlate`, `markptp`, `apk` (`baer`, `stalta`, `aic`), `rot to zne`, and SAC plotters (`p2`, `prs`, `spectrogram`, `beachball`):
    sess.execute_script(
        "\
        r work.sac\n\
        ch t0 1.0 kt0 P_SYN\n\
        cut t0 -0.5 t0 2.5\n\
        markptp length 1.0\n\
        apk baer; apk stalta; apk aic\n\
        hilbert; envelope\n\
        p2 over_plot.svg\n\
        spectrogram spec_plot.svg stft linear\n\
        beachball bb_plot.svg 120 45 90\n\
        ",
        &[],
    )
    .unwrap();
    assert_well_formed_svg(
        &fs::read_to_string(tmp.path.join("over_plot.svg")).unwrap(),
        "SAC p2",
    );
    assert_well_formed_svg(
        &fs::read_to_string(tmp.path.join("spec_plot.svg")).unwrap(),
        "SAC spectrogram",
    );
    assert_well_formed_svg(
        &fs::read_to_string(tmp.path.join("bb_plot.svg")).unwrap(),
        "SAC beachball",
    );

    // 4. `rot to zne` on 3 orthogonal components + `correlate` + `sort` + `merge` + `prs`:
    sess.execute_script(
        "\
        fg sine 2.0 delta 0.02 npts 120; ch kcmpnm BHU cmpaz 0 cmpinc 0 gcarc 30; w c1.sac\n\
        fg sine 2.0 delta 0.02 npts 120; ch kcmpnm BHV cmpaz 0 cmpinc 90 gcarc 15; w c2.sac\n\
        fg sine 2.0 delta 0.02 npts 120; ch kcmpnm BHW cmpaz 90 cmpinc 90 gcarc 45; w c3.sac\n\
        cut off\n\
        r c1.sac c2.sac c3.sac\n\
        rot to zne\n\
        sort gcarc ascend\n\
        prs prs_plot.svg model iasp91 phases P S reduce 8.0\n\
        correlate master 1 length 0.5 normalized\n\
        merge\n\
        ",
        &[],
    )
    .unwrap();
    assert_well_formed_svg(
        &fs::read_to_string(tmp.path.join("prs_plot.svg")).unwrap(),
        "SAC prs",
    );
}

// ============================================================================
// 4. Gate 4 Adversarial Fuzzing (`0` Panics on Arbitrary SAC Macros & CLI Args)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn fuzz_arbitrary_sac_scripts_never_panic(
        raw_script in "[\\PC\\n;\"'$%#*]{0,180}",
        arg1 in "[\\PC]{0,24}",
    ) {
        let tmp = TempWorkDir::new("fuzz_sac");
        let mut sess = SacSession::with_working_dir(&tmp.path);
        // Never allow accidental stdin wait; execute_script never reads stdin:
        let _ = sess.execute_script(&raw_script, &[arg1.as_str()]);
    }

    #[test]
    fn fuzz_arbitrary_cli_arguments_never_panic(
        sub in prop_oneof![
            Just(String::from("inspect")),
            Just(String::from("convert")),
            Just(String::from("filter")),
            Just(String::from("transfer")),
            Just(String::from("pick")),
            Just(String::from("taup")),
            Just(String::from("plot")),
            Just(String::from("fdsn")),
            Just(String::from("verify")),
            "[\\PC]{0,20}",
        ],
        a1 in "[\\PC]{0,32}",
        a2 in "[\\PC]{0,32}",
        a3 in "[\\PC]{0,32}",
    ) {
        // Avoid invoking bare `sac` with zero `-c`/`-m` flags so test stdin is never read:
        if sub != "sac" {
            let args = vec![sub, a1, a2, a3];
            let _ = run_cli_args(&args);
        }
    }
}
