//! Generates the 4 publication showcase SVGs in `results/figures/` using the
//! zero-dependency `earthsnail-sac` W3C SVG 1.1 plotting engine:
//!   1. `results/figures/tohoku_3c_waveforms.svg`
//!   2. `results/figures/taup_global_raypaths.svg`
//!   3. `results/figures/anmo_ppsd_noise.svg`
//!   4. `results/figures/tohoku_moment_tensor_beachball.svg`

use core::f64::consts::TAU;
use std::fs;
use std::path::PathBuf;

use earthsnail_core::{Complex64, KinematicUnit, MomentTensor6, SacDepType, Stream};
use earthsnail_dsp::{
    eval_nlnm_db, irfft, remove_response, rfftfreq, Ppsd, PpsdConfig, ResponseRemovalConfig,
};
use earthsnail_io::{read_sac_file, read_stationxml_file};
use earthsnail_sac::{
    plot_beachball_svg, plot_ppsd_svg, plot_taup_paths_svg, plot_waveforms_svg,
    BeachballPlotOptions, PpsdPlotOptions, TaupPathsPlotOptions, TimeAxisMode, WaveformLayout,
    WaveformPlotOptions,
};
use earthsnail_taup::TauModel;

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64_open(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

fn assert_valid_svg(name: &str, svg: &str) {
    assert!(
        svg.contains("<svg") && svg.trim_end().ends_with("</svg>"),
        "{name} is not a well-formed <svg>...</svg> document"
    );
    assert!(
        svg.len() > 2_000,
        "{name} unexpectedly small ({} B)",
        svg.len()
    );
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let fixtures_io = repo_root.join("tests").join("fixtures").join("io");
    let out_dir = repo_root.join("results").join("figures");
    fs::create_dir_all(&out_dir).expect("create results/figures");

    // ------------------------------------------------------------------------
    // 1. `tohoku_3c_waveforms.svg` (Real 2011 Mw 9.1 Tohoku-Oki @ G.CAN.00)
    // ------------------------------------------------------------------------
    let real_dir = fixtures_io.join("real_event");
    let inv_can = read_stationxml_file(real_dir.join("real_3c_station.xml"))
        .expect("read real_3c_station.xml");
    let mut traces = Vec::with_capacity(3);
    for comp in ["bhz", "bhn", "bhe"] {
        let mut tr =
            read_sac_file(real_dir.join(format!("real_3c_{comp}.sac"))).expect("read real_3c sac");
        let ch_meta = inv_can
            .channels()
            .find(|c| c.channel.channel.eq_ignore_ascii_case(comp))
            .expect("matching channel response");
        let resp = ch_meta.response.as_ref().expect("channel response");
        let cfg = ResponseRemovalConfig {
            output: KinematicUnit::Velocity,
            pre_filt_hz: Some([0.005, 0.01, 7.0, 9.5]),
            water_level_db: Some(60.0),
            zero_mean: true,
            taper: true,
            taper_fraction: 0.05,
            nfft_pow2: true,
            rescale_to_overall_sensitivity: false,
        };
        let vel_m_s = remove_response(&tr.data, tr.stats.sampling_rate, resp, &cfg)
            .expect("deconvolve Tohoku trace");
        // Express in mm/s for crisp axis labels
        tr.data = vel_m_s.into_iter().map(|v| v * 1_000.0).collect();
        if let Some(ref mut sac) = tr.stats.sac {
            sac.idep = SacDepType::VelocityNmPerSec as i32;
        }
        traces.push(tr);
    }
    let stream_3c = Stream::from_traces(traces);
    let wf_svg = plot_waveforms_svg(
        &stream_3c,
        &WaveformPlotOptions {
            width: 1080.0,
            height: 620.0,
            layout: WaveformLayout::Stacked,
            time_axis: TimeAxisMode::UtcClock,
            title: Some(String::from(
                "2011-03-11 Mw 9.1 Tohoku-Oki — GEOSCOPE G.CAN.00 (Canberra, Δ = 73.9°, Deconvolved Velocity)",
            )),
            y_unit: Some(String::from("Vel (mm/s)")),
            show_phase_flags: true,
        },
    );
    assert_valid_svg("tohoku_3c_waveforms.svg", &wf_svg);
    fs::write(out_dir.join("tohoku_3c_waveforms.svg"), &wf_svg).unwrap();
    println!("Wrote tohoku_3c_waveforms.svg ({} bytes)", wf_svg.len());

    // ------------------------------------------------------------------------
    // 2. `taup_global_raypaths.svg` (`iasp91`: P, pP, PP, PcP, PKP, PKIKP, SKS)
    // ------------------------------------------------------------------------
    let iasp91 = TauModel::by_name("iasp91").expect("iasp91 model");
    let ray_queries: &[(&str, f64)] = &[
        ("P", 62.0),
        ("pP", 62.0),
        ("PcP", 52.0),
        ("PP", 92.0),
        ("SKS", 108.0),
        ("PKP", 148.0),
        ("PKIKP", 168.0),
    ];
    let mut all_paths = Vec::new();
    for &(ph, dist_deg) in ray_queries {
        let paths = iasp91
            .get_ray_paths(35.0, dist_deg, &[ph])
            .expect("get_ray_paths");
        if let Some(first) = paths.into_iter().next() {
            all_paths.push(first);
        }
    }
    let taup_svg = plot_taup_paths_svg(
        &all_paths,
        "iasp91",
        &TaupPathsPlotOptions {
            width: 960.0,
            height: 740.0,
            title: Some(String::from(
                "Spherical Earth Ray Paths — IASP91 (z = 35 km: P, pP, PcP, PP, SKS, PKP, PKIKP)",
            )),
        },
    );
    assert_valid_svg("taup_global_raypaths.svg", &taup_svg);
    fs::write(out_dir.join("taup_global_raypaths.svg"), &taup_svg).unwrap();
    println!("Wrote taup_global_raypaths.svg ({} bytes)", taup_svg.len());

    // ------------------------------------------------------------------------
    // 3. `anmo_ppsd_noise.svg` (`IU.ANMO.10.BHZ` + Peterson NLNM/NHNM)
    // ------------------------------------------------------------------------
    let inv_anmo =
        read_stationxml_file(fixtures_io.join("iu_anmo_bhz.xml")).expect("read iu_anmo_bhz.xml");
    let ch_anmo = inv_anmo.channels().next().expect("ANMO channel");
    let fs_anmo = ch_anmo.sample_rate_hz.unwrap_or(40.0);
    let resp_anmo = ch_anmo.response.as_ref().expect("ANMO response");

    let ppsd_cfg = PpsdConfig {
        ppsd_length_s: 600.0,
        overlap: 0.5,
        db_min: -200.0,
        db_max: -50.0,
        db_step: 1.0,
        period_smoothing_width_octaves: 1.0,
        period_step_octaves: 0.125,
    };
    let mut ppsd = Ppsd::new(fs_anmo, resp_anmo, ppsd_cfg).expect("init Ppsd");

    // Synthesize 24 windows of realistic ambient borehole noise coupled through IU.ANMO.10.BHZ's
    // actual 3-stage `H_acc(f)` response (quiet vault ~10-16 dB above NLNM + secondary microseism
    // swell at T ~ 5.5 s and diurnal high-frequency variation).
    let nfft_sim = 32_768_usize; // 819.2 s per realization @ 40 Hz
    let freqs = rfftfreq(nfft_sim, 1.0 / fs_anmo).unwrap();
    let df_bin = fs_anmo / (nfft_sim as f64);
    let h_acc_bins: Vec<Complex64> = freqs
        .iter()
        .map(|&f| {
            if f <= 0.0 {
                Complex64::ZERO
            } else {
                resp_anmo.eval_freq_kinematic(f, KinematicUnit::Acceleration)
            }
        })
        .collect();

    let mut rng = SplitMix64::new(2026_1116);
    for win_idx in 0..24_usize {
        let storm_swell_db = 8.0 * ((win_idx as f64) * 0.35).sin().abs();
        let cultural_db = if win_idx % 3 == 0 { 9.0 } else { 2.5 };
        let mut spec = vec![Complex64::ZERO; freqs.len()];
        for k in 1..(freqs.len() - 1) {
            let f = freqs[k];
            let period = 1.0 / f;
            let base_nlnm = eval_nlnm_db(period.clamp(0.1, 500.0)).unwrap_or(-165.0);
            let microseism =
                storm_swell_db * (-(((period.ln() - 5.6_f64.ln()) / 0.42).powi(2))).exp();
            let hf_cultural = if period < 0.8 { cultural_db } else { 0.0 };
            let target_db = base_nlnm + 13.5 + microseism + hf_cultural;
            let p_acc = 10.0_f64.powf(0.1 * target_db);
            // One-sided PSD `P(f)` -> Fourier bin RMS amplitude `|X[k]| = fs * sqrt(P(f) * df / 2)`
            let amp_acc = fs_anmo * (0.5 * p_acc * df_bin).sqrt();
            let g_re = rng.next_standard_normal() * core::f64::consts::FRAC_1_SQRT_2;
            let g_im = rng.next_standard_normal() * core::f64::consts::FRAC_1_SQRT_2;
            let acc_bin = Complex64::new(g_re * amp_acc, g_im * amp_acc);
            spec[k] = acc_bin * h_acc_bins[k];
        }
        let counts_ts = irfft(&spec, nfft_sim).unwrap();
        ppsd.add_samples(&counts_ts).unwrap();
    }

    let ppsd_svg = plot_ppsd_svg(
        &ppsd,
        &PpsdPlotOptions {
            width: 1020.0,
            height: 600.0,
            max_percent: 25.0,
            show_mean: true,
            show_mode: true,
            title: Some(format!(
                "Probabilistic Power Spectral Density (PPSD) — IU.ANMO.10.BHZ ({} segments vs. Peterson NLNM/NHNM)",
                ppsd.num_segments()
            )),
        },
    );
    assert_valid_svg("anmo_ppsd_noise.svg", &ppsd_svg);
    fs::write(out_dir.join("anmo_ppsd_noise.svg"), &ppsd_svg).unwrap();
    println!("Wrote anmo_ppsd_noise.svg ({} bytes)", ppsd_svg.len());

    // ------------------------------------------------------------------------
    // 4. `tohoku_moment_tensor_beachball.svg` (2011 Mw 9.1 GCMT Moment Tensor)
    // ------------------------------------------------------------------------
    let tohoku_mt = MomentTensor6::new(1.73e22, -0.281e22, -1.45e22, 2.12e22, 4.55e22, -0.657e22);
    let bb_svg = plot_beachball_svg(
        tohoku_mt,
        &BeachballPlotOptions {
            width: 580.0,
            height: 620.0,
            compressional_color: String::from("#dc2626"),
            dilatational_color: String::from("#ffffff"),
            show_axes: true,
            title: Some(String::from(
                "2011-03-11 Mw 9.1 Tohoku-Oki GCMT Moment Tensor",
            )),
        },
    );
    assert_valid_svg("tohoku_moment_tensor_beachball.svg", &bb_svg);
    fs::write(out_dir.join("tohoku_moment_tensor_beachball.svg"), &bb_svg).unwrap();
    println!(
        "Wrote tohoku_moment_tensor_beachball.svg ({} bytes)",
        bb_svg.len()
    );
}
