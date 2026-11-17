//! Zero-dependency structured CLI parser and dispatcher for `earthsnail` (`inspect`, `convert`,
//! `filter`, `transfer`, `pick`, `taup`, `ppsd`, `rf`, `fk`, `plot`, `fdsn`, `seedlink`, `verify`,
//! and `sac`).

use core::time::Duration;
use std::fs;
use std::path::Path;

use earthsnail_core::{
    gps2dist_azimuth, kilometer2degrees, math, ChannelId, ChannelResponse, EarthSnailError,
    KinematicUnit, MomentTensor6, NodalPlane, Result, SacDepType, SacEndian, SacHeader, Stream,
    Trace, TraceStats, UtcTimestamp,
};
use earthsnail_dsp::{
    aic_pick, classic_sta_lta, design_iir_sos, fk_analysis, morse_pick, pk_baer,
    receiver_function_iter, receiver_function_waterlevel, remove_response_trace, sosfilt,
    sosfilt_zerophase_obspy, sosfiltfilt, trigger_onset, ArrayStation, BaerPickParams, BesselNorm,
    FilterBand, FilterFamily, FkMethod, MorsePickConfig, Ppsd, PpsdConfig, ResponseRemovalConfig,
};
use earthsnail_io::{
    detect_waveform_format, read_quakeml_file, read_sacpz_file, read_stationxml_file,
    read_waveforms_bytes, read_waveforms_file, write_geocsv_stream, write_mseed3_stream_bytes,
    write_mseed_stream_bytes, write_quakeml_file, write_sac_alpha_file, write_sac_file,
    write_slist_stream, write_stationxml_file, write_tspair_stream, EventQuery, FdsnClient,
    Mseed3WriteOptions, MseedByteOrder, MseedEncoding, MseedWriteOptions, SeedLinkClient,
    SeedLinkPacket, SeedLinkV4Format, StationLevel, StationQuery, WaveformFormat, WaveformQuery,
};
use earthsnail_taup::{
    locate_hypocenter, HypocenterConfig, PhaseArrivalObservation, SubductionPlumeModel3D,
    TauEngine, TauModel, VelocityModel,
};

use crate::json::{json_f64_slice, JsonArr, JsonObj};
use crate::plot::{
    plot_beachball_svg, plot_ppsd_svg, plot_record_section_svg, plot_spectrogram_svg,
    plot_taup_paths_svg, plot_waveforms_svg, BeachballPlotOptions, DistanceUnit,
    FocalMechanismInput, FrequencyScale, PpsdPlotOptions, RecordSectionOptions, SpectrogramMethod,
    SpectrogramOptions, TaupPathsPlotOptions, TimeAxisMode, WaveformLayout, WaveformPlotOptions,
};
use crate::sac_shell::{stamp_taup_into_sac_header, SacSession};
use crate::verify::run_self_verify;

/// Crate version string.
pub const EARTHSNAIL_VERSION: &str = env!("CARGO_PKG_VERSION");

const TOP_HELP: &str = "\
EarthSnail — Pure-Rust Zero-Bloat Seismological Engine (SAC + TauP + ObsPy)

USAGE:
  earthsnail <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
  inspect    Inspect waveform/metadata files, headers, amplitude stats & gaps [--json]
  convert    Convert between SAC (LE/BE/Alpha), miniSEED v2/v3 (Steim-1/2), GeoCSV, SLIST, TSPAIR
  filter     Apply causal or zero-phase SOS Butterworth / Chebyshev / Bessel digital filters
  transfer   Deconvolve instrument response (StationXML or SACPZ) to VEL, DISP, or ACC
  pick       Automatic phase onset picking (Morse glitch-immune, Baer-Kradolfer, STA/LTA, AIC)
  taup       Spherical & 3D ray tracing (time, pierce, path, curve, setsac, locate) [--json]
  ppsd       McNamara & Buland Probabilistic Power Spectral Density + Peterson NLNM/NHNM SVG
  rf         Ligorría-Ammon iterative or water-level P-receiver function deconvolution + SVG
  fk         Conventional & Capon MVDR frequency-wavenumber array beamforming [--json]
  plot       Publication SVG plotting (waveform, section, spectrogram, ppsd, taup, beachball)
  fdsn       FDSN Web Services client (dataselect, station, event)
  seedlink   SeedLink v3/v4 real-time streaming client (info, stream)
  verify     Run built-in self-verification suite (< 200 ms) [--json]
  sac        Launch interactive SAC> REPL or execute .m macro scripts (-m <file.m>, -c \"<cmds>\")

GLOBAL FLAGS:
  -h, --help       Print help information
  -V, --version    Print version information
  --json           Emit machine-readable RFC-8259 JSON output
";

/// Executes CLI arguments (excluding `argv[0]`), returning formatted stdout text (or launching
/// the interactive `SAC>` stdio REPL when `earthsnail sac` is invoked without `-m` or `-c`).
pub fn run_cli_args(args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Ok(String::from(TOP_HELP));
    }
    let first = args[0].as_str();
    match first {
        "-h" | "--help" | "help" => {
            if let Some(sub) = args.get(1) {
                return Ok(subcommand_help(sub));
            }
            Ok(String::from(TOP_HELP))
        }
        "-V" | "--version" | "version" => Ok(format!("earthsnail {EARTHSNAIL_VERSION}\n")),
        "inspect" => cmd_inspect(&args[1..]),
        "convert" => cmd_convert(&args[1..]),
        "filter" => cmd_filter(&args[1..]),
        "transfer" => cmd_transfer(&args[1..]),
        "pick" => cmd_pick(&args[1..]),
        "taup" => cmd_taup(&args[1..]),
        "ppsd" => cmd_ppsd(&args[1..]),
        "rf" => cmd_rf(&args[1..]),
        "fk" => cmd_fk(&args[1..]),
        "plot" => cmd_plot(&args[1..]),
        "fdsn" => cmd_fdsn(&args[1..]),
        "seedlink" => cmd_seedlink(&args[1..]),
        "verify" => cmd_verify(&args[1..]),
        "sac" => cmd_sac(&args[1..]),
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unknown subcommand '{other}' (run 'earthsnail --help' for usage)"
        ))),
    }
}

/// Entrypoint helper for `es-sac` standalone SAC binary (`es-sac [-m <script.m> [args...]] [-c "<cmds>"] [files...]`).
pub fn run_es_sac_args(args: &[String]) -> Result<String> {
    if args.len() == 1 && matches!(args[0].as_str(), "-V" | "--version") {
        return Ok(format!("es-sac (earthsnail) {EARTHSNAIL_VERSION}\n"));
    }
    cmd_sac(args)
}

fn has_help_flag(args: &[String]) -> bool {
    args.iter().any(|a| a == "--help") || (args.len() == 1 && args[0] == "-h")
}

fn subcommand_help(sub: &str) -> String {
    match sub {
        "inspect" => String::from("USAGE: earthsnail inspect <files...> [--json]\n"),
        "convert" => String::from(
            "USAGE: earthsnail convert <input> <output> [--format sac|sac-le|sac-be|sac-alpha|mseed|ms3|geocsv|slist|tspair] [--encoding steim1|steim2|int16|int32|float32|float64] [--json]\n",
        ),
        "filter" => String::from(
            "USAGE: earthsnail filter <input> <output> --family butter|cheby1|cheby2|bessel --band bp|lp|hp|br --freqs <f1[,f2]> [--corners 4] [--passes 1|2] [--filtfilt] [--json]\n",
        ),
        "transfer" => String::from(
            "USAGE: earthsnail transfer <input> <output> (--stationxml <f> | --sacpz <f>) [--output vel|disp|acc] [--pre-filt f1,f2,f3,f4] [--water-level 60] [--json]\n",
        ),
        "pick" => String::from(
            "USAGE: earthsnail pick <input> [--method morse|baer|stalta|aic] [--thr <val>] [--json]\n",
        ),
        "taup" => String::from(
            "USAGE:\n  earthsnail taup time   -h <evdp_km> -d <deg> [-p P,S,...] [--model iasp91|ak135|prem|<path>] [--json]\n  earthsnail taup pierce -h <evdp_km> -d <deg> [-p P,S,...] [--model ...] [--json]\n  earthsnail taup path   -h <evdp_km> -d <deg> [-p P,S,...] [-o <out.svg>] [--model ...] [--json]\n  earthsnail taup curve  -h <evdp_km> [-p P,S,...] [--points 64] [-o <out.svg>] [--model ...] [--json]\n  earthsnail taup setsac <files...> [-p P,S,...] [--model ...] [--evdpkm] [-o <out.sac>] [--json]\n  earthsnail taup locate --obs <obs_file> [--model ...] [--ellipticity] [--3d] [--json]\n",
        ),
        "ppsd" => String::from(
            "USAGE: earthsnail ppsd <waveform> (--stationxml <f> | --sacpz <f>) --svg <out.svg> [--segment-len <sec>] [--overlap 0.5] [--json]\n",
        ),
        "rf" => String::from(
            "USAGE: earthsnail rf <z_file> <r_file> --svg <out.svg> [--method iter|waterlevel] [--gauss 2.5] [--shift 5.0] [--water-level 0.01] [-o <rf.sac>] [--json]\n",
        ),
        "fk" => String::from(
            "USAGE: earthsnail fk <files...> [--capon] [--fmin 0.5] [--fmax 4.0] [--smax 0.4] [--grid 41] [--json]\n",
        ),
        "plot" => String::from(
            "USAGE:\n  earthsnail plot waveform    <files...> -o <out.svg> [--overlaid] [--utc]\n  earthsnail plot section     <files...> -o <out.svg> [--model iasp91] [--phases P,S] [--reduce <km_s>] [--km]\n  earthsnail plot spectrogram <file>     -o <out.svg> [--cwt|--stft] [--log|--linear]\n  earthsnail plot ppsd        <file> (--stationxml <f> | --sacpz <f>) -o <out.svg>\n  earthsnail plot taup        -o <out.svg> [--model iasp91] [-h 29] [-d 74] [-p P,S,PP,PcP,SKS]\n  earthsnail plot beachball   -o <out.svg> (--sdr <strike,dip,rake> | --mt <mrr,mtt,mpp,mrt,mrp,mtp>)\n",
        ),
        "fdsn" => String::from(
            "USAGE: earthsnail fdsn <dataselect|station|event> [--provider EARTHSCOPE|USGS|<url>] [--net ...] [--sta ...] [--loc ...] [--cha ...] [--start ...] [--end ...] [--minmag ...] [--url-only] [-o <out>] [--json]\n",
        ),
        "seedlink" => String::from(
            "USAGE: earthsnail seedlink <info|stream> --addr <host:port> [--level ID|STREAMS] [--net ...] [--sta ...] [--cha ...] [--max-packets 10] [-o <out.mseed>] [--json]\n",
        ),
        "verify" => String::from("USAGE: earthsnail verify [--json]\n"),
        "sac" => String::from("USAGE: earthsnail sac [-m <script.m> [args...]] [-c \"<cmds>\"] [files...]\n"),
        _ => String::from(TOP_HELP),
    }
}

// ============================================================================
// 1. `earthsnail inspect <files...> [--json]`
// ============================================================================

fn cmd_inspect(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("inspect"));
    }
    let mut json_mode = false;
    let mut files: Vec<String> = Vec::new();
    for a in args {
        if a == "--json" {
            json_mode = true;
        } else if a.starts_with('-') {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unknown flag '{a}' in 'inspect'"
            )));
        } else {
            files.push(a.clone());
        }
    }
    if files.is_empty() {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail inspect' requires one or more input files".into(),
        ));
    }

    let mut file_json_arr = JsonArr::new();
    let mut text_out = String::new();

    for path in &files {
        let raw = fs::read(path)
            .map_err(|e| EarthSnailError::Io(format!("cannot read '{path}': {e}")))?;
        let fmt_opt = detect_waveform_format(&raw);
        let fmt_str = match fmt_opt {
            Some(WaveformFormat::SacBinaryLittle) => "SAC_BINARY_LE",
            Some(WaveformFormat::SacBinaryBig) => "SAC_BINARY_BE",
            Some(WaveformFormat::SacAlpha) => "SAC_ALPHA",
            Some(WaveformFormat::MiniSeed2) => "MINISEED_V2",
            Some(WaveformFormat::MiniSeed3) => "MINISEED_V3",
            Some(WaveformFormat::GeoCsv) => "GEOCSV",
            Some(WaveformFormat::SList) => "SLIST",
            Some(WaveformFormat::TsPair) => "TSPAIR",
            None => "UNKNOWN",
        };
        let stream = read_waveforms_bytes(&raw)?;
        let gaps = stream.get_gaps();

        if json_mode {
            let mut tr_arr = JsonArr::new();
            for tr in &stream.traces {
                let (mn, mx, mean) = tr.min_max_mean().unwrap_or((0.0, 0.0, 0.0));
                let mut to = JsonObj::new();
                to.str("channel", &tr.stats.channel.seed_id())
                    .str("network", &tr.stats.channel.network)
                    .str("station", &tr.stats.channel.station)
                    .str("location", &tr.stats.channel.location)
                    .str("component", &tr.stats.channel.channel)
                    .str("starttime", &tr.stats.starttime.to_iso8601())
                    .str("endtime", &tr.endtime().to_iso8601())
                    .num("sampling_rate", tr.stats.sampling_rate)
                    .num("delta", tr.delta())
                    .uint("npts", tr.npts())
                    .num("min", mn)
                    .num("max", mx)
                    .num("mean", mean);
                if let Some(ref sac) = tr.stats.sac {
                    to.opt_num("gcarc_deg", sac.get_float("gcarc"))
                        .opt_num("dist_km", sac.get_float("dist"))
                        .opt_num("az_deg", sac.get_float("az"))
                        .opt_num("baz_deg", sac.get_float("baz"))
                        .opt_num("evdp_km", sac.get_float("evdp"))
                        .opt_num("o", sac.get_float("o"))
                        .opt_num("a", sac.get_float("a"))
                        .opt_num("t0", sac.get_float("t0"))
                        .opt_str("kt0", sac.get_str("kt0"));
                }
                tr_arr.push_raw(&to.finish());
            }
            let mut gap_arr = JsonArr::new();
            for g in &gaps {
                let mut go = JsonObj::new();
                go.str("channel", &g.channel.seed_id())
                    .num("difference_s", g.difference_s)
                    .int("samples", g.samples);
                gap_arr.push_raw(&go.finish());
            }
            let mut fo = JsonObj::new();
            fo.str("file", path)
                .str("format", fmt_str)
                .uint("size_bytes", raw.len())
                .uint("num_traces", stream.len())
                .raw("traces", &tr_arr.finish())
                .raw("gaps", &gap_arr.finish());
            file_json_arr.push_raw(&fo.finish());
        } else {
            text_out.push_str(&format!(
                "File: {path}  [{fmt_str}, {} bytes, {} trace(s), {} gap(s)]\n",
                raw.len(),
                stream.len(),
                gaps.len()
            ));
            for (i, tr) in stream.traces.iter().enumerate() {
                let (mn, mx, mean) = tr.min_max_mean().unwrap_or((0.0, 0.0, 0.0));
                text_out.push_str(&format!(
                    "  [{i}] {:<16} | {} .. {} | {:>6.2} Hz | {:>7} pts | min={:.4e} max={:.4e} mean={:.4e}\n",
                    tr.stats.channel.seed_id(),
                    tr.stats.starttime.to_iso8601(),
                    tr.endtime().to_iso8601(),
                    tr.stats.sampling_rate,
                    tr.npts(),
                    mn,
                    mx,
                    mean
                ));
            }
        }
    }

    if json_mode {
        let mut root = JsonObj::new();
        root.uint("num_files", files.len())
            .raw("files", &file_json_arr.finish());
        Ok(format!("{}\n", root.finish()))
    } else {
        Ok(text_out)
    }
}

// ============================================================================
// 2. `earthsnail convert <input> <output> [--format ...] [--encoding ...] [--json]`
// ============================================================================

fn cmd_convert(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("convert"));
    }
    let mut fmt_opt: Option<String> = None;
    let mut enc_opt: Option<String> = None;
    let mut json_mode = false;
    let mut pos: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--format" | "-f" => {
                fmt_opt = Some(req_val(args, i, "--format")?);
                i += 2;
            }
            "--encoding" | "-e" => {
                enc_opt = Some(req_val(args, i, "--encoding")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'convert'"
                )));
            }
            other => {
                pos.push(String::from(other));
                i += 1;
            }
        }
    }
    if pos.len() != 2 {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail convert' expects <input> <output>".into(),
        ));
    }
    let input = &pos[0];
    let output = &pos[1];
    let stream = read_waveforms_file(input)?;
    if stream.is_empty() {
        return Err(EarthSnailError::EmptyData(format!(
            "no traces found in '{input}'"
        )));
    }

    let fmt_str = fmt_opt
        .unwrap_or_else(|| infer_format_from_path(output))
        .to_ascii_lowercase();
    let enc = match enc_opt
        .as_deref()
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("steim1") => MseedEncoding::Steim1,
        Some("steim2") => MseedEncoding::Steim2,
        Some("int16" | "i16") => MseedEncoding::Int16,
        Some("int32" | "i32") => MseedEncoding::Int32,
        Some("float32" | "f32") => MseedEncoding::Float32,
        Some("float64" | "f64") => MseedEncoding::Float64,
        None => default_mseed_encoding_for_stream(&stream),
        Some(other) => {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unsupported --encoding '{other}' (expected steim1|steim2|int16|int32|float32|float64)"
            )));
        }
    };

    write_stream_with_format(&stream, output, &fmt_str, enc)?;

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("input", input)
            .str("output", output)
            .str("format", &fmt_str)
            .uint("num_traces", stream.len())
            .uint("total_samples", stream.traces.iter().map(Trace::npts).sum());
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "Converted {} trace(s) from '{}' -> '{}' ({fmt_str})\n",
            stream.len(),
            input,
            output
        ))
    }
}

fn default_mseed_encoding_for_stream(st: &Stream) -> MseedEncoding {
    let all_int = st.traces.iter().all(|tr| {
        tr.data.iter().all(|&v| {
            v.is_finite() && v.abs() <= 500_000_000.0 && (v - math::round(v)).abs() < 1e-9
        })
    });
    if all_int {
        MseedEncoding::Steim2
    } else {
        MseedEncoding::Float64
    }
}

fn infer_format_from_path(path: &str) -> String {
    let lc = path.to_ascii_lowercase();
    if lc.ends_with(".mseed") || lc.ends_with(".msd") {
        String::from("mseed")
    } else if lc.ends_with(".ms3") || lc.ends_with(".mseed3") {
        String::from("ms3")
    } else if lc.ends_with(".alpha") {
        String::from("sac-alpha")
    } else if lc.ends_with(".geocsv") || lc.ends_with(".csv") {
        String::from("geocsv")
    } else if lc.ends_with(".slist") {
        String::from("slist")
    } else if lc.ends_with(".tspair") {
        String::from("tspair")
    } else {
        String::from("sac-le")
    }
}

fn write_stream_with_format(
    stream: &Stream,
    output: &str,
    fmt_lc: &str,
    enc: MseedEncoding,
) -> Result<()> {
    ensure_parent_dir(output)?;
    match fmt_lc {
        "sac" | "sac-le" | "le" => write_sac_file(output, &stream[0], SacEndian::Little),
        "sac-be" | "be" => write_sac_file(output, &stream[0], SacEndian::Big),
        "sac-alpha" | "alpha" => write_sac_alpha_file(output, &stream[0]),
        "mseed" | "mseed2" | "ms2" => {
            let bytes = write_mseed_stream_bytes(
                stream,
                MseedWriteOptions {
                    record_length: 512,
                    encoding: enc,
                    byte_order: MseedByteOrder::Big,
                    quality_indicator: 'D',
                },
            )?;
            fs::write(output, bytes)
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{output}': {e}")))
        }
        "ms3" | "mseed3" => {
            let bytes = write_mseed3_stream_bytes(
                stream,
                &Mseed3WriteOptions {
                    encoding: enc,
                    ..Mseed3WriteOptions::default()
                },
            )?;
            fs::write(output, bytes)
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{output}': {e}")))
        }
        "geocsv" | "csv" => {
            let txt = write_geocsv_stream(stream, true)?;
            fs::write(output, txt.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{output}': {e}")))
        }
        "slist" => {
            let txt = write_slist_stream(stream)?;
            fs::write(output, txt.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{output}': {e}")))
        }
        "tspair" => {
            let txt = write_tspair_stream(stream)?;
            fs::write(output, txt.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{output}': {e}")))
        }
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unsupported output format '{other}'"
        ))),
    }
}

// ============================================================================
// 3. `earthsnail filter <input> <output> --family ... --band ... --freqs ...`
// ============================================================================

fn cmd_filter(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("filter"));
    }
    let mut family_str = String::from("butter");
    let mut band_str = String::from("bandpass");
    let mut freqs: Vec<f64> = Vec::new();
    let mut corners = 4usize;
    let mut passes = 1usize;
    let mut filtfilt = false;
    let mut rp_db = 1.0_f64;
    let mut rs_db = 40.0_f64;
    let mut json_mode = false;
    let mut pos: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--family" => {
                family_str = req_val(args, i, "--family")?;
                i += 2;
            }
            "--band" => {
                band_str = req_val(args, i, "--band")?;
                i += 2;
            }
            "--freqs" | "--freq" => {
                let raw = req_val(args, i, "--freqs")?;
                for tok in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    freqs.push(parse_f64_str(tok, "--freqs")?);
                }
                i += 2;
                while i < args.len() && !args[i].starts_with('-') && freqs.len() < 2 {
                    if let Ok(v) = args[i].parse::<f64>() {
                        freqs.push(v);
                        i += 1;
                    } else {
                        break;
                    }
                }
            }
            "--corners" | "--order" | "-n" => {
                corners = parse_usize_str(&req_val(args, i, "--corners")?, "--corners")?;
                i += 2;
            }
            "--passes" | "-p" => {
                passes = parse_usize_str(&req_val(args, i, "--passes")?, "--passes")?;
                i += 2;
            }
            "--filtfilt" | "--zerophase-filtfilt" => {
                filtfilt = true;
                i += 1;
            }
            "--zerophase" => {
                passes = 2;
                i += 1;
            }
            "--rp" => {
                rp_db = parse_f64_str(&req_val(args, i, "--rp")?, "--rp")?;
                i += 2;
            }
            "--rs" => {
                rs_db = parse_f64_str(&req_val(args, i, "--rs")?, "--rs")?;
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'filter'"
                )));
            }
            other => {
                pos.push(String::from(other));
                i += 1;
            }
        }
    }
    if pos.len() != 2 {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail filter' expects <input> <output>".into(),
        ));
    }

    let family = match family_str.to_ascii_lowercase().as_str() {
        "butter" | "butterworth" | "bu" => FilterFamily::Butterworth,
        "cheby1" | "c1" => FilterFamily::Chebyshev1 { rp_db },
        "cheby2" | "c2" => FilterFamily::Chebyshev2 { rs_db },
        "bessel" | "be" => FilterFamily::Bessel {
            norm: BesselNorm::Phase,
        },
        other => {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unsupported --family '{other}' (expected butter|cheby1|cheby2|bessel)"
            )));
        }
    };

    let band = match band_str.to_ascii_lowercase().as_str() {
        "lowpass" | "lp" => {
            if freqs.len() != 1 {
                return Err(EarthSnailError::InvalidParameter(
                    "lowpass requires 1 frequency in --freqs".into(),
                ));
            }
            FilterBand::Lowpass(freqs[0])
        }
        "highpass" | "hp" => {
            if freqs.len() != 1 {
                return Err(EarthSnailError::InvalidParameter(
                    "highpass requires 1 frequency in --freqs".into(),
                ));
            }
            FilterBand::Highpass(freqs[0])
        }
        "bandpass" | "bp" => {
            if freqs.len() != 2 {
                return Err(EarthSnailError::InvalidParameter(
                    "bandpass requires 2 frequencies in --freqs <fmin,fmax>".into(),
                ));
            }
            FilterBand::Bandpass(freqs[0].min(freqs[1]), freqs[0].max(freqs[1]))
        }
        "bandstop" | "bandrej" | "br" | "bs" => {
            if freqs.len() != 2 {
                return Err(EarthSnailError::InvalidParameter(
                    "bandstop requires 2 frequencies in --freqs <fmin,fmax>".into(),
                ));
            }
            FilterBand::Bandstop(freqs[0].min(freqs[1]), freqs[0].max(freqs[1]))
        }
        other => {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unsupported --band '{other}' (expected bandpass|lowpass|highpass|bandstop)"
            )));
        }
    };

    if passes != 1 && passes != 2 {
        return Err(EarthSnailError::InvalidParameter(
            "--passes must be 1 or 2".into(),
        ));
    }

    let mut stream = read_waveforms_file(&pos[0])?;
    if stream.is_empty() {
        return Err(EarthSnailError::EmptyData("input stream is empty".into()));
    }
    for tr in &mut stream.traces {
        let sos = design_iir_sos(corners, band, tr.stats.sampling_rate, family)?;
        tr.data = if filtfilt {
            sosfiltfilt(&sos, &tr.data)?
        } else if passes == 2 {
            sosfilt_zerophase_obspy(&sos, &tr.data)
        } else {
            sosfilt(&sos, &tr.data)
        };
        if let Some(ref mut sac) = tr.stats.sac {
            sac.update_dep_stats(&tr.data);
        }
    }

    let out_fmt = infer_format_from_path(&pos[1]);
    write_stream_with_format(&stream, &pos[1], &out_fmt, MseedEncoding::Float64)?;

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("input", &pos[0])
            .str("output", &pos[1])
            .str("family", &family_str)
            .str("band", &band_str)
            .raw("freqs_hz", &json_f64_slice(&freqs))
            .uint("corners", corners)
            .uint("passes", passes)
            .bool("filtfilt", filtfilt)
            .uint("num_traces", stream.len());
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "Filtered {} trace(s) ({family_str} {band_str}, corners={corners}, passes={passes}, filtfilt={filtfilt}) -> '{}'\n",
            stream.len(),
            pos[1]
        ))
    }
}

// ============================================================================
// 4. `earthsnail transfer <input> <output> (--stationxml <f> | --sacpz <f>) ...`
// ============================================================================

fn cmd_transfer(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("transfer"));
    }
    let mut stationxml_path: Option<String> = None;
    let mut sacpz_path: Option<String> = None;
    let mut out_unit_str = String::from("vel");
    let mut pre_filt: Option<[f64; 4]> = None;
    let mut water_level = 60.0_f64;
    let mut json_mode = false;
    let mut pos: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--stationxml" | "--xml" => {
                stationxml_path = Some(req_val(args, i, "--stationxml")?);
                i += 2;
            }
            "--sacpz" | "--pz" => {
                sacpz_path = Some(req_val(args, i, "--sacpz")?);
                i += 2;
            }
            "--output" | "--to" => {
                out_unit_str = req_val(args, i, "--output")?;
                i += 2;
            }
            "--pre-filt" | "--freqlimits" => {
                let raw = req_val(args, i, "--pre-filt")?;
                let vals: Result<Vec<f64>> = raw
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|t| parse_f64_str(t, "--pre-filt"))
                    .collect();
                let v = vals?;
                if v.len() != 4 {
                    return Err(EarthSnailError::InvalidParameter(
                        "--pre-filt requires 4 comma-separated frequencies f1,f2,f3,f4".into(),
                    ));
                }
                pre_filt = Some([v[0], v[1], v[2], v[3]]);
                i += 2;
            }
            "--water-level" | "--wl" => {
                water_level = parse_f64_str(&req_val(args, i, "--water-level")?, "--water-level")?;
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'transfer'"
                )));
            }
            other => {
                pos.push(String::from(other));
                i += 1;
            }
        }
    }
    if pos.len() != 2 {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail transfer' expects <input> <output>".into(),
        ));
    }
    if stationxml_path.is_none() && sacpz_path.is_none() {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail transfer' requires --stationxml <file> or --sacpz <file>".into(),
        ));
    }

    let out_unit = match out_unit_str.to_ascii_lowercase().as_str() {
        "vel" | "velocity" => KinematicUnit::Velocity,
        "disp" | "displacement" | "none" => KinematicUnit::Displacement,
        "acc" | "acceleration" => KinematicUnit::Acceleration,
        other => {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unsupported --output '{other}' (expected vel|disp|acc)"
            )));
        }
    };

    let inv_opt = match stationxml_path {
        Some(ref p) => Some(read_stationxml_file(p)?),
        None => None,
    };
    let pz_opt = match sacpz_path {
        Some(ref p) => Some(read_sacpz_file(p)?.response),
        None => None,
    };

    let cfg = ResponseRemovalConfig {
        output: out_unit,
        pre_filt_hz: pre_filt,
        water_level_db: Some(water_level),
        ..ResponseRemovalConfig::default()
    };

    let stream = read_waveforms_file(&pos[0])?;
    if stream.is_empty() {
        return Err(EarthSnailError::EmptyData("input stream is empty".into()));
    }
    let mut out_traces = Vec::with_capacity(stream.len());
    for tr in &stream.traces {
        let resp = resolve_channel_response_for_trace(tr, inv_opt.as_ref(), pz_opt.as_ref())?;
        let mut deconv = remove_response_trace(tr, resp, &cfg)?;
        if let Some(ref mut sac) = deconv.stats.sac {
            sac.idep = match out_unit {
                KinematicUnit::Displacement => SacDepType::DisplacementNm as i32,
                KinematicUnit::Velocity => SacDepType::VelocityNmPerSec as i32,
                KinematicUnit::Acceleration => SacDepType::AccelerationNmPerSec2 as i32,
                KinematicUnit::Other => SacDepType::Unknown as i32,
            };
        }
        out_traces.push(deconv);
    }
    let out_stream = Stream::from_traces(out_traces);
    let out_fmt = infer_format_from_path(&pos[1]);
    write_stream_with_format(&out_stream, &pos[1], &out_fmt, MseedEncoding::Float64)?;

    let peak_abs = out_stream[0]
        .data
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("input", &pos[0])
            .str("output", &pos[1])
            .str("kinematic_unit", &out_unit_str.to_ascii_uppercase())
            .num("water_level_db", water_level)
            .uint("num_traces", out_stream.len())
            .num("first_trace_peak_abs", peak_abs);
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "Deconvolved {} trace(s) to {} (peak={:.4e}) -> '{}'\n",
            out_stream.len(),
            out_unit_str.to_ascii_uppercase(),
            peak_abs,
            pos[1]
        ))
    }
}

fn resolve_channel_response_for_trace<'a>(
    tr: &Trace,
    inv: Option<&'a earthsnail_core::StationInventory>,
    pz: Option<&'a ChannelResponse>,
) -> Result<&'a ChannelResponse> {
    if let Some(r) = pz {
        return Ok(r);
    }
    let inv = inv.ok_or_else(|| {
        EarthSnailError::ResponseError("missing StationXML or SACPZ response".into())
    })?;
    if let Some(c) = inv
        .find_channel(&tr.stats.channel, Some(tr.stats.starttime))
        .or_else(|| inv.find_channel(&tr.stats.channel, None))
    {
        if let Some(ref r) = c.response {
            return Ok(r);
        }
    }
    if let Some(c) = inv.channels().find(|ch| {
        ch.channel
            .station
            .eq_ignore_ascii_case(&tr.stats.channel.station)
            && ch
                .channel
                .channel
                .eq_ignore_ascii_case(&tr.stats.channel.channel)
            && ch.response.is_some()
    }) {
        return Ok(c.response.as_ref().unwrap());
    }
    let all: Vec<&ChannelResponse> = inv.channels().filter_map(|c| c.response.as_ref()).collect();
    if all.len() == 1 {
        return Ok(all[0]);
    }
    Err(EarthSnailError::ResponseError(format!(
        "no matching response in StationXML for '{}'",
        tr.stats.channel.seed_id()
    )))
}

// ============================================================================
// 5. `earthsnail pick <input> [--method morse|baer|stalta|aic] [--json]`
// ============================================================================

fn cmd_pick(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("pick"));
    }
    let mut method = String::from("morse");
    let mut thr_opt: Option<f64> = None;
    let mut json_mode = false;
    let mut input: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--method" | "-m" => {
                method = req_val(args, i, "--method")?.to_ascii_lowercase();
                i += 2;
            }
            "--thr" | "--threshold" => {
                thr_opt = Some(parse_f64_str(&req_val(args, i, "--thr")?, "--thr")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'pick'"
                )));
            }
            other => {
                input = Some(String::from(other));
                i += 1;
            }
        }
    }
    let input_path = input.ok_or_else(|| {
        EarthSnailError::InvalidParameter("'earthsnail pick' requires an <input> file".into())
    })?;
    let stream = read_waveforms_file(&input_path)?;

    let mut picks_arr = JsonArr::new();
    let mut text = String::new();

    for tr in &stream.traces {
        let fs = tr.stats.sampling_rate;
        let dt = tr.delta();
        let n = tr.npts();
        let (sample_opt, metric, quality) = match method.as_str() {
            "morse" => {
                let nlta = ((6.0 * fs) as usize).clamp(16, (n / 3).max(16));
                let nsta = ((1.0 * fs) as usize).clamp(4, nlta / 2);
                if n <= nlta + 4 {
                    (aic_pick(&tr.data), 0.0, String::from("AIC_FALLBACK"))
                } else {
                    let res = morse_pick(
                        &tr.data,
                        &MorsePickConfig {
                            nsta,
                            nlta,
                            thr_on: thr_opt.unwrap_or(3.5),
                            prominence_sigma: 4.0,
                            confirm_samples: 4,
                        },
                    )?;
                    (res.pick_sample, res.peak_ratio, String::from("MORSE"))
                }
            }
            "baer" => {
                let si = (math::round(fs) as usize).max(1);
                let res = pk_baer(
                    &tr.data,
                    &BaerPickParams {
                        samp_int: si,
                        tdownmax: (si / 5).max(2),
                        tupevent: (si / 2).max(4),
                        thr1: thr_opt.unwrap_or(7.0),
                        thr2: 12.0,
                        preset_len: si.min(n.saturating_sub(2)).max(1),
                        p_dur: si,
                    },
                )?;
                let peak = res
                    .characteristic_function
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max);
                (res.pick_sample, peak, res.first_motion)
            }
            "stalta" => {
                let nlta = ((10.0 * fs) as usize).clamp(10, (n / 2).max(10));
                let nsta = ((1.0 * fs) as usize).clamp(2, nlta / 2);
                let cft = classic_sta_lta(&tr.data, nsta, nlta)?;
                let thr = thr_opt.unwrap_or(3.0);
                let trigs = trigger_onset(&cft, thr, (0.4 * thr).max(0.8));
                let pk = cft.iter().copied().fold(0.0_f64, f64::max);
                (trigs.first().map(|p| p[0]), pk, String::from("STALTA"))
            }
            "aic" => (aic_pick(&tr.data), 0.0, String::from("AIC")),
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unsupported picker --method '{other}' (expected morse|baer|stalta|aic)"
                )));
            }
        };

        let rel_s = sample_opt.map(|k| (k as f64) * dt);
        let utc_str = sample_opt
            .and_then(|k| tr.sample_time(k).ok())
            .map(|t| t.to_iso8601());

        if json_mode {
            let mut po = JsonObj::new();
            po.str("channel", &tr.stats.channel.seed_id())
                .str("method", &method)
                .bool("picked", sample_opt.is_some());
            match sample_opt {
                Some(idx) => {
                    po.uint("pick_sample", idx);
                }
                None => {
                    po.raw("pick_sample", "null");
                }
            }
            po.opt_num("pick_rel_s", rel_s)
                .opt_str("pick_utc", utc_str.as_deref())
                .num("peak_metric", metric)
                .str("label", &quality);
            picks_arr.push_raw(&po.finish());
        } else if let (Some(idx), Some(rs), Some(utc)) = (sample_opt, rel_s, utc_str) {
            text.push_str(&format!(
                "{:<16} | method={:<6} | sample={:>7} | +{:.4} s | {} | {}\n",
                tr.stats.channel.seed_id(),
                method,
                idx,
                rs,
                utc,
                quality
            ));
        } else {
            text.push_str(&format!(
                "{:<16} | method={:<6} | no onset detected\n",
                tr.stats.channel.seed_id(),
                method
            ));
        }
    }

    if json_mode {
        let mut root = JsonObj::new();
        root.str("file", &input_path)
            .str("method", &method)
            .raw("picks", &picks_arr.finish());
        Ok(format!("{}\n", root.finish()))
    } else {
        Ok(text)
    }
}

// ============================================================================
// 6. `earthsnail taup time|pierce|path|curve|setsac|locate ...`
// ============================================================================

fn cmd_taup(args: &[String]) -> Result<String> {
    if args.is_empty() || has_help_flag(args) {
        return Ok(subcommand_help("taup"));
    }
    let sub = args[0].as_str();
    let tail = &args[1..];
    match sub {
        "time" => cmd_taup_calc("time", tail),
        "pierce" => cmd_taup_calc("pierce", tail),
        "path" => cmd_taup_calc("path", tail),
        "curve" => cmd_taup_curve(tail),
        "setsac" => cmd_taup_setsac(tail),
        "locate" => cmd_taup_locate(tail),
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unknown 'taup' subcommand '{other}' (expected time|pierce|path|curve|setsac|locate)"
        ))),
    }
}

fn load_tau_model_cli(spec: &str) -> Result<TauModel> {
    if let Ok(m) = TauModel::by_name(spec) {
        return Ok(m);
    }
    let txt = fs::read_to_string(spec)
        .map_err(|e| EarthSnailError::Io(format!("cannot read velocity model '{spec}': {e}")))?;
    let vmod = if spec.to_ascii_lowercase().ends_with(".tvel") {
        VelocityModel::from_tvel_str(spec, &txt)?
    } else {
        VelocityModel::from_nd_str(spec, &txt)?
    };
    TauModel::from_velocity_model(vmod)
}

fn cmd_taup_calc(mode: &str, args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("taup"));
    }
    let mut model_spec = String::from("iasp91");
    let mut evdp_km = 0.0_f64;
    let mut deg_opt: Option<f64> = None;
    let mut sta_latlon: Option<(f64, f64)> = None;
    let mut evt_latlon: Option<(f64, f64)> = None;
    let mut ellipticity = false;
    let mut phases: Vec<String> = Vec::new();
    let mut svg_out: Option<String> = None;
    let mut json_mode = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-mod" | "-m" => {
                model_spec = req_val(args, i, "--model")?;
                i += 2;
            }
            "--evdp" | "--depth" | "-h" => {
                evdp_km = parse_f64_str(&req_val(args, i, "--evdp")?, "--evdp")?;
                i += 2;
            }
            "--deg" | "-d" => {
                deg_opt = Some(parse_f64_str(&req_val(args, i, "--deg")?, "--deg")?);
                i += 2;
            }
            "--km" => {
                let km = parse_f64_str(&req_val(args, i, "--km")?, "--km")?;
                deg_opt = Some(kilometer2degrees(km));
                i += 2;
            }
            "--sta" => {
                sta_latlon = Some(parse_lat_lon(&req_val(args, i, "--sta")?)?);
                i += 2;
            }
            "--evt" => {
                evt_latlon = Some(parse_lat_lon(&req_val(args, i, "--evt")?)?);
                i += 2;
            }
            "--ellipticity" => {
                ellipticity = true;
                i += 1;
            }
            "--phases" | "-p" | "-ph" => {
                let raw = req_val(args, i, "--phases")?;
                for p in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    phases.push(String::from(p));
                }
                i += 2;
            }
            "-o" | "--svg" => {
                svg_out = Some(req_val(args, i, "-o")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'taup {mode}'"
                )));
            }
        }
    }

    if phases.is_empty() {
        phases = vec![String::from("ttbasic")];
    }
    let tm = load_tau_model_cli(&model_spec)?;

    match mode {
        "time" => {
            let arrivals = if let (Some((ela, elo)), Some((sla, slo))) = (evt_latlon, sta_latlon) {
                tm.get_travel_times_geo(ela, elo, evdp_km, sla, slo, &phases, ellipticity)?
            } else {
                let deg = deg_opt.ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "'taup time' requires --deg <deg>, --km <km>, or (--evt <lat,lon> --sta <lat,lon>)"
                            .into(),
                    )
                })?;
                tm.get_travel_times(evdp_km, deg, &phases)?
            };

            if json_mode {
                let mut arr = JsonArr::new();
                for a in &arrivals {
                    let mut ao = JsonObj::new();
                    ao.str("name", &a.name)
                        .str("purist_name", &a.purist_name)
                        .num("time_s", a.time_s)
                        .num("ray_param_s_per_rad", a.ray_param_s_per_rad)
                        .num("ray_param_s_per_deg", a.ray_param_s_per_deg)
                        .num("takeoff_angle_deg", a.takeoff_angle_deg)
                        .num("incident_angle_deg", a.incident_angle_deg)
                        .num("distance_deg", a.distance_deg)
                        .num("purist_distance_deg", a.purist_distance_deg)
                        .num("source_depth_km", a.source_depth_km)
                        .opt_num("ellipticity_corr_s", a.ellipticity_corr_s);
                    arr.push_raw(&ao.finish());
                }
                let mut root = JsonObj::new();
                root.str("model", &model_spec)
                    .num("source_depth_km", evdp_km)
                    .uint("num_arrivals", arrivals.len())
                    .raw("arrivals", &arr.finish());
                Ok(format!("{}\n", root.finish()))
            } else {
                let mut out = format!(
                    "Model: {model_spec}  |  Depth: {evdp_km:.1} km  |  {} arrival(s)\n\
                     Distance   Depth   Phase       Travel(s)   p(s/deg)   Takeoff   Incident   Purist\n\
                     -----------------------------------------------------------------------------------\n",
                    arrivals.len()
                );
                for a in &arrivals {
                    out.push_str(&format!(
                        "{:>8.2} {:>7.1}   {:<10} {:>10.4} {:>10.4} {:>9.2} {:>10.2}   = {}\n",
                        a.distance_deg,
                        a.source_depth_km,
                        a.name,
                        a.time_s,
                        a.ray_param_s_per_deg,
                        a.takeoff_angle_deg,
                        a.incident_angle_deg,
                        a.purist_name
                    ));
                }
                Ok(out)
            }
        }
        "pierce" => {
            let pierces = if let (Some((ela, elo)), Some((sla, slo))) = (evt_latlon, sta_latlon) {
                tm.get_pierce_points_geo(ela, elo, evdp_km, sla, slo, &phases, ellipticity)?
            } else {
                let deg = deg_opt.ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "'taup pierce' requires --deg <deg> or (--evt / --sta)".into(),
                    )
                })?;
                tm.get_pierce_points(evdp_km, deg, &phases)?
            };

            if json_mode {
                let mut arr = JsonArr::new();
                for ap in &pierces {
                    let mut pts = JsonArr::new();
                    for p in &ap.pierce_points {
                        let mut po = JsonObj::new();
                        po.num("dist_deg", p.dist_deg)
                            .num("depth_km", p.depth_km)
                            .num("time_s", p.time_s)
                            .opt_num("lat_deg", p.lat_deg)
                            .opt_num("lon_deg", p.lon_deg);
                        pts.push_raw(&po.finish());
                    }
                    let mut o = JsonObj::new();
                    o.str("name", &ap.arrival.name)
                        .num("time_s", ap.arrival.time_s)
                        .raw("pierce_points", &pts.finish());
                    arr.push_raw(&o.finish());
                }
                let mut root = JsonObj::new();
                root.str("model", &model_spec)
                    .raw("arrivals", &arr.finish());
                Ok(format!("{}\n", root.finish()))
            } else {
                let mut out = String::new();
                for ap in &pierces {
                    out.push_str(&format!(
                        "> {} at {:.3} s ({:.2} deg)\n",
                        ap.arrival.name, ap.arrival.time_s, ap.arrival.distance_deg
                    ));
                    for p in &ap.pierce_points {
                        out.push_str(&format!(
                            "  {:>8.2} deg {:>8.1} km {:>9.3} s\n",
                            p.dist_deg, p.depth_km, p.time_s
                        ));
                    }
                }
                Ok(out)
            }
        }
        "path" => {
            let paths = if let (Some((ela, elo)), Some((sla, slo))) = (evt_latlon, sta_latlon) {
                tm.get_ray_paths_geo(ela, elo, evdp_km, sla, slo, &phases, ellipticity)?
            } else {
                let deg = deg_opt.ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "'taup path' requires --deg <deg> or (--evt / --sta)".into(),
                    )
                })?;
                tm.get_ray_paths(evdp_km, deg, &phases)?
            };

            if let Some(ref svg_path) = svg_out {
                ensure_parent_dir(svg_path)?;
                let svg =
                    plot_taup_paths_svg(&paths, &model_spec, &TaupPathsPlotOptions::default());
                fs::write(svg_path, svg.as_bytes())
                    .map_err(|e| EarthSnailError::Io(format!("cannot write '{svg_path}': {e}")))?;
            }

            if json_mode {
                let mut arr = JsonArr::new();
                for ap in &paths {
                    let mut pts = JsonArr::new();
                    for p in &ap.path_points {
                        let mut po = JsonObj::new();
                        po.num("dist_deg", p.dist_deg)
                            .num("depth_km", p.depth_km)
                            .num("radius_km", p.radius_km)
                            .num("time_s", p.time_s);
                        pts.push_raw(&po.finish());
                    }
                    let mut o = JsonObj::new();
                    o.str("name", &ap.arrival.name)
                        .num("time_s", ap.arrival.time_s)
                        .uint("num_points", ap.path_points.len())
                        .raw("path_points", &pts.finish());
                    arr.push_raw(&o.finish());
                }
                let mut root = JsonObj::new();
                root.str("model", &model_spec)
                    .opt_str("svg", svg_out.as_deref())
                    .raw("paths", &arr.finish());
                Ok(format!("{}\n", root.finish()))
            } else {
                let mut out = format!(
                    "Computed {} ray path(s) on model '{}'\n",
                    paths.len(),
                    model_spec
                );
                for ap in &paths {
                    out.push_str(&format!(
                        "  {:<10} time={:.3}s  points={}\n",
                        ap.arrival.name,
                        ap.arrival.time_s,
                        ap.path_points.len()
                    ));
                }
                Ok(out)
            }
        }
        _ => unreachable!(),
    }
}

fn cmd_taup_curve(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("taup"));
    }
    let mut model_spec = String::from("iasp91");
    let mut evdp_km = 0.0_f64;
    let mut phases = vec![String::from("P"), String::from("S")];
    let mut n_points = 64usize;
    let mut svg_out: Option<String> = None;
    let mut json_mode = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model_spec = req_val(args, i, "--model")?;
                i += 2;
            }
            "--evdp" | "-h" => {
                evdp_km = parse_f64_str(&req_val(args, i, "--evdp")?, "--evdp")?;
                i += 2;
            }
            "--phases" | "-p" => {
                phases = req_val(args, i, "--phases")?
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                i += 2;
            }
            "--points" | "-n" => {
                n_points = parse_usize_str(&req_val(args, i, "--points")?, "--points")?;
                i += 2;
            }
            "-o" | "--svg" => {
                svg_out = Some(req_val(args, i, "-o")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'taup curve'"
                )));
            }
        }
    }

    let tm = load_tau_model_cli(&model_spec)?;
    let curves = tm.get_travel_time_curves(evdp_km, &phases, n_points)?;
    if let Some(ref svg_path) = svg_out {
        ensure_parent_dir(svg_path)?;
        let svg =
            plot_record_section_svg(&Stream::new(), &curves, &RecordSectionOptions::default());
        fs::write(svg_path, svg.as_bytes())
            .map_err(|e| EarthSnailError::Io(format!("cannot write '{svg_path}': {e}")))?;
    }

    if json_mode {
        let mut c_arr = JsonArr::new();
        for c in &curves {
            let mut b_arr = JsonArr::new();
            for br in &c.branches {
                let mut p_arr = JsonArr::new();
                for pt in &br.points {
                    let mut po = JsonObj::new();
                    po.num("dist_deg", pt.dist_deg)
                        .num("time_s", pt.time_s)
                        .num("tau_s", pt.tau_s)
                        .num("ray_param_s_per_rad", pt.ray_param_s_per_rad);
                    p_arr.push_raw(&po.finish());
                }
                b_arr.push_raw(&p_arr.finish());
            }
            let mut co = JsonObj::new();
            co.str("name", &c.name)
                .num("min_distance_deg", c.min_distance_deg)
                .num("max_distance_deg", c.max_distance_deg)
                .raw("branches", &b_arr.finish());
            c_arr.push_raw(&co.finish());
        }
        let mut root = JsonObj::new();
        root.str("model", &model_spec)
            .num("source_depth_km", evdp_km)
            .raw("curves", &c_arr.finish());
        Ok(format!("{}\n", root.finish()))
    } else {
        let mut out = format!("Model: {model_spec} | curves: {}\n", curves.len());
        for c in &curves {
            out.push_str(&format!(
                "  {:<10} [{:.2}° .. {:.2}°]  ({} branch(es))\n",
                c.name,
                c.min_distance_deg,
                c.max_distance_deg,
                c.branches.len()
            ));
        }
        Ok(out)
    }
}

fn cmd_taup_setsac(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("taup"));
    }
    let mut model_spec = String::from("iasp91");
    let mut phases: Vec<String> = vec![
        String::from("P"),
        String::from("S"),
        String::from("PP"),
        String::from("PcP"),
        String::from("SKS"),
    ];
    let mut evdpkm = false;
    let mut out_override: Option<String> = None;
    let mut json_mode = false;
    let mut files: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model_spec = req_val(args, i, "--model")?;
                i += 2;
            }
            "--phases" | "-p" | "-ph" => {
                phases = req_val(args, i, "--phases")?
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                i += 2;
            }
            "--evdpkm" => {
                evdpkm = true;
                i += 1;
            }
            "-o" | "--output" => {
                out_override = Some(req_val(args, i, "-o")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'taup setsac'"
                )));
            }
            other => {
                files.push(String::from(other));
                i += 1;
            }
        }
    }
    if files.is_empty() {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail taup setsac' requires one or more SAC files".into(),
        ));
    }
    if out_override.is_some() && files.len() != 1 {
        return Err(EarthSnailError::InvalidParameter(
            "'-o' output path in 'taup setsac' is only supported when stamping a single input file"
                .into(),
        ));
    }

    let tm = load_tau_model_cli(&model_spec)?;
    let mut engine = TauEngine::new(tm)?;

    let mut f_arr = JsonArr::new();
    let mut text = String::new();

    for path in &files {
        let raw = fs::read(path)
            .map_err(|e| EarthSnailError::Io(format!("cannot read '{path}': {e}")))?;
        let endian = SacHeader::detect_endian(&raw).unwrap_or(SacEndian::Little);
        let mut st = read_waveforms_bytes(&raw)?;
        if st.is_empty() {
            return Err(EarthSnailError::EmptyData(format!(
                "no SAC waveform in '{path}'"
            )));
        }
        let tr = &mut st[0];
        if tr.stats.sac.is_none() {
            tr.stats.sync_to_sac_header(&tr.data);
        }
        let stamped = stamp_taup_into_sac_header(
            tr.stats.sac.as_mut().unwrap(),
            &mut engine,
            &phases,
            evdpkm,
        )?;
        let dst = out_override.as_deref().unwrap_or(path.as_str());
        ensure_parent_dir(dst)?;
        write_sac_file(dst, tr, endian)?;

        if json_mode {
            let mut m_arr = JsonArr::new();
            for (slot, ph, t_val) in &stamped {
                let mut mo = JsonObj::new();
                mo.str("slot", slot).str("phase", ph).num("time_s", *t_val);
                m_arr.push_raw(&mo.finish());
            }
            let mut fo = JsonObj::new();
            fo.str("input", path)
                .str("output", dst)
                .uint("stamped_count", stamped.len())
                .raw("markers", &m_arr.finish());
            f_arr.push_raw(&fo.finish());
        } else {
            let s_list: Vec<String> = stamped
                .iter()
                .map(|(sl, ph, tv)| format!("{sl}={ph}({tv:.3}s)"))
                .collect();
            text.push_str(&format!("{dst}: {}\n", s_list.join(", ")));
        }
    }

    if json_mode {
        let mut root = JsonObj::new();
        root.str("model", &model_spec).raw("files", &f_arr.finish());
        Ok(format!("{}\n", root.finish()))
    } else {
        Ok(text)
    }
}

fn cmd_taup_locate(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("taup"));
    }
    let mut model_spec = String::from("iasp91");
    let mut obs_file: Option<String> = None;
    let mut ellipticity = false;
    let mut use_3d = false;
    let mut json_mode = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                model_spec = req_val(args, i, "--model")?;
                i += 2;
            }
            "--obs" | "-i" => {
                obs_file = Some(req_val(args, i, "--obs")?);
                i += 2;
            }
            "--ellipticity" => {
                ellipticity = true;
                i += 1;
            }
            "--3d" => {
                use_3d = true;
                i += 1;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'taup locate'"
                )));
            }
        }
    }

    let obs_path = obs_file.ok_or_else(|| {
        EarthSnailError::InvalidParameter(
            "'earthsnail taup locate' requires --obs <file> (lines: STA LAT LON PHASE TIME_S [UNC_S])"
                .into(),
        )
    })?;
    let txt = fs::read_to_string(&obs_path)
        .map_err(|e| EarthSnailError::Io(format!("cannot read '{obs_path}': {e}")))?;
    let mut observations = Vec::new();
    for (ln, raw_line) in txt.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .collect();
        if cols.len() < 5 {
            return Err(EarthSnailError::Format(format!(
                "obs line {}: expected 'STA LAT LON PHASE TIME_S [UNC_S]', got '{line}'",
                ln + 1
            )));
        }
        let sta = cols[0];
        let lat = parse_f64_str(cols[1], "obs lat")?;
        let lon = parse_f64_str(cols[2], "obs lon")?;
        let ph = cols[3];
        let t_s = parse_f64_str(cols[4], "obs time_s")?;
        let unc = cols
            .get(5)
            .map(|u| parse_f64_str(u, "obs unc_s"))
            .transpose()?
            .unwrap_or(0.1);
        observations.push(PhaseArrivalObservation::new(sta, lat, lon, ph, t_s, unc));
    }

    let tm = load_tau_model_cli(&model_spec)?;
    let pert3d = SubductionPlumeModel3D::pacific_ring_default();
    let cfg = HypocenterConfig {
        apply_ellipticity: ellipticity,
        perturbation_3d: if use_3d { Some(&pert3d) } else { None },
        ..HypocenterConfig::default()
    };
    let sol = locate_hypocenter(&tm, &observations, &cfg)?;

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("model", &model_spec)
            .num("lat_deg", sol.lat_deg)
            .num("lon_deg", sol.lon_deg)
            .num("depth_km", sol.depth_km)
            .num("origin_time_s", sol.origin_time_s)
            .num("weighted_rms_s", sol.weighted_rms_s)
            .uint("iterations", sol.iterations)
            .num("azimuthal_gap_deg", sol.azimuthal_gap_deg)
            .raw(
                "semi_axes_95pct_km",
                &json_f64_slice(&sol.ellipsoid.semi_axes_95pct_km),
            );
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "Hypocenter: lat={:.4}° lon={:.4}° depth={:.2} km t0={:.3} s (wRMS={:.4} s, iters={})\n",
            sol.lat_deg, sol.lon_deg, sol.depth_km, sol.origin_time_s, sol.weighted_rms_s, sol.iterations
        ))
    }
}

// ============================================================================
// 7. `earthsnail ppsd <waveform> (--stationxml <f> | --sacpz <f>) --svg <out.svg>`
// ============================================================================

fn cmd_ppsd(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("ppsd"));
    }
    let mut stationxml_path: Option<String> = None;
    let mut sacpz_path: Option<String> = None;
    let mut svg_out: Option<String> = None;
    let mut seg_len_opt: Option<f64> = None;
    let mut overlap = 0.5_f64;
    let mut json_mode = false;
    let mut waveform: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--stationxml" | "--xml" => {
                stationxml_path = Some(req_val(args, i, "--stationxml")?);
                i += 2;
            }
            "--sacpz" | "--pz" => {
                sacpz_path = Some(req_val(args, i, "--sacpz")?);
                i += 2;
            }
            "--svg" | "-o" => {
                svg_out = Some(req_val(args, i, "--svg")?);
                i += 2;
            }
            "--segment-len" => {
                seg_len_opt = Some(parse_f64_str(
                    &req_val(args, i, "--segment-len")?,
                    "--segment-len",
                )?);
                i += 2;
            }
            "--overlap" => {
                overlap = parse_f64_str(&req_val(args, i, "--overlap")?, "--overlap")?;
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'ppsd'"
                )));
            }
            other => {
                waveform = Some(String::from(other));
                i += 1;
            }
        }
    }
    let wf_path = waveform.ok_or_else(|| {
        EarthSnailError::InvalidParameter("'earthsnail ppsd' requires <waveform>".into())
    })?;
    let svg_path = svg_out.ok_or_else(|| {
        EarthSnailError::InvalidParameter("'earthsnail ppsd' requires --svg <out.svg>".into())
    })?;

    let stream = read_waveforms_file(&wf_path)?;
    if stream.is_empty() {
        return Err(EarthSnailError::EmptyData("waveform file is empty".into()));
    }
    let tr = &stream[0];
    let inv_opt = match stationxml_path {
        Some(ref p) => Some(read_stationxml_file(p)?),
        None => None,
    };
    let pz_opt = match sacpz_path {
        Some(ref p) => Some(read_sacpz_file(p)?.response),
        None => None,
    };
    let resp = resolve_channel_response_for_trace(tr, inv_opt.as_ref(), pz_opt.as_ref())?;

    let dur_s = tr.duration_s();
    let seg_len_s = seg_len_opt.unwrap_or_else(|| {
        if dur_s >= 3600.0 {
            3600.0
        } else {
            (dur_s * 0.5).max(16.0 / tr.stats.sampling_rate)
        }
    });
    let mut ppsd = Ppsd::new(
        tr.stats.sampling_rate,
        resp,
        PpsdConfig {
            ppsd_length_s: seg_len_s,
            overlap,
            ..PpsdConfig::default()
        },
    )?;
    ppsd.add_stream(&stream)?;

    ensure_parent_dir(&svg_path)?;
    let svg = plot_ppsd_svg(&ppsd, &PpsdPlotOptions::default());
    fs::write(&svg_path, svg.as_bytes())
        .map_err(|e| EarthSnailError::Io(format!("cannot write '{svg_path}': {e}")))?;

    let mean_db = ppsd.mean_curve_db().unwrap_or_default();
    let mode_db = ppsd.mode_curve_db().unwrap_or_default();

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("waveform", &wf_path)
            .str("svg", &svg_path)
            .uint("num_segments", ppsd.num_segments())
            .raw("periods_s", &json_f64_slice(ppsd.periods_s()))
            .raw("mean_db", &json_f64_slice(&mean_db))
            .raw("mode_db", &json_f64_slice(&mode_db));
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "PPSD accumulated {} segment(s) ({} period bins) -> '{}'\n",
            ppsd.num_segments(),
            ppsd.periods_s().len(),
            svg_path
        ))
    }
}

// ============================================================================
// 8. `earthsnail rf <z_file> <r_file> --svg <out.svg> [--method iter|waterlevel]`
// ============================================================================

fn cmd_rf(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("rf"));
    }
    let mut method = String::from("iter");
    let mut gauss_a = 2.5_f64;
    let mut shift_s = 5.0_f64;
    let mut water_level = 0.01_f64;
    let mut max_bumps = 200usize;
    let mut svg_out: Option<String> = None;
    let mut rf_out: Option<String> = None;
    let mut json_mode = false;
    let mut pos: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--method" => {
                method = req_val(args, i, "--method")?.to_ascii_lowercase();
                i += 2;
            }
            "--gauss" | "-a" => {
                gauss_a = parse_f64_str(&req_val(args, i, "--gauss")?, "--gauss")?;
                i += 2;
            }
            "--shift" => {
                shift_s = parse_f64_str(&req_val(args, i, "--shift")?, "--shift")?;
                i += 2;
            }
            "--water-level" => {
                water_level = parse_f64_str(&req_val(args, i, "--water-level")?, "--water-level")?;
                i += 2;
            }
            "--max-bumps" => {
                max_bumps = parse_usize_str(&req_val(args, i, "--max-bumps")?, "--max-bumps")?;
                i += 2;
            }
            "--svg" => {
                svg_out = Some(req_val(args, i, "--svg")?);
                i += 2;
            }
            "-o" | "--output" => {
                rf_out = Some(req_val(args, i, "-o")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'rf'"
                )));
            }
            other => {
                pos.push(String::from(other));
                i += 1;
            }
        }
    }
    if pos.len() != 2 {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail rf' expects <z_file> <r_file>".into(),
        ));
    }
    let svg_path = svg_out.ok_or_else(|| {
        EarthSnailError::InvalidParameter("'earthsnail rf' requires --svg <out.svg>".into())
    })?;

    let z_st = read_waveforms_file(&pos[0])?;
    let r_st = read_waveforms_file(&pos[1])?;
    if z_st.is_empty() || r_st.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "empty Z or R waveform file".into(),
        ));
    }
    let z_tr = &z_st[0];
    let r_tr = &r_st[0];
    let fs = z_tr.stats.sampling_rate;

    let (rf_data, fit_pct, iters) = match method.as_str() {
        "iter" | "iterative" => {
            let res = receiver_function_iter(
                &z_tr.data, &r_tr.data, fs, gauss_a, shift_s, max_bumps, 0.001, false,
            )?;
            (res.receiver_function, res.fit_percent, res.iterations)
        }
        "waterlevel" | "wl" => {
            let rf = receiver_function_waterlevel(
                &z_tr.data,
                &r_tr.data,
                fs,
                gauss_a,
                water_level,
                shift_s,
            )?;
            (rf, 100.0, 1)
        }
        other => {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unsupported RF --method '{other}' (expected iter|waterlevel)"
            )));
        }
    };

    let rf_start = z_tr.stats.starttime.add_secs_f64(-shift_s)?;
    let mut rf_stats = TraceStats::new(
        ChannelId::new(
            &r_tr.stats.channel.network,
            &r_tr.stats.channel.station,
            &r_tr.stats.channel.location,
            "RFR",
        ),
        rf_start,
        fs,
    );
    rf_stats.sync_to_sac_header(&rf_data);
    if let Some(ref mut sac) = rf_stats.sac {
        sac.b = -shift_s;
        sac.e = -shift_s + ((rf_data.len().saturating_sub(1)) as f64) / fs;
        let _ = sac.set_float("o", 0.0);
    }
    let rf_trace = Trace::new(rf_stats, rf_data);

    if let Some(ref out_sac) = rf_out {
        ensure_parent_dir(out_sac)?;
        write_sac_file(out_sac, &rf_trace, SacEndian::Little)?;
    }

    ensure_parent_dir(&svg_path)?;
    let fig_stream = Stream::from_traces(vec![z_tr.clone(), r_tr.clone(), rf_trace.clone()]);
    let svg = plot_waveforms_svg(
        &fig_stream,
        &WaveformPlotOptions {
            title: Some(format!(
                "P-Receiver Function ({method}, a={gauss_a:.2}, fit={fit_pct:.1}%)"
            )),
            ..WaveformPlotOptions::default()
        },
    );
    fs::write(&svg_path, svg.as_bytes())
        .map_err(|e| EarthSnailError::Io(format!("cannot write '{svg_path}': {e}")))?;

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("method", &method)
            .str("svg", &svg_path)
            .num("fit_percent", fit_pct)
            .uint("iterations", iters)
            .uint("npts", rf_trace.npts());
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "Receiver Function ({method}): fit={fit_pct:.2}%, iterations={iters} -> '{svg_path}'\n"
        ))
    }
}

// ============================================================================
// 9. `earthsnail fk <files...> [--capon] [--json]`
// ============================================================================

fn cmd_fk(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("fk"));
    }
    let mut capon = false;
    let mut fmin = 0.5_f64;
    let mut fmax = 4.0_f64;
    let mut smax = 0.4_f64;
    let mut grid = 41usize;
    let mut json_mode = false;
    let mut files: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--capon" | "--mvdr" => {
                capon = true;
                i += 1;
            }
            "--fmin" => {
                fmin = parse_f64_str(&req_val(args, i, "--fmin")?, "--fmin")?;
                i += 2;
            }
            "--fmax" => {
                fmax = parse_f64_str(&req_val(args, i, "--fmax")?, "--fmax")?;
                i += 2;
            }
            "--smax" => {
                smax = parse_f64_str(&req_val(args, i, "--smax")?, "--smax")?;
                i += 2;
            }
            "--grid" => {
                grid = parse_usize_str(&req_val(args, i, "--grid")?, "--grid")?;
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'fk'"
                )));
            }
            other => {
                files.push(String::from(other));
                i += 1;
            }
        }
    }
    if files.len() < 2 {
        return Err(EarthSnailError::InvalidParameter(
            "'earthsnail fk' requires at least 2 station waveform files with STLA/STLO headers"
                .into(),
        ));
    }

    let mut loaded: Vec<Trace> = Vec::new();
    for f in &files {
        let st = read_waveforms_file(f)?;
        loaded.extend(st.traces);
    }
    if loaded.len() < 2 {
        return Err(EarthSnailError::InvalidParameter(
            "fewer than 2 traces loaded for FK beamforming".into(),
        ));
    }

    // Project station coordinates relative to first station in (East_km, North_km):
    let mut latlons = Vec::with_capacity(loaded.len());
    for tr in &loaded {
        let sac = tr.stats.sac.as_ref().ok_or_else(|| {
            EarthSnailError::Header("FK beamforming requires SAC STLA/STLO headers".into())
        })?;
        let la = sac.get_float("stla").ok_or_else(|| {
            EarthSnailError::Header(format!("missing STLA on '{}'", tr.stats.channel))
        })?;
        let lo = sac.get_float("stlo").ok_or_else(|| {
            EarthSnailError::Header(format!("missing STLO on '{}'", tr.stats.channel))
        })?;
        latlons.push((la, lo));
    }
    let (lat0, lon0) = latlons[0];
    let mut stations = Vec::with_capacity(loaded.len());
    for &(la, lo) in &latlons {
        let geo = gps2dist_azimuth(lat0, lon0, la, lo);
        let d_km = geo.distance_m * 1e-3;
        let (saz, caz) = math::sin_cos(geo.azimuth_deg.to_radians());
        stations.push(ArrayStation::new(d_km * saz, d_km * caz));
    }

    let min_len = loaded.iter().map(Trace::npts).min().unwrap_or(0);
    let slices: Vec<&[f64]> = loaded.iter().map(|t| &t.data[..min_len]).collect();
    let method = if capon {
        FkMethod::CaponMvdr {
            diagonal_loading: 1e-2,
        }
    } else {
        FkMethod::Conventional
    };
    let res = fk_analysis(
        &stations,
        &slices,
        loaded[0].stats.sampling_rate,
        fmin,
        fmax,
        smax,
        grid,
        method,
    )?;
    let app_vel = if res.horizontal_slowness_s_per_km > 1e-9 {
        1.0 / res.horizontal_slowness_s_per_km
    } else {
        f64::INFINITY
    };

    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("method", if capon { "capon_mvdr" } else { "conventional" })
            .uint("num_stations", stations.len())
            .num("back_azimuth_deg", res.back_azimuth_deg)
            .num(
                "horizontal_slowness_s_per_km",
                res.horizontal_slowness_s_per_km,
            )
            .num("apparent_velocity_km_s", app_vel)
            .num("semblance", res.semblance)
            .num("abs_power", res.abs_power);
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "FK Analysis ({}): BAZ={:.2}°  slowness={:.4} s/km  Vapp={:.2} km/s  semblance={:.4}\n",
            if capon { "Capon MVDR" } else { "Conventional" },
            res.back_azimuth_deg,
            res.horizontal_slowness_s_per_km,
            app_vel,
            res.semblance
        ))
    }
}

// ============================================================================
// 10. `earthsnail plot waveform|section|spectrogram|ppsd|taup|beachball ...`
// ============================================================================

fn cmd_plot(args: &[String]) -> Result<String> {
    if args.is_empty() || has_help_flag(args) {
        return Ok(subcommand_help("plot"));
    }
    let kind = args[0].as_str();
    let tail = &args[1..];

    match kind {
        "waveform" | "waveforms" | "p1" | "p2" => {
            let mut out_svg: Option<String> = None;
            let mut overlaid = kind == "p2";
            let mut utc = false;
            let mut json_mode = false;
            let mut files = Vec::new();
            let mut i = 0;
            while i < tail.len() {
                match tail[i].as_str() {
                    "-o" | "--svg" | "--output" => {
                        out_svg = Some(req_val(tail, i, "-o")?);
                        i += 2;
                    }
                    "--overlaid" | "--p2" => {
                        overlaid = true;
                        i += 1;
                    }
                    "--utc" => {
                        utc = true;
                        i += 1;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    f if !f.starts_with('-') => {
                        files.push(String::from(f));
                        i += 1;
                    }
                    other => {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown flag '{other}' in 'plot waveform'"
                        )));
                    }
                }
            }
            let dst = out_svg.ok_or_else(|| {
                EarthSnailError::InvalidParameter("'plot waveform' requires -o <out.svg>".into())
            })?;
            let mut all_traces = Vec::new();
            for f in &files {
                all_traces.extend(read_waveforms_file(f)?.traces);
            }
            let st = Stream::from_traces(all_traces);
            let svg = plot_waveforms_svg(
                &st,
                &WaveformPlotOptions {
                    layout: if overlaid {
                        WaveformLayout::Overlaid
                    } else {
                        WaveformLayout::Stacked
                    },
                    time_axis: if utc {
                        TimeAxisMode::UtcClock
                    } else {
                        TimeAxisMode::RelativeSeconds
                    },
                    ..WaveformPlotOptions::default()
                },
            );
            ensure_parent_dir(&dst)?;
            fs::write(&dst, svg.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{dst}': {e}")))?;
            finish_plot_msg("waveform", &dst, svg.len(), json_mode)
        }
        "section" => {
            let mut out_svg: Option<String> = None;
            let mut model_spec = String::from("iasp91");
            let mut phases = vec![String::from("P"), String::from("S")];
            let mut reduce: Option<f64> = None;
            let mut km = false;
            let mut json_mode = false;
            let mut files = Vec::new();
            let mut i = 0;
            while i < tail.len() {
                match tail[i].as_str() {
                    "-o" | "--svg" => {
                        out_svg = Some(req_val(tail, i, "-o")?);
                        i += 2;
                    }
                    "--model" => {
                        model_spec = req_val(tail, i, "--model")?;
                        i += 2;
                    }
                    "--phases" | "-p" => {
                        phases = req_val(tail, i, "--phases")?
                            .split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                            .collect();
                        i += 2;
                    }
                    "--reduce" => {
                        reduce = Some(parse_f64_str(&req_val(tail, i, "--reduce")?, "--reduce")?);
                        i += 2;
                    }
                    "--km" => {
                        km = true;
                        i += 1;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    f if !f.starts_with('-') => {
                        files.push(String::from(f));
                        i += 1;
                    }
                    other => {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown flag '{other}' in 'plot section'"
                        )));
                    }
                }
            }
            let dst = out_svg.ok_or_else(|| {
                EarthSnailError::InvalidParameter("'plot section' requires -o <out.svg>".into())
            })?;
            let mut all_traces = Vec::new();
            for f in &files {
                all_traces.extend(read_waveforms_file(f)?.traces);
            }
            let evdp = all_traces
                .first()
                .and_then(|t| t.stats.sac.as_ref())
                .and_then(|s| s.get_float("evdp"))
                .map(|d| if d > 1000.0 { d * 0.001 } else { d })
                .unwrap_or(10.0)
                .max(0.0);
            let tm = load_tau_model_cli(&model_spec)?;
            let curves = tm.get_travel_time_curves(evdp, &phases, 64)?;
            let svg = plot_record_section_svg(
                &Stream::from_traces(all_traces),
                &curves,
                &RecordSectionOptions {
                    distance_unit: if km {
                        DistanceUnit::Kilometers
                    } else {
                        DistanceUnit::Degrees
                    },
                    reduction_velocity_km_s: reduce,
                    ..RecordSectionOptions::default()
                },
            );
            ensure_parent_dir(&dst)?;
            fs::write(&dst, svg.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{dst}': {e}")))?;
            finish_plot_msg("section", &dst, svg.len(), json_mode)
        }
        "spectrogram" => {
            let mut out_svg: Option<String> = None;
            let mut opts = SpectrogramOptions::default();
            let mut json_mode = false;
            let mut file_in: Option<String> = None;
            let mut i = 0;
            while i < tail.len() {
                match tail[i].as_str() {
                    "-o" | "--svg" => {
                        out_svg = Some(req_val(tail, i, "-o")?);
                        i += 2;
                    }
                    "--cwt" => {
                        opts.method = SpectrogramMethod::MorletCwt { omega0: 6.0 };
                        i += 1;
                    }
                    "--stft" => {
                        opts.method = SpectrogramMethod::Stft { win_samples: 0 };
                        i += 1;
                    }
                    "--linear" => {
                        opts.freq_scale = FrequencyScale::Linear;
                        i += 1;
                    }
                    "--log" => {
                        opts.freq_scale = FrequencyScale::Log10;
                        i += 1;
                    }
                    "--fmin" => {
                        opts.fmin_hz =
                            Some(parse_f64_str(&req_val(tail, i, "--fmin")?, "--fmin")?);
                        i += 2;
                    }
                    "--fmax" => {
                        opts.fmax_hz =
                            Some(parse_f64_str(&req_val(tail, i, "--fmax")?, "--fmax")?);
                        i += 2;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    f if !f.starts_with('-') => {
                        file_in = Some(String::from(f));
                        i += 1;
                    }
                    other => {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown flag '{other}' in 'plot spectrogram'"
                        )));
                    }
                }
            }
            let src = file_in.ok_or_else(|| {
                EarthSnailError::InvalidParameter("'plot spectrogram' requires <file>".into())
            })?;
            let dst = out_svg.ok_or_else(|| {
                EarthSnailError::InvalidParameter(
                    "'plot spectrogram' requires -o <out.svg>".into(),
                )
            })?;
            let st = read_waveforms_file(&src)?;
            if st.is_empty() {
                return Err(EarthSnailError::EmptyData("empty waveform file".into()));
            }
            let svg = plot_spectrogram_svg(&st[0], &opts);
            ensure_parent_dir(&dst)?;
            fs::write(&dst, svg.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{dst}': {e}")))?;
            finish_plot_msg("spectrogram", &dst, svg.len(), json_mode)
        }
        "ppsd" => {
            // Normalize `-o` -> `--svg` and delegate to `cmd_ppsd`:
            let mapped: Vec<String> = tail
                .iter()
                .map(|a| if a == "-o" { String::from("--svg") } else { a.clone() })
                .collect();
            cmd_ppsd(&mapped)
        }
        "taup" => {
            let mut out_svg: Option<String> = None;
            let mut model_spec = String::from("iasp91");
            let mut evdp = 29.0_f64;
            let mut deg = 74.0_f64;
            let mut phases = vec![
                String::from("P"),
                String::from("S"),
                String::from("PP"),
                String::from("PcP"),
                String::from("SKS"),
            ];
            let mut json_mode = false;
            let mut i = 0;
            while i < tail.len() {
                match tail[i].as_str() {
                    "-o" | "--svg" => {
                        out_svg = Some(req_val(tail, i, "-o")?);
                        i += 2;
                    }
                    "--model" | "-m" => {
                        model_spec = req_val(tail, i, "--model")?;
                        i += 2;
                    }
                    "--evdp" | "-h" => {
                        evdp = parse_f64_str(&req_val(tail, i, "--evdp")?, "--evdp")?;
                        i += 2;
                    }
                    "--deg" | "-d" => {
                        deg = parse_f64_str(&req_val(tail, i, "--deg")?, "--deg")?;
                        i += 2;
                    }
                    "--phases" | "-p" => {
                        phases = req_val(tail, i, "--phases")?
                            .split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                            .collect();
                        i += 2;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    other => {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown flag '{other}' in 'plot taup'"
                        )));
                    }
                }
            }
            let dst = out_svg.ok_or_else(|| {
                EarthSnailError::InvalidParameter("'plot taup' requires -o <out.svg>".into())
            })?;
            let tm = load_tau_model_cli(&model_spec)?;
            let paths = tm.get_ray_paths(evdp, deg, &phases)?;
            let svg = plot_taup_paths_svg(&paths, &model_spec, &TaupPathsPlotOptions::default());
            ensure_parent_dir(&dst)?;
            fs::write(&dst, svg.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{dst}': {e}")))?;
            finish_plot_msg("taup", &dst, svg.len(), json_mode)
        }
        "beachball" => {
            let mut out_svg: Option<String> = None;
            let mut fm_in: Option<FocalMechanismInput> = None;
            let mut json_mode = false;
            let mut i = 0;
            while i < tail.len() {
                match tail[i].as_str() {
                    "-o" | "--svg" => {
                        out_svg = Some(req_val(tail, i, "-o")?);
                        i += 2;
                    }
                    "--sdr" => {
                        let v: Vec<f64> = req_val(tail, i, "--sdr")?
                            .split(',')
                            .map(|s| parse_f64_str(s.trim(), "--sdr"))
                            .collect::<Result<_>>()?;
                        if v.len() != 3 {
                            return Err(EarthSnailError::InvalidParameter(
                                "--sdr expects strike,dip,rake".into(),
                            ));
                        }
                        fm_in = Some(FocalMechanismInput::StrikeDipRake(NodalPlane {
                            strike_deg: v[0],
                            dip_deg: v[1],
                            rake_deg: v[2],
                        }));
                        i += 2;
                    }
                    "--mt" => {
                        let v: Vec<f64> = req_val(tail, i, "--mt")?
                            .split(',')
                            .map(|s| parse_f64_str(s.trim(), "--mt"))
                            .collect::<Result<_>>()?;
                        if v.len() != 6 {
                            return Err(EarthSnailError::InvalidParameter(
                                "--mt expects mrr,mtt,mpp,mrt,mrp,mtp".into(),
                            ));
                        }
                        fm_in = Some(FocalMechanismInput::MomentTensor(MomentTensor6::new(
                            v[0], v[1], v[2], v[3], v[4], v[5],
                        )));
                        i += 2;
                    }
                    "--quakeml" => {
                        let cat = read_quakeml_file(&req_val(tail, i, "--quakeml")?)?;
                        let fm = cat
                            .events
                            .iter()
                            .find_map(|e| e.preferred_focal_mechanism())
                            .ok_or_else(|| {
                                EarthSnailError::EmptyData(
                                    "no FocalMechanism found in QuakeML catalog".into(),
                                )
                            })?;
                        fm_in = Some(FocalMechanismInput::from(fm));
                        i += 2;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    other => {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown flag '{other}' in 'plot beachball'"
                        )));
                    }
                }
            }
            let dst = out_svg.ok_or_else(|| {
                EarthSnailError::InvalidParameter("'plot beachball' requires -o <out.svg>".into())
            })?;
            let inp = fm_in.ok_or_else(|| {
                EarthSnailError::InvalidParameter(
                    "'plot beachball' requires --sdr <s,d,r>, --mt <6>, or --quakeml <f>".into(),
                )
            })?;
            let svg = plot_beachball_svg(inp, &BeachballPlotOptions::default());
            ensure_parent_dir(&dst)?;
            fs::write(&dst, svg.as_bytes())
                .map_err(|e| EarthSnailError::Io(format!("cannot write '{dst}': {e}")))?;
            finish_plot_msg("beachball", &dst, svg.len(), json_mode)
        }
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unknown 'plot' target '{other}' (expected waveform|section|spectrogram|ppsd|taup|beachball)"
        ))),
    }
}

fn finish_plot_msg(kind: &str, path: &str, bytes_len: usize, json_mode: bool) -> Result<String> {
    if json_mode {
        let mut obj = JsonObj::new();
        obj.str("plot_type", kind)
            .str("svg", path)
            .uint("bytes", bytes_len);
        Ok(format!("{}\n", obj.finish()))
    } else {
        Ok(format!(
            "Wrote {kind} SVG ({bytes_len} bytes) -> '{path}'\n"
        ))
    }
}

// ============================================================================
// 11. `earthsnail fdsn ...` & `earthsnail seedlink ...`
// ============================================================================

fn cmd_fdsn(args: &[String]) -> Result<String> {
    if args.is_empty() || has_help_flag(args) {
        return Ok(subcommand_help("fdsn"));
    }
    let service = args[0].as_str();
    let tail = &args[1..];

    let mut provider_name = String::from("EARTHSCOPE");
    let mut net: Option<String> = None;
    let mut sta: Option<String> = None;
    let mut loc: Option<String> = None;
    let mut cha: Option<String> = None;
    let mut start_ts: Option<UtcTimestamp> = None;
    let mut end_ts: Option<UtcTimestamp> = None;
    let mut level = StationLevel::Response;
    let mut minmag: Option<f64> = None;
    let mut limit: Option<u32> = None;
    let mut url_only = false;
    let mut out_path: Option<String> = None;
    let mut json_mode = false;

    let mut i = 0;
    while i < tail.len() {
        match tail[i].as_str() {
            "--provider" | "--url" => {
                provider_name = req_val(tail, i, "--provider")?;
                i += 2;
            }
            "--net" | "--network" => {
                net = Some(req_val(tail, i, "--net")?);
                i += 2;
            }
            "--sta" | "--station" => {
                sta = Some(req_val(tail, i, "--sta")?);
                i += 2;
            }
            "--loc" | "--location" => {
                loc = Some(req_val(tail, i, "--loc")?);
                i += 2;
            }
            "--cha" | "--channel" => {
                cha = Some(req_val(tail, i, "--cha")?);
                i += 2;
            }
            "--start" | "--starttime" => {
                start_ts = Some(req_val(tail, i, "--start")?.parse()?);
                i += 2;
            }
            "--end" | "--endtime" => {
                end_ts = Some(req_val(tail, i, "--end")?.parse()?);
                i += 2;
            }
            "--level" => {
                level = match req_val(tail, i, "--level")?.to_ascii_lowercase().as_str() {
                    "network" => StationLevel::Network,
                    "station" => StationLevel::Station,
                    "channel" => StationLevel::Channel,
                    "response" => StationLevel::Response,
                    other => {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "invalid station --level '{other}'"
                        )));
                    }
                };
                i += 2;
            }
            "--minmag" => {
                minmag = Some(parse_f64_str(&req_val(tail, i, "--minmag")?, "--minmag")?);
                i += 2;
            }
            "--limit" => {
                limit = Some(parse_usize_str(&req_val(tail, i, "--limit")?, "--limit")? as u32);
                i += 2;
            }
            "--url-only" | "--dry-run" => {
                url_only = true;
                i += 1;
            }
            "-o" | "--output" => {
                out_path = Some(req_val(tail, i, "-o")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'fdsn'"
                )));
            }
        }
    }

    let client = FdsnClient::from_name(&provider_name);
    let base_url = client.provider.base_url();

    match service {
        "dataselect" | "waveform" => {
            let q = WaveformQuery {
                network: net,
                station: sta,
                location: loc,
                channel: cha,
                starttime: start_ts,
                endtime: end_ts,
                ..WaveformQuery::default()
            };
            let url = q.build_url(base_url);
            if url_only {
                return Ok(if json_mode {
                    let mut o = JsonObj::new();
                    o.str("service", "dataselect").str("url", &url);
                    format!("{}\n", o.finish())
                } else {
                    format!("{url}\n")
                });
            }
            let st = client.get_waveforms(&q)?;
            if let Some(ref dst) = out_path {
                let fmt = infer_format_from_path(dst);
                write_stream_with_format(&st, dst, &fmt, MseedEncoding::Float64)?;
            }
            if json_mode {
                let mut o = JsonObj::new();
                o.str("service", "dataselect")
                    .uint("num_traces", st.len())
                    .opt_str("output", out_path.as_deref());
                Ok(format!("{}\n", o.finish()))
            } else {
                Ok(format!(
                    "Fetched {} trace(s) via FDSNWS dataselect\n",
                    st.len()
                ))
            }
        }
        "station" => {
            let q = StationQuery {
                network: net,
                station: sta,
                location: loc,
                channel: cha,
                starttime: start_ts,
                endtime: end_ts,
                level,
                ..StationQuery::default()
            };
            let url = q.build_url(base_url);
            if url_only {
                return Ok(if json_mode {
                    let mut o = JsonObj::new();
                    o.str("service", "station").str("url", &url);
                    format!("{}\n", o.finish())
                } else {
                    format!("{url}\n")
                });
            }
            let inv = client.get_stations(&q)?;
            if let Some(ref dst) = out_path {
                ensure_parent_dir(dst)?;
                write_stationxml_file(dst, &inv)?;
            }
            let n_ch = inv.channels().count();
            if json_mode {
                let mut o = JsonObj::new();
                o.str("service", "station")
                    .uint("num_networks", inv.networks.len())
                    .uint("num_channels", n_ch);
                Ok(format!("{}\n", o.finish()))
            } else {
                Ok(format!("Fetched StationXML ({} channels)\n", n_ch))
            }
        }
        "event" => {
            let mut q = EventQuery::new();
            q.starttime = start_ts;
            q.endtime = end_ts;
            q.minmagnitude = minmag;
            q.limit = limit;
            let url = q.build_url(base_url);
            if url_only {
                return Ok(if json_mode {
                    let mut o = JsonObj::new();
                    o.str("service", "event").str("url", &url);
                    format!("{}\n", o.finish())
                } else {
                    format!("{url}\n")
                });
            }
            let cat = client.get_events(&q)?;
            if let Some(ref dst) = out_path {
                ensure_parent_dir(dst)?;
                write_quakeml_file(dst, &cat)?;
            }
            if json_mode {
                let mut o = JsonObj::new();
                o.str("service", "event").uint("num_events", cat.len());
                Ok(format!("{}\n", o.finish()))
            } else {
                Ok(format!(
                    "Fetched QuakeML catalog ({} event(s))\n",
                    cat.len()
                ))
            }
        }
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unknown 'fdsn' service '{other}' (expected dataselect|station|event)"
        ))),
    }
}

fn cmd_seedlink(args: &[String]) -> Result<String> {
    if args.is_empty() || has_help_flag(args) {
        return Ok(subcommand_help("seedlink"));
    }
    let mode = args[0].as_str();
    let tail = &args[1..];
    let mut addr: Option<String> = None;
    let mut info_level = String::from("ID");
    let mut net = String::from("IU");
    let mut sta = String::from("ANMO");
    let mut loc = String::from("00");
    let mut cha = String::from("BHZ");
    let mut max_packets = 10usize;
    let mut out_path: Option<String> = None;
    let mut json_mode = false;

    let mut i = 0;
    while i < tail.len() {
        match tail[i].as_str() {
            "--addr" | "--server" => {
                addr = Some(req_val(tail, i, "--addr")?);
                i += 2;
            }
            "--level" => {
                info_level = req_val(tail, i, "--level")?;
                i += 2;
            }
            "--net" => {
                net = req_val(tail, i, "--net")?;
                i += 2;
            }
            "--sta" => {
                sta = req_val(tail, i, "--sta")?;
                i += 2;
            }
            "--loc" => {
                loc = req_val(tail, i, "--loc")?;
                i += 2;
            }
            "--cha" => {
                cha = req_val(tail, i, "--cha")?;
                i += 2;
            }
            "--max-packets" => {
                max_packets =
                    parse_usize_str(&req_val(tail, i, "--max-packets")?, "--max-packets")?;
                i += 2;
            }
            "-o" | "--output" => {
                out_path = Some(req_val(tail, i, "-o")?);
                i += 2;
            }
            "--json" => {
                json_mode = true;
                i += 1;
            }
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'seedlink'"
                )));
            }
        }
    }

    let server_addr = addr.ok_or_else(|| {
        EarthSnailError::InvalidParameter(
            "'earthsnail seedlink' requires --addr <host:port>".into(),
        )
    })?;

    let mut sl = SeedLinkClient::connect(&server_addr, Duration::from_secs(10))?;
    let hello = sl.say_hello()?;
    match mode {
        "info" => {
            let doc = sl.request_info(&info_level)?;
            if json_mode {
                let mut o = JsonObj::new();
                o.str("software", &hello.software_line)
                    .str("organization", &hello.organization_line)
                    .bool("supports_v4", hello.supports_v4)
                    .str("info", &doc);
                Ok(format!("{}\n", o.finish()))
            } else {
                Ok(format!(
                    "Server: {} ({})\n{doc}\n",
                    hello.software_line, hello.organization_line
                ))
            }
        }
        "stream" => {
            sl.select_station(&net, &sta)?;
            sl.select_stream(&loc, &cha)?;
            sl.request_data(None)?;
            sl.end_handshake()?;
            let mut raw_records = Vec::new();
            let mut count = 0usize;
            while count < max_packets {
                let Some(pkt) = sl.read_next_packet()? else {
                    break;
                };
                match pkt {
                    SeedLinkPacket::V3Data { payload, .. } => {
                        raw_records.extend_from_slice(&payload);
                        count += 1;
                    }
                    SeedLinkPacket::V4Packet {
                        format: SeedLinkV4Format::MiniSeed2 | SeedLinkV4Format::MiniSeed3,
                        payload,
                        ..
                    } => {
                        raw_records.extend_from_slice(&payload);
                        count += 1;
                    }
                    SeedLinkPacket::End => break,
                    _ => {}
                }
            }
            if let Some(ref dst) = out_path {
                ensure_parent_dir(dst)?;
                fs::write(dst, &raw_records)
                    .map_err(|e| EarthSnailError::Io(format!("cannot write '{dst}': {e}")))?;
            }
            if json_mode {
                let mut o = JsonObj::new();
                o.uint("packets", count)
                    .uint("bytes", raw_records.len())
                    .opt_str("output", out_path.as_deref());
                Ok(format!("{}\n", o.finish()))
            } else {
                Ok(format!(
                    "Received {count} SeedLink packet(s) ({} bytes)\n",
                    raw_records.len()
                ))
            }
        }
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unknown 'seedlink' subcommand '{other}' (expected info|stream)"
        ))),
    }
}

// ============================================================================
// 12. `earthsnail verify [--json]`
// ============================================================================

fn cmd_verify(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("verify"));
    }
    let mut json_mode = false;
    for a in args {
        if a == "--json" {
            json_mode = true;
        } else {
            return Err(EarthSnailError::InvalidParameter(format!(
                "unknown flag '{a}' in 'verify'"
            )));
        }
    }
    let report = run_self_verify()?;
    let out = if json_mode {
        format!("{}\n", report.to_json())
    } else {
        report.to_pretty_table()
    };
    if !report.all_passed {
        return Err(EarthSnailError::Numerical(format!(
            "self-verification reported failure:\n{out}"
        )));
    }
    Ok(out)
}

// ============================================================================
// 13. `earthsnail sac [-m <script.m> [args...]] [-c "<cmds>"] [files...]`
// ============================================================================

fn cmd_sac(args: &[String]) -> Result<String> {
    if has_help_flag(args) {
        return Ok(subcommand_help("sac"));
    }
    let mut macro_file: Option<(String, Vec<String>)> = None;
    let mut inline_cmds: Vec<String> = Vec::new();
    let mut initial_files: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--macro" => {
                let m = req_val(args, i, "-m")?;
                let mut m_args = Vec::new();
                i += 2;
                while i < args.len() && args[i] != "-c" {
                    m_args.push(args[i].clone());
                    i += 1;
                }
                macro_file = Some((m, m_args));
            }
            "-c" | "--command" => {
                inline_cmds.push(req_val(args, i, "-c")?);
                i += 2;
            }
            other if other.starts_with('-') => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown flag '{other}' in 'sac'"
                )));
            }
            other => {
                initial_files.push(String::from(other));
                i += 1;
            }
        }
    }

    let mut session = SacSession::new();
    if !initial_files.is_empty() {
        let quoted: Vec<String> = initial_files.iter().map(|f| format!("\"{f}\"")).collect();
        session.execute_line(&format!("r {}", quoted.join(" ")))?;
    }

    if macro_file.is_none() && inline_cmds.is_empty() {
        session.run_stdio_repl()?;
        return Ok(session.take_output());
    }

    if let Some((ref mpath, ref margs)) = macro_file {
        session.execute_macro_file(mpath, margs)?;
    }
    for cmd_str in &inline_cmds {
        session.execute_script(cmd_str, &[])?;
    }
    Ok(session.take_output())
}

// ============================================================================
// Argument Parsing Helpers
// ============================================================================

fn req_val(args: &[String], idx: usize, flag: &str) -> Result<String> {
    args.get(idx + 1).cloned().ok_or_else(|| {
        EarthSnailError::InvalidParameter(format!("missing value for option '{flag}'"))
    })
}

fn parse_f64_str(s: &str, ctx: &str) -> Result<f64> {
    let v = s.parse::<f64>().map_err(|_| {
        EarthSnailError::InvalidParameter(format!("invalid floating-point number '{s}' for {ctx}"))
    })?;
    if !v.is_finite() {
        return Err(EarthSnailError::InvalidParameter(format!(
            "non-finite number '{s}' for {ctx}"
        )));
    }
    Ok(v)
}

fn parse_usize_str(s: &str, ctx: &str) -> Result<usize> {
    s.parse::<usize>().map_err(|_| {
        EarthSnailError::InvalidParameter(format!("invalid positive integer '{s}' for {ctx}"))
    })
}

fn parse_lat_lon(s: &str) -> Result<(f64, f64)> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(EarthSnailError::InvalidParameter(format!(
            "expected '<lat>,<lon>', got '{s}'"
        )));
    }
    let lat = parse_f64_str(parts[0], "latitude")?;
    let lon = parse_f64_str(parts[1], "longitude")?;
    if lat.abs() > 90.0 {
        return Err(EarthSnailError::InvalidParameter(format!(
            "latitude {lat} outside [-90, 90]"
        )));
    }
    Ok((lat, lon))
}

fn ensure_parent_dir(path_str: &str) -> Result<()> {
    if let Some(parent) = Path::new(path_str).parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| {
                EarthSnailError::Io(format!(
                    "cannot create parent directory '{}': {e}",
                    parent.display()
                ))
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_help_version_and_malformed_flags_under_miri() {
        let h = run_cli_args(&[]).unwrap();
        assert!(h.contains("EarthSnail"));
        let v = run_cli_args(&[String::from("--version")]).unwrap();
        assert!(v.contains("earthsnail"));

        for sub in [
            "inspect", "convert", "filter", "transfer", "pick", "taup", "ppsd", "rf", "fk", "plot",
            "fdsn", "seedlink", "verify", "sac",
        ] {
            let sh = run_cli_args(&[String::from("help"), String::from(sub)]).unwrap();
            assert!(sh.contains("USAGE"));
        }

        // Dry-run FDSN URL generation works purely in memory under Miri:
        let fdsn_json = run_cli_args(&[
            String::from("fdsn"),
            String::from("dataselect"),
            String::from("--provider"),
            String::from("EARTHSCOPE"),
            String::from("--net"),
            String::from("IU"),
            String::from("--sta"),
            String::from("ANMO"),
            String::from("--cha"),
            String::from("BHZ"),
            String::from("--url-only"),
            String::from("--json"),
        ])
        .unwrap();
        assert!(fdsn_json.contains("fdsnws/dataselect/1/query"));

        // Inline SAC command execution works purely in memory under Miri:
        let sac_out = run_cli_args(&[
            String::from("sac"),
            String::from("-c"),
            String::from("fg sine 2.0 npts 16; rmean; lh depmen"),
        ])
        .unwrap();
        assert!(sac_out.contains("DEPMEN"));

        // Verify subcommand (exercised under Miri in verify::tests; also run via CLI on native):
        #[cfg(not(miri))]
        {
            let ver_json = run_cli_args(&[String::from("verify"), String::from("--json")]).unwrap();
            assert!(ver_json.contains("\"passed\":true"));
        }

        // Adversarial CLI flags return Err without panicking:
        assert!(run_cli_args(&[String::from("nonexistent_cmd")]).is_err());
        assert!(run_cli_args(&[String::from("filter"), String::from("--bogus")]).is_err());
        assert!(run_cli_args(&[String::from("taup"), String::from("time")]).is_err());
    }
}
