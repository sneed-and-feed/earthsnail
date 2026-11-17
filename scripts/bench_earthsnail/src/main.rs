//! Standalone zero-external-dependency release benchmark harness for EarthSnail.
//!
//! Spawns an isolated child OS process per benchmark workload (matching the
//! exact subprocess isolation methodology of `scripts/oracle/run_obspy_baseline_bench.py`),
//! queries OS kernel `WorkingSetSize` & `PeakWorkingSetSize` (`K32GetProcessMemoryInfo`
//! on Windows, `/proc/self/status` on Linux), compares head-to-head against
//! `results/obspy_baseline_bench.json`, and writes `results/benchmark_summary.json`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use earthsnail_core::{ChannelId, KinematicUnit, Trace, TraceStats, UtcTimestamp};
use earthsnail_dsp::{
    bandpass, butter, irfft, remove_response, rfft, FilterBand, ResponseRemovalConfig,
};
use earthsnail_io::{
    read_mseed_bytes, read_stationxml_file, write_mseed_trace_bytes, MseedByteOrder, MseedEncoding,
    MseedWriteOptions,
};
use earthsnail_taup::{TauEngine, TauModel};

// ============================================================================
// OS Kernel RSS & Peak Working Set Measurement
// ============================================================================

#[cfg(target_os = "windows")]
mod os_mem {
    use core::ffi::c_void;
    use core::mem::{size_of, zeroed};

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            ppsmem_counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    pub fn current_and_peak_rss_bytes() -> (u64, u64) {
        // SAFETY: `ProcessMemoryCounters` is POD `#[repr(C)]` matching Win32 `PROCESS_MEMORY_COUNTERS`,
        // `GetCurrentProcess()` returns the pseudo-handle `-1`, and `cb` equals the exact struct size.
        unsafe {
            let mut pmc: ProcessMemoryCounters = zeroed();
            pmc.cb = size_of::<ProcessMemoryCounters>() as u32;
            let ok = K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb);
            if ok != 0 {
                (
                    pmc.working_set_size as u64,
                    pmc.peak_working_set_size as u64,
                )
            } else {
                (0, 0)
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod os_mem {
    use std::fs;

    pub fn current_and_peak_rss_bytes() -> (u64, u64) {
        let Ok(txt) = fs::read_to_string("/proc/self/status") else {
            return (0, 0);
        };
        let mut rss_kb = 0_u64;
        let mut hwm_kb = 0_u64;
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                hwm_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }
        (rss_kb * 1024, hwm_kb.max(rss_kb) * 1024)
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
mod os_mem {
    pub fn current_and_peak_rss_bytes() -> (u64, u64) {
        (0, 0)
    }
}

// ============================================================================
// Deterministic Synthetic Generators (Matching `run_obspy_baseline_bench.py`)
// ============================================================================

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

    fn next_f64_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn next_i64_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1) as u64;
        lo + ((self.next_u64() % span) as i64)
    }
}

fn make_wave_f64(n: usize) -> Vec<f64> {
    let mut rng = SplitMix64::new(42);
    let two_pi = core::f64::consts::TAU;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64) * 0.01;
        let noise = (rng.next_f64_unit() - 0.5) * 0.2;
        let v = (two_pi * 1.5 * t).sin() + 0.5 * (two_pi * 8.0 * t).cos() + noise;
        out.push(v);
    }
    out
}

fn make_steim2_bytes(n: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(42);
    let mut acc = 0_i64;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let d = if i % 500 == 0 {
            rng.next_i64_inclusive(-25_000, 25_000)
        } else {
            rng.next_i64_inclusive(-120, 120)
        };
        acc += d;
        data.push((acc as i32) as f64);
    }
    let t0: UtcTimestamp = "2020-05-15T12:34:56.789000Z".parse().unwrap();
    let stats = TraceStats::new(ChannelId::new("IU", "ANMO", "10", "BHZ"), t0, 40.0);
    let tr = Trace::new(stats, data);
    write_mseed_trace_bytes(
        &tr,
        MseedWriteOptions {
            record_length: 512,
            encoding: MseedEncoding::Steim2,
            byte_order: MseedByteOrder::Big,
            quality_indicator: 'D',
        },
    )
    .expect("Steim-2 benchmark encode failed")
}

// ============================================================================
// Worker Benchmark Loop & Metrics Struct
// ============================================================================

#[derive(Debug, Clone)]
struct BenchMetrics {
    workload: String,
    obspy_workload: String,
    iterations: usize,
    warmup_iterations: usize,
    units_per_iter: usize,
    unit_name: String,
    median_ms: f64,
    min_ms: f64,
    mean_ms: f64,
    p95_ms: f64,
    throughput_per_sec: f64,
    rss_before_bytes: u64,
    rss_after_bytes: u64,
    peak_rss_bytes: u64,
    rss_delta_bytes: u64,
}

impl BenchMetrics {
    fn to_single_line_kv(&self) -> String {
        format!(
            "BENCH_OK\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.6}\t{}\t{}\t{}\t{}",
            self.workload,
            self.obspy_workload,
            self.iterations,
            self.warmup_iterations,
            self.units_per_iter,
            self.unit_name,
            self.median_ms,
            self.min_ms,
            self.mean_ms,
            self.p95_ms,
            self.throughput_per_sec,
            self.rss_before_bytes,
            self.rss_after_bytes,
            self.peak_rss_bytes,
            self.rss_delta_bytes,
        )
    }

    fn from_single_line_kv(line: &str) -> Result<Self, String> {
        let parts: Vec<&str> = line.trim().split('\t').collect();
        if parts.len() != 16 || parts[0] != "BENCH_OK" {
            return Err(format!("unexpected worker line: {line}"));
        }
        Ok(Self {
            workload: parts[1].to_string(),
            obspy_workload: parts[2].to_string(),
            iterations: parts[3].parse().map_err(|e| format!("{e}"))?,
            warmup_iterations: parts[4].parse().map_err(|e| format!("{e}"))?,
            units_per_iter: parts[5].parse().map_err(|e| format!("{e}"))?,
            unit_name: parts[6].to_string(),
            median_ms: parts[7].parse().map_err(|e| format!("{e}"))?,
            min_ms: parts[8].parse().map_err(|e| format!("{e}"))?,
            mean_ms: parts[9].parse().map_err(|e| format!("{e}"))?,
            p95_ms: parts[10].parse().map_err(|e| format!("{e}"))?,
            throughput_per_sec: parts[11].parse().map_err(|e| format!("{e}"))?,
            rss_before_bytes: parts[12].parse().map_err(|e| format!("{e}"))?,
            rss_after_bytes: parts[13].parse().map_err(|e| format!("{e}"))?,
            peak_rss_bytes: parts[14].parse().map_err(|e| format!("{e}"))?,
            rss_delta_bytes: parts[15].parse().map_err(|e| format!("{e}"))?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_loop<C, F>(
    workload: &str,
    obspy_workload: &str,
    setup: impl FnOnce() -> C,
    mut run: F,
    warmup: usize,
    iters: usize,
    units_per_iter: usize,
    unit_name: &str,
) -> BenchMetrics
where
    F: FnMut(&mut C),
{
    let mut ctx = setup();
    for _ in 0..warmup {
        run(&mut ctx);
    }
    let (rss_before, _) = os_mem::current_and_peak_rss_bytes();
    let mut peak_seen = rss_before;
    let mut times_ms = Vec::with_capacity(iters);

    for _ in 0..iters {
        let t0 = Instant::now();
        run(&mut ctx);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(elapsed_ms);
        let (cur_rss, hw_rss) = os_mem::current_and_peak_rss_bytes();
        peak_seen = peak_seen.max(cur_rss).max(hw_rss);
    }
    let (rss_after, hw_end) = os_mem::current_and_peak_rss_bytes();
    peak_seen = peak_seen.max(hw_end);

    times_ms.sort_by(|a, b| a.total_cmp(b));
    let median_ms = if iters % 2 == 1 {
        times_ms[iters / 2]
    } else {
        0.5 * (times_ms[iters / 2 - 1] + times_ms[iters / 2])
    };
    let min_ms = times_ms[0];
    let mean_ms = times_ms.iter().sum::<f64>() / (iters as f64);
    let p95_idx = (((iters.saturating_sub(1)) as f64) * 0.95).round() as usize;
    let p95_ms = times_ms[p95_idx.min(iters - 1)];
    let throughput_per_sec = if median_ms > 0.0 && units_per_iter > 0 {
        (units_per_iter as f64) / (median_ms / 1000.0)
    } else {
        0.0
    };

    BenchMetrics {
        workload: workload.to_string(),
        obspy_workload: obspy_workload.to_string(),
        iterations: iters,
        warmup_iterations: warmup,
        units_per_iter,
        unit_name: unit_name.to_string(),
        median_ms,
        min_ms,
        mean_ms,
        p95_ms,
        throughput_per_sec,
        rss_before_bytes: rss_before,
        rss_after_bytes: rss_after,
        peak_rss_bytes: peak_seen,
        rss_delta_bytes: peak_seen.saturating_sub(rss_before),
    }
}

fn run_worker(workload: &str, xml_path: &Path) {
    let metrics = match workload {
        "cold_startup" => {
            let (rss0, _) = os_mem::current_and_peak_rss_bytes();
            let t0 = Instant::now();
            let engine = TauEngine::by_name("iasp91").expect("TauEngine init");
            let sos = butter(4, FilterBand::Bandpass(0.5, 5.0), 100.0).expect("butter init");
            let inv = read_stationxml_file(xml_path).expect("stationxml init");
            black_box((&engine, &sos, &inv));
            let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let (rss1, peak_rss) = os_mem::current_and_peak_rss_bytes();
            BenchMetrics {
                workload: "cold_startup".into(),
                obspy_workload: "cold_import".into(),
                iterations: 1,
                warmup_iterations: 0,
                units_per_iter: 1,
                unit_name: "init".into(),
                median_ms: dt_ms,
                min_ms: dt_ms,
                mean_ms: dt_ms,
                p95_ms: dt_ms,
                throughput_per_sec: if dt_ms > 0.0 { 1000.0 / dt_ms } else { 0.0 },
                rss_before_bytes: rss0,
                rss_after_bytes: rss1,
                peak_rss_bytes: peak_rss.max(rss1),
                rss_delta_bytes: peak_rss.max(rss1).saturating_sub(rss0),
            }
        }
        "filter_bandpass_causal_100k" => measure_loop(
            workload,
            workload,
            || make_wave_f64(100_000),
            |x| {
                let y = bandpass(black_box(x), 0.5, 5.0, 100.0, 4, false).unwrap();
                black_box(y);
            },
            5,
            50,
            100_000,
            "samples",
        ),
        "filter_bandpass_zerophase_100k" => measure_loop(
            workload,
            workload,
            || make_wave_f64(100_000),
            |x| {
                let y = bandpass(black_box(x), 0.5, 5.0, 100.0, 4, true).unwrap();
                black_box(y);
            },
            5,
            50,
            100_000,
            "samples",
        ),
        "filter_bandpass_causal_1m" => measure_loop(
            workload,
            workload,
            || make_wave_f64(1_000_000),
            |x| {
                let y = bandpass(black_box(x), 0.5, 5.0, 100.0, 4, false).unwrap();
                black_box(y);
            },
            3,
            25,
            1_000_000,
            "samples",
        ),
        "filter_bandpass_zerophase_1m" => measure_loop(
            workload,
            workload,
            || make_wave_f64(1_000_000),
            |x| {
                let y = bandpass(black_box(x), 0.5, 5.0, 100.0, 4, true).unwrap();
                black_box(y);
            },
            3,
            25,
            1_000_000,
            "samples",
        ),
        "steim2_decode_100k" => measure_loop(
            workload,
            workload,
            || make_steim2_bytes(100_000),
            |raw| {
                let st = read_mseed_bytes(black_box(raw)).unwrap();
                black_box(st);
            },
            5,
            50,
            100_000,
            "samples",
        ),
        "steim2_decode_1m" => measure_loop(
            workload,
            workload,
            || make_steim2_bytes(1_000_000),
            |raw| {
                let st = read_mseed_bytes(black_box(raw)).unwrap();
                black_box(st);
            },
            3,
            25,
            1_000_000,
            "samples",
        ),
        "fft_rfft_irfft_100k" => measure_loop(
            workload,
            workload,
            || make_wave_f64(100_000),
            |x| {
                let spec = rfft(black_box(x), Some(x.len())).unwrap();
                let y = irfft(black_box(&spec), x.len()).unwrap();
                black_box(y);
            },
            3,
            25,
            100_000,
            "samples",
        ),
        "fft_rfft_irfft_1m" => measure_loop(
            workload,
            workload,
            || make_wave_f64(1_000_000),
            |x| {
                let spec = rfft(black_box(x), Some(x.len())).unwrap();
                let y = irfft(black_box(&spec), x.len()).unwrap();
                black_box(y);
            },
            2,
            10,
            1_000_000,
            "samples",
        ),
        "fft_rfft_irfft_pow2_131k" => measure_loop(
            workload,
            "fft_rfft_irfft_100k",
            || make_wave_f64(131_072),
            |x| {
                let spec = rfft(black_box(x), Some(x.len())).unwrap();
                let y = irfft(black_box(&spec), x.len()).unwrap();
                black_box(y);
            },
            3,
            25,
            131_072,
            "samples",
        ),
        "fft_rfft_irfft_pow2_1m" => measure_loop(
            workload,
            "fft_rfft_irfft_1m",
            || make_wave_f64(1_048_576),
            |x| {
                let spec = rfft(black_box(x), Some(x.len())).unwrap();
                let y = irfft(black_box(&spec), x.len()).unwrap();
                black_box(y);
            },
            2,
            15,
            1_048_576,
            "samples",
        ),
        "remove_response_100k" => measure_loop(
            workload,
            workload,
            || {
                let inv = read_stationxml_file(xml_path).unwrap();
                let ch = inv.channels().next().unwrap().clone();
                let sr = ch.sample_rate_hz.unwrap_or(40.0);
                let resp = ch.response.unwrap();
                let mut x = make_wave_f64(100_000);
                for v in &mut x {
                    *v *= 1e4;
                }
                let cfg = ResponseRemovalConfig {
                    output: KinematicUnit::Velocity,
                    pre_filt_hz: Some([0.005, 0.01, 15.0, 20.0]),
                    water_level_db: Some(60.0),
                    zero_mean: true,
                    taper: true,
                    taper_fraction: 0.05,
                    nfft_pow2: false,
                    rescale_to_overall_sensitivity: false,
                };
                (x, sr, resp, cfg)
            },
            |(x, sr, resp, cfg)| {
                let y = remove_response(black_box(x), *sr, resp, cfg).unwrap();
                black_box(y);
            },
            2,
            15,
            100_000,
            "samples",
        ),
        "remove_response_1m" => measure_loop(
            workload,
            workload,
            || {
                let inv = read_stationxml_file(xml_path).unwrap();
                let ch = inv.channels().next().unwrap().clone();
                let sr = ch.sample_rate_hz.unwrap_or(40.0);
                let resp = ch.response.unwrap();
                let mut x = make_wave_f64(1_000_000);
                for v in &mut x {
                    *v *= 1e4;
                }
                let cfg = ResponseRemovalConfig {
                    output: KinematicUnit::Velocity,
                    pre_filt_hz: Some([0.005, 0.01, 15.0, 20.0]),
                    water_level_db: Some(60.0),
                    zero_mean: true,
                    taper: true,
                    taper_fraction: 0.05,
                    nfft_pow2: false,
                    rescale_to_overall_sensitivity: false,
                };
                (x, sr, resp, cfg)
            },
            |(x, sr, resp, cfg)| {
                let y = remove_response(black_box(x), *sr, resp, cfg).unwrap();
                black_box(y);
            },
            1,
            5,
            1_000_000,
            "samples",
        ),
        "remove_response_pow2_100k" => measure_loop(
            workload,
            "remove_response_100k",
            || {
                let inv = read_stationxml_file(xml_path).unwrap();
                let ch = inv.channels().next().unwrap().clone();
                let sr = ch.sample_rate_hz.unwrap_or(40.0);
                let resp = ch.response.unwrap();
                let mut x = make_wave_f64(100_000);
                for v in &mut x {
                    *v *= 1e4;
                }
                let cfg = ResponseRemovalConfig {
                    output: KinematicUnit::Velocity,
                    pre_filt_hz: Some([0.005, 0.01, 15.0, 20.0]),
                    water_level_db: Some(60.0),
                    zero_mean: true,
                    taper: true,
                    taper_fraction: 0.05,
                    nfft_pow2: true,
                    rescale_to_overall_sensitivity: false,
                };
                (x, sr, resp, cfg)
            },
            |(x, sr, resp, cfg)| {
                let y = remove_response(black_box(x), *sr, resp, cfg).unwrap();
                black_box(y);
            },
            2,
            15,
            100_000,
            "samples",
        ),
        "remove_response_pow2_1m" => measure_loop(
            workload,
            "remove_response_1m",
            || {
                let inv = read_stationxml_file(xml_path).unwrap();
                let ch = inv.channels().next().unwrap().clone();
                let sr = ch.sample_rate_hz.unwrap_or(40.0);
                let resp = ch.response.unwrap();
                let mut x = make_wave_f64(1_000_000);
                for v in &mut x {
                    *v *= 1e4;
                }
                let cfg = ResponseRemovalConfig {
                    output: KinematicUnit::Velocity,
                    pre_filt_hz: Some([0.005, 0.01, 15.0, 20.0]),
                    water_level_db: Some(60.0),
                    zero_mean: true,
                    taper: true,
                    taper_fraction: 0.05,
                    nfft_pow2: true,
                    rescale_to_overall_sensitivity: false,
                };
                (x, sr, resp, cfg)
            },
            |(x, sr, resp, cfg)| {
                let y = remove_response(black_box(x), *sr, resp, cfg).unwrap();
                black_box(y);
            },
            1,
            5,
            1_000_000,
            "samples",
        ),
        "taup_1000_queries_iasp91" => measure_loop(
            workload,
            workload,
            || {
                let engine = TauEngine::by_name("iasp91").unwrap();
                let depths = [0.0_f64, 15.0, 35.0, 120.0, 450.0];
                let dists: Vec<f64> = (0..200)
                    .map(|i| 5.0 + (170.0 * (i as f64)) / 199.0)
                    .collect();
                let queries: Vec<(f64, f64)> = (0..1000)
                    .map(|i| (depths[i % depths.len()], dists[i % dists.len()]))
                    .collect();
                let phases = ["P", "S", "pP", "PP", "PcP", "PKP", "PKIKP", "SKS"];
                (engine, queries, phases)
            },
            |(engine, queries, phases)| {
                let mut total = 0_usize;
                for &(d, deg) in queries.iter() {
                    let arrs = engine
                        .get_travel_times(black_box(d), black_box(deg), phases)
                        .unwrap();
                    total += arrs.len();
                }
                black_box(total);
            },
            2,
            15,
            1000,
            "queries",
        ),
        "taup_1000_queries_uncached_iasp91" => measure_loop(
            workload,
            "taup_1000_queries_iasp91",
            || {
                let model = TauModel::by_name("iasp91").unwrap();
                let depths = [0.0_f64, 15.0, 35.0, 120.0, 450.0];
                let dists: Vec<f64> = (0..200)
                    .map(|i| 5.0 + (170.0 * (i as f64)) / 199.0)
                    .collect();
                let queries: Vec<(f64, f64)> = (0..1000)
                    .map(|i| (depths[i % depths.len()], dists[i % dists.len()]))
                    .collect();
                let phases = ["P", "S", "pP", "PP", "PcP", "PKP", "PKIKP", "SKS"];
                (model, queries, phases)
            },
            |(model, queries, phases)| {
                let mut total = 0_usize;
                for &(d, deg) in queries.iter() {
                    let arrs = model
                        .get_travel_times(black_box(d), black_box(deg), phases)
                        .unwrap();
                    total += arrs.len();
                }
                black_box(total);
            },
            1,
            5,
            1000,
            "queries",
        ),
        other => panic!("unknown workload `{other}`"),
    };

    println!("{}", metrics.to_single_line_kv());
}

// ============================================================================
// Minimal Zero-Dependency JSON Parser for `obspy_baseline_bench.json`
// ============================================================================

#[allow(dead_code)]
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
    fn get(&self, key: &str) -> Option<&JsonVal> {
        match self {
            JsonVal::Obj(m) => m.get(key),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            JsonVal::Num(v) => Some(*v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            JsonVal::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn parse_val(&mut self) -> JsonVal {
        self.skip_ws();
        match self.b[self.i] {
            b'"' => JsonVal::Str(self.parse_str()),
            b'{' => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.skip_ws();
                if self.b[self.i] == b'}' {
                    self.i += 1;
                    return JsonVal::Obj(m);
                }
                loop {
                    self.skip_ws();
                    let k = self.parse_str();
                    self.skip_ws();
                    assert_eq!(self.b[self.i], b':');
                    self.i += 1;
                    let v = self.parse_val();
                    m.insert(k, v);
                    self.skip_ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    } else {
                        assert_eq!(self.b[self.i], b'}');
                        self.i += 1;
                        break;
                    }
                }
                JsonVal::Obj(m)
            }
            b'[' => {
                self.i += 1;
                let mut arr = Vec::new();
                self.skip_ws();
                if self.b[self.i] == b']' {
                    self.i += 1;
                    return JsonVal::Arr(arr);
                }
                loop {
                    arr.push(self.parse_val());
                    self.skip_ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    } else {
                        assert_eq!(self.b[self.i], b']');
                        self.i += 1;
                        break;
                    }
                }
                JsonVal::Arr(arr)
            }
            b't' => {
                self.i += 4;
                JsonVal::Bool(true)
            }
            b'f' => {
                self.i += 5;
                JsonVal::Bool(false)
            }
            b'n' => {
                self.i += 4;
                JsonVal::Null
            }
            _ => {
                let start = self.i;
                while self.i < self.b.len()
                    && matches!(
                        self.b[self.i],
                        b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
                    )
                {
                    self.i += 1;
                }
                let s = core::str::from_utf8(&self.b[start..self.i]).unwrap();
                JsonVal::Num(s.parse::<f64>().unwrap())
            }
        }
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
                    _ => out.push(esc as char),
                }
            } else {
                out.push(c as char);
            }
        }
        panic!("unterminated JSON string");
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 4 && args[1] == "--worker" {
        run_worker(&args[2], Path::new(&args[3]));
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let xml_path = repo_root
        .join("tests")
        .join("fixtures")
        .join("io")
        .join("iu_anmo_bhz.xml");
    let obspy_json_path = repo_root.join("results").join("obspy_baseline_bench.json");
    let out_json_path = repo_root.join("results").join("benchmark_summary.json");

    let obspy_raw = fs::read_to_string(&obspy_json_path)
        .unwrap_or_else(|e| panic!("failed reading {}: {e}", obspy_json_path.display()));
    let obspy_doc = JsonParser {
        b: obspy_raw.as_bytes(),
        i: 0,
    }
    .parse_val();
    let obspy_benches = obspy_doc.get("benchmarks").expect("benchmarks object");
    let obspy_env = obspy_doc.get("environment").expect("environment object");

    let workloads = [
        "cold_startup",
        "filter_bandpass_causal_100k",
        "filter_bandpass_zerophase_100k",
        "filter_bandpass_causal_1m",
        "filter_bandpass_zerophase_1m",
        "steim2_decode_100k",
        "steim2_decode_1m",
        "fft_rfft_irfft_100k",
        "fft_rfft_irfft_1m",
        "fft_rfft_irfft_pow2_131k",
        "fft_rfft_irfft_pow2_1m",
        "remove_response_100k",
        "remove_response_1m",
        "remove_response_pow2_100k",
        "remove_response_pow2_1m",
        "taup_1000_queries_iasp91",
        "taup_1000_queries_uncached_iasp91",
    ];

    let exe = env::current_exe().expect("current_exe");
    let mut rust_results = Vec::with_capacity(workloads.len());

    println!("==========================================================================================");
    println!(
        "EarthSnail (100% Safe Rust, 0 C/Fortran deps) vs ObsPy 1.5.1 / SciPy 1.18.1 Head-to-Head"
    );
    println!("==========================================================================================");
    println!(
        "{:<36} {:>11} {:>11} {:>10} {:>11} {:>11} {:>9}",
        "Workload", "EarthSnail", "ObsPy", "Speedup", "ES RSS", "ObsPy RSS", "RSS Red."
    );
    println!("------------------------------------------------------------------------------------------");

    for wl in workloads {
        let out = Command::new(&exe)
            .arg("--worker")
            .arg(wl)
            .arg(&xml_path)
            .output()
            .unwrap_or_else(|e| panic!("failed spawning worker for {wl}: {e}"));
        if !out.status.success() {
            panic!(
                "worker {wl} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let bench_line = stdout
            .lines()
            .rev()
            .find(|l| l.starts_with("BENCH_OK\t"))
            .unwrap_or_else(|| panic!("missing BENCH_OK line for {wl}: {stdout}"));
        let m = BenchMetrics::from_single_line_kv(bench_line).unwrap();

        let ob = obspy_benches
            .get(&m.obspy_workload)
            .unwrap_or_else(|| panic!("missing ObsPy baseline `{}`", m.obspy_workload));
        let ob_med_ms = ob.get("median_ms").and_then(JsonVal::as_f64).unwrap_or(0.0);
        let ob_peak_rss = ob
            .get("peak_rss_bytes")
            .and_then(JsonVal::as_f64)
            .unwrap_or(0.0) as u64;

        let speedup = if m.median_ms > 0.0 {
            ob_med_ms / m.median_ms
        } else {
            0.0
        };
        let rss_red = if m.peak_rss_bytes > 0 {
            (ob_peak_rss as f64) / (m.peak_rss_bytes as f64)
        } else {
            0.0
        };

        println!(
            "{:<36} {:>8.3} ms {:>8.2} ms {:>9.2}x {:>7.2} MiB {:>7.1} MiB {:>8.2}x",
            m.workload,
            m.median_ms,
            ob_med_ms,
            speedup,
            (m.peak_rss_bytes as f64) / (1024.0 * 1024.0),
            (ob_peak_rss as f64) / (1024.0 * 1024.0),
            rss_red,
        );

        rust_results.push((m, ob_med_ms, ob_peak_rss, speedup, rss_red));
    }
    println!("==========================================================================================");

    // Emit structured `results/benchmark_summary.json`
    let epoch_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let os_str = obspy_env
        .get("os")
        .and_then(JsonVal::as_str)
        .unwrap_or(env::consts::OS);
    let cpu_str = obspy_env
        .get("cpu")
        .and_then(JsonVal::as_str)
        .unwrap_or(env::consts::ARCH);
    let obspy_ver = obspy_env
        .get("obspy_version")
        .and_then(JsonVal::as_str)
        .unwrap_or("1.5.1");
    let scipy_ver = obspy_env
        .get("scipy_version")
        .and_then(JsonVal::as_str)
        .unwrap_or("1.18.1");
    let numpy_ver = obspy_env
        .get("numpy_version")
        .and_then(JsonVal::as_str)
        .unwrap_or("2.5.5");
    let py_ver = obspy_env
        .get("python_version")
        .and_then(JsonVal::as_str)
        .unwrap_or("3.12.13");

    let mut json = String::with_capacity(16_384);
    json.push_str("{\n");
    json.push_str("  \"schema_version\": \"1.0\",\n");
    json.push_str(&format!("  \"generated_epoch_s\": {epoch_s},\n"));
    json.push_str("  \"methodology\": \"Isolated per-workload OS child process (--release, opt-level=3, thin-LTO); Win32 K32GetProcessMemoryInfo PeakWorkingSetSize RSS vs ObsPy/SciPy/TauPy isolated subprocess baseline.\",\n");
    json.push_str("  \"environment\": {\n");
    json.push_str(&format!("    \"os\": \"{}\",\n", json_escape(os_str)));
    json.push_str(&format!("    \"cpu\": \"{}\",\n", json_escape(cpu_str)));
    json.push_str(&format!(
        "    \"target_arch\": \"{}\",\n",
        env::consts::ARCH
    ));
    json.push_str(&format!(
        "    \"python_version\": \"{}\",\n",
        json_escape(py_ver)
    ));
    json.push_str(&format!(
        "    \"obspy_version\": \"{}\",\n",
        json_escape(obspy_ver)
    ));
    json.push_str(&format!(
        "    \"scipy_version\": \"{}\",\n",
        json_escape(scipy_ver)
    ));
    json.push_str(&format!(
        "    \"numpy_version\": \"{}\"\n",
        json_escape(numpy_ver)
    ));
    json.push_str("  },\n");
    json.push_str("  \"comparisons\": [\n");

    for (idx, (m, ob_med_ms, ob_peak_rss, speedup, rss_red)) in rust_results.iter().enumerate() {
        let comma = if idx + 1 < rust_results.len() {
            ","
        } else {
            ""
        };
        let ob_obj = obspy_benches.get(&m.obspy_workload).unwrap();
        let ob_min_ms = ob_obj
            .get("min_ms")
            .and_then(JsonVal::as_f64)
            .unwrap_or(*ob_med_ms);
        let ob_p95_ms = ob_obj
            .get("p95_ms")
            .and_then(JsonVal::as_f64)
            .unwrap_or(*ob_med_ms);
        let ob_tput = ob_obj
            .get("throughput_per_sec")
            .and_then(JsonVal::as_f64)
            .unwrap_or(0.0);
        let ob_tm_peak = ob_obj
            .get("tracemalloc_peak_bytes")
            .and_then(JsonVal::as_f64)
            .unwrap_or(0.0) as u64;

        json.push_str("    {\n");
        json.push_str(&format!("      \"workload\": \"{}\",\n", m.workload));
        json.push_str(&format!(
            "      \"obspy_baseline_workload\": \"{}\",\n",
            m.obspy_workload
        ));
        json.push_str(&format!(
            "      \"units_per_iter\": {},\n",
            m.units_per_iter
        ));
        json.push_str(&format!("      \"unit_name\": \"{}\",\n", m.unit_name));
        json.push_str("      \"earthsnail_rust\": {\n");
        json.push_str(&format!("        \"iterations\": {},\n", m.iterations));
        json.push_str(&format!(
            "        \"warmup_iterations\": {},\n",
            m.warmup_iterations
        ));
        json.push_str(&format!("        \"median_ms\": {:.6},\n", m.median_ms));
        json.push_str(&format!("        \"min_ms\": {:.6},\n", m.min_ms));
        json.push_str(&format!("        \"mean_ms\": {:.6},\n", m.mean_ms));
        json.push_str(&format!("        \"p95_ms\": {:.6},\n", m.p95_ms));
        json.push_str(&format!(
            "        \"throughput_per_sec\": {:.3},\n",
            m.throughput_per_sec
        ));
        json.push_str(&format!(
            "        \"rss_before_bytes\": {},\n",
            m.rss_before_bytes
        ));
        json.push_str(&format!(
            "        \"rss_after_bytes\": {},\n",
            m.rss_after_bytes
        ));
        json.push_str(&format!(
            "        \"peak_rss_bytes\": {},\n",
            m.peak_rss_bytes
        ));
        json.push_str(&format!(
            "        \"rss_delta_bytes\": {}\n",
            m.rss_delta_bytes
        ));
        json.push_str("      },\n");
        json.push_str("      \"obspy_baseline\": {\n");
        json.push_str(&format!("        \"median_ms\": {:.6},\n", ob_med_ms));
        json.push_str(&format!("        \"min_ms\": {:.6},\n", ob_min_ms));
        json.push_str(&format!("        \"p95_ms\": {:.6},\n", ob_p95_ms));
        json.push_str(&format!(
            "        \"throughput_per_sec\": {:.3},\n",
            ob_tput
        ));
        json.push_str(&format!("        \"peak_rss_bytes\": {},\n", ob_peak_rss));
        json.push_str(&format!(
            "        \"tracemalloc_peak_bytes\": {}\n",
            ob_tm_peak
        ));
        json.push_str("      },\n");
        json.push_str(&format!("      \"speedup_vs_obspy\": {:.4},\n", speedup));
        json.push_str(&format!(
            "      \"peak_rss_reduction_factor\": {:.4}\n",
            rss_red
        ));
        json.push_str(&format!("    }}{comma}\n"));
    }

    json.push_str("  ]\n");
    json.push_str("}\n");

    if let Some(parent) = out_json_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&out_json_path, json).expect("failed writing benchmark_summary.json");
    println!(
        "Wrote head-to-head benchmark summary to: {}",
        out_json_path.display()
    );
}
