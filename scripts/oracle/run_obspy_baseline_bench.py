#!/usr/bin/env python3
"""
Isolated Subprocess Baseline Benchmark Suite for ObsPy + SciPy + TauPy.

Measures execution timings (median, min, mean, p95), throughput, Python
`tracemalloc` peak heap bytes, and OS kernel peak RSS (`peak_wset` / `rss`) in
dedicated per-workload child processes so heap/working-set high-water marks
never bleed across benchmarks.

Outputs: `obspy_baseline_bench.json`
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
OUT_PATH = Path(
    os.environ.get("EARTHSNAIL_BENCH_OUT", str(REPO_ROOT / "results" / "obspy_baseline_bench.json"))
)
XML_FIXTURE = Path(
    os.environ.get(
        "EARTHSNAIL_XML_FIXTURE",
        str(REPO_ROOT / "tests" / "fixtures" / "io" / "iu_anmo_bhz.xml"),
    )
)


WORKER_CODE = r"""
import gc
import io
import json
import statistics
import sys
import time
import tracemalloc
import psutil

proc = psutil.Process()

def get_rss():
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", mi.rss)
    return int(mi.rss), int(peak)

workload = sys.argv[1]

if workload == "cold_import":
    rss0, _ = get_rss()
    tracemalloc.start()
    t0 = time.perf_counter()
    import numpy as np
    import scipy.signal
    import obspy
    from obspy.taup import TauPyModel
    dt_ms = (time.perf_counter() - t0) * 1000.0
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1, peak_rss = get_rss()
    print(json.dumps({
        "workload": workload,
        "iterations": 1,
        "median_ms": dt_ms,
        "min_ms": dt_ms,
        "mean_ms": dt_ms,
        "p95_ms": dt_ms,
        "rss_before_bytes": rss0,
        "rss_after_bytes": rss1,
        "peak_rss_bytes": peak_rss,
        "rss_delta_bytes": max(0, rss1 - rss0),
        "tracemalloc_peak_bytes": int(tm_peak),
    }))
    sys.exit(0)

import numpy as np
import scipy.fft as spfft
import obspy
from obspy import Trace, UTCDateTime, read, read_inventory
from obspy.signal.filter import bandpass
from obspy.taup import TauPyModel

xml_path = sys.argv[2]

def bench_loop(setup_fn, run_fn, warmup, iters, units_count, unit_name):
    ctx = setup_fn()
    for _ in range(warmup):
        run_fn(ctx)
    gc.collect()
    rss_before, _ = get_rss()
    tracemalloc.start()
    times_ms = []
    peak_rss_seen = rss_before
    for _ in range(iters):
        t0 = time.perf_counter()
        run_fn(ctx)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        cur_rss, hw_rss = get_rss()
        if cur_rss > peak_rss_seen:
            peak_rss_seen = cur_rss
        if hw_rss > peak_rss_seen:
            peak_rss_seen = hw_rss
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after, hw_end = get_rss()
    peak_rss_seen = max(peak_rss_seen, hw_end)
    times_sorted = sorted(times_ms)
    med_ms = statistics.median(times_sorted)
    p95_idx = min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))
    throughput = (units_count / (med_ms / 1000.0)) if med_ms > 0 else 0.0
    return {
        "workload": workload,
        "iterations": iters,
        "warmup_iterations": warmup,
        "units_per_iter": units_count,
        "unit_name": unit_name,
        "median_ms": med_ms,
        "min_ms": times_sorted[0],
        "mean_ms": statistics.mean(times_sorted),
        "p95_ms": times_sorted[p95_idx],
        "throughput_per_sec": throughput,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "peak_rss_bytes": peak_rss_seen,
        "rss_delta_bytes": max(0, peak_rss_seen - rss_before),
        "tracemalloc_peak_bytes": int(tm_peak),
    }

def make_wave_f64(n):
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=np.float64) * 0.01
    return np.sin(2.0 * np.pi * 1.5 * t) + 0.5 * np.cos(2.0 * np.pi * 8.0 * t) + 0.1 * rng.standard_normal(n)

def make_steim2_bytes(n):
    rng = np.random.default_rng(42)
    diffs = rng.integers(-120, 121, size=n, dtype=np.int64)
    diffs[::500] = rng.integers(-25000, 25001, size=len(diffs[::500]))
    arr = np.cumsum(diffs).astype(np.int32)
    tr = Trace(data=arr, header={"network": "IU", "station": "ANMO", "location": "10", "channel": "BHZ", "sampling_rate": 40.0})
    buf = io.BytesIO()
    tr.write(buf, format="MSEED", encoding="STEIM2", reclen=512)
    return buf.getvalue()

if workload == "filter_bandpass_causal_100k":
    res = bench_loop(
        lambda: make_wave_f64(100_000),
        lambda x: bandpass(x, 0.5, 5.0, df=100.0, corners=4, zerophase=False),
        warmup=3, iters=25, units_count=100_000, unit_name="samples"
    )
elif workload == "filter_bandpass_zerophase_100k":
    res = bench_loop(
        lambda: make_wave_f64(100_000),
        lambda x: bandpass(x, 0.5, 5.0, df=100.0, corners=4, zerophase=True),
        warmup=3, iters=25, units_count=100_000, unit_name="samples"
    )
elif workload == "filter_bandpass_causal_1m":
    res = bench_loop(
        lambda: make_wave_f64(1_000_000),
        lambda x: bandpass(x, 0.5, 5.0, df=100.0, corners=4, zerophase=False),
        warmup=2, iters=15, units_count=1_000_000, unit_name="samples"
    )
elif workload == "filter_bandpass_zerophase_1m":
    res = bench_loop(
        lambda: make_wave_f64(1_000_000),
        lambda x: bandpass(x, 0.5, 5.0, df=100.0, corners=4, zerophase=True),
        warmup=2, iters=15, units_count=1_000_000, unit_name="samples"
    )
elif workload == "steim2_decode_100k":
    res = bench_loop(
        lambda: make_steim2_bytes(100_000),
        lambda raw: read(io.BytesIO(raw), format="MSEED"),
        warmup=3, iters=25, units_count=100_000, unit_name="samples"
    )
elif workload == "steim2_decode_1m":
    res = bench_loop(
        lambda: make_steim2_bytes(1_000_000),
        lambda raw: read(io.BytesIO(raw), format="MSEED"),
        warmup=2, iters=12, units_count=1_000_000, unit_name="samples"
    )
elif workload == "fft_rfft_irfft_100k":
    res = bench_loop(
        lambda: make_wave_f64(100_000),
        lambda x: spfft.irfft(spfft.rfft(x), n=len(x)),
        warmup=3, iters=25, units_count=100_000, unit_name="samples"
    )
elif workload == "fft_rfft_irfft_1m":
    res = bench_loop(
        lambda: make_wave_f64(1_000_000),
        lambda x: spfft.irfft(spfft.rfft(x), n=len(x)),
        warmup=2, iters=15, units_count=1_000_000, unit_name="samples"
    )
elif workload == "remove_response_100k":
    def setup_rr_100k():
        inv = read_inventory(xml_path)
        ch = inv[0][0][0]
        x = make_wave_f64(100_000) * 1e4
        hdr = {
            "network": inv[0].code, "station": inv[0][0].code,
            "location": ch.location_code, "channel": ch.code,
            "sampling_rate": float(ch.sample_rate),
            "starttime": ch.start_date + 3600.0,
        }
        return inv, x, hdr
    def run_rr(ctx):
        inv, x, hdr = ctx
        tr = Trace(data=x.copy(), header=hdr.copy())
        tr.remove_response(inventory=inv, output="VEL", pre_filt=[0.005, 0.01, 15.0, 20.0], water_level=60.0)
    res = bench_loop(setup_rr_100k, run_rr, warmup=1, iters=5, units_count=100_000, unit_name="samples")
elif workload == "remove_response_1m":
    def setup_rr_1m():
        inv = read_inventory(xml_path)
        ch = inv[0][0][0]
        x = make_wave_f64(1_000_000) * 1e4
        hdr = {
            "network": inv[0].code, "station": inv[0][0].code,
            "location": ch.location_code, "channel": ch.code,
            "sampling_rate": float(ch.sample_rate),
            "starttime": ch.start_date + 3600.0,
        }
        return inv, x, hdr
    def run_rr_1m(ctx):
        inv, x, hdr = ctx
        tr = Trace(data=x.copy(), header=hdr.copy())
        tr.remove_response(inventory=inv, output="VEL", pre_filt=[0.005, 0.01, 15.0, 20.0], water_level=60.0)
    res = bench_loop(setup_rr_1m, run_rr_1m, warmup=1, iters=3, units_count=1_000_000, unit_name="samples")
elif workload == "taup_1000_queries_iasp91":
    def setup_taup():
        m = TauPyModel("iasp91")
        depths = [0.0, 15.0, 35.0, 120.0, 450.0]
        dists = np.linspace(5.0, 175.0, 200)
        queries = [(depths[i % len(depths)], float(dists[i % len(dists)])) for i in range(1000)]
        phases = ["P", "S", "pP", "PP", "PcP", "PKP", "PKIKP", "SKS"]
        return m, queries, phases
    def run_taup(ctx):
        m, queries, phases = ctx
        total = 0
        for d, deg in queries:
            arrs = m.get_travel_times(source_depth_in_km=d, distance_in_degree=deg, phase_list=phases)
            total += len(arrs)
        return total
    res = bench_loop(setup_taup, run_taup, warmup=1, iters=3, units_count=1000, unit_name="queries")
else:
    raise ValueError(f"Unknown workload: {workload}")

print(json.dumps(res))
"""


def main() -> None:
    import numpy as np
    import obspy
    import psutil
    import scipy

    workloads = [
        "cold_import",
        "filter_bandpass_causal_100k",
        "filter_bandpass_zerophase_100k",
        "filter_bandpass_causal_1m",
        "filter_bandpass_zerophase_1m",
        "steim2_decode_100k",
        "steim2_decode_1m",
        "fft_rfft_irfft_100k",
        "fft_rfft_irfft_1m",
        "remove_response_100k",
        "remove_response_1m",
        "taup_1000_queries_iasp91",
    ]

    results = {}
    for wl in workloads:
        print(f"Running isolated benchmark: {wl} ...", flush=True)
        proc_res = subprocess.run(
            [sys.executable, "-c", WORKER_CODE, wl, str(XML_FIXTURE)],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln.strip() for ln in proc_res.stdout.splitlines() if ln.strip()]
        parsed = json.loads(lines[-1])
        results[wl] = parsed
        print(
            f"  -> median={parsed['median_ms']:.2f} ms | "
            f"peak_rss={parsed['peak_rss_bytes'] / (1024*1024):.1f} MiB | "
            f"tracemalloc_peak={parsed['tracemalloc_peak_bytes'] / (1024*1024):.2f} MiB",
            flush=True,
        )

    summary = {
        "schema_version": "1.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_ram_bytes": int(psutil.virtual_memory().total),
            "python_version": platform.python_version(),
            "obspy_version": obspy.__version__,
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
        },
        "benchmarks": results,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved baseline benchmark receipts to: {OUT_PATH}")


if __name__ == "__main__":
    main()
