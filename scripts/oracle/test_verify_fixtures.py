#!/usr/bin/env python3
"""
Pytest verification suite auditing every generated binary & JSON fixture and
benchmark receipt in `oracle_staging/`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from obspy import read, read_events, read_inventory

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
FIXTURES = Path(
    os.environ.get("EARTHSNAIL_FIXTURES_DIR", str(REPO_ROOT / "tests" / "fixtures"))
)
BENCH_JSON = Path(
    os.environ.get("EARTHSNAIL_BENCH_OUT", str(REPO_ROOT / "results" / "obspy_baseline_bench.json"))
)
IO_DIR = FIXTURES / "io"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_all_finite(obj) -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float: {obj}"
    elif isinstance(obj, dict):
        for v in obj.values():
            assert_all_finite(v)
    elif isinstance(obj, list):
        for x in obj:
            assert_all_finite(x)


def test_sac_fixtures():
    meta = json.loads((IO_DIR / "sac_expected.json").read_text(encoding="utf-8"))
    assert_all_finite(meta)
    for key, fmt in [("little_endian", "SAC"), ("big_endian", "SAC"), ("alpha", "SACXY")]:
        entry = meta["files"][key]
        p = IO_DIR / entry["file"]
        assert p.exists() and p.stat().st_size > 0
        assert sha256_of(p) == entry["sha256"]
        tr = read(str(p), format=fmt)[0]
        assert len(tr.data) == 500
        for req_hdr in ["stla", "stlo", "evla", "evlo", "evdp", "b", "o", "t0", "kt0", "kstnm", "knetwk", "kcmpnm"]:
            assert req_hdr in entry["sac_header"], f"Missing SAC header {req_hdr} in {key}"
    # Binary LE & BE must be bit-exact on data
    assert meta["files"]["little_endian"]["data_f32_as_f64"] == meta["files"]["big_endian"]["data_f32_as_f64"]


def test_mseed_fixtures():
    meta = json.loads((IO_DIR / "mseed_expected.json").read_text(encoding="utf-8"))
    assert_all_finite(meta)
    for enc_key in ["steim1", "steim2", "int32", "float64"]:
        entry = meta["files"][enc_key]
        p = IO_DIR / entry["file"]
        assert p.exists() and p.stat().st_size >= 1024
        assert entry["num_records"] >= 2
        assert sha256_of(p) == entry["sha256"]
        tr = read(str(p), format="MSEED")[0]
        assert np.allclose(tr.data, entry["samples"], rtol=0, atol=0)


def test_metadata_and_real_event_fixtures():
    meta = json.loads((IO_DIR / "metadata_expected.json").read_text(encoding="utf-8"))
    assert_all_finite(meta)
    xml_p = IO_DIR / meta["stationxml"]["file"]
    pz_p = IO_DIR / meta["sacpz"]["file"]
    qml_p = IO_DIR / meta["quakeml"]["file"]
    for p in [xml_p, pz_p, qml_p]:
        assert p.exists() and p.stat().st_size > 0
    assert sha256_of(xml_p) == meta["stationxml"]["sha256"]
    assert sha256_of(pz_p) == meta["sacpz"]["sha256"]
    assert sha256_of(qml_p) == meta["quakeml"]["sha256"]
    assert meta["stationxml"]["frequencies_hz"] == [0.01, 0.1, 1.0, 5.0, 15.0]
    for unit in ["DISP", "VEL", "ACC"]:
        assert len(meta["stationxml"]["evalresp_complex_response"][unit]) == 5
        assert len(meta["sacpz"]["complex_response"][unit]) == 5
    inv = read_inventory(str(xml_p))
    assert len(inv[0][0]) >= 1
    cat = read_events(str(qml_p))
    assert len(cat) == meta["quakeml"]["num_events"] >= 3

    real_meta = json.loads((IO_DIR / "real_event" / "real_event_expected.json").read_text(encoding="utf-8"))
    assert_all_finite(real_meta)
    mseed_real = IO_DIR / real_meta["mseed_file"]
    xml_real = IO_DIR / real_meta["stationxml_file"]
    assert mseed_real.exists() and mseed_real.stat().st_size > 0
    assert xml_real.exists() and xml_real.stat().st_size > 0
    st_real = read(str(mseed_real))
    assert len(st_real) == 3
    assert len(real_meta["channels"]) == 3
    for ch_entry in real_meta["channels"]:
        sp = IO_DIR / ch_entry["sac_file"]
        assert sp.exists() and sp.stat().st_size > 0
        assert sha256_of(sp) == ch_entry["sha256"]


def test_geodesics_fixtures():
    geo = json.loads((FIXTURES / "geodesics.json").read_text(encoding="utf-8"))
    assert_all_finite(geo)
    assert geo["num_pairs"] == 25
    assert len(geo["pairs"]) == 25
    cats = {p["category"] for p in geo["pairs"]}
    assert cats == {"regional", "teleseismic", "trans_equatorial", "polar", "near_coincident", "near_antipodal"}
    # Near-antipodal distances must be > 19,800 km (half-meridian ~20,003.9 km)
    for p in geo["pairs"]:
        if p["category"] == "near_antipodal":
            assert p["dist_m"] > 19_800_000.0


def test_dsp_fixtures():
    dsp = json.loads((FIXTURES / "dsp.json").read_text(encoding="utf-8"))
    assert_all_finite(dsp)
    assert set(dsp["detrend"].keys()) >= {"demean", "linear", "polynomial_order_3"}
    for fam in ["butter", "cheby1", "bessel"]:
        for mode in ["bandpass", "lowpass", "highpass", "bandstop"]:
            entry = dsp["filters"][fam][mode]
            assert len(entry["zerophase_false"]) == 1000
            assert len(entry["zerophase_true"]) == 1000
            assert len(entry["zerophase_sosfiltfilt"]) == 1000
            assert len(entry["sos_sections"]) >= 2
    for unit in ["VEL", "DISP", "ACC"]:
        assert len(dsp["remove_response"]["stationxml_outputs"][unit]) == 4096
        assert len(dsp["remove_response"]["sacpz_outputs"][unit]) == 4096
    assert dsp["cross_correlation"]["expected_shift_samples"] == -17
    assert dsp["baer_picker"]["rjob_ehz_1000"]["pick_sample"] > 0
    assert dsp["rotation_and_polarization"]["flinn_p_window"]["rectilinearity"] > 0.99
    assert len(dsp["peterson_noise_models"]["nlnm"]["periods_s"]) == 1001
    assert len(dsp["peterson_noise_models"]["nhnm"]["periods_s"]) == 1001
    assert len(dsp["lanczos_resampling"]["downsample_100hz_to_40hz"]["output"]) == 200


def test_taup_fixtures():
    taup = json.loads((FIXTURES / "taup.json").read_text(encoding="utf-8"))
    assert_all_finite(taup)
    assert taup["models"] == ["iasp91", "ak135", "prem"]
    assert taup["num_grid_queries"] == 3 * 6 * 11
    assert taup["total_arrivals"] > 1500
    assert len(taup["pierce_examples"]) >= 4
    assert len(taup["ray_path_examples"]) >= 3


def test_baseline_benchmarks():
    bench_p = BENCH_JSON
    assert bench_p.exists() and bench_p.stat().st_size > 0
    bench = json.loads(bench_p.read_text(encoding="utf-8"))
    assert_all_finite(bench)
    required = [
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
    for k in required:
        assert k in bench["benchmarks"], f"Missing benchmark {k}"
        assert bench["benchmarks"][k]["median_ms"] > 0.0
        assert bench["benchmarks"][k]["peak_rss_bytes"] > 0
