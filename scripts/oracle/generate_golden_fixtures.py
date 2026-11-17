#!/usr/bin/env python3
"""
Deterministic Ground-Truth Differential Fixture Generator for EarthSnail.

Generates binary & JSON golden reference fixtures using real ObsPy, SciPy, and
TauP (`obspy.taup.TauPyModel`) inside `fixtures/`:
  - `fixtures/io/`          : SAC (LE, BE, Alpha), miniSEED v2 (STEIM1, STEIM2,
                              INT32, FLOAT64), StationXML, SACPZ, QuakeML, and a
                              real 3-component broadband earthquake recording.
  - `fixtures/geodesics.json`: 25 WGS84 coordinate pairs across 6 regimes.
  - `fixtures/dsp.json`      : Detrend, tapers, SOS filters (Butterworth,
                              Chebyshev-I, Bessel; bp/lp/hp/bs; causal & 2-pass),
                              StationXML & SACPZ `remove_response`, Hilbert
                              envelope, normalized cross-correlation, STA/LTA,
                              Baer picker, 3C rotations, Flinn/polarization,
                              Peterson NLNM/NHNM, and Lanczos resampling.
  - `fixtures/taup.json`     : Exact travel times, ray parameters, takeoff &
                              incident angles across `iasp91`, `ak135`, `prem`
                              (6 depths x 11 distances x 24 phases), plus pierce
                              points and full ray-path trajectories.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import obspy
import scipy
import scipy.signal as spsig
from obspy import Catalog, Stream, Trace, UTCDateTime, read, read_events, read_inventory
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.io.sac import SACTrace
from obspy.io.sac.sacpz import attach_paz
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.filter import bandpass, bandstop, envelope, highpass, lowpass
from obspy.signal.interpolation import lanczos_interpolation
from obspy.signal.invsim import cosine_taper, paz_to_freq_resp, simulate_seismometer
from obspy.signal.polarization import flinn, polarization_analysis
from obspy.signal.rotate import rotate_ne_rt, rotate_zne_lqt
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm
from obspy.signal.trigger import classic_sta_lta, pk_baer, recursive_sta_lta, trigger_onset
from obspy.signal.util import _npts2nfft
from obspy.taup import TauPyModel

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
FIXTURES_DIR = Path(
    os.environ.get("EARTHSNAIL_FIXTURES_DIR", str(REPO_ROOT / "tests" / "fixtures"))
)
IO_DIR = FIXTURES_DIR / "io"


def to_py(val: Any) -> Any:
    """Convert NumPy scalars/arrays/complex/UTCDateTime into strict JSON types."""
    if isinstance(val, (np.floating, float)):
        f = float(val)
        if not math.isfinite(f):
            raise ValueError(f"Non-finite float encountered: {f!r}")
        return f
    if isinstance(val, (np.integer, int)) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (complex, np.complexfloating)):
        return {
            "real": to_py(val.real),
            "imag": to_py(val.imag),
            "abs": to_py(abs(val)),
            "phase_rad": to_py(math.atan2(float(val.imag), float(val.real))),
        }
    if isinstance(val, UTCDateTime):
        return val.isoformat() + "Z" if not val.isoformat().endswith("Z") else val.isoformat()
    if isinstance(val, np.ndarray):
        return [to_py(x) for x in val.tolist()]
    if isinstance(val, dict):
        return {str(k): to_py(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_py(x) for x in val]
    return val


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = to_py(payload)
    text = json.dumps(clean, indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================================
# 1. I/O FIXTURES (`fixtures/io/`)
# ============================================================================

def build_steim_torture_int32(npts: int = 2400, include_full_32bit: bool = False) -> np.ndarray:
    """
    Construct a deterministic int32 waveform spanning multiple 512-byte miniSEED
    records whose first-differences systematically exercise every Steim-1 &
    Steim-2 nibble packing regime:
      - d = 0                  (constant run)
      - |d| <= 7               (4-bit signed, Steim-2 dnib=10, 7 diffs/word)
      - |d| <= 15              (5-bit signed, Steim-2 dnib=01, 6 diffs/word)
      - |d| <= 31              (6-bit signed, Steim-2 dnib=00, 5 diffs/word)
      - |d| <= 127             (8-bit signed, Steim-1 c=01, 4 diffs/word)
      - |d| <= 511             (10-bit signed, Steim-2 c=10 dnib=11, 3 diffs/word)
      - |d| <= 16383           (15-bit signed, Steim-2 c=10 dnib=10, 2 diffs/word)
      - |d| <= 32767           (16-bit signed, Steim-1 c=10, 2 diffs/word)
      - |d| <= 536_870_911     (30-bit signed, Steim-2 c=10 dnib=01, 1 diff/word)
      - |d| >  536_870_911     (32-bit signed, Steim-1 c=11 only, if enabled)
    """
    rng = np.random.default_rng(2026111601)
    diffs = np.zeros(npts, dtype=np.int64)

    # Regime slices (each >= 180 samples to fill multiple 64-byte frames)
    diffs[0:150] = 0
    diffs[150:400] = rng.integers(-7, 8, size=250)
    diffs[400:650] = rng.integers(-15, 16, size=250)
    diffs[650:900] = rng.integers(-31, 32, size=250)
    diffs[900:1150] = rng.integers(-120, 121, size=250)
    diffs[1150:1400] = rng.integers(-500, 501, size=250)
    diffs[1400:1650] = rng.integers(-15000, 15001, size=250)
    diffs[1650:1900] = rng.integers(-30000, 30001, size=250)

    # Sharp impulses (30-bit signed safe: max +/- 536_870_911)
    diffs[1900:2400] = rng.integers(-60, 61, size=500)
    impulse_pairs = [
        (1920, 250_000_000),
        (1921, -250_000_000),
        (2010, -480_000_000),
        (2011, 480_000_000),
        (2150, 536_000_000),
        (2151, -536_000_000),
    ]
    for idx, val in impulse_pairs:
        diffs[idx] = val

    if include_full_32bit:
        # Exceeds 30-bit signed range [-2^29, 2^29-1] -> forces true 32-bit Steim-1 word
        diffs[2280] = 1_200_000_000
        diffs[2281] = -1_200_000_000

    x = np.cumsum(diffs) + 12345
    assert np.all((x >= np.iinfo(np.int32).min) & (x <= np.iinfo(np.int32).max))
    return x.astype(np.int32)


def generate_io_fixtures() -> dict[str, Any]:
    IO_DIR.mkdir(parents=True, exist_ok=True)
    obspy_dir = Path(obspy.__file__).resolve().parent

    # ------------------------------------------------------------------
    # 1A. SAC Binary (LE, BE) & SAC Alpha
    # ------------------------------------------------------------------
    npts_sac = 500
    dt_sac = 0.01  # 100 Hz
    t_sac = np.arange(npts_sac, dtype=np.float64) * dt_sac
    sac_wave = (
        125.0 * np.exp(-((t_sac - 1.80) / 0.18) ** 2) * np.sin(2.0 * np.pi * 6.5 * (t_sac - 1.80))
        + 42.5 * np.exp(-((t_sac - 3.40) / 0.35) ** 2) * np.cos(2.0 * np.pi * 2.5 * (t_sac - 3.40))
        + 1.75 * np.sin(2.0 * np.pi * 0.4 * t_sac)
    ).astype(np.float32)

    sac_obj = SACTrace(
        data=sac_wave,
        delta=dt_sac,
        b=0.0,
        o=0.50,
        a=1.80,
        t0=1.80,
        t1=3.40,
        ko="ORIGIN",
        ka="IPU0",
        kt0="P",
        kt1="S",
        stla=34.94598,
        stlo=-106.45713,
        stel=1671.0,
        stdp=145.0,
        evla=38.29700,
        evlo=142.37300,
        evdp=29.0,
        mag=9.1,
        cmpaz=0.0,
        cmpinc=0.0,
        kstnm="ANMO",
        knetwk="IU",
        kcmpnm="BHZ",
        khole="10",
        kevnm="TOHOKU2011MW91",
        nzyear=2011,
        nzjday=70,
        nzhour=5,
        nzmin=46,
        nzsec=24,
        nzmsec=120,
        iztype="ib",
        iftype="itime",
        idep="ivel",
        leven=True,
        lpspol=True,
        lovrok=True,
        lcalda=True,
    )

    sac_le_path = IO_DIR / "test_le.sac"
    sac_be_path = IO_DIR / "test_be.sac"
    sac_alpha_path = IO_DIR / "test_alpha.sac"

    sac_obj.write(str(sac_le_path), byteorder="little")
    sac_obj.write(str(sac_be_path), byteorder="big")

    tr_sac_le = read(str(sac_le_path), format="SAC")[0]
    tr_sac_be = read(str(sac_be_path), format="SAC")[0]
    assert np.array_equal(tr_sac_le.data, tr_sac_be.data), "SAC LE vs BE mismatch!"

    tr_sac_le.write(str(sac_alpha_path), format="SACXY")
    tr_sac_alpha = read(str(sac_alpha_path), format="SACXY")[0]

    def dump_sac_trace(tr: Trace, path: Path, encoding_label: str) -> dict[str, Any]:
        sac_hdr = {}
        for k, v in sorted(tr.stats.sac.items()):
            if k.startswith("_"):
                continue
            if isinstance(v, str):
                sac_hdr[k] = v.strip()
            else:
                sac_hdr[k] = to_py(v)
        return {
            "file": path.name,
            "encoding": encoding_label,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "stats": {
                "network": tr.stats.network,
                "station": tr.stats.station,
                "location": tr.stats.location,
                "channel": tr.stats.channel,
                "starttime": to_py(tr.stats.starttime),
                "endtime": to_py(tr.stats.endtime),
                "sampling_rate": to_py(tr.stats.sampling_rate),
                "delta": to_py(tr.stats.delta),
                "npts": int(tr.stats.npts),
                "calib": to_py(tr.stats.calib),
            },
            "sac_header": sac_hdr,
            "data_f32_as_f64": to_py(tr.data.astype(np.float64)),
        }

    sac_expected = {
        "schema_version": "1.0",
        "description": "ObsPy ground-truth SAC Binary (LE & BE) and SAC Alpha (SACXY) fixtures.",
        "files": {
            "little_endian": dump_sac_trace(tr_sac_le, sac_le_path, "SAC_BINARY_LE"),
            "big_endian": dump_sac_trace(tr_sac_be, sac_be_path, "SAC_BINARY_BE"),
            "alpha": dump_sac_trace(tr_sac_alpha, sac_alpha_path, "SAC_ALPHA"),
        },
    }
    write_json(IO_DIR / "sac_expected.json", sac_expected)

    # ------------------------------------------------------------------
    # 1B. miniSEED v2 (STEIM1, STEIM2, INT32, FLOAT64)
    # ------------------------------------------------------------------
    mseed_start = UTCDateTime("2020-05-15T12:34:56.789000Z")
    base_hdr = {
        "network": "IU",
        "station": "ANMO",
        "location": "10",
        "channel": "BHZ",
        "starttime": mseed_start,
        "sampling_rate": 40.0,
    }

    data_steim1 = build_steim_torture_int32(npts=2400, include_full_32bit=True)
    data_steim2 = build_steim_torture_int32(npts=2400, include_full_32bit=False)
    data_int32 = data_steim1.copy()

    t_f64 = np.arange(1200, dtype=np.float64) / 40.0
    data_f64 = (
        1.23456789012345e-4 * np.sin(2.0 * np.pi * 1.25 * t_f64)
        + 9.87654321098765e-3 * np.exp(-((t_f64 - 12.0) / 1.5) ** 2) * np.cos(2.0 * np.pi * 4.5 * t_f64)
    )
    data_f64[600] += 42.1953125  # sharp float64 impulse

    mseed_specs = [
        ("mseed_steim1.mseed", "STEIM1", 10, data_steim1),
        ("mseed_steim2.mseed", "STEIM2", 11, data_steim2),
        ("mseed_int32.mseed", "INT32", 3, data_int32),
        ("mseed_float64.mseed", "FLOAT64", 5, data_f64),
    ]

    mseed_expected_entries = {}
    for fname, enc_name, enc_code, arr in mseed_specs:
        out_p = IO_DIR / fname
        tr_out = Trace(data=arr.copy(), header=copy.deepcopy(base_hdr))
        tr_out.write(str(out_p), format="MSEED", encoding=enc_name, reclen=512, byteorder=">")
        st_in = read(str(out_p), format="MSEED")
        assert len(st_in) == 1
        tr_in = st_in[0]
        assert np.array_equal(tr_in.data, arr), f"miniSEED roundtrip mismatch for {enc_name}"
        size_bytes = out_p.stat().st_size
        assert size_bytes % 512 == 0 and (size_bytes // 512) >= 2, f"Expected multi-record 512B file for {fname}"

        first_diffs = np.diff(arr.astype(np.int64)) if arr.dtype == np.int32 else np.diff(arr)
        mseed_expected_entries[enc_name.lower()] = {
            "file": fname,
            "encoding_name": enc_name,
            "seed_encoding_code": enc_code,
            "record_length_bytes": 512,
            "num_records": size_bytes // 512,
            "size_bytes": size_bytes,
            "sha256": sha256_file(out_p),
            "dtype": str(arr.dtype),
            "stats": {
                "network": tr_in.stats.network,
                "station": tr_in.stats.station,
                "location": tr_in.stats.location,
                "channel": tr_in.stats.channel,
                "starttime": to_py(tr_in.stats.starttime),
                "endtime": to_py(tr_in.stats.endtime),
                "sampling_rate": to_py(tr_in.stats.sampling_rate),
                "npts": int(tr_in.stats.npts),
            },
            "max_abs_first_diff": to_py(np.max(np.abs(first_diffs))),
            "samples": to_py(tr_in.data),
        }

    write_json(
        IO_DIR / "mseed_expected.json",
        {
            "schema_version": "1.0",
            "description": "ObsPy miniSEED v2 multi-record fixtures (STEIM1, STEIM2, INT32, FLOAT64).",
            "files": mseed_expected_entries,
        },
    )

    # ------------------------------------------------------------------
    # 1C. StationXML (`iu_anmo_bhz.xml`), SACPZ (`iu_anmo_bhz.sacpz`), QuakeML (`events.qml`)
    # ------------------------------------------------------------------
    xml_path = IO_DIR / "iu_anmo_bhz.xml"
    sacpz_path = IO_DIR / "iu_anmo_bhz.sacpz"
    qml_path = IO_DIR / "events.qml"

    # Fetch 40 Hz IU.ANMO.10.BHZ StationXML (with 3 stages: PZ + 2 FIR stages, Nyquist=20 Hz > 15 Hz)
    # Fallback to ObsPy bundled IU_ANMO_00_BHZ.xml if offline.
    try:
        fdsn_client = Client("EARTHSCOPE", timeout=15)
        inv_anmo = fdsn_client.get_stations(
            network="IU",
            station="ANMO",
            location="10",
            channel="BHZ",
            starttime=UTCDateTime("2018-01-01T00:00:00"),
            endtime=UTCDateTime("2018-01-02T00:00:00"),
            level="response",
        )
        inv_anmo.write(str(xml_path), format="STATIONXML")
    except Exception:
        bundled_xml = obspy_dir / "core" / "tests" / "data" / "IU_ANMO_00_BHZ.xml"
        inv_anmo = read_inventory(str(bundled_xml))
        inv_anmo.write(str(xml_path), format="STATIONXML")

    inv_anmo = read_inventory(str(xml_path), format="STATIONXML")
    inv_anmo.write(str(sacpz_path), format="SACPZ")

    net_meta = inv_anmo[0]
    sta_meta = net_meta[0]
    cha_meta = sta_meta[0]
    resp = cha_meta.response

    eval_freqs = np.array([0.01, 0.1, 1.0, 5.0, 15.0], dtype=np.float64)
    evalresp_by_unit = {}
    for unit in ["DISP", "VEL", "ACC"]:
        cvals = resp.get_evalresp_response_for_frequencies(eval_freqs, output=unit)
        evalresp_by_unit[unit] = [to_py(c) for c in cvals]

    # Parse back SACPZ via attach_paz and evaluate exact continuous-time s=i*2*pi*f response
    tr_paz_disp = Trace()
    attach_paz(tr_paz_disp, str(sacpz_path), todisp=False, tovel=False, torad=False, tohz=False)
    paz_disp = tr_paz_disp.stats.paz

    def eval_sacpz_at_freqs(poles: list[complex], zeros: list[complex], constant: float, freqs_hz: np.ndarray) -> list[complex]:
        out = []
        for f in freqs_hz:
            s = 1j * 2.0 * math.pi * float(f)
            num = complex(constant, 0.0)
            for z in zeros:
                num *= (s - z)
            den = complex(1.0, 0.0)
            for p in poles:
                den *= (s - p)
            out.append(num / den)
        return out

    # SACPZ exported from StationXML has INPUT UNIT : M (displacement).
    # Removing one zero at origin yields VEL (M/S); removing two zeros at origin yields ACC (M/S^2).
    zeros_disp = list(paz_disp.zeros)
    zeros_vel = list(zeros_disp)
    zeros_vel.remove(0j)
    zeros_acc = list(zeros_vel)
    zeros_acc.remove(0j)

    sacpz_by_unit = {
        "DISP": [to_py(c) for c in eval_sacpz_at_freqs(list(paz_disp.poles), zeros_disp, float(paz_disp.gain), eval_freqs)],
        "VEL": [to_py(c) for c in eval_sacpz_at_freqs(list(paz_disp.poles), zeros_vel, float(paz_disp.gain), eval_freqs)],
        "ACC": [to_py(c) for c in eval_sacpz_at_freqs(list(paz_disp.poles), zeros_acc, float(paz_disp.gain), eval_freqs)],
    }

    pz_stage = resp.response_stages[0]
    stages_summary = []
    for stg in resp.response_stages:
        stg_info: dict[str, Any] = {
            "stage_sequence_number": int(stg.stage_sequence_number),
            "stage_type": type(stg).__name__,
            "input_units": stg.input_units,
            "output_units": stg.output_units,
            "stage_gain": to_py(stg.stage_gain),
            "stage_gain_frequency": to_py(stg.stage_gain_frequency),
            "decimation_input_sample_rate": to_py(stg.decimation_input_sample_rate) if stg.decimation_input_sample_rate is not None else None,
            "decimation_factor": int(stg.decimation_factor) if stg.decimation_factor is not None else None,
        }
        if hasattr(stg, "poles") and hasattr(stg, "zeros"):
            stg_info["pz_transfer_function_type"] = stg.pz_transfer_function_type
            stg_info["normalization_factor"] = to_py(stg.normalization_factor)
            stg_info["normalization_frequency"] = to_py(stg.normalization_frequency)
            stg_info["zeros"] = [to_py(z) for z in stg.zeros]
            stg_info["poles"] = [to_py(p) for p in stg.poles]
        if hasattr(stg, "numerator"):
            stg_info["cf_transfer_function_type"] = getattr(stg, "cf_transfer_function_type", None)
            stg_info["numerator_len"] = len(stg.numerator)
            stg_info["denominator_len"] = len(stg.denominator)
            stg_info["numerator"] = to_py(np.asarray(stg.numerator, dtype=np.float64))
            stg_info["denominator"] = to_py(np.asarray(stg.denominator, dtype=np.float64))
        stages_summary.append(stg_info)

    # QuakeML event catalog: combine ObsPy's 3-event catalog (picks, origins, magnitudes)
    # with ObsPy's QuakeML 1.2 focal mechanism + moment tensor example event.
    cat = read_events()
    fm_xml = obspy_dir / "io" / "quakeml" / "tests" / "data" / "quakeml_1.2_focalmechanism.xml"
    if fm_xml.exists():
        raw_fm_text = fm_xml.read_text(encoding="utf-8").replace("smi://", "smi:")
        import io as _io
        cat_fm = read_events(_io.BytesIO(raw_fm_text.encode("utf-8")))
        for ev in cat_fm:
            cat.append(ev)
    cat.write(str(qml_path), format="QUAKEML")
    cat_reloaded = read_events(str(qml_path), format="QUAKEML")

    events_summary = []
    for ev in cat_reloaded:
        org = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        ev_dict: dict[str, Any] = {
            "resource_id": str(ev.resource_id),
            "num_origins": len(ev.origins),
            "num_magnitudes": len(ev.magnitudes),
            "num_picks": len(ev.picks),
            "num_focal_mechanisms": len(ev.focal_mechanisms),
            "preferred_origin": {
                "time": to_py(org.time),
                "latitude": to_py(org.latitude),
                "longitude": to_py(org.longitude),
                "depth_m": to_py(org.depth) if org.depth is not None else None,
            } if org else None,
            "preferred_magnitude": {
                "mag": to_py(mag.mag),
                "magnitude_type": mag.magnitude_type,
            } if mag else None,
            "picks": [
                {
                    "time": to_py(pk.time),
                    "phase_hint": pk.phase_hint,
                    "network": pk.waveform_id.network_code if pk.waveform_id else None,
                    "station": pk.waveform_id.station_code if pk.waveform_id else None,
                    "channel": pk.waveform_id.channel_code if pk.waveform_id else None,
                    "polarity": pk.polarity,
                }
                for pk in ev.picks
            ],
            "focal_mechanisms": [],
        }
        for fm in ev.focal_mechanisms:
            fm_entry: dict[str, Any] = {"resource_id": str(fm.resource_id)}
            if fm.nodal_planes and fm.nodal_planes.nodal_plane_1:
                np1 = fm.nodal_planes.nodal_plane_1
                fm_entry["nodal_plane_1"] = {
                    "strike": to_py(np1.strike),
                    "dip": to_py(np1.dip),
                    "rake": to_py(np1.rake),
                }
            if fm.nodal_planes and fm.nodal_planes.nodal_plane_2:
                np2 = fm.nodal_planes.nodal_plane_2
                fm_entry["nodal_plane_2"] = {
                    "strike": to_py(np2.strike),
                    "dip": to_py(np2.dip),
                    "rake": to_py(np2.rake),
                }
            if fm.moment_tensor and fm.moment_tensor.tensor:
                mt = fm.moment_tensor
                ten = mt.tensor
                fm_entry["moment_tensor"] = {
                    "scalar_moment_nm": to_py(mt.scalar_moment) if mt.scalar_moment is not None else None,
                    "m_rr": to_py(ten.m_rr),
                    "m_tt": to_py(ten.m_tt),
                    "m_pp": to_py(ten.m_pp),
                    "m_rt": to_py(ten.m_rt),
                    "m_rp": to_py(ten.m_rp),
                    "m_tp": to_py(ten.m_tp),
                }
            ev_dict["focal_mechanisms"].append(fm_entry)
        events_summary.append(ev_dict)

    metadata_expected = {
        "schema_version": "1.0",
        "stationxml": {
            "file": xml_path.name,
            "sha256": sha256_file(xml_path),
            "network": net_meta.code,
            "station": sta_meta.code,
            "location": cha_meta.location_code,
            "channel": cha_meta.code,
            "latitude": to_py(cha_meta.latitude),
            "longitude": to_py(cha_meta.longitude),
            "elevation_m": to_py(cha_meta.elevation),
            "depth_m": to_py(cha_meta.depth),
            "azimuth_deg": to_py(cha_meta.azimuth),
            "dip_deg": to_py(cha_meta.dip),
            "sample_rate_hz": to_py(cha_meta.sample_rate),
            "overall_sensitivity": to_py(resp.instrument_sensitivity.value),
            "sensitivity_frequency_hz": to_py(resp.instrument_sensitivity.frequency),
            "input_units": resp.instrument_sensitivity.input_units,
            "output_units": resp.instrument_sensitivity.output_units,
            "stages": stages_summary,
            "frequencies_hz": to_py(eval_freqs),
            "evalresp_complex_response": evalresp_by_unit,
        },
        "sacpz": {
            "file": sacpz_path.name,
            "sha256": sha256_file(sacpz_path),
            "input_unit": "M",
            "output_unit": "COUNTS",
            "constant": to_py(paz_disp.gain),
            "a0": to_py(pz_stage.normalization_factor),
            "sensitivity": to_py(resp.instrument_sensitivity.value),
            "zeros_disp_rad_s": [to_py(z) for z in zeros_disp],
            "poles_rad_s": [to_py(p) for p in paz_disp.poles],
            "frequencies_hz": to_py(eval_freqs),
            "complex_response": sacpz_by_unit,
        },
        "quakeml": {
            "file": qml_path.name,
            "sha256": sha256_file(qml_path),
            "num_events": len(events_summary),
            "events": events_summary,
        },
    }
    write_json(IO_DIR / "metadata_expected.json", metadata_expected)

    # ------------------------------------------------------------------
    # 1D. Real 3-Component Broadband Earthquake Recording
    # ------------------------------------------------------------------
    real_dir = IO_DIR / "real_event"
    real_dir.mkdir(parents=True, exist_ok=True)
    real_source = "earthscope_fdsn_tohoku_2011_g_can"
    ev_info = {
        "name": "2011-03-11 Mw 9.1 Tohoku-Oki Earthquake",
        "origin_time": "2011-03-11T05:46:24.120000Z",
        "evla": 38.297,
        "evlo": 142.373,
        "evdp_km": 29.0,
        "mag": 9.1,
    }
    try:
        fdsn_client = Client("EARTHSCOPE", timeout=20)
        t_win_0 = UTCDateTime("2011-03-11T05:56:30")
        t_win_1 = UTCDateTime("2011-03-11T06:08:30")
        st_real = fdsn_client.get_waveforms("G", "CAN", "00", "BH?", t_win_0, t_win_1)
        st_real.merge(method=1, fill_value="interpolate")
        st_real.trim(t_win_0, t_win_1, nearest_sample=True)
        st_real.sort(keys=["channel"])
        inv_real = fdsn_client.get_stations(
            network="G",
            station="CAN",
            location="00",
            channel="BH?",
            starttime=t_win_0,
            endtime=t_win_1,
            level="response",
        )
    except Exception:
        real_source = "obspy_builtin_rjob_2009"
        ev_info = {
            "name": "2009-08-24 Bavaria Local Earthquake (ObsPy Built-in 3C)",
            "origin_time": "2009-08-24T00:20:03.000000Z",
            "evla": 47.74,
            "evlo": 12.86,
            "evdp_km": 6.0,
            "mag": 1.8,
        }
        st_real = read()
        st_real.sort(keys=["channel"])
        inv_real = read_inventory().select(network="BW", station="RJOB")

    real_mseed_path = real_dir / "real_3c.mseed"
    real_xml_path = real_dir / "real_3c_station.xml"
    st_real.write(str(real_mseed_path), format="MSEED")
    inv_real.write(str(real_xml_path), format="STATIONXML")

    # Compute TauP theoretical P and S arrivals and populate 3 SAC files
    taup_iasp = TauPyModel("iasp91")
    sta0 = inv_real[0][0]
    gcarc_deg = float(locations2degrees(ev_info["evla"], ev_info["evlo"], sta0.latitude, sta0.longitude))
    arr_ps = taup_iasp.get_travel_times(
        source_depth_in_km=ev_info["evdp_km"],
        distance_in_degree=gcarc_deg,
        phase_list=["P", "p", "S", "s"],
    )
    p_arr = next((a for a in arr_ps if a.name in ("P", "p")), None)
    s_arr = next((a for a in arr_ps if a.name in ("S", "s")), None)
    ev_origin_utc = UTCDateTime(ev_info["origin_time"])

    sac_files_meta = []
    for tr in st_real:
        ch_code = tr.stats.channel.lower()
        ch_inv = inv_real.select(
            network=tr.stats.network,
            station=tr.stats.station,
            location=tr.stats.location,
            channel=tr.stats.channel,
        )[0][0][0]
        o_rel = float(ev_origin_utc - tr.stats.starttime)
        sac_tr = SACTrace.from_obspy_trace(tr.copy())
        sac_tr.stla = float(ch_inv.latitude)
        sac_tr.stlo = float(ch_inv.longitude)
        sac_tr.stel = float(ch_inv.elevation)
        sac_tr.stdp = float(ch_inv.depth)
        sac_tr.cmpaz = float(ch_inv.azimuth)
        sac_tr.cmpinc = float(ch_inv.dip + 90.0)  # SEED dip [-90=up, +90=down] -> SAC cmpinc [0=up, 90=horiz]
        sac_tr.evla = ev_info["evla"]
        sac_tr.evlo = ev_info["evlo"]
        sac_tr.evdp = ev_info["evdp_km"]
        sac_tr.mag = ev_info["mag"]
        sac_tr.o = o_rel
        sac_tr.ko = "ORIGIN"
        if p_arr is not None:
            sac_tr.t0 = o_rel + float(p_arr.time)
            sac_tr.kt0 = p_arr.name
        if s_arr is not None:
            sac_tr.t1 = o_rel + float(s_arr.time)
            sac_tr.kt1 = s_arr.name
        sac_tr.lcalda = True
        sac_out_p = real_dir / f"real_3c_{ch_code}.sac"
        sac_tr.write(str(sac_out_p), byteorder="little")

        # Also deconvolve velocity for ground-truth PGV verification
        tr_vel = tr.copy().remove_response(
            inventory=inv_real,
            output="VEL",
            pre_filt=[0.005, 0.01, 7.0, 9.5],
            water_level=60.0,
        )
        sac_files_meta.append({
            "channel": tr.stats.channel,
            "sac_file": f"real_event/{sac_out_p.name}",
            "sha256": sha256_file(sac_out_p),
            "npts": int(tr.stats.npts),
            "sampling_rate": to_py(tr.stats.sampling_rate),
            "starttime": to_py(tr.stats.starttime),
            "endtime": to_py(tr.stats.endtime),
            "raw_min": to_py(np.min(tr.data)),
            "raw_max": to_py(np.max(tr.data)),
            "raw_first_16": to_py(tr.data[:16].astype(np.float64)),
            "deconvolved_pgv_m_per_s": to_py(np.max(np.abs(tr_vel.data))),
        })

    write_json(
        real_dir / "real_event_expected.json",
        {
            "schema_version": "1.0",
            "source": real_source,
            "event": ev_info,
            "mseed_file": f"real_event/{real_mseed_path.name}",
            "mseed_sha256": sha256_file(real_mseed_path),
            "stationxml_file": f"real_event/{real_xml_path.name}",
            "stationxml_sha256": sha256_file(real_xml_path),
            "gcarc_deg": to_py(gcarc_deg),
            "theoretical_p_travel_time_s": to_py(p_arr.time) if p_arr else None,
            "theoretical_s_travel_time_s": to_py(s_arr.time) if s_arr else None,
            "channels": sac_files_meta,
        },
    )

    return {
        "inv_anmo": inv_anmo,
        "xml_path": xml_path,
        "sacpz_path": sacpz_path,
    }


# ============================================================================
# 2. GEODESICS FIXTURES (`fixtures/geodesics.json`)
# ============================================================================

def generate_geodesics_fixtures() -> None:
    pairs = [
        # Regional (4)
        ("regional_ridgecrest_pas", "regional", 35.7695, -117.5993, 34.1484, -118.1711),
        ("regional_tohoku_matsushiro", "regional", 38.2970, 142.3730, 36.5417, 138.2039),
        ("regional_alps_zurich_munich", "regional", 47.3769, 8.5417, 48.1351, 11.5820),
        ("regional_andes_santiago_mendoza", "regional", -33.4489, -70.6693, -32.8895, -68.8458),
        # Teleseismic (5)
        ("teleseismic_tohoku_anmo", "teleseismic", 38.2970, 142.3730, 34.94598, -106.45713),
        ("teleseismic_maule_bfo", "teleseismic", -35.8460, -72.7190, 48.3301, 8.3296),
        ("teleseismic_sumatra_hrv", "teleseismic", 3.2950, 95.9820, 42.5064, -71.5583),
        ("teleseismic_kuril_can", "teleseismic", 46.5920, 153.2660, -35.3187, 148.9963),
        ("teleseismic_alaska_cola_kip", "teleseismic", 64.8736, -147.8616, 21.4200, -158.0112),
        # Trans-equatorial (4)
        ("transeq_tokyo_sydney", "trans_equatorial", 35.6762, 139.6503, -33.8688, 151.2093),
        ("transeq_capetown_london_pm", "trans_equatorial", -33.9249, 18.4241, 51.5074, -0.1278),
        ("transeq_fiji_aleutians_dateline", "trans_equatorial", -17.7134, 178.0650, 51.8800, -176.6580),
        ("transeq_quito_singapore", "trans_equatorial", -0.1807, -78.4678, 1.3521, 103.8198),
        # Polar (4)
        ("polar_svalbard_alert", "polar", 78.2232, 15.6267, 82.5018, -62.3481),
        ("polar_cross_north_pole", "polar", 89.5000, 10.0000, 89.5000, -170.0000),
        ("polar_south_pole_qspa_cas", "polar", -89.9289, 144.4382, -66.2833, 110.5167),
        ("polar_exact_north_to_equator", "polar", 90.0000, 0.0000, 0.0000, 45.0000),
        # Near-coincident (4)
        ("coincident_exact_zero", "near_coincident", 34.94598, -106.45713, 34.94598, -106.45713),
        ("coincident_one_meter_north", "near_coincident", 34.94598000, -106.45713000, 34.94598900, -106.45713000),
        ("coincident_sub_centimeter", "near_coincident", 0.00000000, 0.00000000, 0.00000005, 0.00000005),
        ("coincident_borehole_array", "near_coincident", -23.6650, 133.9050, -23.6652, 133.9053),
        # Near-antipodal (4)
        ("antipodal_equatorial_near", "near_antipodal", 0.0000, 0.0000, 0.0010, 179.9990),
        ("antipodal_equatorial_classic_vincenty_trap", "near_antipodal", 0.0000, 0.0000, 0.0000, 179.7000),
        ("antipodal_midlat_madrid_wellington", "near_antipodal", 40.4168, -3.7038, -41.2865, 174.7762),
        ("antipodal_exact_antipode_offset", "near_antipodal", 35.0000, 140.0000, -34.9995, -39.9995),
    ]
    assert len(pairs) == 25

    cases = []
    for case_id, category, lat1, lon1, lat2, lon2 in pairs:
        dist_m, az, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
        dist_deg = locations2degrees(lat1, lon1, lat2, lon2)
        cases.append({
            "id": case_id,
            "category": category,
            "lat1_deg": to_py(lat1),
            "lon1_deg": to_py(lon1),
            "lat2_deg": to_py(lat2),
            "lon2_deg": to_py(lon2),
            "dist_m": to_py(dist_m),
            "az_deg": to_py(az),
            "baz_deg": to_py(baz),
            "spherical_dist_deg": to_py(dist_deg),
        })

    write_json(
        FIXTURES_DIR / "geodesics.json",
        {
            "schema_version": "1.0",
            "ellipsoid": {
                "name": "WGS84",
                "a_m": 6378137.0,
                "f": 1.0 / 298.257223563,
            },
            "num_pairs": len(cases),
            "pairs": cases,
        },
    )


# ============================================================================
# 3. DSP & SIGNAL FIXTURES (`fixtures/dsp.json`)
# ============================================================================

def ricker(t: np.ndarray, t0: float, f0: float) -> np.ndarray:
    u = math.pi * f0 * (t - t0)
    u2 = u * u
    return (1.0 - 2.0 * u2) * np.exp(-u2)


def generate_dsp_fixtures(inv_anmo: Any, sacpz_path: Path) -> None:
    rng = np.random.default_rng(2026111602)

    # 3A. Reference waveforms (100 Hz, 1000 samples = 10.0 s)
    df_100 = 100.0
    dt_100 = 1.0 / df_100
    npts_100 = 1000
    t = np.arange(npts_100, dtype=np.float64) * dt_100

    # Clean arrival trace: quiet pre-event microseism + sharp causal P-onset at t = 2.50 s (sample 250)
    # + secondary S-like wavepacket at t = 5.60 s
    bg_noise = 0.015 * rng.standard_normal(npts_100)
    p_env = np.where(t >= 2.50, (t - 2.50) * np.exp(-(t - 2.50) / 0.22), 0.0)
    p_pkt = 18.0 * p_env * np.sin(2.0 * np.pi * 6.0 * np.maximum(t - 2.50, 0.0))
    s_env = np.where(t >= 5.60, (t - 5.60) * np.exp(-(t - 5.60) / 0.45), 0.0)
    s_pkt = 9.5 * s_env * np.sin(2.0 * np.pi * 2.2 * np.maximum(t - 5.60, 0.0))
    broadband_tones = 0.35 * np.sin(2.0 * np.pi * 0.8 * t) + 0.20 * np.sin(2.0 * np.pi * 14.0 * t)
    clean_arrival = bg_noise + p_pkt + s_pkt + broadband_tones

    # Drifted trace for detrending tests (exact cubic baseline + clean_arrival)
    tn = (t - 5.0) / 5.0  # [-1, 1]
    drift = 3.25 - 1.40 * tn + 0.85 * (tn ** 2) - 0.60 * (tn ** 3)
    drifted_wave = clean_arrival + drift

    # Detrend outputs
    tr_demean = Trace(data=drifted_wave.copy(), header={"sampling_rate": df_100})
    tr_demean.detrend("demean")

    tr_linear = Trace(data=drifted_wave.copy(), header={"sampling_rate": df_100})
    tr_linear.detrend("linear")

    tr_poly3 = Trace(data=drifted_wave.copy(), header={"sampling_rate": df_100})
    tr_poly3.detrend("polynomial", order=3)

    detrend_fixtures = {
        "input": to_py(drifted_wave),
        "demean": to_py(tr_demean.data),
        "linear": to_py(tr_linear.data),
        "polynomial_order_3": to_py(tr_poly3.data),
    }

    # Taper outputs (both pure window weights and applied to clean_arrival)
    def run_taper(taper_type: str, max_pct: float, **kw: Any) -> dict[str, Any]:
        w_tr = Trace(data=np.ones(npts_100, dtype=np.float64), header={"sampling_rate": df_100})
        w_tr.taper(max_percentage=max_pct, type=taper_type, **kw)
        s_tr = Trace(data=clean_arrival.copy(), header={"sampling_rate": df_100})
        s_tr.taper(max_percentage=max_pct, type=taper_type, **kw)
        return {
            "type": taper_type,
            "max_percentage": max_pct,
            "extra_params": kw,
            "window": to_py(w_tr.data),
            "tapered": to_py(s_tr.data),
        }

    sac_cos_win = cosine_taper(npts_100, p=0.10, sactaper=True, halfcosine=False)
    taper_fixtures = {
        "hann_full": run_taper("hann", 0.5),
        "hann_0.05": run_taper("hann", 0.05),
        "cosine_0.05": run_taper("cosine", 0.05),
        "sac_cosine_0.05_each_side": {
            "description": "ObsPy cosine_taper(npts, p=0.10, sactaper=True, halfcosine=False) used by Trace.remove_response(taper_fraction=0.05)",
            "p_total": 0.10,
            "window": to_py(sac_cos_win),
            "tapered": to_py(clean_arrival * sac_cos_win),
        },
        "kaiser_full_beta_8.6": run_taper("kaiser", 0.5, beta=8.6),
        "kaiser_0.05_beta_8.6": run_taper("kaiser", 0.05, beta=8.6),
    }

    # 3B. Butterworth, Chebyshev-I, Bessel SOS filters (bp, lp, hp, bs; causal & zero-phase)
    # Note: Chebyshev-I requires passband ripple `rp` in dB (`rp=1.0`).
    # Bessel uses SciPy's standard `norm='phase'` (matching `obspy.signal.filter`).
    filter_families = [
        ("butter", {"ftype": "butter", "rp": None}),
        ("cheby1", {"ftype": "cheby1", "rp": 1.0}),
        ("bessel", {"ftype": "bessel", "rp": None}),
    ]
    filter_modes = [
        ("bandpass", "band", (0.5, 5.0)),
        ("lowpass", "lowpass", (5.0,)),
        ("highpass", "highpass", (0.5,)),
        ("bandstop", "bandstop", (0.5, 5.0)),
    ]
    corners = 4
    fe = 0.5 * df_100

    filters_out: dict[str, Any] = {}
    for fam_name, fam_cfg in filter_families:
        fam_dict: dict[str, Any] = {}
        ftype = fam_cfg["ftype"]
        rp = fam_cfg["rp"]
        for mode_name, scipy_btype, freqs_hz in filter_modes:
            wn = [f / fe for f in freqs_hz]
            wn_arg = wn[0] if len(wn) == 1 else wn
            sos = spsig.iirfilter(
                corners,
                wn_arg,
                rp=rp,
                btype=scipy_btype,
                ftype=ftype,
                output="sos",
            )
            if mode_name == "bandpass":
                y_causal = bandpass(clean_arrival, freqs_hz[0], freqs_hz[1], df_100, corners=corners, zerophase=False, ftype=ftype, rp=rp)
                y_zp_obspy = bandpass(clean_arrival, freqs_hz[0], freqs_hz[1], df_100, corners=corners, zerophase=True, ftype=ftype, rp=rp)
            elif mode_name == "lowpass":
                y_causal = lowpass(clean_arrival, freqs_hz[0], df_100, corners=corners, zerophase=False, ftype=ftype, rp=rp)
                y_zp_obspy = lowpass(clean_arrival, freqs_hz[0], df_100, corners=corners, zerophase=True, ftype=ftype, rp=rp)
            elif mode_name == "highpass":
                y_causal = highpass(clean_arrival, freqs_hz[0], df_100, corners=corners, zerophase=False, ftype=ftype, rp=rp)
                y_zp_obspy = highpass(clean_arrival, freqs_hz[0], df_100, corners=corners, zerophase=True, ftype=ftype, rp=rp)
            elif mode_name == "bandstop":
                y_causal = bandstop(clean_arrival, freqs_hz[0], freqs_hz[1], df_100, corners=corners, zerophase=False, ftype=ftype, rp=rp)
                y_zp_obspy = bandstop(clean_arrival, freqs_hz[0], freqs_hz[1], df_100, corners=corners, zerophase=True, ftype=ftype, rp=rp)
            else:
                raise RuntimeError(mode_name)

            y_zp_sosfiltfilt = spsig.sosfiltfilt(sos, clean_arrival)

            fam_dict[mode_name] = {
                "ftype": ftype,
                "btype": scipy_btype,
                "corners": corners,
                "rp_db": rp,
                "bessel_norm": "phase" if ftype == "bessel" else None,
                "freqs_hz": list(freqs_hz),
                "sampling_rate_hz": df_100,
                "sos_sections": to_py(sos),
                "zerophase_false": to_py(y_causal),
                "zerophase_true": to_py(y_zp_obspy),
                "zerophase_sosfiltfilt": to_py(y_zp_sosfiltfilt),
            }
        filters_out[fam_name] = fam_dict

    # 3C. Instrument Response Deconvolution (`remove_response` against StationXML & SACPZ)
    cha_anmo = inv_anmo[0][0][0]
    df_resp = float(cha_anmo.sample_rate)  # 40.0 Hz -> Nyquist 20.0 Hz
    npts_resp = 4096  # 102.4 s -> nfft = 8192 -> df_bin = 0.0048828125 Hz < 0.005 Hz!
    t_resp = np.arange(npts_resp, dtype=np.float64) / df_resp
    raw_counts = (
        25000.0 * ricker(t_resp, 35.0, 0.25)
        + 12000.0 * ricker(t_resp, 62.0, 1.20)
        + 4000.0 * np.sin(2.0 * np.pi * 0.08 * t_resp) * np.hanning(npts_resp)
    )
    pre_filt = [0.005, 0.01, 15.0, 20.0]
    water_level = 60.0

    resp_trace_hdr = {
        "network": inv_anmo[0].code,
        "station": inv_anmo[0][0].code,
        "location": cha_anmo.location_code,
        "channel": cha_anmo.code,
        "sampling_rate": df_resp,
        "starttime": cha_anmo.start_date + 3600.0,
    }

    stationxml_deconv = {}
    for out_unit in ["VEL", "DISP", "ACC"]:
        tr_tmp = Trace(data=raw_counts.copy(), header=copy.deepcopy(resp_trace_hdr))
        tr_tmp.remove_response(
            inventory=inv_anmo,
            output=out_unit,
            pre_filt=pre_filt,
            water_level=water_level,
            zero_mean=True,
            taper=True,
            taper_fraction=0.05,
        )
        stationxml_deconv[out_unit] = to_py(tr_tmp.data)

    # SACPZ deconvolution via `simulate_seismometer(..., sacsim=True, pitsasim=False)`
    # (exact algorithmic equivalent of `Trace.remove_response` using the SACPZ transfer function)
    tr_paz_base = Trace()
    attach_paz(tr_paz_base, str(sacpz_path))
    paz_disp = tr_paz_base.stats.paz

    def make_paz_for_unit(unit: str) -> dict[str, Any]:
        zs = list(paz_disp.zeros)
        if unit == "VEL":
            zs.remove(0j)
        elif unit == "ACC":
            zs.remove(0j)
            zs.remove(0j)
        elif unit != "DISP":
            raise ValueError(unit)
        return {
            "poles": list(paz_disp.poles),
            "zeros": zs,
            "gain": float(paz_disp.gain),
            "sensitivity": 1.0,
        }

    sacpz_deconv = {}
    for out_unit in ["VEL", "DISP", "ACC"]:
        paz_u = make_paz_for_unit(out_unit)
        deconv_u = simulate_seismometer(
            raw_counts.copy(),
            df_resp,
            paz_remove=paz_u,
            paz_simulate=None,
            remove_sensitivity=True,
            simulate_sensitivity=False,
            water_level=water_level,
            zero_mean=True,
            taper=True,
            taper_fraction=0.05,
            pre_filt=tuple(pre_filt),
            nfft_pow2=False,
            pitsasim=False,
            sacsim=True,
        )
        sacpz_deconv[out_unit] = to_py(deconv_u)

    remove_response_fixtures = {
        "stationxml_file": "io/iu_anmo_bhz.xml",
        "sacpz_file": "io/iu_anmo_bhz.sacpz",
        "sampling_rate_hz": df_resp,
        "npts": npts_resp,
        "nfft": int(_npts2nfft(npts_resp)),
        "pre_filt_hz": pre_filt,
        "water_level_db": water_level,
        "zero_mean": True,
        "taper": True,
        "taper_fraction": 0.05,
        "input_counts": to_py(raw_counts),
        "stationxml_outputs": stationxml_deconv,
        "sacpz_outputs": sacpz_deconv,
    }

    # 3D. Hilbert envelope, normalized cross-correlation, STA/LTA, Baer picker
    analytic = spsig.hilbert(clean_arrival)
    env = envelope(clean_arrival)
    hilbert_fixtures = {
        "envelope": to_py(env),
        "quadrature_imag": to_py(np.imag(analytic)),
        "instantaneous_phase_rad": to_py(np.angle(analytic)),
    }

    # Normalized cross-correlation: shift by +17 samples + scale + tiny noise
    true_shift_samples = 17
    sig_a = ricker(t, 4.0, 3.5) + 0.45 * ricker(t, 5.2, 6.0)
    sig_b = np.roll(sig_a, true_shift_samples) * 1.75 + 0.005 * rng.standard_normal(npts_100)
    sig_b[:true_shift_samples] = 0.0
    max_shift = 100
    cc_func = correlate(sig_a, sig_b, shift=max_shift, demean=True, normalize="naive")
    cc_shift, cc_val = xcorr_max(cc_func, abs_max=False)
    xcorr_fixtures = {
        "signal_a": to_py(sig_a),
        "signal_b": to_py(sig_b),
        "max_shift_samples": max_shift,
        "demean": True,
        "normalize": "naive",
        "expected_shift_samples": int(cc_shift),
        "expected_max_value": to_py(cc_val),
        "cc_curve": to_py(cc_func),
    }

    # Classic & Recursive STA/LTA
    nsta, nlta = 50, 200
    cft_classic = classic_sta_lta(clean_arrival, nsta=nsta, nlta=nlta)
    cft_recursive = recursive_sta_lta(clean_arrival, nsta=nsta, nlta=nlta)
    on_off_classic = trigger_onset(cft_classic, 3.5, 1.0)
    on_off_recursive = trigger_onset(cft_recursive, 3.5, 1.0)
    sta_lta_fixtures = {
        "nsta": nsta,
        "nlta": nlta,
        "thr_on": 3.5,
        "thr_off": 1.0,
        "classic_cft": to_py(cft_classic),
        "classic_triggers": to_py(on_off_classic),
        "recursive_cft": to_py(cft_recursive),
        "recursive_triggers": to_py(on_off_recursive),
    }

    # Baer picker (`pk_baer`) on ObsPy's canonical real local earthquake P arrival (BW.RJOB..EHZ, first 1000 samples)
    rjob_ehz = read()[0].data[:1000].astype(np.float32)
    baer_params = {
        "samp_int": 100,
        "tdownmax": 20,
        "tupevent": 60,
        "thr1": 7.0,
        "thr2": 12.0,
        "preset_len": 100,
        "p_dur": 100,
    }
    pptime_rjob, pfm_rjob, cf_rjob = pk_baer(
        rjob_ehz,
        baer_params["samp_int"],
        baer_params["tdownmax"],
        baer_params["tupevent"],
        baer_params["thr1"],
        baer_params["thr2"],
        baer_params["preset_len"],
        baer_params["p_dur"],
        return_cf=True,
    )
    pptime_syn, pfm_syn, cf_syn = pk_baer(
        clean_arrival.astype(np.float32),
        baer_params["samp_int"],
        baer_params["tdownmax"],
        baer_params["tupevent"],
        baer_params["thr1"],
        baer_params["thr2"],
        baer_params["preset_len"],
        baer_params["p_dur"],
        return_cf=True,
    )
    baer_fixtures = {
        "params": baer_params,
        "rjob_ehz_1000": {
            "input_f32": to_py(rjob_ehz.astype(np.float64)),
            "pick_sample": int(pptime_rjob),
            "first_motion": pfm_rjob.strip(),
            "characteristic_function": to_py(cf_rjob.astype(np.float64)),
        },
        "clean_arrival_100hz": {
            "pick_sample": int(pptime_syn),
            "first_motion": pfm_syn.strip(),
            "characteristic_function": to_py(cf_syn.astype(np.float64)),
        },
    }

    # 3E. 3C Rotations (`rotate_ne_rt`, `rotate_zne_lqt`) & Polarization (`flinn`, `polarization_analysis`)
    baz_deg = 215.0  # event SW of station -> wave propagates toward az = 35.0 deg
    inc_deg = 28.0
    prop_az_rad = math.radians((baz_deg + 180.0) % 360.0)
    inc_rad = math.radians(inc_deg)

    # Pure rectilinear P wave in [2.0, 4.0] s + transverse SH wave in [5.5, 7.5] s + tiny 3C noise
    p_wave_1d = 10.0 * ricker(t, 3.0, 3.0)
    sh_wave_1d = 6.0 * ricker(t, 6.5, 1.8)
    z_3c = p_wave_1d * math.cos(inc_rad) + 0.01 * rng.standard_normal(npts_100)
    r_true = p_wave_1d * math.sin(inc_rad)
    t_true = sh_wave_1d
    # Since R points along propagation azimuth `prop_az` = baz + 180, N = R*cos(prop_az) - T*sin(prop_az)
    n_3c = r_true * math.cos(prop_az_rad) - t_true * math.sin(prop_az_rad) + 0.01 * rng.standard_normal(npts_100)
    e_3c = r_true * math.sin(prop_az_rad) + t_true * math.cos(prop_az_rad) + 0.01 * rng.standard_normal(npts_100)

    rot_r, rot_t = rotate_ne_rt(n_3c.copy(), e_3c.copy(), baz_deg)
    rot_l, rot_q, rot_lqt_t = rotate_zne_lqt(z_3c.copy(), n_3c.copy(), e_3c.copy(), baz_deg, inc_deg)

    # Flinn eigenstructure polarization on the P-wave window [250:350]
    p_slice = slice(250, 350)
    flinn_az, flinn_inc, flinn_rect, flinn_plan = flinn([z_3c[p_slice], n_3c[p_slice], e_3c[p_slice]], noise_thres=0.0)

    # Sliding-window `polarization_analysis` (`method='flinn'`)
    t0_pol = UTCDateTime("2025-01-01T00:00:00Z")
    st_3c = Stream([
        Trace(data=z_3c.copy(), header={"channel": "BHZ", "sampling_rate": df_100, "starttime": t0_pol}),
        Trace(data=n_3c.copy(), header={"channel": "BHN", "sampling_rate": df_100, "starttime": t0_pol}),
        Trace(data=e_3c.copy(), header={"channel": "BHE", "sampling_rate": df_100, "starttime": t0_pol}),
    ])
    pol_res = polarization_analysis(
        st_3c,
        win_len=1.0,
        win_frac=0.25,
        frqlow=0.5,
        frqhigh=10.0,
        stime=t0_pol + 0.5,
        etime=t0_pol + 9.0,
        method="flinn",
        var_noise=0.0,
    )

    rotation_polarization_fixtures = {
        "back_azimuth_deg": baz_deg,
        "incidence_deg": inc_deg,
        "input_z": to_py(z_3c),
        "input_n": to_py(n_3c),
        "input_e": to_py(e_3c),
        "rotate_ne_rt": {
            "radial": to_py(rot_r),
            "transverse": to_py(rot_t),
        },
        "rotate_zne_lqt": {
            "l": to_py(rot_l),
            "q": to_py(rot_q),
            "t": to_py(rot_lqt_t),
        },
        "flinn_p_window": {
            "start_sample": 250,
            "end_sample_exclusive": 350,
            "noise_thres": 0.0,
            "azimuth_deg": to_py(flinn_az),
            "incidence_deg": to_py(flinn_inc),
            "rectilinearity": to_py(flinn_rect),
            "planarity": to_py(flinn_plan),
        },
        "polarization_analysis_flinn": {
            "win_len_s": 1.0,
            "win_frac": 0.25,
            "frqlow_hz": 0.5,
            "frqhigh_hz": 10.0,
            "timestamps_epoch_s": to_py(pol_res["timestamp"]),
            "azimuth_deg": to_py(pol_res["azimuth"]),
            "incidence_deg": to_py(pol_res["incidence"]),
            "rectilinearity": to_py(pol_res["rectilinearity"]),
            "planarity": to_py(pol_res["planarity"]),
        },
    }

    # 3F. Peterson (1993) NLNM & NHNM curves
    nlnm_per, nlnm_db = get_nlnm()
    nhnm_per, nhnm_db = get_nhnm()
    noise_models_fixtures = {
        "units": "dB rel. 1 (m/s^2)^2/Hz",
        "nlnm": {
            "periods_s": to_py(nlnm_per),
            "power_db": to_py(nlnm_db),
        },
        "nhnm": {
            "periods_s": to_py(nhnm_per),
            "power_db": to_py(nhnm_db),
        },
    }

    # 3G. Lanczos Resampling (`a = 20`)
    # Bandlimited smooth signal at 100 Hz (500 samples = 5.0 s) resampled to 40 Hz (200 samples)
    # and 40 Hz (200 samples) upsampled to 100 Hz (498 samples inside support)
    t_5s = np.arange(500, dtype=np.float64) * 0.01
    bandlim_100 = (
        np.sin(2.0 * np.pi * 1.5 * t_5s)
        + 0.6 * np.cos(2.0 * np.pi * 4.2 * t_5s + 0.4)
        + 0.8 * ricker(t_5s, 2.5, 3.0)
    )
    down_40 = lanczos_interpolation(
        bandlim_100.copy(),
        old_start=0.0,
        old_dt=0.01,
        new_start=0.0,
        new_dt=0.025,
        new_npts=200,
        a=20,
        window="lanczos",
    )
    up_100 = lanczos_interpolation(
        down_40.copy(),
        old_start=0.0,
        old_dt=0.025,
        new_start=0.0,
        new_dt=0.01,
        new_npts=496,
        a=20,
        window="lanczos",
    )
    lanczos_fixtures = {
        "a": 20,
        "window": "lanczos",
        "downsample_100hz_to_40hz": {
            "old_start": 0.0,
            "old_dt": 0.01,
            "new_start": 0.0,
            "new_dt": 0.025,
            "new_npts": 200,
            "input": to_py(bandlim_100),
            "output": to_py(down_40),
        },
        "upsample_40hz_to_100hz": {
            "old_start": 0.0,
            "old_dt": 0.025,
            "new_start": 0.0,
            "new_dt": 0.01,
            "new_npts": 496,
            "input": to_py(down_40),
            "output": to_py(up_100),
        },
    }

    write_json(
        FIXTURES_DIR / "dsp.json",
        {
            "schema_version": "1.0",
            "versions": {
                "obspy": obspy.__version__,
                "scipy": scipy.__version__,
                "numpy": np.__version__,
            },
            "reference_waveforms": {
                "sampling_rate_hz": df_100,
                "npts": npts_100,
                "clean_arrival_100hz": to_py(clean_arrival),
                "drifted_100hz": to_py(drifted_wave),
            },
            "detrend": detrend_fixtures,
            "tapers": taper_fixtures,
            "filters": filters_out,
            "remove_response": remove_response_fixtures,
            "hilbert": hilbert_fixtures,
            "cross_correlation": xcorr_fixtures,
            "sta_lta": sta_lta_fixtures,
            "baer_picker": baer_fixtures,
            "rotation_and_polarization": rotation_polarization_fixtures,
            "peterson_noise_models": noise_models_fixtures,
            "lanczos_resampling": lanczos_fixtures,
        },
    )


# ============================================================================
# 4. TAUP FIXTURES (`fixtures/taup.json`)
# ============================================================================

def generate_taup_fixtures() -> None:
    model_names = ["iasp91", "ak135", "prem"]
    depths_km = [0.0, 15.0, 35.0, 120.0, 450.0, 620.0]
    distances_deg = [0.5, 5.0, 18.0, 35.0, 65.0, 85.0, 105.0, 130.0, 145.0, 155.0, 175.0]
    phases = [
        "p", "s", "P", "S", "Pn", "Sn", "Pg", "Sg",
        "pP", "sP", "PP", "SS", "PcP", "ScS",
        "PKP", "PKIKP", "PKiKP", "SKS", "SKKS",
        "Pdiff", "Sdiff", "P410s", "P660s", "3.5kmps",
    ]

    models = {m: TauPyModel(model=m) for m in model_names}

    travel_time_grid = []
    total_arrivals = 0
    for m_name in model_names:
        model = models[m_name]
        for depth in depths_km:
            for dist in distances_deg:
                arrs = model.get_travel_times(
                    source_depth_in_km=depth,
                    distance_in_degree=dist,
                    phase_list=phases,
                )
                arr_list = []
                for a in arrs:
                    arr_list.append({
                        "name": a.name,
                        "purist_name": a.purist_name,
                        "time_s": to_py(a.time),
                        "ray_param_s_per_rad": to_py(a.ray_param),
                        "ray_param_s_per_deg": to_py(a.ray_param_sec_degree),
                        "takeoff_angle_deg": to_py(a.takeoff_angle),
                        "incident_angle_deg": to_py(a.incident_angle),
                        "distance_deg": to_py(a.distance),
                        "purist_distance_deg": to_py(a.purist_distance),
                    })
                total_arrivals += len(arr_list)
                travel_time_grid.append({
                    "model": m_name,
                    "source_depth_km": depth,
                    "distance_deg": dist,
                    "arrivals": arr_list,
                })

    # Sample `get_pierce_points` across crustal, mantle, converted, core-reflected, and core-transmitted phases
    pierce_specs = [
        ("iasp91", 15.0, 5.0, ["Pg", "Pn", "Sg", "Sn"]),
        ("iasp91", 35.0, 65.0, ["P", "S", "pP", "PcP", "P410s", "P660s"]),
        ("ak135", 120.0, 105.0, ["Pdiff", "Sdiff", "PP", "SKS", "PKiKP"]),
        ("prem", 450.0, 145.0, ["PKP", "PKIKP", "PKiKP", "SKS", "SKKS"]),
    ]
    pierce_fixtures = []
    for m_name, depth, dist, ph_list in pierce_specs:
        arrs = models[m_name].get_pierce_points(
            source_depth_in_km=depth,
            distance_in_degree=dist,
            phase_list=ph_list,
        )
        entries = []
        for a in arrs:
            pts = [
                {
                    "p_s_per_rad": to_py(row["p"]),
                    "time_s": to_py(row["time"]),
                    "dist_rad": to_py(row["dist"]),
                    "dist_deg": to_py(math.degrees(float(row["dist"]))),
                    "depth_km": to_py(row["depth"]),
                }
                for row in a.pierce
            ]
            entries.append({
                "name": a.name,
                "purist_name": a.purist_name,
                "time_s": to_py(a.time),
                "ray_param_s_per_rad": to_py(a.ray_param),
                "pierce_points": pts,
            })
        pierce_fixtures.append({
            "model": m_name,
            "source_depth_km": depth,
            "distance_deg": dist,
            "phases_requested": ph_list,
            "arrivals": entries,
        })

    # Sample `get_ray_paths` full trajectories `(r/depth, theta/dist, time, p)`
    path_specs = [
        ("iasp91", 35.0, 65.0, ["P", "S", "pP", "PcP", "P410s"]),
        ("ak135", 35.0, 105.0, ["Pdiff", "SKS"]),
        ("prem", 120.0, 155.0, ["PKP", "PKIKP", "PKiKP"]),
    ]
    ray_path_fixtures = []
    for m_name, depth, dist, ph_list in path_specs:
        arrs = models[m_name].get_ray_paths(
            source_depth_in_km=depth,
            distance_in_degree=dist,
            phase_list=ph_list,
        )
        entries = []
        for a in arrs:
            path_arr = a.path
            entries.append({
                "name": a.name,
                "purist_name": a.purist_name,
                "time_s": to_py(a.time),
                "ray_param_s_per_rad": to_py(a.ray_param),
                "num_points": int(len(path_arr)),
                "p_s_per_rad": to_py(path_arr["p"]),
                "time_points_s": to_py(path_arr["time"]),
                "dist_points_rad": to_py(path_arr["dist"]),
                "dist_points_deg": to_py(np.degrees(path_arr["dist"])),
                "depth_points_km": to_py(path_arr["depth"]),
            })
        ray_path_fixtures.append({
            "model": m_name,
            "source_depth_km": depth,
            "distance_deg": dist,
            "phases_requested": ph_list,
            "arrivals": entries,
        })

    write_json(
        FIXTURES_DIR / "taup.json",
        {
            "schema_version": "1.0",
            "obspy_version": obspy.__version__,
            "models": model_names,
            "source_depths_km": depths_km,
            "distances_deg": distances_deg,
            "phases": phases,
            "num_grid_queries": len(travel_time_grid),
            "total_arrivals": total_arrivals,
            "travel_times": travel_time_grid,
            "pierce_examples": pierce_fixtures,
            "ray_path_examples": ray_path_fixtures,
        },
    )


def main() -> None:
    print("[1/4] Generating I/O fixtures (SAC, miniSEED v2, StationXML, SACPZ, QuakeML, real 3C event)...")
    io_ctx = generate_io_fixtures()
    print("[2/4] Generating WGS84 Geodesics fixtures (25 coordinate pairs)...")
    generate_geodesics_fixtures()
    print("[3/4] Generating DSP & Signal fixtures (detrend, tapers, filters, response, pickers, 3C, NLNM/NHNM, Lanczos)...")
    generate_dsp_fixtures(io_ctx["inv_anmo"], io_ctx["sacpz_path"])
    print("[4/4] Generating TauP fixtures (iasp91, ak135, prem travel times, pierce points, ray paths)...")
    generate_taup_fixtures()
    print("All golden fixtures successfully generated in:", FIXTURES_DIR)


if __name__ == "__main__":
    main()
