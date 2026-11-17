"""EarthSnail zero-bloat Python bindings powered by the Rust `earthsnail_wasm` C-ABI engine."""

from __future__ import annotations

import pathlib
from typing import Iterable, Optional, Sequence, Union

from ._ffi import (
    ES_ABI_VERSION,
    EarthSnailFFIError,
    EarthSnailLib,
    GeodesicResult,
    PickResult,
    TauPArrival,
    TauPRayPath,
    WaveformTrace,
    find_native_library,
)

__version__ = "0.1.0"
__all__ = [
    "ES_ABI_VERSION",
    "EarthSnailFFIError",
    "EarthSnailLib",
    "GeodesicResult",
    "PickResult",
    "TauPArrival",
    "TauPRayPath",
    "WaveformTrace",
    "encode_mseed2_steim2",
    "encode_sac",
    "filter_trace",
    "find_native_library",
    "get_default_lib",
    "pick_morse",
    "read_waveforms",
    "taup_ray_paths",
    "taup_travel_times",
    "wgs84_geodesic",
]

_DEFAULT_LIB: Optional[EarthSnailLib] = None


def get_default_lib(lib_path: Optional[Union[str, pathlib.Path]] = None) -> EarthSnailLib:
    global _DEFAULT_LIB
    if lib_path is not None:
        return EarthSnailLib(lib_path)
    if _DEFAULT_LIB is None:
        _DEFAULT_LIB = EarthSnailLib()
    return _DEFAULT_LIB


def read_waveforms(source: Union[bytes, bytearray, str, pathlib.Path]) -> list[WaveformTrace]:
    """Auto-detect and decode SAC (LE/BE/Alpha), miniSEED v2/v3 (Steim-1/2/Int/Float), or GeoCSV."""
    return get_default_lib().read_waveforms(source)


def encode_mseed2_steim2(
    samples: Sequence[float],
    sampling_rate_hz: float,
    *,
    channel: str = "XX.STA.00.BHZ",
    start_epoch_s: int = 1_700_000_000,
    start_subsec_ns: int = 0,
    record_len: int = 512,
) -> bytes:
    """Losslessly compress integer-valued waveform samples into SEED v2.4 Steim-2 records."""
    return get_default_lib().encode_mseed2_steim2(
        samples,
        sampling_rate_hz,
        channel=channel,
        start_epoch_s=start_epoch_s,
        start_subsec_ns=start_subsec_ns,
        record_len=record_len,
    )


def encode_sac(
    samples: Sequence[float],
    sampling_rate_hz: float,
    *,
    channel: str = "XX.STA.00.BHZ",
    start_epoch_s: int = 1_700_000_000,
    start_subsec_ns: int = 0,
    big_endian: bool = False,
) -> bytes:
    """Encode a waveform trace into SAC binary (little- or big-endian)."""
    return get_default_lib().encode_sac(
        samples,
        sampling_rate_hz,
        channel=channel,
        start_epoch_s=start_epoch_s,
        start_subsec_ns=start_subsec_ns,
        big_endian=big_endian,
    )


def filter_trace(
    samples: Sequence[float],
    sampling_rate_hz: float,
    f_low_hz: float,
    f_high_hz: float = 0.0,
    *,
    order: int = 4,
    family: str = "butter",
    band: str = "bandpass",
    zero_phase: Union[bool, str] = True,
    ripple_db: float = 1.0,
):
    """Apply a cascaded SOS Butterworth/Chebyshev/Bessel filter (causal or zero-phase)."""
    return get_default_lib().filter_trace(
        samples,
        sampling_rate_hz,
        f_low_hz,
        f_high_hz,
        order=order,
        family=family,
        band=band,
        zero_phase=zero_phase,
        ripple_db=ripple_db,
    )


def pick_morse(
    samples: Sequence[float],
    sampling_rate_hz: float = 100.0,
    *,
    nsta: int = 0,
    nlta: int = 0,
    thr_on: float = 0.0,
    return_cf: bool = False,
) -> PickResult:
    """Run the 3-Point Discrete Morse Causal-Deferral glitch-immune onset picker."""
    return get_default_lib().pick_morse(
        samples,
        sampling_rate_hz,
        nsta=nsta,
        nlta=nlta,
        thr_on=thr_on,
        return_cf=return_cf,
    )


def taup_travel_times(
    depth_km: float,
    distance_deg: float,
    phases: Iterable[str] = ("P", "pP", "PP", "PcP", "S", "SKS", "PKIKP"),
    *,
    model: str = "iasp91",
) -> list[TauPArrival]:
    """Compute global spherical Earth travel times across `iasp91`, `ak135`, or `prem`."""
    return get_default_lib().taup_travel_times(depth_km, distance_deg, phases, model=model)


def taup_ray_paths(
    depth_km: float,
    distance_deg: float,
    phases: Iterable[str] = ("P", "S", "PcP", "PKIKP"),
    *,
    model: str = "iasp91",
) -> list[TauPRayPath]:
    """Compute spherical Earth ray-path trajectories across `iasp91`, `ak135`, or `prem`."""
    return get_default_lib().taup_ray_paths(depth_km, distance_deg, phases, model=model)


def wgs84_geodesic(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> GeodesicResult:
    """Solve the exact Karney WGS84 ellipsoidal inverse geodesic."""
    return get_default_lib().wgs84_geodesic(lat1_deg, lon1_deg, lat2_deg, lon2_deg)
