"""Low-level zero-dependency ctypes loader and wire-frame unpackers for earthsnail_wasm."""

from __future__ import annotations

import array
import ctypes
import os
import pathlib
import struct
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union


ES_OK = 0
ES_ABI_VERSION = 100


class EarthSnailFFIError(RuntimeError):
    """Raised when an EarthSnail C-ABI call returns a non-zero error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"EarthSnail FFI error ({code}): {message}")
        self.code = code
        self.ffi_message = message


@dataclass(frozen=True)
class GeodesicResult:
    distance_m: float
    gcarc_deg: float
    distance_deg_spherical: float
    azimuth_deg: float
    back_azimuth_deg: float


@dataclass
class WaveformTrace:
    network: str
    station: str
    location: str
    channel: str
    sampling_rate_hz: float
    start_epoch_s: int
    start_subsec_ns: int
    calib: float
    samples: "array.array[float]"
    sac_meta: dict[str, float]
    format_code: int = 0

    @property
    def seed_id(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.channel}"

    @property
    def npts(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class TauPArrival:
    name: str
    purist_name: str
    time_s: float
    ray_param_s_per_rad: float
    ray_param_s_per_deg: float
    takeoff_angle_deg: float
    incident_angle_deg: float
    distance_deg: float
    purist_distance_deg: float
    source_depth_km: float


@dataclass
class TauPRayPath:
    arrival: TauPArrival
    points: list[tuple[float, float, float, float]]  # (dist_deg, depth_km, radius_km, time_s)
    planet_radius_km: float
    moho_depth_km: float
    cmb_depth_km: float
    iocb_depth_km: float


@dataclass
class PickResult:
    pick_sample: Optional[int]
    peak_metric: float
    characteristic_function: Optional["array.array[float]"] = None


def _platform_lib_names() -> list[str]:
    if sys.platform == "win32":
        return ["earthsnail_wasm.dll"]
    if sys.platform == "darwin":
        return ["libearthsnail_wasm.dylib", "earthsnail_wasm.dylib"]
    return ["libearthsnail_wasm.so", "earthsnail_wasm.so"]


def find_native_library(explicit_path: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    if explicit_path is not None:
        p = pathlib.Path(explicit_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit EarthSnail library not found: {p}")

    env_path = os.environ.get("EARTHSNAIL_LIB_PATH")
    if env_path:
        p = pathlib.Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"EARTHSNAIL_LIB_PATH does not point to a file: {p}")

    pkg_dir = pathlib.Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent
    names = _platform_lib_names()

    search_dirs: list[pathlib.Path] = [
        pkg_dir,
        repo_root / "target" / "release",
        repo_root / "target_workers" / "studio_wasm" / "release",
    ]
    target_workers = repo_root / "target_workers"
    if target_workers.is_dir():
        for child in sorted(target_workers.iterdir()):
            rel = child / "release"
            if rel.is_dir() and rel not in search_dirs:
                search_dirs.append(rel)
            dbg = child / "debug"
            if dbg.is_dir():
                search_dirs.append(dbg)
    search_dirs.append(repo_root / "target" / "debug")

    for d in search_dirs:
        for name in names:
            cand = d / name
            if cand.is_file():
                return cand

    raise FileNotFoundError(
        f"Could not locate {names[0]}. Build with `cargo build -p earthsnail-wasm --release` "
        f"or set EARTHSNAIL_LIB_PATH."
    )


def _decode_cstr_padded(raw: bytes) -> str:
    nul = raw.find(b"\x00")
    if nul >= 0:
        raw = raw[:nul]
    return raw.decode("utf-8", errors="replace").strip()


def unpack_eswf(buf: bytes) -> list[WaveformTrace]:
    if len(buf) < 16 or buf[:4] != b"ESWF":
        raise ValueError("Invalid ESWF wire header")
    version, trace_count, fmt_code = struct.unpack_from("<III", buf, 4)
    if version != 1:
        raise ValueError(f"Unsupported ESWF wire version: {version}")

    offset = 16
    traces: list[WaveformTrace] = []
    meta_keys = (
        "stla",
        "stlo",
        "stel",
        "stdp",
        "evla",
        "evlo",
        "evdp",
        "mag",
        "dist",
        "az",
        "baz",
        "gcarc",
        "cmpaz",
        "cmpinc",
    )

    for _ in range(trace_count):
        # Exact unpack of first 40 bytes: u64, f64, i64, u32, u32, f64
        npts, fs, epoch_s, subsec_ns, _res, calib = struct.unpack_from("<QdqIId", buf, offset)
        sac_vals = struct.unpack_from("<14d", buf, offset + 40)
        sac_meta = {k: v for k, v in zip(meta_keys, sac_vals)}
        net = _decode_cstr_padded(buf[offset + 152 : offset + 160])
        sta = _decode_cstr_padded(buf[offset + 160 : offset + 168])
        loc = _decode_cstr_padded(buf[offset + 168 : offset + 176])
        cha = _decode_cstr_padded(buf[offset + 176 : offset + 184])
        origin_off = struct.unpack_from("<d", buf, offset + 184)[0]
        sac_meta["origin_offset_s"] = origin_off

        samp_start = offset + 192
        samp_end = samp_start + npts * 8
        samples = array.array("d")
        if sys.byteorder == "little":
            samples.frombytes(buf[samp_start:samp_end])
        else:
            samples.fromlist(list(struct.unpack_from(f"<{npts}d", buf, samp_start)))
        offset = samp_end

        traces.append(
            WaveformTrace(
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                sampling_rate_hz=fs,
                start_epoch_s=epoch_s,
                start_subsec_ns=subsec_ns,
                calib=calib,
                samples=samples,
                sac_meta=sac_meta,
                format_code=fmt_code,
            )
        )
    return traces


def unpack_estt(buf: bytes) -> list[TauPArrival]:
    if len(buf) < 16 or buf[:4] != b"ESTT":
        raise ValueError("Invalid ESTT wire header")
    version, count, _ = struct.unpack_from("<III", buf, 4)
    if version != 1:
        raise ValueError(f"Unsupported ESTT wire version: {version}")

    arrivals: list[TauPArrival] = []
    offset = 16
    for _ in range(count):
        (
            time_s,
            p_rad,
            p_deg,
            takeoff,
            incident,
            dist_deg,
            purist_dist,
            depth_km,
        ) = struct.unpack_from("<8d", buf, offset)
        name = _decode_cstr_padded(buf[offset + 64 : offset + 80])
        purist_name = _decode_cstr_padded(buf[offset + 80 : offset + 96])
        offset += 96
        arrivals.append(
            TauPArrival(
                name=name,
                purist_name=purist_name,
                time_s=time_s,
                ray_param_s_per_rad=p_rad,
                ray_param_s_per_deg=p_deg,
                takeoff_angle_deg=takeoff,
                incident_angle_deg=incident,
                distance_deg=dist_deg,
                purist_distance_deg=purist_dist,
                source_depth_km=depth_km,
            )
        )
    return arrivals


def unpack_esrp(buf: bytes, requested_dist_deg: float) -> list[TauPRayPath]:
    if len(buf) < 48 or buf[:4] not in (b"ESRP", b"ESPP"):
        raise ValueError("Invalid ESRP/ESPP wire header")
    version, count, _ = struct.unpack_from("<III", buf, 4)
    if version != 1:
        raise ValueError(f"Unsupported ESRP wire version: {version}")
    r_planet, moho, cmb, iocb = struct.unpack_from("<4d", buf, 16)

    offset = 48
    paths: list[TauPRayPath] = []
    for _ in range(count):
        time_s, p_rad, p_deg, takeoff, incident, purist_dist, n_pts = struct.unpack_from(
            "<6dQ", buf, offset
        )
        name = _decode_cstr_padded(buf[offset + 56 : offset + 72])
        depth_km = struct.unpack_from("<d", buf, offset + 72)[0]
        offset += 80
        pts: list[tuple[float, float, float, float]] = []
        for _ in range(n_pts):
            pts.append(struct.unpack_from("<4d", buf, offset))
            offset += 32
        arr = TauPArrival(
            name=name,
            purist_name=name,
            time_s=time_s,
            ray_param_s_per_rad=p_rad,
            ray_param_s_per_deg=p_deg,
            takeoff_angle_deg=takeoff,
            incident_angle_deg=incident,
            distance_deg=requested_dist_deg,
            purist_distance_deg=purist_dist,
            source_depth_km=depth_km,
        )
        paths.append(
            TauPRayPath(
                arrival=arr,
                points=pts,
                planet_radius_km=r_planet,
                moho_depth_km=moho,
                cmb_depth_km=cmb,
                iocb_depth_km=iocb,
            )
        )
    return paths


def _to_c_double_array(samples: Sequence[float]) -> tuple[ctypes.Array[ctypes.c_double], int]:
    n = len(samples)
    if isinstance(samples, array.array) and samples.typecode == "d":
        buf_cls = ctypes.c_double * n
        addr, _ = samples.buffer_info()
        # Copy into ctypes buffer for deterministic lifetime independent of Python GC moves:
        arr = buf_cls()
        ctypes.memmove(arr, addr, n * 8)
        return arr, n
    arr_cls = ctypes.c_double * n
    return arr_cls(*samples), n


class EarthSnailLib:
    """Bound ctypes wrapper around the compiled `earthsnail_wasm` native shared library."""

    def __init__(self, lib_path: Optional[Union[str, pathlib.Path]] = None) -> None:
        self.path = find_native_library(lib_path)
        self._dll = ctypes.CDLL(str(self.path))
        self._bind_signatures()
        abi = self._dll.es_abi_version()
        if abi != ES_ABI_VERSION:
            raise RuntimeError(f"ABI mismatch in {self.path}: expected {ES_ABI_VERSION}, got {abi}")

    def _bind_signatures(self) -> None:
        d = self._dll
        c_u8_p = ctypes.c_char_p
        c_void_p = ctypes.c_void_p
        c_size = ctypes.c_size_t
        c_f64 = ctypes.c_double
        c_f64_p = ctypes.POINTER(ctypes.c_double)
        c_i32 = ctypes.c_int32
        c_u32 = ctypes.c_uint32
        c_i64 = ctypes.c_int64

        d.es_abi_version.argtypes = []
        d.es_abi_version.restype = c_u32

        d.es_alloc.argtypes = [c_size]
        d.es_alloc.restype = c_void_p

        d.es_free.argtypes = [c_void_p, c_size]
        d.es_free.restype = None

        d.es_last_error_ptr.argtypes = []
        d.es_last_error_ptr.restype = c_void_p

        d.es_last_error_len.argtypes = []
        d.es_last_error_len.restype = c_size

        d.es_decode_waveforms.argtypes = [c_u8_p, c_size, ctypes.POINTER(c_void_p), ctypes.POINTER(c_size)]
        d.es_decode_waveforms.restype = c_i32

        d.es_encode_sac.argtypes = [
            c_u8_p, c_size, c_i64, c_i32, c_f64, c_f64_p, c_size, c_u32,
            ctypes.POINTER(c_void_p), ctypes.POINTER(c_size),
        ]
        d.es_encode_sac.restype = c_i32

        d.es_encode_mseed2_steim2.argtypes = [
            c_u8_p, c_size, c_i64, c_i32, c_f64, c_f64_p, c_size, c_u32,
            ctypes.POINTER(c_void_p), ctypes.POINTER(c_size),
        ]
        d.es_encode_mseed2_steim2.restype = c_i32

        d.es_encode_mseed3.argtypes = [
            c_u8_p, c_size, c_i64, c_i32, c_f64, c_f64_p, c_size, c_u32,
            ctypes.POINTER(c_void_p), ctypes.POINTER(c_size),
        ]
        d.es_encode_mseed3.restype = c_i32

        d.es_wgs84_geodesic.argtypes = [c_f64, c_f64, c_f64, c_f64, c_f64_p]
        d.es_wgs84_geodesic.restype = c_i32

        d.es_demean_detrend_taper.argtypes = [c_f64_p, c_size, c_u32, c_u32, c_f64]
        d.es_demean_detrend_taper.restype = c_i32

        d.es_sos_filter.argtypes = [
            c_f64_p, c_size, c_f64, c_u32, c_u32, c_u32, c_f64, c_f64, c_f64, c_u32, c_f64_p
        ]
        d.es_sos_filter.restype = c_i32

        d.es_remove_response.argtypes = [
            c_f64_p, c_size, c_f64, c_u8_p, c_size, c_u32, c_u8_p, c_size, c_i64, c_u32,
            c_f64_p, c_f64, c_f64_p,
        ]
        d.es_remove_response.restype = c_i32

        d.es_nlnm_nhnm.argtypes = [c_f64_p, c_size, c_f64_p, c_f64_p]
        d.es_nlnm_nhnm.restype = c_i32

        d.es_hilbert_envelope.argtypes = [c_f64_p, c_size, c_f64_p]
        d.es_hilbert_envelope.restype = c_i32

        d.es_pick_arrivals.argtypes = [
            c_f64_p, c_size, c_f64, c_u32, c_f64, c_f64, c_f64,
            ctypes.POINTER(c_i64), c_f64_p, c_f64_p,
        ]
        d.es_pick_arrivals.restype = c_i32

        d.es_taup_engine_create.argtypes = [c_u8_p, c_size]
        d.es_taup_engine_create.restype = c_void_p

        d.es_taup_engine_free.argtypes = [c_void_p]
        d.es_taup_engine_free.restype = None

        for fn_name in ("es_taup_travel_times", "es_taup_ray_paths", "es_taup_pierce_points"):
            fn = getattr(d, fn_name)
            fn.argtypes = [
                c_void_p, c_u8_p, c_size, c_f64, c_f64, c_u8_p, c_size,
                ctypes.POINTER(c_void_p), ctypes.POINTER(c_size),
            ]
            fn.restype = c_i32

    def _check(self, code: int) -> None:
        if code == ES_OK:
            return
        err_len = self._dll.es_last_error_len()
        err_ptr = self._dll.es_last_error_ptr()
        msg = ctypes.string_at(err_ptr, err_len).decode("utf-8", errors="replace") if (err_ptr and err_len) else "unknown error"
        raise EarthSnailFFIError(code, msg)

    def _consume_owned(self, ptr: ctypes.c_void_p, length: ctypes.c_size_t) -> bytes:
        if not ptr.value or length.value == 0:
            return b""
        try:
            return ctypes.string_at(ptr.value, length.value)
        finally:
            self._dll.es_free(ptr, length)

    def read_waveforms(self, source: Union[bytes, bytearray, str, pathlib.Path]) -> list[WaveformTrace]:
        raw = pathlib.Path(source).read_bytes() if isinstance(source, (str, pathlib.Path)) else bytes(source)
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        rc = self._dll.es_decode_waveforms(raw, len(raw), ctypes.byref(out_ptr), ctypes.byref(out_len))
        self._check(rc)
        payload = self._consume_owned(out_ptr, out_len)
        return unpack_eswf(payload)

    def encode_mseed2_steim2(
        self,
        samples: Sequence[float],
        sampling_rate_hz: float,
        *,
        channel: str = "XX.STA.00.BHZ",
        start_epoch_s: int = 1_700_000_000,
        start_subsec_ns: int = 0,
        record_len: int = 512,
    ) -> bytes:
        ch_b = channel.encode("utf-8")
        c_arr, n = _to_c_double_array(samples)
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        rc = self._dll.es_encode_mseed2_steim2(
            ch_b, len(ch_b), start_epoch_s, start_subsec_ns, sampling_rate_hz,
            c_arr, n, record_len, ctypes.byref(out_ptr), ctypes.byref(out_len)
        )
        self._check(rc)
        return self._consume_owned(out_ptr, out_len)

    def encode_sac(
        self,
        samples: Sequence[float],
        sampling_rate_hz: float,
        *,
        channel: str = "XX.STA.00.BHZ",
        start_epoch_s: int = 1_700_000_000,
        start_subsec_ns: int = 0,
        big_endian: bool = False,
    ) -> bytes:
        ch_b = channel.encode("utf-8")
        c_arr, n = _to_c_double_array(samples)
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        rc = self._dll.es_encode_sac(
            ch_b, len(ch_b), start_epoch_s, start_subsec_ns, sampling_rate_hz,
            c_arr, n, 1 if big_endian else 0, ctypes.byref(out_ptr), ctypes.byref(out_len)
        )
        self._check(rc)
        return self._consume_owned(out_ptr, out_len)

    def wgs84_geodesic(
        self, lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
    ) -> GeodesicResult:
        out5 = (ctypes.c_double * 5)()
        rc = self._dll.es_wgs84_geodesic(lat1_deg, lon1_deg, lat2_deg, lon2_deg, out5)
        self._check(rc)
        return GeodesicResult(
            distance_m=out5[0],
            gcarc_deg=out5[1],
            distance_deg_spherical=out5[2],
            azimuth_deg=out5[3],
            back_azimuth_deg=out5[4],
        )

    def filter_trace(
        self,
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
    ) -> "array.array[float]":
        fam_map = {"butter": 0, "butterworth": 0, "cheby1": 1, "cheby2": 2, "bessel": 3}
        band_map = {"bandpass": 0, "bp": 0, "lowpass": 1, "lp": 1, "highpass": 2, "hp": 2, "bandstop": 3, "bs": 3}
        if isinstance(zero_phase, bool):
            zp_mode = 1 if zero_phase else 0
        else:
            zp_mode = {"causal": 0, "obspy": 1, "sac": 1, "filtfilt": 2}[zero_phase.lower()]
        c_in, n = _to_c_double_array(samples)
        c_out = (ctypes.c_double * n)()
        rc = self._dll.es_sos_filter(
            c_in, n, sampling_rate_hz,
            fam_map[family.lower()], band_map[band.lower()], order,
            f_low_hz, f_high_hz, ripple_db, zp_mode, c_out
        )
        self._check(rc)
        res = array.array("d")
        res.frombytes(ctypes.string_at(c_out, n * 8))
        return res

    def pick_morse(
        self,
        samples: Sequence[float],
        sampling_rate_hz: float = 100.0,
        *,
        nsta: int = 0,
        nlta: int = 0,
        thr_on: float = 0.0,
        return_cf: bool = False,
    ) -> PickResult:
        c_in, n = _to_c_double_array(samples)
        pick_idx = ctypes.c_int64(-1)
        peak = ctypes.c_double(0.0)
        cf_buf = (ctypes.c_double * n)() if return_cf else None
        cf_ptr = ctypes.cast(cf_buf, ctypes.POINTER(ctypes.c_double)) if cf_buf is not None else ctypes.POINTER(ctypes.c_double)()
        rc = self._dll.es_pick_arrivals(
            c_in, n, sampling_rate_hz, 4, float(nsta), float(nlta), float(thr_on),
            ctypes.byref(pick_idx), ctypes.byref(peak), cf_ptr
        )
        self._check(rc)
        cf_arr: Optional[array.array[float]] = None
        if cf_buf is not None:
            cf_arr = array.array("d")
            cf_arr.frombytes(ctypes.string_at(cf_buf, n * 8))
        return PickResult(
            pick_sample=None if pick_idx.value < 0 else int(pick_idx.value),
            peak_metric=float(peak.value),
            characteristic_function=cf_arr,
        )

    def taup_travel_times(
        self,
        depth_km: float,
        distance_deg: float,
        phases: Iterable[str] = ("P", "pP", "PP", "PcP", "S", "SKS", "PKIKP"),
        *,
        model: str = "iasp91",
    ) -> list[TauPArrival]:
        m_bytes = model.encode("utf-8")
        ph_bytes = ",".join(phases).encode("utf-8")
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        rc = self._dll.es_taup_travel_times(
            None, m_bytes, len(m_bytes), depth_km, distance_deg,
            ph_bytes, len(ph_bytes), ctypes.byref(out_ptr), ctypes.byref(out_len)
        )
        self._check(rc)
        return unpack_estt(self._consume_owned(out_ptr, out_len))

    def taup_ray_paths(
        self,
        depth_km: float,
        distance_deg: float,
        phases: Iterable[str] = ("P", "S", "PcP", "PKIKP"),
        *,
        model: str = "iasp91",
    ) -> list[TauPRayPath]:
        m_bytes = model.encode("utf-8")
        ph_bytes = ",".join(phases).encode("utf-8")
        out_ptr = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        rc = self._dll.es_taup_ray_paths(
            None, m_bytes, len(m_bytes), depth_km, distance_deg,
            ph_bytes, len(ph_bytes), ctypes.byref(out_ptr), ctypes.byref(out_len)
        )
        self._check(rc)
        return unpack_esrp(self._consume_owned(out_ptr, out_len), distance_deg)
