"""Executable self-test for EarthSnail Python ctypes bindings (`python -m earthsnail.selftest`)."""

from __future__ import annotations

import json
import math
import sys
import unittest

import earthsnail as es


class EarthSnailBindingsTest(unittest.TestCase):
    def test_steim2_mseed_and_sac_roundtrip(self) -> None:
        raw_samples = [float((i * 17 - (i % 7) * 93) + 1000) for i in range(300)]
        ms2 = es.encode_mseed2_steim2(raw_samples, 20.0, channel="G.CAN.00.BHZ")
        traces = es.read_waveforms(ms2)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].seed_id, "G.CAN.00.BHZ")
        self.assertAlmostEqual(traces[0].sampling_rate_hz, 20.0, places=9)
        self.assertEqual(list(traces[0].samples), raw_samples)

        sac_be = es.encode_sac(raw_samples, 40.0, channel="IU.ANMO.10.BHZ", big_endian=True)
        sac_traces = es.read_waveforms(sac_be)
        self.assertEqual(len(sac_traces), 1)
        self.assertEqual(sac_traces[0].format_code, 2)  # Big-Endian SAC
        for got, exp in zip(sac_traces[0].samples, raw_samples):
            self.assertAlmostEqual(got, exp, places=4)

    def test_wgs84_geodesic_tohoku_to_canberra(self) -> None:
        geo = es.wgs84_geodesic(38.297, 142.373, -35.318715, 148.996325)
        self.assertAlmostEqual(geo.distance_deg_spherical, 73.87078776567722, places=9)
        self.assertTrue(73.4 < geo.gcarc_deg < 74.0)
        self.assertGreater(geo.distance_m, 8_150_000.0)
        self.assertLess(geo.distance_m, 8_300_000.0)
        self.assertTrue(0.0 <= geo.azimuth_deg < 360.0)
        self.assertTrue(0.0 <= geo.back_azimuth_deg < 360.0)

    def test_zero_phase_filter_and_morse_glitch_immunity(self) -> None:
        fs = 50.0
        n = 500
        onset = 320
        sig = []
        for i in range(n):
            t = i / fs
            pi2 = 2.0 * math.pi
            val = 0.01 * (
                math.sin(pi2 * 5.3 * t) + 0.7 * math.cos(pi2 * 9.1 * t) + 0.5 * math.sin(pi2 * 13.7 * t)
            )
            if i >= onset:
                val += 3.0 * math.sin(pi2 * 4.0 * (t - onset / fs))
            sig.append(val)

        # Inject massive 1-sample telemetry glitch before true arrival:
        glitched = list(sig)
        glitched[150] += 5.0e6
        pick = es.pick_morse(glitched, fs, nsta=10, nlta=80, thr_on=4.5, return_cf=True)
        self.assertIsNotNone(pick.pick_sample)
        assert pick.pick_sample is not None
        self.assertLessEqual(abs(pick.pick_sample - onset), 10)
        assert pick.characteristic_function is not None
        self.assertEqual(len(pick.characteristic_function), n)

        # Zero-phase 4-pole Butterworth bandpass preserves signal energy cleanly:
        filtered = es.filter_trace(sig, fs, 1.0, 10.0, order=4, family="butter", zero_phase="filtfilt")
        self.assertEqual(len(filtered), n)
        self.assertGreater(max(abs(x) for x in filtered[onset:]), 2.0)

    def test_taup_travel_times_and_ray_paths_all_models(self) -> None:
        for model in ("iasp91", "ak135", "prem"):
            arrs = es.taup_travel_times(29.0, 73.87078776567722, ("P", "pP", "PcP", "S", "SKS"), model=model)
            names = [a.name for a in arrs]
            self.assertIn("P", names)
            self.assertIn("S", names)
            if model == "iasp91":
                p_arr = next(a for a in arrs if a.name == "P")
                self.assertAlmostEqual(p_arr.time_s, 692.0844477, places=2)

            paths = es.taup_ray_paths(120.0, 145.0, ("PKIKP",), model=model)
            self.assertGreaterEqual(len(paths), 1)
            self.assertGreater(len(paths[0].points), 10)
            max_depth = max(pt[1] for pt in paths[0].points)
            self.assertGreater(max_depth, paths[0].iocb_depth_km)  # penetrates inner core!

    def test_invalid_inputs_raise_typed_ffi_error(self) -> None:
        with self.assertRaises(es.EarthSnailFFIError):
            es.wgs84_geodesic(float("nan"), 0.0, 10.0, 20.0)
        with self.assertRaises(es.EarthSnailFFIError):
            es.read_waveforms(b"not a valid seismic file")


def main() -> int:
    lib = es.get_default_lib()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EarthSnailBindingsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    summary = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "library": str(lib.path),
        "abi_version": es.ES_ABI_VERSION,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }
    print(json.dumps(summary))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
