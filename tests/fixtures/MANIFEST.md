# EarthSnail Ground-Truth Differential Oracle Manifest

**Generator**: `oracle_staging/generate_golden_fixtures.py` + `oracle_staging/run_obspy_baseline_bench.py`  
**Verification Suite**: `oracle_staging/test_verify_fixtures.py`  
**Isolated Runtime (`.venv`)**: CPython `3.12.13` (`uv 0.13.4`), `obspy==1.5.1`, `scipy==1.18.1`, `numpy==2.5.5`, `geographiclib==2.1`, `psutil==7.2.2`

All JSON fixtures are serialized with `allow_nan=False` (strict IEEE-754 finite `f64` precision, zero `NaN`/`Infinity` tokens) and SHA-256 checksums pinning every binary/XML/ASCII fixture.

---

## 1. Directory Layout

```text
oracle_staging/
├── generate_golden_fixtures.py      # Deterministic golden fixture generator
├── run_obspy_baseline_bench.py      # Subprocess-isolated ObsPy/SciPy/TauPy benchmark runner
├── test_verify_fixtures.py          # Pytest integrity & parity audit suite
├── obspy_baseline_bench.json        # Measured ObsPy latency, throughput, tracemalloc & OS peak RSS
├── requirements.txt                 # Exact frozen uv venv package versions
├── MANIFEST.md                      # This schema & algorithmic specification
└── fixtures/
    ├── geodesics.json               # 25 WGS84 coordinate pairs across 6 regimes
    ├── dsp.json                     # Full DSP, filtering, deconvolution, pickers, 3C, noise, Lanczos
    ├── taup.json                    # iasp91 / ak135 / prem travel times, pierce points & ray paths
    └── io/
        ├── test_le.sac              # SAC Binary v6 Little-Endian (500 samples, 100 Hz, LCALDA=true)
        ├── test_be.sac              # SAC Binary v6 Big-Endian (identical waveform & headers)
        ├── test_alpha.sac           # SAC Alphanumeric (SACXY) representation
        ├── sac_expected.json        # Exact SAC headers, stats, SHA-256 & sample arrays
        ├── mseed_steim1.mseed       # Multi-record 512B miniSEED v2 (STEIM1, code 10, 8..32-bit diffs)
        ├── mseed_steim2.mseed       # Multi-record 512B miniSEED v2 (STEIM2, code 11, 4..30-bit diffs)
        ├── mseed_int32.mseed        # Multi-record 512B miniSEED v2 (INT32, code 3)
        ├── mseed_float64.mseed      # Multi-record 512B miniSEED v2 (FLOAT64, code 5)
        ├── mseed_expected.json      # Exact record counts, stats, SHA-256 & decoded sample arrays
        ├── iu_anmo_bhz.xml          # Real StationXML (IU.ANMO.10.BHZ, 40 Hz, PZ + 2 FIR stages)
        ├── iu_anmo_bhz.sacpz        # Exported SAC PoleZero file (INPUT UNIT: M -> COUNTS)
        ├── events.qml               # Valid QuakeML 1.2 catalog (4 events: origins, mags, picks, MT/FM)
        ├── metadata_expected.json   # Parsed metadata + complex H(f) at [0.01, 0.1, 1.0, 5.0, 15.0] Hz
        └── real_event/
            ├── real_3c.mseed        # Real 3C broadband recording (2011 Mw 9.1 Tohoku-Oki @ G.CAN.00.BH*)
            ├── real_3c_station.xml  # Matching 3-channel StationXML with full responses
            ├── real_3c_bhe.sac      # Populated SAC Binary (evla/evlo/evdp/mag/o + TauP P in t0, S in t1)
            ├── real_3c_bhn.sac      # Populated SAC Binary (North component)
            ├── real_3c_bhz.sac      # Populated SAC Binary (Vertical component)
            └── real_event_expected.json
```

---

## 2. Exact JSON Schemas & Algorithmic Conventions

### 2.1 I/O Fixtures (`fixtures/io/`)

#### `sac_expected.json`
- `files.{little_endian, big_endian, alpha}`:
  - `file` (`string`), `encoding` (`"SAC_BINARY_LE" | "SAC_BINARY_BE" | "SAC_ALPHA"`), `size_bytes` (`u64`), `sha256` (`string`).
  - `stats`: `{ network, station, location, channel, starttime, endtime, sampling_rate, delta, npts, calib }`.
  - `sac_header`: Map of defined SAC header keys (lowercase: `stla`, `stlo`, `stel`, `stdp`, `evla`, `evlo`, `evdp`, `mag`, `dist`, `az`, `baz`, `gcarc`, `b`, `e`, `o`, `a`, `t0`, `t1`, `kt0`, `kt1`, `kstnm`, `knetwk`, `kcmpnm`, `khole`, `kevnm`, `nzyear..nzmsec`, etc.). Undefined `-12345` fields are omitted by ObsPy.
  - `data_f32_as_f64`: `Vec<f64>` (`500` samples). Note that `little_endian` and `big_endian` share bit-identical `f32` samples; `alpha` reflects 7-column ASCII `%15.7g` rounding (`~6e-8` max difference).

#### `mseed_expected.json`
- `files.{steim1, steim2, int32, float64}`:
  - `file`, `encoding_name`, `seed_encoding_code` (`10`, `11`, `3`, `5`), `record_length_bytes` (`512`), `num_records` (`>= 2`), `size_bytes`, `sha256`, `dtype`.
  - `stats`: `{ network, station, location, channel, starttime, endtime, sampling_rate, npts }`.
  - `max_abs_first_diff`: Peak `|x[i] - x[i-1]|`.
    - `steim2` exercises all 7 Steim-2 nibble widths (`4, 5, 6, 8, 10, 15, 30`-bit signed diffs up to `±536,000,000`).
    - `steim1` additionally includes `±1,200,000,000` impulses exceeding 30 bits to test true 32-bit Steim-1 words (`c = 0b11`).
  - `samples`: Exact lossless `Vec<i32>` (`steim1`, `steim2`, `int32`) or `Vec<f64>` (`float64`).

#### `metadata_expected.json`
- Complex numbers throughout EarthSnail fixtures use the uniform object schema:
  ```json
  { "real": 1.45411949e9, "imag": 1.98144774e9, "abs": 2.45778e9, "phase_rad": 0.9377 }
  ```
- `stationxml`:
  - Metadata for `iu_anmo_bhz.xml` (`IU.ANMO.10.BHZ`, `40.0 Hz`, Nyquist `20.0 Hz`): `latitude`, `longitude`, `elevation_m`, `depth_m`, `azimuth_deg`, `dip_deg`, `overall_sensitivity`, `sensitivity_frequency_hz`, `stages[]` (Stage 1 `PolesZerosResponseStage` `LAPLACE (RADIANS/SECOND)` + Stages 2–3 FIR `CoefficientsTypeResponseStage`).
  - `frequencies_hz`: `[0.01, 0.1, 1.0, 5.0, 15.0]`.
  - `evalresp_complex_response.{DISP, VEL, ACC}`: 5 complex values each from `Response.get_evalresp_response_for_frequencies`.
- `sacpz`:
  - Parsed from `iu_anmo_bhz.sacpz` (`INPUT UNIT : M` -> displacement response in rad/s): `constant`, `a0`, `sensitivity`, `zeros_disp_rad_s[]`, `poles_rad_s[]`.
  - `complex_response.{DISP, VEL, ACC}`: $H(s) = \text{CONSTANT} \cdot \prod (s - z_k) / \prod (s - p_m)$ at $s = i 2\pi f$, where `DISP` uses all 3 origin zeros, `VEL` removes one `(0, 0)` zero, and `ACC` removes two `(0, 0)` zeros.
- `quakeml`:
  - `num_events` (`4`), `events[]` with `resource_id`, `preferred_origin` (`time`, `latitude`, `longitude`, `depth_m`), `preferred_magnitude` (`mag`, `magnitude_type`), `picks[]`, and `focal_mechanisms[]` (`nodal_plane_1`, `nodal_plane_2`, `moment_tensor` with `scalar_moment_nm`, `m_rr`, `m_tt`, `m_pp`, `m_rt`, `m_rp`, `m_tp`).

#### `real_event/real_event_expected.json`
- Landmark broadband 3-component recording (`G.CAN.00.BHE/BHN/BHZ`, 20 Hz, `2011-03-11 Mw 9.1 Tohoku-Oki`) saved as multi-trace `real_3c.mseed`, 3-channel `real_3c_station.xml`, and 3 `LCALDA`-populated SAC files (`real_3c_bhe.sac`, `real_3c_bhn.sac`, `real_3c_bhz.sac`) with TauP `iasp91` theoretical `P` (`t0`/`kt0`) and `S` (`t1`/`kt1`) markers and deconvolved `PGV` (`m/s`).

---

### 2.2 Geodesics Fixtures (`fixtures/geodesics.json`)

- `ellipsoid`: `{ "name": "WGS84", "a_m": 6378137.0, "f": 0.0033528106647474805 }`.
- `pairs[]` (`25` entries across `regional`, `teleseismic`, `trans_equatorial`, `polar`, `near_coincident`, `near_antipodal`):
  - `id`, `category`, `lat1_deg`, `lon1_deg`, `lat2_deg`, `lon2_deg`.
  - `dist_m`, `az_deg`, `baz_deg`: Exact Karney (2013) `geographiclib` WGS84 ellipsoidal solution via `obspy.geodetics.gps2dist_azimuth`.
  - `spherical_dist_deg`: Great-circle central angle on a sphere via `obspy.geodetics.locations2degrees`.

---

### 2.3 DSP & Signal Fixtures (`fixtures/dsp.json`)

- `reference_waveforms`: `sampling_rate_hz = 100.0`, `npts = 1000`, `clean_arrival_100hz` (`Vec<f64>`), `drifted_100hz` (`Vec<f64>`).
- `detrend`: `{ input, demean, linear, polynomial_order_3 }` (`1000` samples each).
- `tapers`: `{ hann_full, hann_0.05, cosine_0.05, sac_cosine_0.05_each_side, kaiser_full_beta_8.6, kaiser_0.05_beta_8.6 }`, each containing both `window` (`Vec<f64>`) and `tapered` (`Vec<f64>`).
- `filters.{butter, cheby1, bessel}.{bandpass, lowpass, highpass, bandstop}`:
  - `corners = 4`, `sampling_rate_hz = 100.0`, `freqs_hz` (`[0.5, 5.0]` or `[5.0]` or `[0.5]`).
  - Design conventions: `cheby1` uses passband ripple `rp_db = 1.0`; `bessel` uses SciPy's phase-matched normalization `bessel_norm = "phase"`.
  - `sos_sections`: Exact `Vec<[f64; 6]>` rows `[b0, b1, b2, 1.0, a1, a2]`.
  - **Three filter outputs provided per entry**:
    1. `zerophase_false`: Causal 1-pass `scipy.signal.sosfilt(sos, x)`.
    2. `zerophase_true`: ObsPy `zerophase=True`, which executes unpadded forward-backward `flip(sosfilt(sos, flip(sosfilt(sos, x))))` (matching classic SAC `p 2`).
    3. `zerophase_sosfiltfilt`: `scipy.signal.sosfiltfilt(sos, x)` with odd-reflection initial-state edge stabilization.
- `remove_response`:
  - Evaluates `input_counts` (`4096` samples @ `40.0 Hz`, `nfft = 8192`) with `pre_filt_hz = [0.005, 0.01, 15.0, 20.0]`, `water_level_db = 60.0`, `zero_mean = true`, `taper = true`, `taper_fraction = 0.05`.
  - Exact ObsPy spectral deconvolution pipeline:
    1. Subtract mean (`x -= mean(x)`).
    2. Multiply by SAC time-domain cosine taper (`5%` each side = `p=0.10` total).
    3. Real FFT of length `nfft = 8192`.
    4. Multiply complex data spectrum by `cosine_sac_taper(freqs, [0.005, 0.01, 15.0, 20.0])`.
    5. Evaluate instrument transfer function $H(f)$ (`evalresp` on `io/iu_anmo_bhz.xml` for `stationxml_outputs`, or continuous-time `SACPZ` on `io/iu_anmo_bhz.sacpz` for `sacpz_outputs`).
    6. Water-level inversion: `w = max(|H|) * 10^(-60 / 20)`; for `0 < |H[k]| < w`, scale `H[k] *= w / |H[k]|`; invert `H_inv[k] = 1 / H[k]` (with `H_inv[0] = 0`).
    7. Multiply `X[k] *= H_inv[k]`, force Nyquist bin real `X[nfft/2] = |X[nfft/2]| + 0i`, and take `irfft(X)[0..npts]`.
  - Outputs provided for `"VEL"`, `"DISP"`, and `"ACC"` under both `stationxml_outputs` and `sacpz_outputs`.
- `hilbert`: `{ envelope, quadrature_imag, instantaneous_phase_rad }` on `clean_arrival_100hz`.
- `cross_correlation`: `{ signal_a, signal_b, max_shift_samples: 100, expected_shift_samples: -17, expected_max_value, cc_curve }` (`201` lags from `-100` to `+100`). Note ObsPy's sign convention: `correlate(a, b, shift)` evaluates $\sum_n \tilde{a}[n + k]\,\tilde{b}[n] / \sqrt{\sum \tilde{a}^2 \sum \tilde{b}^2}$, so when `signal_b` is delayed by `+17` samples (`b[n] ≈ a[n - 17]`), the peak occurs at lag `k = -17` (`cc_curve[100 + (-17)] = cc_curve[83]`).
- `sta_lta`: `{ nsta: 50, nlta: 200, thr_on: 3.5, thr_off: 1.0, classic_cft, classic_triggers, recursive_cft, recursive_triggers }`.
- `baer_picker`: `pk_baer` (`samp_int=100, tdownmax=20, tupevent=60, thr1=7.0, thr2=12.0, preset_len=100, p_dur=100`) with `pick_sample`, `first_motion`, and `characteristic_function` (`999` samples) on both real `BW.RJOB..EHZ` (`rjob_ehz_1000`) and synthetic `clean_arrival_100hz`.
- `rotation_and_polarization`:
  - `input_z`, `input_n`, `input_e` (`1000` samples, `back_azimuth_deg = 215.0`, `incidence_deg = 28.0`).
  - `rotate_ne_rt.{radial, transverse}` and `rotate_zne_lqt.{l, q, t}`.
  - `flinn_p_window`: Eigenstructure covariance of `[E, N, Z]` on samples `[250..350)` (`rect = 1 - sqrt(lambda_2 / lambda_1)`, `plan = 1 - 2*lambda_3 / (lambda_1 + lambda_2)`, azimuth folded to `[0, 180]`, incidence to `[0, 90]`).
  - `polarization_analysis_flinn`: Sliding-window (`win_len_s = 1.0`, `win_frac = 0.25`) arrays.
- `peterson_noise_models`: `nlnm` and `nhnm` (`1001` `{ periods_s, power_db }` points each in `dB rel. 1 (m/s^2)^2/Hz`).
- `lanczos_resampling`: `a = 20`, `window = "lanczos"`, `downsample_100hz_to_40hz` (`500 -> 200` samples) and `upsample_40hz_to_100hz` (`200 -> 496` samples).

---

### 2.4 TauP Fixtures (`fixtures/taup.json`)

- **Grid Sweep (`travel_times[]`)**:
  - `models`: `["iasp91", "ak135", "prem"]`
  - `source_depths_km`: `[0.0, 15.0, 35.0, 120.0, 450.0, 620.0]`
  - `distances_deg`: `[0.5, 5.0, 18.0, 35.0, 65.0, 85.0, 105.0, 130.0, 145.0, 155.0, 175.0]`
  - `phases` (`24`): `["p", "s", "P", "S", "Pn", "Sn", "Pg", "Sg", "pP", "sP", "PP", "SS", "PcP", "ScS", "PKP", "PKIKP", "PKiKP", "SKS", "SKKS", "Pdiff", "Sdiff", "P410s", "P660s", "3.5kmps"]`
  - Each `arrivals[]` item records **both** ray-parameter unit conventions to prevent unit ambiguity:
    - `name`, `purist_name`, `time_s`, `ray_param_s_per_rad` (`s/rad`), `ray_param_s_per_deg` (`s/deg`), `takeoff_angle_deg`, `incident_angle_deg`, `distance_deg`, `purist_distance_deg`.
- **Pierce Points (`pierce_examples[]`)**:
  - Multi-model pierce tables (`iasp91`, `ak135`, `prem`) with `{ p_s_per_rad, time_s, dist_rad, dist_deg, depth_km }` at every model discontinuity crossing.
- **Ray Paths (`ray_path_examples[]`)**:
  - Full ray-path `(p, time, dist_rad, dist_deg, depth_km)` trajectories across `iasp91`, `ak135`, and `prem` (`P`, `S`, `pP`, `PcP`, `P410s`, `Pdiff`, `SKS`, `PKP`, `PKIKP`, `PKiKP`).

---

## 3. Baseline Benchmark Schema & Measured Results (`obspy_baseline_bench.json`)

Each benchmark runs in a fresh, isolated Python child process:
- `median_ms`, `min_ms`, `mean_ms`, `p95_ms`, `throughput_per_sec`
- `peak_rss_bytes`: Win32 OS kernel `PeakWorkingSetSize` (`peak_wset`), capturing C-extension (`libmseed`, `evalresp`, `PocketFFT`) + Python interpreter memory.
- `tracemalloc_peak_bytes`: Peak Python-managed heap allocation during the hot loop.

| Workload | Median Latency (`ms`) | Throughput | Python `tracemalloc` Peak | OS Process Peak RSS |
| :--- | ---: | ---: | ---: | ---: |
| `cold_import` (`obspy` + `scipy` + `TauPyModel`) | `7,486.34` | — | `75.35 MiB` | `195.6 MiB` |
| `filter_bandpass_causal_100k` | `4.97` | `20.11 MSamp/s` | `0.81 MiB` | `132.9 MiB` |
| `filter_bandpass_zerophase_100k` | `5.80` | `17.24 MSamp/s` | `1.58 MiB` | `131.9 MiB` |
| `filter_bandpass_causal_1m` | `14.23` | `70.27 MSamp/s` | `7.67 MiB` | `159.8 MiB` |
| `filter_bandpass_zerophase_1m` | `25.18` | `39.71 MSamp/s` | `15.30 MiB` | `159.5 MiB` |
| `steim2_decode_100k` | `5.10` | `19.60 MSamp/s` | `0.67 MiB` | `131.9 MiB` |
| `steim2_decode_1m` | `22.13` | `45.19 MSamp/s` | `6.04 MiB` | `149.2 MiB` |
| `fft_rfft_irfft_100k` | `3.39` | `29.52 MSamp/s` | `1.53 MiB` | `134.0 MiB` |
| `fft_rfft_irfft_1m` | `45.47` | `21.99 MSamp/s` | `15.26 MiB` | `167.5 MiB` |
| `remove_response_100k` (`StationXML` `evalresp`) | `102.23` | `0.98 MSamp/s` | `9.97 MiB` | `143.6 MiB` |
| `remove_response_1m` (`StationXML` `evalresp`) | `1,026.00` | `0.97 MSamp/s` | `99.38 MiB` | `238.6 MiB` |
| `taup_1000_queries_iasp91` (8-phase list) | `168,976.13` | `5.92 queries/s` | `34.93 MiB` | `175.6 MiB` |

