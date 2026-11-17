//! Miri-audited unit and adversarial FFI boundary tests for `earthsnail-wasm`.
//!
//! Exercises every `extern "C"` entrypoint across valid inputs, in-place aliased pointers,
//! null pointers, misaligned pointers, overflow lengths, invalid UTF-8, and corrupted payloads.

use super::*;
use core::ptr::{null, null_mut};

fn last_error_string() -> String {
    let len = es_last_error_len();
    if len == 0 {
        return String::new();
    }
    let ptr = es_last_error_ptr();
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn miri_alloc_free_and_adversarial_pointer_safety() {
    assert_eq!(es_abi_version(), ES_ABI_VERSION);

    // Zero & oversize allocations must fail cleanly without UB or abort:
    assert!(es_alloc(0).is_null());
    assert!(!last_error_string().is_empty());
    assert!(es_alloc(MAX_FFI_BYTES + 8).is_null());

    // Valid allocation is 8-byte aligned and zero-initialized:
    let buf = es_alloc(37);
    assert!(!buf.is_null());
    assert_eq!((buf as usize) % 8, 0);
    unsafe {
        let slice = core::slice::from_raw_parts_mut(buf, 37);
        assert!(slice.iter().all(|&b| b == 0));
        slice[0] = 0xAA;
        slice[36] = 0x55;
        // Null or zero-len free is a safe no-op:
        es_free(null_mut(), 37);
        es_free(buf, 0);
        // Misaligned pointer rejected safely without UB:
        es_free(buf.add(1), 36);
        assert!(!last_error_string().is_empty());
        es_clear_error();
        assert_eq!(es_last_error_len(), 0);
        // Exact free:
        es_free(buf, 37);
    }

    // Null & misaligned pointer checks across representative FFI entrypoints:
    let mut out5 = [0.0_f64; 5];
    unsafe {
        assert_eq!(
            es_wgs84_geodesic(0.0, 0.0, 10.0, 20.0, null_mut()),
            ES_ERR_NULL_PTR
        );
        // Misaligned f64 pointer:
        let raw_bytes = [0u8; 64];
        let mis_f64 = raw_bytes.as_ptr().add(1) as *mut f64;
        assert_eq!(
            es_wgs84_geodesic(0.0, 0.0, 10.0, 20.0, mis_f64),
            ES_ERR_NULL_PTR
        );
        // NaN coordinate:
        assert_eq!(
            es_wgs84_geodesic(f64::NAN, 0.0, 10.0, 20.0, out5.as_mut_ptr()),
            ES_ERR_INVALID_ARG
        );

        // Overflow slice length:
        let mut dummy_ptr: *mut u8 = null_mut();
        let mut dummy_len: usize = 0;
        assert_eq!(
            es_decode_waveforms(
                raw_bytes.as_ptr(),
                usize::MAX,
                &mut dummy_ptr,
                &mut dummy_len
            ),
            ES_ERR_ALLOC
        );
        // Invalid UTF-8 channel string:
        let bad_utf8 = [0xFFu8, 0xFE, 0xFD];
        let one_sample = [1.0_f64];
        assert_eq!(
            es_encode_sac(
                bad_utf8.as_ptr(),
                bad_utf8.len(),
                1_700_000_000,
                0,
                20.0,
                one_sample.as_ptr(),
                1,
                0,
                &mut dummy_ptr,
                &mut dummy_len,
            ),
            ES_ERR_UTF8
        );
    }
}

#[test]
fn miri_waveform_codecs_sac_mseed2_steim2_mseed3_roundtrip() {
    // Steim-2 frame codec across multiple difference nibble widths:
    let steps = [0_i32, 3, -5, 14, -29, 110, -450, 12_000, -25_000, 100_000];
    let mut i32_samples = Vec::with_capacity(steps.len() + 1);
    let mut acc = 10_000_i32;
    i32_samples.push(acc);
    for &d in &steps {
        acc += d;
        i32_samples.push(acc);
    }

    unsafe {
        let mut f_ptr: *mut u8 = null_mut();
        let mut f_len: usize = 0;
        assert_eq!(
            es_encode_steim2_frames(
                i32_samples.as_ptr(),
                i32_samples.len(),
                0,
                &mut f_ptr,
                &mut f_len,
            ),
            ES_OK
        );
        assert!(!f_ptr.is_null() && f_len >= 64);

        let mut decoded_i32 = vec![0_i32; i32_samples.len()];
        assert_eq!(
            es_decode_steim2_frames(f_ptr, f_len, i32_samples.len(), decoded_i32.as_mut_ptr(),),
            ES_OK
        );
        assert_eq!(decoded_i32, i32_samples);

        // Corrupt Xn integration check -> must return ES_ERR_PARSE without panic/UB:
        let frame_slice = core::slice::from_raw_parts_mut(f_ptr, f_len);
        frame_slice[11] ^= 0x01;
        assert_eq!(
            es_decode_steim2_frames(f_ptr, f_len, i32_samples.len(), decoded_i32.as_mut_ptr(),),
            ES_ERR_PARSE
        );
        es_free(f_ptr, f_len);
    }

    // Full SAC LE/BE/Alpha, miniSEED v2 Steim-2, and miniSEED v3 encode -> auto-sniff ESWF decode:
    let f64_samples: Vec<f64> = i32_samples.iter().map(|&x| x as f64).collect();
    let chan = b"IU.ANMO.00.BHZ";

    unsafe {
        // 1. SAC Little-Endian + Big-Endian:
        for mode in [0_u32, 1_u32] {
            let mut sac_ptr: *mut u8 = null_mut();
            let mut sac_len: usize = 0;
            assert_eq!(
                es_encode_sac(
                    chan.as_ptr(),
                    chan.len(),
                    1_700_000_000,
                    125_000_000,
                    40.0,
                    f64_samples.as_ptr(),
                    f64_samples.len(),
                    mode,
                    &mut sac_ptr,
                    &mut sac_len,
                ),
                ES_OK
            );
            let mut wf_ptr: *mut u8 = null_mut();
            let mut wf_len: usize = 0;
            assert_eq!(
                es_decode_waveforms(sac_ptr, sac_len, &mut wf_ptr, &mut wf_len),
                ES_OK
            );
            let wf = core::slice::from_raw_parts(wf_ptr, wf_len);
            assert_eq!(&wf[0..4], b"ESWF");
            let trace_count = u32::from_le_bytes(wf[8..12].try_into().unwrap());
            let fmt_code = u32::from_le_bytes(wf[12..16].try_into().unwrap());
            assert_eq!(trace_count, 1);
            assert_eq!(fmt_code, if mode == 0 { 1 } else { 2 });
            let npts = u64::from_le_bytes(wf[16..24].try_into().unwrap()) as usize;
            assert_eq!(npts, f64_samples.len());
            let samp_offset = 16 + 192;
            for (k, &exp) in f64_samples.iter().enumerate() {
                let got = f64::from_le_bytes(
                    wf[samp_offset + k * 8..samp_offset + (k + 1) * 8]
                        .try_into()
                        .unwrap(),
                );
                assert_eq!(got, exp);
            }
            es_free(wf_ptr, wf_len);
            es_free(sac_ptr, sac_len);
        }

        // 2. miniSEED v2 Steim-2:
        let mut ms2_ptr: *mut u8 = null_mut();
        let mut ms2_len: usize = 0;
        assert_eq!(
            es_encode_mseed2_steim2(
                chan.as_ptr(),
                chan.len(),
                1_700_000_000,
                0,
                20.0,
                f64_samples.as_ptr(),
                f64_samples.len(),
                512,
                &mut ms2_ptr,
                &mut ms2_len,
            ),
            ES_OK
        );
        let mut wf_ptr: *mut u8 = null_mut();
        let mut wf_len: usize = 0;
        assert_eq!(
            es_decode_waveforms(ms2_ptr, ms2_len, &mut wf_ptr, &mut wf_len),
            ES_OK
        );
        let wf = core::slice::from_raw_parts(wf_ptr, wf_len);
        assert_eq!(u32::from_le_bytes(wf[12..16].try_into().unwrap()), 4);
        let samp_offset = 16 + 192;
        for (k, &exp) in f64_samples.iter().enumerate() {
            let got = f64::from_le_bytes(
                wf[samp_offset + k * 8..samp_offset + (k + 1) * 8]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(got, exp);
        }
        es_free(wf_ptr, wf_len);
        es_free(ms2_ptr, ms2_len);

        // 3. miniSEED v3 Steim-2:
        let mut ms3_ptr: *mut u8 = null_mut();
        let mut ms3_len: usize = 0;
        assert_eq!(
            es_encode_mseed3(
                chan.as_ptr(),
                chan.len(),
                1_700_000_000,
                0,
                20.0,
                f64_samples.as_ptr(),
                f64_samples.len(),
                11,
                &mut ms3_ptr,
                &mut ms3_len,
            ),
            ES_OK
        );
        assert_eq!(
            es_decode_waveforms(ms3_ptr, ms3_len, &mut wf_ptr, &mut wf_len),
            ES_OK
        );
        let wf3 = core::slice::from_raw_parts(wf_ptr, wf_len);
        assert_eq!(u32::from_le_bytes(wf3[12..16].try_into().unwrap()), 5);
        es_free(wf_ptr, wf_len);
        es_free(ms3_ptr, ms3_len);
    }
}

#[test]
fn miri_geodesics_metadata_dsp_pickers_3c_and_moment_tensor() {
    let n = if cfg!(miri) { 240 } else { 600 };
    let fs = 40.0;

    // 1. WGS84 geodesic:
    let mut geo = [0.0_f64; 5];
    unsafe {
        assert_eq!(
            es_wgs84_geodesic(38.297, 142.373, -35.318715, 148.996325, geo.as_mut_ptr()),
            ES_OK
        );
    }
    assert!(geo[0] > 8_000_000.0 && geo[0] < 8_500_000.0);
    assert!((geo[2] - 73.87078776567722).abs() < 1e-9);
    assert!((geo[1] - 73.6).abs() < 0.5);

    // 2. SACPZ & StationXML eval_response + remove_response:
    let sacpz_txt = b"ZEROS 3\n0.0 0.0\nPOLES 2\n-0.037 0.037\n-0.037 -0.037\nCONSTANT 1.5e9\n";
    let freqs = [0.1_f64, 1.0, 5.0];
    let mut re_im = [0.0_f64; 6];
    let mut pz_const = 0.0_f64;
    unsafe {
        assert_eq!(
            es_parse_sacpz(
                sacpz_txt.as_ptr(),
                sacpz_txt.len(),
                1, // VEL
                freqs.as_ptr(),
                freqs.len(),
                re_im.as_mut_ptr(),
                &mut pz_const,
            ),
            ES_OK
        );
    }
    assert!((pz_const - 1.5e9).abs() < 1.0);
    assert!(re_im[2].hypot(re_im[3]) > 1.0e8);

    let station_xml = br#"<?xml version="1.0" encoding="UTF-8"?>
<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.2">
  <Source>Test</Source>
  <Network code="G">
    <Station code="CAN">
      <Latitude>-35.318715</Latitude>
      <Longitude>148.996325</Longitude>
      <Elevation>700.0</Elevation>
      <Site><Name>Canberra</Name></Site>
      <Channel code="BHZ" locationCode="00">
        <Latitude>-35.318715</Latitude>
        <Longitude>148.996325</Longitude>
        <Elevation>700.0</Elevation>
        <Depth>0.0</Depth>
        <Azimuth>0.0</Azimuth>
        <Dip>-90.0</Dip>
        <SampleRate>40.0</SampleRate>
        <Response>
          <InstrumentSensitivity><Value>1.5e9</Value><Frequency>1.0</Frequency><InputUnits><Name>M/S</Name></InputUnits><OutputUnits><Name>COUNTS</Name></OutputUnits></InstrumentSensitivity>
          <Stage number="1">
            <PolesZeros>
              <InputUnits><Name>M/S</Name></InputUnits>
              <OutputUnits><Name>V</Name></OutputUnits>
              <PzTransferFunctionType>LAPLACE (RADIANS/SECOND)</PzTransferFunctionType>
              <NormalizationFactor>1.0</NormalizationFactor>
              <NormalizationFrequency>1.0</NormalizationFrequency>
              <Zero number="0"><Real>0.0</Real><Imaginary>0.0</Imaginary></Zero>
              <Zero number="1"><Real>0.0</Real><Imaginary>0.0</Imaginary></Zero>
              <Pole number="0"><Real>-0.037</Real><Imaginary>0.037</Imaginary></Pole>
              <Pole number="1"><Real>-0.037</Real><Imaginary>-0.037</Imaginary></Pole>
            </PolesZeros>
            <StageGain><Value>1.5e9</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
        </Response>
      </Channel>
    </Station>
  </Network>
</FDSNStationXML>"#;

    let mut meta7 = [0.0_f64; 7];
    unsafe {
        assert_eq!(
            es_parse_stationxml_eval_response(
                station_xml.as_ptr(),
                station_xml.len(),
                b"G.CAN.00.BHZ".as_ptr(),
                12,
                0,
                1,
                freqs.as_ptr(),
                freqs.len(),
                re_im.as_mut_ptr(),
                meta7.as_mut_ptr(),
            ),
            ES_OK
        );
    }
    assert!((meta7[0] - (-35.318715)).abs() < 1e-6);

    // Build synthetic waveform with trend + passband arrival + 1-sample huge glitch:
    let onset = n * 2 / 3;
    let spike_idx = n / 3;
    let mut sig: Vec<f64> = (0..n)
        .map(|i| {
            let t = (i as f64) / fs;
            let pi2 = 2.0 * core::f64::consts::PI;
            let bg = 0.01
                * (earthsnail_core::math::sin(pi2 * 5.3 * t)
                    + 0.7 * earthsnail_core::math::cos(pi2 * 9.1 * t)
                    + 0.5 * earthsnail_core::math::sin(pi2 * 13.7 * t));
            let arr = if i >= onset {
                2.5 * earthsnail_core::math::sin(pi2 * 3.5 * (t - (onset as f64) / fs))
            } else {
                0.0
            };
            bg + arr
        })
        .collect();

    // Test aliased in-place SOS filter (`input_ptr == out_ptr`) under Miri Stacked/Tree Borrows:
    let mut aliased = sig.clone();
    unsafe {
        assert_eq!(
            es_demean_detrend_taper(aliased.as_mut_ptr(), n, 1 | 2 | 4, 0, 0.05),
            ES_OK
        );
        assert_eq!(
            es_sos_filter(
                aliased.as_ptr(),
                n,
                fs,
                0, // Butterworth
                0, // Bandpass
                4,
                0.5,
                8.0,
                0.0,
                2, // sosfiltfilt
                aliased.as_mut_ptr(),
            ),
            ES_OK
        );
    }

    // Remove response (auto-detect XML):
    let mut deconv = vec![0.0_f64; n];
    let pre_filt = [0.05_f64, 0.1, 12.0, 16.0];
    unsafe {
        assert_eq!(
            es_remove_response(
                sig.as_ptr(),
                n,
                fs,
                station_xml.as_ptr(),
                station_xml.len(),
                2, // auto-sniff
                b"BHZ".as_ptr(),
                3,
                0,
                1, // VEL
                pre_filt.as_ptr(),
                60.0,
                deconv.as_mut_ptr(),
            ),
            ES_OK
        );
    }
    assert!(deconv.iter().all(|v| v.is_finite()));

    // PSD, STFT spectrogram, NLNM/NHNM, and Hilbert envelope:
    unsafe {
        let mut req_bins = 0usize;
        assert_eq!(
            es_rfft_psd(
                sig.as_ptr(),
                n,
                fs,
                64,
                32,
                null_mut(),
                null_mut(),
                0,
                &mut req_bins
            ),
            ES_OK
        );
        assert_eq!(req_bins, 33);
        let mut p_freqs = vec![0.0_f64; req_bins];
        let mut p_vals = vec![0.0_f64; req_bins];
        let mut written = 0usize;
        assert_eq!(
            es_rfft_psd(
                sig.as_ptr(),
                n,
                fs,
                64,
                32,
                p_freqs.as_mut_ptr(),
                p_vals.as_mut_ptr(),
                req_bins,
                &mut written,
            ),
            ES_OK
        );
        assert_eq!(written, 33);

        let mut sp_ptr: *mut u8 = null_mut();
        let mut sp_len: usize = 0;
        assert_eq!(
            es_stft_spectrogram(sig.as_ptr(), n, fs, 32, 16, &mut sp_ptr, &mut sp_len),
            ES_OK
        );
        assert!(!sp_ptr.is_null() && sp_len > 32);
        es_free(sp_ptr, sp_len);

        let periods = [0.2_f64, 1.0, 10.0, 100.0];
        let mut nlnm = [0.0_f64; 4];
        let mut nhnm = [0.0_f64; 4];
        assert_eq!(
            es_nlnm_nhnm(periods.as_ptr(), 4, nlnm.as_mut_ptr(), nhnm.as_mut_ptr()),
            ES_OK
        );
        for k in 0..4 {
            assert!(nlnm[k] < nhnm[k]);
        }

        let mut env = vec![0.0_f64; n];
        assert_eq!(
            es_hilbert_envelope(sig.as_ptr(), n, env.as_mut_ptr()),
            ES_OK
        );
        assert!(env[onset + 5] > 1.0);
    }

    // Glitch-immune Morse vs Classic STA/LTA on a 1-sample 1e6 telemetry spike:
    sig[spike_idx] += 1.0e6;
    unsafe {
        let mut sta_pick = -1_i64;
        let mut sta_peak = 0.0_f64;
        assert_eq!(
            es_pick_arrivals(
                sig.as_ptr(),
                n,
                fs,
                0, // Classic STA/LTA -> fooled by single-sample glitch (ratio ~ nlta/nsta = 8.0)!
                8.0,
                64.0,
                4.0,
                &mut sta_pick,
                &mut sta_peak,
                null_mut(),
            ),
            ES_OK
        );
        assert!(
            (sta_pick - (spike_idx as i64)).abs() <= 2,
            "STA/LTA expected to false-trigger on glitch at {spike_idx}, got {sta_pick}"
        );

        let mut morse_pick_idx = -1_i64;
        let mut morse_peak = 0.0_f64;
        let mut morse_cf = vec![0.0_f64; n];
        assert_eq!(
            es_pick_arrivals(
                sig.as_ptr(),
                n,
                fs,
                4, // Discrete Morse -> ignores 1-sample 1e6 spike completely!
                8.0,
                64.0,
                4.5,
                &mut morse_pick_idx,
                &mut morse_peak,
                morse_cf.as_mut_ptr(),
            ),
            ES_OK
        );
        assert!(
            (morse_pick_idx - (onset as i64)).abs() <= 8,
            "Morse expected true onset ~{onset}, got {morse_pick_idx}"
        );
    }

    // 3C rotation, Flinn polarization, Response Spectrum, and Moment Tensor beachball:
    let mut r1 = vec![0.0_f64; n];
    let mut r2 = vec![0.0_f64; n];
    let mut r3 = vec![0.0_f64; n];
    let n_comp: Vec<f64> = aliased.iter().map(|&v| 0.7 * v).collect();
    let e_comp: Vec<f64> = aliased.iter().map(|&v| -0.5 * v).collect();
    unsafe {
        assert_eq!(
            es_rotate_3c(
                aliased.as_ptr(),
                n_comp.as_ptr(),
                e_comp.as_ptr(),
                n,
                2, // ZNE -> LQT
                125.0,
                28.0,
                r1.as_mut_ptr(),
                r2.as_mut_ptr(),
                r3.as_mut_ptr(),
            ),
            ES_OK
        );
        let mut pol7 = [0.0_f64; 7];
        assert_eq!(
            es_flinn_polarization(
                aliased.as_ptr(),
                n_comp.as_ptr(),
                e_comp.as_ptr(),
                n,
                0.0,
                pol7.as_mut_ptr(),
            ),
            ES_OK
        );
        assert!(pol7[2] > 0.95); // Linearly polarized synthetic -> rectilinearity ~ 1.0

        let rs_periods = [0.1_f64, 0.5, 2.0];
        let mut psa = [0.0_f64; 3];
        assert_eq!(
            es_response_spectrum(
                aliased.as_ptr(),
                n,
                1.0 / fs,
                rs_periods.as_ptr(),
                3,
                0.05,
                psa.as_mut_ptr(),
                null_mut(),
                null_mut(),
            ),
            ES_OK
        );
        assert!(psa.iter().all(|&v| v > 0.0));

        // Moment tensor strike/dip/rake + ISO/CLVD mix + beachball grid:
        let sdr = [203.0_f64, 10.0, 88.0, 3.9e22, 0.0, 0.0];
        let mut mt20 = [0.0_f64; 20];
        let gsize = if cfg!(miri) { 9 } else { 33 };
        let mut grid = vec![0.0_f64; gsize * gsize];
        assert_eq!(
            es_moment_tensor_decompose(
                sdr.as_ptr(),
                1,
                mt20.as_mut_ptr(),
                gsize,
                grid.as_mut_ptr(),
            ),
            ES_OK
        );
        assert!((mt20[1] - 100.0).abs() < 1e-6); // 100% DC
        assert!((mt20[4] - 9.0).abs() < 0.15); // Mw ~ 9.06
        let dc_t_az = mt20[11];
        let dc_p_az = mt20[14];

        // Mixed +15% ISO / 50% DC / -35% CLVD and strong -60% CLVD (preserving T/B/P eigenframe!):
        let sdr_neg_clvd = [203.0_f64, 10.0, 88.0, 3.9e22, 15.0, -35.0];
        assert_eq!(
            es_moment_tensor_decompose(sdr_neg_clvd.as_ptr(), 1, mt20.as_mut_ptr(), 0, null_mut(),),
            ES_OK
        );
        assert!((mt20[0] - 15.0).abs() < 1e-9, "got iso={}", mt20[0]);
        assert!((mt20[1] - 50.0).abs() < 1e-9, "got dc={}", mt20[1]);
        assert!((mt20[2] - (-35.0)).abs() < 1e-9, "got clvd={}", mt20[2]);
        assert!((mt20[11] - dc_t_az).abs() < 1e-6);
        assert!((mt20[14] - dc_p_az).abs() < 1e-6);

        let sdr_strong_neg = [203.0_f64, 10.0, 88.0, 3.9e22, 0.0, -60.0];
        assert_eq!(
            es_moment_tensor_decompose(
                sdr_strong_neg.as_ptr(),
                1,
                mt20.as_mut_ptr(),
                0,
                null_mut(),
            ),
            ES_OK
        );
        assert!((mt20[1] - 40.0).abs() < 1e-9);
        assert!((mt20[2] - (-60.0)).abs() < 1e-9);
        assert!((mt20[11] - dc_t_az).abs() < 1e-6);
        assert!((mt20[14] - dc_p_az).abs() < 1e-6);
    }
}

#[test]
fn miri_taup_engine_travel_times_paths_pierce_and_locator() {
    const COMPACT_ND: &[u8] = b"nd:\n\
0.0    5.80  3.36  2.72\n\
20.0   5.80  3.36  2.72\n\
conrad\n\
20.0   6.50  3.75  2.92\n\
35.0   6.50  3.75  2.92\n\
moho\n\
35.0   8.04  4.47  3.32\n\
410.0  9.03  4.87  3.54\n\
410.0  9.36  5.07  3.72\n\
660.0  10.20 5.60  3.99\n\
660.0  10.79 5.95  4.38\n\
2889.0 13.69 7.30  5.56\n\
outer-core\n\
2889.0 8.00  0.00  9.90\n\
5153.9 10.26 0.00  12.14\n\
inner-core\n\
5153.9 11.12 3.50  12.70\n\
6371.0 11.24 3.56  13.01\n";

    let model_spec: &[u8] = if cfg!(miri) { COMPACT_ND } else { b"iasp91" };

    unsafe {
        let bad = es_taup_engine_create(b"nonexistent".as_ptr(), 11);
        assert!(bad.is_null());

        let eng = es_taup_engine_create(model_spec.as_ptr(), model_spec.len());
        assert!(!eng.is_null());

        let mut tt_ptr: *mut u8 = null_mut();
        let mut tt_len: usize = 0;
        assert_eq!(
            es_taup_travel_times(
                eng,
                null(),
                0,
                29.0,
                73.87078776567722,
                b"P,S".as_ptr(),
                3,
                &mut tt_ptr,
                &mut tt_len,
            ),
            ES_OK
        );
        let tt = core::slice::from_raw_parts(tt_ptr, tt_len);
        assert_eq!(&tt[0..4], b"ESTT");
        let count = u32::from_le_bytes(tt[8..12].try_into().unwrap());
        assert!(count >= 2);
        let p_time = f64::from_le_bytes(tt[16..24].try_into().unwrap());
        let tol_p = if cfg!(miri) { 15.0 } else { 0.05 };
        assert!((p_time - 692.084).abs() < tol_p);
        es_free(tt_ptr, tt_len);

        let mut rp_ptr: *mut u8 = null_mut();
        let mut rp_len: usize = 0;
        assert_eq!(
            es_taup_ray_paths(
                eng,
                null(),
                0,
                29.0,
                73.87,
                b"P".as_ptr(),
                1,
                &mut rp_ptr,
                &mut rp_len,
            ),
            ES_OK
        );
        let rp = core::slice::from_raw_parts(rp_ptr, rp_len);
        assert_eq!(&rp[0..4], b"ESRP");
        es_free(rp_ptr, rp_len);

        let mut pp_ptr: *mut u8 = null_mut();
        let mut pp_len: usize = 0;
        assert_eq!(
            es_taup_pierce_points(
                eng,
                null(),
                0,
                29.0,
                73.87,
                b"P".as_ptr(),
                1,
                &mut pp_ptr,
                &mut pp_len,
            ),
            ES_OK
        );
        let pp = core::slice::from_raw_parts(pp_ptr, pp_len);
        assert_eq!(&pp[0..4], b"ESPP");
        es_free(pp_ptr, pp_len);

        es_taup_engine_free(eng);

        // Full hypocenter location test (skip heavy global seed grid under Miri interpreter for speed;
        // still test FFI validation under Miri and full inversion on native):
        if cfg!(miri) {
            let short_obs = [0.0_f64; 15];
            let mut out16 = [0.0_f64; 16];
            assert_eq!(
                es_locate_hypocenter(
                    b"iasp91".as_ptr(),
                    6,
                    short_obs.as_ptr(),
                    b"P,P,P".as_ptr(),
                    5,
                    3,
                    0,
                    out16.as_mut_ptr(),
                ),
                ES_ERR_INVALID_ARG
            );
        } else {
            let model = TauModel::by_name("iasp91").unwrap();
            let stations = [
                (42.0_f64, 143.0_f64, "P"),
                (35.0_f64, 139.0_f64, "P"),
                (38.0_f64, 128.0_f64, "P"),
                (50.0_f64, 155.0_f64, "P"),
                (35.0_f64, 139.0_f64, "S"),
            ];
            let ev_lat = 38.297;
            let ev_lon = 142.373;
            let ev_dep = 35.0;
            let t0 = 12.5;
            let mut obs_f64 = Vec::new();
            let mut phases_vec = Vec::new();
            for &(slat, slon, ph) in &stations {
                let arrs = model
                    .get_travel_times_geo(ev_lat, ev_lon, ev_dep, slat, slon, &[ph], false)
                    .unwrap();
                obs_f64.extend_from_slice(&[slat, slon, 0.0, t0 + arrs[0].time_s, 0.05]);
                phases_vec.push(ph);
            }
            let csv = phases_vec.join(",");
            let mut out16 = [0.0_f64; 16];
            assert_eq!(
                es_locate_hypocenter(
                    b"iasp91".as_ptr(),
                    6,
                    obs_f64.as_ptr(),
                    csv.as_ptr(),
                    csv.len(),
                    stations.len(),
                    0,
                    out16.as_mut_ptr(),
                ),
                ES_OK
            );
            assert!((out16[0] - ev_lat).abs() < 0.05);
            assert!((out16[1] - ev_lon).abs() < 0.05);
            assert!((out16[2] - ev_dep).abs() < 1.0);
            assert!((out16[3] - t0).abs() < 0.2);
        }
    }
}
