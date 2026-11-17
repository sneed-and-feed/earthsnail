//! Gate 5 Native `x86_64` Reference Harness paired with `studio/verify_wasm_lockstep.mjs`.
//!
//! Executes the exact same C-ABI entrypoints (`es_encode_steim2_frames`, `es_decode_steim2_frames`,
//! `es_wgs84_geodesic`, `es_sos_filter`, `es_rfft_psd`, `es_hilbert_envelope`, `es_nlnm_nhnm`,
//! `es_pick_arrivals`, `es_taup_travel_times`, `es_taup_ray_paths`) on the host target and, when
//! `EARTHSNAIL_LOCKSTEP_DUMP` is set, writes high-precision JSON reference outputs for head-to-head
//! comparison against `earthsnail_wasm.wasm` (`wasm32-unknown-unknown +simd128`).

use core::fmt::Write as _;
use core::ptr::{null, null_mut};

use earthsnail_wasm::*;

// Compact zero-dependency FIPS 180-4 SHA-256 for Steim-2 frame digest parity:
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, slot) in w.iter_mut().enumerate().take(16) {
            let b = i * 4;
            *slot = u32::from_be_bytes([chunk[b], chunk[b + 1], chunk[b + 2], chunk[b + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        let _ = write!(&mut out, "{word:08x}");
    }
    out
}

fn build_steim2_test_vector() -> Vec<i32> {
    let n = 512usize;
    let mut out = vec![0_i32; n];
    let mut cur = 25_000_i32;
    out[0] = cur;
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        let bucket = i % 8;
        let sign: i32 = if i % 2 == 0 { 1 } else { -1 };
        let step: i32 = match bucket {
            0 => 0,
            1 => sign * (((i % 7) as i32) + 1),
            2 => sign * (8 + ((i % 7) as i32)),
            3 => sign * (16 + ((i % 15) as i32)),
            4 => sign * (32 + ((i % 90) as i32)),
            5 => sign * (128 + ((i % 350) as i32)),
            6 => sign * (512 + (((i * 37) % 14_000) as i32)),
            _ => sign * (16_384 + (((i * 7_919) % 400_000) as i32)),
        };
        cur = cur.wrapping_add(step);
        *slot = cur;
    }
    out
}

const GEODESIC_PAIRS: [[f64; 4]; 6] = [
    [38.297, 142.373, -35.318715, 148.996325],
    [34.9459, -106.4572, 35.6895, 139.6917],
    [0.0, 0.0, 0.0, 179.6],
    [89.75, 15.0, 89.8, -165.0],
    [-45.1234, 72.5678, 45.1230, -107.4320],
    [37.7749, -122.4194, 37.7752, -122.4189],
];

fn build_deterministic_waveform(n: usize, fs: f64) -> Vec<f64> {
    let mut x = vec![0.0_f64; n];
    let mut state: u64 = 0x1234_5678_9abc_def0;
    for (i, slot) in x.iter_mut().enumerate() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((state >> 11) as f64) / 9_007_199_254_740_992.0;
        let noise = (u - 0.5) * 0.04;
        let t = (i as f64) / fs;
        let arrival = if i >= 640 {
            let dt = t - 640.0 / fs;
            let env = 1.0 / (1.0 + 0.15 * dt * dt);
            let phase1 = (dt * 0.85) % 1.0;
            let tri1 = 1.0 - 4.0 * (phase1 - 0.5).abs();
            let phase2 = (dt * 2.35) % 1.0;
            let tri2 = 1.0 - 4.0 * (phase2 - 0.5).abs();
            env * (3.2 * tri1 + 1.1 * tri2)
        } else {
            0.0
        };
        *slot = noise + arrival;
    }
    x
}

fn push_f64_json_array(out: &mut String, vals: &[f64]) {
    out.push('[');
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v:.17e}");
    }
    out.push(']');
}

fn push_i32_json_array(out: &mut String, vals: &[i32]) {
    out.push('[');
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v}");
    }
    out.push(']');
}

fn decode_cstr_padded(raw: &[u8]) -> &str {
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    core::str::from_utf8(&raw[..end]).unwrap_or("")
}

#[test]
fn native_lockstep_oracle_execution_and_dump() {
    let mut json = String::with_capacity(131_072);
    json.push('{');

    // 1. Steim-2 codec:
    let steim_in = build_steim2_test_vector();
    let (frames_sha, steim_decoded) = unsafe {
        let mut f_ptr: *mut u8 = null_mut();
        let mut f_len: usize = 0;
        assert_eq!(
            es_encode_steim2_frames(steim_in.as_ptr(), steim_in.len(), 0, &mut f_ptr, &mut f_len),
            ES_OK
        );
        let frame_slice = core::slice::from_raw_parts(f_ptr, f_len);
        let sha = sha256_hex(frame_slice);
        let mut dec = vec![0_i32; steim_in.len()];
        assert_eq!(
            es_decode_steim2_frames(f_ptr, f_len, steim_in.len(), dec.as_mut_ptr()),
            ES_OK
        );
        es_free(f_ptr, f_len);
        (sha, dec)
    };
    assert_eq!(steim_decoded, steim_in);
    let _ = write!(
        &mut json,
        "\"steim2\":{{\"frames_sha256\":\"{frames_sha}\",\"decoded_samples\":"
    );
    push_i32_json_array(&mut json, &steim_decoded);
    json.push_str("},");

    // 2. WGS84 geodesics:
    let mut geo_flat = Vec::with_capacity(GEODESIC_PAIRS.len() * 5);
    let mut out5 = [0.0_f64; 5];
    for [lat1, lon1, lat2, lon2] in GEODESIC_PAIRS {
        unsafe {
            assert_eq!(
                es_wgs84_geodesic(lat1, lon1, lat2, lon2, out5.as_mut_ptr()),
                ES_OK
            );
        }
        geo_flat.extend_from_slice(&out5);
    }
    json.push_str("\"geodesics_flat\":");
    push_f64_json_array(&mut json, &geo_flat);
    json.push(',');

    // 3. 4-pole zero-phase Butterworth bandpass:
    let n_wave = 1024usize;
    let fs_wave = 50.0_f64;
    let base_sig = build_deterministic_waveform(n_wave, fs_wave);
    let mut zp_obspy = vec![0.0_f64; n_wave];
    let mut zp_filtfilt = vec![0.0_f64; n_wave];
    unsafe {
        assert_eq!(
            es_sos_filter(
                base_sig.as_ptr(),
                n_wave,
                fs_wave,
                0,
                0,
                4,
                0.2,
                5.0,
                0.0,
                1,
                zp_obspy.as_mut_ptr(),
            ),
            ES_OK
        );
        assert_eq!(
            es_sos_filter(
                base_sig.as_ptr(),
                n_wave,
                fs_wave,
                0,
                0,
                4,
                0.2,
                5.0,
                0.0,
                2,
                zp_filtfilt.as_mut_ptr(),
            ),
            ES_OK
        );
    }
    json.push_str("\"butterworth_zp_obspy\":");
    push_f64_json_array(&mut json, &zp_obspy);
    json.push_str(",\"butterworth_filtfilt\":");
    push_f64_json_array(&mut json, &zp_filtfilt);
    json.push(',');

    // 4. Welch RFFT PSD, Hilbert envelope, Peterson NLNM/NHNM:
    let bins = 129usize;
    let mut p_freqs = vec![0.0_f64; bins];
    let mut p_vals = vec![0.0_f64; bins];
    let mut written = 0usize;
    let mut env_vals = vec![0.0_f64; n_wave];
    let periods = [
        0.1_f64, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0,
    ];
    let mut nlnm_vals = [0.0_f64; 12];
    let mut nhnm_vals = [0.0_f64; 12];
    unsafe {
        assert_eq!(
            es_rfft_psd(
                base_sig.as_ptr(),
                n_wave,
                fs_wave,
                256,
                128,
                p_freqs.as_mut_ptr(),
                p_vals.as_mut_ptr(),
                bins,
                &mut written,
            ),
            ES_OK
        );
        assert_eq!(written, bins);
        assert_eq!(
            es_hilbert_envelope(base_sig.as_ptr(), n_wave, env_vals.as_mut_ptr()),
            ES_OK
        );
        assert_eq!(
            es_nlnm_nhnm(
                periods.as_ptr(),
                periods.len(),
                nlnm_vals.as_mut_ptr(),
                nhnm_vals.as_mut_ptr(),
            ),
            ES_OK
        );
    }
    json.push_str("\"welch_psd\":");
    push_f64_json_array(&mut json, &p_vals);
    json.push_str(",\"hilbert_envelope\":");
    push_f64_json_array(&mut json, &env_vals);
    json.push_str(",\"nlnm_db\":");
    push_f64_json_array(&mut json, &nlnm_vals);
    json.push_str(",\"nhnm_db\":");
    push_f64_json_array(&mut json, &nhnm_vals);
    json.push(',');

    // 5. Pickers (STA/LTA vs Morse on 1-sample glitched signal + Baer on clean):
    let mut glitched = base_sig.clone();
    glitched[280] += 1.0e6;
    let mut sta_pick = -1_i64;
    let mut sta_peak = 0.0_f64;
    let mut morse_pick = -1_i64;
    let mut morse_peak = 0.0_f64;
    let mut morse_cf = vec![0.0_f64; n_wave];
    let mut baer_pick = -1_i64;
    let mut baer_peak = 0.0_f64;
    let mut baer_cf = vec![0.0_f64; n_wave];
    unsafe {
        assert_eq!(
            es_pick_arrivals(
                glitched.as_ptr(),
                n_wave,
                fs_wave,
                0,
                10.0,
                100.0,
                4.0,
                &mut sta_pick,
                &mut sta_peak,
                null_mut(),
            ),
            ES_OK
        );
        assert_eq!(
            es_pick_arrivals(
                glitched.as_ptr(),
                n_wave,
                fs_wave,
                4,
                15.0,
                120.0,
                4.5,
                &mut morse_pick,
                &mut morse_peak,
                morse_cf.as_mut_ptr(),
            ),
            ES_OK
        );
        assert_eq!(
            es_pick_arrivals(
                base_sig.as_ptr(),
                n_wave,
                fs_wave,
                2,
                7.0,
                12.0,
                20.0,
                &mut baer_pick,
                &mut baer_peak,
                baer_cf.as_mut_ptr(),
            ),
            ES_OK
        );
    }
    assert_eq!(sta_pick, 280);
    assert!((morse_pick - 640).abs() <= 15);
    let _ = write!(
        &mut json,
        "\"pickers\":{{\"sta_lta_glitched_pick\":{sta_pick},\"morse_glitched_pick\":{morse_pick},\"morse_peak_ratio\":{morse_peak:.17e},\"baer_clean_pick\":{baer_pick},\"morse_cf\":"
    );
    push_f64_json_array(&mut json, &morse_cf);
    json.push_str(",\"baer_cf\":");
    push_f64_json_array(&mut json, &baer_cf);
    json.push_str("},");

    // 6. TauP across `iasp91`, `ak135`, `prem`:
    json.push_str("\"taup\":{");
    let models = ["iasp91", "ak135", "prem"];
    let phases_str = b"P,pP,PP,PcP,S,SKS,PKIKP";
    let mut first_key = true;
    for model in models {
        unsafe {
            let eng = es_taup_engine_create(model.as_ptr(), model.len());
            assert!(!eng.is_null());
            for (depth_km, dist_deg, dist_str) in [
                (29.0_f64, 73.87078776567722_f64, "73.87078776567722"),
                (350.0_f64, 145.0_f64, "145"),
            ] {
                if !first_key {
                    json.push(',');
                }
                first_key = false;
                let depth_str = if (depth_km - 29.0).abs() == 0.0 {
                    "29"
                } else {
                    "350"
                };
                let _ = write!(
                    &mut json,
                    "\"{model}_{depth_str}_{dist_str}\":{{\"arrivals\":["
                );

                let mut tt_ptr: *mut u8 = null_mut();
                let mut tt_len: usize = 0;
                assert_eq!(
                    es_taup_travel_times(
                        eng,
                        null(),
                        0,
                        depth_km,
                        dist_deg,
                        phases_str.as_ptr(),
                        phases_str.len(),
                        &mut tt_ptr,
                        &mut tt_len,
                    ),
                    ES_OK
                );
                let tt = core::slice::from_raw_parts(tt_ptr, tt_len);
                let count = u32::from_le_bytes(tt[8..12].try_into().unwrap()) as usize;
                for i in 0..count {
                    if i > 0 {
                        json.push(',');
                    }
                    let off = 16 + i * 96;
                    let t_s = f64::from_le_bytes(tt[off..off + 8].try_into().unwrap());
                    let p_deg = f64::from_le_bytes(tt[off + 16..off + 24].try_into().unwrap());
                    let name = decode_cstr_padded(&tt[off + 64..off + 80]);
                    let _ = write!(
                        &mut json,
                        "{{\"name\":\"{name}\",\"time_s\":{t_s:.17e},\"ray_param_s_per_deg\":{p_deg:.17e}}}"
                    );
                }
                es_free(tt_ptr, tt_len);
                json.push_str("],\"paths\":[");

                let mut rp_ptr: *mut u8 = null_mut();
                let mut rp_len: usize = 0;
                assert_eq!(
                    es_taup_ray_paths(
                        eng,
                        null(),
                        0,
                        depth_km,
                        dist_deg,
                        phases_str.as_ptr(),
                        phases_str.len(),
                        &mut rp_ptr,
                        &mut rp_len,
                    ),
                    ES_OK
                );
                let rp = core::slice::from_raw_parts(rp_ptr, rp_len);
                let p_count = u32::from_le_bytes(rp[8..12].try_into().unwrap()) as usize;
                let mut off = 48usize;
                for i in 0..p_count {
                    if i > 0 {
                        json.push(',');
                    }
                    let t_s = f64::from_le_bytes(rp[off..off + 8].try_into().unwrap());
                    let n_pts =
                        u64::from_le_bytes(rp[off + 48..off + 56].try_into().unwrap()) as usize;
                    let name = decode_cstr_padded(&rp[off + 56..off + 72]);
                    off += 80;
                    let read_pt = |idx: usize| -> [f64; 4] {
                        let po = off + idx * 32;
                        [
                            f64::from_le_bytes(rp[po..po + 8].try_into().unwrap()),
                            f64::from_le_bytes(rp[po + 8..po + 16].try_into().unwrap()),
                            f64::from_le_bytes(rp[po + 16..po + 24].try_into().unwrap()),
                            f64::from_le_bytes(rp[po + 24..po + 32].try_into().unwrap()),
                        ]
                    };
                    let p0 = read_pt(0);
                    let pm = read_pt(n_pts / 2);
                    let pl = read_pt(n_pts - 1);
                    off += n_pts * 32;
                    let _ = write!(
                        &mut json,
                        "{{\"name\":\"{name}\",\"time_s\":{t_s:.17e},\"n_points\":{n_pts},\"first_pt\":"
                    );
                    push_f64_json_array(&mut json, &p0);
                    json.push_str(",\"mid_pt\":");
                    push_f64_json_array(&mut json, &pm);
                    json.push_str(",\"last_pt\":");
                    push_f64_json_array(&mut json, &pl);
                    json.push('}');
                }
                es_free(rp_ptr, rp_len);
                json.push_str("]}");
            }
            es_taup_engine_free(eng);
        }
    }
    json.push_str("}}");

    if let Ok(dump_path) = std::env::var("EARTHSNAIL_LOCKSTEP_DUMP") {
        if !dump_path.is_empty() {
            std::fs::write(&dump_path, &json).expect("failed to write lockstep dump JSON");
        }
    }
}
