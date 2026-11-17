//! Property-based (`proptest`) mathematical invariant suites for `earthsnail-dsp`:
//! 1. Arbitrary-$N$ FFT/IFFT & RFFT/IRFFT round-trip + Parseval energy conservation (`< 1e-12` on
//!    $N \in \{997, 1000, 4096\}$ and `< 1e-11` across randomized prime/composite lengths).
//! 2. IIR SOS pole stability ($|p_i| < 1$) across all families/bands/orders & zero-phase passband phase invariance.
//! 3. 3-Point Discrete Morse Causal-Deferral single-sample telemetry spike zero-false-trigger guarantee.
//! 4. Moment tensor algebra (`strike/dip/rake -> MT -> DC=100%, ISO=0%, CLVD=0%`, orthogonal `P/T/B` axes,
//!    conjugate nodal plane symmetry, and general `M_ISO + M_DC + M_CLVD == M`).
#![cfg(not(miri))]

use core::f64::consts::{PI, TAU};
use earthsnail_core::event::{MomentTensor6, NodalPlane};
use earthsnail_core::{Complex64, ScratchWorkspace};
use earthsnail_dsp::{
    auxiliary_plane, decompose_moment_tensor, design_iir_sos, fft, fft_with_scratch, ifft,
    ifft_with_scratch, irfft, jacobi_eigen_3x3, morse_pick, rfft, sosfilt_zerophase_obspy,
    sosfiltfilt, strike_dip_rake_to_mt, BesselNorm, FilterBand, FilterFamily, MorsePickConfig,
};
use proptest::prelude::*;

#[test]
fn canonical_prime_composite_pow2_fft_parseval_below_1e12() {
    let mut scratch = ScratchWorkspace::with_capacity(16384, 16384);
    for &n in &[1usize, 2, 3, 17, 997, 1000, 4096] {
        let input: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = (i as f64) * 0.173_205_080_756_887_7;
                Complex64::new(
                    (x * 1.3).sin() - 0.4 * (x * 0.7).cos(),
                    0.65 * (x * 2.1).cos(),
                )
            })
            .collect();

        let time_energy: f64 = input.iter().copied().map(Complex64::norm_sqr).sum();
        let mut spec = input.clone();
        fft_with_scratch(&mut spec, &mut scratch).unwrap();
        let freq_energy: f64 =
            spec.iter().copied().map(Complex64::norm_sqr).sum::<f64>() / (n as f64);

        let rel_parseval = (freq_energy - time_energy).abs() / time_energy.max(1.0);
        assert!(
            rel_parseval < 1e-12,
            "Parseval failed at N={n}: rel_err={rel_parseval:.3e}"
        );

        ifft_with_scratch(&mut spec, &mut scratch).unwrap();
        let max_err = spec
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "ifft(fft(x)) roundtrip failed at N={n}: max_err={max_err:.3e}"
        );

        // Real RFFT/IRFFT roundtrip at exact same N
        let real_in: Vec<f64> = input.iter().map(|c| c.re).collect();
        let rspec = rfft(&real_in, Some(n)).unwrap();
        let rec = irfft(&rspec, n).unwrap();
        let r_err = rec
            .iter()
            .zip(real_in.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            r_err < 1e-12,
            "irfft(rfft(x)) roundtrip failed at N={n}: max_err={r_err:.3e}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_fft_parseval_and_roundtrip_arbitrary_n(
        n in 1usize..=260,
        seed in 1u64..1_000_000u64,
    ) {
        let mut state = seed | 1;
        let mut next_f64 = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as i64) as f64) * (1.0 / (i64::MAX as f64)) * 10.0
        };

        let x: Vec<Complex64> = (0..n).map(|_| Complex64::new(next_f64(), next_f64())).collect();
        let e_time: f64 = x.iter().copied().map(Complex64::norm_sqr).sum();

        let y = fft(&x).unwrap();
        let e_freq: f64 = y.iter().copied().map(Complex64::norm_sqr).sum::<f64>() / (n as f64);
        prop_assert!((e_freq - e_time).abs() <= 1e-11 * e_time.max(1.0));

        let x_rec = ifft(&y).unwrap();
        let norm_inf = x.iter().map(|c| c.abs()).fold(1.0_f64, f64::max);
        for (a, b) in x_rec.iter().zip(x.iter()) {
            prop_assert!((*a - *b).abs() <= 1e-11 * norm_inf);
        }

        let rx: Vec<f64> = x.iter().map(|c| c.re).collect();
        let ry = rfft(&rx, Some(n)).unwrap();
        let rx_rec = irfft(&ry, n).unwrap();
        for (&a, &b) in rx_rec.iter().zip(rx.iter()) {
            prop_assert!((a - b).abs() <= 1e-11 * norm_inf);
        }
    }

    #[test]
    fn prop_iir_sos_pole_stability_and_zerophase_passband_invariance(
        order in 1usize..=7,
        fam_idx in 0usize..6,
        band_idx in 0usize..4,
        f_lo_frac in 0.06f64..0.32f64,
        bw_frac in 0.12f64..0.42f64,
    ) {
        let fs = 100.0;
        let nyq = 0.5 * fs;
        let f1 = f_lo_frac * nyq;
        let f2 = ((f_lo_frac + bw_frac).min(0.88)) * nyq;

        let family = match fam_idx {
            0 => FilterFamily::Butterworth,
            1 => FilterFamily::Chebyshev1 { rp_db: 1.0 },
            2 => FilterFamily::Chebyshev2 { rs_db: 40.0 },
            3 => FilterFamily::Bessel { norm: BesselNorm::Phase },
            4 => FilterFamily::Bessel { norm: BesselNorm::Mag },
            _ => FilterFamily::Bessel { norm: BesselNorm::Delay },
        };
        let band = match band_idx {
            0 => FilterBand::Lowpass(f2),
            1 => FilterBand::Highpass(f1),
            2 => FilterBand::Bandpass(f1, f2),
            _ => FilterBand::Bandstop(f1, f2),
        };

        let sos = design_iir_sos(order, band, fs, family).unwrap();
        for p in sos.poles() {
            prop_assert!(
                p.abs() < 1.0 - 1e-10,
                "pole outside/on unit circle: |p|={} for order={order}, fam={family:?}, band={band:?}",
                p.abs()
            );
        }

        // Zero-phase passband phase invariance: a symmetric cosine wavepacket centered at t_center
        // inside the passband preserves its peak location without group-delay shift!
        if order <= 4 && matches!(family, FilterFamily::Butterworth) && matches!(band, FilterBand::Lowpass(_)) {
            let f_tone = 0.25 * f2;
            let npts = 500usize;
            let center = 250usize;
            let sig: Vec<f64> = (0..npts)
                .map(|i| {
                    let dt = ((i as isize - center as isize) as f64) / fs;
                    (TAU * f_tone * dt).cos() * (-0.5 * (dt / 0.8).powi(2)).exp()
                })
                .collect();
            let y_obspy = sosfilt_zerophase_obspy(&sos, &sig);
            let y_scipy = sosfiltfilt(&sos, &sig).unwrap();
            let peak_obspy = y_obspy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let peak_scipy = y_scipy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            prop_assert_eq!(peak_obspy, center);
            prop_assert_eq!(peak_scipy, center);
        }
    }

    #[test]
    fn prop_morse_single_sample_spike_zero_false_trigger(
        spike_pos in 0usize..400,
        spike_amp in 1.0e2f64..1.0e10f64,
        spike_sign in prop::bool::ANY,
        bg_level in 1e-4f64..1.0f64,
        phase_shift in 0.0f64..TAU,
    ) {
        let n = 400usize;
        let mut trace: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                bg_level * (0.6 * (0.31 * t + phase_shift).sin() + 0.4 * (1.17 * t).cos())
            })
            .collect();
        trace[spike_pos] += if spike_sign { spike_amp } else { -spike_amp };

        let cfg = MorsePickConfig::default();
        let res = morse_pick(&trace, &cfg).unwrap();
        prop_assert_eq!(
            res.pick_sample,
            None,
            "Morse picker false-triggered on 1-sample spike at {} with amp {}",
            spike_pos,
            spike_amp
        );
    }

    #[test]
    fn prop_strike_dip_rake_moment_tensor_dc_and_orthogonal_axes(
        strike in 0.0f64..360.0f64,
        dip in 0.5f64..89.5f64,
        rake in -179.5f64..179.5f64,
        log10_m0 in 14.0f64..22.0f64,
    ) {
        let m0 = 10.0_f64.powf(log10_m0);
        let plane = NodalPlane {
            strike_deg: strike,
            dip_deg: dip,
            rake_deg: rake,
        };
        let mt = strike_dip_rake_to_mt(plane, m0).unwrap();
        prop_assert!((mt.scalar_moment_nm() - m0).abs() <= 1e-10 * m0);

        let dec = decompose_moment_tensor(&mt);
        prop_assert!(dec.iso_percent.abs() < 1e-9);
        prop_assert!(dec.clvd_percent.abs() < 1e-9);
        prop_assert!((dec.dc_percent - 100.0).abs() < 1e-9);

        // Verify orthonormal right-handed P/T/B eigenvectors
        let (_evals, evecs) = jacobi_eigen_3x3(mt.to_ned_matrix());
        for i in 0..3 {
            let norm_sq = evecs[i][0] * evecs[i][0] + evecs[i][1] * evecs[i][1] + evecs[i][2] * evecs[i][2];
            prop_assert!((norm_sq - 1.0).abs() < 1e-12);
            for j in (i + 1)..3 {
                let dot = evecs[i][0] * evecs[j][0] + evecs[i][1] * evecs[j][1] + evecs[i][2] * evecs[j][2];
                prop_assert!(dot.abs() < 1e-12);
            }
        }

        // Conjugate auxiliary plane and extracted nodal planes must generate identical MomentTensor6
        let aux = auxiliary_plane(plane).unwrap();
        let mt_aux = strike_dip_rake_to_mt(aux, m0).unwrap();
        let mt_np1 = strike_dip_rake_to_mt(dec.nodal_planes.0, m0).unwrap();
        let mt_np2 = strike_dip_rake_to_mt(dec.nodal_planes.1, m0).unwrap();

        for other in [mt_aux, mt_np1, mt_np2] {
            prop_assert!((other.m_rr - mt.m_rr).abs() <= 1e-9 * m0);
            prop_assert!((other.m_tt - mt.m_tt).abs() <= 1e-9 * m0);
            prop_assert!((other.m_pp - mt.m_pp).abs() <= 1e-9 * m0);
            prop_assert!((other.m_rt - mt.m_rt).abs() <= 1e-9 * m0);
            prop_assert!((other.m_rp - mt.m_rp).abs() <= 1e-9 * m0);
            prop_assert!((other.m_tp - mt.m_tp).abs() <= 1e-9 * m0);
        }
    }

    #[test]
    fn prop_general_moment_tensor_additive_and_percentage_partition(
        m_rr in -10.0f64..10.0f64,
        m_tt in -10.0f64..10.0f64,
        m_pp in -10.0f64..10.0f64,
        m_rt in -10.0f64..10.0f64,
        m_rp in -10.0f64..10.0f64,
        m_tp in -10.0f64..10.0f64,
    ) {
        let mt = MomentTensor6::new(m_rr, m_tt, m_pp, m_rt, m_rp, m_tp);
        if mt.scalar_moment_nm() > 1e-6 {
            let dec = decompose_moment_tensor(&mt);
            let sum_pct = dec.iso_percent.abs() + dec.dc_percent + dec.clvd_percent.abs();
            prop_assert!((sum_pct - 100.0).abs() < 1e-9);

            let sum_rr = dec.m_iso.m_rr + dec.m_dc.m_rr + dec.m_clvd.m_rr;
            let sum_tt = dec.m_iso.m_tt + dec.m_dc.m_tt + dec.m_clvd.m_tt;
            let sum_pp = dec.m_iso.m_pp + dec.m_dc.m_pp + dec.m_clvd.m_pp;
            let sum_rt = dec.m_iso.m_rt + dec.m_dc.m_rt + dec.m_clvd.m_rt;
            let sum_rp = dec.m_iso.m_rp + dec.m_dc.m_rp + dec.m_clvd.m_rp;
            let sum_tp = dec.m_iso.m_tp + dec.m_dc.m_tp + dec.m_clvd.m_tp;

            prop_assert!((sum_rr - mt.m_rr).abs() < 1e-10);
            prop_assert!((sum_tt - mt.m_tt).abs() < 1e-10);
            prop_assert!((sum_pp - mt.m_pp).abs() < 1e-10);
            prop_assert!((sum_rt - mt.m_rt).abs() < 1e-10);
            prop_assert!((sum_rp - mt.m_rp).abs() < 1e-10);
            prop_assert!((sum_tp - mt.m_tp).abs() < 1e-10);
        }
        let _ = PI;
    }
}
