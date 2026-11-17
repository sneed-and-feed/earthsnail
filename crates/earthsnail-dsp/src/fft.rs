//! Arbitrary-length $N \ge 1$ Fast Fourier Transform (`FFT` / `IFFT` / `RFFT` / `IRFFT`)
//! combining in-place radix-2/4 Cooley–Tukey for powers of two with exact Bluestein chirp-Z
//! spectral convolution for prime and arbitrary composite lengths (`0` external dependencies).

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
use earthsnail_core::{math, Complex64, EarthSnailError, Result, ScratchWorkspace};

/// Smallest power of two $\ge n$ (`1` when `n <= 1`).
#[inline]
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Largest power of two $\le n$ (`0` when `n == 0`).
#[inline]
pub fn prev_pow2(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        1_usize << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Returns the largest prime factor of `x` (`1` when `x <= 1`).
pub fn largest_prime_factor(mut x: usize) -> usize {
    if x <= 1 {
        return 1;
    }
    let mut largest = 1;
    while (x & 1) == 0 {
        largest = 2;
        x >>= 1;
    }
    let mut d = 3_usize;
    while d.saturating_mul(d) <= x {
        while x % d == 0 {
            largest = d;
            x /= d;
        }
        d += 2;
    }
    if x > 1 {
        largest = x;
    }
    largest
}

/// Computes zero-padded even FFT length $\ge 2 \cdot \text{npts}$ avoiding large prime factors
/// (`> 500`), matching ObsPy's `obspy.signal.util._npts2nfft(npts, smart=True)` convention.
pub fn npts2nfft(npts: usize) -> usize {
    if npts == 0 {
        return 0;
    }
    let base = if (npts & 1) != 0 {
        2 * (npts + 1)
    } else {
        2 * npts
    };
    if base > 5_000 && largest_prime_factor(base) >= 500 {
        for i in 1..=10 {
            let trial = base + 2 * i;
            if largest_prime_factor(trial) < 500 {
                return trial;
            }
        }
        return next_pow2(base);
    }
    base
}

/// Frequency bins `[0, df, 2*df, ..., fs/2]` (length `nfft / 2 + 1`) of a real-valued `rfft`.
pub fn rfftfreq(nfft: usize, dt_s: f64) -> Result<Vec<f64>> {
    if nfft == 0 {
        return Err(EarthSnailError::EmptyData(
            "rfftfreq requires nfft >= 1".into(),
        ));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "rfftfreq requires finite positive dt_s".into(),
        ));
    }
    let n_bins = nfft / 2 + 1;
    let inv_span = 1.0 / ((nfft as f64) * dt_s);
    let mut out = Vec::with_capacity(n_bins);
    for k in 0..n_bins {
        out.push((k as f64) * inv_span);
    }
    Ok(out)
}

/// Full DFT frequency bins (length `nfft`) matching `numpy.fft.fftfreq(nfft, d=dt_s)`.
pub fn fftfreq(nfft: usize, dt_s: f64) -> Result<Vec<f64>> {
    if nfft == 0 {
        return Err(EarthSnailError::EmptyData(
            "fftfreq requires nfft >= 1".into(),
        ));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "fftfreq requires finite positive dt_s".into(),
        ));
    }
    let inv_span = 1.0 / ((nfft as f64) * dt_s);
    let pos_end = nfft.div_ceil(2);
    let mut out = Vec::with_capacity(nfft);
    for k in 0..pos_end {
        out.push((k as f64) * inv_span);
    }
    let neg_count = nfft - pos_end;
    for k in 0..neg_count {
        let idx = (neg_count - k) as f64;
        out.push(-idx * inv_span);
    }
    Ok(out)
}

/// Forward complex DFT $X[k] = \sum_{n=0}^{N-1} x[n]\,e^{-i 2\pi k n / N}$ for any length $N \ge 1$.
pub fn fft(input: &[Complex64]) -> Result<Vec<Complex64>> {
    if input.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "fft input slice is empty".into(),
        ));
    }
    let mut out = input.to_vec();
    let mut scratch = ScratchWorkspace::new();
    fft_with_scratch(&mut out, &mut scratch)?;
    Ok(out)
}

/// Inverse complex DFT $x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,e^{+i 2\pi k n / N}$ for any length $N \ge 1$.
pub fn ifft(input: &[Complex64]) -> Result<Vec<Complex64>> {
    if input.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "ifft input slice is empty".into(),
        ));
    }
    let mut out = input.to_vec();
    let mut scratch = ScratchWorkspace::new();
    ifft_with_scratch(&mut out, &mut scratch)?;
    Ok(out)
}

/// In-place forward complex FFT for arbitrary $N \ge 1$.
pub fn fft_in_place(buf: &mut [Complex64]) -> Result<()> {
    let mut scratch = ScratchWorkspace::new();
    fft_with_scratch(buf, &mut scratch)
}

/// In-place inverse complex IFFT for arbitrary $N \ge 1$.
pub fn ifft_in_place(buf: &mut [Complex64]) -> Result<()> {
    let mut scratch = ScratchWorkspace::new();
    ifft_with_scratch(buf, &mut scratch)
}

/// In-place forward complex FFT reusing caller-owned `ScratchWorkspace` (zero heap churn once warm).
pub fn fft_with_scratch(buf: &mut [Complex64], scratch: &mut ScratchWorkspace) -> Result<()> {
    let n = buf.len();
    if n == 0 {
        return Err(EarthSnailError::EmptyData("fft buffer is empty".into()));
    }
    if n == 1 {
        return Ok(());
    }
    if n.is_power_of_two() {
        fft_pow2_in_place(buf, false);
        return Ok(());
    }
    bluestein_fft_in_place(buf, false, scratch);
    Ok(())
}

/// In-place inverse complex IFFT reusing caller-owned `ScratchWorkspace` (zero heap churn once warm).
pub fn ifft_with_scratch(buf: &mut [Complex64], scratch: &mut ScratchWorkspace) -> Result<()> {
    let n = buf.len();
    if n == 0 {
        return Err(EarthSnailError::EmptyData("ifft buffer is empty".into()));
    }
    if n == 1 {
        return Ok(());
    }
    if n.is_power_of_two() {
        fft_pow2_in_place(buf, true);
        let inv_n = 1.0 / (n as f64);
        for z in buf.iter_mut() {
            *z *= inv_n;
        }
        return Ok(());
    }
    bluestein_fft_in_place(buf, true, scratch);
    let inv_n = 1.0 / (n as f64);
    for z in buf.iter_mut() {
        *z *= inv_n;
    }
    Ok(())
}

/// One-sided real-input FFT returning `nfft / 2 + 1` Hermitian bins (zero-padding or truncating `real` to `nfft`).
pub fn rfft(real: &[f64], nfft: Option<usize>) -> Result<Vec<Complex64>> {
    let n = nfft.unwrap_or(real.len());
    if n == 0 {
        return Err(EarthSnailError::EmptyData("rfft requires nfft >= 1".into()));
    }
    let mut out = Vec::with_capacity(n / 2 + 1);
    let mut scratch = ScratchWorkspace::new();
    rfft_with_scratch(real, n, &mut out, &mut scratch)?;
    Ok(out)
}

/// Scratch-reusing one-sided real-input FFT writing `nfft / 2 + 1` complex bins into `out`.
///
/// For even `nfft = 2 * M`, packs even/odd real samples into a half-length $M$-point complex FFT
/// and unpacks the Hermitian spectrum in $O(M)$ without allocating temporary vectors.
pub fn rfft_with_scratch(
    real: &[f64],
    nfft: usize,
    out: &mut Vec<Complex64>,
    scratch: &mut ScratchWorkspace,
) -> Result<()> {
    if nfft == 0 {
        return Err(EarthSnailError::EmptyData("rfft requires nfft >= 1".into()));
    }
    let n_bins = nfft / 2 + 1;
    out.clear();
    out.resize(n_bins, Complex64::ZERO);

    if nfft == 1 {
        let x0 = real.first().copied().unwrap_or(0.0);
        out[0] = Complex64::new(x0, 0.0);
        return Ok(());
    }

    if (nfft & 1) == 0 && (nfft / 2).is_power_of_two() {
        // Fast half-length power-of-two real FFT unpack!
        let m = nfft / 2;
        let zbuf = scratch.complex_zeroed(m);
        let copy_pairs = (real.len() / 2).min(m);
        for k in 0..copy_pairs {
            zbuf[k] = Complex64::new(real[2 * k], real[2 * k + 1]);
        }
        if copy_pairs < m && (2 * copy_pairs) < real.len() {
            zbuf[copy_pairs] = Complex64::new(real[2 * copy_pairs], 0.0);
        }
        fft_pow2_in_place(zbuf, false);

        let z0 = zbuf[0];
        out[0] = Complex64::new(z0.re + z0.im, 0.0);
        out[m] = Complex64::new(z0.re - z0.im, 0.0);

        for k in 1..m {
            let zk = zbuf[k];
            let z_conj = zbuf[m - k].conj();
            let even = (zk + z_conj) * 0.5;
            let odd_i = (zk - z_conj) * 0.5; // i * Odd[k]
                                             // -i * odd_i = odd_i.im - i * odd_i.re = Odd[k]
            let odd = Complex64::new(odd_i.im, -odd_i.re);
            let tw = Complex64::cis(-PI * (k as f64) / (m as f64));
            out[k] = even + tw * odd;
        }
        return Ok(());
    }

    // General arbitrary-N real FFT via Bluestein / power-of-two complex FFT:
    let mut full = vec![Complex64::ZERO; nfft];
    let copy_len = real.len().min(nfft);
    for i in 0..copy_len {
        full[i] = Complex64::new(real[i], 0.0);
    }
    fft_with_scratch(&mut full, scratch)?;
    out.copy_from_slice(&full[..n_bins]);
    out[0].im = 0.0;
    if (nfft & 1) == 0 {
        out[n_bins - 1].im = 0.0;
    }
    Ok(())
}

/// Inverse real-output FFT from `nfft / 2 + 1` Hermitian bins back to `nfft` real samples,
/// matching `numpy.fft.irfft(spec, n=nfft)`.
pub fn irfft(spec: &[Complex64], nfft: usize) -> Result<Vec<f64>> {
    if nfft == 0 {
        return Err(EarthSnailError::EmptyData(
            "irfft requires nfft >= 1".into(),
        ));
    }
    let mut out = Vec::with_capacity(nfft);
    let mut scratch = ScratchWorkspace::new();
    irfft_with_scratch(spec, nfft, &mut out, &mut scratch)?;
    Ok(out)
}

/// Scratch-reusing inverse real-output FFT writing `nfft` samples into `out`.
pub fn irfft_with_scratch(
    spec: &[Complex64],
    nfft: usize,
    out: &mut Vec<f64>,
    scratch: &mut ScratchWorkspace,
) -> Result<()> {
    if nfft == 0 {
        return Err(EarthSnailError::EmptyData(
            "irfft requires nfft >= 1".into(),
        ));
    }
    let expected_bins = nfft / 2 + 1;
    if spec.len() < expected_bins {
        return Err(EarthSnailError::InvalidParameter(alloc::format!(
            "irfft(nfft={nfft}) requires at least {expected_bins} bins, got {}",
            spec.len()
        )));
    }
    out.clear();
    out.resize(nfft, 0.0);

    if nfft == 1 {
        out[0] = spec[0].re;
        return Ok(());
    }

    if (nfft & 1) == 0 && (nfft / 2).is_power_of_two() {
        // Fast half-length power-of-two Hermitian inverse unpack!
        let m = nfft / 2;
        let zbuf = scratch.complex_zeroed(m);
        let x0 = spec[0].re;
        let xm = spec[m].re;
        zbuf[0] = Complex64::new(0.5 * (x0 + xm), 0.5 * (x0 - xm));

        for k in 1..m {
            let xk = spec[k];
            let x_mirror = spec[m - k].conj();
            let even = (xk + x_mirror) * 0.5;
            let diff = (xk - x_mirror) * 0.5;
            let tw = Complex64::cis(PI * (k as f64) / (m as f64));
            let i_odd = Complex64::new(-diff.im, diff.re) * tw;
            zbuf[k] = even + i_odd;
        }
        fft_pow2_in_place(zbuf, true);
        let scale = 1.0 / (m as f64);
        for k in 0..m {
            out[2 * k] = zbuf[k].re * scale;
            out[2 * k + 1] = zbuf[k].im * scale;
        }
        return Ok(());
    }

    // General arbitrary-N Hermitian reconstruction:
    let mut full = vec![Complex64::ZERO; nfft];
    full[0] = Complex64::new(spec[0].re, 0.0);
    let last_unique = if (nfft & 1) == 0 {
        full[nfft / 2] = Complex64::new(spec[nfft / 2].re, 0.0);
        nfft / 2
    } else {
        nfft.div_ceil(2)
    };
    for k in 1..last_unique {
        full[k] = spec[k];
        full[nfft - k] = spec[k].conj();
    }
    ifft_with_scratch(&mut full, scratch)?;
    for i in 0..nfft {
        out[i] = full[i].re;
    }
    Ok(())
}

/// In-place mixed radix-4 / radix-2 decimation-in-time Cooley–Tukey FFT for $N = 2^p \ge 2$
/// (unnormalized; caller divides by $N$ on inverse).
fn fft_pow2_in_place(buf: &mut [Complex64], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    // 1. Bit-reversal permutation in-place:
    let shift = usize::BITS - n.trailing_zeros();
    for i in 1..(n - 1) {
        let j = i.reverse_bits() >> shift;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };

    // 2. Execute radix-2 stage first if log2(n) is odd, leaving an even number of stages for radix-4!
    let log2_n = n.trailing_zeros();
    let mut len = if (log2_n & 1) == 1 {
        for base in (0..n).step_by(2) {
            let u = buf[base];
            let v = buf[base + 1];
            buf[base] = u + v;
            buf[base + 1] = u - v;
        }
        2_usize
    } else {
        1_usize
    };

    // 3. Cascaded radix-4 stages (`quarter = len`, `block = 4 * len`):
    while len < n {
        let q = len;
        let block = q << 2;
        let theta_step = sign * TAU / (block as f64);

        // Precompute exact twiddle step + periodic refresh to guarantee < 1 ULP drift
        let w_step = Complex64::cis(theta_step);

        for base in (0..n).step_by(block) {
            let mut w1 = Complex64::ONE;
            for j in 0..q {
                if (j & 31) == 0 && j > 0 {
                    w1 = Complex64::cis(theta_step * (j as f64));
                }
                let w2 = w1 * w1;
                let w3 = w2 * w1;

                // Under radix-2 bit-reversal, quarter-subblocks in `[base .. base + 4*q]`
                // store time-decimated cosets `(4m + 0, 4m + 2, 4m + 1, 4m + 3)` in that order!
                let x0 = buf[base + j];
                let x2 = buf[base + j + q] * w2;
                let x1 = buf[base + j + 2 * q] * w1;
                let x3 = buf[base + j + 3 * q] * w3;

                let t0 = x0 + x2;
                let t1 = x0 - x2;
                let t2 = x1 + x3;
                let diff13 = x1 - x3;
                // Multiply diff13 by (-i for forward, +i for inverse):
                // (-i)*(re + i*im) = im - i*re; (+i)*(re + i*im) = -im + i*re
                let t3 = if inverse {
                    Complex64::new(-diff13.im, diff13.re)
                } else {
                    Complex64::new(diff13.im, -diff13.re)
                };

                buf[base + j] = t0 + t2;
                buf[base + j + q] = t1 + t3;
                buf[base + j + 2 * q] = t0 - t2;
                buf[base + j + 3 * q] = t1 - t3;

                w1 *= w_step;
            }
        }
        len = block;
    }
}

/// Bluestein chirp-Z algorithm for arbitrary prime or composite $N \ge 2$ using power-of-two
/// circular convolution of length $M = \operatorname{next\_pow2}(2N - 1)$, with exact `u128`
/// modular phase reduction `(k * k) % (2 * N)` to prevent mantissa loss at large $N$.
fn bluestein_fft_in_place(buf: &mut [Complex64], inverse: bool, scratch: &mut ScratchWorkspace) {
    let n = buf.len();
    let m = next_pow2(2 * n - 1);
    let sign = if inverse { 1.0 } else { -1.0 };
    let two_n = (2 * n) as u128;
    let pi_over_n = PI / (n as f64);

    let (_, a_buf, b_buf, _) = scratch.split_fft_workspace(0, m, m, 0);

    // Populate chirp sequence w_k = exp(sign * i * pi * (k^2 mod 2N) / N):
    for k in 0..n {
        let k128 = k as u128;
        let rem = ((k128 * k128) % two_n) as f64;
        let angle = sign * pi_over_n * rem;
        let (s, c) = math::sin_cos(angle);
        let chirp = Complex64::new(c, s);
        let chirp_conj = Complex64::new(c, -s);

        a_buf[k] = buf[k] * chirp;
        b_buf[k] = chirp_conj;
        if k > 0 {
            b_buf[m - k] = chirp_conj;
        }
    }

    fft_pow2_in_place(a_buf, false);
    fft_pow2_in_place(b_buf, false);

    for i in 0..m {
        a_buf[i] *= b_buf[i];
    }

    fft_pow2_in_place(a_buf, true);
    let inv_m = 1.0 / (m as f64);

    for (k, out_slot) in buf.iter_mut().enumerate() {
        let k128 = k as u128;
        let rem = ((k128 * k128) % two_n) as f64;
        let angle = sign * pi_over_n * rem;
        let chirp = Complex64::cis(angle);
        *out_slot = (a_buf[k] * inv_m) * chirp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip_and_parseval_prime_composite_pow2() {
        let tol = if cfg!(miri) { 5e-12 } else { 1e-12 };
        let lengths = if cfg!(miri) {
            alloc::vec![1_usize, 2, 7, 29, 30, 64]
        } else {
            alloc::vec![1_usize, 2, 3, 7, 64, 997, 1000, 4096]
        };

        let mut scratch = ScratchWorkspace::new();
        for n in lengths {
            let orig: Vec<Complex64> = (0..n)
                .map(|i| {
                    let t = (i as f64) * 0.173_205_08;
                    Complex64::new(
                        math::sin(t) + 0.35 * math::cos(3.7 * t),
                        -0.6 * math::cos(1.9 * t) + 0.15 * math::sin(5.1 * t),
                    )
                })
                .collect();

            let mut work = orig.clone();
            fft_with_scratch(&mut work, &mut scratch).unwrap();

            // Verify Parseval's identity: sum |x[n]|^2 == (1/N) * sum |X[k]|^2
            let e_time: f64 = orig.iter().map(|z| z.norm_sqr()).sum();
            let e_freq: f64 = work.iter().map(|z| z.norm_sqr()).sum::<f64>() / (n as f64);
            let rel_parseval = (e_time - e_freq).abs() / e_time.max(1.0);
            assert!(
                rel_parseval < tol,
                "Parseval mismatch for N={n}: rel_parseval={rel_parseval}"
            );

            // Direct O(N^2) DFT spot-check on bin k = n / 3
            let k_spot = n / 3;
            let expected_bin: Complex64 = orig
                .iter()
                .enumerate()
                .map(|(idx, &x)| {
                    x * Complex64::cis(-TAU * (k_spot as f64) * (idx as f64) / (n as f64))
                })
                .sum();
            let bin_err = (work[k_spot] - expected_bin).abs() / (n as f64).sqrt();
            assert!(
                bin_err < tol,
                "DFT bin {k_spot} mismatch at N={n}: {bin_err}"
            );

            // Inverse FFT round-trip
            ifft_with_scratch(&mut work, &mut scratch).unwrap();
            for i in 0..n {
                let diff = (work[i] - orig[i]).abs();
                assert!(
                    diff < tol,
                    "IFFT roundtrip mismatch at N={n}, i={i}: {diff}"
                );
            }
        }
    }

    #[test]
    fn rfft_and_irfft_roundtrip_even_odd_and_prime() {
        let tol = if cfg!(miri) { 5e-12 } else { 1e-12 };
        let lengths = if cfg!(miri) {
            alloc::vec![1_usize, 2, 16, 29, 30]
        } else {
            alloc::vec![1_usize, 2, 17, 128, 997, 1000, 4096]
        };
        for n in lengths {
            let x: Vec<f64> = (0..n)
                .map(|i| math::cos(0.31 * (i as f64)) - 0.45 * math::sin(1.17 * (i as f64)) + 0.25)
                .collect();
            let spec = rfft(&x, None).unwrap();
            assert_eq!(spec.len(), n / 2 + 1);
            let rec = irfft(&spec, n).unwrap();
            assert_eq!(rec.len(), n);
            for i in 0..n {
                assert!(
                    (rec[i] - x[i]).abs() < tol,
                    "rfft/irfft mismatch at N={n}, i={i}: got {} vs {}",
                    rec[i],
                    x[i]
                );
            }
        }

        // Verify ObsPy _npts2nfft reference examples from docstring!
        assert_eq!(npts2nfft(4096), 8192);
        assert_eq!(npts2nfft(1_800_028), 3_600_056);
        assert_eq!(npts2nfft(1_800_029), 4_194_304);
        assert_eq!(npts2nfft(1_800_031), 3_600_082);
    }
}
