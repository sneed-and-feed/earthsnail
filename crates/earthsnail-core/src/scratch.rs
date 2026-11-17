//! Caller-owned reusable workspace buffers (`ScratchBuffer` / `ScratchWorkspace`) eliminating
//! heap allocations inside hot DSP, FFT, filter, deconvolution, and ray-tracing loops.

use crate::complex::Complex64;
use alloc::vec::Vec;

/// Reusable scratch memory workspace holding pre-allocated `f64`, `Complex64`, and `usize` buffers.
#[derive(Debug, Clone, Default)]
pub struct ScratchBuffer {
    real_a: Vec<f64>,
    real_b: Vec<f64>,
    real_c: Vec<f64>,
    complex_a: Vec<Complex64>,
    complex_b: Vec<Complex64>,
    complex_c: Vec<Complex64>,
    indices: Vec<usize>,
}

/// Alias matching `ScratchWorkspace` naming in the EarthSnail architecture specification.
pub type ScratchWorkspace = ScratchBuffer;

impl ScratchBuffer {
    /// Creates an empty `ScratchBuffer` with zero initial allocation.
    pub const fn new() -> Self {
        Self {
            real_a: Vec::new(),
            real_b: Vec::new(),
            real_c: Vec::new(),
            complex_a: Vec::new(),
            complex_b: Vec::new(),
            complex_c: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Pre-allocates capacity for primary `f64` and `Complex64` workspaces.
    pub fn with_capacity(real_len: usize, complex_len: usize) -> Self {
        Self {
            real_a: Vec::with_capacity(real_len),
            real_b: Vec::with_capacity(real_len),
            real_c: Vec::new(),
            complex_a: Vec::with_capacity(complex_len),
            complex_b: Vec::with_capacity(complex_len),
            complex_c: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Returns a zero-initialized mutable `f64` slice of length `len`, reusing existing capacity.
    pub fn real_zeroed(&mut self, len: usize) -> &mut [f64] {
        resize_and_zero(&mut self.real_a, len, 0.0)
    }

    /// Returns two disjoint zero-initialized mutable `f64` slices `(a, b)` simultaneously.
    pub fn real_pair_zeroed(&mut self, len_a: usize, len_b: usize) -> (&mut [f64], &mut [f64]) {
        (
            resize_and_zero(&mut self.real_a, len_a, 0.0),
            resize_and_zero(&mut self.real_b, len_b, 0.0),
        )
    }

    /// Returns three disjoint zero-initialized mutable `f64` slices `(a, b, c)` simultaneously.
    pub fn real_triple_zeroed(
        &mut self,
        len_a: usize,
        len_b: usize,
        len_c: usize,
    ) -> (&mut [f64], &mut [f64], &mut [f64]) {
        (
            resize_and_zero(&mut self.real_a, len_a, 0.0),
            resize_and_zero(&mut self.real_b, len_b, 0.0),
            resize_and_zero(&mut self.real_c, len_c, 0.0),
        )
    }

    /// Returns a zero-initialized mutable `Complex64` slice of length `len`, reusing existing capacity.
    pub fn complex_zeroed(&mut self, len: usize) -> &mut [Complex64] {
        resize_and_zero(&mut self.complex_a, len, Complex64::ZERO)
    }

    /// Returns two disjoint zero-initialized mutable `Complex64` slices `(a, b)` simultaneously.
    pub fn complex_pair_zeroed(
        &mut self,
        len_a: usize,
        len_b: usize,
    ) -> (&mut [Complex64], &mut [Complex64]) {
        (
            resize_and_zero(&mut self.complex_a, len_a, Complex64::ZERO),
            resize_and_zero(&mut self.complex_b, len_b, Complex64::ZERO),
        )
    }

    /// Returns three disjoint zero-initialized mutable `Complex64` slices `(a, b, c)` simultaneously.
    pub fn complex_triple_zeroed(
        &mut self,
        len_a: usize,
        len_b: usize,
        len_c: usize,
    ) -> (&mut [Complex64], &mut [Complex64], &mut [Complex64]) {
        (
            resize_and_zero(&mut self.complex_a, len_a, Complex64::ZERO),
            resize_and_zero(&mut self.complex_b, len_b, Complex64::ZERO),
            resize_and_zero(&mut self.complex_c, len_c, Complex64::ZERO),
        )
    }

    /// Simultaneous split borrow of `(real_a, complex_a, complex_b, indices)` for mixed-radix/Bluestein FFTs.
    pub fn split_fft_workspace(
        &mut self,
        real_len: usize,
        complex_a_len: usize,
        complex_b_len: usize,
        idx_len: usize,
    ) -> (&mut [f64], &mut [Complex64], &mut [Complex64], &mut [usize]) {
        (
            resize_and_zero(&mut self.real_a, real_len, 0.0),
            resize_and_zero(&mut self.complex_a, complex_a_len, Complex64::ZERO),
            resize_and_zero(&mut self.complex_b, complex_b_len, Complex64::ZERO),
            resize_and_zero(&mut self.indices, idx_len, 0),
        )
    }

    /// Clears active logical lengths while preserving allocated capacities.
    pub fn clear(&mut self) {
        self.real_a.clear();
        self.real_b.clear();
        self.real_c.clear();
        self.complex_a.clear();
        self.complex_b.clear();
        self.complex_c.clear();
        self.indices.clear();
    }

    /// Releases excess unused capacity back to the allocator.
    pub fn shrink_to_fit(&mut self) {
        self.real_a.shrink_to_fit();
        self.real_b.shrink_to_fit();
        self.real_c.shrink_to_fit();
        self.complex_a.shrink_to_fit();
        self.complex_b.shrink_to_fit();
        self.complex_c.shrink_to_fit();
        self.indices.shrink_to_fit();
    }
}

#[inline]
fn resize_and_zero<T: Copy>(buf: &mut Vec<T>, len: usize, zero: T) -> &mut [T] {
    if buf.len() < len {
        buf.fill(zero);
        buf.resize(len, zero);
    } else {
        buf.truncate(len);
        buf.fill(zero);
    }
    buf.as_mut_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_buffer_reuses_capacity_without_aliasing() {
        let mut scratch = ScratchBuffer::with_capacity(256, 256);
        {
            let (ra, rb) = scratch.real_pair_zeroed(128, 64);
            ra[0] = 42.0;
            rb[63] = -7.0;
            assert_eq!(ra.len(), 128);
            assert_eq!(rb.len(), 64);
        }
        // Subsequent borrow re-zeros previously dirtied slots while keeping capacity >= 256!
        let r = scratch.real_zeroed(128);
        assert_eq!(r[0], 0.0);

        let (ca, cb) = scratch.complex_pair_zeroed(32, 32);
        ca[3] = Complex64::I;
        assert_eq!(cb[3], Complex64::ZERO);
    }
}
