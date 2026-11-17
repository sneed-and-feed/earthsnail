//! Zero-dependency double-precision complex arithmetic (`Complex64`) shared across all crates.

use crate::math;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Interleaved double-precision complex number $z = \text{re} + i\,\text{im}$.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    /// Real part $\operatorname{Re}(z)$.
    pub re: f64,
    /// Imaginary part $\operatorname{Im}(z)$.
    pub im: f64,
}

impl Complex64 {
    /// Additive identity $0 + 0i$.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// Multiplicative identity $1 + 0i$.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// Imaginary unit $0 + 1i$.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Creates a new complex number `re + i * im`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Creates a complex number on the unit circle $\cos\theta + i\sin\theta = e^{i\theta}$.
    #[inline]
    pub fn cis(theta: f64) -> Self {
        let (s, c) = math::sin_cos(theta);
        Self { re: c, im: s }
    }

    /// Creates a complex number from polar coordinates $r e^{i\theta}$.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let (s, c) = math::sin_cos(theta);
        Self {
            re: r * c,
            im: r * s,
        }
    }

    /// Complex conjugate $\bar{z} = \text{re} - i\,\text{im}$.
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared modulus $|z|^2 = \text{re}^2 + \text{im}^2$.
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Overflow-safe Euclidean modulus $|z| = \sqrt{\text{re}^2 + \text{im}^2}$.
    #[inline]
    pub fn abs(self) -> f64 {
        math::hypot(self.re, self.im)
    }

    /// Principal phase angle $\arg(z) \in [-\pi, \pi]$.
    #[inline]
    pub fn arg(self) -> f64 {
        math::atan2(self.im, self.re)
    }

    /// Complex exponential $e^z = e^{\text{re}}(\cos\text{im} + i\sin\text{im})$.
    #[inline]
    pub fn exp(self) -> Self {
        let r = math::exp(self.re);
        let (s, c) = math::sin_cos(self.im);
        Self {
            re: r * c,
            im: r * s,
        }
    }

    /// Principal natural logarithm $\ln|z| + i\arg(z)$.
    #[inline]
    pub fn ln(self) -> Self {
        Self {
            re: math::ln(self.abs()),
            im: self.arg(),
        }
    }

    /// Principal square root $\sqrt{z}$ with $\operatorname{Re}(\sqrt{z}) \ge 0$, avoiding overflow/cancellation.
    #[inline]
    pub fn sqrt(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::ZERO;
        }
        let m = self.abs();
        if self.re >= 0.0 {
            let u = math::sqrt(0.5 * (m + self.re));
            let v = 0.5 * self.im / u;
            Self { re: u, im: v }
        } else {
            let v_abs = math::sqrt(0.5 * (m - self.re));
            let u = 0.5 * self.im.abs() / v_abs;
            Self {
                re: u,
                im: v_abs.copysign(self.im),
            }
        }
    }

    /// Smith's overflow-safe multiplicative inverse $1 / z$.
    #[inline]
    pub fn inv(self) -> Self {
        Self::ONE / self
    }

    /// Multiplies both components by real scalar `s`.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Integer power $z^n$ via binary exponentiation.
    #[inline]
    pub fn powi(mut self, exp: i32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        let mut n = exp.unsigned_abs();
        let mut acc = Self::ONE;
        while n > 0 {
            if (n & 1) != 0 {
                acc *= self;
            }
            n >>= 1;
            if n > 0 {
                self *= self;
            }
        }
        if exp < 0 {
            acc.inv()
        } else {
            acc
        }
    }

    /// Returns `true` if both `re` and `im` are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Returns `true` if either `re` or `im` is `NaN`.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl Neg for Complex64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl SubAssign for Complex64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for Complex64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Complex64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Smith's algorithm prevents intermediate overflow/underflow when |rhs| ~ 1e200 or 1e-200.
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self {
                re: (self.re + self.im * r) / denom,
                im: (self.im - self.re * r) / denom,
            }
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.im + rhs.re * r;
            Self {
                re: (self.re * r + self.im) / denom,
                im: (self.im * r - self.re) / denom,
            }
        }
    }
}

impl DivAssign for Complex64 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Add<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self {
            re: self.re + rhs,
            im: self.im,
        }
    }
}

impl Sub<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl MulAssign<f64> for Complex64 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl Mul<Complex64> for f64 {
    type Output = Complex64;
    #[inline]
    fn mul(self, rhs: Complex64) -> Complex64 {
        rhs.scale(self)
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl DivAssign<f64> for Complex64 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

impl From<f64> for Complex64 {
    #[inline]
    fn from(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
}

impl Sum for Complex64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Complex64> for Complex64 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl Product for Complex64 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<'a> Product<&'a Complex64> for Complex64 {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, &b| a * b)
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_algebra_and_euler_identity() {
        let tol = if cfg!(miri) { 1e-13 } else { 1e-15 };
        let a = Complex64::new(3.0, -4.0);
        let b = Complex64::new(-1.5, 2.5);
        assert!((a.abs() - 5.0).abs() < tol);
        assert_eq!(a * a.conj(), Complex64::new(25.0, 0.0));

        let prod = a * b;
        let quot = prod / b;
        assert!((quot.re - a.re).abs() < 1e-13);
        assert!((quot.im - a.im).abs() < 1e-13);

        // Extreme magnitude division (would overflow naive re^2 + im^2!)
        let huge = Complex64::new(1e250, -2e250);
        let one = huge / huge;
        assert!((one.re - 1.0).abs() < tol);
        assert!(one.im.abs() < tol);

        // Euler identity e^{i pi} + 1 == 0
        let euler = Complex64::new(0.0, core::f64::consts::PI).exp() + Complex64::ONE;
        assert!(euler.abs() < tol);

        // Principal sqrt round-trip
        for &(re, im) in &[
            (4.0, 0.0),
            (-4.0, 0.0),
            (-3.0, 4.0),
            (-3.0, -4.0),
            (0.0, -9.0),
        ] {
            let z = Complex64::new(re, im);
            let s = z.sqrt();
            let sq = s * s;
            assert!((sq - z).abs() < 1e-13);
        }
    }
}
