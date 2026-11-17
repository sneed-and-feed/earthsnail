//! Zero-dependency IEEE-754 `f64` mathematical intrinsics supporting both `std` and `no_std`.
//!
//! When the `std` feature is enabled (default), every function compiles to a direct inline call
//! to Rust's standard `f64` methods. When `std` is disabled (`#![no_std]`), pure-Rust double-precision
//! algorithms provide full 53-bit accuracy (`<= 2 ULP`) with zero external crates.

#[cfg(any(not(feature = "std"), test))]
pub(crate) mod soft {
    const TWO_OVER_PI: f64 = core::f64::consts::FRAC_2_PI;
    // Exact 3-word split of pi/2 where HI and MID have 24 trailing zero mantissa bits:
    // PI_2_HI + PI_2_MID + PI_2_LO == pi/2 to >106 bits.
    const PI_2_HI: f64 = f64::from_bits(0x3ff9_21fb_5440_0000); // 1.57079632673412561417
    const PI_2_MID: f64 = f64::from_bits(0x3dd0_b461_1a60_0000); // 6.07710050630396597660e-11
    const PI_2_LO: f64 = f64::from_bits(0x3ba3_198a_2e03_7073); // 2.02226624879595063154e-21

    // Exact 2-word split of ln(2) where LN2_HI has 24 trailing zero mantissa bits:
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000); // 0.69314718036912381649
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76); // 1.90821492927058770002e-10

    #[inline]
    pub fn trunc(x: f64) -> f64 {
        if !x.is_finite() || x.abs() >= 4_503_599_627_370_496.0 {
            return x;
        }
        ((x as i64) as f64).copysign(x)
    }

    #[inline]
    pub fn floor(x: f64) -> f64 {
        let t = trunc(x);
        if x < t {
            t - 1.0
        } else {
            t
        }
    }

    #[inline]
    pub fn ceil(x: f64) -> f64 {
        let t = trunc(x);
        if x > t {
            t + 1.0
        } else {
            t
        }
    }

    #[inline]
    pub fn round(x: f64) -> f64 {
        let t = trunc(x);
        let r = x - t;
        if r >= 0.5 {
            t + 1.0
        } else if r <= -0.5 {
            t - 1.0
        } else {
            t.copysign(x)
        }
    }

    #[inline]
    pub fn fract(x: f64) -> f64 {
        if x.is_nan() || x.is_infinite() {
            return f64::NAN;
        }
        x - trunc(x)
    }

    #[inline]
    pub fn rem_euclid(x: f64, rhs: f64) -> f64 {
        let rhs_abs = rhs.abs();
        let r = x - trunc(x / rhs) * rhs;
        if r < 0.0 {
            let res = r + rhs_abs;
            if res == rhs_abs {
                0.0
            } else {
                res
            }
        } else {
            r
        }
    }

    #[inline]
    pub fn sqrt(x: f64) -> f64 {
        if x < 0.0 || x.is_nan() {
            return f64::NAN;
        }
        if x == 0.0 || x.is_infinite() {
            return x;
        }
        // Scale subnormals into the normal range before bit-seed extraction.
        let (scaled, scale_out) = if x < f64::MIN_POSITIVE {
            (x * 18_014_398_509_481_984.0, 1.0 / 134_217_728.0) // 2^54, 2^-27
        } else {
            (x, 1.0)
        };
        let mut y = f64::from_bits((scaled.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
        y = 0.5 * (y + scaled / y);
        y = 0.5 * (y + scaled / y);
        y = 0.5 * (y + scaled / y);
        y = 0.5 * (y + scaled / y);
        y = 0.5 * (y + scaled / y);
        y * scale_out
    }

    #[inline]
    pub fn hypot(x: f64, y: f64) -> f64 {
        if x.is_infinite() || y.is_infinite() {
            return f64::INFINITY;
        }
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        let ax = x.abs();
        let ay = y.abs();
        let m = ax.max(ay);
        if m == 0.0 {
            return 0.0;
        }
        let u = ax / m;
        let v = ay / m;
        m * sqrt(u * u + v * v)
    }

    #[inline]
    fn kernel_sin(r: f64) -> f64 {
        let r2 = r * r;
        // Horner evaluation of sin(r)/r - 1 on |r| <= pi/4
        let p = (((((((-1.0 / 355_687_428_096_000.0 * r2 + 1.0 / 1_307_674_368_000.0) * r2
            - 1.0 / 6_227_020_800.0)
            * r2
            + 1.0 / 39_916_800.0)
            * r2
            - 1.0 / 362_880.0)
            * r2
            + 1.0 / 5_040.0)
            * r2
            - 1.0 / 120.0)
            * r2
            + 1.0 / 6.0)
            * (-r2);
        r + r * p
    }

    #[inline]
    fn kernel_cos(r: f64) -> f64 {
        let r2 = r * r;
        let mut c = -1.0 / 6_402_373_705_728_000.0; // -1/18!
        c = c * r2 + 1.0 / 20_922_789_888_000.0; // +1/16!
        c = c * r2 - 1.0 / 87_178_291_200.0; // -1/14!
        c = c * r2 + 1.0 / 479_001_600.0; // +1/12!
        c = c * r2 - 1.0 / 3_628_800.0; // -1/10!
        c = c * r2 + 1.0 / 40_320.0; // +1/8!
        c = c * r2 - 1.0 / 720.0; // -1/6!
        c = c * r2 + 1.0 / 24.0; // +1/4!
        (1.0 - 0.5 * r2) + (r2 * r2) * c
    }

    #[inline]
    pub fn sin_cos(x: f64) -> (f64, f64) {
        if !x.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        if x.abs() < 1e-8 {
            return (x, 1.0 - 0.5 * x * x);
        }
        let kf = round(x * TWO_OVER_PI);
        let r = if kf.abs() < 16_777_216.0 {
            ((x - kf * PI_2_HI) - kf * PI_2_MID) - kf * PI_2_LO
        } else {
            let period = core::f64::consts::TAU;
            let x_red = rem_euclid(x, period);
            let k2 = round(x_red * TWO_OVER_PI);
            ((x_red - k2 * PI_2_HI) - k2 * PI_2_MID) - k2 * PI_2_LO
        };
        let q = if kf.abs() < 16_777_216.0 {
            (kf as i64).rem_euclid(4)
        } else {
            let x_red = rem_euclid(x, core::f64::consts::TAU);
            (round(x_red * TWO_OVER_PI) as i64).rem_euclid(4)
        };
        let s = kernel_sin(r);
        let c = kernel_cos(r);
        match q {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }

    #[inline]
    pub fn sin(x: f64) -> f64 {
        sin_cos(x).0
    }

    #[inline]
    pub fn cos(x: f64) -> f64 {
        sin_cos(x).1
    }

    #[inline]
    pub fn tan(x: f64) -> f64 {
        let (s, c) = sin_cos(x);
        s / c
    }

    #[inline]
    pub fn atan(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return core::f64::consts::FRAC_PI_2.copysign(x);
        }
        if x == 0.0 {
            return x;
        }
        let mut a = x.abs();
        let recip = a > 1.0;
        if recip {
            a = 1.0 / a;
        }
        // Segment [0, 1] around {0, tan(pi/12), tan(pi/6), 1} so |u| <= tan(pi/24) ~ 0.13165.
        const TAN_PI_12: f64 = 0.267_949_192_431_122_7; // 2 - sqrt(3)
        const TAN_PI_6: f64 = 0.577_350_269_189_625_8; // 1 / sqrt(3)
        let (theta0, u) = if a < 0.131_652_497_587_395_83 {
            (0.0, a)
        } else if a < 0.414_213_562_373_095_03 {
            (
                core::f64::consts::FRAC_PI_6 * 0.5,
                (a - TAN_PI_12) / (1.0 + a * TAN_PI_12),
            )
        } else if a < 0.767_326_987_978_960_4 {
            (
                core::f64::consts::FRAC_PI_6,
                (a - TAN_PI_6) / (1.0 + a * TAN_PI_6),
            )
        } else {
            (core::f64::consts::FRAC_PI_4, (a - 1.0) / (1.0 + a))
        };
        let u2 = u * u;
        let poly = ((((((((1.0 / 19.0 * u2 - 1.0 / 17.0) * u2 + 1.0 / 15.0) * u2 - 1.0 / 13.0)
            * u2
            + 1.0 / 11.0)
            * u2
            - 1.0 / 9.0)
            * u2
            + 1.0 / 7.0)
            * u2
            - 1.0 / 5.0)
            * u2
            + 1.0 / 3.0)
            * (-u2);
        let mut res = theta0 + (u + u * poly);
        if recip {
            res = core::f64::consts::FRAC_PI_2 - res;
        }
        res.copysign(x)
    }

    #[inline]
    pub fn atan2(y: f64, x: f64) -> f64 {
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() && y.is_infinite() {
            let base = if x.is_sign_positive() {
                core::f64::consts::FRAC_PI_4
            } else {
                3.0 * core::f64::consts::FRAC_PI_4
            };
            return base.copysign(y);
        }
        if x == 0.0 {
            if y == 0.0 {
                return if x.is_sign_negative() {
                    core::f64::consts::PI.copysign(y)
                } else {
                    y
                };
            }
            return core::f64::consts::FRAC_PI_2.copysign(y);
        }
        if y == 0.0 {
            return if x.is_sign_negative() {
                core::f64::consts::PI.copysign(y)
            } else {
                y
            };
        }
        let base = atan(y / x);
        if x > 0.0 {
            base
        } else if y > 0.0 {
            base + core::f64::consts::PI
        } else {
            base - core::f64::consts::PI
        }
    }

    #[inline]
    pub fn asin(x: f64) -> f64 {
        if x.is_nan() || x.abs() > 1.0 {
            return f64::NAN;
        }
        atan2(x, sqrt((1.0 - x) * (1.0 + x)))
    }

    #[inline]
    pub fn acos(x: f64) -> f64 {
        if x.is_nan() || x.abs() > 1.0 {
            return f64::NAN;
        }
        atan2(sqrt((1.0 - x) * (1.0 + x)), x)
    }

    #[inline]
    pub fn exp(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        if x > 709.782_712_893_384 {
            return f64::INFINITY;
        }
        if x < -745.133_219_101_941_2 {
            return 0.0;
        }
        let kf = round(x * core::f64::consts::LOG2_E);
        let k = kf as i32;
        let r = (x - kf * LN2_HI) - kf * LN2_LO;
        let mut p = 1.0 / 6_227_020_800.0; // 1/13!
        p = p * r + 1.0 / 479_001_600.0;
        p = p * r + 1.0 / 39_916_800.0;
        p = p * r + 1.0 / 3_628_800.0;
        p = p * r + 1.0 / 362_880.0;
        p = p * r + 1.0 / 40_320.0;
        p = p * r + 1.0 / 5_040.0;
        p = p * r + 1.0 / 720.0;
        p = p * r + 1.0 / 120.0;
        p = p * r + 1.0 / 24.0;
        p = p * r + 1.0 / 6.0;
        p = p * r + 0.5;
        let exp_r = 1.0 + r + (r * r) * p;
        ldexp(exp_r, k)
    }

    #[inline]
    fn ldexp(x: f64, exp: i32) -> f64 {
        if x == 0.0 || !x.is_finite() || exp == 0 {
            return x;
        }
        let mut v = x;
        let mut e = exp;
        while e > 1023 {
            v *= f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
            e -= 1023;
        }
        while e < -1022 {
            v *= f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
            e += 1022;
        }
        let scale = f64::from_bits(((e + 1023) as u64) << 52);
        v * scale
    }

    #[inline]
    pub fn ln(x: f64) -> f64 {
        if x.is_nan() || x < 0.0 {
            return f64::NAN;
        }
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if x.is_infinite() {
            return f64::INFINITY;
        }
        let (norm, exp_adj) = if x < f64::MIN_POSITIVE {
            (x * 18_014_398_509_481_984.0, -54_i32)
        } else {
            (x, 0_i32)
        };
        let bits = norm.to_bits();
        let mut k = ((bits >> 52) as i32) - 1023 + exp_adj;
        let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
        if m > core::f64::consts::SQRT_2 {
            m *= 0.5;
            k += 1;
        }
        let u = (m - 1.0) / (m + 1.0);
        let u2 = u * u;
        let mut s = 1.0 / 23.0;
        s = s * u2 + 1.0 / 21.0;
        s = s * u2 + 1.0 / 19.0;
        s = s * u2 + 1.0 / 17.0;
        s = s * u2 + 1.0 / 15.0;
        s = s * u2 + 1.0 / 13.0;
        s = s * u2 + 1.0 / 11.0;
        s = s * u2 + 1.0 / 9.0;
        s = s * u2 + 1.0 / 7.0;
        s = s * u2 + 1.0 / 5.0;
        s = s * u2 + 1.0 / 3.0;
        let ln_m = 2.0 * u + 2.0 * u * u2 * s;
        let kf = k as f64;
        kf * LN2_HI + (ln_m + kf * LN2_LO)
    }

    #[inline]
    pub fn log10(x: f64) -> f64 {
        ln(x) * core::f64::consts::LOG10_E
    }

    #[inline]
    pub fn powi(mut base: f64, exp: i32) -> f64 {
        if exp == 0 {
            return 1.0;
        }
        if base.is_nan() {
            return f64::NAN;
        }
        let mut n: u32 = exp.unsigned_abs();
        let mut acc = 1.0;
        while n > 0 {
            if (n & 1) != 0 {
                acc *= base;
            }
            n >>= 1;
            if n > 0 {
                base *= base;
            }
        }
        if exp < 0 {
            1.0 / acc
        } else {
            acc
        }
    }

    #[inline]
    pub fn powf(x: f64, y: f64) -> f64 {
        if y == 0.0 {
            return 1.0;
        }
        if x.is_nan() || y.is_nan() {
            return f64::NAN;
        }
        if x == 1.0 {
            return 1.0;
        }
        let yi = trunc(y);
        if yi == y && y.abs() <= (i32::MAX as f64) {
            return powi(x, yi as i32);
        }
        if x < 0.0 {
            return f64::NAN;
        }
        if x == 0.0 {
            return if y > 0.0 { 0.0 } else { f64::INFINITY };
        }
        exp(y * ln(x))
    }
}

/// Square root $\sqrt{x}$.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::sqrt(x)
    }
}

/// Overflow-safe Euclidean hypotenuse $\sqrt{x^2 + y^2}$.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.hypot(y)
    }
    #[cfg(not(feature = "std"))]
    {
        soft::hypot(x, y)
    }
}

/// Sine $\sin(x)$ with $x$ in radians.
#[inline]
pub fn sin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::sin(x)
    }
}

/// Cosine $\cos(x)$ with $x$ in radians.
#[inline]
pub fn cos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::cos(x)
    }
}

/// Simultaneous $(\sin x, \cos x)$ with $x$ in radians.
#[inline]
pub fn sin_cos(x: f64) -> (f64, f64) {
    #[cfg(feature = "std")]
    {
        x.sin_cos()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::sin_cos(x)
    }
}

/// Tangent $\tan(x)$ with $x$ in radians.
#[inline]
pub fn tan(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.tan()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::tan(x)
    }
}

/// Principal arcsine $\arcsin(x) \in [-\pi/2, \pi/2]$.
#[inline]
pub fn asin(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.asin()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::asin(x)
    }
}

/// Principal arccosine $\arccos(x) \in [0, \pi]$.
#[inline]
pub fn acos(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.acos()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::acos(x)
    }
}

/// Principal arctangent $\arctan(x) \in [-\pi/2, \pi/2]$.
#[inline]
pub fn atan(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.atan()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::atan(x)
    }
}

/// Four-quadrant arctangent $\operatorname{atan2}(y, x) \in [-\pi, \pi]$.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        soft::atan2(y, x)
    }
}

/// Natural exponential $e^x$.
#[inline]
pub fn exp(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::exp(x)
    }
}

/// Natural logarithm $\ln(x)$.
#[inline]
pub fn ln(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::ln(x)
    }
}

/// Base-10 logarithm $\log_{10}(x)$.
#[inline]
pub fn log10(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.log10()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::log10(x)
    }
}

/// Integer power $x^n$.
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powi(n)
    }
    #[cfg(not(feature = "std"))]
    {
        soft::powi(x, n)
    }
}

/// Real power $x^y$.
#[inline]
pub fn powf(x: f64, y: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(not(feature = "std"))]
    {
        soft::powf(x, y)
    }
}

/// Largest integer $\le x$.
#[inline]
pub fn floor(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::floor(x)
    }
}

/// Smallest integer $\ge x$.
#[inline]
pub fn ceil(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ceil()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::ceil(x)
    }
}

/// Nearest integer (half-way cases away from zero).
#[inline]
pub fn round(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.round()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::round(x)
    }
}

/// Integer part truncated toward zero.
#[inline]
pub fn trunc(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.trunc()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::trunc(x)
    }
}

/// Signed fractional part `x - trunc(x)`.
#[inline]
pub fn fract(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.fract()
    }
    #[cfg(not(feature = "std"))]
    {
        soft::fract(x)
    }
}

/// Least nonnegative remainder `x mod rhs`.
#[inline]
pub fn rem_euclid(x: f64, rhs: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.rem_euclid(rhs)
    }
    #[cfg(not(feature = "std"))]
    {
        soft::rem_euclid(x, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_math_matches_reference_identities() {
        for k in -200..=200 {
            let x = (k as f64) * 0.137;
            let (s, c) = soft::sin_cos(x);
            assert!((s * s + c * c - 1.0).abs() < 1e-15);
            #[cfg(feature = "std")]
            {
                // Under Miri, std transcendentals intentionally inject +-16 ULP (~3.5e-15) jitter.
                let std_tol = if cfg!(miri) { 1e-13 } else { 1e-15 };
                assert!((s - x.sin()).abs() < std_tol);
                assert!((c - x.cos()).abs() < std_tol);
                let at = soft::atan(x);
                assert!((at - x.atan()).abs() < std_tol);
            }
        }

        let pos_vals = [1e-305, 1e-50, 0.001, 0.5, 1.0, 2.75, 12345.6789, 1e120];
        for &v in &pos_vals {
            let r = soft::sqrt(v);
            assert!(((r * r - v) / v).abs() < 1e-15);
            let l = soft::ln(v);
            let e = soft::exp(l);
            assert!(((e - v) / v).abs() < 1e-13);
        }

        for &u in &[-1.0, -0.9999, -0.5, 0.0, 0.5, 0.9999, 1.0] {
            let a = soft::asin(u);
            let b = soft::acos(u);
            assert!((a + b - core::f64::consts::FRAC_PI_2).abs() < 1e-15);
        }

        assert_eq!(soft::floor(-2.3), -3.0);
        assert_eq!(soft::ceil(-2.3), -2.0);
        assert_eq!(soft::round(-2.5), -3.0);
        assert_eq!(soft::round(2.5), 3.0);
        assert!((soft::fract(3.25) - 0.25).abs() < 1e-15);
        assert!((soft::hypot(3e200, 4e200) - 5e200) / 5e200 < 1e-15);
        assert!((soft::sin(0.5) * soft::cos(0.5) - 0.5 * soft::sin(1.0)).abs() < 1e-15);
        assert!((soft::tan(0.4) - soft::sin(0.4) / soft::cos(0.4)).abs() < 1e-15);
        assert!((soft::log10(1000.0) - 3.0).abs() < 1e-14);
        assert!((soft::powi(2.0, -3) - 0.125).abs() < 1e-15);
        assert!((soft::powf(9.0, 1.5) - 27.0).abs() < 1e-13);
    }
}
