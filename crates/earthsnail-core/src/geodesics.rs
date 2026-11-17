//! Exact WGS84 / GRS80 ellipsoidal inverse & direct geodesic solver (Karney safeguarded
//! dual-number Newton-Raphson + 16-point Gauss-Legendre elliptic quadrature), geocentric <->
//! geographic latitude conversions, Dziewonski-Gilbert ellipticity weights, and great-circle waypoints.

use crate::error::{EarthSnailError, Result};
use crate::math;
use alloc::vec::Vec;

/// Mean spherical Earth radius in kilometers (`obspy.geodetics` standard `6371.0 km`).
pub const EARTH_RADIUS_KM: f64 = 6_371.0;
/// Mean spherical Earth radius in meters (`6_371_000.0 m`).
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Kilometers per degree on the `6371.0 km` reference sphere (`~111.19492664455873 km/deg`).
pub const KM_PER_DEG: f64 = EARTH_RADIUS_KM * core::f64::consts::PI / 180.0;

/// Reference ellipsoid of revolution defined by equatorial radius `a_m` and flattening `f`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipsoid {
    /// Equatorial semi-major axis $a$ in meters.
    pub a_m: f64,
    /// Flattening $f = (a - b) / a$.
    pub f: f64,
}

impl Ellipsoid {
    /// WGS84 reference ellipsoid ($a = 6378137.0\text{ m}$, $1/f = 298.257223563$).
    pub const WGS84: Self = Self {
        a_m: 6_378_137.0,
        f: 1.0 / 298.257_223_563,
    };

    /// GRS80 reference ellipsoid ($a = 6378137.0\text{ m}$, $1/f = 298.257222101$).
    pub const GRS80: Self = Self {
        a_m: 6_378_137.0,
        f: 1.0 / 298.257_222_101,
    };

    /// Polar semi-minor axis $b = a(1 - f)$ in meters.
    #[inline]
    pub fn b_m(self) -> f64 {
        self.a_m * (1.0 - self.f)
    }

    /// Second eccentricity squared $e'^2 = \frac{a^2 - b^2}{b^2} = \frac{f(2 - f)}{(1 - f)^2}$.
    #[inline]
    pub fn ep2(self) -> f64 {
        let omf = 1.0 - self.f;
        self.f * (2.0 - self.f) / (omf * omf)
    }
}

/// Result of an inverse geodesic query from `(lat1, lon1)` to `(lat2, lon2)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicResult {
    /// Shortest ellipsoidal geodesic distance $s_{12}$ in meters.
    pub distance_m: f64,
    /// Exact spherical central angle on `(lat1, lon1)` -> `(lat2, lon2)` in degrees (`locations2degrees`).
    pub distance_deg_spherical: f64,
    /// Great-circle arc distance in degrees (`kilometer2degrees(distance_m / 1000.0)` matching `SAC` / `ObsPy`).
    pub gcarc_deg: f64,
    /// Forward azimuth at point 1 clockwise from North in `[0.0, 360.0)` degrees.
    pub azimuth_deg: f64,
    /// Back-azimuth at point 2 (direction from point 2 back to point 1) in `[0.0, 360.0)` degrees.
    pub back_azimuth_deg: f64,
}

/// Result of a direct geodesic query starting from `(lat1, lon1)` along `azimuth_deg` for `distance_m`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectGeodesicResult {
    /// Destination geographic latitude in `[-90.0, 90.0]` degrees.
    pub lat_deg: f64,
    /// Destination geographic longitude in `[-180.0, 180.0]` degrees.
    pub lon_deg: f64,
    /// Forward continuation azimuth at destination in `[0.0, 360.0)` degrees.
    pub final_azimuth_deg: f64,
    /// Back-azimuth from destination looking back to start in `[0.0, 360.0)` degrees.
    pub back_azimuth_deg: f64,
    /// Auxiliary spherical arc length $\sigma_{12}$ in degrees.
    pub arc_deg: f64,
}

// 16-point Gauss-Legendre positive abscissas and weights on [-1, 1] (exact to >30 digits).
const GL8_HALF_NODES: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_37,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];

const GL8_HALF_WEIGHTS: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_58,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_74,
    0.124_628_971_255_533_88,
    0.095_158_511_682_492_79,
    0.062_253_523_938_647_894,
    0.027_152_459_411_754_096,
];

/// Converts kilometers to degrees on the mean `6371.0 km` Earth sphere (`obspy.geodetics.kilometer2degrees`).
#[inline]
pub fn kilometer2degrees(km: f64) -> f64 {
    km / KM_PER_DEG
}

/// Converts degrees to kilometers on the mean `6371.0 km` Earth sphere (`obspy.geodetics.degrees2kilometers`).
#[inline]
pub fn degrees2kilometers(deg: f64) -> f64 {
    deg * KM_PER_DEG
}

/// Converts geographic (geodetic) latitude in degrees to geocentric latitude in degrees:
/// $\theta_{\text{geoc}} = \arctan((1 - f)^2 \tan\phi)$ on WGS84.
#[inline]
pub fn geographic_to_geocentric_lat(lat_deg: f64) -> f64 {
    geographic_to_geocentric_lat_ellipsoid(lat_deg, Ellipsoid::WGS84)
}

/// Converts geographic latitude in degrees to geocentric latitude on a given `Ellipsoid`.
#[inline]
pub fn geographic_to_geocentric_lat_ellipsoid(lat_deg: f64, ellip: Ellipsoid) -> f64 {
    let omf2 = (1.0 - ellip.f) * (1.0 - ellip.f);
    let (s, c) = sin_cos_deg(lat_deg);
    math::atan2(omf2 * s, c).to_degrees()
}

/// Converts geocentric latitude in degrees back to geographic (geodetic) latitude in degrees:
/// $\phi = \arctan(\tan\theta_{\text{geoc}} / (1 - f)^2)$ on WGS84.
#[inline]
pub fn geocentric_to_geographic_lat(geocentric_lat_deg: f64) -> f64 {
    geocentric_to_geographic_lat_ellipsoid(geocentric_lat_deg, Ellipsoid::WGS84)
}

/// Converts geocentric latitude in degrees back to geographic latitude on a given `Ellipsoid`.
#[inline]
pub fn geocentric_to_geographic_lat_ellipsoid(geocentric_lat_deg: f64, ellip: Ellipsoid) -> f64 {
    let omf2 = (1.0 - ellip.f) * (1.0 - ellip.f);
    let (s, c) = sin_cos_deg(geocentric_lat_deg);
    math::atan2(s, omf2 * c).to_degrees()
}

/// Dziewonski & Gilbert (1976) / Kennett & Gudmundsson (1996 `ak135` `ellip`) degree-2 real
/// spherical harmonic geometric weights `[sc0, sc1, sc2]` evaluated at `source_lat_deg` (converted
/// to WGS84 geocentric co-latitude $\theta_0$) and event-to-station `azimuth_deg` $\zeta$:
///
/// - `sc0` = $\frac{1}{4}(1 + 3\cos 2\theta_0) = P_2^0(\cos\theta_0)$
/// - `sc1` = $\frac{\sqrt{3}}{2}\sin 2\theta_0 \cos\zeta$
/// - `sc2` = $\frac{\sqrt{3}}{2}\sin^2\theta_0 \cos 2\zeta$
///
/// Travel-time ellipticity correction is $\delta t = \tau_0(\Delta, h)\,\text{sc}_0 + \tau_1(\Delta, h)\,\text{sc}_1 + \tau_2(\Delta, h)\,\text{sc}_2$.
pub fn dziewonski_gilbert_weights(source_lat_deg: f64, azimuth_deg: f64) -> [f64; 3] {
    let geoc_lat_deg = geographic_to_geocentric_lat(source_lat_deg);
    let colat_rad = (90.0 - geoc_lat_deg).to_radians();
    let az_rad = azimuth_deg.to_radians();
    let sin_th = math::sin(colat_rad);
    let cos_2th = math::cos(2.0 * colat_rad);
    let sin_2th = math::sin(2.0 * colat_rad);
    let half_sqrt3 = 0.866_025_403_784_438_6; // sqrt(3) / 2
    let sc0 = 0.25 * (1.0 + 3.0 * cos_2th);
    let sc1 = half_sqrt3 * sin_2th * math::cos(az_rad);
    let sc2 = half_sqrt3 * (sin_th * sin_th) * math::cos(2.0 * az_rad);
    [sc0, sc1, sc2]
}

/// Exact spherical great-circle central angle in degrees (`obspy.geodetics.locations2degrees`).
pub fn locations2degrees(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    let (s1, c1) = sin_cos_deg(lat1_deg);
    let (s2, c2) = sin_cos_deg(lat2_deg);
    let dlon_deg = ang_diff_deg(lon1_deg, lon2_deg);
    let (sdlon, cdlon) = sin_cos_deg(dlon_deg);
    let x = c2 * sdlon;
    let y = c1 * s2 - s1 * c2 * cdlon;
    let num = math::hypot(x, y);
    let den = s1 * s2 + c1 * c2 * cdlon;
    math::atan2(num, den).to_degrees()
}

/// Geocentric great-circle central angle in degrees using WGS84 geocentric latitudes.
pub fn geocentric_central_angle_deg(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> f64 {
    locations2degrees(
        geographic_to_geocentric_lat(lat1_deg),
        lon1_deg,
        geographic_to_geocentric_lat(lat2_deg),
        lon2_deg,
    )
}

/// Validated WGS84 inverse geodesic solver returning `Err` if coordinates are `NaN`/`Inf` or `|lat| > 90`.
pub fn try_gps2dist_azimuth(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> Result<GeodesicResult> {
    try_gps2dist_azimuth_ellipsoid(lat1_deg, lon1_deg, lat2_deg, lon2_deg, Ellipsoid::WGS84)
}

/// Validated inverse geodesic solver on an arbitrary `Ellipsoid`.
pub fn try_gps2dist_azimuth_ellipsoid(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    ellip: Ellipsoid,
) -> Result<GeodesicResult> {
    if !lat1_deg.is_finite()
        || !lon1_deg.is_finite()
        || !lat2_deg.is_finite()
        || !lon2_deg.is_finite()
    {
        return Err(EarthSnailError::InvalidParameter(
            "non-finite geographic coordinate".into(),
        ));
    }
    if lat1_deg.abs() > 90.0 || lat2_deg.abs() > 90.0 {
        return Err(EarthSnailError::InvalidParameter(
            "latitude outside [-90, 90] degrees".into(),
        ));
    }
    Ok(gps2dist_azimuth_ellipsoid(
        lat1_deg, lon1_deg, lat2_deg, lon2_deg, ellip,
    ))
}

/// Antipodal-safe, polar-safe, and near-coincident-exact WGS84 inverse geodesic solver.
///
/// Matches `GeographicLib` / `obspy.geodetics.gps2dist_azimuth` to `< 1e-8 m` (`~10 nm`) and
/// `< 1e-11 deg` across the entire ellipsoid.
pub fn gps2dist_azimuth(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> GeodesicResult {
    gps2dist_azimuth_ellipsoid(lat1_deg, lon1_deg, lat2_deg, lon2_deg, Ellipsoid::WGS84)
}

/// Antipodal-safe inverse geodesic solver on an arbitrary `Ellipsoid`.
pub fn gps2dist_azimuth_ellipsoid(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    ellip: Ellipsoid,
) -> GeodesicResult {
    let lat1 = lat1_deg.clamp(-90.0, 90.0);
    let lat2 = lat2_deg.clamp(-90.0, 90.0);
    let distance_deg_spherical = locations2degrees(lat1, lon1_deg, lat2, lon2_deg);

    let dlon_initial = ang_diff_deg(lon1_deg, lon2_deg);
    if (lat1 == lat2 && dlon_initial == 0.0)
        || (lat1 == 90.0 && lat2 == 90.0)
        || (lat1 == -90.0 && lat2 == -90.0)
    {
        return GeodesicResult {
            distance_m: 0.0,
            distance_deg_spherical: 0.0,
            gcarc_deg: 0.0,
            azimuth_deg: 0.0,
            back_azimuth_deg: 0.0,
        };
    }

    // Step A: Ensure |phi1| >= |phi2|.
    let (mut phi1, mut phi2, mut dlon_deg, swapped) = if lat1.abs() < lat2.abs() {
        (lat2, lat1, -dlon_initial, true)
    } else {
        (lat1, lat2, dlon_initial, false)
    };

    // Step B: Ensure phi1 <= 0 (southern hemisphere start).
    let lat_flipped = if phi1 > 0.0 {
        phi1 = -phi1;
        phi2 = -phi2;
        true
    } else {
        false
    };

    // Step C: Ensure 0 <= dlon_deg <= 180.
    let lon_flipped = if dlon_deg < 0.0 {
        dlon_deg = -dlon_deg;
        true
    } else {
        false
    };

    let lam_target = dlon_deg.to_radians();
    let (sbet1, cbet1, n1) = reduced_lat_sc(phi1, ellip.f);
    let (sbet2, cbet2, n2) = reduced_lat_sc(phi2, ellip.f);
    let dphi_deg = phi2 - phi1; // >= 0 since phi1 <= phi2 <= -phi1

    let (dist_m, mut salp1, mut calp1, mut salp2, mut calp2) = solve_canonical_inverse(
        ellip, phi1, phi2, dphi_deg, sbet1, cbet1, n1, sbet2, cbet2, n2, lam_target,
    );

    // Undo Step C, Step B, Step A in exact reverse order.
    if lon_flipped {
        salp1 = -salp1;
        salp2 = -salp2;
    }
    if lat_flipped {
        calp1 = -calp1;
        calp2 = -calp2;
    }
    if swapped {
        let (tmp_s, tmp_c) = (salp1, calp1);
        salp1 = -salp2;
        calp1 = -calp2;
        salp2 = -tmp_s;
        calp2 = -tmp_c;
    }

    let azimuth_deg = wrap_deg_360(math::atan2(salp1, calp1).to_degrees());
    let back_azimuth_deg = wrap_deg_360(math::atan2(-salp2, -calp2).to_degrees());
    let gcarc_deg = kilometer2degrees(dist_m * 1e-3);

    GeodesicResult {
        distance_m: dist_m,
        distance_deg_spherical,
        gcarc_deg,
        azimuth_deg,
        back_azimuth_deg,
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_canonical_inverse(
    ellip: Ellipsoid,
    phi1: f64,
    phi2: f64,
    dphi_deg: f64,
    sbet1: f64,
    cbet1: f64,
    n1: f64,
    sbet2: f64,
    cbet2: f64,
    n2: f64,
    lam_target: f64,
) -> (f64, f64, f64, f64, f64) {
    let b = ellip.b_m();
    let ep2 = ellip.ep2();

    // Case 1: Point 1 is at the South Pole (phi1 == -90, cbet1 == 0) or meridian dlon == 0.
    if cbet1 == 0.0 || lam_target == 0.0 {
        // Along a meridian: salp0 = 0, calp0 = 1, k^2 = ep2.
        let (salp1, calp1) = if cbet1 == 0.0 {
            math::sin_cos(lam_target)
        } else {
            (0.0, 1.0)
        };
        let (salp2, calp2) = (0.0, 1.0);
        let sig1 = math::atan2(sbet1, cbet1);
        let sig2 = math::atan2(sbet2, cbet2);
        let dist_m = b * integrate_i1(sig1, sig2, ep2);
        return (dist_m, salp1, calp1, salp2, calp2);
    }

    // Case 2: Both points lie on the equator (phi1 == 0 && phi2 == 0).
    if phi1 == 0.0 && phi2 == 0.0 {
        let omf_pi = (1.0 - ellip.f) * core::f64::consts::PI;
        if lam_target <= omf_pi {
            // Pure equatorial geodesic.
            return (ellip.a_m * lam_target, 1.0, 0.0, 1.0, 0.0);
        }
        // Near-antipodal or exact-antipodal equatorial points: geodesic arches into hemisphere!
        if (core::f64::consts::PI - lam_target).abs() <= 1e-16 {
            // Exact equatorial antipodes: meridian over North Pole (azi1 = 0, azi2 = 180).
            let dist_m = b * integrate_i1(0.0, core::f64::consts::PI, ep2);
            return (dist_m, 0.0, 1.0, 0.0, -1.0);
        }
        // Solve pi - f(2 - f) * sin(alp1) * int_0^pi dsigma / (1 + (1-f)sqrt(1 + k^2 sin^2 sigma)) = lam_target
        // for alp1 in (0, pi/2) (Northern arch: calp1 > 0, calp2 = -calp1 < 0).
        let mut lo = 0.0_f64;
        let mut hi = core::f64::consts::FRAC_PI_2;
        let mut alp1 = math::asin(
            ((core::f64::consts::PI - lam_target) / (ellip.f * core::f64::consts::PI))
                .clamp(0.0, 1.0),
        );
        for _ in 0..48 {
            let (lam_val, dlam) = equatorial_arch_lambda_dual(ellip, alp1);
            let err = lam_val - lam_target;
            // On alp1 in [0, pi/2], lam_val decreases strictly from pi at 0 to (1-f)pi at pi/2 (dlam < 0).
            if err > 0.0 {
                lo = alp1;
            } else {
                hi = alp1;
            }
            if (hi - lo) <= 1e-15 {
                break;
            }
            if dlam.abs() > 0.0 {
                let step = err / dlam;
                if step.abs() <= 1e-15 {
                    alp1 = (alp1 - step).clamp(lo, hi);
                    break;
                }
                let cand = alp1 - step;
                alp1 = if cand > lo && cand < hi {
                    cand
                } else {
                    0.5 * (lo + hi)
                };
            } else {
                alp1 = 0.5 * (lo + hi);
            }
        }
        let (salp1, calp1) = math::sin_cos(alp1);
        let k2 = ep2 * calp1 * calp1;
        let dist_m = b * integrate_i1(0.0, core::f64::consts::PI, k2);
        return (dist_m, salp1, calp1, salp1, -calp1);
    }

    // Case 3: Exact meridional passage over the South Pole (lam_target == pi and beta2 < -beta1,
    // OR exact antipodes when meridian applies? Wait: for beta2 == -beta1 and lam_target == pi,
    // sin(alp1) = 0 -> alp1 = pi IS the exact root of lambda12(pi) = pi!).
    if (core::f64::consts::PI - lam_target).abs() == 0.0 {
        let sig1 = -core::f64::consts::PI - math::atan2(sbet1, cbet1);
        let sig2 = math::atan2(sbet2, cbet2);
        let dist_m = b * integrate_i1(sig1, sig2, ep2);
        return (dist_m, 0.0, -1.0, 0.0, 1.0);
    }

    // Precompute cancellation-free latitude differences for arbitrary proximity down to 1e-300 deg.
    let ctx = CanonicalPairCtx::new(ellip, dphi_deg, sbet1, cbet1, n1, sbet2, cbet2, n2);

    // Smart initial seed for alp1 in (0, pi):
    let (sdlon, cdlon) = math::sin_cos(lam_target);
    let sph_y = cbet2 * sdlon;
    let sph_x = if dphi_deg <= 45.0 && lam_target < 0.5 {
        // Cancellation-free cbet1*sbet2 - sbet1*cbet2*cdlon = sin(dbeta) + sbet1*cbet2*(1 - cdlon)
        let half_dlon = math::sin(0.5 * lam_target);
        ctx.sin_dbeta + sbet1 * cbet2 * (2.0 * half_dlon * half_dlon)
    } else {
        cbet1 * sbet2 - sbet1 * cbet2 * cdlon
    };
    let mut alp1 =
        math::atan2(sph_y, sph_x * (1.0 - ellip.f)).clamp(1e-15, core::f64::consts::PI - 1e-15);

    let mut lo = 0.0_f64;
    let mut hi = core::f64::consts::PI;

    for _ in 0..48 {
        let eval = ctx.eval_lambda_dual(alp1);
        let err = eval.lam12 - lam_target;
        if err < 0.0 {
            if alp1 > lo {
                lo = alp1;
            }
        } else if alp1 < hi {
            hi = alp1;
        }
        if (hi - lo) <= 1e-15 * (1.0 + alp1) {
            break;
        }
        if eval.dlam12 > 0.0 {
            let step = err / eval.dlam12;
            if step.abs() <= 1e-15 {
                alp1 = (alp1 - step).clamp(lo, hi);
                break;
            }
            let next = alp1 - step;
            alp1 = if next > lo && next < hi {
                next
            } else {
                0.5 * (lo + hi)
            };
        } else {
            alp1 = 0.5 * (lo + hi);
        }
    }

    ctx.finalize_geodesic(alp1)
}

#[derive(Clone, Copy)]
struct Dual {
    v: f64,
    d: f64,
}

impl Dual {
    #[inline]
    const fn c(v: f64) -> Self {
        Self { v, d: 0.0 }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            v: self.v + o.v,
            d: self.d + o.d,
        }
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            v: self.v - o.v,
            d: self.d - o.d,
        }
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            d: self.v * o.d + self.d * o.v,
        }
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        Self {
            v: self.v * s,
            d: self.d * s,
        }
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        Self {
            v: self.v * inv,
            d: (self.d * o.v - self.v * o.d) * (inv * inv),
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        let r = math::sqrt(self.v);
        let d = if r > 0.0 { 0.5 * self.d / r } else { 0.0 };
        Self { v: r, d }
    }
    #[inline]
    fn hypot(self, o: Self) -> Self {
        let h = math::hypot(self.v, o.v);
        let d = if h > 0.0 {
            (self.v * self.d + o.v * o.d) / h
        } else {
            0.0
        };
        Self { v: h, d }
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = math::sin_cos(self.v);
        (
            Self {
                v: s,
                d: c * self.d,
            },
            Self {
                v: c,
                d: -s * self.d,
            },
        )
    }
    #[inline]
    fn atan2(y: Self, x: Self) -> Self {
        let v = math::atan2(y.v, x.v);
        let r2 = x.v * x.v + y.v * y.v;
        let d = if r2 > 0.0 {
            (x.v * y.d - y.v * x.d) / r2
        } else {
            0.0
        };
        Self { v, d }
    }
}

struct LambdaEval {
    lam12: f64,
    dlam12: f64,
}

struct CanonicalPairCtx {
    ellip: Ellipsoid,
    sbet1: f64,
    cbet1: f64,
    sbet2: f64,
    cbet2: f64,
    sin_dbeta: f64,
    dsbet: f64,
    cbet2_sq_minus_cbet1_sq: f64,
}

impl CanonicalPairCtx {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ellip: Ellipsoid,
        dphi_deg: f64,
        sbet1: f64,
        cbet1: f64,
        n1: f64,
        sbet2: f64,
        cbet2: f64,
        n2: f64,
    ) -> Self {
        // Exact sin(beta2 - beta1) without subtracting nearly equal latitudes:
        let sin_dbeta = if dphi_deg <= 45.0 {
            ((1.0 - ellip.f) * sin_deg(dphi_deg) / (n1 * n2)).clamp(-1.0, 1.0)
        } else {
            sbet2 * cbet1 - cbet2 * sbet1
        };
        let (dsbet, cbet2_sq_minus_cbet1_sq) = if dphi_deg <= 45.0 {
            let dbeta = math::asin(sin_dbeta);
            let beta1 = math::atan2(sbet1, cbet1);
            let ds = 2.0 * math::sin(0.5 * dbeta) * math::cos(beta1 + 0.5 * dbeta);
            let dc2 = -sin_dbeta * math::sin(2.0 * beta1 + dbeta);
            (ds, dc2.max(0.0))
        } else {
            (sbet2 - sbet1, ((cbet2 - cbet1) * (cbet2 + cbet1)).max(0.0))
        };
        Self {
            ellip,
            sbet1,
            cbet1,
            sbet2,
            cbet2,
            sin_dbeta,
            dsbet,
            cbet2_sq_minus_cbet1_sq,
        }
    }

    fn eval_lambda_dual(&self, alp1: f64) -> LambdaEval {
        let (s1, c1) = math::sin_cos(alp1);
        let salp1 = Dual { v: s1, d: c1 };
        let calp1 = Dual { v: c1, d: -s1 };

        let salp0 = salp1.scale(self.cbet1);
        let p = calp1.scale(self.cbet1);
        let calp0 = p.hypot(Dual::c(self.sbet1));

        let q2_sq = p.mul(p).add(Dual::c(self.cbet2_sq_minus_cbet1_sq));
        let q2 = if q2_sq.v > 0.0 {
            q2_sq.sqrt()
        } else {
            // Cusp at beta2 == -beta1 and alp1 == pi/2: take right-sided limit |P|.
            Dual {
                v: 0.0,
                d: self.cbet1,
            }
        };

        let ssig1 = Dual::c(self.sbet1).div(calp0);
        let csig1 = p.div(calp0);
        let ssig2 = Dual::c(self.sbet2).div(calp0);
        let csig2 = q2.div(calp0);

        let (sig1, sig12, omg12) = if p.v > 0.0 {
            // Northbound throughout: use cancellation-free difference formulas for sig12 & omg12!
            let q2_minus_p = Dual::c(self.cbet2_sq_minus_cbet1_sq).div(q2.add(p));
            // c_alp0^2 * sin(sig2 - sig1) = dsbet * P - sbet1 * (Q2 - P)
            let num_ssig12 = p.scale(self.dsbet).sub(q2_minus_p.scale(self.sbet1));
            let calp0_sq = calp0.mul(calp0);
            let ssig12 = num_ssig12.div(calp0_sq);
            let csig12 = csig1.mul(csig2).add(ssig1.mul(ssig2));
            let sig12_d = Dual::atan2(ssig12, csig12);
            let sig1_d = Dual::atan2(ssig1, csig1);

            let somg12 = salp0.mul(ssig12);
            let comg12 = csig1.mul(csig2).add(salp0.mul(salp0).mul(ssig1).mul(ssig2));
            let omg12_d = Dual::atan2(somg12, comg12);
            (sig1_d, sig12_d, omg12_d)
        } else {
            // Southern vertex crossed (csig1 <= 0, csig2 >= 0): omg1 in [-pi, -pi/2], omg2 in [-pi/2, pi/2].
            let sig1_d = Dual::c(-core::f64::consts::PI)
                .add(Dual::atan2(ssig1.scale(-1.0), csig1.scale(-1.0)));
            let sig2_d = Dual::atan2(ssig2, csig2);
            let omg1_d = Dual::c(-core::f64::consts::PI)
                .add(Dual::atan2(salp0.mul(ssig1).scale(-1.0), csig1.scale(-1.0)));
            let omg2_d = Dual::atan2(salp0.mul(ssig2), csig2);
            (sig1_d, sig2_d.sub(sig1_d), omg2_d.sub(omg1_d))
        };

        let k2 = calp0.mul(calp0).scale(self.ellip.ep2());
        let f = self.ellip.f;
        let omf = 1.0 - f;

        // 16-point Gauss-Legendre quadrature of int_{sig1}^{sig1 + sig12} dsigma / (1 + (1-f)*sqrt(1 + k^2 sin^2 sigma))
        let half = sig12.scale(0.5);
        let mid = sig1.add(half);
        let mut sum = Dual::c(0.0);
        for i in 0..8 {
            let off = half.scale(GL8_HALF_NODES[i]);
            let w = GL8_HALF_WEIGHTS[i];
            for s_node in [mid.sub(off), mid.add(off)] {
                let (sn, _) = s_node.sin_cos();
                let rad = Dual::c(1.0).add(k2.mul(sn).mul(sn)).sqrt();
                let integrand = Dual::c(1.0).div(Dual::c(1.0).add(rad.scale(omf)));
                sum = sum.add(integrand.scale(w));
            }
        }
        let integral = half.mul(sum);
        let corr = salp0.scale(f * (2.0 - f)).mul(integral);
        let lam = omg12.sub(corr);
        LambdaEval {
            lam12: lam.v,
            dlam12: lam.d,
        }
    }

    fn finalize_geodesic(&self, alp1: f64) -> (f64, f64, f64, f64, f64) {
        let (salp1, calp1) = math::sin_cos(alp1);
        let salp0 = self.cbet1 * salp1;
        let p = self.cbet1 * calp1;
        let calp0 = math::hypot(p, self.sbet1);
        let q2 = math::sqrt((self.cbet2_sq_minus_cbet1_sq + p * p).max(0.0));

        // Azimuth at point 2: salp2 * cbet2 = salp0, calp2 * cbet2 = q2 >= 0.
        let (salp2, calp2) = if self.cbet2 > 0.0 {
            let norm = math::hypot(salp0, q2);
            if norm > 0.0 {
                (salp0 / norm, q2 / norm)
            } else {
                (0.0, 1.0)
            }
        } else {
            // Arriving at North Pole (beta2 == +90 deg -> cbet2 == 0).
            (0.0, 1.0)
        };

        let ssig1 = self.sbet1 / calp0;
        let csig1 = p / calp0;
        let ssig2 = self.sbet2 / calp0;
        let csig2 = q2 / calp0;

        let (sig1, sig2) = if p > 0.0 {
            let q2_minus_p = self.cbet2_sq_minus_cbet1_sq / (q2 + p);
            let ssig12 = (self.dsbet * p - self.sbet1 * q2_minus_p) / (calp0 * calp0);
            let csig12 = csig1 * csig2 + ssig1 * ssig2;
            let sig12 = math::atan2(ssig12, csig12);
            let s1 = math::atan2(ssig1, csig1);
            (s1, s1 + sig12)
        } else {
            let s1 = -core::f64::consts::PI + math::atan2(-ssig1, -csig1);
            let s2 = math::atan2(ssig2, csig2);
            (s1, s2)
        };

        let k2 = self.ellip.ep2() * calp0 * calp0;
        let dist_m = self.ellip.b_m() * integrate_i1(sig1, sig2, k2);
        (dist_m, salp1, calp1, salp2, calp2)
    }
}

fn equatorial_arch_lambda_dual(ellip: Ellipsoid, alp1: f64) -> (f64, f64) {
    let (s1, c1) = math::sin_cos(alp1);
    let salp0 = Dual { v: s1, d: c1 };
    let calp0 = Dual { v: c1, d: -s1 };
    let k2 = calp0.mul(calp0).scale(ellip.ep2());
    let f = ellip.f;
    let omf = 1.0 - f;

    let mid = 0.5 * core::f64::consts::PI;
    let half = 0.5 * core::f64::consts::PI;
    let mut sum = Dual::c(0.0);
    for i in 0..8 {
        let off = half * GL8_HALF_NODES[i];
        let w = GL8_HALF_WEIGHTS[i];
        for s_val in [mid - off, mid + off] {
            let sn = math::sin(s_val);
            let rad = Dual::c(1.0).add(k2.scale(sn * sn)).sqrt();
            let integrand = Dual::c(1.0).div(Dual::c(1.0).add(rad.scale(omf)));
            sum = sum.add(integrand.scale(w));
        }
    }
    let integral = sum.scale(half);
    let lam = Dual::c(core::f64::consts::PI).sub(salp0.scale(f * (2.0 - f)).mul(integral));
    (lam.v, lam.d)
}

/// 16-point Gauss-Legendre integration of $\int_{\sigma_1}^{\sigma_2} \sqrt{1 + k^2\sin^2\sigma}\,d\sigma$
/// with exact half-period splitting for arbitrarily long arcs.
fn integrate_i1(sig1: f64, sig2: f64, k2: f64) -> f64 {
    let dsig = sig2 - sig1;
    if dsig == 0.0 {
        return 0.0;
    }
    let pi = core::f64::consts::PI;
    let turns = math::trunc(dsig / pi);
    let mut total = 0.0;
    if turns != 0.0 {
        let half_turn_val = gl16_i1_segment(0.0, pi, k2);
        total += turns * half_turn_val;
    }
    let rem_start = sig1 + turns * pi;
    total += gl16_i1_segment(rem_start, sig2, k2);
    total
}

#[inline]
fn gl16_i1_segment(a: f64, b: f64, k2: f64) -> f64 {
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    if half == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..8 {
        let off = half * GL8_HALF_NODES[i];
        let s_lo = math::sin(mid - off);
        let s_hi = math::sin(mid + off);
        let f_lo = math::sqrt(1.0 + k2 * s_lo * s_lo);
        let f_hi = math::sqrt(1.0 + k2 * s_hi * s_hi);
        sum += GL8_HALF_WEIGHTS[i] * (f_lo + f_hi);
    }
    half * sum
}

/// 16-point Gauss-Legendre integration of $\int_{\sigma_1}^{\sigma_2} \frac{d\sigma}{1 + (1-f)\sqrt{1 + k^2\sin^2\sigma}}$.
fn integrate_lon_corr(sig1: f64, sig2: f64, k2: f64, omf: f64) -> f64 {
    let dsig = sig2 - sig1;
    if dsig == 0.0 {
        return 0.0;
    }
    let pi = core::f64::consts::PI;
    let turns = math::trunc(dsig / pi);
    let mut total = 0.0;
    if turns != 0.0 {
        total += turns * gl16_lon_segment(0.0, pi, k2, omf);
    }
    total + gl16_lon_segment(sig1 + turns * pi, sig2, k2, omf)
}

#[inline]
fn gl16_lon_segment(a: f64, b: f64, k2: f64, omf: f64) -> f64 {
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    if half == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..8 {
        let off = half * GL8_HALF_NODES[i];
        let s_lo = math::sin(mid - off);
        let s_hi = math::sin(mid + off);
        let g_lo = 1.0 / (1.0 + omf * math::sqrt(1.0 + k2 * s_lo * s_lo));
        let g_hi = 1.0 / (1.0 + omf * math::sqrt(1.0 + k2 * s_hi * s_hi));
        sum += GL8_HALF_WEIGHTS[i] * (g_lo + g_hi);
    }
    half * sum
}

/// Solves the WGS84 direct geodesic problem starting at `(lat1_deg, lon1_deg)` along `azimuth_deg`
/// clockwise from North for `distance_m` meters.
pub fn direct_geodesic(
    lat1_deg: f64,
    lon1_deg: f64,
    azimuth_deg: f64,
    distance_m: f64,
) -> DirectGeodesicResult {
    direct_geodesic_ellipsoid(
        lat1_deg,
        lon1_deg,
        azimuth_deg,
        distance_m,
        Ellipsoid::WGS84,
    )
}

/// Solves the direct geodesic problem on an arbitrary `Ellipsoid`.
pub fn direct_geodesic_ellipsoid(
    lat1_deg: f64,
    lon1_deg: f64,
    azimuth_deg: f64,
    distance_m: f64,
    ellip: Ellipsoid,
) -> DirectGeodesicResult {
    let lat1 = lat1_deg.clamp(-90.0, 90.0);
    let (sbet1, cbet1, _) = reduced_lat_sc(lat1, ellip.f);
    let (salp1, calp1) = sin_cos_deg(azimuth_deg);

    if distance_m == 0.0 {
        let az = wrap_deg_360(azimuth_deg);
        return DirectGeodesicResult {
            lat_deg: lat1,
            lon_deg: wrap_lon_180(lon1_deg),
            final_azimuth_deg: az,
            back_azimuth_deg: wrap_deg_360(az + 180.0),
            arc_deg: 0.0,
        };
    }

    let salp0 = cbet1 * salp1;
    let calp0 = math::hypot(cbet1 * calp1, sbet1);
    let k2 = ellip.ep2() * calp0 * calp0;
    let b = ellip.b_m();

    let sig1 = if cbet1 == 0.0 {
        // Starting at North or South Pole: meridian starts at sig1 = +-pi/2
        core::f64::consts::FRAC_PI_2.copysign(sbet1)
    } else {
        math::atan2(sbet1, cbet1 * calp1)
    };

    // Invert b * int_{sig1}^{sig1 + sig12} sqrt(1 + k^2 sin^2 sigma) dsigma = distance_m via Newton!
    let target_i1 = distance_m / b;
    let mut sig12 = target_i1 / math::sqrt(1.0 + 0.5 * k2);
    for _ in 0..8 {
        let f_val = integrate_i1(sig1, sig1 + sig12, k2) - target_i1;
        if f_val.abs() <= 1e-16 * (1.0 + target_i1.abs()) {
            break;
        }
        let s_end = math::sin(sig1 + sig12);
        let df = math::sqrt(1.0 + k2 * s_end * s_end);
        sig12 -= f_val / df;
    }

    let sig2 = sig1 + sig12;
    let (ssig2, csig2) = math::sin_cos(sig2);
    let sbet2 = calp0 * ssig2;
    let cbet2 = math::hypot(salp0, calp0 * csig2);
    let omf = 1.0 - ellip.f;
    let lat2_deg = math::atan2(sbet2, omf * cbet2).to_degrees();

    let (salp2, calp2) = if cbet1 == 0.0 {
        // Left a pole along meridian: longitude steps by azimuth (at South Pole) or 180 - azimuth (North Pole)
        (0.0, if sbet1 > 0.0 { -1.0 } else { 1.0 })
    } else {
        (salp0, calp0 * csig2)
    };

    let dlon_rad = if cbet1 == 0.0 {
        if sbet1 > 0.0 {
            // Left North Pole along azimuth_deg: meridian has lon = lon1 + 180 - azimuth_deg
            (180.0 - azimuth_deg).to_radians()
                + if csig2 < 0.0 {
                    0.0
                } else {
                    core::f64::consts::PI
                }
                - core::f64::consts::PI
        } else {
            azimuth_deg.to_radians()
        }
    } else {
        // Compute unwrapped auxiliary longitude change omg12 matching sig12 turns!
        let (ssig1, csig1) = math::sin_cos(sig1);
        let pi = core::f64::consts::PI;
        let turns = math::round(sig12 / pi);
        let somg1 = salp0 * ssig1;
        let comg1 = csig1;
        let somg2 = salp0 * ssig2;
        let comg2 = csig2;
        let base_omg = math::atan2(somg2 * comg1 - comg2 * somg1, comg2 * comg1 + somg2 * somg1);
        // Unwrap base_omg near sig12 * sign(salp0) when salp0 != 0
        let omg12 = if salp0 == 0.0 {
            // Pure meridian: jumps by pi each pole crossing
            turns * pi
        } else {
            let approx = salp0.signum() * sig12;
            base_omg + math::round((approx - base_omg) / (2.0 * pi)) * (2.0 * pi)
        };
        let corr = ellip.f * (2.0 - ellip.f) * salp0 * integrate_lon_corr(sig1, sig2, k2, omf);
        omg12 - corr
    };

    let lon2_deg = wrap_lon_180(lon1_deg + dlon_rad.to_degrees());
    let final_azimuth_deg = wrap_deg_360(math::atan2(salp2, calp2).to_degrees());
    let back_azimuth_deg = wrap_deg_360(math::atan2(-salp2, -calp2).to_degrees());

    DirectGeodesicResult {
        lat_deg: lat2_deg,
        lon_deg: lon2_deg,
        final_azimuth_deg,
        back_azimuth_deg,
        arc_deg: sig12.to_degrees(),
    }
}

/// Computes the exact midpoint `(lat_deg, lon_deg)` along the shorter spherical great-circle arc.
pub fn spherical_midpoint(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> (f64, f64) {
    spherical_interpolate(lat1_deg, lon1_deg, lat2_deg, lon2_deg, 0.5)
}

/// Computes `n_points` evenly spaced `(lat_deg, lon_deg)` waypoints (including start and end when
/// `n_points >= 2`) along the spherical great-circle arc from `(lat1, lon1)` to `(lat2, lon2)`.
pub fn spherical_waypoints(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    n_points: usize,
) -> Vec<(f64, f64)> {
    if n_points == 0 {
        return Vec::new();
    }
    if n_points == 1 {
        return alloc::vec![(lat1_deg, wrap_lon_180(lon1_deg))];
    }
    let mut pts = Vec::with_capacity(n_points);
    let denom = (n_points - 1) as f64;
    for i in 0..n_points {
        let frac = (i as f64) / denom;
        pts.push(spherical_interpolate(
            lat1_deg, lon1_deg, lat2_deg, lon2_deg, frac,
        ));
    }
    pts
}

/// Spherical linear interpolation (`SLERP`) along the great circle at fraction `f in [0, 1]`.
pub fn spherical_interpolate(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    fraction: f64,
) -> (f64, f64) {
    let v1 = latlon_to_unit_xyz(lat1_deg, lon1_deg);
    let v2 = latlon_to_unit_xyz(lat2_deg, lon2_deg);
    let cross = [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ];
    let sin_omega = math::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let omega = math::atan2(sin_omega, dot);

    if sin_omega < 1e-12 {
        if dot > 0.0 {
            // Nearly coincident: linear chord + renormalization
            let p = [
                (1.0 - fraction) * v1[0] + fraction * v2[0],
                (1.0 - fraction) * v1[1] + fraction * v2[1],
                (1.0 - fraction) * v1[2] + fraction * v2[2],
            ];
            return unit_xyz_to_latlon(p);
        }
        // Exact spherical antipodes: choose northward orthogonal meridian normal!
        let ortho = if v1[2].abs() < 0.9 {
            // Project +Z perpendicular to v1
            let gz = [
                -dot_z(&v1) * v1[0],
                -dot_z(&v1) * v1[1],
                1.0 - dot_z(&v1) * v1[2],
            ];
            normalize_xyz(gz)
        } else {
            normalize_xyz([1.0 - v1[0] * v1[0], -v1[0] * v1[1], -v1[0] * v1[2]])
        };
        let (s, c) = math::sin_cos(fraction * core::f64::consts::PI);
        return unit_xyz_to_latlon([
            c * v1[0] + s * ortho[0],
            c * v1[1] + s * ortho[1],
            c * v1[2] + s * ortho[2],
        ]);
    }

    let a = math::sin((1.0 - fraction) * omega) / sin_omega;
    let b = math::sin(fraction * omega) / sin_omega;
    unit_xyz_to_latlon([
        a * v1[0] + b * v2[0],
        a * v1[1] + b * v2[1],
        a * v1[2] + b * v2[2],
    ])
}

#[inline]
fn dot_z(v: &[f64; 3]) -> f64 {
    v[2]
}

#[inline]
fn normalize_xyz(v: [f64; 3]) -> [f64; 3] {
    let n = math::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    [v[0] / n, v[1] / n, v[2] / n]
}

#[inline]
fn latlon_to_unit_xyz(lat_deg: f64, lon_deg: f64) -> [f64; 3] {
    let (slat, clat) = sin_cos_deg(lat_deg);
    let (slon, clon) = sin_cos_deg(lon_deg);
    [clat * clon, clat * slon, slat]
}

#[inline]
fn unit_xyz_to_latlon(p: [f64; 3]) -> (f64, f64) {
    let r_xy = math::hypot(p[0], p[1]);
    let lat = math::atan2(p[2], r_xy).to_degrees();
    let lon = if r_xy == 0.0 {
        0.0
    } else {
        math::atan2(p[1], p[0]).to_degrees()
    };
    (lat, wrap_lon_180(lon))
}

/// Reduced latitude $(\sin\beta, \cos\beta, \operatorname{hypot}((1-f)\sin\phi, \cos\phi))$ with exact polar zeros.
#[inline]
fn reduced_lat_sc(lat_deg: f64, f: f64) -> (f64, f64, f64) {
    let (sphi, cphi) = sin_cos_deg(lat_deg);
    let sy = (1.0 - f) * sphi;
    let norm = math::hypot(sy, cphi);
    (sy / norm, cphi / norm, norm)
}

/// Exact degree-argument sine & cosine preserving exact `0.0` / `+-1.0` at multiples of `90 deg`.
#[inline]
fn sin_cos_deg(deg: f64) -> (f64, f64) {
    let r = math::rem_euclid(deg, 360.0);
    let q = math::round(r / 90.0);
    let d = r - 90.0 * q;
    let (s, c) = math::sin_cos(d.to_radians());
    match (q as i32) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[inline]
fn sin_deg(deg: f64) -> f64 {
    sin_cos_deg(deg).0
}

/// Exact signed angle difference `(lon2 - lon1)` reduced into `[-180.0, 180.0]` without
/// sub-ULP rounding loss near zero (`5e-8 deg` stays exact to 53 bits).
#[inline]
fn ang_diff_deg(lon1_deg: f64, lon2_deg: f64) -> f64 {
    let raw = lon2_deg - lon1_deg;
    if (-180.0..=180.0).contains(&raw) {
        return if raw == -180.0 { 180.0 } else { raw };
    }
    let turns = math::round(raw / 360.0);
    // Two-sum style reduction `(lon2 - turns * 360) - lon1` preserves precision across +-180 wrap
    let r = (lon2_deg - turns * 360.0) - lon1_deg;
    if r == -180.0 {
        180.0
    } else {
        r
    }
}

#[inline]
fn wrap_lon_180(deg: f64) -> f64 {
    if (-180.0..=180.0).contains(&deg) {
        return if deg == -180.0 && deg.is_sign_positive() {
            180.0
        } else {
            deg
        };
    }
    let r = deg - 360.0 * math::round(deg / 360.0);
    if r == -180.0 && deg > 0.0 {
        180.0
    } else {
        r
    }
}

#[inline]
fn wrap_deg_360(deg: f64) -> f64 {
    let r = math::rem_euclid(deg, 360.0);
    if r >= 360.0 || r == 0.0 {
        0.0
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn karney_reference_vectors_and_antipodes() {
        // Classic Karney (2013) Table / GeographicLib benchmark vector:
        // (-30.0, 0.0) -> (29.9, 179.8) (near-antipodal conjugate astroid!)
        // GeographicLib exact WGS84:
        // s12 = 19989832.82761036 m, azi1 = 161.8905247363307 deg, azi2 = 18.0907372457452 deg -> baz = 198.0907372457452 deg
        let r_antip = gps2dist_azimuth(-30.0, 0.0, 29.9, 179.8);
        assert!(
            (r_antip.distance_m - 19_989_832.827_610_36).abs() < 1e-6,
            "got {}",
            r_antip.distance_m
        );
        assert!(
            (r_antip.azimuth_deg - 161.890_524_736_330_7).abs() < 1e-9,
            "got {}",
            r_antip.azimuth_deg
        );
        assert!(
            (r_antip.back_azimuth_deg - 198.090_737_245_745_2).abs() < 1e-9,
            "got {}",
            r_antip.back_azimuth_deg
        );

        // Quarter meridian (0, 0) -> (90, 0): WGS84 exact meridional quadrant = 10_001_965.729312722 m
        let r_pole = gps2dist_azimuth(0.0, 0.0, 90.0, 0.0);
        assert!((r_pole.distance_m - 10_001_965.729_312_722).abs() < 1e-7);
        assert!((r_pole.azimuth_deg - 0.0).abs() < 1e-12);
        assert!((r_pole.back_azimuth_deg - 180.0).abs() < 1e-12);

        // Equatorial short & near-antipodal (0, 0) -> (0, 179.5): arches over pole!
        // Wait: (1 - f) * 180 = 179.396494... deg, so 179.5 deg > 179.3965 deg leaves equator!
        let r_eq_arch = gps2dist_azimuth(0.0, 0.0, 0.0, 179.5);
        assert!(r_eq_arch.distance_m < Ellipsoid::WGS84.a_m * (179.5_f64).to_radians());
        assert!(r_eq_arch.azimuth_deg > 0.0 && r_eq_arch.azimuth_deg < 90.0);

        // Exact equatorial antipodes (0, 0) -> (0, 180): twice quarter meridian = 20_003_931.458625445 m
        let r_eq_antip = gps2dist_azimuth(0.0, 0.0, 0.0, 180.0);
        assert!((r_eq_antip.distance_m - 20_003_931.458_625_445).abs() < 1e-7);

        // Near-coincident (~0.14 m apart): check inverse <-> direct round-trip to sub-nanometer!
        let r_tiny = gps2dist_azimuth(35.0, -120.0, 35.000_001, -119.999_999);
        assert!(r_tiny.distance_m > 0.1 && r_tiny.distance_m < 0.2);
        let dir_tiny = direct_geodesic(35.0, -120.0, r_tiny.azimuth_deg, r_tiny.distance_m);
        assert!((dir_tiny.lat_deg - 35.000_001).abs() < 1e-13);
        assert!((dir_tiny.lon_deg - (-119.999_999)).abs() < 1e-13);
    }

    #[test]
    fn inverse_direct_roundtrip_grid() {
        let pairs = [
            (38.640, 43.400, 49.144, 12.878),     // Van earthquake -> Wettzell
            (38.297, 142.373, 34.9459, -106.457), // Tohoku 2011 -> IU.ANMO
            (-54.4, -71.2, 64.873, -147.861),     // Tierra del Fuego -> IU.COLA
            (-30.0, 0.0, 29.9999, 179.9999),      // Deep inside antipodal astroid
            (89.9, 45.0, -89.8, -134.5),          // Polar-to-polar
            (-45.0, 170.0, -45.0, -170.0),        // Antimeridian crossing
        ];
        let lat_lon_tol = if cfg!(miri) { 1e-8 } else { 1e-11 };
        let sym_dist_tol = if cfg!(miri) { 1e-6 } else { 1e-9 };
        let sym_az_tol = if cfg!(miri) { 1e-8 } else { 1e-10 };
        for &(lat1, lon1, lat2, lon2) in &pairs {
            let inv = gps2dist_azimuth(lat1, lon1, lat2, lon2);
            let dir = direct_geodesic(lat1, lon1, inv.azimuth_deg, inv.distance_m);
            assert!(
                (dir.lat_deg - lat2).abs() < lat_lon_tol,
                "lat mismatch for ({lat1},{lon1})->({lat2},{lon2}): got {}",
                dir.lat_deg
            );
            assert!(
                ang_diff_deg(dir.lon_deg, lon2).abs() < lat_lon_tol,
                "lon mismatch for ({lat1},{lon1})->({lat2},{lon2}): got {}",
                dir.lon_deg
            );
            // Reverse symmetry check
            let rev = gps2dist_azimuth(lat2, lon2, lat1, lon1);
            assert!((rev.distance_m - inv.distance_m).abs() < sym_dist_tol);
            assert!((rev.azimuth_deg - inv.back_azimuth_deg).abs() < sym_az_tol);
            assert!((rev.back_azimuth_deg - inv.azimuth_deg).abs() < sym_az_tol);
        }
    }

    #[test]
    fn obspy_geographiclib_25_golden_pairs() {
        // Exact 25 reference vectors generated by obspy.geodetics.gps2dist_azimuth & locations2degrees:
        // (lat1, lon1, lat2, lon2, dist_m, az_deg, baz_deg, spherical_dist_deg)
        #[allow(clippy::excessive_precision)]
        let golden: &[[f64; 8]] = &[
            [
                35.7695,
                -117.5993,
                34.1484,
                -118.1711,
                187272.61363770146,
                196.35710171095963,
                16.029431650238507,
                1.6874651998601158,
            ],
            [
                38.297,
                142.373,
                36.5417,
                138.2039,
                417251.5656356074,
                243.4500440313435,
                60.91570875793451,
                3.7470744486501744,
            ],
            [
                47.3769,
                8.5417,
                48.1351,
                11.582,
                243018.9379935025,
                68.58362093664441,
                250.83460895449298,
                2.1798679606003866,
            ],
            [
                -33.4489,
                -70.6693,
                -32.8895,
                -68.8458,
                181044.55745757444,
                70.46022624972619,
                249.46249387774066,
                1.6256234866785102,
            ],
            [
                38.297,
                142.373,
                34.94598,
                -106.45713,
                9244669.364680447,
                50.33531054485132,
                312.5089818795567,
                82.95378175889837,
            ],
            [
                -35.846,
                -72.719,
                48.3301,
                8.3296,
                12290858.150267815,
                44.73642256777512,
                239.0409251400997,
                110.70700618879994,
            ],
            [
                3.295,
                95.982,
                42.5064,
                -71.5583,
                14779911.169347636,
                347.526332037086,
                16.981266709034003,
                132.82908752983283,
            ],
            [
                46.592,
                153.266,
                -35.3187,
                148.9963,
                9081926.697675403,
                183.53220558890774,
                2.976195887252601,
                82.00074815716054,
            ],
            [
                64.8736,
                -147.8616,
                21.42,
                -158.0112,
                4885037.94085824,
                193.69595909081937,
                6.2141538591000085,
                43.96650874841913,
            ],
            [
                35.6762,
                139.6503,
                -33.8688,
                151.2093,
                7792174.827115238,
                169.77466835911878,
                349.9975177879703,
                70.37927765834007,
            ],
            [
                -33.9249,
                18.4241,
                51.5074,
                -0.1278,
                9636428.841325592,
                348.50728027157294,
                164.61208933939105,
                86.97339960544333,
            ],
            [
                -17.7134,
                178.065,
                51.88,
                -176.658,
                7726305.78787892,
                3.4889221505150494,
                185.37893754244382,
                69.74568551263596,
            ],
            [
                -0.1807,
                -78.4678,
                1.3521,
                103.8198,
                19743445.680717908,
                303.14785681667536,
                56.875987895388874,
                177.4301414485363,
            ],
            [
                78.2232,
                15.6267,
                82.5018,
                -62.3481,
                1400962.7008588647,
                324.008149451007,
                66.78967397322891,
                12.544384207271014,
            ],
            [
                89.5,
                10.0,
                89.5,
                -170.0,
                111693.95089655284,
                0.0,
                0.0,
                0.9999999999999957,
            ],
            [
                -89.9289,
                144.4382,
                -66.2833,
                110.5167,
                2640949.5790527915,
                325.98789473011067,
                179.9010658788741,
                23.657732362170147,
            ],
            [90.0, 0.0, 0.0, 45.0, 10001965.729312724, 135.0, 360.0, 90.0],
            [
                34.94598, -106.45713, 34.94598, -106.45713, 0.0, 180.0, 360.0, 0.0,
            ],
            [
                34.94598,
                -106.45713,
                34.945989,
                -106.45713,
                0.9984562725672707,
                0.0,
                180.0,
                9.000000006961076e-6,
            ],
            [
                0.0,
                0.0,
                5e-8,
                5e-8,
                0.007845173596652742,
                45.19242321515819,
                225.1924232151582,
                7.071067811865474e-8,
            ],
            [
                -23.665,
                133.905,
                -23.6652,
                133.9053,
                37.77906786284545,
                125.89646673168335,
                305.8963463146972,
                0.0003398525407548675,
            ],
            [
                0.0,
                0.0,
                0.001,
                179.999,
                20003820.79219665,
                0.09486171375751155,
                359.9051382862281,
                179.99858578643767,
            ],
            [
                0.0,
                0.0,
                0.0,
                179.7,
                19995624.889961265,
                29.828768395683454,
                330.1712316043165,
                179.7,
            ],
            [
                40.4168,
                -3.7038,
                -41.2865,
                174.7762,
                19854610.375957005,
                134.27522291982757,
                226.50368614165228,
                178.558418562012,
            ],
            [
                35.0,
                140.0,
                -34.9995,
                -39.9995,
                20003875.96529261,
                359.9421372544226,
                0.057862393605148554,
                179.9993536614332,
            ],
        ];

        for (idx, &[lat1, lon1, lat2, lon2, exp_d, exp_az, exp_baz, exp_sph]) in
            golden.iter().enumerate()
        {
            let res = gps2dist_azimuth(lat1, lon1, lat2, lon2);
            assert!(
                (res.distance_m - exp_d).abs() <= 1e-6,
                "pair #{idx}: dist got {} vs expected {}",
                res.distance_m,
                exp_d
            );
            assert!(
                (res.distance_deg_spherical - exp_sph).abs() <= 1e-9,
                "pair #{idx}: sph got {} vs expected {}",
                res.distance_deg_spherical,
                exp_sph
            );
            if exp_d > 0.0 {
                assert!(
                    ang_diff_deg(res.azimuth_deg, exp_az).abs() <= 1e-9,
                    "pair #{idx}: az got {} vs expected {}",
                    res.azimuth_deg,
                    exp_az
                );
                assert!(
                    ang_diff_deg(res.back_azimuth_deg, exp_baz).abs() <= 1e-9,
                    "pair #{idx}: baz got {} vs expected {}",
                    res.back_azimuth_deg,
                    exp_baz
                );
            }
        }
    }

    #[test]
    fn geocentric_latitude_and_waypoints() {
        for &lat in &[-90.0, -45.0, -12.345, 0.0, 35.678, 45.0, 90.0] {
            let geoc = geographic_to_geocentric_lat(lat);
            let back = geocentric_to_geographic_lat(geoc);
            assert!((back - lat).abs() < 1e-13);
        }
        let mid = spherical_midpoint(0.0, -10.0, 0.0, 30.0);
        assert!(mid.0.abs() < 1e-13);
        assert!((mid.1 - 10.0).abs() < 1e-13);

        let wps = spherical_waypoints(10.0, 20.0, -20.0, 80.0, 5);
        assert_eq!(wps.len(), 5);
        let total = locations2degrees(10.0, 20.0, -20.0, 80.0);
        for (i, &(la, lo)) in wps.iter().enumerate() {
            let d = locations2degrees(10.0, 20.0, la, lo);
            assert!((d - total * (i as f64) / 4.0).abs() < 1e-12);
        }
    }
}
