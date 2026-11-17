//! Seismic moment tensor algebra, symmetric 3x3 Jacobi eigensolver, `ISO + DC + CLVD`
//! decomposition, Aki & Richards fault-plane conversions, and focal-sphere beachball projection.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{FRAC_1_SQRT_2, PI};
use earthsnail_core::event::{MomentTensor6, NodalPlane, PrincipalAxes, PrincipalAxis};
use earthsnail_core::math::{acos, asin, atan, atan2, cos, hypot, sin, sqrt};
use earthsnail_core::{EarthSnailError, Result};

#[inline]
fn abs(x: f64) -> f64 {
    x.abs()
}

const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;

/// Projection convention for lower-hemisphere focal mechanism ("beachball") grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeachballProjection {
    /// Schmidt equal-area lower-hemisphere projection (`r = sqrt(2) * sin(takeoff / 2)`).
    LambertEqualArea,
    /// Wulff stereographic lower-hemisphere projection (`r = tan(takeoff / 2)`).
    Stereographic,
}

/// Complete `ISO + DC + CLVD` decomposition of a symmetric 6-component moment tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentTensorDecomposition {
    /// Eigenvalues sorted descending `lambda_1 >= lambda_2 >= lambda_3` in `N*m`.
    pub eigenvalues_nm: [f64; 3],
    /// Principal `T`, `P`, and `N` (`B`) axes in lower-hemisphere trend/plunge convention.
    pub principal_axes: PrincipalAxes,
    /// Signed isotropic percentage in `[-100, 100]` (`+` explosion, `-` implosion).
    pub iso_percent: f64,
    /// Non-negative double-couple percentage in `[0, 100]`.
    pub dc_percent: f64,
    /// Signed compensated linear vector dipole percentage in `[-100, 100]`.
    pub clvd_percent: f64,
    /// Pure isotropic sub-tensor `M_ISO`.
    pub m_iso: MomentTensor6,
    /// Pure double-couple sub-tensor `M_DC`.
    pub m_dc: MomentTensor6,
    /// Pure CLVD sub-tensor `M_CLVD` (`m_iso + m_dc + m_clvd == mt`).
    pub m_clvd: MomentTensor6,
    /// Best-fitting double-couple primary and auxiliary fault nodal planes.
    pub nodal_planes: (NodalPlane, NodalPlane),
}

/// Sampled lower-hemisphere P-wave radiation pattern grid on `[-1, 1] x [-1, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct BeachballGrid {
    /// Grid dimension (`size * size` samples, row-major from South `-1` to North `+1` or `y` index).
    pub size: usize,
    /// Far-field P-wave radiation amplitude `gamma^T * M_NED * gamma` (`0.0` outside unit disk).
    pub amplitudes: Vec<f64>,
    /// `true` when the grid pixel lies inside the unit focal circle (`x^2 + y^2 <= 1`).
    pub inside_mask: Vec<bool>,
}

/// Deterministic cyclic 3x3 real symmetric Jacobi eigensolver.
///
/// Returns `(eigenvalues_desc, eigenvectors_desc)` where `eigenvalues_desc[0] >= eigenvalues_desc[1] >= eigenvalues_desc[2]`
/// and `eigenvectors_desc[k]` is the unit 3-vector `[v_0, v_1, v_2]` corresponding to `eigenvalues_desc[k]`,
/// oriented to form a right-handed orthonormal basis (`det > 0`).
pub fn jacobi_eigen_3x3(mat: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = [
        [
            mat[0][0],
            0.5 * (mat[0][1] + mat[1][0]),
            0.5 * (mat[0][2] + mat[2][0]),
        ],
        [
            0.5 * (mat[0][1] + mat[1][0]),
            mat[1][1],
            0.5 * (mat[1][2] + mat[2][1]),
        ],
        [
            0.5 * (mat[0][2] + mat[2][0]),
            0.5 * (mat[1][2] + mat[2][1]),
            mat[2][2],
        ],
    ];
    // Columns of `v` hold eigenvectors during sweeps.
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let scale = (abs(a[0][0])
        + abs(a[1][1])
        + abs(a[2][2])
        + 2.0 * (abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])))
    .max(1e-300);

    for _sweep in 0..50 {
        let off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2]);
        if off <= 1e-15 * scale {
            break;
        }
        for &(p, q) in &[(0usize, 1usize), (0usize, 2usize), (1usize, 2usize)] {
            let apq = a[p][q];
            if abs(apq) <= 1e-18 * scale {
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                continue;
            }
            let diff = a[q][q] - a[p][p];
            let t = if abs(apq) <= 1e-18 * abs(diff) {
                apq / diff
            } else {
                let tau = diff / (2.0 * apq);
                let root = hypot(1.0, tau);
                if tau >= 0.0 {
                    1.0 / (tau + root)
                } else {
                    -1.0 / (-tau + root)
                }
            };
            let c = 1.0 / hypot(1.0, t);
            let s = t * c;
            let tau_rot = s / (1.0 + c);

            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            let r = 3 - p - q;
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = arp - s * (arq + tau_rot * arp);
            a[p][r] = a[r][p];
            a[r][q] = arq + s * (arp - tau_rot * arq);
            a[q][r] = a[r][q];

            for row in &mut v {
                let vp = row[p];
                let vq = row[q];
                row[p] = vp - s * (vq + tau_rot * vp);
                row[q] = vq + s * (vp - tau_rot * vq);
            }
        }
    }

    let mut evals = [a[0][0], a[1][1], a[2][2]];
    let mut evecs = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];

    // Normalize each eigenvector and sort descending by eigenvalue.
    for vec_k in &mut evecs {
        let nrm = sqrt(vec_k[0] * vec_k[0] + vec_k[1] * vec_k[1] + vec_k[2] * vec_k[2]);
        if nrm > 0.0 {
            vec_k[0] /= nrm;
            vec_k[1] /= nrm;
            vec_k[2] /= nrm;
        }
    }
    for &(i, j) in &[(0usize, 1usize), (1usize, 2usize), (0usize, 1usize)] {
        if evals[j] > evals[i] {
            evals.swap(i, j);
            evecs.swap(i, j);
        }
    }

    // Enforce right-handed basis det([e0, e1, e2]) > 0.
    let cross_0 = evecs[0][1] * evecs[1][2] - evecs[0][2] * evecs[1][1];
    let cross_1 = evecs[0][2] * evecs[1][0] - evecs[0][0] * evecs[1][2];
    let cross_2 = evecs[0][0] * evecs[1][1] - evecs[0][1] * evecs[1][0];
    let det = cross_0 * evecs[2][0] + cross_1 * evecs[2][1] + cross_2 * evecs[2][2];
    if det < 0.0 {
        evecs[2][0] = -evecs[2][0];
        evecs[2][1] = -evecs[2][1];
        evecs[2][2] = -evecs[2][2];
    }

    (evals, evecs)
}

#[inline]
fn clean_angle_360(deg: f64) -> f64 {
    let mut r = deg % 360.0;
    if r < 0.0 {
        r += 360.0;
    }
    if abs(r - 360.0) < 1e-12 || abs(r) < 1e-12 {
        0.0
    } else {
        r
    }
}

fn ned_vec_to_principal_axis(vec_ned: [f64; 3], length: f64) -> PrincipalAxis {
    // Lower-hemisphere convention: ensure `v_D >= 0`.
    let (vn, ve, vd) = if vec_ned[2] < -1e-15
        || (abs(vec_ned[2]) <= 1e-15
            && (vec_ned[0] < -1e-15 || (abs(vec_ned[0]) <= 1e-15 && vec_ned[1] < 0.0)))
    {
        (-vec_ned[0], -vec_ned[1], -vec_ned[2])
    } else {
        (vec_ned[0], vec_ned[1], vec_ned[2])
    };
    let horiz = hypot(vn, ve);
    let plunge_deg = (atan2(vd.max(0.0), horiz) * RAD_TO_DEG).clamp(0.0, 90.0);
    let azimuth_deg = if horiz <= 1e-12 {
        0.0
    } else {
        clean_angle_360(atan2(ve, vn) * RAD_TO_DEG)
    };
    PrincipalAxis {
        azimuth_deg,
        plunge_deg,
        length,
    }
}

fn reconstruct_mt_from_ned_spectral(evals: [f64; 3], evecs: [[f64; 3]; 3]) -> MomentTensor6 {
    let mut m = [[0.0_f64; 3]; 3];
    for k in 0..3 {
        let lam = evals[k];
        let v = evecs[k];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += lam * v[r] * v[c];
            }
        }
    }
    MomentTensor6::from_ned(m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2])
}

/// Convert Aki & Richards `(strike, dip, rake)` fault geometry and scalar seismic moment `M_0` (`N*m`)
/// into a Harvard/GCMT/QuakeML [`MomentTensor6`].
pub fn strike_dip_rake_to_mt(plane: NodalPlane, scalar_moment_nm: f64) -> Result<MomentTensor6> {
    if !scalar_moment_nm.is_finite() || scalar_moment_nm < 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "scalar seismic moment must be finite and non-negative".into(),
        ));
    }
    if !plane.strike_deg.is_finite()
        || !plane.dip_deg.is_finite()
        || !plane.rake_deg.is_finite()
        || !(0.0..=90.0).contains(&plane.dip_deg)
    {
        return Err(EarthSnailError::InvalidParameter(
            "strike/rake must be finite and dip_deg must lie in [0, 90]".into(),
        ));
    }

    let phi = plane.strike_deg * DEG_TO_RAD;
    let delta = plane.dip_deg * DEG_TO_RAD;
    let lam = plane.rake_deg * DEG_TO_RAD;

    let (sp, cp) = (sin(phi), cos(phi));
    let (sd, cd) = (sin(delta), cos(delta));
    let (sl, cl) = (sin(lam), cos(lam));

    // Fault normal `n` and hanging-wall slip vector `d` in `(North, East, Down)`.
    let n = [-sd * sp, sd * cp, -cd];
    let d = [cl * cp + cd * sl * sp, cl * sp - cd * sl * cp, -sl * sd];

    let m0 = scalar_moment_nm;
    let m_nn = 2.0 * m0 * n[0] * d[0];
    let m_ee = 2.0 * m0 * n[1] * d[1];
    let m_dd = 2.0 * m0 * n[2] * d[2];
    let m_ne = m0 * (n[0] * d[1] + n[1] * d[0]);
    let m_nd = m0 * (n[0] * d[2] + n[2] * d[0]);
    let m_ed = m0 * (n[1] * d[2] + n[2] * d[1]);

    Ok(MomentTensor6::from_ned(m_nn, m_ee, m_dd, m_ne, m_nd, m_ed))
}

/// Convert orthogonal unit fault-normal `n` and slip `d` vectors (in `North, East, Down`)
/// into Aki & Richards `(strike, dip, rake)`.
pub fn normal_slip_to_nodal_plane(mut n: [f64; 3], mut d: [f64; 3]) -> NodalPlane {
    // Fault normal must point into upper hemisphere (`n_D <= 0`).
    // For strictly vertical faults (`n_D == 0`), pick deterministic sign where `(n_E, -n_N)` is canonical.
    if n[2] > 1e-14 || (abs(n[2]) <= 1e-14 && (n[1] < -1e-14 || (abs(n[1]) <= 1e-14 && n[0] > 0.0)))
    {
        n = [-n[0], -n[1], -n[2]];
        d = [-d[0], -d[1], -d[2]];
    }

    let cos_dip = (-n[2]).clamp(0.0, 1.0);
    let dip_rad = acos(cos_dip);
    let sin_dip = hypot(n[0], n[1]);

    if sin_dip <= 1e-10 {
        // Horizontal fault (`dip == 0`): choose `strike = 0`, `d = (cos(rake), -sin(rake), 0)`.
        let mut rake_deg = atan2(-d[1], d[0]) * RAD_TO_DEG;
        if rake_deg <= -180.0 + 1e-12 {
            rake_deg = 180.0;
        }
        return NodalPlane {
            strike_deg: 0.0,
            dip_deg: 0.0,
            rake_deg,
        };
    }

    let strike_rad = atan2(-n[0], n[1]);
    let strike_deg = clean_angle_360(strike_rad * RAD_TO_DEG);
    let phi = strike_deg * DEG_TO_RAD;
    let (sp, cp) = (sin(phi), cos(phi));
    let (sd, cd) = (sin(dip_rad), cos(dip_rad));

    // Orthonormal in-plane basis: along-strike `s_hat` and up-dip `u_hat`.
    let s_hat = [cp, sp, 0.0];
    let u_hat = [cd * sp, -cd * cp, -sd];

    let cos_rake = d[0] * s_hat[0] + d[1] * s_hat[1] + d[2] * s_hat[2];
    let sin_rake = d[0] * u_hat[0] + d[1] * u_hat[1] + d[2] * u_hat[2];
    let mut rake_deg = atan2(sin_rake, cos_rake) * RAD_TO_DEG;
    if rake_deg <= -180.0 + 1e-12 {
        rake_deg = 180.0;
    }

    NodalPlane {
        strike_deg,
        dip_deg: (dip_rad * RAD_TO_DEG).clamp(0.0, 90.0),
        rake_deg: rake_deg.clamp(-180.0, 180.0),
    }
}

/// Compute the conjugate auxiliary fault plane `(strike_2, dip_2, rake_2)` for a double-couple fault plane.
pub fn auxiliary_plane(plane: NodalPlane) -> Result<NodalPlane> {
    if !plane.strike_deg.is_finite()
        || !plane.dip_deg.is_finite()
        || !plane.rake_deg.is_finite()
        || !(0.0..=90.0).contains(&plane.dip_deg)
    {
        return Err(EarthSnailError::InvalidParameter(
            "strike/rake must be finite and dip_deg must lie in [0, 90]".into(),
        ));
    }
    let phi = plane.strike_deg * DEG_TO_RAD;
    let delta = plane.dip_deg * DEG_TO_RAD;
    let lam = plane.rake_deg * DEG_TO_RAD;
    let (sp, cp) = (sin(phi), cos(phi));
    let (sd, cd) = (sin(delta), cos(delta));
    let (sl, cl) = (sin(lam), cos(lam));

    let n = [-sd * sp, sd * cp, -cd];
    let d = [cl * cp + cd * sl * sp, cl * sp - cd * sl * cp, -sl * sd];
    // Swapping normal `n` and slip `d` yields the exact conjugate auxiliary plane!
    Ok(normal_slip_to_nodal_plane(d, n))
}

/// Extract principal `T`, `P`, and `N` (`B`) axes from a [`MomentTensor6`].
pub fn mt_to_principal_axes(mt: &MomentTensor6) -> PrincipalAxes {
    let (evals, evecs) = jacobi_eigen_3x3(mt.to_ned_matrix());
    PrincipalAxes {
        t_axis: ned_vec_to_principal_axis(evecs[0], evals[0]),
        n_axis: Some(ned_vec_to_principal_axis(evecs[1], evals[1])),
        p_axis: ned_vec_to_principal_axis(evecs[2], evals[2]),
    }
}

/// Extract the best-fitting double-couple conjugate nodal planes `(plane_1, plane_2)` from a [`MomentTensor6`].
pub fn mt_to_nodal_planes(mt: &MomentTensor6) -> (NodalPlane, NodalPlane) {
    let (_evals, evecs) = jacobi_eigen_3x3(mt.to_ned_matrix());
    let t_vec = evecs[0];
    let p_vec = evecs[2];
    let v1 = [
        (t_vec[0] + p_vec[0]) * FRAC_1_SQRT_2,
        (t_vec[1] + p_vec[1]) * FRAC_1_SQRT_2,
        (t_vec[2] + p_vec[2]) * FRAC_1_SQRT_2,
    ];
    let v2 = [
        (t_vec[0] - p_vec[0]) * FRAC_1_SQRT_2,
        (t_vec[1] - p_vec[1]) * FRAC_1_SQRT_2,
        (t_vec[2] - p_vec[2]) * FRAC_1_SQRT_2,
    ];
    (
        normal_slip_to_nodal_plane(v1, v2),
        normal_slip_to_nodal_plane(v2, v1),
    )
}

/// Decompose a [`MomentTensor6`] into exact `ISO + DC + CLVD` percentages, component sub-tensors,
/// principal `T`/`P`/`N` axes, and best-fitting double-couple nodal planes.
pub fn decompose_moment_tensor(mt: &MomentTensor6) -> MomentTensorDecomposition {
    let (evals, evecs) = jacobi_eigen_3x3(mt.to_ned_matrix());
    let principal_axes = PrincipalAxes {
        t_axis: ned_vec_to_principal_axis(evecs[0], evals[0]),
        n_axis: Some(ned_vec_to_principal_axis(evecs[1], evals[1])),
        p_axis: ned_vec_to_principal_axis(evecs[2], evals[2]),
    };

    let m_iso_scalar = (evals[0] + evals[1] + evals[2]) / 3.0;
    let d0 = evals[0] - m_iso_scalar;
    let d1 = evals[1] - m_iso_scalar;
    let d2 = evals[2] - m_iso_scalar;

    let (dc_evals, clvd_evals, dc_mag) = if d1 <= 0.0 {
        // `d0 >= 0 >= d1 >= d2`, `d0 == -d1 - d2 = |d|_max`
        let m_dc = (d1 - d2).max(0.0);
        ([m_dc, 0.0, -m_dc], [-2.0 * d1, d1, d1], m_dc)
    } else {
        // `d0 >= d1 > 0 >= d2`, `-d2 == d0 + d1 = |d|_max`
        let m_dc = (d0 - d1).max(0.0);
        ([m_dc, 0.0, -m_dc], [d1, d1, -2.0 * d1], m_dc)
    };

    let clvd_signed = -2.0 * d1;
    let m_tot = abs(m_iso_scalar) + dc_mag + abs(clvd_signed);
    let (iso_percent, dc_percent, clvd_percent) = if m_tot <= 1e-300 {
        (0.0, 0.0, 0.0)
    } else {
        (
            100.0 * m_iso_scalar / m_tot,
            100.0 * dc_mag / m_tot,
            100.0 * clvd_signed / m_tot,
        )
    };

    let m_iso = MomentTensor6::new(m_iso_scalar, m_iso_scalar, m_iso_scalar, 0.0, 0.0, 0.0);
    let m_dc = reconstruct_mt_from_ned_spectral(dc_evals, evecs);
    let m_clvd = reconstruct_mt_from_ned_spectral(clvd_evals, evecs);

    let t_vec = evecs[0];
    let p_vec = evecs[2];
    let v1 = [
        (t_vec[0] + p_vec[0]) * FRAC_1_SQRT_2,
        (t_vec[1] + p_vec[1]) * FRAC_1_SQRT_2,
        (t_vec[2] + p_vec[2]) * FRAC_1_SQRT_2,
    ];
    let v2 = [
        (t_vec[0] - p_vec[0]) * FRAC_1_SQRT_2,
        (t_vec[1] - p_vec[1]) * FRAC_1_SQRT_2,
        (t_vec[2] - p_vec[2]) * FRAC_1_SQRT_2,
    ];
    let nodal_planes = (
        normal_slip_to_nodal_plane(v1, v2),
        normal_slip_to_nodal_plane(v2, v1),
    );

    MomentTensorDecomposition {
        eigenvalues_nm: evals,
        principal_axes,
        iso_percent,
        dc_percent,
        clvd_percent,
        m_iso,
        m_dc,
        m_clvd,
        nodal_planes,
    }
}

/// Evaluate far-field P-wave radiation amplitude `gamma^T * M_NED * gamma` at `(azimuth_deg, takeoff_deg)`.
///
/// `takeoff_deg` is measured from downward vertical (`0°` = straight down, `90°` = horizontal).
pub fn p_radiation_amplitude(mt: &MomentTensor6, azimuth_deg: f64, takeoff_deg: f64) -> f64 {
    let az = azimuth_deg * DEG_TO_RAD;
    let ih = takeoff_deg * DEG_TO_RAD;
    let s_ih = sin(ih);
    let g = [s_ih * cos(az), s_ih * sin(az), cos(ih)];
    let m = mt.to_ned_matrix();
    let mut amp = 0.0_f64;
    for r in 0..3 {
        for c in 0..3 {
            amp += g[r] * m[r][c] * g[c];
        }
    }
    amp
}

/// Render a lower-hemisphere P-wave radiation amplitude grid (`size x size`) on `[-1, 1]^2`
/// for beachball visualization using either Schmidt equal-area or Wulff stereographic projection.
///
/// Row index `iy = 0 .. size` maps `y = -1 .. +1` (South to North); column index `ix = 0 .. size`
/// maps `x = -1 .. +1` (West to East).
pub fn beachball_p_radiation_grid(
    mt: &MomentTensor6,
    size: usize,
    projection: BeachballProjection,
) -> Result<BeachballGrid> {
    if size < 2 {
        return Err(EarthSnailError::InvalidParameter(
            "beachball grid size must be at least 2".into(),
        ));
    }
    let m = mt.to_ned_matrix();
    let total = size * size;
    let mut amplitudes = vec![0.0_f64; total];
    let mut inside_mask = vec![false; total];
    let inv_span = 2.0 / ((size - 1) as f64);

    for iy in 0..size {
        let y = -1.0 + (iy as f64) * inv_span; // North
        for ix in 0..size {
            let x = -1.0 + (ix as f64) * inv_span; // East
            let r2 = x * x + y * y;
            let idx = iy * size + ix;
            if r2 > 1.0 + 1e-12 {
                continue;
            }
            inside_mask[idx] = true;
            let r = sqrt(r2.min(1.0));
            let ih = match projection {
                BeachballProjection::LambertEqualArea => {
                    2.0 * asin((r * FRAC_1_SQRT_2).clamp(0.0, 1.0))
                }
                BeachballProjection::Stereographic => 2.0 * atan(r),
            };
            let az = if r <= 1e-15 { 0.0 } else { atan2(x, y) };
            let s_ih = sin(ih);
            let g = [s_ih * cos(az), s_ih * sin(az), cos(ih)];
            let mut amp = 0.0_f64;
            for r_i in 0..3 {
                for c_i in 0..3 {
                    amp += g[r_i] * m[r_i][c_i] * g[c_i];
                }
            }
            amplitudes[idx] = amp;
        }
    }

    Ok(BeachballGrid {
        size,
        amplitudes,
        inside_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_roundtrip_and_beachball_quadrants() {
        // 45-deg pure thrust fault striking North: T-axis vertical (center compressional > 0).
        let thrust = NodalPlane {
            strike_deg: 0.0,
            dip_deg: 45.0,
            rake_deg: 90.0,
        };
        let m0 = 1.0e18;
        let mt = strike_dip_rake_to_mt(thrust, m0).unwrap();
        assert!(abs(mt.scalar_moment_nm() - m0) < 1e-4 * m0);

        let dec = decompose_moment_tensor(&mt);
        assert!(abs(dec.iso_percent) < 1e-10);
        assert!(abs(dec.clvd_percent) < 1e-10);
        assert!(abs(dec.dc_percent - 100.0) < 1e-10);
        assert!(abs(dec.principal_axes.t_axis.plunge_deg - 90.0) < 1e-8);

        // Both recovered nodal planes must generate the exact same MomentTensor6!
        let mt_p1 = strike_dip_rake_to_mt(dec.nodal_planes.0, m0).unwrap();
        let mt_p2 = strike_dip_rake_to_mt(dec.nodal_planes.1, m0).unwrap();
        assert!(abs(mt_p1.m_rr - mt.m_rr) < 1e-9 * m0);
        assert!(abs(mt_p2.m_rr - mt.m_rr) < 1e-9 * m0);
        assert!(abs(mt_p1.m_pp - mt.m_pp) < 1e-9 * m0);
        assert!(abs(mt_p2.m_pp - mt.m_pp) < 1e-9 * m0);

        let grid =
            beachball_p_radiation_grid(&mt, 21, BeachballProjection::LambertEqualArea).unwrap();
        let center = grid.amplitudes[10 * 21 + 10];
        assert!(center > 0.99 * m0);
    }
}
