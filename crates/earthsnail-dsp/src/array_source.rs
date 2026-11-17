//! 3-Component rotations, Flinn/Jurkevics covariance polarization, receiver functions
//! (water-level & Ligorría–Ammon iterative), Silver–Chan shear-wave splitting,
//! Conventional/Capon MVDR FK array beamforming, vespagrams, and Nigam–Jennings
//! engineering ground-motion intensity & response spectra.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;
use earthsnail_core::math::{atan2, cos, exp, floor, hypot, log10, powf, sin, sqrt};
use earthsnail_core::{Complex64, EarthSnailError, Result, ScratchWorkspace, Trace};

use crate::fft::{irfft_with_scratch, next_pow2, rfft_with_scratch, rfftfreq};
use crate::moment_tensor::jacobi_eigen_3x3;
use crate::preprocess::{cosine_taper, integrate_trapezoidal};
use crate::spectral::analytic_signal_with_scratch;

#[inline]
fn abs(x: f64) -> f64 {
    x.abs()
}

const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;
const STANDARD_GRAVITY_M_S2: f64 = 9.80665;

// ============================================================================
// 1. 3-Component Coordinate Rotations
// ============================================================================

/// Rotate horizontal `(North, East)` seismograms into `(Radial, Transverse)` given `back_azimuth_deg` (`[0, 360]`).
///
/// Matches `obspy.signal.rotate.rotate_ne_rt`.
pub fn rotate_ne_rt(
    north: &[f64],
    east: &[f64],
    back_azimuth_deg: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if north.len() != east.len() {
        return Err(EarthSnailError::IncompatibleTraces(
            "North and East components must have equal length".into(),
        ));
    }
    if !back_azimuth_deg.is_finite() || !(0.0..=360.0).contains(&back_azimuth_deg) {
        return Err(EarthSnailError::InvalidParameter(
            "back_azimuth_deg must lie in [0, 360]".into(),
        ));
    }
    let ba = back_azimuth_deg * DEG_TO_RAD;
    let (sba, cba) = (sin(ba), cos(ba));
    let mut radial = Vec::with_capacity(north.len());
    let mut transverse = Vec::with_capacity(north.len());
    for (&n, &e) in north.iter().zip(east.iter()) {
        radial.push(-e * sba - n * cba);
        transverse.push(-e * cba + n * sba);
    }
    Ok((radial, transverse))
}

/// Inverse rotation from `(Radial, Transverse)` back to `(North, East)` (`obspy.signal.rotate.rotate_rt_ne`).
pub fn rotate_rt_ne(
    radial: &[f64],
    transverse: &[f64],
    back_azimuth_deg: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if !back_azimuth_deg.is_finite() || !(0.0..=360.0).contains(&back_azimuth_deg) {
        return Err(EarthSnailError::InvalidParameter(
            "back_azimuth_deg must lie in [0, 360]".into(),
        ));
    }
    rotate_ne_rt(radial, transverse, 360.0 - back_azimuth_deg)
}

/// Rotate 3-component `(Z, North, East)` seismograms into ray-coordinate `(L, Q, T)` (`obspy.signal.rotate.rotate_zne_lqt`).
pub fn rotate_zne_lqt(
    z: &[f64],
    north: &[f64],
    east: &[f64],
    back_azimuth_deg: f64,
    incidence_deg: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = z.len();
    if north.len() != len || east.len() != len {
        return Err(EarthSnailError::IncompatibleTraces(
            "Z, North, and East components must have equal length".into(),
        ));
    }
    if !back_azimuth_deg.is_finite() || !(0.0..=360.0).contains(&back_azimuth_deg) {
        return Err(EarthSnailError::InvalidParameter(
            "back_azimuth_deg must lie in [0, 360]".into(),
        ));
    }
    if !incidence_deg.is_finite() || !(0.0..=360.0).contains(&incidence_deg) {
        return Err(EarthSnailError::InvalidParameter(
            "incidence_deg must lie in [0, 360]".into(),
        ));
    }

    let ba = back_azimuth_deg * DEG_TO_RAD;
    let inc = incidence_deg * DEG_TO_RAD;
    let (sba, cba) = (sin(ba), cos(ba));
    let (sinc, cinc) = (sin(inc), cos(inc));

    let mut l_out = Vec::with_capacity(len);
    let mut q_out = Vec::with_capacity(len);
    let mut t_out = Vec::with_capacity(len);

    for i in 0..len {
        let (zv, nv, ev) = (z[i], north[i], east[i]);
        l_out.push(zv * cinc - nv * sinc * cba - ev * sinc * sba);
        q_out.push(zv * sinc + nv * cinc * cba + ev * cinc * sba);
        t_out.push(nv * sba - ev * cba);
    }
    Ok((l_out, q_out, t_out))
}

/// Inverse rotation from ray-coordinate `(L, Q, T)` back to `(Z, North, East)` (`obspy.signal.rotate.rotate_lqt_zne`).
pub fn rotate_lqt_zne(
    l: &[f64],
    q: &[f64],
    t: &[f64],
    back_azimuth_deg: f64,
    incidence_deg: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = l.len();
    if q.len() != len || t.len() != len {
        return Err(EarthSnailError::IncompatibleTraces(
            "L, Q, and T components must have equal length".into(),
        ));
    }
    if !back_azimuth_deg.is_finite() || !(0.0..=360.0).contains(&back_azimuth_deg) {
        return Err(EarthSnailError::InvalidParameter(
            "back_azimuth_deg must lie in [0, 360]".into(),
        ));
    }
    if !incidence_deg.is_finite() || !(0.0..=360.0).contains(&incidence_deg) {
        return Err(EarthSnailError::InvalidParameter(
            "incidence_deg must lie in [0, 360]".into(),
        ));
    }

    let ba = back_azimuth_deg * DEG_TO_RAD;
    let inc = incidence_deg * DEG_TO_RAD;
    let (sba, cba) = (sin(ba), cos(ba));
    let (sinc, cinc) = (sin(inc), cos(inc));

    let mut z_out = Vec::with_capacity(len);
    let mut n_out = Vec::with_capacity(len);
    let mut e_out = Vec::with_capacity(len);

    for i in 0..len {
        let (lv, qv, tv) = (l[i], q[i], t[i]);
        z_out.push(lv * cinc + qv * sinc);
        n_out.push(-lv * sinc * cba + qv * cinc * cba + tv * sba);
        e_out.push(-lv * sinc * sba + qv * cinc * sba - tv * cba);
    }
    Ok((z_out, n_out, e_out))
}

/// Rotate `(Z, North, East)` into symmetric triaxial Streckeisen STS-2 `(U, V, W)` coordinates.
pub fn rotate_zne_uvw(
    z: &[f64],
    north: &[f64],
    east: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = z.len();
    if north.len() != len || east.len() != len {
        return Err(EarthSnailError::IncompatibleTraces(
            "Z, North, and East components must have equal length".into(),
        ));
    }
    let s2_3 = sqrt(2.0 / 3.0);
    let s1_6 = sqrt(1.0 / 6.0);
    let s1_2 = sqrt(0.5);
    let s1_3 = sqrt(1.0 / 3.0);

    let mut u = Vec::with_capacity(len);
    let mut v = Vec::with_capacity(len);
    let mut w = Vec::with_capacity(len);
    for i in 0..len {
        let (zv, nv, ev) = (z[i], north[i], east[i]);
        u.push(-s2_3 * ev + s1_3 * zv);
        v.push(s1_6 * ev + s1_2 * nv + s1_3 * zv);
        w.push(s1_6 * ev - s1_2 * nv + s1_3 * zv);
    }
    Ok((u, v, w))
}

/// Inverse rotation from symmetric triaxial `(U, V, W)` back to `(Z, North, East)`.
pub fn rotate_uvw_zne(u: &[f64], v: &[f64], w: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = u.len();
    if v.len() != len || w.len() != len {
        return Err(EarthSnailError::IncompatibleTraces(
            "U, V, and W components must have equal length".into(),
        ));
    }
    let s2_3 = sqrt(2.0 / 3.0);
    let s1_6 = sqrt(1.0 / 6.0);
    let s1_2 = sqrt(0.5);
    let s1_3 = sqrt(1.0 / 3.0);

    let mut z = Vec::with_capacity(len);
    let mut north = Vec::with_capacity(len);
    let mut east = Vec::with_capacity(len);
    for i in 0..len {
        let (uv, vv, wv) = (u[i], v[i], w[i]);
        z.push(s1_3 * (uv + vv + wv));
        north.push(s1_2 * (vv - wv));
        east.push(-s2_3 * uv + s1_6 * (vv + wv));
    }
    Ok((z, north, east))
}

/// General arbitrary 3-component basis rotation to/from `(Z, North, East)` given SEED `(azimuth_deg, dip_deg)`
/// (`obspy.signal.rotate.rotate2zne`).
#[allow(clippy::too_many_arguments)]
pub fn rotate_to_zne(
    d1: &[f64],
    az1_deg: f64,
    dip1_deg: f64,
    d2: &[f64],
    az2_deg: f64,
    dip2_deg: f64,
    d3: &[f64],
    az3_deg: f64,
    dip3_deg: f64,
    inverse: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = d1.len();
    if d2.len() != len || d3.len() != len {
        return Err(EarthSnailError::IncompatibleTraces(
            "all three component slices must have equal length".into(),
        ));
    }
    let base_vec = |az_deg: f64, dip_deg: f64| -> [f64; 3] {
        let az = az_deg * DEG_TO_RAD;
        let dip = dip_deg * DEG_TO_RAD;
        [-sin(dip), cos(az) * cos(dip), sin(az) * cos(dip)]
    };
    let m = [
        base_vec(az1_deg, dip1_deg),
        base_vec(az2_deg, dip2_deg),
        base_vec(az3_deg, dip3_deg),
    ];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !(1e-6..1e6).contains(&abs(det)) {
        return Err(EarthSnailError::SingularMatrix(
            "component directions are not linearly independent".into(),
        ));
    }

    let op = if inverse {
        m
    } else {
        let inv_det = 1.0 / det;
        [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ]
    };

    let clean_zero = |v: f64| if v == 0.0 { 0.0 } else { v };
    let mut z = Vec::with_capacity(len);
    let mut n = Vec::with_capacity(len);
    let mut e = Vec::with_capacity(len);
    for i in 0..len {
        let (x0, x1, x2) = (d1[i], d2[i], d3[i]);
        z.push(clean_zero(op[0][0] * x0 + op[0][1] * x1 + op[0][2] * x2));
        n.push(clean_zero(op[1][0] * x0 + op[1][1] * x1 + op[1][2] * x2));
        e.push(clean_zero(op[2][0] * x0 + op[2][1] * x1 + op[2][2] * x2));
    }
    Ok((z, n, e))
}

// ============================================================================
// 2. Flinn (1965) & Jurkevics (1988) 3C Covariance Polarization
// ============================================================================

/// 3-component covariance polarization metrics for a single window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolarizationMetrics {
    /// Principal polarization azimuth in `[0, 180]` degrees clockwise from North.
    pub azimuth_deg: f64,
    /// Principal incidence angle from vertical in `[0, 90]` degrees.
    pub incidence_deg: f64,
    /// Rectilinearity in `[0, 1]`.
    pub rectilinearity: f64,
    /// Planarity in `[0, 1]`.
    pub planarity: f64,
    /// Covariance eigenvalues sorted descending `[lambda_0, lambda_1, lambda_2]`.
    pub eigenvalues: [f64; 3],
}

/// Sliding-window Flinn polarization time series (`obspy.signal.polarization.polarization_analysis(..., method='flinn')`).
#[derive(Debug, Clone, PartialEq)]
pub struct SlidingPolarizationResult {
    /// Window center POSIX epoch timestamps in seconds.
    pub timestamps_epoch_s: Vec<f64>,
    /// Window azimuths in `[0, 180]` degrees.
    pub azimuth_deg: Vec<f64>,
    /// Window incidence angles in `[0, 90]` degrees.
    pub incidence_deg: Vec<f64>,
    /// Montalbetti & Kanasewich (1970) rectilinearity `1 - sqrt(l1 / l0)`.
    pub rectilinearity: Vec<f64>,
    /// Jurkevics (1988) planarity `1 - 2 * l2 / (l0 + l1)`.
    pub planarity: Vec<f64>,
}

fn cov_3x3_enz(
    z: &[f64],
    north: &[f64],
    east: &[f64],
    noise_thres: f64,
) -> Result<([[f64; 3]; 3], usize)> {
    let len = z.len();
    if north.len() != len || east.len() != len {
        return Err(EarthSnailError::IncompatibleTraces(
            "Z, North, and East slices must have equal length".into(),
        ));
    }
    let mut mean_e = 0.0_f64;
    let mut mean_n = 0.0_f64;
    let mut mean_z = 0.0_f64;
    let mut count = 0usize;
    for i in 0..len {
        let pwr = z[i] * z[i] + north[i] * north[i] + east[i] * east[i];
        if pwr > noise_thres {
            mean_e += east[i];
            mean_n += north[i];
            mean_z += z[i];
            count += 1;
        }
    }
    if count < 2 {
        return Err(EarthSnailError::InvalidParameter(
            "fewer than 2 samples exceed noise_thres for sample covariance".into(),
        ));
    }
    let inv_n = 1.0 / (count as f64);
    mean_e *= inv_n;
    mean_n *= inv_n;
    mean_z *= inv_n;

    let mut cov = [[0.0_f64; 3]; 3];
    for i in 0..len {
        let pwr = z[i] * z[i] + north[i] * north[i] + east[i] * east[i];
        if pwr > noise_thres {
            let d = [east[i] - mean_e, north[i] - mean_n, z[i] - mean_z];
            for r in 0..3 {
                for c in r..3 {
                    cov[r][c] += d[r] * d[c];
                }
            }
        }
    }
    let inv_dof = 1.0 / ((count - 1) as f64);
    for row in &mut cov {
        for slot in row.iter_mut() {
            *slot *= inv_dof;
        }
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];
    Ok((cov, count))
}

fn fold_flinn_angles(v_enz: [f64; 3]) -> (f64, f64) {
    let mut azimuth = atan2(v_enz[0], v_enz[1]) * RAD_TO_DEG;
    let eve = hypot(v_enz[0], v_enz[1]);
    let mut incidence = atan2(eve, v_enz[2]) * RAD_TO_DEG;
    if azimuth < 0.0 {
        azimuth += 360.0;
    }
    if incidence < 0.0 {
        incidence += 180.0;
    }
    if incidence > 90.0 {
        incidence = 180.0 - incidence;
        if azimuth > 180.0 {
            azimuth -= 180.0;
        } else {
            azimuth += 180.0;
        }
    }
    if azimuth > 180.0 {
        azimuth -= 180.0;
    }
    (azimuth, incidence)
}

/// Flinn (1965) / Montalbetti & Kanasewich (1970) 3C covariance eigenstructure polarization.
///
/// Matches `obspy.signal.polarization.flinn([z, n, e], noise_thres)`.
pub fn flinn_polarization(
    z: &[f64],
    north: &[f64],
    east: &[f64],
    noise_thres: f64,
) -> Result<PolarizationMetrics> {
    let (cov, _cnt) = cov_3x3_enz(z, north, east, noise_thres)?;
    let (evals, evecs) = jacobi_eigen_3x3(cov);
    let l0 = evals[0].max(0.0);
    let l1 = evals[1].max(0.0);
    let l2 = evals[2].max(0.0);

    let rectilinearity = if l0 <= 1e-300 {
        0.0
    } else {
        1.0 - sqrt((l1 / l0).clamp(0.0, 1.0))
    };
    let planarity = if l0 + l1 <= 1e-300 {
        0.0
    } else {
        1.0 - (2.0 * l2) / (l0 + l1)
    };
    let (azimuth_deg, incidence_deg) = fold_flinn_angles(evecs[0]);

    Ok(PolarizationMetrics {
        azimuth_deg,
        incidence_deg,
        rectilinearity,
        planarity,
        eigenvalues: [l0, l1, l2],
    })
}

/// Jurkevics (1988) 3C covariance polarization (`rect = 1 - (l1 + l2) / (2 * l0)`).
pub fn jurkevics_polarization(
    z: &[f64],
    north: &[f64],
    east: &[f64],
    noise_thres: f64,
) -> Result<PolarizationMetrics> {
    let (cov, _cnt) = cov_3x3_enz(z, north, east, noise_thres)?;
    let (evals, evecs) = jacobi_eigen_3x3(cov);
    let l0 = evals[0].max(0.0);
    let l1 = evals[1].max(0.0);
    let l2 = evals[2].max(0.0);

    let rectilinearity = if l0 <= 1e-300 {
        0.0
    } else {
        1.0 - (l1 + l2) / (2.0 * l0)
    };
    let planarity = if l0 + l1 <= 1e-300 {
        0.0
    } else {
        1.0 - (2.0 * l2) / (l0 + l1)
    };
    let (azimuth_deg, incidence_deg) = fold_flinn_angles(evecs[0]);

    Ok(PolarizationMetrics {
        azimuth_deg,
        incidence_deg,
        rectilinearity,
        planarity,
        eigenvalues: [l0, l1, l2],
    })
}

/// Sliding-window Flinn polarization analysis on synchronized `(Z, N, E)` traces.
///
/// Matches `obspy.signal.polarization.polarization_analysis(..., method='flinn')`.
#[allow(clippy::too_many_arguments)]
pub fn polarization_analysis_flinn(
    trace_z: &Trace,
    trace_n: &Trace,
    trace_e: &Trace,
    win_len_s: f64,
    win_frac: f64,
    stime_epoch_s: f64,
    etime_epoch_s: f64,
    var_noise: f64,
) -> Result<SlidingPolarizationResult> {
    let fs = trace_z.stats.sampling_rate;
    if !fs.is_finite()
        || fs <= 0.0
        || abs(trace_n.stats.sampling_rate - fs) > 1e-6 * fs
        || abs(trace_e.stats.sampling_rate - fs) > 1e-6 * fs
    {
        return Err(EarthSnailError::IncompatibleTraces(
            "3C traces must share equal positive sampling_rate".into(),
        ));
    }
    if win_len_s <= 0.0 || win_frac <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "window length and step fraction must be positive".into(),
        ));
    }

    let starts_s = [
        trace_z.stats.starttime.as_epoch_nanos() as f64 * 1e-9,
        trace_n.stats.starttime.as_epoch_nanos() as f64 * 1e-9,
        trace_e.stats.starttime.as_epoch_nanos() as f64 * 1e-9,
    ];
    let ends_s = [
        trace_z.endtime().as_epoch_nanos() as f64 * 1e-9,
        trace_n.endtime().as_epoch_nanos() as f64 * 1e-9,
        trace_e.endtime().as_epoch_nanos() as f64 * 1e-9,
    ];
    let slatest = starts_s[0].max(starts_s[1]).max(starts_s[2]);
    let eearliest = ends_s[0].min(ends_s[1]).min(ends_s[2]);
    if slatest > stime_epoch_s + 1e-12 || eearliest + 1e-12 < etime_epoch_s {
        return Err(EarthSnailError::InvalidParameter(
            "requested analysis interval falls outside common 3C trace span".into(),
        ));
    }

    let delta = 1.0 / fs;
    let mut spoint = [0usize; 3];
    for i in 0..3 {
        let offset = floor((stime_epoch_s - slatest) / delta + 1.0) as usize;
        let diffstart = (slatest - starts_s[i]).max(0.0);
        spoint[i] = (floor(diffstart) as usize) + offset;
    }

    let nsamp = (win_len_s * fs) as usize;
    let nstep = ((nsamp as f64) * win_frac) as usize;
    if nsamp < 2 || nstep == 0 {
        return Err(EarthSnailError::InvalidParameter(
            "sliding window must contain >= 2 samples with positive step".into(),
        ));
    }

    let tap = cosine_taper(nsamp, 0.22, true, false)?;
    let z_raw = &trace_z.data;
    let n_raw = &trace_n.data;
    let e_raw = &trace_e.data;

    let mut timestamps_epoch_s = Vec::new();
    let mut azimuth_deg = Vec::new();
    let mut incidence_deg = Vec::new();
    let mut rectilinearity = Vec::new();
    let mut planarity = Vec::new();

    let mut newstart = stime_epoch_s;
    let mut offset = 0usize;
    let mut z_win = vec![0.0_f64; nsamp];
    let mut n_win = vec![0.0_f64; nsamp];
    let mut e_win = vec![0.0_f64; nsamp];

    while (newstart + ((nsamp + nstep) as f64) / fs) < etime_epoch_s {
        if spoint[0] + offset + nsamp > z_raw.len()
            || spoint[1] + offset + nsamp > n_raw.len()
            || spoint[2] + offset + nsamp > e_raw.len()
        {
            break;
        }
        let timestamp = newstart + (nsamp as f64) / 2.0 / fs;

        let prepare_win = |src: &[f64], dst: &mut [f64]| {
            let mean = src.iter().copied().sum::<f64>() / (nsamp as f64);
            for ((o, &s), &w) in dst.iter_mut().zip(src.iter()).zip(tap.iter()) {
                *o = (s - mean) * w;
            }
        };
        prepare_win(
            &z_raw[(spoint[0] + offset)..(spoint[0] + offset + nsamp)],
            &mut z_win,
        );
        prepare_win(
            &n_raw[(spoint[1] + offset)..(spoint[1] + offset + nsamp)],
            &mut n_win,
        );
        prepare_win(
            &e_raw[(spoint[2] + offset)..(spoint[2] + offset + nsamp)],
            &mut e_win,
        );

        let pm = flinn_polarization(&z_win, &n_win, &e_win, var_noise)?;
        timestamps_epoch_s.push(timestamp);
        azimuth_deg.push(pm.azimuth_deg);
        incidence_deg.push(pm.incidence_deg);
        rectilinearity.push(pm.rectilinearity);
        planarity.push(pm.planarity);

        offset += nstep;
        newstart += (nstep as f64) / fs;
    }

    Ok(SlidingPolarizationResult {
        timestamps_epoch_s,
        azimuth_deg,
        incidence_deg,
        rectilinearity,
        planarity,
    })
}

// ============================================================================
// 3. Receiver Functions (Water-Level & Ligorría–Ammon 1999 Iterative)
// ============================================================================

/// Result of Ligorría & Ammon (1999) iterative time-domain receiver function deconvolution.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativeRfResult {
    /// Bandlimited Gaussian-filtered receiver function of length `vertical.len()`.
    pub receiver_function: Vec<f64>,
    /// Percentage of variance of the Gaussian-filtered radial trace explained (`[0, 100]`).
    pub fit_percent: f64,
    /// Number of spike iterations executed.
    pub iterations: usize,
}

/// Clayton & Wiggins (1976) / Ammon (1991) spectral water-level receiver function deconvolution.
pub fn receiver_function_waterlevel(
    vertical: &[f64],
    radial: &[f64],
    sampling_rate_hz: f64,
    gaussian_a: f64,
    water_level: f64,
    t_shift_s: f64,
) -> Result<Vec<f64>> {
    let n = vertical.len();
    if n == 0 || radial.len() != n {
        return Err(EarthSnailError::IncompatibleTraces(
            "vertical and radial traces must be non-empty and equal length".into(),
        ));
    }
    if !sampling_rate_hz.is_finite()
        || sampling_rate_hz <= 0.0
        || !gaussian_a.is_finite()
        || gaussian_a <= 0.0
        || !water_level.is_finite()
        || water_level <= 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "sampling_rate_hz, gaussian_a, and water_level must be finite and positive".into(),
        ));
    }

    let nfft = next_pow2(2 * n);
    let mut scratch = ScratchWorkspace::new();
    let mut z_spec = Vec::new();
    let mut r_spec = Vec::new();
    rfft_with_scratch(vertical, nfft, &mut z_spec, &mut scratch)?;
    rfft_with_scratch(radial, nfft, &mut r_spec, &mut scratch)?;
    let freqs = rfftfreq(nfft, 1.0 / sampling_rate_hz)?;

    let mut max_zz = 0.0_f64;
    for c in &z_spec {
        let p = c.norm_sqr();
        if p > max_zz {
            max_zz = p;
        }
    }
    let wl_floor = (water_level * max_zz).max(1e-300);

    let mut rf_spec = Vec::with_capacity(z_spec.len());
    let mut zz_spec = Vec::with_capacity(z_spec.len());
    for (k, &f) in freqs.iter().enumerate() {
        let w = 2.0 * PI * f;
        let gauss = exp(-(w * w) / (4.0 * gaussian_a * gaussian_a));
        let phase = -w * t_shift_s;
        let shift_phasor = Complex64::new(cos(phase), sin(phase)) * gauss;
        let zz = z_spec[k].norm_sqr();
        let denom = zz.max(wl_floor);
        let r_zconj = r_spec[k] * z_spec[k].conj();
        rf_spec.push(r_zconj * (shift_phasor / denom));
        zz_spec.push(shift_phasor * (zz / denom));
    }

    let mut rf_td = Vec::new();
    let mut zz_td = Vec::new();
    irfft_with_scratch(&rf_spec, nfft, &mut rf_td, &mut scratch)?;
    irfft_with_scratch(&zz_spec, nfft, &mut zz_td, &mut scratch)?;
    let norm = zz_td
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(abs(v)))
        .max(1e-30);
    rf_td.truncate(n);
    for v in &mut rf_td {
        *v /= norm;
    }
    Ok(rf_td)
}

/// Ligorría & Ammon (1999) iterative time-domain Gaussian-pulse receiver function deconvolution.
#[allow(clippy::too_many_arguments)]
pub fn receiver_function_iter(
    vertical: &[f64],
    radial: &[f64],
    sampling_rate_hz: f64,
    gaussian_a: f64,
    t_shift_s: f64,
    max_bumps: usize,
    min_improvement_pct: f64,
    only_positive: bool,
) -> Result<IterativeRfResult> {
    let n = vertical.len();
    if n == 0 || radial.len() != n {
        return Err(EarthSnailError::IncompatibleTraces(
            "vertical and radial traces must be non-empty and equal length".into(),
        ));
    }
    if !sampling_rate_hz.is_finite()
        || sampling_rate_hz <= 0.0
        || !gaussian_a.is_finite()
        || gaussian_a <= 0.0
        || max_bumps == 0
    {
        return Err(EarthSnailError::InvalidParameter(
            "sampling_rate_hz, gaussian_a, and max_bumps must be positive".into(),
        ));
    }

    let nfft = next_pow2(2 * n);
    let dt = 1.0 / sampling_rate_hz;
    let mut scratch = ScratchWorkspace::new();
    let freqs = rfftfreq(nfft, dt)?;

    let mut g_spec: Vec<Complex64> = freqs
        .iter()
        .map(|&f| {
            let w = 2.0 * PI * f;
            Complex64::new(exp(-(w * w) / (4.0 * gaussian_a * gaussian_a)), 0.0)
        })
        .collect();
    let mut g_impulse = Vec::new();
    irfft_with_scratch(&g_spec, nfft, &mut g_impulse, &mut scratch)?;
    let g_peak = abs(g_impulse[0]).max(1e-30);
    for c in &mut g_spec {
        *c /= g_peak;
    }

    let mut z_f = Vec::new();
    let mut r_f = Vec::new();
    rfft_with_scratch(vertical, nfft, &mut z_f, &mut scratch)?;
    rfft_with_scratch(radial, nfft, &mut r_f, &mut scratch)?;
    for k in 0..g_spec.len() {
        z_f[k] *= g_spec[k];
        r_f[k] *= g_spec[k];
    }
    let mut z_g = Vec::new();
    let mut r_g = Vec::new();
    irfft_with_scratch(&z_f, nfft, &mut z_g, &mut scratch)?;
    irfft_with_scratch(&r_f, nfft, &mut r_g, &mut scratch)?;

    let z_power: f64 = z_g.iter().map(|&v| v * v).sum();
    let r_power: f64 = r_g.iter().map(|&v| v * v).sum();
    if z_power <= 1e-300 || r_power <= 1e-300 {
        return Ok(IterativeRfResult {
            receiver_function: vec![0.0; n],
            fit_percent: 100.0,
            iterations: 0,
        });
    }

    let mut residual = r_g;
    let mut spikes = vec![0.0_f64; nfft];
    let z_g_conj: Vec<Complex64> = z_f.iter().map(|c| c.conj()).collect();
    let shift_samples = ((t_shift_s * sampling_rate_hz + 0.5).max(0.0)) as usize;
    let max_pos_lag = n.saturating_sub(1);
    let min_neg_lag = shift_samples.min(nfft / 2);

    let mut prev_fit = 0.0_f64;
    let mut iters_done = 0usize;
    let mut res_f = Vec::new();
    let mut corr = Vec::new();

    for iter in 0..max_bumps {
        rfft_with_scratch(&residual, nfft, &mut res_f, &mut scratch)?;
        for k in 0..res_f.len() {
            res_f[k] *= z_g_conj[k];
        }
        irfft_with_scratch(&res_f, nfft, &mut corr, &mut scratch)?;
        // Note: `irfft_with_scratch` divides by `nfft`, whereas raw discrete dot-product
        // `sum_t residual[t] * z_g[t - lag]` equals `(1/nfft) * sum_k R_res[k] * conj(Z_g[k]) * nfft`
        // wait: `irfft` ALREADY computes `(1/nfft) * sum_k R_res[k] * conj(Z_g[k])`, which by Parseval IS
        // `sum_t residual[t] * z_g[t - lag]`! Let's verify: for `residual = z_g = [1, 0, ...]`,
        // `R_res[k] = 1`, `Z_g[k] = 1`, `irfft([1; nfft/2+1]) = [1, 0, ...]`, so `corr[0] == 1.0 == z_power`!

        let mut best_lag = 0usize;
        let mut best_metric = -1.0_f64;
        for (lag, &cval) in corr.iter().enumerate() {
            let valid = lag <= max_pos_lag || lag >= nfft - min_neg_lag;
            if !valid {
                continue;
            }
            let metric = if only_positive { cval } else { abs(cval) };
            if metric > best_metric {
                best_metric = metric;
                best_lag = lag;
            }
        }
        if best_metric <= 0.0 {
            break;
        }

        let amp = corr[best_lag] / z_power;
        spikes[best_lag] += amp;

        for (t_idx, res_slot) in residual.iter_mut().enumerate() {
            let src_idx = (t_idx + nfft - best_lag) % nfft;
            *res_slot -= amp * z_g[src_idx];
        }

        let res_power: f64 = residual.iter().map(|&v| v * v).sum();
        let fit_pct = (100.0 * (1.0 - res_power / r_power)).clamp(0.0, 100.0);
        iters_done = iter + 1;
        if iter > 0 && (fit_pct - prev_fit) < min_improvement_pct {
            prev_fit = fit_pct;
            break;
        }
        prev_fit = fit_pct;
    }

    let mut spikes_f = Vec::new();
    rfft_with_scratch(&spikes, nfft, &mut spikes_f, &mut scratch)?;
    for (k, &f) in freqs.iter().enumerate() {
        let phase = -2.0 * PI * f * t_shift_s;
        let shift_phasor = Complex64::new(cos(phase), sin(phase));
        spikes_f[k] *= g_spec[k] * shift_phasor;
    }
    let mut rf_td = Vec::new();
    irfft_with_scratch(&spikes_f, nfft, &mut rf_td, &mut scratch)?;
    rf_td.truncate(n);

    Ok(IterativeRfResult {
        receiver_function: rf_td,
        fit_percent: prev_fit,
        iterations: iters_done,
    })
}

// ============================================================================
// 4. Silver & Chan (1991) Shear-Wave Splitting
// ============================================================================

/// Result of Silver & Chan (1991) horizontal covariance eigenvalue minimization.
#[derive(Debug, Clone, PartialEq)]
pub struct ShearWaveSplittingResult {
    /// Best-fitting fast-axis polarization azimuth in `[0, 180)` degrees clockwise from North.
    pub fast_phi_deg: f64,
    /// Best-fitting splitting delay time `dt` in seconds (`>= 0`).
    pub dt_s: f64,
    /// Minimum smaller covariance eigenvalue `lambda_2(fast_phi, dt)`.
    pub lambda2_min: f64,
    /// Larger covariance eigenvalue `lambda_1(fast_phi, dt)` at the optimum.
    pub lambda1_at_min: f64,
    /// Rectilinearity `1 - lambda_2 / lambda_1` after removing splitting.
    pub corrected_linearity: f64,
    /// Sampled `phi` grid in degrees.
    pub phi_grid_deg: Vec<f64>,
    /// Sampled `dt` grid in seconds.
    pub dt_grid_s: Vec<f64>,
    /// Row-major `(phi_idx * dt_grid.len() + dt_idx)` surface of `lambda_2(phi, dt)`.
    pub lambda2_surface: Vec<f64>,
}

/// Silver & Chan (1991) shear-wave splitting estimation over `(phi_deg, dt_s)` grid.
pub fn silver_chan_splitting(
    north: &[f64],
    east: &[f64],
    sampling_rate_hz: f64,
    max_dt_s: f64,
    phi_step_deg: f64,
    dt_step_s: f64,
) -> Result<ShearWaveSplittingResult> {
    let n = north.len();
    if n < 16 || east.len() != n {
        return Err(EarthSnailError::IncompatibleTraces(
            "north and east slices must have equal length >= 16".into(),
        ));
    }
    if !sampling_rate_hz.is_finite()
        || sampling_rate_hz <= 0.0
        || !max_dt_s.is_finite()
        || max_dt_s <= 0.0
        || !phi_step_deg.is_finite()
        || phi_step_deg <= 0.0
        || !dt_step_s.is_finite()
        || dt_step_s <= 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "sampling_rate_hz, max_dt_s, phi_step_deg, and dt_step_s must be finite and positive"
                .into(),
        ));
    }

    let max_shift_samp = ((max_dt_s * sampling_rate_hz) as usize) + 2;
    if n <= max_shift_samp + 8 {
        return Err(EarthSnailError::InvalidParameter(
            "trace window is too short for requested max_dt_s".into(),
        ));
    }
    let eval_len = n - max_shift_samp;

    let n_phi = ((180.0 / phi_step_deg).max(1.0)) as usize;
    let n_dt = (floor(max_dt_s / dt_step_s) as usize) + 1;
    let mut phi_grid_deg = Vec::with_capacity(n_phi);
    for ip in 0..n_phi {
        let p = (ip as f64) * phi_step_deg;
        if p < 180.0 - 1e-12 {
            phi_grid_deg.push(p);
        }
    }
    let mut dt_grid_s = Vec::with_capacity(n_dt);
    for id in 0..n_dt {
        let d = (id as f64) * dt_step_s;
        if d <= max_dt_s + 1e-12 {
            dt_grid_s.push(d);
        }
    }

    let mut lambda2_surface = vec![0.0_f64; phi_grid_deg.len() * dt_grid_s.len()];
    let mut best_l2 = f64::INFINITY;
    let mut best_l1 = 0.0_f64;
    let mut best_phi = 0.0_f64;
    let mut best_dt = 0.0_f64;

    let mut fast = vec![0.0_f64; n];
    let mut slow = vec![0.0_f64; n];

    for (ip, &phi_deg) in phi_grid_deg.iter().enumerate() {
        let phi = phi_deg * DEG_TO_RAD;
        let (sp, cp) = (sin(phi), cos(phi));
        for i in 0..n {
            fast[i] = north[i] * cp + east[i] * sp;
            slow[i] = -north[i] * sp + east[i] * cp;
        }
        let mean_f = fast[..eval_len].iter().copied().sum::<f64>() / (eval_len as f64);
        let mut c_ff = 0.0_f64;
        for &fv in &fast[..eval_len] {
            let df = fv - mean_f;
            c_ff += df * df;
        }

        for (id, &dt_cand) in dt_grid_s.iter().enumerate() {
            let shift_f = dt_cand * sampling_rate_hz;
            let k0 = floor(shift_f) as usize;
            let frac = shift_f - (k0 as f64);

            let sample_slow = |idx: usize| -> f64 {
                let base = idx + k0;
                if frac <= 1e-14 {
                    slow[base]
                } else if base >= 1 && base + 2 < n {
                    let y0 = slow[base - 1];
                    let y1 = slow[base];
                    let y2 = slow[base + 1];
                    let y3 = slow[base + 2];
                    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                    let a2 = -0.5 * y0 + 0.5 * y2;
                    ((a0 * frac + a1) * frac + a2) * frac + y1
                } else {
                    (1.0 - frac) * slow[base] + frac * slow[base + 1]
                }
            };

            let mut mean_s = 0.0_f64;
            for i in 0..eval_len {
                mean_s += sample_slow(i);
            }
            mean_s /= eval_len as f64;

            let mut c_ss = 0.0_f64;
            let mut c_fs = 0.0_f64;
            for (i, &fv) in fast.iter().enumerate().take(eval_len) {
                let df = fv - mean_f;
                let ds = sample_slow(i) - mean_s;
                c_ss += ds * ds;
                c_fs += df * ds;
            }

            let tr = c_ff + c_ss;
            let disc = hypot(c_ff - c_ss, 2.0 * c_fs);
            let l1 = 0.5 * (tr + disc);
            let l2 = (0.5 * (tr - disc)).max(0.0);
            lambda2_surface[ip * dt_grid_s.len() + id] = l2;

            if l2 < best_l2 {
                best_l2 = l2;
                best_l1 = l1;
                best_phi = phi_deg;
                best_dt = dt_cand;
            }
        }
    }

    let corrected_linearity = if best_l1 <= 1e-300 {
        0.0
    } else {
        (1.0 - best_l2 / best_l1).clamp(0.0, 1.0)
    };

    Ok(ShearWaveSplittingResult {
        fast_phi_deg: best_phi,
        dt_s: best_dt,
        lambda2_min: best_l2,
        lambda1_at_min: best_l1,
        corrected_linearity,
        phi_grid_deg,
        dt_grid_s,
        lambda2_surface,
    })
}

// ============================================================================
// 5. Array Seismology: Conventional & Capon (1969) MVDR FK + Vespagram
// ============================================================================

/// Local Cartesian coordinates `(x_east_km, y_north_km)` of a seismic array element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrayStation {
    /// Eastward offset from array reference point in kilometers.
    pub x_east_km: f64,
    /// Northward offset from array reference point in kilometers.
    pub y_north_km: f64,
}

impl ArrayStation {
    /// Construct an `ArrayStation` directly from `(x_east_km, y_north_km)`.
    pub const fn new(x_east_km: f64, y_north_km: f64) -> Self {
        Self {
            x_east_km,
            y_north_km,
        }
    }
}

/// Beamforming estimator for [`fk_analysis`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FkMethod {
    /// Conventional delay-and-sum frequency-domain Bartlett beamformer.
    Conventional,
    /// Capon (1969) Minimum-Variance Distortionless Response (`MVDR`) high-resolution beamformer.
    CaponMvdr {
        /// Fractional diagonal regularization added to the cross-spectral matrix (`> 0`, e.g. `1e-2`).
        diagonal_loading: f64,
    },
}

/// Output of frequency-wavenumber (`FK`) array analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct FkAnalysisResult {
    /// Estimated wave arrival back-azimuth in `[0, 360)` degrees clockwise from North.
    pub back_azimuth_deg: f64,
    /// Estimated horizontal slowness `|s|` in `s/km`.
    pub horizontal_slowness_s_per_km: f64,
    /// Multi-channel frequency-band semblance / normalized coherent power in `[0, 1]`.
    pub semblance: f64,
    /// Peak absolute beam power.
    pub abs_power: f64,
    /// Peak relative beam power in `[0, 1]`.
    pub rel_power: f64,
    /// Sampled slowness grid coordinates `[-s_max .. +s_max]` in `s/km`.
    pub slowness_axis_s_per_km: Vec<f64>,
    /// Row-major `(iy_north * grid_size + ix_east)` beam power surface.
    pub power_grid: Vec<f64>,
}

fn invert_complex_matrix(mat: &[Complex64], m: usize) -> Option<Vec<Complex64>> {
    let mut a = mat.to_vec();
    let mut inv = vec![Complex64::ZERO; m * m];
    for i in 0..m {
        inv[i * m + i] = Complex64::ONE;
    }

    for col in 0..m {
        let mut piv_row = col;
        let mut piv_mag = a[col * m + col].norm_sqr();
        for r in (col + 1)..m {
            let mag = a[r * m + col].norm_sqr();
            if mag > piv_mag {
                piv_mag = mag;
                piv_row = r;
            }
        }
        if piv_mag <= 1e-60 {
            return None;
        }
        if piv_row != col {
            for c in 0..m {
                a.swap(col * m + c, piv_row * m + c);
                inv.swap(col * m + c, piv_row * m + c);
            }
        }
        let piv = a[col * m + col];
        let inv_piv = piv.inv();
        for c in 0..m {
            a[col * m + c] *= inv_piv;
            inv[col * m + c] *= inv_piv;
        }
        for r in 0..m {
            if r == col {
                continue;
            }
            let factor = a[r * m + col];
            if factor.norm_sqr() == 0.0 {
                continue;
            }
            for c in 0..m {
                let a_col_c = a[col * m + c];
                let inv_col_c = inv[col * m + c];
                a[r * m + c] -= factor * a_col_c;
                inv[r * m + c] -= factor * inv_col_c;
            }
        }
    }
    Some(inv)
}

/// Conventional Delay-and-Sum & Capon (1969) MVDR Frequency-Wavenumber (`FK`) array analysis.
#[allow(clippy::too_many_arguments)]
pub fn fk_analysis(
    stations: &[ArrayStation],
    traces: &[&[f64]],
    sampling_rate_hz: f64,
    fmin_hz: f64,
    fmax_hz: f64,
    s_max_s_per_km: f64,
    grid_size: usize,
    method: FkMethod,
) -> Result<FkAnalysisResult> {
    let m = stations.len();
    if m < 2 || traces.len() != m {
        return Err(EarthSnailError::InvalidParameter(
            "require at least 2 stations and matching trace count".into(),
        ));
    }
    let n = traces[0].len();
    if n < 16 || traces.iter().any(|t| t.len() != n) {
        return Err(EarthSnailError::IncompatibleTraces(
            "all array traces must have identical length >= 16".into(),
        ));
    }
    if !sampling_rate_hz.is_finite()
        || sampling_rate_hz <= 0.0
        || fmin_hz < 0.0
        || fmax_hz <= fmin_hz
        || fmax_hz > 0.5 * sampling_rate_hz
        || s_max_s_per_km <= 0.0
        || grid_size < 3
    {
        return Err(EarthSnailError::InvalidParameter(
            "invalid frequency band, sampling rate, slowness bound, or grid_size (< 3)".into(),
        ));
    }

    let sub_len = match method {
        FkMethod::Conventional => n,
        FkMethod::CaponMvdr { .. } => (n / 4).max(32).min(n),
    };
    let step = (sub_len / 2).max(1);
    let nfft = next_pow2(sub_len);
    let freqs = rfftfreq(nfft, 1.0 / sampling_rate_hz)?;

    let mut active_bins = Vec::new();
    for (k, &f) in freqs.iter().enumerate() {
        if f >= fmin_hz && f <= fmax_hz && f > 0.0 {
            active_bins.push((k, f));
        }
    }
    if active_bins.is_empty() {
        return Err(EarthSnailError::InvalidParameter(
            "no FFT frequency bins fall inside [fmin_hz, fmax_hz]".into(),
        ));
    }

    let mut scratch = ScratchWorkspace::new();
    let mut csd_per_bin = vec![vec![Complex64::ZERO; m * m]; active_bins.len()];
    let mut n_subwins = 0usize;

    let mut start = 0usize;
    let mut win_weights = vec![0.0_f64; sub_len];
    if sub_len > 1 {
        for (i, slot) in win_weights.iter_mut().enumerate() {
            *slot = 0.5 * (1.0 - cos(2.0 * PI * (i as f64) / ((sub_len - 1) as f64)));
        }
    } else {
        win_weights[0] = 1.0;
    }
    let win_energy: f64 = win_weights.iter().map(|&w| w * w).sum::<f64>().max(1e-30);

    let mut seg_buf = vec![0.0_f64; sub_len];
    let mut ch_specs: Vec<Vec<Complex64>> = vec![Vec::new(); m];

    while start + sub_len <= n {
        for (ch_idx, tr) in traces.iter().enumerate() {
            let slice = &tr[start..(start + sub_len)];
            let mean = slice.iter().copied().sum::<f64>() / (sub_len as f64);
            for i in 0..sub_len {
                seg_buf[i] = (slice[i] - mean) * win_weights[i];
            }
            rfft_with_scratch(&seg_buf, nfft, &mut ch_specs[ch_idx], &mut scratch)?;
        }

        for (bi, &(k, _f)) in active_bins.iter().enumerate() {
            let csd = &mut csd_per_bin[bi];
            for r in 0..m {
                let ur = ch_specs[r][k];
                for c in 0..m {
                    csd[r * m + c] += ur * ch_specs[c][k].conj();
                }
            }
        }
        n_subwins += 1;
        if sub_len == n {
            break;
        }
        start += step;
    }

    let scale_csd = 1.0 / ((n_subwins.max(1) as f64) * win_energy);
    let mut total_trace_power = 0.0_f64;
    let mut inv_csd_per_bin: Vec<Vec<Complex64>> = Vec::new();

    for csd in &mut csd_per_bin {
        for v in csd.iter_mut() {
            *v *= scale_csd;
        }
        let mut tr_bin = 0.0_f64;
        for d in 0..m {
            tr_bin += csd[d * m + d].re.max(0.0);
        }
        total_trace_power += tr_bin;

        if let FkMethod::CaponMvdr { diagonal_loading } = method {
            let load = diagonal_loading.max(1e-6) * (tr_bin / (m as f64)).max(1e-30);
            let mut reg = csd.clone();
            for d in 0..m {
                reg[d * m + d] += Complex64::new(load, 0.0);
            }
            let inv = invert_complex_matrix(&reg, m).ok_or_else(|| {
                EarthSnailError::SingularMatrix(
                    "Capon MVDR cross-spectral matrix is singular".into(),
                )
            })?;
            inv_csd_per_bin.push(inv);
        }
    }

    let mut slowness_axis = Vec::with_capacity(grid_size);
    for idx in 0..grid_size {
        let s = -s_max_s_per_km + 2.0 * s_max_s_per_km * (idx as f64) / ((grid_size - 1) as f64);
        slowness_axis.push(s);
    }

    let inv_sqrt_m = 1.0 / sqrt(m as f64);
    let mut power_grid = vec![0.0_f64; grid_size * grid_size];
    let mut best_power = -1.0_f64;
    let mut best_conv_power = 0.0_f64;
    let mut best_sx = 0.0_f64;
    let mut best_sy = 0.0_f64;
    let mut steer = vec![Complex64::ZERO; m];

    for (iy, &sy_north) in slowness_axis.iter().enumerate() {
        for (ix, &sx_east) in slowness_axis.iter().enumerate() {
            let mut p_sum = 0.0_f64;
            let mut conv_sum = 0.0_f64;

            for (bi, &(_k, freq)) in active_bins.iter().enumerate() {
                let w = 2.0 * PI * freq;
                for (j, st) in stations.iter().enumerate() {
                    let phase = w * (sx_east * st.x_east_km + sy_north * st.y_north_km);
                    steer[j] = Complex64::new(cos(phase) * inv_sqrt_m, sin(phase) * inv_sqrt_m);
                }

                let csd = &csd_per_bin[bi];
                let mut quad_conv = 0.0_f64;
                for r in 0..m {
                    let mut row_acc = Complex64::ZERO;
                    for c in 0..m {
                        row_acc += csd[r * m + c] * steer[c];
                    }
                    quad_conv += (steer[r].conj() * row_acc).re;
                }
                let quad_conv = quad_conv.max(0.0);
                conv_sum += quad_conv;

                match method {
                    FkMethod::Conventional => {
                        p_sum += quad_conv;
                    }
                    FkMethod::CaponMvdr { .. } => {
                        let inv_r = &inv_csd_per_bin[bi];
                        let mut denom = 0.0_f64;
                        for r in 0..m {
                            let mut row_acc = Complex64::ZERO;
                            for c in 0..m {
                                row_acc += inv_r[r * m + c] * steer[c];
                            }
                            denom += (steer[r].conj() * row_acc).re;
                        }
                        if denom > 1e-30 {
                            p_sum += 1.0 / denom;
                        }
                    }
                }
            }

            power_grid[iy * grid_size + ix] = p_sum;
            if p_sum > best_power {
                best_power = p_sum;
                best_conv_power = conv_sum;
                best_sx = sx_east;
                best_sy = sy_north;
            }
        }
    }

    let horizontal_slowness_s_per_km = hypot(best_sx, best_sy);
    let mut back_azimuth_deg = if horizontal_slowness_s_per_km <= 1e-12 {
        0.0
    } else {
        (atan2(best_sx, best_sy) * RAD_TO_DEG) % 360.0
    };
    if back_azimuth_deg < 0.0 {
        back_azimuth_deg += 360.0;
    }

    let semblance = if total_trace_power <= 1e-30 {
        0.0
    } else {
        (best_conv_power / total_trace_power).clamp(0.0, 1.0)
    };

    Ok(FkAnalysisResult {
        back_azimuth_deg,
        horizontal_slowness_s_per_km,
        semblance,
        abs_power: best_power.max(0.0),
        rel_power: semblance,
        slowness_axis_s_per_km: slowness_axis,
        power_grid,
    })
}

/// Stacking mode for [`vespagram`] (`slant_stack`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StackMethod {
    /// Standard linear delay-and-sum beam.
    Linear,
    /// Muirhead & Datt (1976) non-linear `N`-th root stack (`root >= 1`).
    NthRoot(usize),
    /// Schimmel & Paulssen (1997) instantaneous Phase-Weighted Stack (`PWS`) with coherence power `nu`.
    PhaseWeighted(f64),
}

/// Velocity spectral analysis (`vespagram` / `tau-p` slant stack).
#[derive(Debug, Clone, PartialEq)]
pub struct VespagramResult {
    /// Slowness grid in `s/km` (`n_slow`).
    pub slownesses_s_per_km: Vec<f64>,
    /// Number of time samples per slowness beam (`n_samples`).
    pub n_samples: usize,
    /// Row-major `(s_idx * n_samples + t_idx)` stacked beam matrix.
    pub beams: Vec<f64>,
}

/// Compute a `tau-p` vespagram (`slant_stack`) across array trace offsets `offsets_km`.
pub fn vespagram(
    offsets_km: &[f64],
    traces: &[&[f64]],
    sampling_rate_hz: f64,
    slownesses_s_per_km: &[f64],
    method: StackMethod,
) -> Result<VespagramResult> {
    let m = offsets_km.len();
    if m == 0 || traces.len() != m || slownesses_s_per_km.is_empty() {
        return Err(EarthSnailError::InvalidParameter(
            "require non-empty matching offsets/traces and non-empty slowness grid".into(),
        ));
    }
    let n = traces[0].len();
    if n < 2 || traces.iter().any(|t| t.len() != n) || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::IncompatibleTraces(
            "all traces must have equal length >= 2 and positive sampling_rate_hz".into(),
        ));
    }

    let mut scratch = ScratchWorkspace::new();
    let unit_phasors: Vec<Vec<Complex64>> = if matches!(method, StackMethod::PhaseWeighted(_)) {
        let mut ph = Vec::with_capacity(m);
        for &tr in traces {
            let z = analytic_signal_with_scratch(tr, &mut scratch)?;
            ph.push(
                z.into_iter()
                    .map(|c| {
                        let mag = hypot(c.re, c.im);
                        if mag > 1e-30 {
                            Complex64::new(c.re / mag, c.im / mag)
                        } else {
                            Complex64::ZERO
                        }
                    })
                    .collect(),
            );
        }
        ph
    } else {
        Vec::new()
    };

    let inv_m = 1.0 / (m as f64);
    let mut beams = vec![0.0_f64; slownesses_s_per_km.len() * n];

    let interp_f64 = |arr: &[f64], t_samp: f64| -> f64 {
        if t_samp < 0.0 || t_samp > ((n - 1) as f64) {
            return 0.0;
        }
        let k0 = floor(t_samp) as usize;
        if k0 + 1 >= n {
            arr[n - 1]
        } else {
            let f = t_samp - (k0 as f64);
            (1.0 - f) * arr[k0] + f * arr[k0 + 1]
        }
    };

    for (si, &p_slow) in slownesses_s_per_km.iter().enumerate() {
        let row = &mut beams[(si * n)..((si + 1) * n)];
        for (ti, slot) in row.iter_mut().enumerate() {
            match method {
                StackMethod::Linear => {
                    let mut acc = 0.0_f64;
                    for j in 0..m {
                        let t_samp = (ti as f64) + p_slow * offsets_km[j] * sampling_rate_hz;
                        acc += interp_f64(traces[j], t_samp);
                    }
                    *slot = acc * inv_m;
                }
                StackMethod::NthRoot(root_k) => {
                    let rk = root_k.max(1) as f64;
                    let inv_rk = 1.0 / rk;
                    let mut acc = 0.0_f64;
                    for j in 0..m {
                        let t_samp = (ti as f64) + p_slow * offsets_km[j] * sampling_rate_hz;
                        let v = interp_f64(traces[j], t_samp);
                        if v >= 0.0 {
                            acc += powf(v, inv_rk);
                        } else {
                            acc -= powf(-v, inv_rk);
                        }
                    }
                    let mean_r = acc * inv_m;
                    *slot = if mean_r >= 0.0 {
                        powf(mean_r, rk)
                    } else {
                        -powf(-mean_r, rk)
                    };
                }
                StackMethod::PhaseWeighted(nu) => {
                    let mut lin_acc = 0.0_f64;
                    let mut ph_acc = Complex64::ZERO;
                    for j in 0..m {
                        let t_samp = (ti as f64) + p_slow * offsets_km[j] * sampling_rate_hz;
                        lin_acc += interp_f64(traces[j], t_samp);
                        if t_samp >= 0.0 && t_samp <= ((n - 1) as f64) {
                            let k0 = floor(t_samp) as usize;
                            let u = if k0 + 1 >= n {
                                unit_phasors[j][n - 1]
                            } else {
                                let f = t_samp - (k0 as f64);
                                let c =
                                    unit_phasors[j][k0] * (1.0 - f) + unit_phasors[j][k0 + 1] * f;
                                let mag = hypot(c.re, c.im);
                                if mag > 1e-30 {
                                    c / mag
                                } else {
                                    Complex64::ZERO
                                }
                            };
                            ph_acc += u;
                        }
                    }
                    let coh = (hypot(ph_acc.re, ph_acc.im) * inv_m).clamp(0.0, 1.0);
                    *slot = (lin_acc * inv_m) * powf(coh, nu.max(0.0));
                }
            }
        }
    }

    Ok(VespagramResult {
        slownesses_s_per_km: slownesses_s_per_km.to_vec(),
        n_samples: n,
        beams,
    })
}

/// Alias for [`vespagram`] (`tau-p` slant stack).
pub fn slant_stack(
    offsets_km: &[f64],
    traces: &[&[f64]],
    sampling_rate_hz: f64,
    slownesses_s_per_km: &[f64],
    method: StackMethod,
) -> Result<VespagramResult> {
    vespagram(
        offsets_km,
        traces,
        sampling_rate_hz,
        slownesses_s_per_km,
        method,
    )
}

// ============================================================================
// 6. Engineering Seismology: Nigam & Jennings (1969) Response Spectra & Intensity
// ============================================================================

/// Damped SDOF response spectra computed via Nigam & Jennings (1969) exact piecewise-linear integration.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseSpectrum {
    /// Oscillator natural periods `T` in seconds (`0.0` allowed for rigid PGA limit).
    pub periods_s: Vec<f64>,
    /// Viscous damping ratio `zeta` (e.g. `0.05` for 5% critical damping).
    pub damping_ratio: f64,
    /// Relative spectral displacement `SD(T) = max |u(t)|` (`m`).
    pub sd_m: Vec<f64>,
    /// Relative spectral velocity `SV(T) = max |u'(t)|` (`m/s`).
    pub sv_m_s: Vec<f64>,
    /// Pseudo-spectral velocity `PSV(T) = (2*pi/T) * SD(T)` (`m/s`).
    pub psv_m_s: Vec<f64>,
    /// Pseudo-spectral acceleration `PSA(T) = (2*pi/T)^2 * SD(T)` (`m/s^2`).
    pub psa_m_s2: Vec<f64>,
    /// True absolute spectral acceleration `SA(T) = max |u''(t) + a_g(t)|` (`m/s^2`).
    pub sa_m_s2: Vec<f64>,
}

fn sdof_nigam_jennings_single(
    acc_m_s2: &[f64],
    dt_s: f64,
    period_s: f64,
    zeta: f64,
) -> (f64, f64, f64, f64, f64) {
    let pga_val = peak_ground_motion(acc_m_s2);
    if period_s <= 1e-9 {
        return (0.0, 0.0, 0.0, pga_val, pga_val);
    }

    let wn = 2.0 * PI / period_s;
    let wn2 = wn * wn;
    let wn3 = wn2 * wn;
    let sqrt_1mz2 = sqrt((1.0 - zeta * zeta).max(1e-12));
    let wd = wn * sqrt_1mz2;

    let use_rk4 = wn * dt_s < 1e-3;

    let e_term = exp(-zeta * wn * dt_s);
    let c_term = cos(wd * dt_s);
    let s_term = sin(wd * dt_s);
    let z_ratio = zeta / sqrt_1mz2;

    let a11 = e_term * (c_term + z_ratio * s_term);
    let a12 = e_term * s_term / wd;
    let a21 = -(wn2 / wd) * e_term * s_term;
    let a22 = e_term * (c_term - z_ratio * s_term);

    let mut u = 0.0_f64;
    let mut v = 0.0_f64;
    let mut max_u = 0.0_f64;
    let mut max_v = 0.0_f64;
    let mut max_abs_acc = 0.0_f64;

    for i in 0..(acc_m_s2.len() - 1) {
        let a0 = acc_m_s2[i];
        let a1 = acc_m_s2[i + 1];

        if use_rk4 {
            let a_mid = 0.5 * (a0 + a1);
            let f = |uu: f64, vv: f64, ag: f64| -> (f64, f64) {
                (vv, -ag - 2.0 * zeta * wn * vv - wn2 * uu)
            };
            let (k1u, k1v) = f(u, v, a0);
            let (k2u, k2v) = f(u + 0.5 * dt_s * k1u, v + 0.5 * dt_s * k1v, a_mid);
            let (k3u, k3v) = f(u + 0.5 * dt_s * k2u, v + 0.5 * dt_s * k2v, a_mid);
            let (k4u, k4v) = f(u + dt_s * k3u, v + dt_s * k3v, a1);
            u += (dt_s / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            v += (dt_s / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        } else {
            let alpha = (a1 - a0) / dt_s;
            let c1 = -alpha / wn2;
            let c0 = -a0 / wn2 + (2.0 * zeta * alpha) / wn3;
            let up1 = -a1 / wn2 + (2.0 * zeta * alpha) / wn3;
            let vp1 = c1;

            let du0 = u - c0;
            let dv0 = v - c1;
            u = up1 + a11 * du0 + a12 * dv0;
            v = vp1 + a21 * du0 + a22 * dv0;
        }

        let au = abs(u);
        let av = abs(v);
        let aa = abs(2.0 * zeta * wn * v + wn2 * u);
        if au > max_u {
            max_u = au;
        }
        if av > max_v {
            max_v = av;
        }
        if aa > max_abs_acc {
            max_abs_acc = aa;
        }
    }

    let psv = wn * max_u;
    let psa = wn2 * max_u;
    (max_u, max_v, psv, psa, max_abs_acc)
}

/// Compute `SD`, `SV`, `PSV`, `PSA`, and `SA` response spectra via Nigam & Jennings (1969).
pub fn nigam_jennings_response_spectra(
    acc_m_s2: &[f64],
    dt_s: f64,
    periods_s: &[f64],
    damping_ratio: f64,
) -> Result<ResponseSpectrum> {
    if acc_m_s2.len() < 2 || periods_s.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "nigam_jennings_response_spectra requires >= 2 acceleration samples and >= 1 period"
                .into(),
        ));
    }
    if !dt_s.is_finite()
        || dt_s <= 0.0
        || !damping_ratio.is_finite()
        || !(0.0..1.0).contains(&damping_ratio)
        || periods_s.iter().any(|&p| !p.is_finite() || p < 0.0)
    {
        return Err(EarthSnailError::InvalidParameter(
            "require dt_s > 0, damping_ratio in [0, 1), and non-negative periods".into(),
        ));
    }

    let np = periods_s.len();
    let mut sd_m = Vec::with_capacity(np);
    let mut sv_m_s = Vec::with_capacity(np);
    let mut psv_m_s = Vec::with_capacity(np);
    let mut psa_m_s2 = Vec::with_capacity(np);
    let mut sa_m_s2 = Vec::with_capacity(np);

    for &t_per in periods_s {
        let (sd, sv, psv, psa, sa) =
            sdof_nigam_jennings_single(acc_m_s2, dt_s, t_per, damping_ratio);
        sd_m.push(sd);
        sv_m_s.push(sv);
        psv_m_s.push(psv);
        psa_m_s2.push(psa);
        sa_m_s2.push(sa);
    }

    Ok(ResponseSpectrum {
        periods_s: periods_s.to_vec(),
        damping_ratio,
        sd_m,
        sv_m_s,
        psv_m_s,
        psa_m_s2,
        sa_m_s2,
    })
}

/// Peak absolute value (`PGA`, `PGV`, or `PGD`) of a kinematic ground-motion trace.
pub fn peak_ground_motion(samples: &[f64]) -> f64 {
    samples
        .iter()
        .copied()
        .fold(0.0_f64, |mx, v| mx.max(abs(v)))
}

/// Summary engineering ground-motion intensity parameters computed from an acceleration record (`m/s^2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundMotionSummary {
    /// Peak Ground Acceleration in `m/s^2`.
    pub pga_m_s2: f64,
    /// Peak Ground Velocity in `m/s`.
    pub pgv_m_s: f64,
    /// Peak Ground Displacement in `m`.
    pub pgd_m: f64,
    /// Arias Intensity `I_A = (pi / (2 g)) * int a(t)^2 dt` in `m/s`.
    pub arias_intensity_m_s: f64,
    /// Worden et al. (2012) ShakeMap Instrumental Modified Mercalli Intensity (`[1.0, 10.0]`).
    pub worden_mmi: f64,
}

/// Arias (1970) intensity `I_A = (pi / (2 * g)) * integral(a(t)^2 dt)` in `m/s` for `acc_m_s2` in `m/s^2`.
pub fn arias_intensity(acc_m_s2: &[f64], dt_s: f64) -> Result<f64> {
    if acc_m_s2.is_empty() {
        return Err(EarthSnailError::EmptyData(
            "arias_intensity input is empty".into(),
        ));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "dt_s must be finite and positive".into(),
        ));
    }
    if acc_m_s2.len() == 1 {
        return Ok(0.0);
    }
    let mut integral_a2 = 0.0_f64;
    for w in acc_m_s2.windows(2) {
        integral_a2 += 0.5 * (w[0] * w[0] + w[1] * w[1]) * dt_s;
    }
    Ok((PI / (2.0 * STANDARD_GRAVITY_M_S2)) * integral_a2)
}

/// Worden et al. (2012, BSSA) ShakeMap Instrumental Modified Mercalli Intensity (`MMI` in `[1.0, 10.0]`).
pub fn worden_mmi(pga_m_s2: f64, pgv_m_s: f64) -> f64 {
    let pga_cgs = abs(pga_m_s2) * 100.0;
    let pgv_cgs = abs(pgv_m_s) * 100.0;
    if pga_cgs <= 1e-12 && pgv_cgs <= 1e-12 {
        return 1.0;
    }

    let mmi_pga = if pga_cgs <= 1e-12 {
        1.0
    } else {
        let lp = log10(pga_cgs);
        if lp <= 1.57 {
            1.78 + 1.55 * lp
        } else {
            -1.60 + 3.70 * lp
        }
    };

    let mmi_pgv = if pgv_cgs <= 1e-12 {
        mmi_pga
    } else {
        let lv = log10(pgv_cgs);
        if lv <= 0.53 {
            3.78 + 1.47 * lv
        } else {
            2.89 + 3.16 * lv
        }
    };

    let blended = if mmi_pga <= 3.0 {
        mmi_pga
    } else if mmi_pga >= 7.0 {
        mmi_pgv
    } else {
        let w = (mmi_pga - 3.0) / 4.0;
        (1.0 - w) * mmi_pga + w * mmi_pgv
    };
    blended.clamp(1.0, 10.0)
}

/// Compute [`GroundMotionSummary`] (`PGA`, `PGV`, `PGD`, Arias Intensity, Worden `MMI`).
pub fn summarize_ground_motion(acc_m_s2: &[f64], dt_s: f64) -> Result<GroundMotionSummary> {
    if acc_m_s2.len() < 2 {
        return Err(EarthSnailError::InvalidParameter(
            "summarize_ground_motion requires at least 2 samples".into(),
        ));
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "dt_s must be finite and positive".into(),
        ));
    }
    let mean_a = acc_m_s2.iter().copied().sum::<f64>() / (acc_m_s2.len() as f64);
    let acc_zero: Vec<f64> = acc_m_s2.iter().map(|&v| v - mean_a).collect();
    let vel = integrate_trapezoidal(&acc_zero, dt_s)?;
    let mean_v = vel.iter().copied().sum::<f64>() / (vel.len() as f64);
    let vel_zero: Vec<f64> = vel.iter().map(|&v| v - mean_v).collect();
    let disp = integrate_trapezoidal(&vel_zero, dt_s)?;

    let pga_m_s2 = peak_ground_motion(acc_m_s2);
    let pgv_m_s = peak_ground_motion(&vel);
    let pgd_m = peak_ground_motion(&disp);
    let arias_intensity_m_s = arias_intensity(acc_m_s2, dt_s)?;
    let mmi = worden_mmi(pga_m_s2, pgv_m_s);

    Ok(GroundMotionSummary {
        pga_m_s2,
        pgv_m_s,
        pgd_m,
        arias_intensity_m_s,
        worden_mmi: mmi,
    })
}

/// Japan Meteorological Agency (1996) 3-component Instrumental Seismic Intensity (`JMA Shindo`).
pub fn jma_instrumental_shindo(
    acc_z_m_s2: &[f64],
    acc_n_m_s2: &[f64],
    acc_e_m_s2: &[f64],
    sampling_rate_hz: f64,
) -> Result<f64> {
    let n = acc_z_m_s2.len();
    if n < 4 || acc_n_m_s2.len() != n || acc_e_m_s2.len() != n {
        return Err(EarthSnailError::IncompatibleTraces(
            "3C acceleration traces must have equal length >= 4".into(),
        ));
    }
    if !sampling_rate_hz.is_finite() || sampling_rate_hz <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "sampling_rate_hz must be finite and positive".into(),
        ));
    }
    let min_samples_0p3s = ((0.3 * sampling_rate_hz + 0.999999) as usize).max(1);
    if n < min_samples_0p3s {
        return Err(EarthSnailError::InvalidParameter(
            "trace duration must be at least 0.3 seconds for JMA Shindo".into(),
        ));
    }

    let nfft = next_pow2(n);
    let freqs = rfftfreq(nfft, 1.0 / sampling_rate_hz)?;
    let mut jma_gain = vec![0.0_f64; freqs.len()];
    for (k, &f) in freqs.iter().enumerate() {
        if f <= 0.0 {
            continue;
        }
        let f1 = sqrt(1.0 / f);
        let y = f / 10.0;
        let y2 = y * y;
        let y4 = y2 * y2;
        let y6 = y4 * y2;
        let y8 = y4 * y4;
        let y10 = y8 * y2;
        let y12 = y6 * y6;
        let f2 = 1.0
            / sqrt(
                1.0 + 0.694 * y2
                    + 0.241 * y4
                    + 0.0557 * y6
                    + 0.009664 * y8
                    + 0.00134 * y10
                    + 0.000155 * y12,
            );
        let u = f / 0.5;
        let f3 = sqrt((1.0 - exp(-(u * u * u))).max(0.0));
        jma_gain[k] = f1 * f2 * f3;
    }

    let mut scratch = ScratchWorkspace::new();
    let filter_gal = |src_m_s2: &[f64], scratch: &mut ScratchWorkspace| -> Result<Vec<f64>> {
        let mean = src_m_s2.iter().copied().sum::<f64>() / (n as f64);
        let scaled: Vec<f64> = src_m_s2.iter().map(|&v| (v - mean) * 100.0).collect();
        let mut spec = Vec::new();
        rfft_with_scratch(&scaled, nfft, &mut spec, scratch)?;
        for (c, &g) in spec.iter_mut().zip(jma_gain.iter()) {
            *c *= g;
        }
        let mut td = Vec::new();
        irfft_with_scratch(&spec, nfft, &mut td, scratch)?;
        td.truncate(n);
        Ok(td)
    };

    let fz = filter_gal(acc_z_m_s2, &mut scratch)?;
    let fn_ = filter_gal(acc_n_m_s2, &mut scratch)?;
    let fe = filter_gal(acc_e_m_s2, &mut scratch)?;

    let mut vec_gal: Vec<f64> = (0..n)
        .map(|i| sqrt(fz[i] * fz[i] + fn_[i] * fn_[i] + fe[i] * fe[i]))
        .collect();
    vec_gal.sort_by(|a, b| b.total_cmp(a));
    let a0 = vec_gal[min_samples_0p3s - 1];
    if a0 <= 1e-12 {
        return Ok(0.0);
    }
    Ok(2.0 * log10(a0) + 0.94)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotations_roundtrip_and_sts2_uvw() {
        let z = [1.0, -2.5, 0.75, 4.2];
        let n = [-0.5, 1.2, 3.1, -1.8];
        let e = [2.2, 0.4, -1.5, 0.9];

        let (r, t) = rotate_ne_rt(&n, &e, 137.0).unwrap();
        let (n2, e2) = rotate_rt_ne(&r, &t, 137.0).unwrap();
        for i in 0..z.len() {
            assert!(abs(n2[i] - n[i]) < 1e-14);
            assert!(abs(e2[i] - e[i]) < 1e-14);
        }

        let (l, q, t_lqt) = rotate_zne_lqt(&z, &n, &e, 215.0, 28.0).unwrap();
        let (z3, n3, e3) = rotate_lqt_zne(&l, &q, &t_lqt, 215.0, 28.0).unwrap();
        for i in 0..z.len() {
            assert!(abs(z3[i] - z[i]) < 1e-14);
            assert!(abs(n3[i] - n[i]) < 1e-14);
            assert!(abs(e3[i] - e[i]) < 1e-14);
        }

        let (u, v, w) = rotate_zne_uvw(&z, &n, &e).unwrap();
        let (zu, nu, eu) = rotate_uvw_zne(&u, &v, &w).unwrap();
        for i in 0..z.len() {
            assert!(abs(zu[i] - z[i]) < 1e-14);
            assert!(abs(nu[i] - n[i]) < 1e-14);
            assert!(abs(eu[i] - e[i]) < 1e-14);
        }
    }

    #[test]
    fn test_silver_chan_and_fk_and_nigam_jennings() {
        let fs = 50.0;
        let n = if cfg!(miri) { 80 } else { 200 };
        let true_phi = 35.0 * DEG_TO_RAD;
        let delay_samp = 4usize;
        let mut fast = vec![0.0_f64; n];
        for (i, slot) in fast.iter_mut().enumerate() {
            let t = (i as f64) / fs;
            let u = (t - 0.7) * 6.0;
            *slot = (1.0 - 2.0 * u * u) * exp(-u * u);
        }
        let pol_diff = 40.0 * DEG_TO_RAD;
        let mut north = vec![0.0_f64; n];
        let mut east = vec![0.0_f64; n];
        for i in 0..n {
            let f_val = fast[i] * cos(pol_diff);
            let s_val = if i >= delay_samp {
                fast[i - delay_samp] * sin(pol_diff)
            } else {
                0.0
            };
            north[i] = f_val * cos(true_phi) - s_val * sin(true_phi);
            east[i] = f_val * sin(true_phi) + s_val * cos(true_phi);
        }
        let phi_step = if cfg!(miri) { 35.0 } else { 5.0 };
        let sc = silver_chan_splitting(&north, &east, fs, 0.20, phi_step, 0.02).unwrap();
        assert!(abs(sc.fast_phi_deg - 35.0) <= phi_step + 1e-9);
        assert!(abs(sc.dt_s - 0.08) <= 0.02 + 1e-9);

        let stations = [
            ArrayStation::new(0.0, 0.0),
            ArrayStation::new(2.0, 0.0),
            ArrayStation::new(0.0, 2.0),
            ArrayStation::new(-1.5, -1.5),
        ];
        let sx_true = 0.10_f64;
        let sy_true = 0.05_f64;
        let slow = hypot(sx_true, sy_true);
        let expected_baz = atan2(sx_true, sy_true) * RAD_TO_DEG;
        let n_fk = if cfg!(miri) { 64 } else { 256 };
        let mut arr_traces = vec![vec![0.0_f64; n_fk]; 4];
        for (j, st) in stations.iter().enumerate() {
            let adv_s = sx_true * st.x_east_km + sy_true * st.y_north_km;
            for (i, slot) in arr_traces[j].iter_mut().enumerate() {
                let t = (i as f64) / 20.0 + adv_s;
                *slot = sin(2.0 * PI * 2.5 * t) + 0.5 * cos(2.0 * PI * 3.75 * t);
            }
        }
        let refs: Vec<&[f64]> = arr_traces.iter().map(|v| v.as_slice()).collect();
        let fk = fk_analysis(
            &stations,
            &refs,
            20.0,
            1.5,
            4.5,
            0.20,
            if cfg!(miri) { 9 } else { 41 },
            FkMethod::Conventional,
        )
        .unwrap();
        assert!(abs(fk.horizontal_slowness_s_per_km - slow) < 0.02);
        assert!(abs(fk.back_azimuth_deg - expected_baz) < 5.0);
        assert!(fk.semblance > 0.95);

        let mut acc = vec![0.0_f64; 100];
        for (i, slot) in acc.iter_mut().enumerate() {
            *slot = 2.0 * sin(2.0 * PI * 2.0 * (i as f64) * 0.01);
        }
        let spec =
            nigam_jennings_response_spectra(&acc, 0.01, &[0.0, 0.005, 0.5, 50.0], 0.05).unwrap();
        let pga = peak_ground_motion(&acc);
        assert!(abs(spec.psa_m_s2[0] - pga) < 1e-12);
        assert!(abs(spec.psa_m_s2[1] - pga) < 0.05 * pga);
        assert!(spec.psa_m_s2[2] > 3.0 * pga);
    }
}
