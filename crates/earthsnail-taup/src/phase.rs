//! General IASPEI / TauP phase-name grammar, zero-regex leg tokenizer, branch-sequence
//! compiler, LVZ shadow-zone splitter, Brent shooting refiner, and pierce/path evaluators.

use crate::models::MaterialProperty;
use crate::tau_model::{TauModel, TimeDist};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
use earthsnail_core::{math, EarthSnailError, Result};

/// Default Brent ray-parameter tolerance (`s/rad`) for travel-time queries (`10x` tighter than
/// `obspy.taup`'s `0.1 s/rad`, achieving `< 1e-9 s` travel-time accuracy via stationary $\theta$).
pub const DEFAULT_RAY_PARAM_TOL: f64 = 1e-2;
/// Tighter Brent ray-parameter tolerance (`s/rad`) for ray-path and pierce-point trajectories.
pub const DEFAULT_PATH_RAY_PARAM_TOL: f64 = 1e-6;
/// Maximum head-wave refraction arc along Moho (`20 deg` in radians).
pub const MAX_REFRACTION_RAD: f64 = 20.0 * PI / 180.0;
/// Maximum core-mantle / inner-core boundary diffraction arc (`60 deg` in radians).
pub const MAX_DIFFRACTION_RAD: f64 = 60.0 * PI / 180.0;
/// Maximum Dekker-Brent root-finding iterations per arrival.
pub const MAX_REFINE_ITER: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndAction {
    Turn,
    ReflectUnderside,
    ReflectTopside,
    TransUp,
    TransDown,
    Diffract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoreSubbranch {
    None,
    PkpAb,
    PkpBc,
}

#[derive(Debug, Clone, Copy)]
struct PhaseSegment {
    start_branch: usize,
    end_branch: usize,
    end_action: EndAction,
    is_down_going: bool,
}

/// Raw phase arrival produced by `SeismicPhase` before high-level geographic decoration.
#[derive(Debug, Clone)]
pub struct RawArrival {
    /// Requested phase name (e.g. `"P"`, `"PKP"`, `"PKPab"`, `"P410s"`, `"3.5kmps"`).
    pub name: String,
    /// Discontinuity-snapped purist phase name (e.g. `"P400.0s"` on `prem`).
    pub purist_name: String,
    /// Requested epicentral distance in degrees.
    pub distance_deg: f64,
    /// Unwrapped great-circle travel distance in radians.
    pub purist_dist_rad: f64,
    /// Travel time in seconds.
    pub time_s: f64,
    /// Spherical ray parameter $p$ in `s/rad`.
    pub ray_param_s_per_rad: f64,
    /// Bracketing index in `SeismicPhase::ray_param`.
    pub ray_param_index: usize,
    /// Source focal depth in kilometers.
    pub source_depth_km: f64,
    /// Receiver burial depth in kilometers.
    pub receiver_depth_km: f64,
    /// Takeoff angle from downward vertical at source in `[0, 180]` degrees.
    pub takeoff_angle_deg: f64,
    /// Incident angle from vertical at receiver in `[0, 180]` degrees.
    pub incident_angle_deg: f64,
}

/// Expands `ttimes` / `TauP` phase-list aliases (`"ttp"`, `"tts"`, `"ttp+"`, `"tts+"`,
/// `"ttbasic"`, `"ttall"`) and returns sorted, deduplicated phase names.
pub fn parse_phase_list<S: AsRef<str>>(phase_list: &[S]) -> Vec<String> {
    let mut names = Vec::with_capacity(phase_list.len().max(8));
    for item in phase_list {
        for sub in item.as_ref().split(',') {
            let trimmed = sub.trim();
            if trimmed.is_empty() {
                continue;
            }
            expand_phase_alias(trimmed, &mut names);
        }
    }
    names.sort();
    names.dedup();
    names
}

fn expand_phase_alias(phase_name: &str, out: &mut Vec<String>) {
    let lower = phase_name.to_ascii_lowercase();
    match lower.as_str() {
        "ttp" | "tts" | "ttbasic" | "tts+" | "ttp+" | "ttall" => {
            if matches!(lower.as_str(), "ttp" | "ttp+" | "ttbasic" | "ttall") {
                for p in ["p", "P", "Pn", "Pdiff", "PKP", "PKiKP", "PKIKP"] {
                    out.push(p.to_string());
                }
            }
            if matches!(lower.as_str(), "tts" | "tts+" | "ttbasic" | "ttall") {
                for p in ["s", "S", "Sn", "Sdiff", "SKS", "SKIKS"] {
                    out.push(p.to_string());
                }
            }
            if matches!(lower.as_str(), "ttp+" | "ttbasic" | "ttall") {
                for p in [
                    "PcP", "pP", "pPdiff", "pPKP", "pPKIKP", "pPKiKP", "sP", "sPdiff", "sPKP",
                    "sPKIKP", "sPKiKP",
                ] {
                    out.push(p.to_string());
                }
            }
            if matches!(lower.as_str(), "tts+" | "ttbasic" | "ttall") {
                for p in [
                    "sS", "sSdiff", "sSKS", "sSKIKS", "ScS", "pS", "pSdiff", "pSKS", "pSKIKS",
                ] {
                    out.push(p.to_string());
                }
            }
            if matches!(lower.as_str(), "ttbasic" | "ttall") {
                for p in [
                    "ScP",
                    "SKP",
                    "SKIKP",
                    "PKKP",
                    "PKIKKIKP",
                    "SKKP",
                    "SKIKKIKP",
                    "PP",
                    "PKPPKP",
                    "PKIKPPKIKP",
                ] {
                    out.push(p.to_string());
                }
            }
            if lower == "ttall" {
                for p in [
                    "SKiKP",
                    "PP",
                    "ScS",
                    "PcS",
                    "PKS",
                    "PKIKS",
                    "PKKS",
                    "PKIKKIKS",
                    "SKKS",
                    "SKIKKIKS",
                    "SKSSKS",
                    "SKIKSSKIKS",
                    "SS",
                    "SP",
                    "PS",
                ] {
                    out.push(p.to_string());
                }
            }
        }
        _ => out.push(phase_name.to_string()),
    }
}

/// Maps explicit IASPEI core triplication labels (`PKPab`, `PKPbc`, `PKPdf`, `SKSac`, `SKSdf`,
/// and depth-phase variants `pPKPab`, `sPKPbc`, etc.) onto canonical TauP phase legs plus
/// optional caustic sub-branch filtering.
fn normalize_core_alias(raw_name: &str) -> (String, CoreSubbranch) {
    if let Some(prefix) = raw_name.strip_suffix("PKPdf") {
        return (format!("{prefix}PKIKP"), CoreSubbranch::None);
    }
    if let Some(prefix) = raw_name.strip_suffix("SKSdf") {
        return (format!("{prefix}SKIKS"), CoreSubbranch::None);
    }
    if let Some(prefix) = raw_name.strip_suffix("SKSac") {
        return (format!("{prefix}SKS"), CoreSubbranch::None);
    }
    if let Some(prefix) = raw_name.strip_suffix("PKPab") {
        return (format!("{prefix}PKP"), CoreSubbranch::PkpAb);
    }
    if let Some(prefix) = raw_name.strip_suffix("PKPbc") {
        return (format!("{prefix}PKP"), CoreSubbranch::PkpBc);
    }
    (raw_name.to_string(), CoreSubbranch::None)
}

/// Tokenizes a TauP phase name into individual legs terminated by `"END"`, without regex overhead.
pub fn leg_puller(name: &str) -> Result<Vec<String>> {
    if name.is_empty() {
        return Err(EarthSnailError::PhaseParse("empty phase name".into()));
    }
    // Normalize Unicode modifier circumflex `ˆ` (U+02C6) to ASCII `^` if present.
    let normalized: String = name
        .chars()
        .map(|c| if c == '\u{02c6}' { '^' } else { c })
        .collect();
    let s = normalized.as_str();
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut legs = Vec::with_capacity(8);

    while pos < len {
        let rem = &s[pos..];

        // 1. Surface-wave horizontal velocity `<number>kmps`
        if let Some(num_len) = scan_unsigned_float(rem) {
            if rem[num_len..].starts_with("kmps") {
                let speed_str = &rem[..num_len];
                let speed = speed_str.parse::<f64>().map_err(|_| {
                    EarthSnailError::PhaseParse(format!(
                        "invalid surface-wave velocity '{speed_str}' in '{name}'"
                    ))
                })?;
                if !speed.is_finite() || speed <= 0.0 {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "non-positive surface-wave velocity {speed} in '{name}'"
                    )));
                }
                legs.push(rem[..num_len + 4].to_string());
                pos += num_len + 4;
                continue;
            }
        }

        // 2. Multi-character composite legs
        let composites = [
            "Pdiff", "Sdiff", "Kdiff", "Ped", "Sed", "Pn", "Sn", "Pg", "Sg", "Pb", "Sb",
        ];
        if let Some(&comp) = composites.iter().find(|&&c| rem.starts_with(c)) {
            legs.push(comp.to_string());
            pos += comp.len();
            continue;
        }

        // 3. Underside `^` or topside `v` reflection tokens `^[mci|<depth>]`, `v[mci|<depth>]`
        let b0 = bytes[pos];
        if b0 == b'^' || b0 == b'v' {
            if pos + 1 >= len {
                return Err(EarthSnailError::PhaseParse(format!(
                    "dangling reflection '{}' at end of phase '{name}'",
                    b0 as char
                )));
            }
            let after = &rem[1..];
            if after.starts_with('m') || after.starts_with('c') || after.starts_with('i') {
                legs.push(rem[..2].to_string());
                pos += 2;
                continue;
            }
            if let Some(num_len) = scan_unsigned_float(after) {
                legs.push(rem[..1 + num_len].to_string());
                pos += 1 + num_len;
                continue;
            }
            return Err(EarthSnailError::PhaseParse(format!(
                "invalid reflection target in '{rem}' of phase '{name}'"
            )));
        }

        // 4. Reject invalid direct `[PS][ps]` without an intervening interface.
        if (b0 == b'P' || b0 == b'S') && pos + 1 < len {
            let b1 = bytes[pos + 1];
            if b1 == b'p' || b1 == b's' {
                return Err(EarthSnailError::PhaseParse(format!(
                    "invalid phase '{name}': '{}' cannot be followed directly by '{}'",
                    b0 as char, b1 as char
                )));
            }
        }

        // 5. Single-letter wave or boundary symbols.
        if matches!(
            b0,
            b'K' | b'k' | b'I' | b'i' | b'J' | b'm' | b'c' | b'P' | b'p' | b'S' | b's'
        ) {
            legs.push((b0 as char).to_string());
            pos += 1;
            continue;
        }

        // 6. Discontinuity depth number `<depth>`
        if let Some(num_len) = scan_unsigned_float(rem) {
            legs.push(rem[..num_len].to_string());
            pos += num_len;
            continue;
        }

        return Err(EarthSnailError::PhaseParse(format!(
            "unrecognized token '{}' in phase '{name}'",
            rem
        )));
    }

    if legs.is_empty() {
        return Err(EarthSnailError::PhaseParse(format!(
            "no valid legs in phase '{name}'"
        )));
    }
    legs.push("END".to_string());
    Ok(legs)
}

/// Scans `\.?\d+\.?\d*` (requiring at least one ASCII digit and at most one decimal point).
fn scan_unsigned_float(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    let mut has_digit = false;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            has_digit = true;
            i += 1;
        }
        return has_digit.then_some(i);
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        has_digit = true;
        i += 1;
    }
    if !has_digit {
        return None;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    Some(i)
}

/// Compiled seismic phase over a depth-corrected `TauModel`.
#[derive(Debug, Clone)]
pub struct SeismicPhase {
    /// Requested phase name.
    pub name: String,
    /// Canonical TauP phase name after alias expansion.
    pub canonical_name: String,
    /// Discontinuity-snapped purist phase name.
    pub purist_name: String,
    /// Source depth in kilometers.
    pub source_depth: f64,
    /// Receiver depth in kilometers.
    pub receiver_depth: f64,
    /// Tokenized phase legs ending with `"END"`.
    pub legs: Vec<String>,
    /// Minimum valid ray parameter (`s/rad`), or `-1.0` if phase cannot exist.
    pub min_ray_param: f64,
    /// Maximum valid ray parameter (`s/rad`), or `-1.0` if phase cannot exist.
    pub max_ray_param: f64,
    /// Index of `min_ray_param` in `TauModel::ray_params`.
    pub min_ray_param_index: usize,
    /// Index of `max_ray_param` in `TauModel::ray_params`.
    pub max_ray_param_index: usize,
    /// Minimum observable distance (rad).
    pub min_distance: f64,
    /// Maximum observable distance (rad).
    pub max_distance: f64,
    /// Ray-parameter grid (`s/rad`) for this phase (with repeated values marking LVZ shadow zones).
    pub ray_param: Vec<f64>,
    /// Cumulative epicentral distance $\Delta(p_k)$ (rad) along `ray_param`.
    pub dist: Vec<f64>,
    /// Cumulative travel time $T(p_k)$ (s) along `ray_param`.
    pub time: Vec<f64>,
    /// Ordered sequence of `TauBranch` indices traversed by the phase.
    pub branch_seq: Vec<usize>,
    /// Direction (`true` = downgoing, `false` = upgoing) for each entry in `branch_seq`.
    pub down_going: Vec<bool>,
    /// Wave type (`true` = P-type, `false` = S-type) for each entry in `branch_seq`.
    pub wave_type: Vec<bool>,
    /// Indices into `branch_seq` after which a Moho head wave or CMB/ICB diffraction occurs.
    pub head_or_diffract_seq: Vec<usize>,
    /// Precomputed non-zero `(branch_idx, p_top_l, p_bot_l, pm, s_top_l, s_bot_l, sm)` multipliers.
    active_branch_mult: Vec<(usize, usize, usize, u32, usize, usize, u32)>,
    /// Precomputed `(seg_start, seg_end_excl, min_dist, max_dist)` bounding boxes for O(1) block culling.
    dist_blocks: Vec<(usize, usize, f64, f64)>,
    is_kmps: bool,
    subbranch: CoreSubbranch,
}

impl SeismicPhase {
    /// Compiles `name` against `tau_model` for a surface receiver (`receiver_depth = 0.0`).
    pub fn new(name: &str, tau_model: &TauModel) -> Result<Self> {
        Self::with_receiver_depth(name, tau_model, 0.0)
    }

    /// Compiles `name` against `tau_model` for an arbitrary `receiver_depth` (km).
    pub fn with_receiver_depth(
        name: &str,
        tau_model: &TauModel,
        receiver_depth: f64,
    ) -> Result<Self> {
        let (canonical_name, subbranch) = normalize_core_alias(name);
        let legs = leg_puller(&canonical_name)?;
        let is_kmps = legs.len() == 2 && legs[0].ends_with("kmps");
        let purist_name = if is_kmps {
            name.to_string()
        } else {
            create_purist_name(&legs, tau_model)?
        };

        let mut phase = Self {
            name: name.to_string(),
            canonical_name,
            purist_name,
            source_depth: tau_model.source_depth,
            receiver_depth,
            legs,
            min_ray_param: -1.0,
            max_ray_param: -1.0,
            min_ray_param_index: 0,
            max_ray_param_index: 0,
            min_distance: 0.0,
            max_distance: -1.0,
            ray_param: Vec::new(),
            dist: Vec::new(),
            time: Vec::new(),
            branch_seq: Vec::new(),
            down_going: Vec::new(),
            wave_type: Vec::new(),
            head_or_diffract_seq: Vec::new(),
            active_branch_mult: Vec::new(),
            dist_blocks: Vec::new(),
            is_kmps,
            subbranch,
        };

        phase.parse_name(tau_model)?;
        phase.sum_branches(tau_model)?;
        Ok(phase)
    }

    /// Returns `true` if this phase has at least one valid ray-parameter branch at `source_depth`.
    #[inline]
    pub fn has_arrivals(&self) -> bool {
        !self.ray_param.is_empty() && self.max_distance >= 0.0
    }

    fn parse_name(&mut self, tau_model: &TauModel) -> Result<()> {
        if self.is_kmps {
            return Ok(());
        }
        if self.canonical_name.contains('J') && !tau_model.s_mod.config.allow_inner_core_s {
            return Err(EarthSnailError::PhaseParse(format!(
                "inner-core shear 'J' phases are disabled for model: {}",
                self.name
            )));
        }

        let first_leg = self.legs[0].as_str();
        let mut is_p_wave = if matches!(first_leg, "p" | "k" | "I")
            || first_leg.starts_with('P')
            || first_leg.starts_with('K')
        {
            true
        } else if matches!(first_leg, "s" | "J") || first_leg.starts_with('S') {
            false
        } else {
            return Err(EarthSnailError::PhaseParse(format!(
                "unknown starting phase leg '{first_leg}' in '{}'",
                self.name
            )));
        };
        let mut is_p_wave_prev = is_p_wave;

        let upgoing_rec_branch = tau_model.find_branch(self.receiver_depth)?;
        let downgoing_rec_branch = upgoing_rec_branch.wrapping_sub(1);

        // Exclude S sources inside the fluid outer core.
        if first_leg.starts_with('s') || first_leg.starts_with('S') {
            let sdep = tau_model.source_depth;
            if tau_model.cmb_depth < sdep && sdep < tau_model.iocb_depth {
                self.max_ray_param = -1.0;
                self.min_ray_param = -1.0;
                return Ok(());
            }
        }

        let mut current_branch: usize;
        let mut end_action: EndAction;

        if first_leg.starts_with('P') || first_leg.starts_with('S') {
            current_branch = tau_model.source_branch;
            end_action = EndAction::ReflectUnderside;
            self.max_ray_param = tau_model
                .get_tau_branch(tau_model.source_branch, is_p_wave)
                .max_ray_param;
        } else if first_leg == "p" || first_leg == "s" {
            end_action = EndAction::ReflectTopside;
            let s_layer_num = tau_model
                .s_mod
                .layer_number_above(tau_model.source_depth, is_p_wave_prev)?;
            let layer = tau_model
                .s_mod
                .get_slowness_layer(s_layer_num, is_p_wave_prev);
            self.max_ray_param = layer.bot_p;
            if let Some(hsz) = tau_model.s_mod.high_slowness_range_for(
                tau_model.source_depth,
                self.max_ray_param,
                is_p_wave_prev,
            ) {
                self.max_ray_param = self.max_ray_param.min(hsz.ray_param);
            }
            if tau_model.source_branch != 0 {
                current_branch = tau_model.source_branch - 1;
            } else {
                self.max_ray_param = -1.0;
                self.min_ray_param = -1.0;
                return Ok(());
            }
        } else {
            return Err(EarthSnailError::PhaseParse(format!(
                "starting leg '{first_leg}' not valid from crustal/mantle source in '{}'",
                self.name
            )));
        }

        if self.receiver_depth != 0.0 {
            let penult = self.legs[self.legs.len() - 2].as_str();
            if penult == "Ped" || penult == "Sed" {
                if upgoing_rec_branch == 0 {
                    self.max_ray_param = -1.0;
                    self.min_ray_param = -1.0;
                    return Ok(());
                }
                self.max_ray_param = self.max_ray_param.min(
                    tau_model
                        .get_tau_branch(downgoing_rec_branch, is_p_wave)
                        .min_turn_ray_param,
                );
            } else {
                self.max_ray_param = self.max_ray_param.min(
                    tau_model
                        .get_tau_branch(upgoing_rec_branch, is_p_wave)
                        .max_ray_param,
                );
            }
        }

        self.min_ray_param = 0.0;

        let legs_local = self.legs.clone();
        let mut is_next_leg_depth = false;
        let mut current_leg_str = "START";
        let mut next_leg_str = legs_local[0].as_str();
        let mut segments: Vec<PhaseSegment> = Vec::with_capacity(8);

        let num_real_legs = legs_local.len() - 1;
        for leg_num in 0..num_real_legs {
            let prev_leg = current_leg_str;
            current_leg_str = next_leg_str;
            next_leg_str = legs_local[leg_num + 1].as_str();
            let is_leg_depth = is_next_leg_depth;

            let (next_leg_depth, next_is_num) = match next_leg_str.parse::<f64>() {
                Ok(d) if d.is_finite() => (d, true),
                _ => (-1.0, false),
            };
            is_next_leg_depth = next_is_num;

            is_p_wave_prev = is_p_wave;
            if matches!(current_leg_str, "p" | "k" | "I") || current_leg_str.starts_with('P') {
                is_p_wave = true;
            } else if matches!(current_leg_str, "s" | "J") || current_leg_str.starts_with('S') {
                is_p_wave = false;
            } else if current_leg_str.starts_with('K') {
                // Retain `is_p_wave` from entering mantle/inner-core leg so outer-core LVZ logic
                // distinguishes PKP (mantle P -> K) from SKS (mantle S -> K).
            }

            if !self.branch_seq.is_empty() && is_p_wave_prev != is_p_wave {
                let last_b = *self.branch_seq.last().unwrap();
                self.phase_conversion(tau_model, last_b, end_action, is_p_wave_prev)?;
            }

            if current_leg_str == "Ped" || current_leg_str == "Sed" {
                if next_leg_str == "END" {
                    if self.receiver_depth > 0.0 && upgoing_rec_branch > 0 {
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            downgoing_rec_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        self.max_ray_param = -1.0;
                        self.min_ray_param = -1.0;
                        return Ok(());
                    }
                } else if next_leg_str == "Pdiff" || next_leg_str == "Sdiff" {
                    end_action = EndAction::Diffract;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.cmb_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                    if current_leg_str.as_bytes()[0] != next_leg_str.as_bytes()[0] {
                        is_p_wave = !is_p_wave;
                    }
                } else if next_leg_str.starts_with('K') {
                    end_action = EndAction::TransDown;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.cmb_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str == "m" {
                    end_action = EndAction::TransDown;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.moho_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if let Some(rest) = next_leg_str.strip_prefix('v') {
                    let discon_branch = closest_branch_to_depth(tau_model, rest)?;
                    if discon_branch > 0 && current_branch < discon_branch {
                        end_action = EndAction::ReflectTopside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        return Err(EarthSnailError::PhaseParse(format!(
                            "invalid topside reflection '{next_leg_str}' after '{current_leg_str}'"
                        )));
                    }
                } else if next_leg_str == "I" || next_leg_str == "J" {
                    if tau_model.cmb_depth == tau_model.iocb_depth {
                        end_action = EndAction::TransDown;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.cmb_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        self.max_ray_param = -1.0;
                        return Ok(());
                    }
                } else {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "phase leg '{current_leg_str}' followed by '{next_leg_str}' not recognized"
                    )));
                }
            } else if matches!(current_leg_str, "p" | "s" | "k") {
                if next_leg_str.starts_with('v') {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "upgoing '{current_leg_str}' cannot precede topside reflection '{next_leg_str}'"
                    )));
                } else if let Some(rest) = next_leg_str.strip_prefix('^') {
                    let discon_branch = closest_branch_to_depth(tau_model, rest)?;
                    if current_branch >= discon_branch {
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        self.max_ray_param = -1.0;
                        self.min_ray_param = -1.0;
                        return Ok(());
                    }
                } else if next_leg_str == "m" && current_branch >= tau_model.moho_branch {
                    end_action = EndAction::TransUp;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.moho_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str.starts_with('P')
                    || next_leg_str.starts_with('S')
                    || next_leg_str.starts_with('K')
                    || next_leg_str == "END"
                {
                    let discon_branch = if next_leg_str == "END" {
                        upgoing_rec_branch
                    } else if next_leg_str.starts_with('K') {
                        tau_model.cmb_branch
                    } else {
                        0
                    };
                    end_action = if current_leg_str == "k" && !next_leg_str.starts_with('K') {
                        EndAction::TransUp
                    } else {
                        EndAction::ReflectUnderside
                    };
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        discon_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if is_next_leg_depth {
                    let discon_branch = closest_branch_to_depth(tau_model, next_leg_str)?;
                    end_action = EndAction::TransUp;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        discon_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "phase leg '{current_leg_str}' followed by '{next_leg_str}' not recognized"
                    )));
                }
            } else if current_leg_str == "P" || current_leg_str == "S" {
                if matches!(
                    next_leg_str,
                    "P" | "S" | "Pn" | "Sn" | "Pdiff" | "Sdiff" | "END"
                ) {
                    if end_action == EndAction::TransDown
                        || end_action == EndAction::ReflectUnderside
                    {
                        end_action = EndAction::Turn;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.cmb_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    }
                    end_action = EndAction::ReflectUnderside;
                    let target = if next_leg_str == "END" {
                        upgoing_rec_branch
                    } else {
                        0
                    };
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        target,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if let Some(rest) = next_leg_str.strip_prefix('v') {
                    let discon_branch = closest_branch_to_depth(tau_model, rest)?;
                    if discon_branch > 0 && current_branch < discon_branch {
                        end_action = EndAction::ReflectTopside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        // Source is deeper than the requested topside reflector -> phase cannot exist.
                        self.max_ray_param = -1.0;
                        self.min_ray_param = -1.0;
                        return Ok(());
                    }
                } else if let Some(rest) = next_leg_str.strip_prefix('^') {
                    let discon_branch = closest_branch_to_depth(tau_model, rest)?;
                    if prev_leg.starts_with('K') {
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else if prev_leg.starts_with('^')
                        || matches!(prev_leg, "P" | "S" | "p" | "s" | "START")
                    {
                        end_action = EndAction::Turn;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.cmb_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else if (prev_leg.starts_with('v')
                        && discon_branch < closest_branch_to_depth(tau_model, &prev_leg[1..])?)
                        || (prev_leg == "m" && discon_branch < tau_model.moho_branch)
                        || (prev_leg == "c" && discon_branch < tau_model.cmb_branch)
                    {
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        return Err(EarthSnailError::PhaseParse(format!(
                            "unrecognized underside reflection '{next_leg_str}' after '{current_leg_str}'"
                        )));
                    }
                } else if next_leg_str == "c" {
                    end_action = EndAction::ReflectTopside;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.cmb_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str.starts_with('K') {
                    end_action = EndAction::TransDown;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.cmb_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str == "m"
                    || (is_next_leg_depth && next_leg_depth < tau_model.cmb_depth)
                {
                    let discon_branch = closest_branch_to_depth(tau_model, next_leg_str)?;
                    if matches!(
                        end_action,
                        EndAction::Turn | EndAction::ReflectTopside | EndAction::TransUp
                    ) {
                        if discon_branch > current_branch {
                            return Err(EarthSnailError::PhaseParse(format!(
                                "discontinuity '{next_leg_str}' is deeper than upgoing branch"
                            )));
                        }
                        end_action = EndAction::TransUp;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            discon_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        let next_next = self
                            .legs
                            .get(leg_num + 2)
                            .map(String::as_str)
                            .unwrap_or("END");
                        if next_next == "p" || next_next == "s" {
                            end_action = EndAction::Turn;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                tau_model.cmb_branch - 1,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                            end_action = EndAction::TransUp;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                discon_branch,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                        } else if next_next == "P" || next_next == "S" {
                            if discon_branch > current_branch {
                                end_action = EndAction::TransDown;
                                current_branch = self.add_to_branch(
                                    tau_model,
                                    current_branch,
                                    discon_branch - 1,
                                    is_p_wave,
                                    end_action,
                                    &mut segments,
                                )?;
                            } else {
                                self.max_ray_param = -1.0;
                                return Ok(());
                            }
                        } else {
                            return Err(EarthSnailError::PhaseParse(format!(
                                "phase sequence '{current_leg_str}' -> '{next_leg_str}' -> '{next_next}' not recognized"
                            )));
                        }
                    }
                } else {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "phase leg '{current_leg_str}' followed by '{next_leg_str}' not recognized"
                    )));
                }
            } else if current_leg_str.starts_with('P') || current_leg_str.starts_with('S') {
                if current_leg_str == "Pdiff" || current_leg_str == "Sdiff" {
                    let cmb_turn_p = tau_model
                        .get_tau_branch(tau_model.cmb_branch - 1, is_p_wave)
                        .min_turn_ray_param;
                    if self.max_ray_param >= cmb_turn_p && cmb_turn_p >= self.min_ray_param {
                        if (current_branch < tau_model.cmb_branch - 1
                            || (current_branch == tau_model.cmb_branch - 1
                                && end_action != EndAction::Diffract))
                            && !prev_leg.starts_with('K')
                        {
                            end_action = EndAction::Diffract;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                tau_model.cmb_branch - 1,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                        }
                        self.head_or_diffract_seq.push(self.branch_seq.len() - 1);
                        self.max_ray_param = cmb_turn_p;
                        self.min_ray_param = cmb_turn_p;

                        if next_leg_str == "END" {
                            end_action = EndAction::ReflectUnderside;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                upgoing_rec_branch,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                        } else if next_leg_str.starts_with('K') {
                            end_action = EndAction::TransDown;
                            current_branch += 1;
                        } else if next_leg_str.starts_with('P') || next_leg_str.starts_with('S') {
                            end_action = EndAction::ReflectUnderside;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                0,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                        } else {
                            return Err(EarthSnailError::PhaseParse(format!(
                                "'{current_leg_str}' followed by '{next_leg_str}' not recognized"
                            )));
                        }
                    } else {
                        self.max_ray_param = -1.0;
                        return Ok(());
                    }
                } else if matches!(current_leg_str, "Pg" | "Sg" | "Pn" | "Sn" | "Pb" | "Sb") {
                    if current_branch >= tau_model.moho_branch {
                        self.max_ray_param = -1.0;
                        return Ok(());
                    }
                    if current_leg_str == "Pg" || current_leg_str == "Sg" {
                        end_action = EndAction::Turn;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.moho_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                        let target = if next_leg_str == "END" {
                            upgoing_rec_branch
                        } else {
                            0
                        };
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            target,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else if current_leg_str == "Pb" || current_leg_str == "Sb" {
                        // Lower-crustal turning ray between Conrad discontinuity and Moho!
                        let Some(cb) = tau_model.conrad_branch else {
                            self.max_ray_param = -1.0;
                            return Ok(());
                        };
                        if cb >= tau_model.moho_branch {
                            self.max_ray_param = -1.0;
                            return Ok(());
                        }
                        if current_branch < cb {
                            end_action = EndAction::TransDown;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                cb - 1,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                        }
                        end_action = EndAction::Turn;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.moho_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                        let target = if next_leg_str == "END" {
                            upgoing_rec_branch
                        } else {
                            0
                        };
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            target,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else {
                        // Pn / Sn Moho head wave
                        let moho_head_p = tau_model
                            .get_tau_branch(tau_model.moho_branch, is_p_wave)
                            .max_ray_param;
                        if self.max_ray_param >= moho_head_p && moho_head_p >= self.min_ray_param {
                            end_action = EndAction::Turn;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                tau_model.moho_branch,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                            self.head_or_diffract_seq.push(self.branch_seq.len() - 1);
                            end_action = EndAction::TransUp;
                            current_branch = self.add_to_branch(
                                tau_model,
                                current_branch,
                                tau_model.moho_branch,
                                is_p_wave,
                                end_action,
                                &mut segments,
                            )?;
                            self.min_ray_param = self.max_ray_param;
                            if next_leg_str == "END" {
                                end_action = EndAction::ReflectUnderside;
                                current_branch = self.add_to_branch(
                                    tau_model,
                                    current_branch,
                                    upgoing_rec_branch,
                                    is_p_wave,
                                    end_action,
                                    &mut segments,
                                )?;
                            } else if next_leg_str.starts_with('P') || next_leg_str.starts_with('S')
                            {
                                end_action = EndAction::ReflectUnderside;
                                current_branch = self.add_to_branch(
                                    tau_model,
                                    current_branch,
                                    0,
                                    is_p_wave,
                                    end_action,
                                    &mut segments,
                                )?;
                            }
                        } else {
                            self.max_ray_param = -1.0;
                            return Ok(());
                        }
                    }
                } else {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "unrecognized leg '{current_leg_str}'"
                    )));
                }
            } else if current_leg_str == "K" {
                if tau_model.cmb_depth == tau_model.radius_of_planet
                    || tau_model.cmb_depth == tau_model.iocb_depth
                {
                    self.max_ray_param = -1.0;
                    return Ok(());
                }
                if matches!(next_leg_str, "P" | "S" | "p" | "s" | "Pdiff" | "Sdiff") {
                    if matches!(
                        prev_leg,
                        "P" | "S" | "Pdiff" | "Sdiff" | "K" | "k" | "Kdiff" | "START"
                    ) {
                        end_action = EndAction::Turn;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.iocb_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    }
                    end_action = EndAction::TransUp;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.cmb_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str.starts_with('K') {
                    if matches!(prev_leg, "P" | "S" | "K" | "Pdiff" | "Sdiff" | "Kdiff") {
                        end_action = EndAction::Turn;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.iocb_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    }
                    end_action = EndAction::ReflectUnderside;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.cmb_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str == "I" || next_leg_str == "J" {
                    if tau_model.iocb_depth == tau_model.radius_of_planet {
                        self.max_ray_param = -1.0;
                        return Ok(());
                    }
                    end_action = EndAction::TransDown;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.iocb_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str == "i" {
                    end_action = EndAction::ReflectTopside;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.iocb_branch - 1,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "'K' followed by '{next_leg_str}' not recognized"
                    )));
                }
            } else if current_leg_str == "Kdiff" {
                let iocb_turn_p = tau_model
                    .get_tau_branch(tau_model.iocb_branch - 1, is_p_wave)
                    .min_turn_ray_param;
                if self.max_ray_param >= iocb_turn_p && iocb_turn_p >= self.min_ray_param {
                    if current_branch < tau_model.iocb_branch - 1
                        || (current_branch == tau_model.iocb_branch - 1
                            && end_action != EndAction::Diffract)
                    {
                        end_action = EndAction::Diffract;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.iocb_branch - 1,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    }
                    self.head_or_diffract_seq.push(self.branch_seq.len() - 1);
                    self.max_ray_param = iocb_turn_p;
                    self.min_ray_param = iocb_turn_p;
                    if next_leg_str.starts_with('K') {
                        end_action = EndAction::ReflectUnderside;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.cmb_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    } else if next_leg_str.starts_with('P') || next_leg_str.starts_with('S') {
                        end_action = EndAction::TransUp;
                        current_branch = self.add_to_branch(
                            tau_model,
                            current_branch,
                            tau_model.cmb_branch,
                            is_p_wave,
                            end_action,
                            &mut segments,
                        )?;
                    }
                } else {
                    self.max_ray_param = -1.0;
                    return Ok(());
                }
            } else if current_leg_str == "I" || current_leg_str == "J" {
                end_action = EndAction::Turn;
                current_branch = self.add_to_branch(
                    tau_model,
                    current_branch,
                    tau_model.num_branches() - 1,
                    is_p_wave,
                    end_action,
                    &mut segments,
                )?;
                if next_leg_str == "I" || next_leg_str == "J" {
                    end_action = EndAction::ReflectUnderside;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.iocb_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else if next_leg_str.starts_with('K') {
                    end_action = EndAction::TransUp;
                    current_branch = self.add_to_branch(
                        tau_model,
                        current_branch,
                        tau_model.iocb_branch,
                        is_p_wave,
                        end_action,
                        &mut segments,
                    )?;
                } else {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "inner-core leg '{current_leg_str}' followed by '{next_leg_str}' not recognized"
                    )));
                }
            } else if matches!(current_leg_str, "m" | "c" | "i") || current_leg_str.starts_with('^')
            {
                // Handled during preceding leg lookahead.
            } else if let Some(rest) = current_leg_str.strip_prefix('v') {
                let b = closest_branch_to_depth(tau_model, rest)?;
                if b == 0 {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "'{current_leg_str}' attempts topside reflection at the free surface"
                    )));
                }
            } else if is_leg_depth {
                let b = closest_branch_to_depth(tau_model, current_leg_str)?;
                if b == 0 && (next_leg_str == "p" || next_leg_str == "s") {
                    return Err(EarthSnailError::PhaseParse(format!(
                        "discontinuity '{current_leg_str}' snaps to free surface 0 km before '{next_leg_str}'"
                    )));
                }
            } else {
                return Err(EarthSnailError::PhaseParse(format!(
                    "unrecognized phase leg '{current_leg_str}'"
                )));
            }
        }

        if self.max_ray_param != -1.0 && !self.branch_seq.is_empty() {
            let last_b = *self.branch_seq.last().unwrap();
            let invalid_end_reflection = (end_action == EndAction::ReflectUnderside
                && upgoing_rec_branch > 0
                && downgoing_rec_branch == last_b)
                || (end_action == EndAction::ReflectTopside && upgoing_rec_branch == last_b);
            if invalid_end_reflection {
                self.min_ray_param = -1.0;
                self.max_ray_param = -1.0;
            }
        }

        Ok(())
    }

    fn phase_conversion(
        &mut self,
        tau_model: &TauModel,
        from_branch: usize,
        end_action: EndAction,
        is_p_to_s: bool,
    ) -> Result<()> {
        match end_action {
            EndAction::Turn => {
                return Err(EarthSnailError::PhaseParse(
                    "phase conversion is not allowed at turning points".into(),
                ));
            }
            EndAction::ReflectUnderside => {
                self.max_ray_param = self
                    .max_ray_param
                    .min(
                        tau_model
                            .get_tau_branch(from_branch, is_p_to_s)
                            .max_ray_param,
                    )
                    .min(
                        tau_model
                            .get_tau_branch(from_branch, !is_p_to_s)
                            .max_ray_param,
                    );
            }
            EndAction::ReflectTopside => {
                self.max_ray_param = self
                    .max_ray_param
                    .min(
                        tau_model
                            .get_tau_branch(from_branch, is_p_to_s)
                            .min_turn_ray_param,
                    )
                    .min(
                        tau_model
                            .get_tau_branch(from_branch, !is_p_to_s)
                            .min_turn_ray_param,
                    );
            }
            EndAction::TransUp => {
                if from_branch == 0 {
                    self.max_ray_param = -1.0;
                    return Ok(());
                }
                self.max_ray_param = self
                    .max_ray_param
                    .min(
                        tau_model
                            .get_tau_branch(from_branch, is_p_to_s)
                            .max_ray_param,
                    )
                    .min(
                        tau_model
                            .get_tau_branch(from_branch - 1, !is_p_to_s)
                            .min_turn_ray_param,
                    );
            }
            EndAction::TransDown => {
                if from_branch + 1 >= tau_model.num_branches() {
                    self.max_ray_param = -1.0;
                    return Ok(());
                }
                self.max_ray_param = self
                    .max_ray_param
                    .min(
                        tau_model
                            .get_tau_branch(from_branch, is_p_to_s)
                            .min_ray_param,
                    )
                    .min(
                        tau_model
                            .get_tau_branch(from_branch + 1, !is_p_to_s)
                            .max_ray_param,
                    );
            }
            EndAction::Diffract => {}
        }
        Ok(())
    }

    fn add_to_branch(
        &mut self,
        tau_model: &TauModel,
        start_branch: usize,
        mut end_branch: usize,
        is_p_wave: bool,
        end_action: EndAction,
        segments: &mut Vec<PhaseSegment>,
    ) -> Result<usize> {
        let num_branches = tau_model.num_branches();
        if end_branch > num_branches {
            return Err(EarthSnailError::PhaseParse(format!(
                "end_branch {end_branch} exceeds num_branches {num_branches}"
            )));
        }

        let (end_offset, is_down_going): (isize, bool) = match end_action {
            EndAction::Turn => {
                if end_branch >= num_branches {
                    self.min_ray_param = -1.0;
                    self.max_ray_param = -1.0;
                    return Ok(0);
                }
                self.min_ray_param = self.min_ray_param.max(
                    tau_model
                        .get_tau_branch(end_branch, is_p_wave)
                        .min_turn_ray_param,
                );
                let mut b_num = end_branch as isize;
                while b_num >= start_branch as isize {
                    let bu = b_num as usize;
                    let branch_top_in_hsz = tau_model.s_mod.depth_in_high_slowness(
                        tau_model.get_tau_branch(bu, is_p_wave).top_depth,
                        self.min_ray_param,
                        is_p_wave,
                    );
                    let is_bottom_branch = bu + 1 >= num_branches;
                    let not_crit_reflect = is_bottom_branch
                        || self.min_ray_param
                            <= tau_model.get_tau_branch(bu + 1, is_p_wave).max_ray_param;
                    if branch_top_in_hsz && not_crit_reflect {
                        if bu == 0 {
                            self.min_ray_param = -1.0;
                            self.max_ray_param = -1.0;
                            return Ok(0);
                        }
                        end_branch = bu - 1;
                        b_num -= 1;
                    } else {
                        break;
                    }
                }
                (0, true)
            }
            EndAction::ReflectUnderside => {
                if end_branch < num_branches {
                    self.max_ray_param = self.max_ray_param.min(
                        tau_model
                            .get_tau_branch(end_branch, is_p_wave)
                            .max_ray_param,
                    );
                }
                (0, false)
            }
            EndAction::ReflectTopside => {
                self.max_ray_param = self.max_ray_param.min(
                    tau_model
                        .get_tau_branch(end_branch, is_p_wave)
                        .min_turn_ray_param,
                );
                (0, true)
            }
            EndAction::TransUp => {
                self.max_ray_param = self.max_ray_param.min(
                    tau_model
                        .get_tau_branch(end_branch, is_p_wave)
                        .max_ray_param,
                );
                (-1, false)
            }
            EndAction::TransDown => {
                self.max_ray_param = self.max_ray_param.min(
                    tau_model
                        .get_tau_branch(end_branch, is_p_wave)
                        .min_ray_param,
                );
                (1, true)
            }
            EndAction::Diffract => {
                self.max_ray_param = self.max_ray_param.min(
                    tau_model
                        .get_tau_branch(end_branch, is_p_wave)
                        .min_turn_ray_param,
                );
                (0, true)
            }
        };

        let seg = PhaseSegment {
            start_branch,
            end_branch,
            end_action,
            is_down_going,
        };
        if let Some(prev) = segments.last() {
            check_segment_consistency(prev, &seg)?;
        }
        segments.push(seg);

        if is_down_going {
            if start_branch > end_branch {
                self.min_ray_param = -1.0;
                self.max_ray_param = -1.0;
            } else {
                for b in start_branch..=end_branch {
                    self.branch_seq.push(b);
                    self.down_going.push(true);
                    self.wave_type.push(is_p_wave);
                }
            }
        } else if start_branch < end_branch {
            self.min_ray_param = -1.0;
            self.max_ray_param = -1.0;
        } else {
            let mut b = start_branch;
            loop {
                self.branch_seq.push(b);
                self.down_going.push(false);
                self.wave_type.push(is_p_wave);
                if b == end_branch {
                    break;
                }
                b -= 1;
            }
        }

        let next_branch = (end_branch as isize + end_offset).max(0) as usize;
        Ok(next_branch)
    }

    fn sum_branches(&mut self, tau_model: &TauModel) -> Result<()> {
        if self.is_kmps {
            let speed_str = &self.legs[0][..self.legs[0].len() - 4];
            let speed = speed_str.parse::<f64>().map_err(|_| {
                EarthSnailError::PhaseParse(format!("invalid kmps speed '{speed_str}'"))
            })?;
            let rp = tau_model.radius_of_planet / speed;
            self.ray_param = vec![rp, rp];
            self.dist = vec![0.0, TAU];
            self.time = vec![0.0, TAU * rp];
            self.min_ray_param = rp;
            self.max_ray_param = rp;
            self.min_distance = 0.0;
            self.max_distance = TAU;
            self.rebuild_dist_blocks();
            return Ok(());
        }

        if self.max_ray_param < 0.0 || self.min_ray_param > self.max_ray_param {
            self.ray_param.clear();
            self.dist.clear();
            self.time.clear();
            self.min_ray_param = -1.0;
            self.max_ray_param = -1.0;
            self.max_distance = -1.0;
            return Ok(());
        }

        // Locate min_ray_param_index and max_ray_param_index in decreasing `tau_model.ray_params`.
        let mut min_idx = None;
        let mut max_idx = None;
        for (i, &rp) in tau_model.ray_params.iter().enumerate() {
            if rp >= self.min_ray_param {
                min_idx = Some(i);
            }
            if rp >= self.max_ray_param {
                max_idx = Some(i);
            }
        }
        let (Some(min_i), Some(max_i)) = (min_idx, max_idx) else {
            self.max_distance = -1.0;
            return Ok(());
        };
        self.min_ray_param_index = min_i;
        self.max_ray_param_index = max_i;

        if max_i == min_i {
            self.ray_param = vec![self.min_ray_param, self.min_ray_param];
        } else {
            self.ray_param = tau_model.ray_params[max_i..=min_i].to_vec();
        }

        let n_branches = tau_model.num_branches();
        let mut p_mult = vec![0u32; n_branches];
        let mut s_mult = vec![0u32; n_branches];
        for (&wt, &bs) in self.wave_type.iter().zip(self.branch_seq.iter()) {
            if wt {
                p_mult[bs] += 1;
            } else {
                s_mult[bs] += 1;
            }
        }
        self.active_branch_mult.clear();
        let s_mod = &tau_model.s_mod;
        for b in 0..n_branches {
            if p_mult[b] != 0 || s_mult[b] != 0 {
                let (p_top, p_bot) = if p_mult[b] != 0 {
                    let br = &tau_model.tau_branches[0][b];
                    (
                        s_mod.layer_number_below(br.top_depth, true)?,
                        s_mod.layer_number_above(br.bot_depth, true)?,
                    )
                } else {
                    (0, 0)
                };
                let (s_top, s_bot) = if s_mult[b] != 0 {
                    let br = &tau_model.tau_branches[1][b];
                    (
                        s_mod.layer_number_below(br.top_depth, false)?,
                        s_mod.layer_number_above(br.bot_depth, false)?,
                    )
                } else {
                    (0, 0)
                };
                self.active_branch_mult
                    .push((b, p_top, p_bot, p_mult[b], s_top, s_bot, s_mult[b]));
            }
        }

        let n_pts = self.ray_param.len();
        self.dist = vec![0.0; n_pts];
        self.time = vec![0.0; n_pts];

        let size = min_i - max_i + 1;
        for &(b, _, _, pm, _, _, sm) in &self.active_branch_mult {
            if pm != 0 {
                let pf = pm as f64;
                let taub = &tau_model.tau_branches[0][b];
                for k in 0..size {
                    self.dist[k] += pf * taub.dist[max_i + k];
                    self.time[k] += pf * taub.time[max_i + k];
                }
            }
            if sm != 0 {
                let sf = sm as f64;
                let taubs = &tau_model.tau_branches[1][b];
                for k in 0..size {
                    self.dist[k] += sf * taubs.dist[max_i + k];
                    self.time[k] += sf * taubs.time[max_i + k];
                }
            }
        }

        if self.canonical_name.contains("Sdiff") || self.canonical_name.contains("Pdiff") {
            let starts_p =
                self.canonical_name.starts_with('P') || self.canonical_name.starts_with('p');
            if tau_model.cmb_depth == tau_model.radius_of_planet
                || tau_model.s_mod.depth_in_high_slowness(
                    tau_model.cmb_depth - 1e-10,
                    self.min_ray_param,
                    starts_p,
                )
            {
                self.ray_param.clear();
                self.dist.clear();
                self.time.clear();
                self.min_ray_param = -1.0;
                self.max_ray_param = -1.0;
                self.max_distance = -1.0;
                return Ok(());
            }
            self.dist[1] = self.dist[0] + MAX_DIFFRACTION_RAD;
            self.time[1] = self.time[0] + MAX_DIFFRACTION_RAD * self.min_ray_param;
        } else if self.canonical_name.contains("Kdiff") {
            self.dist[1] = self.dist[0] + MAX_DIFFRACTION_RAD;
            self.time[1] = self.time[0] + MAX_DIFFRACTION_RAD * self.min_ray_param;
        } else if self.canonical_name.contains("Pn") || self.canonical_name.contains("Sn") {
            self.dist[1] = self.dist[0] + MAX_REFRACTION_RAD;
            self.time[1] = self.time[0] + MAX_REFRACTION_RAD * self.min_ray_param;
        } else if max_i == min_i {
            self.dist[1] = self.dist[0];
            self.time[1] = self.time[0];
        }

        // Insert repeated-ray-parameter shadow-zone breaks for any crossed LVZ.
        for is_pwave in [true, false] {
            let hsz_list = if is_pwave {
                &tau_model.s_mod.high_slowness_layer_depths_p
            } else {
                &tau_model.s_mod.high_slowness_layer_depths_s
            };
            let mut index_offset = 0usize;
            for hszi in hsz_list {
                if self.max_ray_param > hszi.ray_param && hszi.ray_param > self.min_ray_param {
                    let branch_num = tau_model.find_branch(hszi.top_depth)?;
                    let mut found_overlap = false;
                    for leg_idx in 1..self.branch_seq.len() {
                        if self.branch_seq[leg_idx] == branch_num
                            && self.wave_type[leg_idx] == is_pwave
                            && self.down_going[leg_idx]
                            && branch_num > 0
                            && self.branch_seq[leg_idx - 1] == branch_num - 1
                            && self.wave_type[leg_idx - 1] == is_pwave
                            && self.down_going[leg_idx - 1]
                        {
                            found_overlap = true;
                            break;
                        }
                    }
                    if found_overlap {
                        if let Some(hsz_idx) =
                            self.ray_param.iter().position(|&p| p == hszi.ray_param)
                        {
                            let model_rp_idx = max_i + hsz_idx - index_offset;
                            let mut pre_d = 0.0;
                            let mut pre_t = 0.0;
                            for &(b, _, _, pm, _, _, sm) in &self.active_branch_mult {
                                if pm != 0
                                    && tau_model.tau_branches[0][b].top_depth < hszi.top_depth
                                {
                                    pre_d += (pm as f64)
                                        * tau_model.tau_branches[0][b].dist[model_rp_idx];
                                    pre_t += (pm as f64)
                                        * tau_model.tau_branches[0][b].time[model_rp_idx];
                                }
                                if sm != 0
                                    && tau_model.tau_branches[1][b].top_depth < hszi.top_depth
                                {
                                    pre_d += (sm as f64)
                                        * tau_model.tau_branches[1][b].dist[model_rp_idx];
                                    pre_t += (sm as f64)
                                        * tau_model.tau_branches[1][b].time[model_rp_idx];
                                }
                            }
                            self.ray_param.insert(hsz_idx, hszi.ray_param);
                            self.dist.insert(hsz_idx, pre_d);
                            self.time.insert(hsz_idx, pre_t);
                            index_offset += 1;
                        }
                    }
                }
            }
        }

        // Apply optional PKPab / PKPbc caustic split if explicitly requested!
        if self.subbranch != CoreSubbranch::None && self.dist.len() >= 2 {
            // Find the PKP B-caustic minimum-distance index among continuous samples.
            let mut min_d = f64::INFINITY;
            let mut b_idx = 0usize;
            for (i, &d) in self.dist.iter().enumerate() {
                if d < min_d {
                    min_d = d;
                    b_idx = i;
                }
            }
            match self.subbranch {
                CoreSubbranch::PkpAb => {
                    // Shallow outer-core retrograde branch (larger p, indices 0..=b_idx).
                    if b_idx == 0 {
                        self.ray_param.clear();
                        self.dist.clear();
                        self.time.clear();
                        self.max_distance = -1.0;
                        return Ok(());
                    }
                    self.ray_param.truncate(b_idx + 1);
                    self.dist.truncate(b_idx + 1);
                    self.time.truncate(b_idx + 1);
                    self.min_ray_param = *self.ray_param.last().unwrap();
                }
                CoreSubbranch::PkpBc => {
                    // Deep outer-core prograde branch (smaller p, indices b_idx..).
                    if b_idx + 1 >= self.dist.len() {
                        self.ray_param.clear();
                        self.dist.clear();
                        self.time.clear();
                        self.max_distance = -1.0;
                        return Ok(());
                    }
                    self.ray_param.drain(0..b_idx);
                    self.dist.drain(0..b_idx);
                    self.time.drain(0..b_idx);
                    self.max_ray_param = self.ray_param[0];
                }
                CoreSubbranch::None => {}
            }
        }

        self.min_distance = self.dist.iter().copied().fold(f64::INFINITY, f64::min);
        self.max_distance = self.dist.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        self.rebuild_dist_blocks();
        Ok(())
    }

    fn rebuild_dist_blocks(&mut self) {
        self.dist_blocks.clear();
        let n = self.dist.len();
        if n < 2 {
            return;
        }
        let is_non_body = self.is_kmps || !self.head_or_diffract_seq.is_empty();
        const BLOCK_LEN: usize = 24;
        let mut cur_start = 0usize;
        while cur_start < n - 1 {
            if !is_non_body && self.ray_param[cur_start] == self.ray_param[cur_start + 1] {
                cur_start += 1;
                continue;
            }
            let mut end = cur_start;
            let mut bmin = f64::INFINITY;
            let mut bmax = f64::NEG_INFINITY;
            while end < n - 1 && end - cur_start < BLOCK_LEN {
                if !is_non_body && self.ray_param[end] == self.ray_param[end + 1] {
                    break;
                }
                let d0 = self.dist[end];
                let d1 = self.dist[end + 1];
                bmin = bmin.min(d0.min(d1));
                bmax = bmax.max(d0.max(d1));
                end += 1;
            }
            if end > cur_start {
                self.dist_blocks.push((cur_start, end, bmin, bmax));
                cur_start = end;
            } else {
                cur_start += 1;
            }
        }
    }

    /// Returns index ranges `[start..end]` splitting `self.ray_param` across any LVZ shadow zones.
    pub fn shadow_zone_splits(&self) -> Vec<core::ops::Range<usize>> {
        let n = self.ray_param.len();
        if n == 0 {
            return Vec::new();
        }
        if !self.head_or_diffract_seq.is_empty() || self.is_kmps {
            return core::iter::once(0..n).collect();
        }
        let mut ranges = Vec::new();
        let mut start = 0usize;
        for i in 0..(n - 1) {
            if self.ray_param[i] == self.ray_param[i + 1] {
                if i + 1 > start {
                    ranges.push(start..(i + 1));
                }
                start = i + 1;
            }
        }
        if n > start {
            ranges.push(start..n);
        }
        ranges
    }

    /// Computes all arrivals of this phase at epicentral distance `degrees` (including minor-arc
    /// and major-arc multi-wraps).
    pub fn calc_time(
        &self,
        tau_model: &TauModel,
        degrees: f64,
        ray_param_tol: f64,
    ) -> Result<Vec<RawArrival>> {
        if !self.has_arrivals() || self.dist.len() < 2 {
            return Ok(Vec::new());
        }

        let mut temp_deg = degrees.abs();
        if temp_deg > 360.0 {
            temp_deg = math::rem_euclid(temp_deg, 360.0);
        }
        if temp_deg > 180.0 {
            temp_deg = 360.0 - temp_deg;
        }
        let rad = temp_deg.to_radians();

        let is_non_body = self.is_kmps || !self.head_or_diffract_seq.is_empty();
        let n = self.dist.len();
        let mut arrivals = Vec::with_capacity(4);

        // Enumerate distinct target radian distances <= max_distance across wraps.
        let mut wrap_n = 0usize;
        loop {
            let d_minor = TAU * (wrap_n as f64) + rad;
            let d_major = TAU * ((wrap_n + 1) as f64) - rad;
            if d_minor > self.max_distance && d_major > self.max_distance {
                break;
            }

            let mut targets = [0.0; 2];
            let mut n_targets = 0usize;
            if d_minor <= self.max_distance {
                targets[n_targets] = d_minor;
                n_targets += 1;
            }
            // Avoid duplicate target distances at exact 0 deg (`2*pi*n` vs `2*pi*(n+1)`) and 180 deg (`pi == 2*pi - pi`).
            if temp_deg > 0.0 && temp_deg < 180.0 && d_major <= self.max_distance {
                targets[n_targets] = d_major;
                n_targets += 1;
            }

            for &search_dist in &targets[..n_targets] {
                if search_dist < self.min_distance {
                    continue;
                }
                for &(blk_start, blk_end, bmin, bmax) in &self.dist_blocks {
                    if search_dist < bmin || search_dist > bmax {
                        continue;
                    }
                    for i in blk_start..blk_end {
                        let d0 = self.dist[i];
                        let d1 = self.dist[i + 1];
                        let strictly_inside = (d0 - search_dist) * (search_dist - d1) > 0.0;
                        let left_hit = d0 == search_dist;
                        let is_subbranch_end = i + 1 == n - 1
                            || (!is_non_body && self.ray_param[i + 1] == self.ray_param[i + 2]);
                        let right_hit = d1 == search_dist && is_subbranch_end;
                        if strictly_inside || left_hit || right_hit {
                            let arr = self.refine_arrival(
                                tau_model,
                                degrees,
                                i,
                                search_dist,
                                ray_param_tol,
                                MAX_REFINE_ITER,
                            )?;
                            arrivals.push(arr);
                        }
                    }
                }
            }

            wrap_n += 1;
            if wrap_n > 64 {
                break;
            }
        }

        Ok(arrivals)
    }

    fn refine_arrival(
        &self,
        tau_model: &TauModel,
        degrees: f64,
        ray_index: usize,
        dist_radian: f64,
        tolerance: f64,
        recursion_limit: usize,
    ) -> Result<RawArrival> {
        let left_p = self.ray_param[ray_index];
        let left_t = self.time[ray_index];
        let left_d = self.dist[ray_index];

        let right_p = self.ray_param[ray_index + 1];
        let right_t = self.time[ray_index + 1];
        let right_d = self.dist[ray_index + 1];

        if left_d == dist_radian {
            return self.make_raw_arrival(tau_model, degrees, left_t, left_d, left_p, ray_index);
        }
        if right_d == dist_radian {
            return self.make_raw_arrival(tau_model, degrees, right_t, right_d, right_p, ray_index);
        }

        if recursion_limit == 0
            || self.is_kmps
            || !self.head_or_diffract_seq.is_empty()
            || left_p == right_p
        {
            return self.linear_interp_arrival(
                tau_model,
                degrees,
                dist_radian,
                ray_index,
                (left_p, left_t, left_d),
                (right_p, right_t, right_d),
            );
        }

        let fa = dist_radian - left_d;
        let fb = dist_radian - right_d;
        if fa * fb > 0.0 {
            return self.linear_interp_arrival(
                tau_model,
                degrees,
                dist_radian,
                ray_index,
                (left_p, left_t, left_d),
                (right_p, right_t, right_d),
            );
        }

        // Exact 4-DOF Buland & Chapman (1983) Cubic Hermite tau(p) spline on [right_p, left_p]
        // matching both tau(p) and dtau/dp = -Delta(p) at the two endpoints:
        let mut a = left_p;
        let mut b = right_p;
        let mut f_a = fa;
        let mut f_b = fb;
        let lin_denom = left_d - right_d;
        let lin_p = if lin_denom == 0.0 {
            right_p
        } else {
            (dist_radian - right_d) * ((left_p - right_p) / lin_denom) + right_p
        };
        let lin_t = right_t + right_p * (dist_radian - right_d);
        let mut last_shot = (lin_p, lin_t, dist_radian);

        let h_p = left_p - right_p;
        if h_p > 0.0 {
            let tau0 = right_t - right_p * right_d;
            let tau1 = left_t - left_p * left_d;
            let c2 = 3.0 * (tau1 - tau0) + h_p * (2.0 * right_d + left_d);
            let c3 = 2.0 * (tau0 - tau1) - h_p * (right_d + left_d);
            let aq = 3.0 * c3;
            let bq = c2;
            let cq = h_p * (dist_radian - right_d);
            let disc = bq * bq - aq * cq;

            if disc >= 0.0 && bq != 0.0 {
                let sq = math::sqrt(disc).copysign(bq);
                let denom_q = bq + sq;
                if denom_q != 0.0 {
                    let u_h = -cq / denom_q;
                    if u_h > 0.0 && u_h < 1.0 {
                        let p_h = right_p + u_h * h_p;
                        let d_span = (left_d - right_d).abs();
                        let away_from_sqrt_tip =
                            ray_index > 0 && self.ray_param[ray_index - 1] != left_p;
                        let c3_lim = if away_from_sqrt_tip { 5e-2 } else { 2e-3 };
                        if tolerance >= 1e-3 && c3.abs() <= c3_lim && d_span > 1e-6 {
                            let tau_h = tau0 + u_h * (-h_p * right_d + u_h * (c2 + u_h * c3));
                            let theta_h = tau_h + p_h * dist_radian;
                            return self.make_raw_arrival(
                                tau_model,
                                degrees,
                                theta_h,
                                dist_radian,
                                p_h,
                                ray_index,
                            );
                        }

                        // Warm-start Dekker-Brent at p_h so cusp cells finish in 1-2 shots!
                        let (shot_t0, shot_d0) = self.shoot_ray_td(tau_model, p_h)?;
                        let f_h = dist_radian - shot_d0;
                        last_shot = (p_h, shot_t0, shot_d0);
                        if f_h.abs() <= 1e-9 || (b - a).abs() <= tolerance {
                            let theta_h = shot_t0 + p_h * f_h;
                            return self.make_raw_arrival(
                                tau_model,
                                degrees,
                                theta_h,
                                dist_radian,
                                p_h,
                                ray_index,
                            );
                        }
                        if (f_a > 0.0 && f_h < 0.0) || (f_a < 0.0 && f_h > 0.0) {
                            b = p_h;
                            f_b = f_h;
                        } else {
                            a = p_h;
                            f_a = f_h;
                        }
                    }
                }
            }
        }

        // Safeguarded Dekker-Brent root-finding on `residual(p) = dist_radian - shoot_dist(p)`
        // with zero heap allocations!
        let mut c = a;
        let mut fc = f_a;
        let mut d_step = b - a;
        let mut e_step = d_step;

        for _ in 0..recursion_limit {
            if (f_b > 0.0 && fc > 0.0) || (f_b < 0.0 && fc < 0.0) {
                c = a;
                fc = f_a;
                d_step = b - a;
                e_step = d_step;
            }
            if fc.abs() < f_b.abs() {
                a = b;
                b = c;
                c = a;
                f_a = f_b;
                f_b = fc;
                fc = f_a;
            }
            let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tolerance;
            let xm = 0.5 * (c - b);
            if xm.abs() <= tol1 || f_b == 0.0 {
                break;
            }
            if e_step.abs() >= tol1 && f_a.abs() > f_b.abs() {
                let s = f_b / f_a;
                let (mut p_num, mut q_den) = if a == c {
                    (2.0 * xm * s, 1.0 - s)
                } else {
                    let q = f_a / fc;
                    let r = f_b / fc;
                    (
                        s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0)),
                        (q - 1.0) * (r - 1.0) * (s - 1.0),
                    )
                };
                if p_num > 0.0 {
                    q_den = -q_den;
                }
                p_num = p_num.abs();
                let min1 = 3.0 * xm * q_den - (tol1 * q_den).abs();
                let min2 = (e_step * q_den).abs();
                if 2.0 * p_num < min1.min(min2) {
                    e_step = d_step;
                    d_step = p_num / q_den;
                } else {
                    d_step = xm;
                    e_step = d_step;
                }
            } else {
                d_step = xm;
                e_step = d_step;
            }
            a = b;
            f_a = f_b;
            if d_step.abs() > tol1 {
                b += d_step;
            } else {
                b += tol1.copysign(xm);
            }

            let (shot_t, shot_d) = self.shoot_ray_td(tau_model, b)?;
            last_shot = (b, shot_t, shot_d);
            f_b = dist_radian - shot_d;
        }

        // Buland & Chapman (1983) stationary theta travel-time evaluation:
        let theta_time = last_shot.1 + last_shot.0 * (dist_radian - last_shot.2);
        self.make_raw_arrival(tau_model, degrees, theta_time, dist_radian, b, ray_index)
    }

    /// Zero-allocation exact ray shooting returning `(time_s, dist_rad)` for `ray_param`.
    #[inline]
    pub fn shoot_ray_td(&self, tau_model: &TauModel, ray_param: f64) -> Result<(f64, f64)> {
        let s_mod = &tau_model.s_mod;
        let mut tot_time = 0.0;
        let mut tot_dist = 0.0;

        for &(j, p_top, p_bot, pm, s_top, s_bot, sm) in &self.active_branch_mult {
            if pm != 0 {
                let br = &tau_model.tau_branches[0][j];
                if ray_param <= br.max_ray_param {
                    let (t, d) = br.eval_single_p(s_mod, p_top, p_bot, ray_param, true)?;
                    tot_time += (pm as f64) * t;
                    tot_dist += (pm as f64) * d;
                }
            }
            if sm != 0 {
                let br = &tau_model.tau_branches[1][j];
                if ray_param <= br.max_ray_param {
                    let (t, d) = br.eval_single_p(s_mod, s_top, s_bot, ray_param, true)?;
                    tot_time += (sm as f64) * t;
                    tot_dist += (sm as f64) * d;
                }
            }
        }
        Ok((tot_time, tot_dist))
    }

    fn linear_interp_arrival(
        &self,
        tau_model: &TauModel,
        degrees: f64,
        search_dist: f64,
        ray_index: usize,
        left: (f64, f64, f64),
        right: (f64, f64, f64),
    ) -> Result<RawArrival> {
        let (left_p, left_t, left_d) = left;
        let (right_p, right_t, right_d) = right;
        if left_d == search_dist {
            return self.make_raw_arrival(tau_model, degrees, left_t, left_d, left_p, ray_index);
        }
        if right_d == search_dist {
            return self.make_raw_arrival(tau_model, degrees, right_t, right_d, right_p, ray_index);
        }
        let denom = left_d - right_d;
        let dp_dx = if denom == 0.0 {
            0.0
        } else {
            (left_p - right_p) / denom
        };
        let ray_param = (search_dist - right_d) * dp_dx + right_p;
        let left_theta = left_t + left_p * (search_dist - left_d);
        let right_theta = right_t + right_p * (search_dist - right_d);
        let arrival_time = if dp_dx > 0.0 {
            left_theta.max(right_theta)
        } else {
            left_theta.min(right_theta)
        };
        self.make_raw_arrival(
            tau_model,
            degrees,
            arrival_time,
            search_dist,
            ray_param,
            ray_index,
        )
    }

    fn make_raw_arrival(
        &self,
        tau_model: &TauModel,
        degrees: f64,
        time_s: f64,
        purist_dist_rad: f64,
        ray_param_s_per_rad: f64,
        ray_param_index: usize,
    ) -> Result<RawArrival> {
        let takeoff_angle_deg = self.calc_takeoff_angle(tau_model, ray_param_s_per_rad)?;
        let incident_angle_deg = self.calc_incident_angle(tau_model, ray_param_s_per_rad)?;
        Ok(RawArrival {
            name: self.name.clone(),
            purist_name: self.purist_name.clone(),
            distance_deg: degrees,
            purist_dist_rad,
            time_s,
            ray_param_s_per_rad,
            ray_param_index,
            source_depth_km: self.source_depth,
            receiver_depth_km: self.receiver_depth,
            takeoff_angle_deg,
            incident_angle_deg,
        })
    }

    /// Computes source takeoff angle in degrees from downward vertical (`[0, 180]`).
    pub fn calc_takeoff_angle(&self, tau_model: &TauModel, ray_param: f64) -> Result<f64> {
        if self.is_kmps {
            return Ok(0.0);
        }
        let prop = if self.wave_type.first().copied().unwrap_or(true) {
            MaterialProperty::PVelocity
        } else {
            MaterialProperty::SVelocity
        };
        let v_mod = &tau_model.s_mod.v_mod;
        let down = self.down_going.first().copied().unwrap_or(true);
        let vel = if down {
            v_mod.evaluate_below(self.source_depth, prop)?
        } else {
            v_mod.evaluate_above(self.source_depth, prop)?
        };
        let r_src = tau_model.radius_of_planet - self.source_depth;
        let arg = if r_src == 0.0 {
            0.0
        } else {
            (vel * ray_param / r_src).clamp(-1.0, 1.0)
        };
        let mut angle = math::asin(arg).to_degrees();
        if !down {
            angle = 180.0 - angle;
        }
        Ok(angle)
    }

    /// Computes receiver incident angle in degrees from vertical (`[0, 180]`).
    pub fn calc_incident_angle(&self, tau_model: &TauModel, ray_param: f64) -> Result<f64> {
        if self.is_kmps {
            return Ok(0.0);
        }
        let prop = if self.wave_type.last().copied().unwrap_or(true) {
            MaterialProperty::PVelocity
        } else {
            MaterialProperty::SVelocity
        };
        let v_mod = &tau_model.s_mod.v_mod;
        let down_end = self.down_going.last().copied().unwrap_or(false);
        let vel = if down_end {
            v_mod.evaluate_above(self.receiver_depth, prop)?
        } else {
            v_mod.evaluate_below(self.receiver_depth, prop)?
        };
        let r_rec = tau_model.radius_of_planet - self.receiver_depth;
        let arg = if r_rec == 0.0 {
            0.0
        } else {
            (vel * ray_param / r_rec).clamp(-1.0, 1.0)
        };
        let mut angle = math::asin(arg).to_degrees();
        if down_end {
            angle = 180.0 - angle;
        }
        Ok(angle)
    }

    /// Evaluates discontinuity & turning pierce points `(p, time, dist_rad, depth_km)` for `arrival`.
    pub fn calc_pierce_for_arrival(
        &self,
        tau_model: &TauModel,
        arrival: &RawArrival,
    ) -> Result<Vec<TimeDist>> {
        let mut ray_num = 0usize;
        for (i, &rp) in tau_model
            .ray_params
            .iter()
            .enumerate()
            .take(tau_model.ray_params.len().saturating_sub(1))
        {
            if rp >= arrival.ray_param_s_per_rad {
                ray_num = i;
            } else {
                break;
            }
        }

        let ray_param_a = self.ray_param[arrival.ray_param_index];
        let ray_param_b = self.ray_param[arrival.ray_param_index + 1];
        let dist_a = self.dist[arrival.ray_param_index];
        let dist_b = self.dist[arrival.ray_param_index + 1];
        let dist_ratio = if dist_b == dist_a {
            0.0
        } else {
            (arrival.purist_dist_rad - dist_a) / (dist_b - dist_a)
        };
        let dist_ray_param = dist_ratio * (ray_param_b - ray_param_a) + ray_param_a;

        let mut pierce =
            Vec::with_capacity(self.branch_seq.len() + self.head_or_diffract_seq.len() + 2);
        pierce.push(TimeDist {
            p: dist_ray_param,
            time: 0.0,
            dist: 0.0,
            depth: tau_model.source_depth,
        });

        let mut branch_dist = 0.0;
        let mut branch_time = 0.0;

        for (i, ((&branch_num, &is_p_wave), &is_down_going)) in self
            .branch_seq
            .iter()
            .zip(self.wave_type.iter())
            .zip(self.down_going.iter())
            .enumerate()
        {
            let tau_branch = tau_model.get_tau_branch(branch_num, is_p_wave);
            let turn_depth = if dist_ray_param > tau_branch.max_ray_param {
                tau_branch.top_depth
            } else if dist_ray_param <= tau_branch.min_ray_param {
                tau_branch.bot_depth
            } else {
                let mid_d = 0.5 * (tau_branch.top_depth + tau_branch.bot_depth);
                let use_p = is_p_wave || tau_model.s_mod.depth_in_fluid(mid_d);
                tau_model.s_mod.find_depth_from_depths(
                    dist_ray_param,
                    tau_branch.top_depth,
                    tau_branch.bot_depth,
                    use_p,
                )?
            };

            let (da, ta, db, tb) = if !self.head_or_diffract_seq.is_empty() {
                let d0 = tau_branch.dist[ray_num];
                let t0 = tau_branch.time[ray_num];
                (d0, t0, d0, t0)
            } else {
                (
                    tau_branch.dist[ray_num],
                    tau_branch.time[ray_num],
                    tau_branch.dist[ray_num + 1],
                    tau_branch.time[ray_num + 1],
                )
            };

            branch_dist += dist_ratio * (db - da) + da;
            let prev_branch_time = branch_time;
            branch_time += dist_ratio * (tb - ta) + ta;
            let branch_depth = if is_down_going {
                tau_branch.bot_depth.min(turn_depth)
            } else {
                tau_branch.top_depth.min(turn_depth)
            };

            if (prev_branch_time - branch_time).abs() > 1e-10 {
                pierce.push(TimeDist {
                    p: dist_ray_param,
                    time: branch_time,
                    dist: branch_dist,
                    depth: branch_depth,
                });
            }

            if self.head_or_diffract_seq.contains(&i) {
                let refract_dist = (arrival.purist_dist_rad - self.dist[0])
                    / (self.head_or_diffract_seq.len() as f64);
                let refract_time = refract_dist * arrival.ray_param_s_per_rad;
                pierce.push(TimeDist {
                    p: dist_ray_param,
                    time: branch_time + refract_time,
                    dist: branch_dist + refract_dist,
                    depth: branch_depth,
                });
                branch_dist += refract_dist;
                branch_time += refract_time;
            }
        }

        if self.is_kmps {
            pierce.push(TimeDist {
                p: dist_ray_param,
                time: arrival.time_s,
                dist: arrival.purist_dist_rad,
                depth: 0.0,
            });
        }

        Ok(pierce)
    }

    /// Evaluates the full spherical ray-path trajectory `(p, time, dist_rad, depth_km)` for `arrival`.
    pub fn calc_path_for_arrival(
        &self,
        tau_model: &TauModel,
        arrival: &RawArrival,
    ) -> Result<Vec<TimeDist>> {
        let mut path_list = Vec::with_capacity(256);
        path_list.push(TimeDist {
            p: arrival.ray_param_s_per_rad,
            time: 0.0,
            dist: 0.0,
            depth: tau_model.source_depth,
        });

        for (i, ((&branch_num, &is_p_wave), &is_down_going)) in self
            .branch_seq
            .iter()
            .zip(self.wave_type.iter())
            .zip(self.down_going.iter())
            .enumerate()
        {
            let br = tau_model.get_tau_branch(branch_num, is_p_wave);
            let start_len = path_list.len();
            br.path(
                arrival.ray_param_s_per_rad,
                is_down_going,
                &tau_model.s_mod,
                &mut path_list,
            )?;
            for td in &path_list[start_len..] {
                if td.dist < 0.0 {
                    return Err(EarthSnailError::Numerical(
                        "ray path backtracked with negative distance increment".into(),
                    ));
                }
            }

            if self.head_or_diffract_seq.contains(&i) {
                let dist_diff = (arrival.purist_dist_rad - self.dist[0])
                    / (self.head_or_diffract_seq.len() as f64);
                let branch_depth = if self.canonical_name.contains("Pdiff")
                    || self.canonical_name.contains("Sdiff")
                {
                    tau_model.cmb_depth
                } else if self.canonical_name.contains("Pn") || self.canonical_name.contains("Sn") {
                    tau_model.moho_depth
                } else if self.canonical_name.contains("Kdiff") {
                    tau_model.iocb_depth
                } else {
                    return Err(EarthSnailError::PhaseParse(
                        "unexpected head_or_diffract_seq phase".into(),
                    ));
                };
                path_list.push(TimeDist {
                    p: arrival.ray_param_s_per_rad,
                    time: dist_diff * arrival.ray_param_s_per_rad,
                    dist: dist_diff,
                    depth: branch_depth,
                });
            }
        }

        if self.is_kmps {
            path_list.push(TimeDist {
                p: arrival.ray_param_s_per_rad,
                time: arrival.purist_dist_rad * arrival.ray_param_s_per_rad,
                dist: arrival.purist_dist_rad,
                depth: 0.0,
            });
        }

        // Convert incremental `(dt, d_dist)` into cumulative `(time, dist_rad)`.
        let mut cum_t = 0.0;
        let mut cum_d = 0.0;
        for pt in &mut path_list {
            cum_t += pt.time;
            cum_d += pt.dist;
            pt.time = cum_t;
            pt.dist = cum_d;
        }
        Ok(path_list)
    }
}

fn check_segment_consistency(prev: &PhaseSegment, curr: &PhaseSegment) -> Result<()> {
    if curr.is_down_going {
        if prev.end_branch > curr.start_branch {
            return Err(EarthSnailError::PhaseParse(
                "segment is downgoing, but previous segment ended deeper".into(),
            ));
        }
        if matches!(
            prev.end_action,
            EndAction::ReflectTopside | EndAction::Turn | EndAction::TransUp
        ) {
            return Err(EarthSnailError::PhaseParse(
                "segment is downgoing after an upward reflection/turn/transmission".into(),
            ));
        }
        if prev.end_branch == curr.start_branch
            && !prev.is_down_going
            && prev.end_action != EndAction::ReflectUnderside
        {
            return Err(EarthSnailError::PhaseParse(
                "segment is downgoing without an underside reflection".into(),
            ));
        }
    } else {
        if prev.end_branch < curr.start_branch {
            return Err(EarthSnailError::PhaseParse(
                "segment is upgoing, but previous segment ended shallower".into(),
            ));
        }
        if matches!(
            prev.end_action,
            EndAction::ReflectUnderside | EndAction::TransDown
        ) {
            return Err(EarthSnailError::PhaseParse(
                "segment is upgoing after a downward reflection/transmission".into(),
            ));
        }
        if prev.end_branch == curr.start_branch
            && prev.is_down_going
            && !matches!(
                prev.end_action,
                EndAction::Turn | EndAction::Diffract | EndAction::ReflectTopside
            )
        {
            return Err(EarthSnailError::PhaseParse(
                "segment is upgoing without turning, diffracting, or topside reflection".into(),
            ));
        }
    }
    Ok(())
}

fn closest_branch_to_depth(tau_model: &TauModel, depth_string: &str) -> Result<usize> {
    match depth_string {
        "m" => return Ok(tau_model.moho_branch),
        "c" => return Ok(tau_model.cmb_branch),
        "i" => return Ok(tau_model.iocb_branch),
        _ => {}
    }
    let target = depth_string.parse::<f64>().map_err(|_| {
        EarthSnailError::PhaseParse(format!("invalid discontinuity depth '{depth_string}'"))
    })?;
    let mut best_branch = None;
    let mut best_diff = 1e300_f64;
    for (i, tb) in tau_model.tau_branches[0].iter().enumerate() {
        let diff = (target - tb.top_depth).abs();
        if diff < best_diff && !tau_model.no_discon_depths.contains(&tb.top_depth) {
            best_branch = Some(i);
            best_diff = diff;
        }
    }
    best_branch.ok_or_else(|| {
        EarthSnailError::PhaseParse(format!(
            "no valid discontinuity branch near depth {depth_string} km"
        ))
    })
}

fn create_purist_name(legs: &[String], tau_model: &TauModel) -> Result<String> {
    let mut out = String::with_capacity(16);
    for leg in &legs[..legs.len() - 1] {
        let b0 = leg.as_bytes()[0];
        if b0 == b'v' || b0 == b'^' {
            let discon_branch = closest_branch_to_depth(tau_model, &leg[1..])?;
            let leg_depth = branch_top_depth_safe(tau_model, discon_branch);
            out.push(b0 as char);
            let rounded = math::round(leg_depth) as i64;
            out.push_str(&format!("{rounded}"));
        } else if leg.parse::<f64>().is_ok() {
            let discon_branch = closest_branch_to_depth(tau_model, leg)?;
            let leg_depth = branch_top_depth_safe(tau_model, discon_branch);
            out.push_str(&format_py_float(leg_depth));
        } else {
            out.push_str(leg);
        }
    }
    Ok(out)
}

fn branch_top_depth_safe(tau_model: &TauModel, branch_idx: usize) -> f64 {
    if branch_idx < tau_model.num_branches() {
        tau_model.tau_branches[0][branch_idx].top_depth
    } else {
        tau_model.radius_of_planet
    }
}

fn format_py_float(val: f64) -> String {
    if math::fract(val) == 0.0 && val.abs() < 1e15 {
        format!("{val:.1}")
    } else {
        format!("{val}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leg_puller_and_phase_compiler_cover_all_families() {
        assert_eq!(
            leg_puller("PKiKP").unwrap(),
            vec!["P", "K", "i", "K", "P", "END"]
        );
        assert_eq!(leg_puller("p^410S").unwrap(), vec!["p", "^410", "S", "END"]);
        assert_eq!(leg_puller("Pv410p").unwrap(), vec!["P", "v410", "p", "END"]);
        assert_eq!(leg_puller("P660s").unwrap(), vec!["P", "660", "s", "END"]);
        assert_eq!(leg_puller("3.5kmps").unwrap(), vec!["3.5kmps", "END"]);
        assert!(leg_puller("Ps").is_err());
        assert!(leg_puller("").is_err());
        assert!(leg_puller("P^").is_err());

        let iasp = crate::tau_model::test_iasp91_model()
            .with_source_depth(35.0)
            .unwrap();
        for ph in [
            "p", "s", "Pg", "Sg", "Pb", "Sb", "Pn", "Sn", "P", "S", "pP", "sP", "pS", "sS", "PP",
            "SS", "PPP", "SSS", "PS", "SP", "PcP", "ScS", "PcS", "ScP", "PKiKP", "SKiKS", "PKP",
            "PKPab", "PKPbc", "PKPdf", "PKIKP", "SKS", "SKSac", "SKSdf", "SKKS", "PKKP", "PKJKP",
            "Pdiff", "Sdiff", "P410s", "P660s", "S410p", "S660p", "Pms", "Pv410p", "3.5kmps",
            "4.0kmps",
        ] {
            let compiled = SeismicPhase::new(ph, &iasp)
                .unwrap_or_else(|e| panic!("failed to compile {ph}: {e}"));
            assert!(!compiled.purist_name.is_empty());
        }
    }
}
