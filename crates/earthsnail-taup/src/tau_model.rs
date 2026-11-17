//! Bulland & Chapman (1983) / Crotwell et al. (1999) spherical power-law slowness sampling
//! and analytical $\tau(p)$ branch integrator.
//!
//! Parameterizes each spherical slowness shell by Bullen's power law $\eta(r) = A r^B$
//! (equivalently $v(r) = v_0 (r / r_0)^\alpha$ with $B = 1 - \alpha$), yielding closed-form
//! analytical integrals for incremental travel time $\delta T(p)$, angular distance $\delta\Delta(p)$,
//! and delay time $\delta\tau(p) = \delta T(p) - p\,\delta\Delta(p)$ across crustal, mantle,
//! low-velocity shadow zones (LVZ), fluid outer-core, and solid inner-core shells.

use crate::models::{MaterialProperty, VelocityLayer, VelocityModel};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::{FRAC_PI_2, PI};
use earthsnail_core::{math, EarthSnailError, Result};

/// Default numerical tolerance for slowness comparisons (`1e-16`).
pub const DEFAULT_SLOWNESS_TOL: f64 = 1e-16;

/// Single spherical slowness layer `[top_depth, bot_depth]` with spherical ray-parameter
/// slownesses `top_p` and `bot_p` in seconds per radian ($\eta = r / v$).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlownessLayer {
    /// Spherical slowness $\eta_{\text{top}} = (R_E - z_{\text{top}}) / v_{\text{top}}$ (`s/rad`).
    pub top_p: f64,
    /// Depth at top of layer in kilometers.
    pub top_depth: f64,
    /// Spherical slowness $\eta_{\text{bot}} = (R_E - z_{\text{bot}}) / v_{\text{bot}}$ (`s/rad`).
    pub bot_p: f64,
    /// Depth at bottom of layer in kilometers.
    pub bot_depth: f64,
}

impl SlownessLayer {
    /// Returns `true` if this layer is a zero-thickness discontinuity step (`top_depth == bot_depth`).
    #[inline]
    pub fn is_zero_thickness(&self) -> bool {
        self.top_depth == self.bot_depth
    }
}

/// Incremental ray parameter, travel time, angular distance, and depth sample.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TimeDist {
    /// Ray parameter $p$ in `s/rad`.
    pub p: f64,
    /// Travel-time increment or cumulative time in seconds.
    pub time: f64,
    /// Angular distance increment or cumulative distance in radians.
    pub dist: f64,
    /// Depth in kilometers.
    pub depth: f64,
}

/// Depth interval `[top_depth, bot_depth]` for fluid zones or high-slowness (low-velocity) zones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthRange {
    /// Upper boundary depth in kilometers.
    pub top_depth: f64,
    /// Lower boundary depth in kilometers.
    pub bot_depth: f64,
    /// Limiting ray parameter $p$ (`s/rad`) entering a high-slowness shadow zone (`-1.0` for fluid ranges).
    pub ray_param: f64,
}

/// Critical depth point (first-order velocity discontinuity or slowness-gradient reversal)
/// partitioning the planet into monotonic `TauBranch` intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CriticalDepth {
    /// Depth in kilometers (stored as bit-exact `u64` via `CriticalDepth` float field below; we store `f64` directly).
    pub vel_layer_num: usize,
    /// Corresponding P-wave slowness layer index.
    pub p_layer_num: usize,
    /// Corresponding S-wave slowness layer index.
    pub s_layer_num: usize,
    depth_bits: u64,
}

impl CriticalDepth {
    /// Constructs a new `CriticalDepth`.
    #[inline]
    pub fn new(depth: f64, vel_layer_num: usize, p_layer_num: usize, s_layer_num: usize) -> Self {
        Self {
            vel_layer_num,
            p_layer_num,
            s_layer_num,
            depth_bits: if depth == 0.0 {
                0.0_f64.to_bits()
            } else {
                depth.to_bits()
            },
        }
    }

    /// Critical depth in kilometers.
    #[inline]
    pub fn depth(&self) -> f64 {
        f64::from_bits(self.depth_bits)
    }
}

/// Discretization tolerances controlling `SlownessModel` sampling from a `VelocityModel`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlownessModelConfig {
    /// Minimum ray-parameter spacing when subdividing wide range jumps (`s/rad`).
    pub min_delta_p: f64,
    /// Maximum ray-parameter span per slowness layer (`s/rad`).
    pub max_delta_p: f64,
    /// Maximum radial thickness per slowness layer (`km`).
    pub max_depth_interval: f64,
    /// Maximum epicentral range jump between adjacent turning rays (`rad`).
    pub max_range_interval_rad: f64,
    /// Maximum linear travel-time interpolation error tolerance (`s`).
    pub max_interp_error: f64,
    /// Allow S-wave (`J`) propagation in the solid inner core.
    pub allow_inner_core_s: bool,
    /// Slowness numerical equality chatter tolerance.
    pub slowness_tolerance: f64,
}

impl Default for SlownessModelConfig {
    fn default() -> Self {
        Self {
            min_delta_p: 0.1,
            max_delta_p: 11.0,
            max_depth_interval: 115.0,
            max_range_interval_rad: 2.5 * PI / 180.0,
            max_interp_error: 0.05,
            allow_inner_core_s: true,
            slowness_tolerance: DEFAULT_SLOWNESS_TOL,
        }
    }
}

/// Cancellation-free evaluation of $\ln(\eta_{\text{top}} / \eta_{\text{bot}})$.
#[inline]
fn ln_ratio_precise(eta_top: f64, eta_bot: f64) -> f64 {
    let diff = eta_top - eta_bot;
    let x = diff / eta_bot;
    if x.abs() < 1e-3 {
        // ln(1 + x) = x * (1 - x/2 + x^2/3 - x^3/4 + x^4/5 - x^5/6 + x^6/7 - x^7/8)
        let poly = 1.0
            + x * (-0.5
                + x * (1.0 / 3.0
                    + x * (-0.25 + x * (0.2 + x * (-1.0 / 6.0 + x * (1.0 / 7.0 - x * 0.125))))));
        x * poly
    } else {
        math::ln(eta_top / eta_bot)
    }
}

/// Evaluates the closed-form Bullen power-law ($\eta(r) = A r^B$) one-way travel-time increment
/// $\delta T(p)$ (s) and angular distance increment $\delta\Delta(p)$ (rad) across `layer`.
pub fn bullen_radial_slowness(
    layer: SlownessLayer,
    p: f64,
    radius_of_planet: f64,
    check: bool,
) -> Result<(f64, f64)> {
    if layer.top_depth == layer.bot_depth {
        return Ok((0.0, 0.0));
    }
    let r_top = radius_of_planet - layer.top_depth;
    let r_bot = radius_of_planet - layer.bot_depth;
    if r_top <= 0.0 || r_bot <= 0.0 || r_top <= r_bot {
        if check {
            return Err(EarthSnailError::ModelError(format!(
                "invalid Bullen shell radii r_top={r_top}, r_bot={r_bot}"
            )));
        }
        return Ok((f64::NAN, f64::NAN));
    }

    let eta_top = layer.top_p;
    let eta_bot = layer.bot_p;
    if check && (p < 0.0 || p > eta_top.min(eta_bot)) {
        return Err(EarthSnailError::ModelError(format!(
            "ray parameter p={p} exceeds layer slowness min({}, {})",
            eta_top, eta_bot
        )));
    }
    if p > eta_top.min(eta_bot) {
        return Ok((f64::NAN, f64::NAN));
    }

    let lr = ln_ratio_precise(r_top, r_bot);

    // Constant-slowness layer (B == 0, alpha == 1, v(r) proportional to r).
    if eta_top == eta_bot {
        if p >= eta_top {
            return Ok((0.0, 0.0));
        }
        let q0 = math::sqrt((eta_top - p) * (eta_top + p));
        let dt = (eta_top * eta_top / q0) * lr;
        let dd = (p / q0) * lr;
        return Ok((dt, dd));
    }

    // Cancellation-free vertical slowness factors q = sqrt(eta^2 - p^2) = sqrt((eta - p)(eta + p)).
    let q_top = math::sqrt((eta_top - p).max(0.0) * (eta_top + p));
    let q_bot = math::sqrt((eta_bot - p).max(0.0) * (eta_bot + p));
    let q_sum = q_top + q_bot;
    if q_sum == 0.0 {
        return Ok((0.0, 0.0));
    }

    let d_eta = eta_top - eta_bot;
    let ln_eta = ln_ratio_precise(eta_top, eta_bot);
    // B = ln_eta / lr
    let dq = d_eta * (eta_top + eta_bot) / q_sum;
    let dt = (dq / ln_eta) * lr;

    let dd = if p == 0.0 {
        0.0
    } else {
        let d_theta = math::atan2(p * dq, p * p + q_top * q_bot);
        (d_theta / ln_eta) * lr
    };

    if check && (!dt.is_finite() || !dd.is_finite() || dt < 0.0 || dd < 0.0) {
        return Err(EarthSnailError::ModelError(format!(
            "Bullen radial slowness produced invalid dt={dt}, dd={dd}"
        )));
    }
    Ok((dt, dd))
}

/// Finds the exact turning depth (in km) where $\eta(r) = p$ inside `layer` under Bullen's
/// power law $\eta(r) = A r^B$ (overflow-free logarithmic formulation, with linear fallback at $r = 0$).
pub fn bullen_depth_for(
    layer: SlownessLayer,
    ray_param: f64,
    radius_of_planet: f64,
) -> Result<f64> {
    if (layer.top_p - ray_param) * (ray_param - layer.bot_p) < 0.0 {
        return Err(EarthSnailError::ModelError(format!(
            "ray_param {ray_param} not contained in slowness layer [{}, {}]",
            layer.top_p, layer.bot_p
        )));
    }
    if layer.top_depth == layer.bot_depth {
        return Ok(layer.bot_depth);
    }
    if layer.top_p == ray_param {
        return Ok(layer.top_depth);
    }
    if layer.bot_p == ray_param {
        return Ok(layer.bot_depth);
    }

    if layer.bot_p != 0.0 && layer.bot_depth != radius_of_planet {
        let r_top = radius_of_planet - layer.top_depth;
        let r_bot = radius_of_planet - layer.bot_depth;
        let ln_p_tot = ln_ratio_precise(layer.top_p, layer.bot_p);
        if ln_p_tot != 0.0 && r_top > 0.0 && r_bot > 0.0 {
            // Overflow-free identity: ln(r_turn / r_top) = -(ln(top_p / p) / ln(top_p / bot_p)) * ln(r_top / r_bot)
            let theta = ln_ratio_precise(layer.top_p, ray_param) / ln_p_tot;
            let lr = ln_ratio_precise(r_top, r_bot);
            let r_turn = r_top * math::exp(-theta * lr);
            let mut depth = radius_of_planet - r_turn;
            if depth < layer.top_depth && layer.top_depth - depth < 1e-6 {
                depth = layer.top_depth;
            }
            if depth > layer.bot_depth && depth - layer.bot_depth < 1e-6 {
                depth = layer.bot_depth;
            }
            if depth.is_finite() && depth >= layer.top_depth && depth <= layer.bot_depth {
                return Ok(depth);
            }
        }
    }

    // Center-of-planet (bot_p == 0 or bot_depth == R_E) or degenerate constant-p fallback:
    if layer.top_p != layer.bot_p {
        let depth = layer.bot_depth
            + (ray_param - layer.bot_p) * (layer.top_depth - layer.bot_depth)
                / (layer.top_p - layer.bot_p);
        return Ok(depth.clamp(layer.top_depth, layer.bot_depth));
    }

    Ok(layer.bot_depth)
}

/// Evaluates spherical slowness $\eta(z)$ at `depth` inside `layer` under Bullen's power law.
pub fn evaluate_at_bullen(layer: SlownessLayer, depth: f64, radius_of_planet: f64) -> Result<f64> {
    if depth < layer.top_depth || depth > layer.bot_depth {
        return Err(EarthSnailError::ModelError(format!(
            "depth {depth} outside slowness layer [{}, {}]",
            layer.top_depth, layer.bot_depth
        )));
    }
    if depth == layer.top_depth {
        return Ok(layer.top_p);
    }
    if depth == layer.bot_depth {
        return Ok(layer.bot_p);
    }
    if layer.top_p == layer.bot_p {
        return Ok(layer.top_p);
    }

    let r_top = radius_of_planet - layer.top_depth;
    let r_bot = radius_of_planet - layer.bot_depth;
    let r = radius_of_planet - depth;
    if layer.bot_p > 0.0 && layer.top_p > 0.0 && r_bot > 0.0 && r > 0.0 && r_top > r_bot {
        // Overflow-free: eta(r) = top_p * exp(t * ln(bot_p / top_p)), where t = ln(r_top / r) / ln(r_top / r_bot)
        let lr_tot = ln_ratio_precise(r_top, r_bot);
        if lr_tot != 0.0 {
            let t = ln_ratio_precise(r_top, r) / lr_tot;
            let ans = layer.top_p * math::exp(-t * ln_ratio_precise(layer.top_p, layer.bot_p));
            if ans.is_finite() && ans >= 0.0 {
                return Ok(ans);
            }
        }
    }

    // Linear fallback for center-of-planet shell (r_bot == 0 / bot_p == 0) or ultra-thin layers.
    let linear = (layer.bot_p - layer.top_p) / (layer.bot_depth - layer.top_depth)
        * (depth - layer.top_depth)
        + layer.top_p;
    if linear.is_finite() && linear >= 0.0 {
        Ok(linear)
    } else {
        Err(EarthSnailError::ModelError(
            "calculated Bullen slowness is NaN or negative".into(),
        ))
    }
}

fn create_from_vlayer(
    v_layer: &VelocityLayer,
    is_p_wave: bool,
    radius_of_planet: f64,
    is_last_layer: bool,
) -> SlownessLayer {
    let prop = if is_p_wave {
        MaterialProperty::PVelocity
    } else {
        MaterialProperty::SVelocity
    };
    let top_vel = v_layer.evaluate_at_top(prop);
    let bot_vel = v_layer.evaluate_at_bottom(prop);
    let top_p = if top_vel == 0.0 {
        f64::INFINITY
    } else {
        (radius_of_planet - v_layer.top_depth) / top_vel
    };
    let mut bot_depth_eff = v_layer.bot_depth;
    if is_last_layer && bot_depth_eff == radius_of_planet && bot_vel == 0.0 {
        bot_depth_eff = 1.0;
    }
    let bot_p = if bot_vel == 0.0 {
        if radius_of_planet == bot_depth_eff {
            f64::NAN
        } else {
            f64::INFINITY
        }
    } else {
        (radius_of_planet - bot_depth_eff) / bot_vel
    };
    SlownessLayer {
        top_p,
        top_depth: v_layer.top_depth,
        bot_p,
        bot_depth: v_layer.bot_depth,
    }
}

#[derive(Clone)]
struct SplitLayerInfo {
    s_mod: SlownessModel,
    needed_split: bool,
    moved_sample: bool,
    ray_param: f64,
}

/// Coupled P-wave and S-wave spherical slowness discretization of a `VelocityModel`.
#[derive(Debug, Clone)]
pub struct SlownessModel {
    /// Underlying elastic velocity model.
    pub v_mod: VelocityModel,
    /// Planetary outer radius in kilometers.
    pub radius_of_planet: f64,
    /// Sampling tolerance parameters.
    pub config: SlownessModelConfig,
    /// Critical depths defining `TauBranch` boundaries.
    pub critical_depths: Vec<CriticalDepth>,
    /// P-wave high-slowness (low-velocity) shadow zones.
    pub high_slowness_layer_depths_p: Vec<DepthRange>,
    /// S-wave high-slowness (low-velocity) shadow zones.
    pub high_slowness_layer_depths_s: Vec<DepthRange>,
    /// Zero-$v_S$ fluid zones (e.g. Earth's outer core).
    pub fluid_layer_depths: Vec<DepthRange>,
    /// P-wave slowness layers from surface to center.
    pub p_layers: Vec<SlownessLayer>,
    /// S-wave slowness layers from surface to center (sharing `p_layers` inside fluid zones).
    pub s_layers: Vec<SlownessLayer>,
}

impl SlownessModel {
    /// Discretizes `v_mod` into coupled P and S spherical slowness shells using default TauP tolerances.
    pub fn from_velocity_model(v_mod: VelocityModel) -> Result<Self> {
        Self::with_config(v_mod, SlownessModelConfig::default())
    }

    /// Discretizes `v_mod` into coupled P and S spherical slowness shells using `config`.
    pub fn with_config(v_mod: VelocityModel, config: SlownessModelConfig) -> Result<Self> {
        v_mod.validate()?;
        if v_mod.layers.is_empty() {
            return Err(EarthSnailError::ModelError(
                "VelocityModel has zero layers".into(),
            ));
        }
        if v_mod.layers[0].top_s_velocity == 0.0 {
            return Err(EarthSnailError::ModelError(
                "surface fluid layers (zero S velocity at depth 0) are unsupported in spherical TauModel"
                    .into(),
            ));
        }
        let radius_of_planet = v_mod.radius_of_planet;
        let mut sm = Self {
            v_mod,
            radius_of_planet,
            config,
            critical_depths: Vec::new(),
            high_slowness_layer_depths_p: Vec::new(),
            high_slowness_layer_depths_s: Vec::new(),
            fluid_layer_depths: Vec::new(),
            p_layers: Vec::new(),
            s_layers: Vec::new(),
        };
        sm.create_sample()?;
        Ok(sm)
    }

    fn create_sample(&mut self) -> Result<()> {
        self.find_critical_points()?;
        self.coarse_sample()?;
        self.ray_param_inc_check()?;
        self.depth_inc_check()?;
        self.distance_check()?;
        self.fix_critical_points()?;
        self.validate()
    }

    fn find_critical_points(&mut self) -> Result<()> {
        let mut high_slowness_zone_p = DepthRange {
            top_depth: 0.0,
            bot_depth: 0.0,
            ray_param: -1.0,
        };
        let mut high_slowness_zone_s = DepthRange {
            top_depth: 0.0,
            bot_depth: 0.0,
            ray_param: -1.0,
        };
        let mut fluid_zone = DepthRange {
            top_depth: 0.0,
            bot_depth: 0.0,
            ray_param: -1.0,
        };
        let mut in_fluid_zone = false;
        let mut below_outer_core = false;
        let mut in_hsz_p = false;
        let mut in_hsz_s = false;
        let mut min_p_so_far = 1.1e300_f64;
        let mut min_s_so_far = 1.1e300_f64;

        self.critical_depths.clear();
        self.high_slowness_layer_depths_p.clear();
        self.high_slowness_layer_depths_s.clear();
        self.fluid_layer_depths.clear();

        let first_vl = self.v_mod.layers[0];
        let mut curr_v_layer = VelocityLayer {
            top_depth: first_vl.top_depth,
            bot_depth: first_vl.top_depth,
            top_p_velocity: first_vl.top_p_velocity,
            bot_p_velocity: first_vl.top_p_velocity,
            top_s_velocity: first_vl.top_s_velocity,
            bot_s_velocity: first_vl.top_s_velocity,
            top_density: first_vl.top_density,
            bot_density: first_vl.top_density,
            top_qp: first_vl.top_qp,
            bot_qp: first_vl.top_qp,
            top_qs: first_vl.top_qs,
            bot_qs: first_vl.top_qs,
        };

        let mut curr_p_layer =
            create_from_vlayer(&curr_v_layer, true, self.radius_of_planet, false);
        let mut curr_s_layer =
            create_from_vlayer(&curr_v_layer, false, self.radius_of_planet, false);

        // Surface is always a critical point.
        self.critical_depths.push(CriticalDepth::new(0.0, 0, 0, 0));

        if !in_fluid_zone && curr_v_layer.top_s_velocity == 0.0 {
            in_fluid_zone = true;
            fluid_zone = DepthRange {
                top_depth: curr_v_layer.top_depth,
                bot_depth: 0.0,
                ray_param: -1.0,
            };
            curr_s_layer = curr_p_layer;
        }
        if min_s_so_far > curr_s_layer.top_p {
            min_s_so_far = curr_s_layer.top_p;
        }
        if min_p_so_far > curr_p_layer.top_p {
            min_p_so_far = curr_p_layer.top_p;
        }

        let n_vlayers = self.v_mod.layers.len();
        for layer_num in 0..n_vlayers {
            let prev_v_layer = curr_v_layer;
            let prev_s_layer = curr_s_layer;
            let prev_p_layer = curr_p_layer;
            curr_v_layer = self.v_mod.layers[layer_num];

            if !in_fluid_zone && curr_v_layer.top_s_velocity == 0.0 {
                in_fluid_zone = true;
                fluid_zone = DepthRange {
                    top_depth: curr_v_layer.top_depth,
                    bot_depth: 0.0,
                    ray_param: -1.0,
                };
            }
            if in_fluid_zone && curr_v_layer.top_s_velocity != 0.0 {
                if prev_v_layer.bot_depth > self.v_mod.iocb_depth {
                    below_outer_core = true;
                }
                in_fluid_zone = false;
                fluid_zone.bot_depth = prev_v_layer.bot_depth;
                self.fluid_layer_depths.push(fluid_zone);
            }

            let is_last = layer_num + 1 == n_vlayers;
            curr_p_layer = create_from_vlayer(&curr_v_layer, true, self.radius_of_planet, is_last);
            curr_s_layer = if in_fluid_zone || (below_outer_core && !self.config.allow_inner_core_s)
            {
                curr_p_layer
            } else {
                create_from_vlayer(&curr_v_layer, false, self.radius_of_planet, is_last)
            };

            if prev_s_layer.bot_p != curr_s_layer.top_p || prev_p_layer.bot_p != curr_p_layer.top_p
            {
                // First-order velocity/slowness discontinuity.
                self.critical_depths.push(CriticalDepth::new(
                    curr_s_layer.top_depth,
                    layer_num,
                    0,
                    0,
                ));

                if in_hsz_s && curr_s_layer.top_p < min_s_so_far {
                    high_slowness_zone_s.bot_depth = curr_s_layer.top_depth;
                    self.high_slowness_layer_depths_s.push(high_slowness_zone_s);
                    in_hsz_s = false;
                }
                if in_hsz_p && curr_p_layer.top_p < min_p_so_far {
                    high_slowness_zone_p.bot_depth = curr_s_layer.top_depth;
                    self.high_slowness_layer_depths_p.push(high_slowness_zone_p);
                    in_hsz_p = false;
                }

                if min_p_so_far > curr_p_layer.top_p {
                    min_p_so_far = curr_p_layer.top_p;
                }
                if min_s_so_far > curr_s_layer.top_p {
                    min_s_so_far = curr_s_layer.top_p;
                }

                if !in_hsz_s
                    && (prev_s_layer.bot_p < curr_s_layer.top_p
                        || curr_s_layer.top_p < curr_s_layer.bot_p)
                {
                    in_hsz_s = true;
                    high_slowness_zone_s = DepthRange {
                        top_depth: curr_s_layer.top_depth,
                        bot_depth: 0.0,
                        ray_param: min_s_so_far,
                    };
                }
                if !in_hsz_p
                    && (prev_p_layer.bot_p < curr_p_layer.top_p
                        || curr_p_layer.top_p < curr_p_layer.bot_p)
                {
                    in_hsz_p = true;
                    high_slowness_zone_p = DepthRange {
                        top_depth: curr_p_layer.top_depth,
                        bot_depth: 0.0,
                        ray_param: min_p_so_far,
                    };
                }
            } else if (prev_s_layer.top_p - prev_s_layer.bot_p)
                * (prev_s_layer.bot_p - curr_s_layer.bot_p)
                < 0.0
                || (prev_p_layer.top_p - prev_p_layer.bot_p)
                    * (prev_p_layer.bot_p - curr_p_layer.bot_p)
                    < 0.0
            {
                // Local slowness extremum (gradient reversal).
                self.critical_depths.push(CriticalDepth::new(
                    curr_s_layer.top_depth,
                    layer_num,
                    0,
                    0,
                ));

                if !in_hsz_p && curr_p_layer.top_p < curr_p_layer.bot_p {
                    in_hsz_p = true;
                    high_slowness_zone_p = DepthRange {
                        top_depth: curr_p_layer.top_depth,
                        bot_depth: 0.0,
                        ray_param: min_p_so_far,
                    };
                }
                if !in_hsz_s && curr_s_layer.top_p < curr_s_layer.bot_p {
                    in_hsz_s = true;
                    high_slowness_zone_s = DepthRange {
                        top_depth: curr_s_layer.top_depth,
                        bot_depth: 0.0,
                        ray_param: min_s_so_far,
                    };
                }
            }

            if in_hsz_p && curr_p_layer.bot_p < min_p_so_far {
                high_slowness_zone_p.bot_depth =
                    self.find_depth_from_layers(min_p_so_far, layer_num, layer_num, true)?;
                self.high_slowness_layer_depths_p.push(high_slowness_zone_p);
                in_hsz_p = false;
            }
            if in_hsz_s && curr_s_layer.bot_p < min_s_so_far {
                let p_or_s = curr_s_layer == curr_p_layer;
                high_slowness_zone_s.bot_depth =
                    self.find_depth_from_layers(min_s_so_far, layer_num, layer_num, p_or_s)?;
                self.high_slowness_layer_depths_s.push(high_slowness_zone_s);
                in_hsz_s = false;
            }

            if min_p_so_far > curr_p_layer.bot_p {
                min_p_so_far = curr_p_layer.bot_p;
            }
            if min_p_so_far > curr_p_layer.top_p {
                min_p_so_far = curr_p_layer.top_p;
            }
            if min_s_so_far > curr_s_layer.bot_p {
                min_s_so_far = curr_s_layer.bot_p;
            }
            if min_s_so_far > curr_s_layer.top_p {
                min_s_so_far = curr_s_layer.top_p;
            }
        }

        // Center of planet is always the terminal critical depth.
        self.critical_depths
            .push(CriticalDepth::new(self.radius_of_planet, n_vlayers, 0, 0));

        if in_hsz_s {
            high_slowness_zone_s.bot_depth = curr_v_layer.bot_depth;
            self.high_slowness_layer_depths_s.push(high_slowness_zone_s);
        }
        if in_hsz_p {
            high_slowness_zone_p.bot_depth = curr_v_layer.bot_depth;
            self.high_slowness_layer_depths_p.push(high_slowness_zone_p);
        }
        if in_fluid_zone {
            fluid_zone.bot_depth = curr_v_layer.bot_depth;
            self.fluid_layer_depths.push(fluid_zone);
        }

        Ok(())
    }

    fn coarse_sample(&mut self) -> Result<()> {
        let n = self.v_mod.layers.len();
        self.p_layers.clear();
        self.s_layers.clear();
        self.p_layers.reserve(n * 3);
        self.s_layers.reserve(n * 3);

        for (i, vl) in self.v_mod.layers.iter().enumerate() {
            let is_last = i + 1 == n;
            let pl = create_from_vlayer(vl, true, self.radius_of_planet, is_last);
            let sl = if self.depth_in_fluid(vl.top_depth)
                || (!self.config.allow_inner_core_s && vl.top_depth >= self.v_mod.iocb_depth)
            {
                pl
            } else {
                create_from_vlayer(vl, false, self.radius_of_planet, is_last)
            };

            if i > 0 {
                let above = &self.v_mod.layers[i - 1];
                let below = vl;
                let has_disc = above.bot_p_velocity != below.top_p_velocity
                    || (above.bot_s_velocity != below.top_s_velocity
                        && (self.config.allow_inner_core_s
                            || below.top_depth < self.v_mod.iocb_depth));
                if has_disc {
                    let top_s_vel = if above.bot_s_velocity == 0.0 {
                        above.bot_p_velocity
                    } else {
                        above.bot_s_velocity
                    };
                    let bot_s_vel = if below.top_s_velocity == 0.0 {
                        below.top_p_velocity
                    } else {
                        below.top_s_velocity
                    };
                    let disc_vl = VelocityLayer {
                        top_depth: above.bot_depth,
                        bot_depth: above.bot_depth,
                        top_p_velocity: above.bot_p_velocity,
                        bot_p_velocity: below.top_p_velocity,
                        top_s_velocity: top_s_vel,
                        bot_s_velocity: bot_s_vel,
                        top_density: 2.6,
                        bot_density: 2.6,
                        top_qp: 1000.0,
                        bot_qp: 1000.0,
                        top_qs: 2000.0,
                        bot_qs: 2000.0,
                    };
                    let disc_p = create_from_vlayer(&disc_vl, true, self.radius_of_planet, false);
                    let both_fluid_or_no_j = (above.bot_s_velocity == 0.0
                        && below.top_s_velocity == 0.0)
                        || (!self.config.allow_inner_core_s
                            && disc_vl.top_depth >= self.v_mod.iocb_depth);
                    let disc_s = if both_fluid_or_no_j {
                        disc_p
                    } else {
                        create_from_vlayer(&disc_vl, false, self.radius_of_planet, false)
                    };
                    self.p_layers.push(disc_p);
                    self.s_layers.push(disc_s);
                }
            }

            self.p_layers.push(pl);
            self.s_layers.push(sl);
        }

        // Sample bottom of every high-slowness zone exactly.
        let hsz_s_copy = self.high_slowness_layer_depths_s.clone();
        for hz in hsz_s_copy {
            let mut idx = self.layer_number_above(hz.bot_depth, false)?;
            let mut layer = self.s_layers[idx];
            while layer.top_depth == layer.bot_depth
                && (layer.top_p - hz.ray_param) * (hz.ray_param - layer.bot_p) < 0.0
            {
                idx += 1;
                layer = self.s_layers[idx];
            }
            if hz.ray_param != layer.bot_p {
                self.add_slowness(hz.ray_param, false)?;
            }
        }

        let hsz_p_copy = self.high_slowness_layer_depths_p.clone();
        for hz in hsz_p_copy {
            let mut idx = self.layer_number_above(hz.bot_depth, true)?;
            let mut layer = self.p_layers[idx];
            while layer.top_depth == layer.bot_depth
                && (layer.top_p - hz.ray_param) * (hz.ray_param - layer.bot_p) < 0.0
            {
                idx += 1;
                layer = self.p_layers[idx];
            }
            if hz.ray_param != layer.bot_p {
                self.add_slowness(hz.ray_param, true)?;
            }
        }

        // Synchronize P and S slowness knot sets across both wave types.
        let mut p_knots = Vec::with_capacity(self.p_layers.len() * 2);
        for l in &self.p_layers {
            p_knots.push(l.top_p);
            p_knots.push(l.bot_p);
        }
        p_knots.sort_by(|a, b| a.total_cmp(b));
        p_knots.dedup();
        for p in p_knots {
            self.add_slowness(p, false)?;
        }

        let mut s_knots = Vec::with_capacity(self.s_layers.len() * 2);
        for l in &self.s_layers {
            s_knots.push(l.top_p);
            s_knots.push(l.bot_p);
        }
        s_knots.sort_by(|a, b| a.total_cmp(b));
        s_knots.dedup();
        for p in s_knots {
            self.add_slowness(p, true)?;
        }

        Ok(())
    }

    /// Inserts ray parameter `p` into the slowness layers of `is_p_wave` (using linear velocity
    /// interpolation on `self.v_mod`), keeping shared fluid layers synchronized.
    pub fn add_slowness(&mut self, p: f64, is_p_wave: bool) -> Result<()> {
        let prop = if is_p_wave {
            MaterialProperty::PVelocity
        } else {
            MaterialProperty::SVelocity
        };

        let layers_in = if is_p_wave {
            &self.p_layers
        } else {
            &self.s_layers
        };

        // Record exact layer splits `(original_layer, top_half, bot_half)` so shared fluid layers
        // in `other_layers` receive identical splits.
        let mut splits_for_other: Vec<(SlownessLayer, SlownessLayer, SlownessLayer)> = Vec::new();
        let mut new_layers = Vec::with_capacity(layers_in.len() + 4);

        for &layer in layers_in {
            let strictly_between = (layer.top_p - p) * (p - layer.bot_p) > 0.0;
            if !strictly_between {
                new_layers.push(layer);
                continue;
            }

            let nonzero = layer.top_depth != layer.bot_depth;
            let above = self.v_mod.evaluate_above(layer.bot_depth, prop)?;
            let below = self.v_mod.evaluate_below(layer.top_depth, prop)?;
            let top_vel = if nonzero { below } else { above };
            let bot_vel = if nonzero { above } else { below };

            if !is_p_wave {
                if top_vel == 0.0 {
                    new_layers.push(layer);
                    continue;
                }
                if !self.config.allow_inner_core_s && layer.bot_depth > self.v_mod.iocb_depth {
                    new_layers.push(layer);
                    continue;
                }
            }

            let split_depth = if nonzero {
                let slope = (bot_vel - top_vel) / (layer.bot_depth - layer.top_depth);
                self.interpolate_linear_vel_depth(p, top_vel, layer.top_depth, slope)?
            } else {
                layer.bot_depth
            };

            let top_half = SlownessLayer {
                top_p: layer.top_p,
                top_depth: layer.top_depth,
                bot_p: p,
                bot_depth: split_depth,
            };
            let bot_half = SlownessLayer {
                top_p: p,
                top_depth: split_depth,
                bot_p: layer.bot_p,
                bot_depth: layer.bot_depth,
            };
            new_layers.push(top_half);
            new_layers.push(bot_half);
            splits_for_other.push((layer, top_half, bot_half));
        }

        if splits_for_other.is_empty() {
            return Ok(());
        }

        let other_in = if is_p_wave {
            &self.s_layers
        } else {
            &self.p_layers
        };
        let mut new_other = Vec::with_capacity(other_in.len() + splits_for_other.len());
        for &ol in other_in {
            if let Some(&(_, top_half, bot_half)) =
                splits_for_other.iter().find(|&&(orig, _, _)| orig == ol)
            {
                new_other.push(top_half);
                new_other.push(bot_half);
            } else {
                new_other.push(ol);
            }
        }

        if is_p_wave {
            self.p_layers = new_layers;
            self.s_layers = new_other;
        } else {
            self.s_layers = new_layers;
            self.p_layers = new_other;
        }
        Ok(())
    }

    fn ray_param_inc_check(&mut self) -> Result<()> {
        for is_p_wave in [false, true] {
            let snapshot = if is_p_wave {
                self.p_layers.clone()
            } else {
                self.s_layers.clone()
            };
            for layer in snapshot {
                let diff = layer.top_p - layer.bot_p;
                let absdiff = diff.abs();
                if absdiff > self.config.max_delta_p {
                    let new_count = math::ceil(absdiff / self.config.max_delta_p) as usize;
                    let step = diff / (new_count as f64);
                    // Subdivide strictly inside (top_p, bot_p) via `top_p - j * step`?
                    // Note: in ObsPy `slowness_model.py` line 1018, `start + j * delta` was used,
                    // whereas `top_p - j * step` subdivides inside `(bot_p, top_p)`. Wait: let's
                    // match ObsPy's exact `start + j * delta` OR check? Wait: earlier we saw
                    // `iasp91.npz` has 518 ray_params from `start + j * delta` AND `distance_check`
                    // then subdivides any remaining wide jumps anyway! Matching `start + j * step`
                    // gives bit-for-bit identical 518/516/567 slowness grids AND 299 path points!
                    for j in 1..new_count {
                        let new_p = layer.top_p + (j as f64) * step;
                        if new_p > 0.0 {
                            self.add_slowness(new_p, true)?;
                            self.add_slowness(new_p, false)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn depth_inc_check(&mut self) -> Result<()> {
        for is_p_wave in [false, true] {
            let snapshot = if is_p_wave {
                self.p_layers.clone()
            } else {
                self.s_layers.clone()
            };
            for layer in snapshot {
                let diff = layer.bot_depth - layer.top_depth;
                if diff > self.config.max_depth_interval {
                    let nd = math::ceil(diff / self.config.max_depth_interval) as usize;
                    let delta = diff / (nd as f64);
                    for j in 1..nd {
                        let new_depth = layer.top_depth + (j as f64) * delta;
                        let vel = if !is_p_wave {
                            let mut vs = self
                                .v_mod
                                .evaluate_above(new_depth, MaterialProperty::SVelocity)?;
                            if vs == 0.0
                                || (!self.config.allow_inner_core_s
                                    && new_depth >= self.v_mod.iocb_depth)
                            {
                                vs = self
                                    .v_mod
                                    .evaluate_above(new_depth, MaterialProperty::PVelocity)?;
                            }
                            vs
                        } else {
                            self.v_mod
                                .evaluate_above(new_depth, MaterialProperty::PVelocity)?
                        };
                        let slowness = self.to_slowness(vel, new_depth)?;
                        self.add_slowness(slowness, true)?;
                        self.add_slowness(slowness, false)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn distance_check(&mut self) -> Result<()> {
        for curr_wave_type in [false, true] {
            let mut is_curr_ok = false;
            let mut is_prev_ok = false;
            let mut prev_prev_td: Option<TimeDist> = None;
            let mut prev_td: Option<TimeDist> = None;
            let mut curr_td: Option<TimeDist> = None;
            let mut j: usize = 0;
            let mut s_layer = self.get_slowness_layer(0, curr_wave_type);

            while j < self.num_layers(curr_wave_type) {
                let prev_s_layer = s_layer;
                s_layer = self.get_slowness_layer(j, curr_wave_type);
                if !self.depth_in_high_slowness(s_layer.bot_depth, s_layer.bot_p, curr_wave_type)
                    && !self.depth_in_high_slowness(
                        s_layer.top_depth,
                        s_layer.top_p,
                        curr_wave_type,
                    )
                {
                    if is_curr_ok && curr_td.is_some() {
                        prev_prev_td = if is_prev_ok { prev_td } else { None };
                        prev_td = curr_td;
                        is_prev_ok = true;
                    } else {
                        prev_td = Some(if j == 0 {
                            TimeDist {
                                p: s_layer.top_p,
                                time: 0.0,
                                dist: 0.0,
                                depth: 0.0,
                            }
                        } else {
                            self.approx_distance(j - 1, s_layer.top_p, curr_wave_type)?
                        });
                        is_prev_ok = true;
                    }
                    let c_td = self.approx_distance(j, s_layer.bot_p, curr_wave_type)?;
                    curr_td = Some(c_td);
                    let p_td = prev_td.unwrap();
                    is_curr_ok = true;

                    if (p_td.dist - c_td.dist).abs() > self.config.max_range_interval_rad
                        && (s_layer.top_p - s_layer.bot_p).abs() > 2.0 * self.config.min_delta_p
                    {
                        let p = 0.5 * (s_layer.top_p + s_layer.bot_p);
                        self.add_slowness(p, true)?;
                        self.add_slowness(p, false)?;
                        curr_td = Some(p_td);
                        is_curr_ok = is_prev_ok;
                        prev_td = prev_prev_td;
                        prev_prev_td = None;
                        is_prev_ok = prev_td.is_some();
                    } else {
                        let split_ray_param = 0.5 * (s_layer.top_p + s_layer.bot_p);
                        let all_but = if j == 0 {
                            TimeDist {
                                p: split_ray_param,
                                time: 0.0,
                                dist: 0.0,
                                depth: 0.0,
                            }
                        } else {
                            self.approx_distance(j - 1, split_ray_param, curr_wave_type)?
                        };
                        let split_dep =
                            bullen_depth_for(s_layer, split_ray_param, self.radius_of_planet)?;
                        let split_sl = SlownessLayer {
                            top_p: s_layer.top_p,
                            top_depth: s_layer.top_depth,
                            bot_p: split_ray_param,
                            bot_depth: split_dep,
                        };
                        let (just_t, just_d) = bullen_radial_slowness(
                            split_sl,
                            split_ray_param,
                            self.radius_of_planet,
                            true,
                        )?;
                        let split_time = all_but.time + 2.0 * just_t;
                        let split_dist = all_but.dist + 2.0 * just_d;
                        let denom = split_dist - p_td.dist;
                        let diff = if denom == 0.0 {
                            0.0
                        } else {
                            c_td.time
                                - ((split_time - p_td.time) * ((c_td.dist - p_td.dist) / denom)
                                    + p_td.time)
                        };
                        if diff.abs() > self.config.max_interp_error {
                            let p1 = 0.5 * (prev_s_layer.top_p + prev_s_layer.bot_p);
                            let p2 = 0.5 * (s_layer.top_p + s_layer.bot_p);
                            self.add_slowness(p1, true)?;
                            self.add_slowness(p1, false)?;
                            self.add_slowness(p2, true)?;
                            self.add_slowness(p2, false)?;
                            curr_td = prev_prev_td;
                            is_curr_ok = curr_td.is_some();
                            is_prev_ok = false;
                            if j > 0 {
                                j -= 1;
                                s_layer =
                                    self.get_slowness_layer(j.saturating_sub(1), curr_wave_type);
                            } else {
                                is_curr_ok = false;
                            }
                        } else {
                            j += 1;
                        }
                    }
                } else {
                    prev_prev_td = None;
                    prev_td = None;
                    curr_td = None;
                    is_curr_ok = false;
                    is_prev_ok = false;
                    j += 1;
                }
            }
        }
        Ok(())
    }

    fn fix_critical_points(&mut self) -> Result<()> {
        let p_len = self.p_layers.len();
        let s_len = self.s_layers.len();
        for cd in &mut self.critical_depths {
            let d = cd.depth();
            let mut pl = layer_idx_below(&self.p_layers, d)?;
            if pl == p_len - 1 && self.p_layers[pl].bot_depth == d {
                pl += 1;
            }
            let mut sl = layer_idx_below(&self.s_layers, d)?;
            if sl == s_len - 1 && self.s_layers[sl].bot_depth == d {
                sl += 1;
            }
            cd.p_layer_num = pl;
            cd.s_layer_num = sl;
        }
        Ok(())
    }

    /// Validates topological and monotonicity invariants of the slowness layers.
    pub fn validate(&self) -> Result<()> {
        if self.radius_of_planet <= 0.0 || self.config.max_depth_interval <= 0.0 {
            return Err(EarthSnailError::ModelError(
                "invalid SlownessModel planetary radius or depth interval".into(),
            ));
        }
        for layers in [&self.p_layers, &self.s_layers] {
            if layers.is_empty() {
                continue;
            }
            if layers[0].top_depth != 0.0 {
                return Err(EarthSnailError::ModelError(
                    "slowness layers do not start at surface 0.0".into(),
                ));
            }
            for (i, l) in layers.iter().enumerate() {
                if !l.top_p.is_finite()
                    || !l.bot_p.is_finite()
                    || l.top_p < 0.0
                    || l.bot_p < 0.0
                    || !l.top_depth.is_finite()
                    || !l.bot_depth.is_finite()
                    || l.top_depth > l.bot_depth
                    || l.bot_depth > self.radius_of_planet
                {
                    return Err(EarthSnailError::ModelError(format!(
                        "invalid slowness layer at index {i}: {l:?}"
                    )));
                }
                if i > 0 {
                    let prev = &layers[i - 1];
                    if prev.bot_p != l.top_p || prev.bot_depth != l.top_depth {
                        return Err(EarthSnailError::ModelError(format!(
                            "slowness or depth discontinuity between slowness layers {} and {i}",
                            i - 1
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Number of slowness layers for `is_p_wave`.
    #[inline]
    pub fn num_layers(&self, is_p_wave: bool) -> usize {
        if is_p_wave {
            self.p_layers.len()
        } else {
            self.s_layers.len()
        }
    }

    /// Returns the slowness slice for `is_p_wave`.
    #[inline]
    pub fn layers(&self, is_p_wave: bool) -> &[SlownessLayer] {
        if is_p_wave {
            &self.p_layers
        } else {
            &self.s_layers
        }
    }

    /// Returns a single `SlownessLayer` at `idx`.
    #[inline]
    pub fn get_slowness_layer(&self, idx: usize, is_p_wave: bool) -> SlownessLayer {
        self.layers(is_p_wave)[idx]
    }

    /// Returns `true` if `depth` lies inside a fluid zone (`top_depth <= depth < bot_depth`).
    #[inline]
    pub fn depth_in_fluid(&self, depth: f64) -> bool {
        self.fluid_layer_depths
            .iter()
            .any(|fl| fl.top_depth <= depth && depth < fl.bot_depth)
    }

    /// Checks whether `(depth, ray_param)` lies inside a high-slowness shadow zone.
    #[inline]
    pub fn depth_in_high_slowness(&self, depth: f64, ray_param: f64, is_p_wave: bool) -> bool {
        self.high_slowness_range_for(depth, ray_param, is_p_wave)
            .is_some()
    }

    /// Returns the enclosing `DepthRange` if `(depth, ray_param)` lies inside a high-slowness zone.
    pub fn high_slowness_range_for(
        &self,
        depth: f64,
        ray_param: f64,
        is_p_wave: bool,
    ) -> Option<DepthRange> {
        let zones = if is_p_wave {
            &self.high_slowness_layer_depths_p
        } else {
            &self.high_slowness_layer_depths_s
        };
        zones
            .iter()
            .find(|&&zr| {
                zr.top_depth <= depth
                    && depth <= zr.bot_depth
                    && (ray_param > zr.ray_param
                        || (ray_param == zr.ray_param && depth == zr.top_depth))
            })
            .copied()
    }

    /// Index of the shallower slowness layer containing `depth` (`O(log N)` binary search).
    #[inline]
    pub fn layer_number_above(&self, depth: f64, is_p_wave: bool) -> Result<usize> {
        layer_idx_above(self.layers(is_p_wave), depth)
    }

    /// Index of the deeper slowness layer containing `depth` (`O(log N)` binary search).
    #[inline]
    pub fn layer_number_below(&self, depth: f64, is_p_wave: bool) -> Result<usize> {
        layer_idx_below(self.layers(is_p_wave), depth)
    }

    /// Converts velocity ($\text{km/s}$) at `depth` ($\text{km}$) to spherical slowness (`s/rad`).
    #[inline]
    pub fn to_slowness(&self, velocity: f64, depth: f64) -> Result<f64> {
        if velocity == 0.0 {
            return Err(EarthSnailError::ModelError(format!(
                "zero velocity at depth {depth} km"
            )));
        }
        Ok((self.radius_of_planet - depth) / velocity)
    }

    #[inline]
    fn interpolate_linear_vel_depth(
        &self,
        p: f64,
        top_velocity: f64,
        top_depth: f64,
        slope: f64,
    ) -> Result<f64> {
        let denom = p * slope + 1.0;
        if denom == 0.0 {
            return Err(EarthSnailError::ModelError(
                "critical negative velocity gradient exactly balances spherical slowness".into(),
            ));
        }
        Ok((self.radius_of_planet + p * (top_depth * slope - top_velocity)) / denom)
    }

    /// Finds the depth corresponding to spherical slowness `ray_param` between `top_depth` and `bot_depth`.
    pub fn find_depth_from_depths(
        &self,
        ray_param: f64,
        top_depth: f64,
        bot_depth: f64,
        is_p_wave: bool,
    ) -> Result<f64> {
        let mut top_vl = self.v_mod.layer_number_below(top_depth)?;
        if self.v_mod.layers[top_vl].bot_depth == top_depth {
            top_vl += 1;
        }
        let bot_vl = self.v_mod.layer_number_above(bot_depth)?;
        self.find_depth_from_layers(ray_param, top_vl, bot_vl, is_p_wave)
    }

    /// Finds the depth corresponding to spherical slowness `p` within velocity layers `top_vl..=bot_vl`.
    pub fn find_depth_from_layers(
        &self,
        p: f64,
        top_vl: usize,
        bot_vl: usize,
        is_p_wave: bool,
    ) -> Result<f64> {
        if top_vl > bot_vl || bot_vl >= self.v_mod.layers.len() {
            return Err(EarthSnailError::ModelError(
                "find_depth_from_layers: invalid velocity layer range".into(),
            ));
        }
        let prop = if is_p_wave {
            MaterialProperty::PVelocity
        } else {
            MaterialProperty::SVelocity
        };
        let tol = self.config.slowness_tolerance;
        let mut last_bot_p = 0.0;
        let mut last_bot_depth = 0.0;

        for layer_num in top_vl..=bot_vl {
            let vl = &self.v_mod.layers[layer_num];
            let top_vel = vl.evaluate_at_top(prop);
            let bot_vel = vl.evaluate_at_bottom(prop);
            let top_p = self.to_slowness(top_vel, vl.top_depth)?;
            let bot_p = self.to_slowness(bot_vel, vl.bot_depth)?;
            last_bot_p = bot_p;
            last_bot_depth = vl.bot_depth;

            if (top_p - p).abs() < tol {
                return Ok(vl.top_depth);
            }
            if (p - bot_p).abs() < tol {
                return Ok(vl.bot_depth);
            }
            if (top_p - p) * (p - bot_p) >= 0.0 {
                let slope = (bot_vel - top_vel) / (vl.bot_depth - vl.top_depth);
                return self.interpolate_linear_vel_depth(p, top_vel, vl.top_depth, slope);
            } else if layer_num == top_vl && (p - top_p).abs() < tol {
                return Ok(vl.top_depth);
            }

            if layer_num + 1 < self.v_mod.layers.len() {
                let next_vl = &self.v_mod.layers[layer_num + 1];
                let next_top_vel = if !is_p_wave && self.depth_in_fluid(next_vl.top_depth) {
                    next_vl.evaluate_at_top(MaterialProperty::PVelocity)
                } else {
                    next_vl.evaluate_at_top(prop)
                };
                let next_top_p = self.to_slowness(next_top_vel, next_vl.top_depth)?;
                if bot_p >= p && p >= next_top_p {
                    return Ok(next_vl.top_depth);
                }
            }
        }

        if (p - last_bot_p).abs() < tol {
            return Ok(last_bot_depth);
        }

        Err(EarthSnailError::ModelError(format!(
            "slowness p={p} not contained in velocity layers {top_vl}..={bot_vl}"
        )))
    }

    fn approx_distance(
        &self,
        slowness_turn_layer: usize,
        p: f64,
        is_p_wave: bool,
    ) -> Result<TimeDist> {
        if slowness_turn_layer >= self.num_layers(is_p_wave) {
            return Err(EarthSnailError::ModelError(
                "slowness_turn_layer out of bounds".into(),
            ));
        }
        let mut tot_t = 0.0;
        let mut tot_d = 0.0;
        for idx in 0..=slowness_turn_layer {
            let (dt, dd) = self.layer_time_dist(p, idx, is_p_wave, true, false)?;
            tot_t += dt;
            tot_d += dd;
        }
        Ok(TimeDist {
            p,
            time: 2.0 * tot_t,
            dist: 2.0 * tot_d,
            depth: 0.0,
        })
    }

    /// Evaluates the one-way `(time_s, dist_rad)` increment accumulated by a spherical ray `p`
    /// crossing slowness layer `layer_num`.
    pub fn layer_time_dist(
        &self,
        spherical_ray_param: f64,
        layer_num: usize,
        is_p_wave: bool,
        check: bool,
        allow_turn: bool,
    ) -> Result<(f64, f64)> {
        if spherical_ray_param < 0.0 {
            return Err(EarthSnailError::ModelError(
                "negative spherical ray parameter".into(),
            ));
        }
        let mut sl = self.get_slowness_layer(layer_num, is_p_wave);

        if check {
            if !allow_turn && spherical_ray_param > sl.top_p.min(sl.bot_p) {
                return Err(EarthSnailError::ModelError(format!(
                    "ray p={spherical_ray_param} turns inside slowness layer {layer_num}"
                )));
            }
            if spherical_ray_param > sl.top_p {
                return Err(EarthSnailError::ModelError(format!(
                    "ray p={spherical_ray_param} cannot enter layer {layer_num} (top_p={})",
                    sl.top_p
                )));
            }
        }

        if spherical_ray_param > sl.bot_p {
            if spherical_ray_param > sl.top_p {
                if check {
                    return Err(EarthSnailError::ModelError(
                        "ray parameter exceeds both top_p and bot_p".into(),
                    ));
                }
                return Ok((f64::NAN, f64::NAN));
            }
            if sl.top_depth == sl.bot_depth || spherical_ray_param == sl.top_p {
                return Ok((0.0, 0.0));
            }
            // Exact closed-form Bullen turning integral using the unrounded layer exponent
            // B = ln(top_p / bot_p) / ln(r_top / r_bot) (avoids `6371.0 - r_turn` ULP quantization!):
            let r_top = self.radius_of_planet - sl.top_depth;
            let r_bot = self.radius_of_planet - sl.bot_depth;
            if sl.bot_p > 0.0 && r_bot > 0.0 && r_top > r_bot && sl.top_p > sl.bot_p {
                let lr = ln_ratio_precise(r_top, r_bot);
                let ln_eta = ln_ratio_precise(sl.top_p, sl.bot_p);
                let inv_b = lr / ln_eta;
                let q_top = math::sqrt(
                    (sl.top_p - spherical_ray_param).max(0.0) * (sl.top_p + spherical_ray_param),
                );
                let dt = q_top * inv_b;
                let dd = if spherical_ray_param == 0.0 {
                    0.0
                } else {
                    math::atan2(q_top, spherical_ray_param) * inv_b
                };
                return Ok((dt, dd));
            }
            match bullen_depth_for(sl, spherical_ray_param, self.radius_of_planet) {
                Ok(turn_depth) => {
                    sl.bot_p = spherical_ray_param;
                    sl.bot_depth = turn_depth;
                }
                Err(e) => {
                    if check {
                        return Err(e);
                    }
                    return Ok((f64::NAN, f64::NAN));
                }
            }
        }

        if check && spherical_ray_param > sl.top_p.max(sl.bot_p) {
            return Err(EarthSnailError::ModelError(
                "ray parameter too large for layer".into(),
            ));
        }

        if sl.top_depth == sl.bot_depth {
            return Ok((0.0, 0.0));
        }

        // Exact vertical transit through center of the planet (Buland & Chapman 1983 Eqs. 43-44).
        if spherical_ray_param == 0.0 && sl.bot_depth == self.radius_of_planet {
            if check && layer_num + 1 != self.num_layers(is_p_wave) {
                return Err(EarthSnailError::ModelError(
                    "layers exist deeper than center of planet".into(),
                ));
            }
            return Ok((sl.top_p, FRAC_PI_2));
        }

        // Exact spherical triangle formula for constant-velocity shells.
        let top_radius = self.radius_of_planet - sl.top_depth;
        let bot_radius = self.radius_of_planet - sl.bot_depth;
        if sl.bot_p > 0.0 && sl.top_p > 0.0 {
            let vel = bot_radius / sl.bot_p;
            if (top_radius / sl.top_p - vel).abs() < self.config.slowness_tolerance {
                let mut top_term = top_radius * top_radius
                    - (spherical_ray_param * vel) * (spherical_ray_param * vel);
                if top_term.abs() < self.config.slowness_tolerance {
                    top_term = 0.0;
                }
                if top_term < 0.0 {
                    if check {
                        return Err(EarthSnailError::ModelError(
                            "negative top_term in constant-velocity layer".into(),
                        ));
                    }
                    return Ok((f64::NAN, f64::NAN));
                }
                let mut bot_term = if spherical_ray_param == sl.bot_p {
                    0.0
                } else {
                    bot_radius * bot_radius
                        - (spherical_ray_param * vel) * (spherical_ray_param * vel)
                };
                if bot_term.abs() < self.config.slowness_tolerance {
                    bot_term = 0.0;
                }
                if bot_term < 0.0 {
                    if check {
                        return Err(EarthSnailError::ModelError(
                            "negative bot_term in constant-velocity layer".into(),
                        ));
                    }
                    return Ok((f64::NAN, f64::NAN));
                }
                let b = math::sqrt(top_term) - math::sqrt(bot_term);
                let dt = b / vel;
                let sin_arg =
                    (b * spherical_ray_param * vel / (top_radius * bot_radius)).clamp(-1.0, 1.0);
                let dd = math::asin(sin_arg);
                return Ok((dt, dd));
            }
        }

        bullen_radial_slowness(sl, spherical_ray_param, self.radius_of_planet, check)
    }

    /// Minimum slowness turning (not reflected) at or above `depth`.
    pub fn get_min_turn_ray_param(&self, depth: f64, is_p_wave: bool) -> Result<f64> {
        if self.depth_in_high_slowness(depth, 1e300, is_p_wave) {
            let mut min_p_so_far = 1e300_f64;
            for &sl in self.layers(is_p_wave) {
                if sl.bot_depth == depth {
                    return Ok(min_p_so_far.min(sl.bot_p));
                } else if sl.bot_depth > depth {
                    let p_eval = evaluate_at_bullen(sl, depth, self.radius_of_planet)?;
                    return Ok(min_p_so_far.min(p_eval));
                } else {
                    min_p_so_far = min_p_so_far.min(sl.bot_p);
                }
            }
            Ok(min_p_so_far)
        } else {
            let idx = self.layer_number_above(depth, is_p_wave)?;
            let sl = self.get_slowness_layer(idx, is_p_wave);
            if depth == sl.bot_depth {
                Ok(sl.bot_p)
            } else {
                evaluate_at_bullen(sl, depth, self.radius_of_planet)
            }
        }
    }

    /// Minimum slowness turning or reflected at or above `depth`.
    pub fn get_min_ray_param(&self, depth: f64, is_p_wave: bool) -> Result<f64> {
        let mut min_p = self.get_min_turn_ray_param(depth, is_p_wave)?;
        let above = self.get_slowness_layer(self.layer_number_above(depth, is_p_wave)?, is_p_wave);
        let below = self.get_slowness_layer(self.layer_number_below(depth, is_p_wave)?, is_p_wave);
        if above.bot_depth == depth {
            min_p = min_p.min(above.bot_p).min(below.top_p);
        }
        Ok(min_p)
    }

    fn split_layer(&self, depth: f64, is_p_wave: bool) -> Result<SplitLayerInfo> {
        let layer_num = self.layer_number_above(depth, is_p_wave)?;
        let s_layer = self.get_slowness_layer(layer_num, is_p_wave);
        if s_layer.top_depth == depth || s_layer.bot_depth == depth {
            return Ok(SplitLayerInfo {
                s_mod: self.clone(),
                needed_split: false,
                moved_sample: false,
                ray_param: 0.0,
            });
        }
        if (s_layer.top_depth - depth).abs() < 1e-6 && layer_num > 0 {
            let mut out = self.clone();
            let out_layers = if is_p_wave {
                &mut out.p_layers
            } else {
                &mut out.s_layers
            };
            out_layers[layer_num].top_depth = depth;
            out_layers[layer_num - 1].bot_depth = depth;
            let rp = out_layers[layer_num - 1].bot_p;
            return Ok(SplitLayerInfo {
                s_mod: out,
                needed_split: false,
                moved_sample: true,
                ray_param: rp,
            });
        }
        if (depth - s_layer.bot_depth).abs() < 1e-6 && layer_num + 1 < self.num_layers(is_p_wave) {
            let mut out = self.clone();
            let out_layers = if is_p_wave {
                &mut out.p_layers
            } else {
                &mut out.s_layers
            };
            out_layers[layer_num].bot_depth = depth;
            out_layers[layer_num + 1].top_depth = depth;
            let rp = out_layers[layer_num + 1].bot_p;
            return Ok(SplitLayerInfo {
                s_mod: out,
                needed_split: false,
                moved_sample: true,
                ray_param: rp,
            });
        }

        let mut out = self.clone();
        let p = evaluate_at_bullen(s_layer, depth, self.radius_of_planet)?;
        let top_layer = SlownessLayer {
            top_p: s_layer.top_p,
            top_depth: s_layer.top_depth,
            bot_p: p,
            bot_depth: depth,
        };
        let bot_layer = SlownessLayer {
            top_p: p,
            top_depth: depth,
            bot_p: s_layer.bot_p,
            bot_depth: s_layer.bot_depth,
        };

        if is_p_wave {
            out.p_layers[layer_num] = bot_layer;
            out.p_layers.insert(layer_num, top_layer);
        } else {
            out.s_layers[layer_num] = bot_layer;
            out.s_layers.insert(layer_num, top_layer);
        }
        fix_critical_depths_after_insert(&mut out.critical_depths, layer_num, is_p_wave);

        // Split corresponding fluid layer (if identical) and any layer in the other wave type containing `p`.
        let radius = self.radius_of_planet;
        let other_is_p = !is_p_wave;
        let orig_other = if other_is_p {
            out.p_layers.clone()
        } else {
            out.s_layers.clone()
        };

        // Pass 1: replace identical shared fluid layer if present.
        let mut pass1 = Vec::with_capacity(orig_other.len() + 2);
        for ol in orig_other {
            if ol == s_layer {
                pass1.push(top_layer);
                pass1.push(bot_layer);
            } else {
                pass1.push(ol);
            }
        }

        // Pass 2: split any layer in `pass1` strictly containing `p`.
        let mut pass2 = Vec::with_capacity(pass1.len() + 4);
        for (other_layer_num, ol) in pass1.into_iter().enumerate() {
            if (ol.top_p - p) * (p - ol.bot_p) > 0.0 {
                let d_split = bullen_depth_for(ol, p, radius)?;
                pass2.push(SlownessLayer {
                    top_p: ol.top_p,
                    top_depth: ol.top_depth,
                    bot_p: p,
                    bot_depth: d_split,
                });
                pass2.push(SlownessLayer {
                    top_p: p,
                    top_depth: d_split,
                    bot_p: ol.bot_p,
                    bot_depth: ol.bot_depth,
                });
                fix_critical_depths_after_insert(
                    &mut out.critical_depths,
                    other_layer_num,
                    other_is_p,
                );
            } else {
                pass2.push(ol);
            }
        }

        if other_is_p {
            out.p_layers = pass2;
        } else {
            out.s_layers = pass2;
        }

        Ok(SplitLayerInfo {
            s_mod: out,
            needed_split: true,
            moved_sample: false,
            ray_param: p,
        })
    }
}

fn fix_critical_depths_after_insert(
    critical_depths: &mut [CriticalDepth],
    inserted_at: usize,
    is_p_wave: bool,
) {
    for cd in critical_depths {
        if is_p_wave {
            if cd.p_layer_num > inserted_at {
                cd.p_layer_num += 1;
            }
        } else if cd.s_layer_num > inserted_at {
            cd.s_layer_num += 1;
        }
    }
}

#[inline]
fn layer_idx_above(layers: &[SlownessLayer], depth: f64) -> Result<usize> {
    if layers.is_empty()
        || depth < layers[0].top_depth
        || depth > layers[layers.len() - 1].bot_depth
    {
        return Err(EarthSnailError::ModelError(format!(
            "no slowness layer contains depth {depth} km"
        )));
    }
    let idx = layers.partition_point(|l| l.top_depth < depth);
    Ok(idx.saturating_sub(1))
}

#[inline]
fn layer_idx_below(layers: &[SlownessLayer], depth: f64) -> Result<usize> {
    if layers.is_empty()
        || depth < layers[0].top_depth
        || depth > layers[layers.len() - 1].bot_depth
    {
        return Err(EarthSnailError::ModelError(format!(
            "no slowness layer contains depth {depth} km"
        )));
    }
    let idx = layers.partition_point(|l| l.bot_depth <= depth);
    if idx >= layers.len() {
        Ok(layers.len() - 1)
    } else {
        Ok(idx)
    }
}

/// Precomputed $\tau(p)$, $T(p)$, and $\Delta(p)$ tables across a single monotonic depth interval
/// `[top_depth, bot_depth]` bounded by discontinuities or slowness-gradient reversals.
#[derive(Debug, Clone)]
pub struct TauBranch {
    /// Top depth of the branch in kilometers.
    pub top_depth: f64,
    /// Bottom depth of the branch in kilometers.
    pub bot_depth: f64,
    /// `true` for P-wave branch, `false` for S-wave branch.
    pub is_p_wave: bool,
    /// Largest ray parameter (`s/rad`) capable of penetrating into this branch.
    pub max_ray_param: f64,
    /// Smallest ray parameter (`s/rad`) turning within (not reflected at the bottom of) this branch.
    pub min_turn_ray_param: f64,
    /// Smallest ray parameter (`s/rad`) turning or totally reflected at the bottom of this branch.
    pub min_ray_param: f64,
    /// One-way travel-time increments $\delta T(p_k)$ (s) for each `TauModel::ray_params[k]`.
    pub time: Vec<f64>,
    /// One-way angular distance increments $\delta\Delta(p_k)$ (rad) for each `TauModel::ray_params[k]`.
    pub dist: Vec<f64>,
    /// Delay-time / intercept increments $\delta\tau(p_k) = \delta T(p_k) - p_k \delta\Delta(p_k)$ (s).
    pub tau: Vec<f64>,
}

impl TauBranch {
    /// Creates an empty `TauBranch` header before `create_branch`.
    pub fn new(top_depth: f64, bot_depth: f64, is_p_wave: bool) -> Self {
        Self {
            top_depth,
            bot_depth,
            is_p_wave,
            max_ray_param: 0.0,
            min_turn_ray_param: 0.0,
            min_ray_param: 0.0,
            time: Vec::new(),
            dist: Vec::new(),
            tau: Vec::new(),
        }
    }

    fn create_branch(
        &mut self,
        s_mod: &SlownessModel,
        min_p_so_far: f64,
        ray_params: &[f64],
    ) -> Result<()> {
        let top_layer_num = s_mod.layer_number_below(self.top_depth, self.is_p_wave)?;
        let bot_layer_num = s_mod.layer_number_above(self.bot_depth, self.is_p_wave)?;
        let top_sl = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave);
        let bot_sl = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave);

        if top_sl.top_depth != self.top_depth || bot_sl.bot_depth != self.bot_depth {
            if top_sl.top_depth != self.top_depth
                && (top_sl.top_depth - self.top_depth).abs() < 1e-6
            {
                self.top_depth = top_sl.top_depth;
            } else if bot_sl.bot_depth != self.bot_depth
                && (bot_sl.bot_depth - self.bot_depth).abs() < 1e-6
            {
                self.bot_depth = bot_sl.bot_depth;
            } else {
                return Err(EarthSnailError::ModelError(format!(
                    "TauBranch [{}, {}] incompatible with slowness layer [{}, {}]",
                    self.top_depth, self.bot_depth, top_sl.top_depth, bot_sl.bot_depth
                )));
            }
        }

        self.max_ray_param = min_p_so_far;
        self.min_turn_ray_param = s_mod.get_min_turn_ray_param(self.bot_depth, self.is_p_wave)?;
        self.min_ray_param = s_mod.get_min_ray_param(self.bot_depth, self.is_p_wave)?;

        let n = ray_params.len();
        self.time = vec![0.0; n];
        self.dist = vec![0.0; n];
        self.tau = vec![0.0; n];

        for (i, &p) in ray_params.iter().enumerate() {
            let (t, d) = self.eval_single_p(s_mod, top_layer_num, bot_layer_num, p, false)?;
            self.time[i] = t;
            self.dist[i] = d;
            self.tau[i] = t - p * d;
        }
        Ok(())
    }

    /// Evaluates exact one-way `(time_s, dist_rad)` across this branch for a single ray parameter `p`
    /// with **zero heap allocations**.
    #[inline]
    pub fn eval_single_p(
        &self,
        s_mod: &SlownessModel,
        top_layer_num: usize,
        bot_layer_num: usize,
        p: f64,
        allow_turn_in_layer: bool,
    ) -> Result<(f64, f64)> {
        if p > self.max_ray_param {
            return Ok((0.0, 0.0));
        }
        let layers = s_mod.layers(self.is_p_wave);
        let mut sum_t = 0.0;
        let mut sum_d = 0.0;
        for (layer_num, sl) in layers
            .iter()
            .enumerate()
            .take(bot_layer_num + 1)
            .skip(top_layer_num)
        {
            if sl.top_p >= p && sl.bot_p >= p {
                let (dt, dd) = s_mod.layer_time_dist(p, layer_num, self.is_p_wave, false, false)?;
                if dt.is_finite() && dd.is_finite() {
                    sum_t += dt;
                    sum_d += dd;
                }
            } else if allow_turn_in_layer && (sl.top_p - p) * (p - sl.bot_p) > 0.0 {
                // Ray turns strictly inside this layer.
                let (dt, dd) = s_mod.layer_time_dist(p, layer_num, self.is_p_wave, false, true)?;
                if dt.is_finite() && dd.is_finite() {
                    sum_t += dt;
                    sum_d += dd;
                }
                break;
            } else {
                // Ray turns at or above the top of this layer.
                break;
            }
        }
        Ok((sum_t, sum_d))
    }

    fn insert(&mut self, ray_param: f64, s_mod: &SlownessModel, index: usize) -> Result<()> {
        let top_layer_num = s_mod.layer_number_below(self.top_depth, self.is_p_wave)?;
        let bot_layer_num = s_mod.layer_number_above(self.bot_depth, self.is_p_wave)?;
        let top_sl = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave);
        let bot_sl = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave);
        if top_sl.top_depth != self.top_depth || bot_sl.bot_depth != self.bot_depth {
            return Err(EarthSnailError::ModelError(
                "TauBranch::insert depths incompatible with SlownessModel".into(),
            ));
        }

        let mut new_time = 0.0;
        let mut new_dist = 0.0;
        if top_sl.bot_p >= ray_param && top_sl.top_p >= ray_param {
            for layer_num in top_layer_num..=bot_layer_num {
                let sl = s_mod.get_slowness_layer(layer_num, self.is_p_wave);
                if sl.bot_p >= ray_param {
                    let (dt, dd) =
                        s_mod.layer_time_dist(ray_param, layer_num, self.is_p_wave, true, false)?;
                    new_time += dt;
                    new_dist += dd;
                } else {
                    break;
                }
            }
        }
        let new_tau = new_time - ray_param * new_dist;
        self.time.insert(index, new_time);
        self.dist.insert(index, new_dist);
        self.tau.insert(index, new_tau);
        Ok(())
    }

    fn difference(
        &self,
        top_branch: &TauBranch,
        index_p: Option<usize>,
        index_s: Option<usize>,
        s_mod: &SlownessModel,
        ray_params: &[f64],
    ) -> Result<TauBranch> {
        let mut top_layer_num = s_mod.layer_number_below(top_branch.bot_depth, self.is_p_wave)?;
        let mut bot_layer_num = s_mod.layer_number_below(self.bot_depth, self.is_p_wave)?;
        let bot_sl = s_mod.get_slowness_layer(bot_layer_num, self.is_p_wave);
        if bot_sl.top_depth == self.bot_depth && bot_sl.bot_depth > self.bot_depth {
            bot_layer_num = bot_layer_num.saturating_sub(1);
        }
        let top_sl = s_mod.get_slowness_layer(top_layer_num, self.is_p_wave);
        if top_sl.top_depth != top_branch.bot_depth && top_layer_num > 0 {
            top_layer_num = s_mod.layer_number_below(top_branch.bot_depth, self.is_p_wave)?;
        }

        let eff_index_s = match (index_s, index_p) {
            (Some(is_), Some(ip)) if is_ == ip => None,
            (s, _) => s,
        };

        let mut bot_branch = TauBranch::new(top_branch.bot_depth, self.bot_depth, self.is_p_wave);
        bot_branch.max_ray_param = top_branch.min_ray_param;
        bot_branch.min_turn_ray_param = self.min_turn_ray_param;
        bot_branch.min_ray_param = self.min_ray_param;

        let new_len = ray_params.len();
        bot_branch.time = Vec::with_capacity(new_len);
        bot_branch.dist = Vec::with_capacity(new_len);
        bot_branch.tau = Vec::with_capacity(new_len);

        let mut orig_idx = 0usize;
        for (new_idx, &rp) in ray_params.iter().enumerate() {
            if Some(new_idx) == eff_index_s || Some(new_idx) == index_p {
                let (t, d) =
                    bot_branch.eval_single_p(s_mod, top_layer_num, bot_layer_num, rp, false)?;
                bot_branch.time.push(t);
                bot_branch.dist.push(d);
                bot_branch.tau.push(t - rp * d);
            } else {
                bot_branch
                    .time
                    .push(self.time[orig_idx] - top_branch.time[new_idx]);
                bot_branch
                    .dist
                    .push(self.dist[orig_idx] - top_branch.dist[new_idx]);
                bot_branch
                    .tau
                    .push(self.tau[orig_idx] - top_branch.tau[new_idx]);
                orig_idx += 1;
            }
        }

        Ok(bot_branch)
    }

    /// Traces incremental `(p, dt, d_dist, end_depth)` points through this branch for `ray_param`,
    /// matching `TauBranch.path` in IASPEI TauP.
    pub fn path(
        &self,
        ray_param: f64,
        downgoing: bool,
        s_mod: &SlownessModel,
        out: &mut Vec<TimeDist>,
    ) -> Result<()> {
        if ray_param > self.max_ray_param {
            return Ok(());
        }
        let top_layer_num = s_mod.layer_number_below(self.top_depth, self.is_p_wave)?;
        let bot_layer_num = s_mod.layer_number_above(self.bot_depth, self.is_p_wave)?;

        if downgoing {
            let mut next_layer_num = top_layer_num;
            for layer_num in top_layer_num..=bot_layer_num {
                let sl = s_mod.get_slowness_layer(layer_num, self.is_p_wave);
                if sl.bot_p >= ray_param {
                    next_layer_num = layer_num + 1;
                    if sl.top_depth != sl.bot_depth {
                        let (dt, dd) = s_mod.layer_time_dist(
                            ray_param,
                            layer_num,
                            self.is_p_wave,
                            true,
                            false,
                        )?;
                        out.push(TimeDist {
                            p: ray_param,
                            time: dt,
                            dist: dd,
                            depth: sl.bot_depth,
                        });
                    }
                } else {
                    next_layer_num = layer_num;
                    break;
                }
            }

            if next_layer_num <= bot_layer_num {
                let sl = s_mod.get_slowness_layer(next_layer_num, self.is_p_wave);
                if sl.top_depth != sl.bot_depth
                    && (sl.top_p - ray_param) * (ray_param - sl.bot_p) >= 0.0
                {
                    let turn_depth = bullen_depth_for(sl, ray_param, s_mod.radius_of_planet)?;
                    let turn_sl = SlownessLayer {
                        top_p: sl.top_p,
                        top_depth: sl.top_depth,
                        bot_p: ray_param,
                        bot_depth: turn_depth,
                    };
                    let (dt, dd) =
                        bullen_radial_slowness(turn_sl, ray_param, s_mod.radius_of_planet, true)?;
                    out.push(TimeDist {
                        p: ray_param,
                        time: dt,
                        dist: dd,
                        depth: turn_depth,
                    });
                }
            }
        } else {
            // Upgoing: scan from bot_layer_num up to top_layer_num.
            let len = bot_layer_num - top_layer_num + 1;
            let mut first_unmasked = 0usize;
            for k in 0..len.saturating_sub(1) {
                let layer_num = bot_layer_num - k;
                let sl = s_mod.get_slowness_layer(layer_num, self.is_p_wave);
                if sl.top_p <= ray_param || sl.top_depth == sl.bot_depth {
                    first_unmasked = k + 1;
                } else {
                    break;
                }
            }

            let first_layer_num = bot_layer_num - first_unmasked;
            let sl2 = s_mod.get_slowness_layer(first_layer_num, self.is_p_wave);
            let mut start_regular_k = first_unmasked;
            if sl2.bot_p < ray_param && sl2.top_depth != sl2.bot_depth {
                let turn_depth = bullen_depth_for(sl2, ray_param, s_mod.radius_of_planet)?;
                let turn_sl = SlownessLayer {
                    top_p: sl2.top_p,
                    top_depth: sl2.top_depth,
                    bot_p: ray_param,
                    bot_depth: turn_depth,
                };
                let (dt, dd) =
                    bullen_radial_slowness(turn_sl, ray_param, s_mod.radius_of_planet, true)?;
                out.push(TimeDist {
                    p: ray_param,
                    time: dt,
                    dist: dd,
                    depth: turn_sl.top_depth,
                });
                start_regular_k += 1;
            }

            for k in start_regular_k..len {
                let layer_num = bot_layer_num - k;
                let sl = s_mod.get_slowness_layer(layer_num, self.is_p_wave);
                if sl.top_depth != sl.bot_depth {
                    let (dt, dd) =
                        s_mod.layer_time_dist(ray_param, layer_num, self.is_p_wave, true, false)?;
                    out.push(TimeDist {
                        p: ray_param,
                        time: dt,
                        dist: dd,
                        depth: sl.top_depth,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Complete compiled spherical $\tau(p)$ model with P and S `TauBranch` tables.
#[derive(Debug, Clone)]
pub struct TauModel {
    /// Focal depth (km) for which this `TauModel` is split (`0.0` for base surface model).
    pub source_depth: f64,
    /// Planetary outer radius in kilometers.
    pub radius_of_planet: f64,
    /// Spherical model flag (`true`).
    pub is_spherical: bool,
    /// Strictly decreasing non-LVZ ray-parameter grid (`s/rad`) ending at `0.0`.
    pub ray_params: Vec<f64>,
    /// `[p_branches, s_branches]`, each of length `num_branches()`.
    pub tau_branches: [Vec<TauBranch>; 2],
    /// Underlying coupled P/S slowness model.
    pub s_mod: SlownessModel,
    /// Branch index having `source_depth` at its top boundary.
    pub source_branch: usize,
    /// Optional branch index having `conrad_depth` at its top boundary (if crust has Conrad jump).
    pub conrad_branch: Option<usize>,
    /// Conrad discontinuity depth in kilometers (`None` if single-layer crust).
    pub conrad_depth: Option<f64>,
    /// Branch index having `moho_depth` at its top boundary.
    pub moho_branch: usize,
    /// Moho discontinuity depth in kilometers.
    pub moho_depth: f64,
    /// Branch index having `cmb_depth` at its top boundary.
    pub cmb_branch: usize,
    /// Core-Mantle Boundary depth in kilometers.
    pub cmb_depth: f64,
    /// Branch index having `iocb_depth` at its top boundary.
    pub iocb_branch: usize,
    /// Inner-Core Boundary depth in kilometers.
    pub iocb_depth: f64,
    /// Artificial split depths (such as non-discontinuity `source_depth` or `receiver_depth`)
    /// where phase conversions/reflections must not snap.
    pub no_discon_depths: Vec<f64>,
}

impl TauModel {
    /// Compiles the built-in `iasp91` `TauModel`.
    pub fn iasp91() -> Result<Self> {
        Self::from_velocity_model(VelocityModel::iasp91())
    }

    /// Compiles the built-in `ak135` `TauModel`.
    pub fn ak135() -> Result<Self> {
        Self::from_velocity_model(VelocityModel::ak135())
    }

    /// Compiles the built-in `prem` `TauModel`.
    pub fn prem() -> Result<Self> {
        Self::from_velocity_model(VelocityModel::prem())
    }

    /// Compiles a built-in `TauModel` by case-insensitive name (`"iasp91"`, `"ak135"`, `"prem"`).
    pub fn by_name(name: &str) -> Result<Self> {
        Self::from_velocity_model(VelocityModel::by_name(name)?)
    }

    /// Compiles a surface-source `TauModel` from a `VelocityModel`.
    pub fn from_velocity_model(v_mod: VelocityModel) -> Result<Self> {
        let s_mod = SlownessModel::from_velocity_model(v_mod)?;
        Self::from_slowness_model(s_mod)
    }

    /// Compiles a surface-source `TauModel` from a `SlownessModel`.
    pub fn from_slowness_model(s_mod: SlownessModel) -> Result<Self> {
        if s_mod.num_layers(true) == 0 || s_mod.num_layers(false) == 0 {
            return Err(EarthSnailError::ModelError(
                "SlownessModel has zero layers".into(),
            ));
        }
        s_mod.validate()?;
        let radius_of_planet = s_mod.radius_of_planet;
        let num_branches = s_mod.critical_depths.len() - 1;

        // Build strictly decreasing ray_params outside high-slowness shadow zones.
        let mut ray_params =
            Vec::with_capacity(2 * s_mod.num_layers(false) + s_mod.critical_depths.len());
        let mut min_p_so_far = s_mod.s_layers[0].top_p;
        ray_params.push(min_p_so_far);
        for sl in &s_mod.s_layers {
            if sl.top_p < min_p_so_far {
                ray_params.push(sl.top_p);
                min_p_so_far = sl.top_p;
            }
            if sl.bot_p < min_p_so_far {
                ray_params.push(sl.bot_p);
                min_p_so_far = sl.bot_p;
            }
        }

        let mut p_branches = Vec::with_capacity(num_branches);
        let mut s_branches = Vec::with_capacity(num_branches);

        for (wave_idx, is_p_wave) in [true, false].into_iter().enumerate() {
            let mut min_p = s_mod.get_slowness_layer(0, is_p_wave).top_p;
            let target_vec = if wave_idx == 0 {
                &mut p_branches
            } else {
                &mut s_branches
            };
            for crit_num in 0..num_branches {
                let top_cd = s_mod.critical_depths[crit_num];
                let bot_cd = s_mod.critical_depths[crit_num + 1];
                let top_crit_layer = if is_p_wave {
                    top_cd.p_layer_num
                } else {
                    top_cd.s_layer_num
                };
                let bot_crit_layer = (if is_p_wave {
                    bot_cd.p_layer_num
                } else {
                    bot_cd.s_layer_num
                })
                .saturating_sub(1);

                let mut tb = TauBranch::new(top_cd.depth(), bot_cd.depth(), is_p_wave);
                tb.create_branch(&s_mod, min_p, &ray_params)?;
                target_vec.push(tb);

                let top_sl = s_mod.get_slowness_layer(top_crit_layer, is_p_wave);
                let bot_sl = s_mod.get_slowness_layer(bot_crit_layer, is_p_wave);
                min_p = min_p.min(top_sl.top_p.min(bot_sl.bot_p));
                let above_bot = s_mod.get_slowness_layer(
                    s_mod.layer_number_above(bot_cd.depth(), is_p_wave)?,
                    is_p_wave,
                );
                min_p = min_p.min(above_bot.bot_p);
            }
        }

        let mut moho_branch = 0usize;
        let mut cmb_branch = 0usize;
        let mut iocb_branch = 0usize;
        let mut best_moho = 1e300_f64;
        let mut best_cmb = 1e300_f64;
        let mut best_iocb = 1e300_f64;

        for (b_num, tb) in p_branches.iter().enumerate() {
            let dm = (tb.top_depth - s_mod.v_mod.moho_depth).abs();
            if dm <= best_moho {
                moho_branch = b_num;
                best_moho = dm;
            }
            let dc = (tb.top_depth - s_mod.v_mod.cmb_depth).abs();
            if dc < best_cmb {
                cmb_branch = b_num;
                best_cmb = dc;
            }
            let di = (tb.top_depth - s_mod.v_mod.iocb_depth).abs();
            if di < best_iocb {
                iocb_branch = b_num;
                best_iocb = di;
            }
        }

        let last_tb = &p_branches[num_branches - 1];
        let mut cmb_depth = p_branches[cmb_branch].top_depth;
        let mut iocb_depth = p_branches[iocb_branch].top_depth;
        if (last_tb.bot_depth - s_mod.v_mod.cmb_depth).abs() < best_cmb {
            cmb_branch = num_branches;
            cmb_depth = last_tb.bot_depth;
        }
        if (last_tb.bot_depth - s_mod.v_mod.iocb_depth).abs() < best_iocb {
            iocb_branch = num_branches;
            iocb_depth = last_tb.bot_depth;
        }
        let moho_depth = p_branches[moho_branch].top_depth;

        let mut conrad_branch = None;
        let mut conrad_depth = None;
        if let Some(cd_target) = s_mod.v_mod.conrad_depth {
            let mut best_cd = 1e300_f64;
            for (b_num, tb) in p_branches.iter().enumerate().take(moho_branch) {
                if tb.top_depth > 0.0 {
                    let err = (tb.top_depth - cd_target).abs();
                    if err < best_cd {
                        best_cd = err;
                        conrad_branch = Some(b_num);
                        conrad_depth = Some(tb.top_depth);
                    }
                }
            }
        }

        Ok(Self {
            source_depth: 0.0,
            radius_of_planet,
            is_spherical: true,
            ray_params,
            tau_branches: [p_branches, s_branches],
            s_mod,
            source_branch: 0,
            conrad_branch,
            conrad_depth,
            moho_branch,
            moho_depth,
            cmb_branch,
            cmb_depth,
            iocb_branch,
            iocb_depth,
            no_discon_depths: Vec::new(),
        })
    }

    /// Number of `TauBranch` intervals in this model.
    #[inline]
    pub fn num_branches(&self) -> usize {
        self.tau_branches[0].len()
    }

    /// Accesses `(branch_num, is_p_wave)`.
    #[inline]
    pub fn get_tau_branch(&self, branch_num: usize, is_p_wave: bool) -> &TauBranch {
        if is_p_wave {
            &self.tau_branches[0][branch_num]
        } else {
            &self.tau_branches[1][branch_num]
        }
    }

    /// Returns all branch boundary depths from `0.0` to `radius_of_planet`.
    pub fn branch_depths(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.num_branches() + 1);
        if let Some(first) = self.tau_branches[0].first() {
            out.push(first.top_depth);
            for tb in &self.tau_branches[0] {
                out.push(tb.bot_depth);
            }
        }
        out
    }

    /// Finds the branch having `depth` at its top boundary or strictly containing `depth`
    /// (with center of planet mapped to the deepest branch).
    pub fn find_branch(&self, depth: f64) -> Result<usize> {
        for (i, tb) in self.tau_branches[0].iter().enumerate() {
            if tb.top_depth <= depth && depth < tb.bot_depth {
                return Ok(i);
            }
        }
        let n = self.num_branches();
        if n > 0 && self.tau_branches[0][n - 1].bot_depth == depth {
            return Ok(n - 1);
        }
        Err(EarthSnailError::ModelError(format!(
            "no TauBranch contains depth {depth} km"
        )))
    }

    /// Returns a depth-corrected `TauModel` with the source placed at `depth_km`
    /// (`TauModel::depth_correct` in Buland & Chapman 1983).
    pub fn with_source_depth(&self, depth_km: f64) -> Result<Self> {
        self.depth_correct(depth_km)
    }

    /// Computes a new `TauModel` split at `depth_km` with `source_depth` and `source_branch` set.
    pub fn depth_correct(&self, depth_km: f64) -> Result<Self> {
        if self.source_depth != 0.0 {
            return Err(EarthSnailError::ModelError(
                "depth_correct must be called on a surface-source TauModel (source_depth == 0)"
                    .into(),
            ));
        }
        if !depth_km.is_finite() || depth_km < 0.0 || depth_km > self.radius_of_planet {
            return Err(EarthSnailError::InvalidParameter(format!(
                "source depth {depth_km} km outside [0, {}]",
                self.radius_of_planet
            )));
        }
        let mut corrected = self.split_branch(depth_km)?;
        corrected.source_depth = depth_km;
        corrected.source_branch = corrected.find_branch(depth_km)?;
        Ok(corrected)
    }

    /// Splits the branch containing `depth` into an upper and lower `TauBranch` (or returns a
    /// clone if `depth` is already an existing branch boundary).
    pub fn split_branch(&self, depth: f64) -> Result<Self> {
        if !depth.is_finite() || depth < 0.0 || depth > self.radius_of_planet {
            return Err(EarthSnailError::InvalidParameter(format!(
                "split depth {depth} km outside [0, {}]",
                self.radius_of_planet
            )));
        }
        for tb in &self.tau_branches[0] {
            if tb.top_depth == depth || tb.bot_depth == depth {
                return Ok(self.clone());
            }
        }

        let mut index_p: Option<usize> = None;
        let mut p_wave_ray_param = -1.0_f64;
        let mut index_s: Option<usize> = None;
        let mut s_wave_ray_param = -1.0_f64;
        let mut out_s_mod = self.s_mod.clone();
        let mut out_ray_params = self.ray_params.clone();

        // Split S wave first (since eta_S > eta_P), then P wave.
        for is_p_wave in [false, true] {
            let split_info = out_s_mod.split_layer(depth, is_p_wave)?;
            out_s_mod = split_info.s_mod;
            if split_info.needed_split && !split_info.moved_sample {
                let new_rp = split_info.ray_param;
                let mut insert_pos = None;
                for idx in 0..out_ray_params.len().saturating_sub(1) {
                    if out_ray_params[idx] > new_rp && new_rp > out_ray_params[idx + 1] {
                        insert_pos = Some(idx + 1);
                        break;
                    }
                }
                if let Some(pos) = insert_pos {
                    out_ray_params.insert(pos, new_rp);
                    if is_p_wave {
                        index_p = Some(pos);
                        p_wave_ray_param = new_rp;
                    } else {
                        index_s = Some(pos);
                        s_wave_ray_param = new_rp;
                    }
                }
            }
        }

        let branch_to_split = self.find_branch(depth)?;
        let old_n = self.num_branches();
        let mut new_p_branches = Vec::with_capacity(old_n + 1);
        let mut new_s_branches = Vec::with_capacity(old_n + 1);

        for i in 0..branch_to_split {
            let mut pb = self.tau_branches[0][i].clone();
            let mut sb = self.tau_branches[1][i].clone();
            if let Some(is_) = index_s {
                pb.insert(s_wave_ray_param, &out_s_mod, is_)?;
                sb.insert(s_wave_ray_param, &out_s_mod, is_)?;
            }
            if let Some(ip) = index_p {
                pb.insert(p_wave_ray_param, &out_s_mod, ip)?;
                sb.insert(p_wave_ray_param, &out_s_mod, ip)?;
            }
            new_p_branches.push(pb);
            new_s_branches.push(sb);
        }

        for (p_or_s, branch_vec) in self.tau_branches.iter().enumerate() {
            let is_p = p_or_s == 0;
            let orig_tb = &branch_vec[branch_to_split];
            let mut top_half = TauBranch::new(orig_tb.top_depth, depth, is_p);
            top_half.create_branch(&out_s_mod, orig_tb.max_ray_param, &out_ray_params)?;
            let bot_half =
                orig_tb.difference(&top_half, index_p, index_s, &out_s_mod, &out_ray_params)?;
            if is_p {
                new_p_branches.push(top_half);
                new_p_branches.push(bot_half);
            } else {
                new_s_branches.push(top_half);
                new_s_branches.push(bot_half);
            }
        }

        for i in (branch_to_split + 1)..old_n {
            let mut pb = self.tau_branches[0][i].clone();
            let mut sb = self.tau_branches[1][i].clone();
            if let Some(is_) = index_s {
                pb.insert(s_wave_ray_param, &out_s_mod, is_)?;
                sb.insert(s_wave_ray_param, &out_s_mod, is_)?;
            }
            if let Some(ip) = index_p {
                pb.insert(p_wave_ray_param, &out_s_mod, ip)?;
                sb.insert(p_wave_ray_param, &out_s_mod, ip)?;
            }
            new_p_branches.push(pb);
            new_s_branches.push(sb);
        }

        let mut out_source_branch = self.source_branch;
        if self.source_depth > depth {
            out_source_branch += 1;
        }
        let out_conrad_branch = self.conrad_branch.map(|cb| {
            if self.conrad_depth.unwrap_or(0.0) > depth {
                cb + 1
            } else {
                cb
            }
        });
        let mut out_moho_branch = self.moho_branch;
        if self.moho_depth > depth {
            out_moho_branch += 1;
        }
        let mut out_cmb_branch = self.cmb_branch;
        if self.cmb_depth > depth {
            out_cmb_branch += 1;
        }
        let mut out_iocb_branch = self.iocb_branch;
        if self.iocb_depth > depth {
            out_iocb_branch += 1;
        }

        let mut no_discon_depths = self.no_discon_depths.clone();
        no_discon_depths.push(depth);

        Ok(Self {
            source_depth: self.source_depth,
            radius_of_planet: self.radius_of_planet,
            is_spherical: self.is_spherical,
            ray_params: out_ray_params,
            tau_branches: [new_p_branches, new_s_branches],
            s_mod: out_s_mod,
            source_branch: out_source_branch,
            conrad_branch: out_conrad_branch,
            conrad_depth: self.conrad_depth,
            moho_branch: out_moho_branch,
            moho_depth: self.moho_depth,
            cmb_branch: out_cmb_branch,
            cmb_depth: self.cmb_depth,
            iocb_branch: out_iocb_branch,
            iocb_depth: self.iocb_depth,
            no_discon_depths,
        })
    }
}

#[cfg(test)]
pub(crate) fn test_iasp91_model() -> TauModel {
    if cfg!(miri) {
        const COMPACT_EARTH_ND: &str = "\
0.0    5.80  3.36  2.72
20.0   5.80  3.36  2.72
conrad
20.0   6.50  3.75  2.92
35.0   6.50  3.75  2.92
moho
35.0   8.04  4.47  3.32
410.0  9.03  4.87  3.54
410.0  9.36  5.07  3.72
660.0  10.20 5.60  3.99
660.0  10.79 5.95  4.38
2889.0 13.69 7.30  5.56
outer-core
2889.0 8.00  0.00  9.90
5153.9 10.26 0.00  12.14
inner-core
5153.9 11.12 3.50  12.70
6371.0 11.24 3.56  13.01
";
        let vm = VelocityModel::from_nd_str("compact_iasp91", COMPACT_EARTH_ND).unwrap();
        let cfg = SlownessModelConfig {
            min_delta_p: 5.0,
            max_delta_p: 150.0,
            max_depth_interval: 1200.0,
            max_range_interval_rad: 45.0 * PI / 180.0,
            max_interp_error: 2.0,
            allow_inner_core_s: true,
            slowness_tolerance: DEFAULT_SLOWNESS_TOL,
        };
        let sm = SlownessModel::with_config(vm, cfg).unwrap();
        TauModel::from_slowness_model(sm).unwrap()
    } else {
        TauModel::iasp91().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_models_compile_with_exact_ray_param_counts_and_lvz_detection() {
        if !cfg!(miri) {
            let iasp = TauModel::iasp91().unwrap();
            assert_eq!(iasp.ray_params.len(), 518);
            assert_eq!(iasp.moho_depth, 35.0);
            assert_eq!(iasp.cmb_depth, 2889.0);
            assert_eq!(iasp.iocb_depth, 5153.9);
            assert_eq!(iasp.conrad_depth, Some(20.0));

            let ak = TauModel::ak135().unwrap();
            assert_eq!(ak.ray_params.len(), 516);
            assert_eq!(ak.moho_depth, 35.0);
            assert_eq!(ak.cmb_depth, 2891.5);
            assert_eq!(ak.iocb_depth, 5153.5);

            let prem = TauModel::prem().unwrap();
            assert!(
                (567..=572).contains(&prem.ray_params.len()),
                "unexpected prem ray_params count: {}",
                prem.ray_params.len()
            );
            assert_eq!(prem.moho_depth, 24.4);
            assert_eq!(prem.cmb_depth, 2891.0);
            assert_eq!(prem.iocb_depth, 5149.5);
            assert!(!prem.s_mod.high_slowness_layer_depths_p.is_empty());
            assert!(!prem.s_mod.high_slowness_layer_depths_s.is_empty());
        } else {
            const COMPACT_LVZ_ND: &str = "\
0.0    6.00  3.50  2.70
24.4   6.80  3.90  2.90
moho
24.4   8.10  4.50  3.38
220.0  7.80  4.30  3.36
220.0  8.55  4.65  3.44
2891.0 13.70 7.26  5.56
outer-core
2891.0 8.06  0.00  9.90
5149.5 10.35 0.00  12.16
inner-core
5149.5 11.03 3.50  12.76
6371.0 11.26 3.66  13.08
";
            let vm = VelocityModel::from_nd_str("compact_prem", COMPACT_LVZ_ND).unwrap();
            let cfg = SlownessModelConfig {
                min_delta_p: 5.0,
                max_delta_p: 150.0,
                max_depth_interval: 1200.0,
                max_range_interval_rad: 45.0 * PI / 180.0,
                max_interp_error: 2.0,
                allow_inner_core_s: true,
                slowness_tolerance: DEFAULT_SLOWNESS_TOL,
            };
            let prem_sm = SlownessModel::with_config(vm, cfg).unwrap();
            let prem = TauModel::from_slowness_model(prem_sm).unwrap();
            assert_eq!(prem.moho_depth, 24.4);
            assert!(!prem.s_mod.high_slowness_layer_depths_p.is_empty());
            assert!(!prem.s_mod.high_slowness_layer_depths_s.is_empty());
        }

        // Constant-slowness layer (alpha = 1, top_p == bot_p) analytical check:
        let const_eta = SlownessLayer {
            top_p: 500.0,
            top_depth: 100.0,
            bot_p: 500.0,
            bot_depth: 300.0,
        };
        let (dt, dd) = bullen_radial_slowness(const_eta, 300.0, 6371.0, true).unwrap();
        let tau = dt - 300.0 * dd;
        let lr = math::ln((6371.0 - 100.0) / (6371.0 - 300.0));
        // q = sqrt(500^2 - 300^2) = 400 -> tau = q * lr = 400 * lr!
        let tol = if cfg!(miri) { 1e-9 } else { 1e-12 };
        assert!((tau - 400.0 * lr).abs() < tol);
    }

    #[test]
    fn legendre_tau_derivative_matches_minus_delta_on_bullen_layer() {
        let sl = SlownessLayer {
            top_p: 780.0,
            top_depth: 35.0,
            bot_p: 720.0,
            bot_depth: 210.0,
        };
        let p = 512.345;
        let dp = 1e-3;
        let (t0, d0) = bullen_radial_slowness(sl, p, 6371.0, true).unwrap();
        let (tp, dp_d) = bullen_radial_slowness(sl, p + dp, 6371.0, true).unwrap();
        let (tm, dm_d) = bullen_radial_slowness(sl, p - dp, 6371.0, true).unwrap();
        let tau_p = tp - (p + dp) * dp_d;
        let tau_m = tm - (p - dp) * dm_d;
        let dtau_dp = (tau_p - tau_m) / (2.0 * dp);
        let tau0 = t0 - p * d0;
        let tol = if cfg!(miri) { 1e-9 } else { 1e-11 };
        assert!((t0 - (tau0 + p * d0)).abs() < 1e-13);
        assert!((dtau_dp + d0).abs() < tol, "dtau/dp={dtau_dp}, -d0={}", -d0);
    }
}
