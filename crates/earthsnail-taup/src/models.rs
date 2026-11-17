//! Spherical planetary velocity models (`iasp91`, `ak135`, `prem`), runtime `.nd` and `.tvel`
//! text model compilers, and Earth-flattening transformations.

use crate::model_tables::{AK135_KNOTS, IASP91_KNOTS, PREM_KNOTS};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use earthsnail_core::{math, EarthSnailError, Result};

/// Default compressional attenuation factor $Q_P$ when omitted by `.tvel` or `.nd` inputs.
pub const DEFAULT_QP: f64 = 1000.0;
/// Default shear attenuation factor $Q_S$ when omitted by `.tvel` or `.nd` inputs.
pub const DEFAULT_QS: f64 = 2000.0;
/// Default density ($\text{g/cm}^3$) when omitted by 3-column `.nd` inputs.
pub const DEFAULT_DENSITY: f64 = 2.6;
/// Default initial Moho discontinuity search depth in kilometers (`35.0 km`).
pub const DEFAULT_MOHO_KM: f64 = 35.0;
/// Default initial Core-Mantle Boundary search depth in kilometers (`2889.0 km`).
pub const DEFAULT_CMB_KM: f64 = 2889.0;
/// Default initial Inner-Core Boundary search depth in kilometers (`5153.9 km`).
pub const DEFAULT_IOCB_KM: f64 = 5153.9;

/// Body-wave polarization / propagation mode (`P` compressional or `S` shear).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WaveType {
    /// Compressional (`P`, `K`, `I`) wave.
    P,
    /// Shear (`S`, `J`) wave.
    S,
}

impl WaveType {
    /// Returns `true` if `self == WaveType::P`.
    #[inline]
    pub const fn is_p(self) -> bool {
        matches!(self, Self::P)
    }

    /// Converts a boolean `is_p_wave` flag to `WaveType`.
    #[inline]
    pub const fn from_is_p(is_p_wave: bool) -> Self {
        if is_p_wave {
            Self::P
        } else {
            Self::S
        }
    }
}

/// Material property selector for evaluating a `VelocityLayer` or `VelocityModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialProperty {
    /// Compressional wave speed $v_P$ ($\text{km/s}$).
    PVelocity,
    /// Shear wave speed $v_S$ ($\text{km/s}$).
    SVelocity,
    /// Mass density $\rho$ ($\text{g/cm}^3$).
    Density,
}

impl MaterialProperty {
    /// Maps a `WaveType` to its corresponding velocity `MaterialProperty`.
    #[inline]
    pub const fn from_wave(wave: WaveType) -> Self {
        match wave {
            WaveType::P => Self::PVelocity,
            WaveType::S => Self::SVelocity,
        }
    }
}

/// Spherical shell between `top_depth` and `bot_depth` (in km) with linear depth-interpolated
/// elastic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityLayer {
    /// Depth at top of layer in kilometers.
    pub top_depth: f64,
    /// Depth at bottom of layer in kilometers (`>= top_depth`).
    pub bot_depth: f64,
    /// Compressional velocity at `top_depth` ($\text{km/s}$).
    pub top_p_velocity: f64,
    /// Compressional velocity at `bot_depth` ($\text{km/s}$).
    pub bot_p_velocity: f64,
    /// Shear velocity at `top_depth` ($\text{km/s}$; `0.0` in fluid layers).
    pub top_s_velocity: f64,
    /// Shear velocity at `bot_depth` ($\text{km/s}$; `0.0` in fluid layers).
    pub bot_s_velocity: f64,
    /// Density at `top_depth` ($\text{g/cm}^3$).
    pub top_density: f64,
    /// Density at `bot_depth` ($\text{g/cm}^3$).
    pub bot_density: f64,
    /// Compressional quality factor $Q_P$ at `top_depth`.
    pub top_qp: f64,
    /// Compressional quality factor $Q_P$ at `bot_depth`.
    pub bot_qp: f64,
    /// Shear quality factor $Q_S$ at `top_depth`.
    pub top_qs: f64,
    /// Shear quality factor $Q_S$ at `bot_depth`.
    pub bot_qs: f64,
}

impl VelocityLayer {
    /// Evaluates `prop` at the top boundary of this layer.
    #[inline]
    pub const fn evaluate_at_top(&self, prop: MaterialProperty) -> f64 {
        match prop {
            MaterialProperty::PVelocity => self.top_p_velocity,
            MaterialProperty::SVelocity => self.top_s_velocity,
            MaterialProperty::Density => self.top_density,
        }
    }

    /// Evaluates `prop` at the bottom boundary of this layer.
    #[inline]
    pub const fn evaluate_at_bottom(&self, prop: MaterialProperty) -> f64 {
        match prop {
            MaterialProperty::PVelocity => self.bot_p_velocity,
            MaterialProperty::SVelocity => self.bot_s_velocity,
            MaterialProperty::Density => self.bot_density,
        }
    }

    /// Linearly interpolates `prop` at `depth` within `[top_depth, bot_depth]`.
    #[inline]
    pub fn evaluate_at(&self, depth: f64, prop: MaterialProperty) -> f64 {
        let thick = self.bot_depth - self.top_depth;
        let top_val = self.evaluate_at_top(prop);
        if thick == 0.0 {
            return top_val;
        }
        let bot_val = self.evaluate_at_bottom(prop);
        let slope = (bot_val - top_val) / thick;
        slope * (depth - self.top_depth) + top_val
    }

    /// Layer thickness `bot_depth - top_depth` in kilometers.
    #[inline]
    pub fn thickness_km(&self) -> f64 {
        self.bot_depth - self.top_depth
    }
}

/// Spherically symmetric planetary seismic velocity model.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocityModel {
    /// Model identifier (e.g. `"iasp91"`, `"ak135"`, `"prem"`).
    pub model_name: String,
    /// Reference outer planetary radius $R_E$ in kilometers (`6371.0 km` for Earth).
    pub radius_of_planet: f64,
    /// Minimum radius in kilometers (`0.0` for whole-planet models).
    pub min_radius: f64,
    /// Maximum radius in kilometers (`radius_of_planet`).
    pub max_radius: f64,
    /// Optional intracrustal Conrad discontinuity depth in kilometers (`20.0 km` in `iasp91`/`ak135`,
    /// `15.0 km` in `prem`), enabling `Pb`/`Sb` lower-crustal phase interpretation.
    pub conrad_depth: Option<f64>,
    /// Crust-mantle boundary (Moho) depth in kilometers.
    pub moho_depth: f64,
    /// Core-mantle boundary (CMB) depth in kilometers.
    pub cmb_depth: f64,
    /// Inner-core / outer-core boundary (ICB / IOCB) depth in kilometers.
    pub iocb_depth: f64,
    /// Whether the model represents a spherical planet (`true`).
    pub is_spherical: bool,
    /// Contiguous non-zero-thickness spherical velocity shells from surface (`0.0`) to center (`R_E`).
    pub layers: Vec<VelocityLayer>,
}

impl VelocityModel {
    /// Constructs the built-in zero-I/O `iasp91` model (Kennett & Engdahl 1991).
    pub fn iasp91() -> Self {
        Self::from_const_knots("iasp91", IASP91_KNOTS, Some(20.0), 35.0, 2889.0, 5153.9)
    }

    /// Constructs the built-in zero-I/O `ak135` model (Kennett, Engdahl & Buland 1995).
    pub fn ak135() -> Self {
        Self::from_const_knots("ak135", AK135_KNOTS, Some(20.0), 35.0, 2891.5, 5153.5)
    }

    /// Constructs the built-in zero-I/O isotropic 1s `prem` model (Dziewonski & Anderson 1981).
    pub fn prem() -> Self {
        Self::from_const_knots("prem", PREM_KNOTS, Some(15.0), 24.4, 2891.0, 5149.5)
    }

    /// Looks up a built-in spherical Earth model by case-insensitive name (`"iasp91"`, `"ak135"`, `"prem"`).
    pub fn by_name(name: &str) -> Result<Self> {
        let lower = name.trim().to_ascii_lowercase();
        match lower.as_str() {
            "iasp91" => Ok(Self::iasp91()),
            "ak135" => Ok(Self::ak135()),
            "prem" => Ok(Self::prem()),
            _ => Err(EarthSnailError::ModelError(format!(
                "unknown built-in velocity model '{name}' (expected 'iasp91', 'ak135', or 'prem')"
            ))),
        }
    }

    fn from_const_knots(
        name: &str,
        knots: &[[f64; 6]],
        conrad_depth: Option<f64>,
        moho_depth: f64,
        cmb_depth: f64,
        iocb_depth: f64,
    ) -> Self {
        let mut layers = Vec::with_capacity(knots.len().saturating_sub(1));
        for pair in knots.windows(2) {
            let top = pair[0];
            let bot = pair[1];
            if top[0] == bot[0] {
                continue;
            }
            layers.push(VelocityLayer {
                top_depth: top[0],
                bot_depth: bot[0],
                top_p_velocity: top[1],
                bot_p_velocity: bot[1],
                top_s_velocity: top[2],
                bot_s_velocity: bot[2],
                top_density: top[3],
                bot_density: bot[3],
                top_qp: top[4],
                bot_qp: bot[4],
                top_qs: top[5],
                bot_qs: bot[5],
            });
        }
        let radius_of_planet = knots.last().map_or(6371.0, |k| k[0]);
        Self {
            model_name: name.to_string(),
            radius_of_planet,
            min_radius: 0.0,
            max_radius: radius_of_planet,
            conrad_depth,
            moho_depth,
            cmb_depth,
            iocb_depth,
            is_spherical: true,
            layers,
        }
    }

    /// Parses a TauP `.nd` (named discontinuity) ASCII velocity model.
    ///
    /// Supports `#` line comments, `/* ... */` inline/block comments, 3/4/6-column knot rows
    /// (`depth vp vs [rho [qp qs]]`), and named boundaries (`conrad`/`crust`, `mantle`/`moho`,
    /// `outer-core`/`cmb`, `inner-core`/`iocb`/`icocb`).
    pub fn from_nd_str(model_name: &str, text: &str) -> Result<Self> {
        let stripped = strip_c_block_comments(text)?;
        let mut knots: Vec<[f64; 6]> = Vec::new();
        let mut explicit_conrad: Option<f64> = None;
        let mut moho_depth = DEFAULT_MOHO_KM;
        let mut cmb_depth = DEFAULT_CMB_KM;
        let mut iocb_depth = DEFAULT_IOCB_KM;

        for (line_idx, raw_line) in stripped.lines().enumerate() {
            let without_hash = raw_line.split('#').next().unwrap_or("");
            let tokens: Vec<&str> = without_hash.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            if tokens.len() == 1 {
                let Some(last_knot) = knots.last() else {
                    return Err(EarthSnailError::ModelError(format!(
                        ".nd line {}: named discontinuity '{}' before first knot",
                        line_idx + 1,
                        tokens[0]
                    )));
                };
                let d = last_knot[0];
                let dc = tokens[0].to_ascii_lowercase();
                match dc.as_str() {
                    "conrad" | "crust" => explicit_conrad = Some(d),
                    "mantle" | "moho" => moho_depth = d,
                    "outer-core" | "cmb" => cmb_depth = d,
                    "inner-core" | "iocb" | "icocb" => iocb_depth = d,
                    _ => {
                        return Err(EarthSnailError::ModelError(format!(
                            ".nd line {}: unrecognized discontinuity label '{}'",
                            line_idx + 1,
                            tokens[0]
                        )));
                    }
                }
                continue;
            }
            if tokens.len() < 3 {
                return Err(EarthSnailError::ModelError(format!(
                    ".nd line {}: expected at least 3 numeric columns (depth vp vs), got {}",
                    line_idx + 1,
                    tokens.len()
                )));
            }
            let row = parse_knot_tokens(&tokens, line_idx + 1, ".nd")?;
            knots.push(row);
        }

        Self::build_from_parsed_knots(
            model_name,
            &knots,
            explicit_conrad,
            moho_depth,
            cmb_depth,
            iocb_depth,
        )
    }

    /// Parses a TauP `.tvel` ASCII velocity model.
    ///
    /// Skips the standard 2-line `.tvel` header (plus any `#` comments or blank lines) and
    /// reads `depth vp vs [rho [qp qs]]` knot rows.
    pub fn from_tvel_str(model_name: &str, text: &str) -> Result<Self> {
        let mut lines_iter = text.lines().enumerate();
        // Skip up to 2 header lines in standard .tvel format.
        for _ in 0..2 {
            if lines_iter.next().is_none() {
                return Err(EarthSnailError::ModelError(
                    ".tvel input has fewer than 2 header lines".into(),
                ));
            }
        }

        let mut knots: Vec<[f64; 6]> = Vec::new();
        for (line_idx, raw_line) in lines_iter {
            let without_hash = raw_line.split('#').next().unwrap_or("");
            let tokens: Vec<&str> = without_hash.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            if tokens.len() < 3 {
                return Err(EarthSnailError::ModelError(format!(
                    ".tvel line {}: expected at least 3 numeric columns (depth vp vs), got {}",
                    line_idx + 1,
                    tokens.len()
                )));
            }
            let row = parse_knot_tokens(&tokens, line_idx + 1, ".tvel")?;
            knots.push(row);
        }

        Self::build_from_parsed_knots(
            model_name,
            &knots,
            None,
            DEFAULT_MOHO_KM,
            DEFAULT_CMB_KM,
            DEFAULT_IOCB_KM,
        )
    }

    fn build_from_parsed_knots(
        model_name: &str,
        knots: &[[f64; 6]],
        explicit_conrad: Option<f64>,
        moho_depth: f64,
        cmb_depth: f64,
        iocb_depth: f64,
    ) -> Result<Self> {
        if knots.len() < 2 {
            return Err(EarthSnailError::ModelError(
                "velocity model requires at least 2 depth knots".into(),
            ));
        }
        if knots[0][0] != 0.0 {
            return Err(EarthSnailError::ModelError(format!(
                "velocity model must start at surface depth 0.0 km, got {}",
                knots[0][0]
            )));
        }

        let mut layers = Vec::with_capacity(knots.len() - 1);
        for (idx, pair) in knots.windows(2).enumerate() {
            let top = pair[0];
            let bot = pair[1];
            if bot[0] < top[0] {
                return Err(EarthSnailError::ModelError(format!(
                    "decreasing depth from {} to {} at knot {}",
                    top[0],
                    bot[0],
                    idx + 1
                )));
            }
            if top[0] == bot[0] {
                continue;
            }
            layers.push(VelocityLayer {
                top_depth: top[0],
                bot_depth: bot[0],
                top_p_velocity: top[1],
                bot_p_velocity: bot[1],
                top_s_velocity: top[2],
                bot_s_velocity: bot[2],
                top_density: top[3],
                bot_density: bot[3],
                top_qp: top[4],
                bot_qp: bot[4],
                top_qs: top[5],
                bot_qs: bot[5],
            });
        }

        if layers.is_empty() {
            return Err(EarthSnailError::ModelError(
                "velocity model has zero non-degenerate layers".into(),
            ));
        }

        let radius_of_planet = knots[knots.len() - 1][0];
        let mut model = Self {
            model_name: model_name.to_string(),
            radius_of_planet,
            min_radius: 0.0,
            max_radius: radius_of_planet,
            conrad_depth: explicit_conrad,
            moho_depth,
            cmb_depth,
            iocb_depth,
            is_spherical: true,
            layers,
        };
        model.fix_discontinuity_depths();
        model.validate()?;
        Ok(model)
    }

    /// Snaps `moho_depth`, `cmb_depth`, `iocb_depth`, and `conrad_depth` to actual first-order
    /// velocity discontinuities in `self.layers`, matching IASPEI `TauP` conventions.
    pub fn fix_discontinuity_depths(&mut self) -> bool {
        let moho_min = 65.0_f64;
        let mut cmb_min = self.radius_of_planet;
        let iocb_min = self.radius_of_planet - 100.0;

        let mut temp_moho_depth = 0.0_f64;
        let mut temp_cmb_depth = self.radius_of_planet;
        let mut temp_iocb_depth = self.radius_of_planet;

        if self.layers.len() < 2 {
            let changed = self.moho_depth != temp_moho_depth
                || self.cmb_depth != temp_cmb_depth
                || self.iocb_depth != temp_iocb_depth;
            self.moho_depth = temp_moho_depth;
            self.cmb_depth = temp_cmb_depth;
            self.iocb_depth = temp_iocb_depth;
            self.conrad_depth = None;
            return changed;
        }

        let n_pairs = self.layers.len() - 1;
        let mut best_moho_diff = moho_min;
        for i in 0..n_pairs {
            let above = &self.layers[i];
            let below = &self.layers[i + 1];
            let is_disc = above.bot_p_velocity != below.top_p_velocity
                || above.bot_s_velocity != below.top_s_velocity;
            if !is_disc {
                continue;
            }
            let d = (self.moho_depth - above.bot_depth).abs();
            if d < best_moho_diff {
                best_moho_diff = d;
                temp_moho_depth = above.bot_depth;
            }
        }

        for i in 0..n_pairs {
            let above = &self.layers[i];
            let below = &self.layers[i + 1];
            let is_disc = above.bot_p_velocity != below.top_p_velocity
                || above.bot_s_velocity != below.top_s_velocity;
            if !is_disc {
                continue;
            }
            let d = (self.cmb_depth - above.bot_depth).abs();
            if d < cmb_min && above.bot_depth > temp_moho_depth {
                cmb_min = d;
                temp_cmb_depth = above.bot_depth;
            }
        }

        let mut best_iocb_diff = iocb_min;
        for i in 0..n_pairs {
            let above = &self.layers[i];
            let below = &self.layers[i + 1];
            let is_disc = above.bot_p_velocity != below.top_p_velocity
                || above.bot_s_velocity != below.top_s_velocity;
            if !is_disc {
                continue;
            }
            // IOCB must transition from fluid (S == 0) above to solid (S > 0) below.
            if above.bot_s_velocity == 0.0 && below.top_s_velocity > 0.0 {
                let d = (self.iocb_depth - above.bot_depth).abs();
                if d < best_iocb_diff {
                    best_iocb_diff = d;
                    temp_iocb_depth = above.bot_depth;
                }
            }
        }

        let final_layer = &self.layers[self.layers.len() - 1];
        if final_layer.bot_depth > temp_moho_depth
            && (self.cmb_depth - final_layer.bot_depth).abs() < cmb_min
        {
            temp_cmb_depth = final_layer.bot_depth;
        }

        if final_layer.bot_depth == temp_cmb_depth
            || (final_layer.bot_s_velocity == 0.0 && temp_iocb_depth == temp_cmb_depth)
        {
            temp_iocb_depth = final_layer.bot_depth;
        }

        let final_iocb = if temp_cmb_depth != temp_iocb_depth {
            temp_iocb_depth
        } else {
            self.radius_of_planet
        };

        let changed = self.moho_depth != temp_moho_depth
            || self.cmb_depth != temp_cmb_depth
            || self.iocb_depth != final_iocb;

        self.moho_depth = temp_moho_depth;
        self.cmb_depth = temp_cmb_depth;
        self.iocb_depth = final_iocb;

        // Identify intracrustal Conrad discontinuity if present strictly inside `(0, moho_depth)`.
        if let Some(target_conrad) = self.conrad_depth {
            let mut best: Option<(f64, f64)> = None;
            for i in 0..n_pairs {
                let above = &self.layers[i];
                let below = &self.layers[i + 1];
                let is_disc = above.bot_p_velocity != below.top_p_velocity
                    || above.bot_s_velocity != below.top_s_velocity;
                if is_disc && above.bot_depth > 0.0 && above.bot_depth < self.moho_depth {
                    let err = (above.bot_depth - target_conrad).abs();
                    if best.is_none_or(|(b_err, _)| err < b_err) {
                        best = Some((err, above.bot_depth));
                    }
                }
            }
            self.conrad_depth = best.map(|(_, d)| d);
        } else {
            for i in 0..n_pairs {
                let above = &self.layers[i];
                let below = &self.layers[i + 1];
                let is_disc = above.bot_p_velocity != below.top_p_velocity
                    || above.bot_s_velocity != below.top_s_velocity;
                if is_disc && above.bot_depth > 0.0 && above.bot_depth < self.moho_depth {
                    self.conrad_depth = Some(above.bot_depth);
                    break;
                }
            }
        }

        changed
    }

    /// Validates internal physical and topological consistency of the velocity model.
    pub fn validate(&self) -> Result<()> {
        if !self.radius_of_planet.is_finite() || self.radius_of_planet <= 0.0 {
            return Err(EarthSnailError::ModelError(
                "radius_of_planet must be finite and positive".into(),
            ));
        }
        if self.moho_depth < 0.0
            || self.cmb_depth < self.moho_depth
            || self.iocb_depth < self.cmb_depth
        {
            return Err(EarthSnailError::ModelError(format!(
                "invalid discontinuity ordering: moho={}, cmb={}, iocb={}",
                self.moho_depth, self.cmb_depth, self.iocb_depth
            )));
        }
        if self.min_radius < 0.0 || self.max_radius <= self.min_radius {
            return Err(EarthSnailError::ModelError(
                "invalid min_radius / max_radius bounds".into(),
            ));
        }
        if self.layers.is_empty() {
            return Err(EarthSnailError::ModelError(
                "velocity model contains zero layers".into(),
            ));
        }

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.bot_depth <= layer.top_depth {
                return Err(EarthSnailError::ModelError(format!(
                    "zero or negative thickness at velocity layer {i}"
                )));
            }
            if layer.top_p_velocity <= 0.0 || layer.bot_p_velocity <= 0.0 {
                return Err(EarthSnailError::ModelError(format!(
                    "non-positive P velocity at layer {i}"
                )));
            }
            if layer.top_s_velocity < 0.0 || layer.bot_s_velocity < 0.0 {
                return Err(EarthSnailError::ModelError(format!(
                    "negative S velocity at layer {i}"
                )));
            }
            if layer.top_s_velocity > layer.top_p_velocity
                || layer.bot_s_velocity > layer.bot_p_velocity
            {
                return Err(EarthSnailError::ModelError(format!(
                    "S velocity exceeds P velocity at layer {i}"
                )));
            }
            if layer.top_depth != 0.0
                && ((layer.top_s_velocity != 0.0 && layer.bot_s_velocity == 0.0)
                    || (layer.top_s_velocity == 0.0 && layer.bot_s_velocity != 0.0))
            {
                return Err(EarthSnailError::ModelError(format!(
                    "S velocity transitions continuously to zero inside layer {i} without a discontinuity"
                )));
            }
            if i > 0 && self.layers[i - 1].bot_depth != layer.top_depth {
                return Err(EarthSnailError::ModelError(format!(
                    "depth gap or overlap between velocity layers {} and {i}",
                    i - 1
                )));
            }
        }
        Ok(())
    }

    /// Number of spherical velocity shells in the model.
    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if `self.layers` is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns all discontinuity depths (surface `0.0`, internal 1st-order velocity jumps,
    /// and center `bot_depth`).
    pub fn discontinuity_depths(&self) -> Vec<f64> {
        if self.layers.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(12);
        out.push(self.layers[0].top_depth);
        for pair in self.layers.windows(2) {
            if pair[0].bot_p_velocity != pair[1].top_p_velocity
                || pair[0].bot_s_velocity != pair[1].top_s_velocity
            {
                out.push(pair[0].bot_depth);
            }
        }
        out.push(self.layers[self.layers.len() - 1].bot_depth);
        out
    }

    /// Returns `true` if `depth` matches a model discontinuity depth.
    pub fn is_discontinuity(&self, depth: f64) -> bool {
        self.discontinuity_depths().contains(&depth)
    }

    /// Finds the index of the layer containing `depth`, returning the upper layer if `depth`
    /// lies on an internal boundary (`top_depth < depth <= bot_depth`, with `0` for `depth == 0.0`).
    pub fn layer_number_above(&self, depth: f64) -> Result<usize> {
        if self.layers.is_empty() {
            return Err(EarthSnailError::ModelError("empty velocity model".into()));
        }
        if depth == self.layers[0].top_depth {
            return Ok(0);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.top_depth < depth && depth <= layer.bot_depth {
                return Ok(i);
            }
        }
        Err(EarthSnailError::ModelError(format!(
            "no velocity layer above depth {depth} km"
        )))
    }

    /// Finds the index of the layer containing `depth`, returning the lower layer if `depth`
    /// lies on an internal boundary (`top_depth <= depth < bot_depth`, or last layer at center).
    pub fn layer_number_below(&self, depth: f64) -> Result<usize> {
        if self.layers.is_empty() {
            return Err(EarthSnailError::ModelError("empty velocity model".into()));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.top_depth <= depth && depth < layer.bot_depth {
                return Ok(i);
            }
        }
        let last = self.layers.len() - 1;
        if depth == self.layers[last].bot_depth {
            return Ok(last);
        }
        Err(EarthSnailError::ModelError(format!(
            "no velocity layer below depth {depth} km"
        )))
    }

    /// Evaluates `prop` at `depth` from the upper side if `depth` is a discontinuity.
    pub fn evaluate_above(&self, depth: f64, prop: MaterialProperty) -> Result<f64> {
        let idx = self.layer_number_above(depth)?;
        Ok(self.layers[idx].evaluate_at(depth, prop))
    }

    /// Evaluates `prop` at `depth` from the lower side if `depth` is a discontinuity.
    pub fn evaluate_below(&self, depth: f64, prop: MaterialProperty) -> Result<f64> {
        let idx = self.layer_number_below(depth)?;
        Ok(self.layers[idx].evaluate_at(depth, prop))
    }
}

/// Earth-flattening transformation mapping spherical radius $r \in (0, R_E]$ to flat-Earth depth
/// $z_f = -R_E \ln(r / R_E)$.
#[inline]
pub fn earth_flatten_depth(radius_km: f64, planet_radius_km: f64) -> Result<f64> {
    if !radius_km.is_finite()
        || !planet_radius_km.is_finite()
        || radius_km <= 0.0
        || planet_radius_km <= 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "earth_flatten_depth requires positive finite radii".into(),
        ));
    }
    Ok(-planet_radius_km * math::ln(radius_km / planet_radius_km))
}

/// Earth-flattening velocity transformation $v_f(z_f) = v(r) \cdot (R_E / r)$.
#[inline]
pub fn earth_flatten_velocity(
    spherical_vel_km_s: f64,
    radius_km: f64,
    planet_radius_km: f64,
) -> Result<f64> {
    if !spherical_vel_km_s.is_finite()
        || !radius_km.is_finite()
        || !planet_radius_km.is_finite()
        || radius_km <= 0.0
        || planet_radius_km <= 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "earth_flatten_velocity requires positive finite radii".into(),
        ));
    }
    Ok(spherical_vel_km_s * (planet_radius_km / radius_km))
}

/// Inverse Earth-flattening transformation mapping flat depth $z_f$ back to spherical radius
/// $r = R_E \exp(-z_f / R_E)$.
#[inline]
pub fn earth_unflatten_radius(flat_depth_km: f64, planet_radius_km: f64) -> Result<f64> {
    if !flat_depth_km.is_finite() || !planet_radius_km.is_finite() || planet_radius_km <= 0.0 {
        return Err(EarthSnailError::InvalidParameter(
            "earth_unflatten_radius requires finite flat depth and positive planet radius".into(),
        ));
    }
    Ok(planet_radius_km * math::exp(-flat_depth_km / planet_radius_km))
}

/// Inverse Earth-flattening velocity transformation $v(r) = v_f(z_f) \cdot (r / R_E)$.
#[inline]
pub fn earth_unflatten_velocity(
    flat_vel_km_s: f64,
    radius_km: f64,
    planet_radius_km: f64,
) -> Result<f64> {
    if !flat_vel_km_s.is_finite()
        || !radius_km.is_finite()
        || !planet_radius_km.is_finite()
        || radius_km <= 0.0
        || planet_radius_km <= 0.0
    {
        return Err(EarthSnailError::InvalidParameter(
            "earth_unflatten_velocity requires positive finite radii".into(),
        ));
    }
    Ok(flat_vel_km_s * (radius_km / planet_radius_km))
}

fn strip_c_block_comments(input: &str) -> Result<String> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        out.push(' ');
        let after_open = &rest[start + 2..];
        let Some(end) = after_open.find("*/") else {
            return Err(EarthSnailError::ModelError(
                "unclosed '/* ... */' block comment in velocity model".into(),
            ));
        };
        // Preserve newline count inside multi-line block comments for accurate line numbers.
        for ch in after_open[..end].chars() {
            if ch == '\n' {
                out.push('\n');
            }
        }
        rest = &after_open[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

fn parse_knot_tokens(tokens: &[&str], line_no: usize, fmt_label: &str) -> Result<[f64; 6]> {
    let parse_num = |s: &str, col_name: &str| -> Result<f64> {
        let v = s.parse::<f64>().map_err(|_| {
            EarthSnailError::ModelError(format!(
                "{fmt_label} line {line_no}: invalid float '{s}' for {col_name}"
            ))
        })?;
        if !v.is_finite() {
            return Err(EarthSnailError::ModelError(format!(
                "{fmt_label} line {line_no}: non-finite float '{s}' for {col_name}"
            )));
        }
        Ok(v)
    };

    let depth = parse_num(tokens[0], "depth")?;
    let vp = parse_num(tokens[1], "vp")?;
    let vs = parse_num(tokens[2], "vs")?;
    let rho = if tokens.len() >= 4 {
        parse_num(tokens[3], "density")?
    } else {
        DEFAULT_DENSITY
    };
    let qp = if tokens.len() >= 5 {
        parse_num(tokens[4], "qp")?
    } else {
        DEFAULT_QP
    };
    let qs = if tokens.len() >= 6 {
        parse_num(tokens[5], "qs")?
    } else {
        DEFAULT_QS
    };

    if depth < 0.0 {
        return Err(EarthSnailError::ModelError(format!(
            "{fmt_label} line {line_no}: negative depth {depth}"
        )));
    }
    if vp <= 0.0 || vs < 0.0 {
        return Err(EarthSnailError::ModelError(format!(
            "{fmt_label} line {line_no}: invalid velocities vp={vp}, vs={vs}"
        )));
    }
    if vs > vp {
        return Err(EarthSnailError::ModelError(format!(
            "{fmt_label} line {line_no}: vs ({vs}) > vp ({vp})"
        )));
    }

    Ok([depth, vp, vs, rho, qp, qs])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_models_validate_and_match_discontinuities() {
        let iasp = VelocityModel::iasp91();
        iasp.validate().unwrap();
        assert_eq!(iasp.radius_of_planet, 6371.0);
        assert_eq!(iasp.conrad_depth, Some(20.0));
        assert_eq!(iasp.moho_depth, 35.0);
        assert_eq!(iasp.cmb_depth, 2889.0);
        assert_eq!(iasp.iocb_depth, 5153.9);
        assert!(iasp.is_discontinuity(410.0));
        assert!(iasp.is_discontinuity(660.0));

        let ak = VelocityModel::ak135();
        ak.validate().unwrap();
        assert_eq!(ak.conrad_depth, Some(20.0));
        assert_eq!(ak.moho_depth, 35.0);
        assert_eq!(ak.cmb_depth, 2891.5);
        assert_eq!(ak.iocb_depth, 5153.5);

        let prem = VelocityModel::prem();
        prem.validate().unwrap();
        assert_eq!(prem.conrad_depth, Some(15.0));
        assert_eq!(prem.moho_depth, 24.4);
        assert_eq!(prem.cmb_depth, 2891.0);
        assert_eq!(prem.iocb_depth, 5149.5);
        assert!(prem.is_discontinuity(220.0));
        assert!(prem.is_discontinuity(400.0));
        assert!(prem.is_discontinuity(670.0));
    }

    #[test]
    fn custom_nd_and_tvel_parsers_and_malformed_rejection() {
        let nd_src = "\
            # Synthetic whole-planet model
            0.0   5.8  3.4  2.7
            18.0  5.8  3.4  2.7
            conrad
            18.0  6.6  3.8  2.9
            33.0  6.6  3.8  2.9
            mantle
            33.0  8.0  4.5  3.3
            410.0 9.0  4.9  3.5
            410.0 9.4  5.1  3.7
            2890.0 13.7 7.2 5.5
            outer-core
            2890.0 8.0 0.0  9.9
            5150.0 10.3 0.0 12.1
            inner-core
            5150.0 11.0 3.5 12.7
            6371.0 11.2 3.6 13.0
        ";
        let vm = VelocityModel::from_nd_str("custom_nd", nd_src).unwrap();
        assert_eq!(vm.conrad_depth, Some(18.0));
        assert_eq!(vm.moho_depth, 33.0);
        assert_eq!(vm.cmb_depth, 2890.0);
        assert_eq!(vm.iocb_depth, 5150.0);
        assert_eq!(vm.radius_of_planet, 6371.0);

        let tvel_src = "\
            Header P
            Header S
            0.0    6.0  3.5  2.7
            30.0   6.0  3.5  2.7
            30.0   8.1  4.5  3.3
            2890.0 13.6 7.2  5.5
            2890.0 8.1  0.0  9.9
            5150.0 10.2 0.0 12.1
            5150.0 11.1 3.5 12.7
            6371.0 11.3 3.6 13.0
        ";
        let tv = VelocityModel::from_tvel_str("custom_tvel", tvel_src).unwrap();
        assert_eq!(tv.moho_depth, 30.0);
        assert_eq!(tv.cmb_depth, 2890.0);
        assert_eq!(tv.iocb_depth, 5150.0);

        // Malformed inputs must return typed Err, never panic:
        assert!(VelocityModel::from_nd_str("bad1", "0.0 5.0 6.0 2.7\n100.0 5.0 6.0 2.7").is_err());
        assert!(VelocityModel::from_nd_str("bad2", "0.0 NaN 3.0 2.7\n100.0 6.0 3.0 2.7").is_err());
        assert!(VelocityModel::from_nd_str("bad3", "0.0 6.0 3.0\n/* unclosed").is_err());
        assert!(VelocityModel::from_tvel_str("bad4", "only_one_header_line").is_err());
    }

    #[test]
    fn earth_flattening_roundtrip() {
        let re = 6371.0;
        for &z in &[0.0, 35.0, 410.0, 660.0, 2889.0, 5153.9] {
            let r = re - z;
            let zf = earth_flatten_depth(r, re).unwrap();
            let r_back = earth_unflatten_radius(zf, re).unwrap();
            let tol = if cfg!(miri) { 1e-9 } else { 1e-11 };
            assert!((r_back - r).abs() < tol);

            let vf = earth_flatten_velocity(8.0, r, re).unwrap();
            let v_back = earth_unflatten_velocity(vf, r, re).unwrap();
            assert!((v_back - 8.0).abs() < tol);
        }
    }
}
