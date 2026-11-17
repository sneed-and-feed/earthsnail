//! Zero-dependency W3C SVG 1.1 builder (`SvgCanvas`, `SvgStyle`, `SvgPathBuilder`, `PlotViewport`)
//! for publication-grade waveform plots, spectrograms, PPSD, TauP ray paths, and focal beachballs.

use crate::math;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

/// Escapes special XML characters (`&`, `<`, `>`, `"`, `'`) in text or attribute strings.
pub fn xml_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Stroke, fill, opacity, and typography attributes for SVG elements.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SvgStyle {
    /// Stroke color (e.g. `"#1f2937"`, `"none"`).
    pub stroke: Option<String>,
    /// Stroke width in SVG user units.
    pub stroke_width: Option<f64>,
    /// Optional dash pattern (e.g. `"4,3"`).
    pub stroke_dasharray: Option<String>,
    /// Fill color (e.g. `"none"`, `"#2563eb"`).
    pub fill: Option<String>,
    /// Overall element opacity in `[0.0, 1.0]`.
    pub opacity: Option<f64>,
    /// Font family for `<text>` elements.
    pub font_family: Option<String>,
    /// Font size in pixels/points.
    pub font_size: Option<f64>,
    /// Font weight (e.g. `"bold"`, `"600"`).
    pub font_weight: Option<String>,
    /// Horizontal text alignment (`"start"`, `"middle"`, `"end"`).
    pub text_anchor: Option<String>,
    /// Vertical text baseline (`"auto"`, `"middle"`, `"hanging"`, `"central"`).
    pub dominant_baseline: Option<String>,
}

impl SvgStyle {
    /// Creates an empty style.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a unfilled stroke-only line style (`fill="none"`).
    pub fn stroke_only(color: &str, width: f64) -> Self {
        Self {
            stroke: Some(String::from(color)),
            stroke_width: Some(width),
            fill: Some(String::from("none")),
            ..Self::default()
        }
    }

    /// Creates a solid filled polygon/shape style (`stroke="none"`).
    pub fn fill_only(color: &str) -> Self {
        Self {
            fill: Some(String::from(color)),
            stroke: Some(String::from("none")),
            ..Self::default()
        }
    }

    /// Creates a text style with font size, color, and anchor (`"start"`, `"middle"`, `"end"`).
    pub fn label(color: &str, font_size: f64, anchor: &str) -> Self {
        Self {
            fill: Some(String::from(color)),
            font_family: Some(String::from(
                "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            )),
            font_size: Some(font_size),
            text_anchor: Some(String::from(anchor)),
            ..Self::default()
        }
    }

    /// Sets `stroke_dasharray`.
    pub fn with_dasharray(mut self, pattern: &str) -> Self {
        self.stroke_dasharray = Some(String::from(pattern));
        self
    }

    /// Sets `opacity`.
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = Some(opacity);
        self
    }

    /// Sets `font_weight`.
    pub fn with_font_weight(mut self, weight: &str) -> Self {
        self.font_weight = Some(String::from(weight));
        self
    }

    /// Sets `dominant_baseline`.
    pub fn with_baseline(mut self, baseline: &str) -> Self {
        self.dominant_baseline = Some(String::from(baseline));
        self
    }

    fn write_attrs(&self, out: &mut String) {
        if let Some(ref s) = self.stroke {
            let _ = write!(out, " stroke=\"{}\"", xml_escape(s));
        }
        if let Some(w) = self.stroke_width {
            let _ = write!(out, " stroke-width=\"{w:.4}\"");
        }
        if let Some(ref da) = self.stroke_dasharray {
            let _ = write!(out, " stroke-dasharray=\"{}\"", xml_escape(da));
        }
        if let Some(ref fl) = self.fill {
            let _ = write!(out, " fill=\"{}\"", xml_escape(fl));
        }
        if let Some(op) = self.opacity {
            let _ = write!(out, " opacity=\"{op:.4}\"");
        }
        if let Some(ref ff) = self.font_family {
            let _ = write!(out, " font-family=\"{}\"", xml_escape(ff));
        }
        if let Some(fs) = self.font_size {
            let _ = write!(out, " font-size=\"{fs:.2}\"");
        }
        if let Some(ref fw) = self.font_weight {
            let _ = write!(out, " font-weight=\"{}\"", xml_escape(fw));
        }
        if let Some(ref ta) = self.text_anchor {
            let _ = write!(out, " text-anchor=\"{}\"", xml_escape(ta));
        }
        if let Some(ref db) = self.dominant_baseline {
            let _ = write!(out, " dominant-baseline=\"{}\"", xml_escape(db));
        }
    }
}

/// Builder for SVG `<path d="...">` command strings (`M`, `L`, `C`, `Q`, `A`, `Z`).
#[derive(Debug, Clone, Default)]
pub struct SvgPathBuilder {
    buf: String,
}

impl SvgPathBuilder {
    /// Creates an empty path builder.
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    fn sep(&mut self) {
        if !self.buf.is_empty() {
            self.buf.push(' ');
        }
    }

    /// Appends `M x y`.
    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.sep();
        let _ = write!(self.buf, "M {x:.3} {y:.3}");
        self
    }

    /// Appends `L x y`.
    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.sep();
        let _ = write!(self.buf, "L {x:.3} {y:.3}");
        self
    }

    /// Appends cubic Bezier `C x1 y1, x2 y2, x y`.
    pub fn cubic_to(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, x: f64, y: f64) -> &mut Self {
        self.sep();
        let _ = write!(
            self.buf,
            "C {x1:.3} {y1:.3}, {x2:.3} {y2:.3}, {x:.3} {y:.3}"
        );
        self
    }

    /// Appends elliptical arc `A rx ry x_rot large_arc sweep x y`.
    #[allow(clippy::too_many_arguments)]
    pub fn arc_to(
        &mut self,
        rx: f64,
        ry: f64,
        x_rotation_deg: f64,
        large_arc: bool,
        sweep: bool,
        x: f64,
        y: f64,
    ) -> &mut Self {
        self.sep();
        let la = u8::from(large_arc);
        let sw = u8::from(sweep);
        let _ = write!(
            self.buf,
            "A {rx:.3} {ry:.3} {x_rotation_deg:.2} {la} {sw} {x:.3} {y:.3}"
        );
        self
    }

    /// Closes current subpath (`Z`).
    pub fn close(&mut self) -> &mut Self {
        self.sep();
        self.buf.push('Z');
        self
    }

    /// Returns the built `d` path string slice.
    pub fn as_str(&self) -> &str {
        &self.buf
    }

    /// Consumes the builder and returns the `d` string.
    pub fn finish(self) -> String {
        self.buf
    }
}

/// Rectangular data-to-pixel linear coordinate viewport (`[x_min, x_max] x [y_min, y_max]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlotViewport {
    /// Left pixel coordinate of plot frame.
    pub left: f64,
    /// Top pixel coordinate of plot frame.
    pub top: f64,
    /// Pixel width of plot frame.
    pub width: f64,
    /// Pixel height of plot frame.
    pub height: f64,
    /// Data-space minimum X value.
    pub x_min: f64,
    /// Data-space maximum X value.
    pub x_max: f64,
    /// Data-space minimum Y value (mapped to `top + height`).
    pub y_min: f64,
    /// Data-space maximum Y value (mapped to `top`).
    pub y_max: f64,
}

impl PlotViewport {
    /// Maps a data-space X value to SVG pixel X.
    #[inline]
    pub fn map_x(&self, x: f64) -> f64 {
        let span = self.x_max - self.x_min;
        if span.abs() < 1e-30 {
            self.left + 0.5 * self.width
        } else {
            self.left + ((x - self.x_min) / span) * self.width
        }
    }

    /// Maps a data-space Y value to SVG pixel Y (inverted screen Y).
    #[inline]
    pub fn map_y(&self, y: f64) -> f64 {
        let span = self.y_max - self.y_min;
        if span.abs() < 1e-30 {
            self.top + 0.5 * self.height
        } else {
            self.top + self.height - ((y - self.y_min) / span) * self.height
        }
    }

    /// Maps `(x, y)` data coordinates to `(px, py)` SVG coordinates.
    #[inline]
    pub fn map_point(&self, x: f64, y: f64) -> (f64, f64) {
        (self.map_x(x), self.map_y(y))
    }
}

/// Computes human-readable `{1, 2, 5} * 10^k` tick values across `[min, max]`.
pub fn nice_ticks(min: f64, max: f64, target_count: usize) -> Vec<f64> {
    if !min.is_finite() || !max.is_finite() || target_count == 0 {
        return Vec::new();
    }
    if (max - min).abs() < 1e-30 {
        return alloc::vec![min];
    }
    let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
    let span = hi - lo;
    let raw_step = span / (target_count.max(2) as f64);
    let exp10 = math::powf(10.0, math::floor(math::log10(raw_step)));
    let norm = raw_step / exp10;
    let nice_mult = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    };
    let step = nice_mult * exp10;
    let first_idx = math::ceil(lo / step - 1e-12);
    let last_idx = math::floor(hi / step + 1e-12);
    let count = ((last_idx - first_idx) as isize + 1).clamp(0, 256) as usize;
    let mut ticks = Vec::with_capacity(count);
    for k in 0..count {
        let v = (first_idx + (k as f64)) * step;
        let clean = if v.abs() < 1e-12 * step { 0.0 } else { v };
        ticks.push(clean);
    }
    ticks
}

/// Standalone zero-dependency SVG document builder.
#[derive(Debug, Clone)]
pub struct SvgCanvas {
    width: f64,
    height: f64,
    defs: String,
    body: String,
}

impl SvgCanvas {
    /// Creates a new `SvgCanvas` with dimensions `(width, height)` in pixels.
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            defs: String::new(),
            body: String::with_capacity(4096),
        }
    }

    /// Canvas width in user units.
    #[inline]
    pub const fn width(&self) -> f64 {
        self.width
    }

    /// Canvas height in user units.
    #[inline]
    pub const fn height(&self) -> f64 {
        self.height
    }

    /// Fills the entire canvas background with `color`.
    pub fn set_background(&mut self, color: &str) {
        self.rect(
            0.0,
            0.0,
            self.width,
            self.height,
            &SvgStyle::fill_only(color),
        );
    }

    /// Registers a rectangular `<clipPath id="...">` in `<defs>`.
    pub fn define_clip_rect(&mut self, id: &str, x: f64, y: f64, width: f64, height: f64) {
        let esc_id = xml_escape(id);
        let _ = writeln!(
            self.defs,
            "  <clipPath id=\"{esc_id}\"><rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{width:.2}\" height=\"{height:.2}\"/></clipPath>"
        );
    }

    /// Begins a `<g>` group with optional `transform` and `clip_id`.
    pub fn begin_group(
        &mut self,
        transform: Option<&str>,
        clip_id: Option<&str>,
        style: Option<&SvgStyle>,
    ) {
        self.body.push_str("  <g");
        if let Some(tr) = transform {
            let _ = write!(self.body, " transform=\"{}\"", xml_escape(tr));
        }
        if let Some(cid) = clip_id {
            let _ = write!(self.body, " clip-path=\"url(#{})\"", xml_escape(cid));
        }
        if let Some(st) = style {
            st.write_attrs(&mut self.body);
        }
        self.body.push_str(">\n");
    }

    /// Closes a `<g>` group opened by `begin_group`.
    pub fn end_group(&mut self) {
        self.body.push_str("  </g>\n");
    }

    /// Appends a `<rect>` element.
    pub fn rect(&mut self, x: f64, y: f64, width: f64, height: f64, style: &SvgStyle) {
        let _ = write!(
            self.body,
            "  <rect x=\"{x:.3}\" y=\"{y:.3}\" width=\"{width:.3}\" height=\"{height:.3}\""
        );
        style.write_attrs(&mut self.body);
        self.body.push_str("/>\n");
    }

    /// Appends a `<circle>` element.
    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, style: &SvgStyle) {
        let _ = write!(
            self.body,
            "  <circle cx=\"{cx:.3}\" cy=\"{cy:.3}\" r=\"{r:.3}\""
        );
        style.write_attrs(&mut self.body);
        self.body.push_str("/>\n");
    }

    /// Appends a `<line>` segment.
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, style: &SvgStyle) {
        let _ = write!(
            self.body,
            "  <line x1=\"{x1:.3}\" y1=\"{y1:.3}\" x2=\"{x2:.3}\" y2=\"{y2:.3}\""
        );
        style.write_attrs(&mut self.body);
        self.body.push_str("/>\n");
    }

    /// Appends a `<polyline>` element from `(x, y)` coordinates.
    pub fn polyline(&mut self, points: &[(f64, f64)], style: &SvgStyle) {
        if points.is_empty() {
            return;
        }
        self.body.push_str("  <polyline points=\"");
        for (i, &(x, y)) in points.iter().enumerate() {
            if i > 0 {
                self.body.push(' ');
            }
            let _ = write!(self.body, "{x:.3},{y:.3}");
        }
        self.body.push('"');
        style.write_attrs(&mut self.body);
        self.body.push_str("/>\n");
    }

    /// Appends a closed `<polygon>` element from `(x, y)` vertices.
    pub fn polygon(&mut self, points: &[(f64, f64)], style: &SvgStyle) {
        if points.is_empty() {
            return;
        }
        self.body.push_str("  <polygon points=\"");
        for (i, &(x, y)) in points.iter().enumerate() {
            if i > 0 {
                self.body.push(' ');
            }
            let _ = write!(self.body, "{x:.3},{y:.3}");
        }
        self.body.push('"');
        style.write_attrs(&mut self.body);
        self.body.push_str("/>\n");
    }

    /// Appends a `<path d="...">` element.
    pub fn path(&mut self, d: &str, style: &SvgStyle) {
        let _ = write!(self.body, "  <path d=\"{}\"", xml_escape(d));
        style.write_attrs(&mut self.body);
        self.body.push_str("/>\n");
    }

    /// Appends an XML-escaped `<text>` element at `(x, y)`.
    pub fn text(&mut self, x: f64, y: f64, content: &str, style: &SvgStyle) {
        let _ = write!(self.body, "  <text x=\"{x:.3}\" y=\"{y:.3}\"");
        style.write_attrs(&mut self.body);
        let _ = writeln!(self.body, ">{}</text>", xml_escape(content));
    }

    /// Appends a rotated XML-escaped `<text>` element at `(x, y)` (useful for Y-axis titles).
    pub fn text_rotated(
        &mut self,
        x: f64,
        y: f64,
        angle_deg: f64,
        content: &str,
        style: &SvgStyle,
    ) {
        let _ = write!(
            self.body,
            "  <text x=\"{x:.3}\" y=\"{y:.3}\" transform=\"rotate({angle_deg:.2} {x:.3} {y:.3})\""
        );
        style.write_attrs(&mut self.body);
        let _ = writeln!(self.body, ">{}</text>", xml_escape(content));
    }

    /// Renders gridlines, border frame, tick marks, formatted tick labels, and axis titles for `viewport`.
    pub fn draw_axes(
        &mut self,
        viewport: &PlotViewport,
        x_label: &str,
        y_label: &str,
        x_tick_count: usize,
        y_tick_count: usize,
    ) {
        let grid_style = SvgStyle::stroke_only("#e5e7eb", 0.75).with_dasharray("3,3");
        let tick_style = SvgStyle::stroke_only("#374151", 1.0);
        let frame_style = SvgStyle::stroke_only("#111827", 1.2);
        let x_lbl_style = SvgStyle::label("#374151", 11.0, "middle").with_baseline("hanging");
        let y_lbl_style = SvgStyle::label("#374151", 11.0, "end").with_baseline("central");
        let title_style = SvgStyle::label("#111827", 12.5, "middle").with_font_weight("600");

        let bottom = viewport.top + viewport.height;
        let right = viewport.left + viewport.width;

        for tx in nice_ticks(viewport.x_min, viewport.x_max, x_tick_count) {
            let px = viewport.map_x(tx);
            if px >= viewport.left - 0.5 && px <= right + 0.5 {
                self.line(px, viewport.top, px, bottom, &grid_style);
                self.line(px, bottom, px, bottom + 5.0, &tick_style);
                self.text(px, bottom + 8.0, &format_tick(tx), &x_lbl_style);
            }
        }

        for ty in nice_ticks(viewport.y_min, viewport.y_max, y_tick_count) {
            let py = viewport.map_y(ty);
            if py >= viewport.top - 0.5 && py <= bottom + 0.5 {
                self.line(viewport.left, py, right, py, &grid_style);
                self.line(viewport.left - 5.0, py, viewport.left, py, &tick_style);
                self.text(viewport.left - 8.0, py, &format_tick(ty), &y_lbl_style);
            }
        }

        self.rect(
            viewport.left,
            viewport.top,
            viewport.width,
            viewport.height,
            &frame_style,
        );

        if !x_label.is_empty() {
            self.text(
                viewport.left + 0.5 * viewport.width,
                bottom + 28.0,
                x_label,
                &title_style,
            );
        }
        if !y_label.is_empty() {
            self.text_rotated(
                (viewport.left - 48.0).max(14.0),
                viewport.top + 0.5 * viewport.height,
                -90.0,
                y_label,
                &title_style,
            );
        }
    }

    /// Produces the complete standalone W3C SVG 1.1 XML string.
    pub fn finish(&self) -> String {
        let mut svg = String::with_capacity(self.defs.len() + self.body.len() + 256);
        let _ = writeln!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.2} {:.2}\" width=\"{:.2}\" height=\"{:.2}\">",
            self.width, self.height, self.width, self.height
        );
        if !self.defs.is_empty() {
            svg.push_str("<defs>\n");
            svg.push_str(&self.defs);
            svg.push_str("</defs>\n");
        }
        svg.push_str(&self.body);
        svg.push_str("</svg>\n");
        svg
    }
}

fn format_tick(v: f64) -> String {
    let av = v.abs();
    if av == 0.0 {
        String::from("0")
    } else if !(1e-3..1e5).contains(&av) {
        format!("{v:.2e}")
    } else {
        let s = format!("{v:.6}");
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        String::from(if trimmed == "-0" { "0" } else { trimmed })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_builder_escapes_xml_and_renders_valid_document() {
        assert_eq!(
            xml_escape("IU.ANMO <BHZ> & \"P'P\""),
            "IU.ANMO &lt;BHZ&gt; &amp; &quot;P&apos;P&quot;"
        );
        let mut canvas = SvgCanvas::new(800.0, 300.0);
        canvas.set_background("#ffffff");
        let vp = PlotViewport {
            left: 65.0,
            top: 20.0,
            width: 710.0,
            height: 230.0,
            x_min: 0.0,
            x_max: 60.0,
            y_min: -1.0,
            y_max: 1.0,
        };
        canvas.draw_axes(&vp, "Time (s) & Offset", "Velocity <nm/s>", 6, 5);
        canvas.polyline(
            &[
                vp.map_point(0.0, 0.0),
                vp.map_point(30.0, 0.8),
                vp.map_point(60.0, -0.4),
            ],
            &SvgStyle::stroke_only("#2563eb", 1.25),
        );
        let out = canvas.finish();
        assert!(out.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(out.ends_with("</svg>\n"));
        assert!(out.contains("Time (s) &amp; Offset"));
        assert!(out.contains("Velocity &lt;nm/s&gt;"));
        assert!(out.contains("<polyline "));
    }
}
