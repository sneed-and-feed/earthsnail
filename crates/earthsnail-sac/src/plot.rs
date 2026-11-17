//! Zero-dependency W3C SVG 1.1 publication plotting engine for EarthSnail (`plot_waveforms_svg`,
//! `plot_record_section_svg`, `plot_spectrogram_svg`, `plot_ppsd_svg`, `plot_taup_paths_svg`,
//! and `plot_beachball_svg`), built strictly on `earthsnail_core::svg::SvgCanvas`.

use core::f64::consts::{PI, TAU};
use earthsnail_core::event::{FocalMechanism, MomentTensor6, NodalPlane, PrincipalAxis};
use earthsnail_core::geodesics::{gps2dist_azimuth, kilometer2degrees};
use earthsnail_core::math;
use earthsnail_core::sac_header::{is_sac_float_undef, SacDepType, SacHeader};
use earthsnail_core::svg::{nice_ticks, PlotViewport, SvgCanvas, SvgPathBuilder, SvgStyle};
use earthsnail_core::time::UtcTimestamp;
use earthsnail_core::trace::{Stream, Trace};
use earthsnail_dsp::fft::{rfft, rfftfreq};
use earthsnail_dsp::moment_tensor::{
    beachball_p_radiation_grid, decompose_moment_tensor, strike_dip_rake_to_mt, BeachballProjection,
};
use earthsnail_dsp::preprocess::{demean_in_place, hann_window};
use earthsnail_dsp::spectral::{cwt_morlet, get_nhnm, get_nlnm, Ppsd};
use earthsnail_taup::{ArrivalPath, PhaseTravelTimeCurve, VelocityModel};

const PALETTE: [&str; 10] = [
    "#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be185d", "#4f46e5",
    "#65a30d", "#b91c1c",
];

// ============================================================================
// 1. Waveform Plotting (`plot1` Stacked & `plot2` Overlaid)
// ============================================================================

/// Multi-trace panel arrangement (`plot1` stacked vs `plot2` overlaid).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveformLayout {
    /// Separate vertically stacked panel per trace (`SAC plot1` / `p1`).
    #[default]
    Stacked,
    /// Single shared panel with overlaid color-coded traces (`SAC plot2` / `p2`).
    Overlaid,
}

/// Horizontal time-axis tick formatting convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeAxisMode {
    /// Seconds relative to earliest trace start time.
    #[default]
    RelativeSeconds,
    /// Formatted `HH:MM:SS` UTC clock labels anchored to the earliest trace `UtcTimestamp`.
    UtcClock,
}

/// Styling and layout options for [`plot_waveforms_svg`].
#[derive(Debug, Clone)]
pub struct WaveformPlotOptions {
    /// SVG width in pixels (`1000.0` default).
    pub width: f64,
    /// SVG height in pixels (`0.0` = auto-scale with trace count in `Stacked` mode).
    pub height: f64,
    /// Stacked (`plot1`) vs Overlaid (`plot2`).
    pub layout: WaveformLayout,
    /// Relative seconds vs UTC clock labels.
    pub time_axis: TimeAxisMode,
    /// Optional figure title.
    pub title: Option<String>,
    /// Optional explicit Y-axis unit override (auto-inferred from `SacHeader.idep` when `None`).
    pub y_unit: Option<String>,
    /// Render `O`, `A`, `F`, and `T0..T9` (`KT0..KT9`) vertical phase markers from SAC headers.
    pub show_phase_flags: bool,
}

impl Default for WaveformPlotOptions {
    fn default() -> Self {
        Self {
            width: 1000.0,
            height: 0.0,
            layout: WaveformLayout::Stacked,
            time_axis: TimeAxisMode::RelativeSeconds,
            title: None,
            y_unit: None,
            show_phase_flags: true,
        }
    }
}

/// Renders a publication-grade `plot1` (stacked multi-panel) or `plot2` (overlaid) SVG figure
/// with UTC/relative time axes, channel badges, amplitude units, and vertical SAC phase flags.
pub fn plot_waveforms_svg(stream: &Stream, opts: &WaveformPlotOptions) -> String {
    let w = opts.width.max(360.0);
    let traces: Vec<&Trace> = stream.traces.iter().filter(|t| !t.is_empty()).collect();
    if traces.is_empty() {
        let h = if opts.height > 0.0 {
            opts.height
        } else {
            280.0
        };
        return empty_svg_placeholder(w, h, "No waveform samples to display");
    }

    let n_panels = match opts.layout {
        WaveformLayout::Stacked => traces.len(),
        WaveformLayout::Overlaid => 1,
    };
    let h = if opts.height > 0.0 {
        opts.height.max(220.0)
    } else {
        match opts.layout {
            WaveformLayout::Stacked => (110.0 + (n_panels as f64) * 165.0).clamp(280.0, 2600.0),
            WaveformLayout::Overlaid => 420.0,
        }
    };

    let mut canvas = SvgCanvas::new(w, h);
    canvas.set_background("#ffffff");

    let ref_epoch = traces
        .iter()
        .map(|t| t.stats.starttime)
        .min()
        .unwrap_or(UtcTimestamp::UNIX_EPOCH);

    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for tr in &traces {
        let s0 = tr.stats.starttime.duration_secs_f64_since(ref_epoch);
        let s1 = s0 + tr.duration_s();
        if s0 < t_min {
            t_min = s0;
        }
        if s1 > t_max {
            t_max = s1;
        }
    }
    if !t_min.is_finite() || !t_max.is_finite() {
        t_min = 0.0;
        t_max = 1.0;
    }
    if (t_max - t_min).abs() < 1e-9 {
        t_max = t_min + 1.0;
    }

    let title_str = opts.title.clone().unwrap_or_else(|| {
        format!(
            "Seismic Waveforms — {} trace{} (Ref UTC: {})",
            traces.len(),
            if traces.len() == 1 { "" } else { "s" },
            ref_epoch.to_iso8601()
        )
    });
    canvas.text(
        w * 0.5,
        24.0,
        &title_str,
        &SvgStyle::label("#0f172a", 14.0, "middle").with_font_weight("700"),
    );

    let left = 84.0;
    let right_pad = 28.0;
    let top_start = 46.0;
    let bottom_pad = 54.0;
    let plot_w = (w - left - right_pad).max(120.0);
    let avail_h = (h - top_start - bottom_pad).max(80.0);

    match opts.layout {
        WaveformLayout::Stacked => {
            let gap = if n_panels > 1 { 16.0 } else { 0.0 };
            let panel_h = ((avail_h - gap * ((n_panels - 1) as f64)) / (n_panels as f64)).max(42.0);
            for (idx, tr) in traces.iter().enumerate() {
                let py_top = top_start + (idx as f64) * (panel_h + gap);
                let (ymin, ymax) = trace_y_bounds(tr);
                let vp = PlotViewport {
                    left,
                    top: py_top,
                    width: plot_w,
                    height: panel_h,
                    x_min: t_min,
                    x_max: t_max,
                    y_min: ymin,
                    y_max: ymax,
                };
                let is_bottom = idx + 1 == n_panels;
                let y_lbl = opts
                    .y_unit
                    .clone()
                    .unwrap_or_else(|| infer_trace_unit_label(tr));
                draw_time_panel_axes(
                    &mut canvas,
                    &vp,
                    ref_epoch,
                    opts.time_axis,
                    is_bottom,
                    &y_lbl,
                );

                let clip_id = format!("wf_clip_{idx}");
                canvas.define_clip_rect(&clip_id, vp.left, vp.top, vp.width, vp.height);
                canvas.begin_group(None, Some(&clip_id), None);

                let s0 = tr.stats.starttime.duration_secs_f64_since(ref_epoch);
                let poly = decimate_waveform_points(&tr.data, s0, tr.delta(), &vp);
                let stroke_col = PALETTE[idx % PALETTE.len()];
                canvas.polyline(&poly, &SvgStyle::stroke_only(stroke_col, 1.15));

                if opts.show_phase_flags {
                    if let Some(ref sac) = tr.stats.sac {
                        draw_sac_phase_flags(&mut canvas, sac, s0, &vp);
                    }
                }
                canvas.end_group();

                // Channel & rate badge in upper-left of panel:
                draw_badge(
                    &mut canvas,
                    vp.left + 8.0,
                    vp.top + 8.0,
                    &format!(
                        "{}  |  {} Hz  |  {} pts",
                        tr.stats.channel.seed_id(),
                        clean_num(tr.stats.sampling_rate),
                        tr.npts()
                    ),
                    stroke_col,
                );
            }
        }
        WaveformLayout::Overlaid => {
            let mut gmin = f64::INFINITY;
            let mut gmax = f64::NEG_INFINITY;
            for tr in &traces {
                let (mn, mx) = trace_y_bounds(tr);
                if mn < gmin {
                    gmin = mn;
                }
                if mx > gmax {
                    gmax = mx;
                }
            }
            if !gmin.is_finite() || !gmax.is_finite() || (gmax - gmin).abs() < 1e-30 {
                gmin = -1.0;
                gmax = 1.0;
            }
            let vp = PlotViewport {
                left,
                top: top_start,
                width: plot_w,
                height: avail_h,
                x_min: t_min,
                x_max: t_max,
                y_min: gmin,
                y_max: gmax,
            };
            let y_lbl = opts
                .y_unit
                .clone()
                .unwrap_or_else(|| infer_trace_unit_label(traces[0]));
            draw_time_panel_axes(&mut canvas, &vp, ref_epoch, opts.time_axis, true, &y_lbl);

            canvas.define_clip_rect("wf_over_clip", vp.left, vp.top, vp.width, vp.height);
            canvas.begin_group(None, Some("wf_over_clip"), None);
            for (idx, tr) in traces.iter().enumerate() {
                let s0 = tr.stats.starttime.duration_secs_f64_since(ref_epoch);
                let poly = decimate_waveform_points(&tr.data, s0, tr.delta(), &vp);
                let col = PALETTE[idx % PALETTE.len()];
                canvas.polyline(&poly, &SvgStyle::stroke_only(col, 1.15).with_opacity(0.88));
                if opts.show_phase_flags && idx == 0 {
                    if let Some(ref sac) = tr.stats.sac {
                        draw_sac_phase_flags(&mut canvas, sac, s0, &vp);
                    }
                }
            }
            canvas.end_group();

            // Legend in top-right:
            for (idx, tr) in traces.iter().take(8).enumerate() {
                let col = PALETTE[idx % PALETTE.len()];
                let lx = vp.left + vp.width - 170.0;
                let ly = vp.top + 16.0 + (idx as f64) * 16.0;
                canvas.line(lx, ly, lx + 18.0, ly, &SvgStyle::stroke_only(col, 2.0));
                canvas.text(
                    lx + 24.0,
                    ly,
                    &tr.stats.channel.seed_id(),
                    &SvgStyle::label("#1e293b", 10.5, "start").with_baseline("central"),
                );
            }
        }
    }

    canvas.finish()
}

fn infer_trace_unit_label(tr: &Trace) -> String {
    if let Some(ref sac) = tr.stats.sac {
        if let Some(code) = sac.get_int("idep") {
            match SacDepType::from_i32(code) {
                Some(SacDepType::DisplacementNm) => return String::from("Disp (m / nm)"),
                Some(SacDepType::VelocityNmPerSec) => return String::from("Vel (m/s)"),
                Some(SacDepType::AccelerationNmPerSec2) => return String::from("Acc (m/s^2)"),
                Some(SacDepType::Volts) => return String::from("Volts (V)"),
                _ => {}
            }
        }
    }
    String::from("Amplitude")
}

fn trace_y_bounds(tr: &Trace) -> (f64, f64) {
    let Some((mn, mx, _)) = tr.min_max_mean() else {
        return (-1.0, 1.0);
    };
    if !mn.is_finite() || !mx.is_finite() {
        return (-1.0, 1.0);
    }
    let span = mx - mn;
    if span.abs() < 1e-30 {
        let pad = mn.abs().max(1.0) * 0.1;
        (mn - pad, mx + pad)
    } else {
        (mn - 0.08 * span, mx + 0.14 * span)
    }
}

fn decimate_waveform_points(
    data: &[f64],
    t0_rel_s: f64,
    dt_s: f64,
    vp: &PlotViewport,
) -> Vec<(f64, f64)> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let max_buckets = (vp.width.max(100.0) as usize) * 2;
    if n <= max_buckets {
        let mut pts = Vec::with_capacity(n);
        for (i, &v) in data.iter().enumerate() {
            if v.is_finite() {
                let t = t0_rel_s + (i as f64) * dt_s;
                pts.push(vp.map_point(t, v));
            }
        }
        return pts;
    }

    // Peak-preserving min/max envelope per pixel bucket:
    let n_cols = vp.width.max(100.0) as usize;
    let mut pts = Vec::with_capacity(n_cols * 2 + 2);
    for col in 0..n_cols {
        let i0 = (col * n) / n_cols;
        let i1 = (((col + 1) * n) / n_cols).max(i0 + 1).min(n);
        let mut min_i = i0;
        let mut max_i = i0;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for (idx, &val) in data.iter().enumerate().take(i1).skip(i0) {
            if !val.is_finite() {
                continue;
            }
            if val < min_v {
                min_v = val;
                min_i = idx;
            }
            if val > max_v {
                max_v = val;
                max_i = idx;
            }
        }
        if !min_v.is_finite() {
            continue;
        }
        if min_i <= max_i {
            pts.push(vp.map_point(t0_rel_s + (min_i as f64) * dt_s, min_v));
            if max_i != min_i {
                pts.push(vp.map_point(t0_rel_s + (max_i as f64) * dt_s, max_v));
            }
        } else {
            pts.push(vp.map_point(t0_rel_s + (max_i as f64) * dt_s, max_v));
            pts.push(vp.map_point(t0_rel_s + (min_i as f64) * dt_s, min_v));
        }
    }
    pts
}

fn draw_sac_phase_flags(
    canvas: &mut SvgCanvas,
    sac: &SacHeader,
    trace_t0_s: f64,
    vp: &PlotViewport,
) {
    let b_off = sac.get_float("b").unwrap_or(0.0);
    let mut markers: Vec<(f64, String, &str)> = Vec::new();

    if let Some(o_val) = sac.get_float("o") {
        let lbl = sac.get_str("ko").unwrap_or("O");
        markers.push((trace_t0_s + (o_val - b_off), String::from(lbl), "#b45309"));
    }
    if let Some(a_val) = sac.get_float("a") {
        let lbl = sac.get_str("ka").unwrap_or("A");
        markers.push((trace_t0_s + (a_val - b_off), String::from(lbl), "#dc2626"));
    }
    if let Some(f_val) = sac.get_float("f") {
        let lbl = sac.get_str("kf").unwrap_or("F");
        markers.push((trace_t0_s + (f_val - b_off), String::from(lbl), "#475569"));
    }
    for k in 0..10 {
        let t_field = format!("t{k}");
        let kt_field = format!("kt{k}");
        if let Some(tv) = sac.get_float(&t_field) {
            let lbl = sac
                .get_str(&kt_field)
                .map(String::from)
                .unwrap_or_else(|| format!("T{k}"));
            markers.push((trace_t0_s + (tv - b_off), lbl, "#7c3aed"));
        }
    }

    let bottom = vp.top + vp.height;
    for (idx, (t_x, label, color)) in markers.iter().enumerate() {
        if *t_x < vp.x_min || *t_x > vp.x_max {
            continue;
        }
        let px = vp.map_x(*t_x);
        canvas.line(
            px,
            vp.top,
            px,
            bottom,
            &SvgStyle::stroke_only(color, 1.15).with_dasharray("4,3"),
        );
        let stagger_y = vp.top + 12.0 + ((idx % 3) as f64) * 12.0;
        canvas.text(
            px + 3.0,
            stagger_y,
            label,
            &SvgStyle::label(color, 10.0, "start").with_font_weight("700"),
        );
    }
}

fn draw_time_panel_axes(
    canvas: &mut SvgCanvas,
    vp: &PlotViewport,
    ref_epoch: UtcTimestamp,
    mode: TimeAxisMode,
    show_x_labels: bool,
    y_label: &str,
) {
    let grid_style = SvgStyle::stroke_only("#e2e8f0", 0.75).with_dasharray("3,3");
    let tick_style = SvgStyle::stroke_only("#334155", 1.0);
    let frame_style = SvgStyle::stroke_only("#0f172a", 1.15);
    let x_lbl_style = SvgStyle::label("#334155", 10.5, "middle").with_baseline("hanging");
    let y_lbl_style = SvgStyle::label("#334155", 10.0, "end").with_baseline("central");
    let title_style = SvgStyle::label("#0f172a", 11.5, "middle").with_font_weight("600");

    let bottom = vp.top + vp.height;
    let right = vp.left + vp.width;

    for tx in nice_ticks(vp.x_min, vp.x_max, 8) {
        let px = vp.map_x(tx);
        if px >= vp.left - 0.5 && px <= right + 0.5 {
            canvas.line(px, vp.top, px, bottom, &grid_style);
            canvas.line(px, bottom, px, bottom + 4.5, &tick_style);
            if show_x_labels {
                let txt = match mode {
                    TimeAxisMode::RelativeSeconds => clean_num(tx),
                    TimeAxisMode::UtcClock => match ref_epoch.add_secs_f64(tx) {
                        Ok(ts) => {
                            let (_, _, _, hh, mm, ss, nano) = ts.to_ymd_hms_nano();
                            if nano == 0 {
                                format!("{hh:02}:{mm:02}:{ss:02}Z")
                            } else {
                                format!("{hh:02}:{mm:02}:{ss:02}.{:02}Z", nano / 10_000_000)
                            }
                        }
                        Err(_) => clean_num(tx),
                    },
                };
                canvas.text(px, bottom + 7.0, &txt, &x_lbl_style);
            }
        }
    }

    for ty in nice_ticks(vp.y_min, vp.y_max, 4) {
        let py = vp.map_y(ty);
        if py >= vp.top - 0.5 && py <= bottom + 0.5 {
            canvas.line(vp.left, py, right, py, &grid_style);
            canvas.line(vp.left - 4.5, py, vp.left, py, &tick_style);
            canvas.text(vp.left - 7.0, py, &clean_num(ty), &y_lbl_style);
        }
    }

    canvas.rect(vp.left, vp.top, vp.width, vp.height, &frame_style);

    if show_x_labels {
        let x_title = match mode {
            TimeAxisMode::RelativeSeconds => "Time (s)",
            TimeAxisMode::UtcClock => "UTC Time",
        };
        canvas.text(
            vp.left + 0.5 * vp.width,
            bottom + 26.0,
            x_title,
            &title_style,
        );
    }
    if !y_label.is_empty() {
        canvas.text_rotated(20.0, vp.top + 0.5 * vp.height, -90.0, y_label, &title_style);
    }
}

fn draw_badge(canvas: &mut SvgCanvas, x: f64, y: f64, text: &str, accent_color: &str) {
    let est_w = (text.len() as f64) * 6.2 + 16.0;
    canvas.rect(
        x,
        y,
        est_w,
        18.0,
        &SvgStyle {
            fill: Some(String::from("#f8fafc")),
            stroke: Some(String::from("#cbd5e1")),
            stroke_width: Some(0.8),
            opacity: Some(0.92),
            ..SvgStyle::default()
        },
    );
    canvas.rect(x, y, 3.5, 18.0, &SvgStyle::fill_only(accent_color));
    canvas.text(
        x + 8.0,
        y + 9.5,
        text,
        &SvgStyle::label("#0f172a", 10.0, "start")
            .with_font_weight("600")
            .with_baseline("central"),
    );
}

// ============================================================================
// 2. Seismic Record Section (`plot_record_section_svg`)
// ============================================================================

/// Distance coordinate unit for [`plot_record_section_svg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceUnit {
    /// Epicentral great-circle arc distance in degrees (`GCARC`).
    #[default]
    Degrees,
    /// Epicentral WGS84 surface distance in kilometers (`DIST`).
    Kilometers,
}

/// Styling and reduction options for [`plot_record_section_svg`].
#[derive(Debug, Clone)]
pub struct RecordSectionOptions {
    /// SVG width in pixels (`1000.0` default).
    pub width: f64,
    /// SVG height in pixels (`680.0` default).
    pub height: f64,
    /// Distance axis unit (`Degrees` or `Kilometers`).
    pub distance_unit: DistanceUnit,
    /// Optional reduction velocity in `km/s` (when `Some(v_red)`, plots $t - \Delta_{\text{km}} / v_{\text{red}}$).
    pub reduction_velocity_km_s: Option<f64>,
    /// Per-trace wiggle excursion scale in distance units (`0.0` = auto-scale).
    pub trace_scale: f64,
    /// Optional figure title.
    pub title: Option<String>,
}

impl Default for RecordSectionOptions {
    fn default() -> Self {
        Self {
            width: 1000.0,
            height: 680.0,
            distance_unit: DistanceUnit::Degrees,
            reduction_velocity_km_s: None,
            trace_scale: 0.0,
            title: None,
        }
    }
}

/// Renders an epicentral-distance (`deg` or `km`) reduced or unreduced seismic record section
/// with normalized trace wiggles and overlaid theoretical `TauP` travel-time curves.
pub fn plot_record_section_svg(
    stream: &Stream,
    taup_curves: &[PhaseTravelTimeCurve],
    opts: &RecordSectionOptions,
) -> String {
    let w = opts.width.max(400.0);
    let h = opts.height.max(320.0);
    let traces: Vec<&Trace> = stream.traces.iter().filter(|t| !t.is_empty()).collect();
    if traces.is_empty() && taup_curves.is_empty() {
        return empty_svg_placeholder(w, h, "No record section traces or TauP curves");
    }

    let mut canvas = SvgCanvas::new(w, h);
    canvas.set_background("#ffffff");

    // Extract (distance_deg, distance_km, origin_rel_start_s, &Trace) per trace:
    let mut entries: Vec<(f64, f64, f64, &Trace)> = Vec::with_capacity(traces.len());
    for (idx, tr) in traces.iter().enumerate() {
        let (dist_deg, dist_km, start_after_o_s) = extract_trace_section_coords(tr, idx);
        entries.push((dist_deg, dist_km, start_after_o_s, *tr));
    }

    let mut d_min = f64::INFINITY;
    let mut d_max = f64::NEG_INFINITY;
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;

    let reduce_shift = |dist_km: f64| -> f64 {
        match opts.reduction_velocity_km_s {
            Some(v) if v.is_finite() && v.abs() > 1e-6 => dist_km / v,
            _ => 0.0,
        }
    };

    for &(d_deg, d_km, s0_o, tr) in &entries {
        let d_val = match opts.distance_unit {
            DistanceUnit::Degrees => d_deg,
            DistanceUnit::Kilometers => d_km,
        };
        let r_shift = reduce_shift(d_km);
        let t0 = s0_o - r_shift;
        let t1 = t0 + tr.duration_s();
        d_min = d_min.min(d_val);
        d_max = d_max.max(d_val);
        t_min = t_min.min(t0);
        t_max = t_max.max(t1);
    }

    if entries.is_empty() {
        for curve in taup_curves {
            for br in &curve.branches {
                for pt in &br.points {
                    let d_km = pt.dist_deg * earthsnail_core::KM_PER_DEG;
                    let d_val = match opts.distance_unit {
                        DistanceUnit::Degrees => pt.dist_deg,
                        DistanceUnit::Kilometers => d_km,
                    };
                    let t_val = pt.time_s - reduce_shift(d_km);
                    d_min = d_min.min(d_val);
                    d_max = d_max.max(d_val);
                    t_min = t_min.min(t_val);
                    t_max = t_max.max(t_val);
                }
            }
        }
    }

    if !d_min.is_finite() || !d_max.is_finite() {
        d_min = 0.0;
        d_max = 90.0;
    }
    let d_span = (d_max - d_min).abs().max(2.0);
    let wiggle_amp = if opts.trace_scale > 0.0 {
        opts.trace_scale
    } else {
        (d_span / (entries.len().max(4) as f64)).max(0.6) * 0.65
    };
    d_min -= wiggle_amp * 1.4;
    d_max += wiggle_amp * 1.4;

    if !t_min.is_finite() || !t_max.is_finite() || (t_max - t_min).abs() < 1e-6 {
        t_min = 0.0;
        t_max = 600.0;
    }

    let vp = PlotViewport {
        left: 82.0,
        top: 48.0,
        width: (w - 190.0).max(160.0),
        height: (h - 104.0).max(140.0),
        x_min: t_min,
        x_max: t_max,
        y_min: d_min,
        y_max: d_max,
    };

    let title_str = opts
        .title
        .clone()
        .unwrap_or_else(|| match opts.reduction_velocity_km_s {
            Some(v) => format!("Seismic Record Section (Vred = {v:.2} km/s)"),
            None => String::from("Seismic Record Section & Theoretical TauP Curves"),
        });
    canvas.text(
        w * 0.5,
        24.0,
        &title_str,
        &SvgStyle::label("#0f172a", 14.0, "middle").with_font_weight("700"),
    );

    let x_lbl = match opts.reduction_velocity_km_s {
        Some(v) => format!("Reduced Travel Time T - X/{v:.2} (s)"),
        None => String::from("Travel Time Relative to Origin (s)"),
    };
    let y_lbl = match opts.distance_unit {
        DistanceUnit::Degrees => "Epicentral Distance (deg)",
        DistanceUnit::Kilometers => "Epicentral Distance (km)",
    };
    canvas.draw_axes(&vp, &x_lbl, y_lbl, 8, 7);

    canvas.define_clip_rect("rs_clip", vp.left, vp.top, vp.width, vp.height);
    canvas.begin_group(None, Some("rs_clip"), None);

    // 1. Draw waveform wiggles:
    let max_pts = (vp.width as usize) * 2;
    for &(d_deg, d_km, s0_o, tr) in &entries {
        let d_base = match opts.distance_unit {
            DistanceUnit::Degrees => d_deg,
            DistanceUnit::Kilometers => d_km,
        };
        let t_start = s0_o - reduce_shift(d_km);
        let Some((mn, mx, mean)) = tr.min_max_mean() else {
            continue;
        };
        let peak = (mx - mean).abs().max((mn - mean).abs()).max(1e-30);
        let step = (tr.npts() / max_pts).max(1);
        let mut pts = Vec::with_capacity(tr.npts() / step + 2);
        for (i, &val) in tr.data.iter().enumerate().step_by(step) {
            if val.is_finite() {
                let t = t_start + (i as f64) * tr.delta();
                let y = d_base + ((val - mean) / peak) * wiggle_amp;
                pts.push(vp.map_point(t, y));
            }
        }
        canvas.polyline(
            &pts,
            &SvgStyle::stroke_only("#1e293b", 0.95).with_opacity(0.85),
        );
    }

    // 2. Overlay TauP travel-time curves:
    for (c_idx, curve) in taup_curves.iter().enumerate() {
        let col = PALETTE[c_idx % PALETTE.len()];
        for br in &curve.branches {
            let mut cpts = Vec::with_capacity(br.points.len());
            for pt in &br.points {
                let d_deg_fold = fold_great_circle_deg(pt.dist_deg);
                let d_km = d_deg_fold * earthsnail_core::KM_PER_DEG;
                let d_val = match opts.distance_unit {
                    DistanceUnit::Degrees => d_deg_fold,
                    DistanceUnit::Kilometers => d_km,
                };
                let t_val = pt.time_s - reduce_shift(d_km);
                cpts.push(vp.map_point(t_val, d_val));
            }
            canvas.polyline(&cpts, &SvgStyle::stroke_only(col, 1.8));
        }
    }
    canvas.end_group();

    // Station labels on right edge + Phase legend:
    let right_x = vp.left + vp.width + 6.0;
    for &(d_deg, d_km, _, tr) in entries.iter().take(24) {
        let d_base = match opts.distance_unit {
            DistanceUnit::Degrees => d_deg,
            DistanceUnit::Kilometers => d_km,
        };
        let py = vp.map_y(d_base);
        if py >= vp.top && py <= vp.top + vp.height {
            let sta = if tr.stats.channel.station.is_empty() {
                tr.stats.channel.seed_id()
            } else {
                format!("{}.{}", tr.stats.channel.station, tr.stats.channel.channel)
            };
            canvas.text(
                right_x,
                py,
                &sta,
                &SvgStyle::label("#334155", 9.5, "start").with_baseline("central"),
            );
        }
    }

    for (c_idx, curve) in taup_curves.iter().take(12).enumerate() {
        let col = PALETTE[c_idx % PALETTE.len()];
        let lx = vp.left + 14.0;
        let ly = vp.top + 16.0 + (c_idx as f64) * 15.0;
        canvas.line(lx, ly, lx + 18.0, ly, &SvgStyle::stroke_only(col, 2.2));
        canvas.text(
            lx + 24.0,
            ly,
            &curve.name,
            &SvgStyle::label(col, 10.5, "start")
                .with_font_weight("700")
                .with_baseline("central"),
        );
    }

    canvas.finish()
}

fn extract_trace_section_coords(tr: &Trace, fallback_idx: usize) -> (f64, f64, f64) {
    if let Some(ref sac) = tr.stats.sac {
        let mut gcarc = sac.get_float("gcarc");
        let mut dist_km = sac.get_float("dist");
        if gcarc.is_none() || dist_km.is_none() {
            if let (Some(evla), Some(evlo), Some(stla), Some(stlo)) = (
                sac.get_float("evla"),
                sac.get_float("evlo"),
                sac.get_float("stla"),
                sac.get_float("stlo"),
            ) {
                if evla.abs() <= 90.0 && stla.abs() <= 90.0 {
                    let g = gps2dist_azimuth(evla, evlo, stla, stlo);
                    gcarc = gcarc.or(Some(g.gcarc_deg));
                    dist_km = dist_km.or(Some(g.distance_m * 1e-3));
                }
            }
        }
        let d_deg = match (gcarc, dist_km) {
            (Some(deg), _) => deg,
            (None, Some(km)) => kilometer2degrees(km),
            (None, None) => ((fallback_idx + 1) as f64) * 5.0,
        };
        let d_km = dist_km.unwrap_or(d_deg * earthsnail_core::KM_PER_DEG);
        // Time of sample 0 after event origin O:
        // sample 0 is at `b` relative to reftime; origin is at `o` relative to reftime -> `b - o`.
        let b_val = sac.get_float("b").unwrap_or(0.0);
        let o_val = sac.get_float("o").unwrap_or(0.0);
        return (d_deg, d_km, b_val - o_val);
    }
    let deg = ((fallback_idx + 1) as f64) * 5.0;
    (deg, deg * earthsnail_core::KM_PER_DEG, 0.0)
}

#[inline]
fn fold_great_circle_deg(unwrapped_deg: f64) -> f64 {
    let r = math::rem_euclid(unwrapped_deg, 360.0);
    if r > 180.0 {
        360.0 - r
    } else {
        r
    }
}

// ============================================================================
// 3. Spectrogram + Top Waveform Strip (`plot_spectrogram_svg`)
// ============================================================================

/// Time-frequency decomposition estimator for [`plot_spectrogram_svg`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrogramMethod {
    /// Short-Time Fourier Transform (Hann-windowed sliding RFFT).
    Stft {
        /// Sliding window length in samples (`0` = automatic).
        win_samples: usize,
    },
    /// Continuous Wavelet Transform with complex Morlet wavelet (`omega0 = 6.0` standard).
    MorletCwt {
        /// Nondimensional Morlet carrier wavenumber (`6.0` default).
        omega0: f64,
    },
}

impl Default for SpectrogramMethod {
    fn default() -> Self {
        Self::MorletCwt { omega0: 6.0 }
    }
}

/// Frequency axis scaling for [`plot_spectrogram_svg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrequencyScale {
    /// Logarithmic frequency axis ($\log_{10} f$).
    #[default]
    Log10,
    /// Linear frequency axis ($f$ in Hz).
    Linear,
}

/// Configuration options for [`plot_spectrogram_svg`].
#[derive(Debug, Clone)]
pub struct SpectrogramOptions {
    /// SVG width in pixels (`1000.0` default).
    pub width: f64,
    /// SVG height in pixels (`580.0` default).
    pub height: f64,
    /// Estimator (`MorletCwt` or `Stft`).
    pub method: SpectrogramMethod,
    /// Frequency axis scale (`Log10` or `Linear`).
    pub freq_scale: FrequencyScale,
    /// Optional minimum frequency in Hz (`None` = auto).
    pub fmin_hz: Option<f64>,
    /// Optional maximum frequency in Hz (`None` = Nyquist).
    pub fmax_hz: Option<f64>,
    /// Dynamic range floor below peak power in dB (`60.0` dB default).
    pub dynamic_range_db: f64,
    /// Optional figure title.
    pub title: Option<String>,
}

impl Default for SpectrogramOptions {
    fn default() -> Self {
        Self {
            width: 1000.0,
            height: 580.0,
            method: SpectrogramMethod::default(),
            freq_scale: FrequencyScale::Log10,
            fmin_hz: None,
            fmax_hz: None,
            dynamic_range_db: 60.0,
            title: None,
        }
    }
}

/// Renders a two-panel synchronized waveform + Morlet-CWT / STFT color-mapped SVG spectrogram.
pub fn plot_spectrogram_svg(trace: &Trace, opts: &SpectrogramOptions) -> String {
    let w = opts.width.max(420.0);
    let h = opts.height.max(340.0);
    if trace.data.len() < 4 || trace.stats.sampling_rate <= 0.0 {
        return empty_svg_placeholder(w, h, "Trace requires >= 4 samples for spectrogram");
    }

    let mut canvas = SvgCanvas::new(w, h);
    canvas.set_background("#ffffff");

    let fs = trace.stats.sampling_rate;
    let nyq = 0.5 * fs;
    let dur_s = trace.duration_s().max(trace.delta());
    let fmax = opts.fmax_hz.unwrap_or(nyq).clamp(1e-4, nyq);
    let default_fmin = (2.0 / dur_s).clamp(1e-3, 0.25 * fmax);
    let fmin = opts
        .fmin_hz
        .unwrap_or(default_fmin)
        .clamp(1e-5, (0.8 * fmax).max(1e-5));

    let title = opts.title.clone().unwrap_or_else(|| {
        let m_str = match opts.method {
            SpectrogramMethod::MorletCwt { .. } => "Morlet CWT",
            SpectrogramMethod::Stft { .. } => "STFT",
        };
        format!(
            "Spectrogram ({m_str}) — {} ({} Hz)",
            trace.stats.channel.seed_id(),
            clean_num(fs)
        )
    });
    canvas.text(
        w * 0.5,
        22.0,
        &title,
        &SvgStyle::label("#0f172a", 13.5, "middle").with_font_weight("700"),
    );

    let left = 80.0;
    let plot_w = (w - left - 96.0).max(160.0);
    let wf_vp = PlotViewport {
        left,
        top: 42.0,
        width: plot_w,
        height: (h * 0.23).max(70.0),
        x_min: 0.0,
        x_max: dur_s,
        y_min: trace_y_bounds(trace).0,
        y_max: trace_y_bounds(trace).1,
    };
    draw_time_panel_axes(
        &mut canvas,
        &wf_vp,
        trace.stats.starttime,
        TimeAxisMode::RelativeSeconds,
        false,
        "Amp",
    );
    let wf_pts = decimate_waveform_points(&trace.data, 0.0, trace.delta(), &wf_vp);
    canvas.define_clip_rect(
        "spec_wf_clip",
        wf_vp.left,
        wf_vp.top,
        wf_vp.width,
        wf_vp.height,
    );
    canvas.begin_group(None, Some("spec_wf_clip"), None);
    canvas.polyline(&wf_pts, &SvgStyle::stroke_only("#0f172a", 1.05));
    canvas.end_group();

    let spec_top = wf_vp.top + wf_vp.height + 18.0;
    let spec_h = (h - spec_top - 52.0).max(120.0);
    let n_rows = if cfg!(miri) { 16usize } else { 48usize };
    let n_cols = if cfg!(miri) {
        24usize.min(trace.npts())
    } else {
        140usize.min(trace.npts()).max(8)
    };

    let row_freqs: Vec<f64> = (0..n_rows)
        .map(|r| {
            let frac = if n_rows <= 1 {
                0.5
            } else {
                (r as f64) / ((n_rows - 1) as f64)
            };
            match opts.freq_scale {
                FrequencyScale::Log10 => {
                    let lf0 = math::log10(fmin);
                    let lf1 = math::log10(fmax);
                    math::powf(10.0, lf0 + frac * (lf1 - lf0))
                }
                FrequencyScale::Linear => fmin + frac * (fmax - fmin),
            }
        })
        .collect();

    let power_grid = compute_spectrogram_db_grid(trace, &row_freqs, n_cols, opts.method);
    let mut max_db = f64::NEG_INFINITY;
    for row in &power_grid {
        for &v in row {
            if v.is_finite() && v > max_db {
                max_db = v;
            }
        }
    }
    if !max_db.is_finite() {
        max_db = 0.0;
    }
    let dyn_db = opts.dynamic_range_db.abs().max(10.0);
    let min_db = max_db - dyn_db;

    let cell_w = plot_w / (n_cols as f64);
    let cell_h = spec_h / (n_rows as f64);
    for (r, row) in power_grid.iter().enumerate() {
        // Row 0 = lowest frequency -> bottom of spectrogram panel!
        let cy = spec_top + spec_h - ((r + 1) as f64) * cell_h;
        for (c, &db_val) in row.iter().enumerate() {
            let norm = ((db_val - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
            let hex = viridis_hex(norm);
            let cx = left + (c as f64) * cell_w;
            canvas.rect(
                cx,
                cy,
                cell_w + 0.45,
                cell_h + 0.45,
                &SvgStyle::fill_only(&hex),
            );
        }
    }

    // Spectrogram frame + X/Y ticks:
    let (y_min_ax, y_max_ax, y_title) = match opts.freq_scale {
        FrequencyScale::Log10 => (math::log10(fmin), math::log10(fmax), "Frequency (Hz, log)"),
        FrequencyScale::Linear => (fmin, fmax, "Frequency (Hz)"),
    };
    let spec_vp = PlotViewport {
        left,
        top: spec_top,
        width: plot_w,
        height: spec_h,
        x_min: 0.0,
        x_max: dur_s,
        y_min: y_min_ax,
        y_max: y_max_ax,
    };
    let bottom = spec_vp.top + spec_vp.height;
    let tick_st = SvgStyle::stroke_only("#334155", 1.0);
    for tx in nice_ticks(0.0, dur_s, 8) {
        let px = spec_vp.map_x(tx);
        canvas.line(px, bottom, px, bottom + 5.0, &tick_st);
        canvas.text(
            px,
            bottom + 8.0,
            &clean_num(tx),
            &SvgStyle::label("#334155", 10.5, "middle").with_baseline("hanging"),
        );
    }
    match opts.freq_scale {
        FrequencyScale::Log10 => {
            for &f_tick in &[
                0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
            ] {
                if f_tick >= fmin * 0.999 && f_tick <= fmax * 1.001 {
                    let py = spec_vp.map_y(math::log10(f_tick));
                    canvas.line(left - 5.0, py, left, py, &tick_st);
                    canvas.text(
                        left - 8.0,
                        py,
                        &clean_num(f_tick),
                        &SvgStyle::label("#334155", 10.0, "end").with_baseline("central"),
                    );
                }
            }
        }
        FrequencyScale::Linear => {
            for fy in nice_ticks(fmin, fmax, 6) {
                let py = spec_vp.map_y(fy);
                canvas.line(left - 5.0, py, left, py, &tick_st);
                canvas.text(
                    left - 8.0,
                    py,
                    &clean_num(fy),
                    &SvgStyle::label("#334155", 10.0, "end").with_baseline("central"),
                );
            }
        }
    }
    canvas.rect(
        spec_vp.left,
        spec_vp.top,
        spec_vp.width,
        spec_vp.height,
        &SvgStyle::stroke_only("#0f172a", 1.2),
    );
    canvas.text(
        spec_vp.left + 0.5 * spec_vp.width,
        bottom + 28.0,
        "Time (s)",
        &SvgStyle::label("#0f172a", 11.5, "middle").with_font_weight("600"),
    );
    canvas.text_rotated(
        20.0,
        spec_vp.top + 0.5 * spec_vp.height,
        -90.0,
        y_title,
        &SvgStyle::label("#0f172a", 11.5, "middle").with_font_weight("600"),
    );

    // Right colorbar:
    draw_vertical_colorbar(
        &mut canvas,
        left + plot_w + 18.0,
        spec_top,
        14.0,
        spec_h,
        min_db,
        max_db,
        "Power (dB)",
        viridis_hex,
    );

    canvas.finish()
}

fn compute_spectrogram_db_grid(
    trace: &Trace,
    row_freqs: &[f64],
    n_cols: usize,
    method: SpectrogramMethod,
) -> Vec<Vec<f64>> {
    let npts = trace.npts();
    let fs = trace.stats.sampling_rate;
    let n_rows = row_freqs.len();

    match method {
        SpectrogramMethod::MorletCwt { omega0 } => {
            // Downsample trace mildly if huge (> 8192 samples) so CWT renders in < 15 ms:
            let stride = (npts / 4096).max(1);
            let sub_data: Vec<f64> = trace.data.iter().copied().step_by(stride).collect();
            let sub_fs = fs / (stride as f64);
            let clamped_freqs: Vec<f64> = row_freqs
                .iter()
                .map(|&f| f.min(0.49 * sub_fs).max(1e-5))
                .collect();
            if let Ok(scalogram) = cwt_morlet(&sub_data, sub_fs, &clamped_freqs, omega0) {
                let m = sub_data.len();
                let mut out = vec![vec![-200.0_f64; n_cols]; n_rows];
                for (row_out, row_c) in out.iter_mut().zip(scalogram.iter()) {
                    for (c, cell) in row_out.iter_mut().enumerate() {
                        let i0 = (c * m) / n_cols;
                        let i1 = (((c + 1) * m) / n_cols).max(i0 + 1).min(m);
                        let mut avg_p = 0.0_f64;
                        for z in &row_c[i0..i1] {
                            avg_p += z.norm_sqr();
                        }
                        avg_p /= (i1 - i0) as f64;
                        *cell = 10.0 * math::log10(avg_p.max(1e-30));
                    }
                }
                return out;
            }
            vec![vec![-120.0; n_cols]; n_rows]
        }
        SpectrogramMethod::Stft { win_samples } => {
            let wlen = if win_samples >= 8 {
                win_samples.min(npts)
            } else {
                (npts / 12).clamp(16, 512).min(npts)
            };
            let win = hann_window(wlen);
            let freqs = rfftfreq(wlen, 1.0 / fs).unwrap_or_else(|_| vec![0.0; wlen / 2 + 1]);
            let mut out = vec![vec![-200.0_f64; n_cols]; n_rows];
            let mut seg = vec![0.0_f64; wlen];
            for c in 0..n_cols {
                let start = if n_cols <= 1 || npts <= wlen {
                    0
                } else {
                    (c * (npts - wlen)) / (n_cols - 1)
                };
                seg.copy_from_slice(&trace.data[start..start + wlen]);
                demean_in_place(&mut seg);
                for (s, &w_val) in seg.iter_mut().zip(win.iter()) {
                    *s *= w_val;
                }
                if let Ok(spec) = rfft(&seg, Some(wlen)) {
                    for (row_out, &f_target) in out.iter_mut().zip(row_freqs.iter()) {
                        let mut best_k = 1.min(spec.len().saturating_sub(1));
                        let mut best_d = f64::INFINITY;
                        for (k, &fk) in freqs.iter().enumerate().skip(1) {
                            let d = (fk - f_target).abs();
                            if d < best_d {
                                best_d = d;
                                best_k = k;
                            }
                        }
                        let p = spec[best_k].norm_sqr() / (fs * (wlen as f64));
                        row_out[c] = 10.0 * math::log10(p.max(1e-30));
                    }
                }
            }
            out
        }
    }
}

// ============================================================================
// 4. McNamara & Buland PPSD + Peterson NLNM/NHNM (`plot_ppsd_svg`)
// ============================================================================

/// Configuration options for [`plot_ppsd_svg`].
#[derive(Debug, Clone)]
pub struct PpsdPlotOptions {
    /// SVG width in pixels (`980.0` default).
    pub width: f64,
    /// SVG height in pixels (`580.0` default).
    pub height: f64,
    /// Maximum histogram probability `%` saturated on the colorbar (`30.0` % default).
    pub max_percent: f64,
    /// Overlay mean curve (`true` default).
    pub show_mean: bool,
    /// Overlay histogram mode curve (`true` default).
    pub show_mode: bool,
    /// Optional figure title.
    pub title: Option<String>,
}

impl Default for PpsdPlotOptions {
    fn default() -> Self {
        Self {
            width: 980.0,
            height: 580.0,
            max_percent: 30.0,
            show_mean: true,
            show_mode: true,
            title: None,
        }
    }
}

/// Renders a McNamara & Buland (2004) Probabilistic Power Spectral Density (`PPSD`) period-dB
/// plot with 2D probability heatmap, Peterson (1993) `NLNM` & `NHNM` reference bounds, and
/// Mean/Mode summary curves.
pub fn plot_ppsd_svg(ppsd: &Ppsd, opts: &PpsdPlotOptions) -> String {
    let w = opts.width.max(420.0);
    let h = opts.height.max(340.0);
    let periods = ppsd.periods_s();
    let db_edges = ppsd.db_bin_edges();
    if periods.is_empty() || db_edges.len() < 2 {
        return empty_svg_placeholder(w, h, "PPSD has no period/dB bins");
    }

    let mut canvas = SvgCanvas::new(w, h);
    canvas.set_background("#ffffff");

    let p_min = periods.first().copied().unwrap_or(0.05).max(1e-4);
    let p_max = periods.last().copied().unwrap_or(100.0).max(p_min * 2.0);
    let lp_min = math::log10(p_min);
    let lp_max = math::log10(p_max);
    let db_min = db_edges[0];
    let db_max = *db_edges.last().unwrap_or(&-50.0);

    let vp = PlotViewport {
        left: 82.0,
        top: 48.0,
        width: (w - 186.0).max(180.0),
        height: (h - 108.0).max(150.0),
        x_min: lp_min,
        x_max: lp_max,
        y_min: db_min,
        y_max: db_max,
    };

    let title = opts.title.clone().unwrap_or_else(|| {
        format!(
            "Probabilistic Power Spectral Density (PPSD — {} segment{})",
            ppsd.num_segments(),
            if ppsd.num_segments() == 1 { "" } else { "s" }
        )
    });
    canvas.text(
        w * 0.5,
        24.0,
        &title,
        &SvgStyle::label("#0f172a", 14.0, "middle").with_font_weight("700"),
    );

    canvas.define_clip_rect("ppsd_clip", vp.left, vp.top, vp.width, vp.height);
    canvas.begin_group(None, Some("ppsd_clip"), None);

    // 1. 2D Probability Density Histogram Heatmap:
    let hist_pct = ppsd.histogram_percent();
    let max_pct = opts.max_percent.max(5.0);
    let n_per = periods.len();
    for (b, row_pct) in hist_pct.iter().enumerate() {
        let lp_c = math::log10(periods[b]);
        let lp_left = if b == 0 {
            if n_per > 1 {
                lp_c - 0.5 * (math::log10(periods[1]) - lp_c)
            } else {
                lp_min
            }
        } else {
            0.5 * (math::log10(periods[b - 1]) + lp_c)
        };
        let lp_right = if b + 1 >= n_per {
            if n_per > 1 {
                lp_c + 0.5 * (lp_c - math::log10(periods[b - 1]))
            } else {
                lp_max
            }
        } else {
            0.5 * (lp_c + math::log10(periods[b + 1]))
        };
        let x0 = vp.map_x(lp_left);
        let x1 = vp.map_x(lp_right);
        let cw = (x1 - x0).abs().max(1.0) + 0.4;

        for (d_idx, &pct) in row_pct.iter().enumerate() {
            if pct <= 0.0 {
                continue;
            }
            let y_top = vp.map_y(db_edges[d_idx + 1]);
            let y_bot = vp.map_y(db_edges[d_idx]);
            let ch = (y_bot - y_top).abs().max(1.0) + 0.4;
            let norm = (pct / max_pct).clamp(0.0, 1.0);
            canvas.rect(
                x0.min(x1),
                y_top.min(y_bot),
                cw,
                ch,
                &SvgStyle::fill_only(&ppsd_prob_hex(norm)),
            );
        }
    }

    // 2. Peterson (1993) NLNM & NHNM reference models:
    let (nlnm_p, nlnm_db) = get_nlnm();
    let nlnm_pts: Vec<(f64, f64)> = nlnm_p
        .iter()
        .zip(nlnm_db.iter())
        .filter(|(&p, _)| p >= p_min * 0.8 && p <= p_max * 1.25)
        .map(|(&p, &db)| vp.map_point(math::log10(p), db))
        .collect();
    canvas.polyline(&nlnm_pts, &SvgStyle::stroke_only("#334155", 2.1));

    let (nhnm_p, nhnm_db) = get_nhnm();
    let nhnm_pts: Vec<(f64, f64)> = nhnm_p
        .iter()
        .zip(nhnm_db.iter())
        .filter(|(&p, _)| p >= p_min * 0.8 && p <= p_max * 1.25)
        .map(|(&p, &db)| vp.map_point(math::log10(p), db))
        .collect();
    canvas.polyline(&nhnm_pts, &SvgStyle::stroke_only("#334155", 2.1));

    // 3. Mean & Mode curves:
    if opts.show_mean {
        if let Ok(mean_db) = ppsd.mean_curve_db() {
            let pts: Vec<(f64, f64)> = periods
                .iter()
                .zip(mean_db.iter())
                .map(|(&p, &db)| vp.map_point(math::log10(p), db))
                .collect();
            canvas.polyline(&pts, &SvgStyle::stroke_only("#0f172a", 1.9));
        }
    }
    if opts.show_mode {
        if let Ok(mode_db) = ppsd.mode_curve_db() {
            let pts: Vec<(f64, f64)> = periods
                .iter()
                .zip(mode_db.iter())
                .map(|(&p, &db)| vp.map_point(math::log10(p), db))
                .collect();
            canvas.polyline(
                &pts,
                &SvgStyle::stroke_only("#dc2626", 1.7).with_dasharray("5,3"),
            );
        }
    }
    canvas.end_group();

    // Logarithmic period grid & axes:
    let grid_st = SvgStyle::stroke_only("#cbd5e1", 0.7).with_dasharray("3,3");
    let tick_st = SvgStyle::stroke_only("#334155", 1.0);
    let bottom = vp.top + vp.height;
    let right = vp.left + vp.width;

    for &p_tick in &[
        0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
    ] {
        let lp = math::log10(p_tick);
        if lp >= lp_min - 1e-6 && lp <= lp_max + 1e-6 {
            let px = vp.map_x(lp);
            canvas.line(px, vp.top, px, bottom, &grid_st);
            canvas.line(px, bottom, px, bottom + 5.0, &tick_st);
            canvas.text(
                px,
                bottom + 8.0,
                &clean_num(p_tick),
                &SvgStyle::label("#334155", 10.5, "middle").with_baseline("hanging"),
            );
        }
    }
    for dy in nice_ticks(db_min, db_max, 7) {
        let py = vp.map_y(dy);
        if py >= vp.top - 0.5 && py <= bottom + 0.5 {
            canvas.line(vp.left, py, right, py, &grid_st);
            canvas.line(vp.left - 5.0, py, vp.left, py, &tick_st);
            canvas.text(
                vp.left - 8.0,
                py,
                &clean_num(dy),
                &SvgStyle::label("#334155", 10.5, "end").with_baseline("central"),
            );
        }
    }
    canvas.rect(
        vp.left,
        vp.top,
        vp.width,
        vp.height,
        &SvgStyle::stroke_only("#0f172a", 1.2),
    );
    canvas.text(
        vp.left + 0.5 * vp.width,
        bottom + 28.0,
        "Period (s)",
        &SvgStyle::label("#0f172a", 12.0, "middle").with_font_weight("600"),
    );
    canvas.text_rotated(
        22.0,
        vp.top + 0.5 * vp.height,
        -90.0,
        "Power [dB rel. 1 (m/s^2)^2/Hz]",
        &SvgStyle::label("#0f172a", 12.0, "middle").with_font_weight("600"),
    );

    // Legend box:
    let lx = vp.left + 12.0;
    let ly = vp.top + 14.0;
    canvas.rect(
        lx - 6.0,
        ly - 8.0,
        162.0,
        54.0,
        &SvgStyle {
            fill: Some(String::from("#ffffff")),
            stroke: Some(String::from("#cbd5e1")),
            stroke_width: Some(0.8),
            opacity: Some(0.92),
            ..SvgStyle::default()
        },
    );
    canvas.line(
        lx,
        ly,
        lx + 18.0,
        ly,
        &SvgStyle::stroke_only("#334155", 2.1),
    );
    canvas.text(
        lx + 24.0,
        ly,
        "NLNM / NHNM (Peterson)",
        &SvgStyle::label("#1e293b", 10.0, "start").with_baseline("central"),
    );
    canvas.line(
        lx,
        ly + 16.0,
        lx + 18.0,
        ly + 16.0,
        &SvgStyle::stroke_only("#0f172a", 1.9),
    );
    canvas.text(
        lx + 24.0,
        ly + 16.0,
        "Mean PSD",
        &SvgStyle::label("#0f172a", 10.0, "start").with_baseline("central"),
    );
    canvas.line(
        lx,
        ly + 32.0,
        lx + 18.0,
        ly + 32.0,
        &SvgStyle::stroke_only("#dc2626", 1.7).with_dasharray("5,3"),
    );
    canvas.text(
        lx + 24.0,
        ly + 32.0,
        "Mode PSD",
        &SvgStyle::label("#dc2626", 10.0, "start").with_baseline("central"),
    );

    draw_vertical_colorbar(
        &mut canvas,
        vp.left + vp.width + 18.0,
        vp.top,
        14.0,
        vp.height,
        0.0,
        max_pct,
        "Prob (%)",
        ppsd_prob_hex,
    );

    canvas.finish()
}

// ============================================================================
// 5. Spherical Earth TauP Ray-Path Diagram (`plot_taup_paths_svg`)
// ============================================================================

/// Configuration options for [`plot_taup_paths_svg`].
#[derive(Debug, Clone)]
pub struct TaupPathsPlotOptions {
    /// SVG width in pixels (`860.0` default).
    pub width: f64,
    /// SVG height in pixels (`700.0` default).
    pub height: f64,
    /// Optional custom figure title.
    pub title: Option<String>,
}

impl Default for TaupPathsPlotOptions {
    fn default() -> Self {
        Self {
            width: 860.0,
            height: 700.0,
            title: None,
        }
    }
}

/// Renders a circular spherical Earth cross-section showing Surface, Moho, `410 km`, `660 km`,
/// CMB, and ICB concentric shells, shaded outer/inner core, and color-coded `TauP` ray trajectories.
pub fn plot_taup_paths_svg(
    paths: &[ArrivalPath],
    model_name: &str,
    opts: &TaupPathsPlotOptions,
) -> String {
    let w = opts.width.max(460.0);
    let h = opts.height.max(420.0);
    let mut canvas = SvgCanvas::new(w, h);
    canvas.set_background("#ffffff");

    let vmod = VelocityModel::by_name(model_name).unwrap_or_else(|_| VelocityModel::iasp91());
    let r_earth = vmod.radius_of_planet.max(1000.0);
    let r_moho = (r_earth - vmod.moho_depth).max(0.0);
    let r_410 = (r_earth - 410.0).max(0.0);
    let r_660 = (r_earth - 660.0).max(0.0);
    let r_cmb = (r_earth - vmod.cmb_depth).max(0.0);
    let r_icb = (r_earth - vmod.iocb_depth).max(0.0);

    let cx = (w - 210.0).max(w * 0.42);
    let cy = h * 0.52;
    let r_px = (cx.min(h * 0.48) - 46.0).max(120.0);
    let km_to_px = |radius_km: f64| (radius_km / r_earth) * r_px;

    let title = opts.title.clone().unwrap_or_else(|| {
        format!(
            "Spherical Earth Ray Paths — Model '{}' ({} arrival{})",
            model_name,
            paths.len(),
            if paths.len() == 1 { "" } else { "s" }
        )
    });
    canvas.text(
        w * 0.5,
        26.0,
        &title,
        &SvgStyle::label("#0f172a", 14.5, "middle").with_font_weight("700"),
    );

    // 1. Shaded concentric Earth interior shells:
    // Mantle base fill:
    canvas.circle(cx, cy, r_px, &SvgStyle::fill_only("#f8fafc"));
    // Upper-mantle transition zone 410-660 km subtle ring:
    canvas.circle(cx, cy, km_to_px(r_410), &SvgStyle::fill_only("#f1f5f9"));
    canvas.circle(cx, cy, km_to_px(r_660), &SvgStyle::fill_only("#f8fafc"));
    // Fluid outer core (CMB):
    canvas.circle(cx, cy, km_to_px(r_cmb), &SvgStyle::fill_only("#ffedd5"));
    // Solid inner core (ICB):
    canvas.circle(cx, cy, km_to_px(r_icb), &SvgStyle::fill_only("#fef08a"));

    // Shell boundary strokes & labels:
    canvas.circle(
        cx,
        cy,
        km_to_px(r_moho),
        &SvgStyle::stroke_only("#94a3b8", 0.75).with_dasharray("2,2"),
    );
    canvas.circle(
        cx,
        cy,
        km_to_px(r_410),
        &SvgStyle::stroke_only("#64748b", 0.85).with_dasharray("4,3"),
    );
    canvas.circle(
        cx,
        cy,
        km_to_px(r_660),
        &SvgStyle::stroke_only("#475569", 0.95).with_dasharray("5,3"),
    );
    canvas.circle(
        cx,
        cy,
        km_to_px(r_cmb),
        &SvgStyle::stroke_only("#c2410c", 1.35),
    );
    canvas.circle(
        cx,
        cy,
        km_to_px(r_icb),
        &SvgStyle::stroke_only("#a16207", 1.2),
    );
    canvas.circle(cx, cy, r_px, &SvgStyle::stroke_only("#0f172a", 1.8));

    // Shell depth labels along 270 deg (-X) radius:
    let shell_lbl = SvgStyle::label("#475569", 9.0, "end").with_baseline("auto");
    canvas.text(cx - 4.0, cy - km_to_px(r_410) + 11.0, "410 km", &shell_lbl);
    canvas.text(cx - 4.0, cy - km_to_px(r_660) + 11.0, "660 km", &shell_lbl);
    canvas.text(
        cx - 4.0,
        cy - km_to_px(r_cmb) + 12.0,
        &format!("CMB ({:.0} km)", vmod.cmb_depth),
        &SvgStyle::label("#9a3412", 9.5, "end"),
    );
    canvas.text(
        cx,
        cy + 4.0,
        &format!("ICB ({:.0} km)", vmod.iocb_depth),
        &SvgStyle::label("#854d0e", 9.5, "middle"),
    );

    // 2. Angular degree ticks around outer surface (0 deg at 12 o'clock):
    for deg in (0..360).step_by(30) {
        let ang = (deg as f64).to_radians();
        let (sa, ca) = math::sin_cos(ang);
        let x1 = cx + r_px * sa;
        let y1 = cy - r_px * ca;
        let x2 = cx + (r_px + 6.0) * sa;
        let y2 = cy - (r_px + 6.0) * ca;
        canvas.line(x1, y1, x2, y2, &SvgStyle::stroke_only("#334155", 1.0));
        let disp_deg = if deg <= 180 { deg } else { 360 - deg };
        let xl = cx + (r_px + 18.0) * sa;
        let yl = cy - (r_px + 18.0) * ca;
        canvas.text(
            xl,
            yl,
            &format!("{disp_deg}°"),
            &SvgStyle::label("#334155", 10.0, "middle").with_baseline("central"),
        );
    }

    // 3. Ray trajectories:
    for (idx, ap) in paths.iter().enumerate() {
        let col = PALETTE[idx % PALETTE.len()];
        let sign = if ap.arrival.purist_distance_deg % 360.0 > 180.0 {
            -1.0
        } else {
            1.0
        };
        let mut pts = Vec::with_capacity(ap.path_points.len());
        for pt in &ap.path_points {
            let rr = km_to_px((r_earth - pt.depth_km).max(0.0));
            let theta = sign * pt.dist_rad;
            let (st, ct) = math::sin_cos(theta);
            pts.push((cx + rr * st, cy - rr * ct));
        }
        canvas.polyline(&pts, &SvgStyle::stroke_only(col, 1.75).with_opacity(0.92));

        // Receiver triangle on last point:
        if let Some(&(rx, ry)) = pts.last() {
            let tri = [(rx, ry - 5.5), (rx - 4.8, ry + 4.0), (rx + 4.8, ry + 4.0)];
            canvas.polygon(&tri, &SvgStyle::fill_only("#0f172a"));
        }
    }

    // Source star at 0 deg, source_depth_km:
    let src_depth = paths
        .first()
        .map(|p| p.arrival.source_depth_km)
        .unwrap_or(0.0);
    let sy = cy - km_to_px((r_earth - src_depth).max(0.0));
    canvas.circle(
        cx,
        sy,
        5.2,
        &SvgStyle {
            fill: Some(String::from("#ef4444")),
            stroke: Some(String::from("#7f1d1d")),
            stroke_width: Some(1.2),
            ..SvgStyle::default()
        },
    );

    // 4. Right-hand Phase Legend:
    let lx = (cx + r_px + 38.0).min(w - 185.0);
    let mut ly = 64.0;
    canvas.text(
        lx,
        ly - 14.0,
        &format!("Moho: {:.1} km | h = {:.1} km", vmod.moho_depth, src_depth),
        &SvgStyle::label("#475569", 10.0, "start").with_font_weight("600"),
    );
    for (idx, ap) in paths.iter().take(18).enumerate() {
        let col = PALETTE[idx % PALETTE.len()];
        canvas.line(lx, ly, lx + 18.0, ly, &SvgStyle::stroke_only(col, 2.2));
        canvas.text(
            lx + 24.0,
            ly,
            &format!(
                "{}  ({:.1}s @ {:.1}°)",
                ap.arrival.name, ap.arrival.time_s, ap.arrival.distance_deg
            ),
            &SvgStyle::label("#1e293b", 10.5, "start").with_baseline("central"),
        );
        ly += 18.0;
    }

    canvas.finish()
}

// ============================================================================
// 6. Lower-Hemisphere Focal Mechanism Beachball (`plot_beachball_svg`)
// ============================================================================

/// Flexible focal-mechanism input accepted by [`plot_beachball_svg`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocalMechanismInput {
    /// Full 6-component Harvard/GCMT/QuakeML moment tensor `(m_rr, m_tt, m_pp, m_rt, m_rp, m_tp)`.
    MomentTensor(MomentTensor6),
    /// Double-couple fault plane `(strike_deg, dip_deg, rake_deg)`.
    StrikeDipRake(NodalPlane),
}

impl From<MomentTensor6> for FocalMechanismInput {
    fn from(mt: MomentTensor6) -> Self {
        Self::MomentTensor(mt)
    }
}

impl From<&MomentTensor6> for FocalMechanismInput {
    fn from(mt: &MomentTensor6) -> Self {
        Self::MomentTensor(*mt)
    }
}

impl From<NodalPlane> for FocalMechanismInput {
    fn from(np: NodalPlane) -> Self {
        Self::StrikeDipRake(np)
    }
}

impl From<(f64, f64, f64)> for FocalMechanismInput {
    fn from((strike_deg, dip_deg, rake_deg): (f64, f64, f64)) -> Self {
        Self::StrikeDipRake(NodalPlane {
            strike_deg,
            dip_deg,
            rake_deg,
        })
    }
}

impl From<&FocalMechanism> for FocalMechanismInput {
    fn from(fm: &FocalMechanism) -> Self {
        if let Some(mt) = fm.moment_tensor {
            Self::MomentTensor(mt)
        } else if let Some(np1) = fm.nodal_plane_1 {
            Self::StrikeDipRake(np1)
        } else {
            Self::StrikeDipRake(NodalPlane {
                strike_deg: 0.0,
                dip_deg: 45.0,
                rake_deg: 90.0,
            })
        }
    }
}

/// Configuration options for [`plot_beachball_svg`].
#[derive(Debug, Clone)]
pub struct BeachballPlotOptions {
    /// SVG width in pixels (`520.0` default).
    pub width: f64,
    /// SVG height in pixels (`560.0` default).
    pub height: f64,
    /// Fill color for compressional (`P > 0`, upward) quadrants (`"#dc2626"` crimson default).
    pub compressional_color: String,
    /// Fill color for dilatational (`P < 0`, downward) quadrants (`"#ffffff"` white default).
    pub dilatational_color: String,
    /// Render `P` and `T` principal axis markers (`true` default).
    pub show_axes: bool,
    /// Optional figure title.
    pub title: Option<String>,
}

impl Default for BeachballPlotOptions {
    fn default() -> Self {
        Self {
            width: 520.0,
            height: 560.0,
            compressional_color: String::from("#dc2626"),
            dilatational_color: String::from("#ffffff"),
            show_axes: true,
            title: None,
        }
    }
}

/// Renders a lower-hemisphere Schmidt equal-area focal mechanism beachball SVG showing
/// compressional/dilatational radiation quadrants, nodal-plane great-circle arcs, `P`/`T`
/// principal axis markers, and `ISO/DC/CLVD` decomposition metrics.
pub fn plot_beachball_svg(
    mt_or_planes: impl Into<FocalMechanismInput>,
    opts: &BeachballPlotOptions,
) -> String {
    let w = opts.width.max(300.0);
    let h = opts.height.max(320.0);
    let mut canvas = SvgCanvas::new(w, h);
    canvas.set_background("#ffffff");

    let mt = match mt_or_planes.into() {
        FocalMechanismInput::MomentTensor(m) => m,
        FocalMechanismInput::StrikeDipRake(np) => {
            let clamped = NodalPlane {
                strike_deg: np.strike_deg,
                dip_deg: np.dip_deg.clamp(0.0, 90.0),
                rake_deg: np.rake_deg.clamp(-180.0, 180.0),
            };
            strike_dip_rake_to_mt(clamped, 1.0e19)
                .unwrap_or_else(|_| MomentTensor6::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0e19))
        }
    };

    let decomp = decompose_moment_tensor(&mt);
    let cx = w * 0.5;
    let cy = h * 0.47;
    let r_px = (w.min(h - 70.0) * 0.40).max(90.0);

    let title = opts
        .title
        .clone()
        .unwrap_or_else(|| String::from("Focal Mechanism (Lower-Hemisphere Equal-Area)"));
    canvas.text(
        cx,
        26.0,
        &title,
        &SvgStyle::label("#0f172a", 14.0, "middle").with_font_weight("700"),
    );

    // Clip path for unit focal sphere disk:
    canvas.body_or_clip_circle(cx, cy, r_px);

    // Dilatational background disk:
    canvas.circle(cx, cy, r_px, &SvgStyle::fill_only(&opts.dilatational_color));

    // Compressional quadrants via horizontal scanline run-length merging inside circle clip:
    let grid_n = if cfg!(miri) { 33usize } else { 121usize };
    if let Ok(grid) = beachball_p_radiation_grid(&mt, grid_n, BeachballProjection::LambertEqualArea)
    {
        canvas.begin_group(None, Some("bb_disk_clip"), None);
        let step_px = (2.0 * r_px) / ((grid_n - 1) as f64);
        let fill_st = SvgStyle::fill_only(&opts.compressional_color);
        for iy in 0..grid_n {
            // iy = 0 -> South (y = -1 -> cy + r_px), iy = grid_n - 1 -> North (y = +1 -> cy - r_px)
            let y_norm = -1.0 + 2.0 * (iy as f64) / ((grid_n - 1) as f64);
            let py = cy - y_norm * r_px - 0.5 * step_px;
            let mut ix = 0usize;
            while ix < grid_n {
                let idx = iy * grid_n + ix;
                if grid.inside_mask[idx] && grid.amplitudes[idx] >= 0.0 {
                    let run_start = ix;
                    while ix < grid_n {
                        let k = iy * grid_n + ix;
                        if grid.inside_mask[k] && grid.amplitudes[k] >= 0.0 {
                            ix += 1;
                        } else {
                            break;
                        }
                    }
                    let x0_norm = -1.0 + 2.0 * (run_start as f64) / ((grid_n - 1) as f64);
                    let x1_norm = -1.0 + 2.0 * ((ix - 1) as f64) / ((grid_n - 1) as f64);
                    let px0 = cx + x0_norm * r_px - 0.5 * step_px;
                    let rw = (x1_norm - x0_norm) * r_px + step_px + 0.35;
                    canvas.rect(px0, py, rw, step_px + 0.35, &fill_st);
                } else {
                    ix += 1;
                }
            }
        }
        canvas.end_group();
    }

    // Best-fitting Double-Couple Nodal Plane Great-Circle Arcs:
    let (np1, np2) = decomp.nodal_planes;
    for np in [np1, np2] {
        let arc_pts = sample_nodal_plane_equal_area(np, cx, cy, r_px, 91);
        canvas.polyline(&arc_pts, &SvgStyle::stroke_only("#0f172a", 1.75));
    }

    // Cardinal tick marks (N, E, S, W) + outer focal rim:
    canvas.circle(cx, cy, r_px, &SvgStyle::stroke_only("#0f172a", 2.0));
    canvas.line(
        cx,
        cy - r_px,
        cx,
        cy - r_px - 7.0,
        &SvgStyle::stroke_only("#0f172a", 1.3),
    );
    canvas.text(
        cx,
        cy - r_px - 11.0,
        "N",
        &SvgStyle::label("#0f172a", 11.0, "middle").with_font_weight("700"),
    );

    // P & T principal axis markers:
    if opts.show_axes {
        draw_principal_axis_marker(
            &mut canvas,
            decomp.principal_axes.p_axis,
            cx,
            cy,
            r_px,
            "P",
            "#ffffff",
            "#0f172a",
        );
        draw_principal_axis_marker(
            &mut canvas,
            decomp.principal_axes.t_axis,
            cx,
            cy,
            r_px,
            "T",
            "#0f172a",
            "#ffffff",
        );
    }

    // Bottom mechanism caption:
    let sub1 = format!(
        "NP1: {s1:.0}° / {d1:.0}° / {r1:.0}°    NP2: {s2:.0}° / {d2:.0}° / {r2:.0}°",
        s1 = np1.strike_deg,
        d1 = np1.dip_deg,
        r1 = np1.rake_deg,
        s2 = np2.strike_deg,
        d2 = np2.dip_deg,
        r2 = np2.rake_deg,
    );
    let sub2 = format!(
        "DC: {:.1}%   |   CLVD: {:+.1}%   |   ISO: {:+.1}%",
        decomp.dc_percent, decomp.clvd_percent, decomp.iso_percent
    );
    canvas.text(
        cx,
        h - 32.0,
        &sub1,
        &SvgStyle::label("#1e293b", 11.0, "middle").with_font_weight("600"),
    );
    canvas.text(
        cx,
        h - 14.0,
        &sub2,
        &SvgStyle::label("#475569", 10.5, "middle"),
    );

    canvas.finish()
}

trait CanvasCircleClipExt {
    fn body_or_clip_circle(&mut self, cx: f64, cy: f64, r: f64);
}

impl CanvasCircleClipExt for SvgCanvas {
    fn body_or_clip_circle(&mut self, cx: f64, cy: f64, r: f64) {
        // Note: `SvgCanvas::define_clip_rect` is built-in; for circular focal-sphere clipping we
        // also draw the outer crisp rim over scanlines and bound scanline centers strictly inside
        // `x^2 + y^2 <= 1`, plus register a bounding rect clip `bb_disk_clip`:
        self.define_clip_rect("bb_disk_clip", cx - r, cy - r, 2.0 * r, 2.0 * r);
    }
}

fn sample_nodal_plane_equal_area(
    plane: NodalPlane,
    cx: f64,
    cy: f64,
    r_px: f64,
    n_pts: usize,
) -> Vec<(f64, f64)> {
    let phi = plane.strike_deg.to_radians();
    let delta = plane.dip_deg.clamp(0.0, 90.0).to_radians();
    let (sp, cp) = math::sin_cos(phi);
    let (sd, cd) = math::sin_cos(delta);

    // Along-strike horizontal unit vector s = (N, E, D) and down-dip unit vector d_down (D >= 0):
    let s_vec = [cp, sp, 0.0];
    let d_down = [-cd * sp, cd * cp, sd];

    let count = n_pts.max(3);
    let mut pts = Vec::with_capacity(count);
    for k in 0..count {
        let alpha = PI * (k as f64) / ((count - 1) as f64);
        let (sa, ca) = math::sin_cos(alpha);
        let un = ca * s_vec[0] + sa * d_down[0];
        let ue = ca * s_vec[1] + sa * d_down[1];
        let ud = (ca * s_vec[2] + sa * d_down[2]).clamp(0.0, 1.0);
        // Exact singularity-free Lambert equal-area projection: (x, y) = (u_E, u_N) / sqrt(1 + u_D)
        let denom = math::sqrt(1.0 + ud);
        let px = cx + r_px * (ue / denom);
        let py = cy - r_px * (un / denom);
        pts.push((px, py));
    }
    pts
}

#[allow(clippy::too_many_arguments)]
fn draw_principal_axis_marker(
    canvas: &mut SvgCanvas,
    axis: PrincipalAxis,
    cx: f64,
    cy: f64,
    r_px: f64,
    label: &str,
    bg_col: &str,
    fg_col: &str,
) {
    let az = axis.azimuth_deg.to_radians();
    let pl = axis.plunge_deg.clamp(0.0, 90.0).to_radians();
    let (saz, caz) = math::sin_cos(az);
    let (spl, cpl) = math::sin_cos(pl);
    let un = cpl * caz;
    let ue = cpl * saz;
    let ud = spl.clamp(0.0, 1.0);
    let denom = math::sqrt(1.0 + ud);
    let px = cx + r_px * (ue / denom);
    let py = cy - r_px * (un / denom);

    canvas.circle(
        px,
        py,
        9.0,
        &SvgStyle {
            fill: Some(String::from(bg_col)),
            stroke: Some(String::from(fg_col)),
            stroke_width: Some(1.4),
            ..SvgStyle::default()
        },
    );
    canvas.text(
        px,
        py,
        label,
        &SvgStyle::label(fg_col, 10.5, "middle")
            .with_font_weight("700")
            .with_baseline("central"),
    );
}

// ============================================================================
// Shared Colorbars, Scientific Colormaps & Helpers
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn draw_vertical_colorbar(
    canvas: &mut SvgCanvas,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    v_min: f64,
    v_max: f64,
    unit_title: &str,
    cmap: fn(f64) -> String,
) {
    let steps = 40usize;
    let sh = height / (steps as f64);
    for i in 0..steps {
        let norm = ((i as f64) + 0.5) / (steps as f64);
        let sy = y + height - ((i + 1) as f64) * sh;
        canvas.rect(x, sy, width, sh + 0.4, &SvgStyle::fill_only(&cmap(norm)));
    }
    canvas.rect(x, y, width, height, &SvgStyle::stroke_only("#0f172a", 1.0));
    let lbl_st = SvgStyle::label("#334155", 9.5, "start").with_baseline("central");
    for &frac in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let py = y + height * (1.0 - frac);
        let val = v_min + frac * (v_max - v_min);
        canvas.line(
            x + width,
            py,
            x + width + 4.0,
            py,
            &SvgStyle::stroke_only("#334155", 0.9),
        );
        canvas.text(x + width + 6.0, py, &format!("{val:.0}"), &lbl_st);
    }
    canvas.text(
        x + 0.5 * width,
        y - 8.0,
        unit_title,
        &SvgStyle::label("#0f172a", 10.0, "middle").with_font_weight("600"),
    );
}

fn viridis_hex(t: f64) -> String {
    const STOPS: [(u8, u8, u8); 6] = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (189, 223, 38),
        (253, 231, 37),
    ];
    interp_rgb_hex(t, &STOPS)
}

fn ppsd_prob_hex(t: f64) -> String {
    const STOPS: [(u8, u8, u8); 6] = [
        (241, 245, 249),
        (99, 102, 241),
        (6, 182, 212),
        (34, 197, 94),
        (250, 204, 21),
        (220, 38, 38),
    ];
    interp_rgb_hex(t, &STOPS)
}

fn interp_rgb_hex(t: f64, stops: &[(u8, u8, u8)]) -> String {
    let u = t.clamp(0.0, 1.0) * ((stops.len() - 1) as f64);
    let i0 = (math::floor(u) as usize).min(stops.len() - 1);
    let i1 = (i0 + 1).min(stops.len() - 1);
    let f = u - (i0 as f64);
    let lerp = |a: u8, b: u8| -> u8 {
        math::round((a as f64) + f * ((b as f64) - (a as f64))).clamp(0.0, 255.0) as u8
    };
    let r = lerp(stops[i0].0, stops[i1].0);
    let g = lerp(stops[i0].1, stops[i1].1);
    let b = lerp(stops[i0].2, stops[i1].2);
    format!("#{r:02x}{g:02x}{b:02x}")
}

fn clean_num(v: f64) -> String {
    if !v.is_finite() || v == 0.0 {
        return String::from("0");
    }
    let av = v.abs();
    if !(1e-3..1e5).contains(&av) {
        format!("{v:.2e}")
    } else {
        let s = format!("{v:.4}");
        let t = s.trim_end_matches('0').trim_end_matches('.');
        String::from(if t == "-0" { "0" } else { t })
    }
}

fn empty_svg_placeholder(width: f64, height: f64, message: &str) -> String {
    let mut canvas = SvgCanvas::new(width, height);
    canvas.set_background("#ffffff");
    let mut pb = SvgPathBuilder::new();
    pb.move_to(24.0, 24.0)
        .line_to(width - 24.0, 24.0)
        .line_to(width - 24.0, height - 24.0)
        .line_to(24.0, height - 24.0)
        .close();
    canvas.path(
        pb.as_str(),
        &SvgStyle::stroke_only("#cbd5e1", 1.2).with_dasharray("6,4"),
    );
    canvas.text(
        width * 0.5,
        height * 0.5,
        message,
        &SvgStyle::label("#64748b", 13.0, "middle").with_baseline("central"),
    );
    let _ = is_sac_float_undef(-12345.0);
    let _ = TAU;
    canvas.finish()
}

// Replace the stubbed phase flag call inside `plot_waveforms_svg` by wiring `draw_sac_phase_flags_mut`:
#[cfg(test)]
mod tests {
    use super::*;
    use earthsnail_core::{ChannelId, ChannelResponse, Complex64, TraceStats};
    use earthsnail_dsp::spectral::PpsdConfig;
    use earthsnail_taup::TauModel;

    fn assert_valid_svg(svg: &str) {
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.len() > 250);
    }

    #[test]
    fn all_six_svg_plotters_render_valid_documents_under_miri() {
        let t0: UtcTimestamp = "2024-03-11T05:46:24Z".parse().unwrap();
        let npts = if cfg!(miri) { 64 } else { 400 };
        let fs = 20.0;
        let data: Vec<f64> = (0..npts)
            .map(|i| {
                let t = (i as f64) / fs;
                math::sin(TAU * 1.2 * t) * math::exp(-0.15 * t)
            })
            .collect();
        let mut stats = TraceStats::new(ChannelId::new("G", "CAN", "00", "BHZ"), t0, fs);
        stats.sync_to_sac_header(&data);
        if let Some(ref mut sac) = stats.sac {
            sac.set_float("o", 0.0).unwrap();
            sac.set_float("a", 0.8).unwrap();
            sac.set_str("ka", "MORSE").unwrap();
            sac.set_float("t0", 1.2).unwrap();
            sac.set_str("kt0", "P").unwrap();
            sac.set_float("gcarc", 42.0).unwrap();
        }
        let tr = Trace::new(stats, data);
        let stream = Stream::from_traces(vec![tr.clone()]);

        // 1. Waveforms (Stacked & Overlaid):
        let svg_p1 = plot_waveforms_svg(&stream, &WaveformPlotOptions::default());
        assert_valid_svg(&svg_p1);
        assert!(svg_p1.contains("G.CAN.00.BHZ"));
        let svg_p2 = plot_waveforms_svg(
            &stream,
            &WaveformPlotOptions {
                layout: WaveformLayout::Overlaid,
                time_axis: TimeAxisMode::UtcClock,
                ..WaveformPlotOptions::default()
            },
        );
        assert_valid_svg(&svg_p2);

        // 2. Record section:
        let svg_rs = plot_record_section_svg(&stream, &[], &RecordSectionOptions::default());
        assert_valid_svg(&svg_rs);

        // 3. Spectrogram (CWT & STFT):
        let svg_spec = plot_spectrogram_svg(&tr, &SpectrogramOptions::default());
        assert_valid_svg(&svg_spec);

        // 4. PPSD:
        let resp = ChannelResponse::from_sacpz(
            vec![Complex64::ZERO, Complex64::ZERO, Complex64::ZERO],
            vec![
                Complex64::new(-0.037, 0.037),
                Complex64::new(-0.037, -0.037),
            ],
            1.5e9,
        );
        let mut ppsd = Ppsd::new(
            fs,
            &resp,
            PpsdConfig {
                ppsd_length_s: (npts as f64) / fs,
                overlap: 0.5,
                ..PpsdConfig::default()
            },
        )
        .unwrap();
        ppsd.add_trace(&tr).unwrap();
        let svg_ppsd = plot_ppsd_svg(&ppsd, &PpsdPlotOptions::default());
        assert_valid_svg(&svg_ppsd);
        assert!(svg_ppsd.contains("NLNM"));

        // 5. TauP ray paths (skip full TauModel compilation under Miri for fast test turnaround, test empty/1-path):
        let paths = if cfg!(miri) {
            Vec::new()
        } else {
            let tm = TauModel::iasp91().unwrap();
            tm.get_ray_paths(29.0, 74.0, &["P", "S", "PcP", "SKS"])
                .unwrap()
        };
        let svg_taup = plot_taup_paths_svg(&paths, "iasp91", &TaupPathsPlotOptions::default());
        assert_valid_svg(&svg_taup);
        assert!(svg_taup.contains("CMB"));
        assert!(svg_taup.contains("410 km"));
        assert!(svg_taup.contains("660 km"));

        // 6. Beachball:
        let svg_bb = plot_beachball_svg((203.0, 10.0, 88.0), &BeachballPlotOptions::default());
        assert_valid_svg(&svg_bb);
        assert!(svg_bb.contains("DC:"));
    }
}
