//! Interactive `SAC>` shell and `.m` macro script engine (`SacSession`) supporting unlimited
//! traces in memory, multi-step `undo`, blackboard variables (`setbb`/`getbb`/`unsetbb` with
//! `$var` & `%var%` expansion), positional macro arguments (`$1..$9`), `do ... enddo` loops,
//! classic SAC commands & standard abbreviations, built-in `taup` header stamping, and
//! zero-dependency publication SVG plotting.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use core::f64::consts::TAU;
use earthsnail_core::sac_header::{SAC_FLOAT_NAMES, SAC_INT_NAMES, SAC_STR_NAMES};
use earthsnail_core::{
    is_sac_int_undef, is_sac_str_undef, kilometer2degrees, math, ChannelId, ChannelResponse,
    Complex64, EarthSnailError, KinematicUnit, MomentTensor6, NodalPlane, Result, SacDepType,
    SacEndian, SacFileType, SacHeader, Stream, Trace, TraceStats, UtcTimestamp,
};
use earthsnail_dsp::{
    aic_pick, analytic_signal, classic_sta_lta, correlate, decimate_trace, design_iir_sos,
    differentiate_gradient, differentiate_spectral, envelope, integrate_spectral,
    integrate_trapezoidal, irfft, lanczos_resample, merge_stream, morse_pick, pk_baer, rfft,
    rmean_in_place, rotate_ne_rt, rotate_to_zne, rotate_uvw_zne, rtrend_in_place, sac_transfer,
    sosfilt, sosfilt_zerophase_obspy, sosfiltfilt, taper_in_place, trigger_onset,
    wood_anderson_response_iaspei, BaerPickParams, BesselNorm, FilterBand, FilterFamily,
    LanczosWindow, MergeStrategy, MorsePickConfig, TaperSide, TaperType, XcorrNormalize,
};
use earthsnail_io::{
    read_sacpz_str, read_stationxml_bytes, read_waveforms_bytes, write_geocsv_stream,
    write_mseed3_stream_bytes, write_mseed_stream_bytes, write_sac_alpha_bytes, write_sac_bytes,
    write_slist_stream, write_tspair_stream, Mseed3WriteOptions, MseedWriteOptions,
};
use earthsnail_taup::{TauEngine, TauModel, VelocityModel};

use crate::plot::{
    plot_beachball_svg, plot_record_section_svg, plot_spectrogram_svg, plot_taup_paths_svg,
    plot_waveforms_svg, BeachballPlotOptions, DistanceUnit, FocalMechanismInput, FrequencyScale,
    RecordSectionOptions, SpectrogramMethod, SpectrogramOptions, TaupPathsPlotOptions,
    WaveformLayout, WaveformPlotOptions,
};

const MAX_MACRO_DEPTH: usize = 16;
const MAX_LOOP_ITERATIONS: usize = 50_000;
const MAX_FUNCGEN_NPTS: usize = 50_000_000;

/// Saved complex spectrum enabling lossless `fft` -> `ifft` round-trip in `SAC>`.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedSpectrum {
    /// Half-complex Hermitian RFFT bins `0..=nfft/2`.
    pub rfft_bins: Vec<Complex64>,
    /// FFT transform length.
    pub nfft: usize,
    /// Original time-domain `NPTS` before `fft`.
    pub orig_npts: usize,
    /// Original time-domain sampling rate in Hz.
    pub orig_sampling_rate: f64,
    /// Original time-domain `B` offset in seconds.
    pub orig_b: f64,
}

/// Cut window anchor in SAC header coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CutAnchor {
    /// Absolute offset relative to SAC reference zero-time (`KZDATE`/`KZTIME`).
    Zero,
    /// Trace begin marker `B`.
    Begin,
    /// Trace end marker `E`.
    End,
    /// Origin marker `O`.
    Origin,
    /// First arrival pick marker `A`.
    Arrival,
    /// Finale / coda marker `F`.
    Finale,
    /// Theoretical or analyst time pick `T0..T9`.
    Tn(u8),
}

/// Active `SAC cut` specification (`[anchor1 + offset1, anchor2 + offset2]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutSpec {
    pub start_anchor: CutAnchor,
    pub start_offset_s: f64,
    pub end_anchor: CutAnchor,
    pub end_offset_s: f64,
}

#[derive(Debug, Clone)]
struct SacSnapshot {
    traces: Vec<Trace>,
    filenames: Vec<String>,
    spectra: Vec<Option<SavedSpectrum>>,
    cut_enabled: bool,
    cut_spec: Option<CutSpec>,
}

/// Stateful `SAC>` interactive session and `.m` macro execution engine with unlimited trace slots.
#[derive(Debug, Clone)]
pub struct SacSession {
    /// Active waveforms in memory (unbounded count — no 200-file SAC limit!).
    pub traces: Vec<Trace>,
    /// Source filenames corresponding 1-to-1 with `traces` (used by `write over` and `wh`).
    pub filenames: Vec<String>,
    /// Per-trace saved RFFT spectra when `fft` has transformed a trace.
    pub spectra: Vec<Option<SavedSpectrum>>,
    /// Case-insensitive blackboard variable store (`setbb` / `getbb` / `unsetbb`).
    pub blackboard: BTreeMap<String, String>,
    /// Active `cut` mode flag (`cut on` / `cut off`).
    pub cut_enabled: bool,
    /// Configured `cut` window bounds.
    pub cut_spec: Option<CutSpec>,
    /// Base working directory for relative file and macro resolution.
    pub working_dir: PathBuf,
    /// Accumulated text output from `lh`, `getbb`, `taup`, `apk`, `echo`, etc.
    pub output_log: String,
    /// Most recently rendered SVG document (populated by `p1`, `p2`, `prs`, `spectrogram`, `beachball`).
    pub last_svg: Option<String>,
    /// Whether `quit` / `exit` was invoked.
    pub exited: bool,
    /// Maximum number of `undo` snapshots retained (`32` default).
    pub max_undo_depth: usize,
    undo_stack: Vec<SacSnapshot>,
    /// Optional in-memory virtual files (enabling full I/O + macro testing under `cargo miri`!).
    pub virtual_files: BTreeMap<String, Vec<u8>>,
    /// Optional in-memory macro scripts keyed by macro name.
    pub virtual_macros: BTreeMap<String, String>,
    tau_engines: BTreeMap<String, TauEngine>,
}

impl Default for SacSession {
    fn default() -> Self {
        Self::new()
    }
}

impl SacSession {
    /// Creates a new `SacSession`. Avoids touching OS `current_dir()` under Miri strict isolation.
    pub fn new() -> Self {
        #[cfg(not(miri))]
        let wd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        #[cfg(miri)]
        let wd = PathBuf::from(".");

        Self {
            traces: Vec::new(),
            filenames: Vec::new(),
            spectra: Vec::new(),
            blackboard: BTreeMap::new(),
            cut_enabled: false,
            cut_spec: None,
            working_dir: wd,
            output_log: String::new(),
            last_svg: None,
            exited: false,
            max_undo_depth: 32,
            undo_stack: Vec::new(),
            virtual_files: BTreeMap::new(),
            virtual_macros: BTreeMap::new(),
            tau_engines: BTreeMap::new(),
        }
    }

    /// Creates a `SacSession` anchored to `working_dir`.
    pub fn with_working_dir(working_dir: impl Into<PathBuf>) -> Self {
        let mut s = Self::new();
        s.working_dir = working_dir.into();
        s
    }

    /// Borrows all active traces as a `Stream` slice view.
    pub fn to_stream(&self) -> Stream {
        Stream::from_traces(self.traces.clone())
    }

    /// Clears and returns accumulated `output_log` text.
    pub fn take_output(&mut self) -> String {
        core::mem::take(&mut self.output_log)
    }

    /// Registers an in-memory file (`SAC`, `miniSEED`, `StationXML`, `SACPZ`, etc.) for virtual I/O.
    pub fn put_virtual_file(&mut self, name: impl Into<String>, bytes: Vec<u8>) {
        self.virtual_files.insert(name.into(), bytes);
    }

    /// Retrieves bytes written to an in-memory virtual file (when `virtual_files` mode or path matches).
    pub fn get_virtual_file(&self, name: &str) -> Option<&[u8]> {
        self.virtual_files.get(name).map(Vec::as_slice)
    }

    /// Registers an in-memory `.m` macro script body.
    pub fn put_virtual_macro(&mut self, name: impl Into<String>, script: impl Into<String>) {
        self.virtual_macros.insert(name.into(), script.into());
    }

    /// Launches an interactive `SAC>` REPL reading from `reader` and writing output/diagnostics to `writer`.
    pub fn run_repl<R: BufRead, W: Write>(&mut self, mut reader: R, mut writer: W) -> Result<()> {
        let mut line_buf = String::new();
        let mut multiline_block = String::new();
        let mut do_depth = 0usize;

        while !self.exited {
            let prompt = if do_depth > 0 { "SAC(do)> " } else { "SAC> " };
            let _ = write!(writer, "{prompt}");
            let _ = writer.flush();

            line_buf.clear();
            let n = reader
                .read_line(&mut line_buf)
                .map_err(|e| EarthSnailError::Io(format!("REPL stdin read error: {e}")))?;
            if n == 0 {
                break;
            }

            // Track nested `do ... enddo` blocks interactively:
            for stmt in split_semicolon_statements(line_buf.trim_start_matches('\u{feff}')) {
                let first = stmt
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if first == "do" {
                    do_depth += 1;
                } else if first == "enddo" {
                    do_depth = do_depth.saturating_sub(1);
                }
            }

            if do_depth > 0 || !multiline_block.is_empty() {
                multiline_block.push_str(&line_buf);
                if do_depth > 0 {
                    continue;
                }
                let script = core::mem::take(&mut multiline_block);
                if let Err(err) = self.execute_script(&script, &[]) {
                    let _ = writeln!(writer, "SAC Error: {err}");
                }
            } else if let Err(err) = self.execute_line(&line_buf) {
                let _ = writeln!(writer, "SAC Error: {err}");
            }

            let out = self.take_output();
            if !out.is_empty() {
                let _ = write!(writer, "{out}");
            }
        }
        Ok(())
    }

    /// Launches an interactive `SAC>` REPL on process `stdin` / `stdout`.
    pub fn run_stdio_repl(&mut self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        self.run_repl(stdin.lock(), stdout.lock())
    }

    /// Executes a single line (which may contain multiple `;`-separated SAC commands).
    pub fn execute_line(&mut self, line: &str) -> Result<()> {
        self.execute_script_inner(line, &[], 0)
    }

    /// Executes a multi-line SAC `.m` macro script with optional positional `$1..$9` arguments.
    pub fn execute_script(&mut self, script: &str, args: &[&str]) -> Result<()> {
        let owned_args: Vec<String> = args.iter().map(|s| String::from(*s)).collect();
        self.execute_script_inner(script, &owned_args, 0)
    }

    /// Executes a `.m` macro file from disk (or `virtual_macros`) with optional positional arguments.
    pub fn execute_macro_file(&mut self, macro_path: &str, args: &[String]) -> Result<()> {
        self.run_macro_with_depth(macro_path, args, 0)
    }

    fn run_macro_with_depth(
        &mut self,
        macro_path: &str,
        args: &[String],
        depth: usize,
    ) -> Result<()> {
        if depth >= MAX_MACRO_DEPTH {
            return Err(EarthSnailError::CapacityExceeded(format!(
                "SAC macro recursion depth exceeded maximum limit ({MAX_MACRO_DEPTH}) at '{macro_path}'"
            )));
        }
        let body = if let Some(virt) = self.virtual_macros.get(macro_path) {
            virt.clone()
        } else if let Some(virt_bytes) = self.virtual_files.get(macro_path) {
            String::from_utf8(virt_bytes.clone()).map_err(|_| {
                EarthSnailError::Format(format!("macro '{macro_path}' is not valid UTF-8"))
            })?
        } else {
            let resolved = self.resolve_path(macro_path);
            fs::read_to_string(&resolved).map_err(|e| {
                EarthSnailError::Io(format!(
                    "cannot read SAC macro '{}': {e}",
                    resolved.display()
                ))
            })?
        };
        self.execute_script_inner(&body, args, depth + 1)
    }

    fn execute_script_inner(
        &mut self,
        script: &str,
        macro_args: &[String],
        depth: usize,
    ) -> Result<()> {
        let script = script.strip_prefix('\u{feff}').unwrap_or(script);
        // Flatten lines & `;`-separated statements after stripping `*` and `#` comments:
        let mut stmts: Vec<String> = Vec::new();
        for raw_line in script.lines() {
            let trimmed_start = raw_line.trim_start();
            if trimmed_start.is_empty()
                || trimmed_start.starts_with('*')
                || trimmed_start.starts_with('#')
            {
                continue;
            }
            let cleaned = strip_inline_hash_comment(trimmed_start);
            for part in split_semicolon_statements(&cleaned) {
                let st = part.trim();
                if st.is_empty() || st.starts_with('*') || st.starts_with('#') {
                    continue;
                }
                stmts.push(String::from(st));
            }
        }
        self.execute_stmt_slice(&stmts, macro_args, depth)
    }

    fn execute_stmt_slice(
        &mut self,
        stmts: &[String],
        macro_args: &[String],
        depth: usize,
    ) -> Result<()> {
        let mut idx = 0usize;
        while idx < stmts.len() && !self.exited {
            let raw_stmt = &stmts[idx];
            let first_word = raw_stmt
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();

            if first_word == "do" {
                let end_idx = find_matching_enddo(stmts, idx)?;
                let header_expanded = self.expand_variables(raw_stmt, macro_args)?;
                let body_slice = &stmts[(idx + 1)..end_idx];
                let (var_name, values) = parse_do_header(&header_expanded)?;
                if values.len() > MAX_LOOP_ITERATIONS {
                    return Err(EarthSnailError::CapacityExceeded(format!(
                        "SAC 'do' loop count ({}) exceeds safety limit ({MAX_LOOP_ITERATIONS})",
                        values.len()
                    )));
                }
                let prev_val = self.blackboard.get(&var_name.to_ascii_uppercase()).cloned();
                for val in values {
                    self.blackboard.insert(var_name.to_ascii_uppercase(), val);
                    self.execute_stmt_slice(body_slice, macro_args, depth)?;
                    if self.exited {
                        break;
                    }
                }
                match prev_val {
                    Some(old) => {
                        self.blackboard.insert(var_name.to_ascii_uppercase(), old);
                    }
                    None => {
                        self.blackboard.remove(&var_name.to_ascii_uppercase());
                    }
                }
                idx = end_idx + 1;
                continue;
            }

            if first_word == "enddo" {
                return Err(EarthSnailError::Format(
                    "SAC syntax error: unmatched 'enddo' without corresponding 'do'".into(),
                ));
            }

            let expanded = self.expand_variables(raw_stmt, macro_args)?;
            let tokens = tokenize_sac_command(&expanded)?;
            if !tokens.is_empty() {
                self.dispatch_command(&tokens, depth)?;
            }
            idx += 1;
        }
        Ok(())
    }

    fn expand_variables(&self, input: &str, macro_args: &[String]) -> Result<String> {
        let mut out = String::with_capacity(input.len() + 16);
        let chars: Vec<char> = input.chars().collect();
        let n = chars.len();
        let mut i = 0usize;

        while i < n {
            let ch = chars[i];
            if ch == '$' && i + 1 < n {
                // Positional $1..$9:
                if let Some(d) = chars[i + 1].to_digit(10) {
                    if (1..=9).contains(&d) {
                        let arg_idx = (d as usize) - 1;
                        let val = macro_args.get(arg_idx).ok_or_else(|| {
                            EarthSnailError::InvalidParameter(format!(
                                "SAC macro argument '${d}' not provided (got {} args)",
                                macro_args.len()
                            ))
                        })?;
                        out.push_str(val);
                        i += 2;
                        continue;
                    }
                }
                // Braced ${VAR}:
                if chars[i + 1] == '{' {
                    if let Some(rel_end) = chars[(i + 2)..].iter().position(|&c| c == '}') {
                        let name: String = chars[(i + 2)..(i + 2 + rel_end)].iter().collect();
                        let val = self.lookup_bb(&name)?;
                        out.push_str(val);
                        i = i + 2 + rel_end + 1;
                        continue;
                    }
                }
                // Identifier $VAR or $VAR$:
                if chars[i + 1].is_ascii_alphabetic() || chars[i + 1] == '_' {
                    let mut j = i + 1;
                    while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                        j += 1;
                    }
                    let name: String = chars[(i + 1)..j].iter().collect();
                    let val = self.lookup_bb(&name)?;
                    out.push_str(val);
                    if j < n && chars[j] == '$' {
                        j += 1;
                    }
                    i = j;
                    continue;
                }
            }

            // Classic SAC %VAR% blackboard syntax:
            if ch == '%' && i + 2 < n && (chars[i + 1].is_ascii_alphabetic() || chars[i + 1] == '_')
            {
                let mut j = i + 1;
                while j < n && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                    j += 1;
                }
                if j < n && chars[j] == '%' {
                    let name: String = chars[(i + 1)..j].iter().collect();
                    let val = self.lookup_bb(&name)?;
                    out.push_str(val);
                    i = j + 1;
                    continue;
                }
            }

            out.push(ch);
            i += 1;
        }
        Ok(out)
    }

    fn lookup_bb<'a>(&'a self, name: &str) -> Result<&'a str> {
        let key = name.trim().to_ascii_uppercase();
        self.blackboard
            .get(&key)
            .map(String::as_str)
            .ok_or_else(|| {
                EarthSnailError::InvalidParameter(format!(
                    "undefined SAC blackboard variable '{name}' (set it first with 'setbb {name} <value>')"
                ))
            })
    }

    fn push_undo(&mut self) {
        if self.max_undo_depth == 0 {
            return;
        }
        if self.undo_stack.len() >= self.max_undo_depth {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(SacSnapshot {
            traces: self.traces.clone(),
            filenames: self.filenames.clone(),
            spectra: self.spectra.clone(),
            cut_enabled: self.cut_enabled,
            cut_spec: self.cut_spec,
        });
    }

    fn rollback_last_undo(&mut self) {
        if let Some(snap) = self.undo_stack.pop() {
            self.traces = snap.traces;
            self.filenames = snap.filenames;
            self.spectra = snap.spectra;
            self.cut_enabled = snap.cut_enabled;
            self.cut_spec = snap.cut_spec;
        }
    }

    fn run_mutating<F>(&mut self, op: F) -> Result<()>
    where
        F: FnOnce(&mut Self) -> Result<()>,
    {
        self.push_undo();
        match op(self) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.rollback_last_undo();
                Err(e)
            }
        }
    }

    fn require_traces(&self, cmd_name: &str) -> Result<()> {
        if self.traces.is_empty() {
            return Err(EarthSnailError::EmptyData(format!(
                "SAC '{cmd_name}': no waveforms in memory (load files with 'read <file>' or synthesize with 'funcgen')"
            )));
        }
        Ok(())
    }

    fn resolve_path(&self, raw: &str) -> PathBuf {
        let p = Path::new(raw);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.working_dir.join(p)
        }
    }

    fn read_bytes_any(&self, raw_path: &str) -> Result<Vec<u8>> {
        if let Some(v) = self.virtual_files.get(raw_path) {
            return Ok(v.clone());
        }
        let full = self.resolve_path(raw_path);
        if let Some(s) = full.to_str() {
            if let Some(v) = self.virtual_files.get(s) {
                return Ok(v.clone());
            }
        }
        #[cfg(miri)]
        {
            let _ = full;
            Err(EarthSnailError::Io(format!(
                "virtual file '{raw_path}' not found under Miri isolation"
            )))
        }
        #[cfg(not(miri))]
        {
            fs::read(&full).map_err(|e| {
                EarthSnailError::Io(format!("failed to read '{}': {e}", full.display()))
            })
        }
    }

    fn write_bytes_any(&mut self, raw_path: &str, bytes: &[u8]) -> Result<()> {
        #[cfg(miri)]
        {
            self.virtual_files
                .insert(String::from(raw_path), bytes.to_vec());
            Ok(())
        }
        #[cfg(not(miri))]
        {
            if let Some(rest) = raw_path.strip_prefix("mem://") {
                self.virtual_files
                    .insert(String::from(rest), bytes.to_vec());
                self.virtual_files
                    .insert(String::from(raw_path), bytes.to_vec());
                return Ok(());
            }
            let full = self.resolve_path(raw_path);
            if let Some(parent) = full.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    fs::create_dir_all(parent).map_err(|e| {
                        EarthSnailError::Io(format!(
                            "cannot create directory '{}': {e}",
                            parent.display()
                        ))
                    })?;
                }
            }
            fs::write(&full, bytes).map_err(|e| {
                EarthSnailError::Io(format!("failed to write '{}': {e}", full.display()))
            })
        }
    }

    fn get_or_compile_tau_model(&mut self, model_spec: &str) -> Result<&mut TauEngine> {
        let key = model_spec.trim().to_string();
        if !self.tau_engines.contains_key(&key) {
            let tm = match TauModel::by_name(&key) {
                Ok(m) => m,
                Err(_) => {
                    let bytes = self.read_bytes_any(&key)?;
                    let txt = String::from_utf8(bytes).map_err(|_| {
                        EarthSnailError::ModelError(format!(
                            "velocity model file '{key}' is not UTF-8"
                        ))
                    })?;
                    let vmod = if key.to_ascii_lowercase().ends_with(".tvel") {
                        VelocityModel::from_tvel_str(&key, &txt)?
                    } else {
                        VelocityModel::from_nd_str(&key, &txt)?
                    };
                    TauModel::from_velocity_model(vmod)?
                }
            };
            let eng = TauEngine::new(tm)?;
            self.tau_engines.insert(key.clone(), eng);
        }
        Ok(self.tau_engines.get_mut(&key).unwrap())
    }

    // ========================================================================
    // Command Dispatcher
    // ========================================================================

    fn dispatch_command(&mut self, tokens: &[String], depth: usize) -> Result<()> {
        let cmd = tokens[0].to_ascii_lowercase();
        let args = &tokens[1..];

        match cmd.as_str() {
            "q" | "quit" | "exit" => {
                self.exited = true;
                Ok(())
            }
            "help" | "h" => {
                self.cmd_help();
                Ok(())
            }
            "echo" | "message" | "ms" => {
                self.output_log.push_str(&args.join(" "));
                self.output_log.push('\n');
                Ok(())
            }
            "pwd" => {
                self.output_log
                    .push_str(&format!("{}\n", self.working_dir.display()));
                Ok(())
            }
            "cd" => {
                let target = args.first().ok_or_else(|| {
                    EarthSnailError::InvalidParameter("SAC 'cd' requires a directory path".into())
                })?;
                self.working_dir = self.resolve_path(target);
                Ok(())
            }
            "undo" => self.cmd_undo(args),
            "setbb" => self.cmd_setbb(args),
            "getbb" => self.cmd_getbb(args),
            "unsetbb" => self.cmd_unsetbb(args),
            "m" | "macro" => {
                let mpath = args.first().ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "SAC 'macro' requires a .m script path".into(),
                    )
                })?;
                self.run_macro_with_depth(mpath, &args[1..], depth)
            }

            // --- I/O & Headers ---
            "r" | "read" => self.run_mutating(|s| s.cmd_read(args)),
            "rmore" => {
                let mut v = vec![String::from("more")];
                v.extend_from_slice(args);
                self.run_mutating(|s| s.cmd_read(&v))
            }
            "w" | "write" => self.cmd_write(args),
            "lh" | "listhdr" => self.cmd_listhdr(args),
            "ch" | "chnhdr" => self.run_mutating(|s| s.cmd_chnhdr(args)),
            "wh" | "writehdr" => self.cmd_writehdr(),
            "cut" => self.run_mutating(|s| s.cmd_cut(args, false)),
            "cutim" => self.run_mutating(|s| s.cmd_cut(args, true)),
            "merge" => self.run_mutating(|s| s.cmd_merge(args)),
            "sort" => self.run_mutating(|s| s.cmd_sort(args)),
            "dc" | "deletechannel" => self.run_mutating(|s| s.cmd_deletechannel(args)),
            "fg" | "funcgen" => self.run_mutating(|s| s.cmd_funcgen(args)),

            // --- Preprocessing & Math ---
            "rmean" | "demean" => self.run_mutating(|s| s.cmd_rmean()),
            "rtrend" => self.run_mutating(|s| s.cmd_rtrend(args)),
            "rglitches" => self.run_mutating(|s| s.cmd_rglitches(args)),
            "taper" => self.run_mutating(|s| s.cmd_taper(args)),
            "decimate" => self.run_mutating(|s| s.cmd_decimate(args)),
            "interpolate" | "interp" => self.run_mutating(|s| s.cmd_interpolate(args)),
            "dif" | "differentiate" => self.run_mutating(|s| s.cmd_dif(args)),
            "int" | "integrate" => self.run_mutating(|s| s.cmd_int(args)),
            "mul" => self.run_mutating(|s| s.cmd_scalar_math("mul", args)),
            "div" => self.run_mutating(|s| s.cmd_scalar_math("div", args)),
            "add" => self.run_mutating(|s| s.cmd_scalar_math("add", args)),
            "sub" => self.run_mutating(|s| s.cmd_scalar_math("sub", args)),
            "sqr" => self.run_mutating(|s| s.cmd_unary_math("sqr")),
            "sqrt" => self.run_mutating(|s| s.cmd_unary_math("sqrt")),
            "abs" => self.run_mutating(|s| s.cmd_unary_math("abs")),
            "log10" => self.run_mutating(|s| s.cmd_unary_math("log10")),
            "exp" => self.run_mutating(|s| s.cmd_unary_math("exp")),
            "reverse" | "rev" => self.run_mutating(|s| s.cmd_unary_math("reverse")),
            "addf" => self.run_mutating(|s| s.cmd_file_math("addf", args)),
            "subf" => self.run_mutating(|s| s.cmd_file_math("subf", args)),
            "mulf" => self.run_mutating(|s| s.cmd_file_math("mulf", args)),
            "divf" => self.run_mutating(|s| s.cmd_file_math("divf", args)),
            "chpf" => self.cmd_chpf(args),

            // --- Filtering & Deconvolution ---
            "bp" | "bandpass" => self.run_mutating(|s| s.cmd_filter("bp", args)),
            "lp" | "lowpass" => self.run_mutating(|s| s.cmd_filter("lp", args)),
            "hp" | "highpass" => self.run_mutating(|s| s.cmd_filter("hp", args)),
            "br" | "bandrej" | "bandstop" => self.run_mutating(|s| s.cmd_filter("br", args)),
            "transfer" | "trans" => self.run_mutating(|s| s.cmd_transfer(args)),

            // --- Spectral, Picking, 3C & Native TauP ---
            "fft" => self.run_mutating(|s| s.cmd_fft(args)),
            "ifft" => self.run_mutating(|s| s.cmd_ifft()),
            "hilbert" => self.run_mutating(|s| s.cmd_hilbert()),
            "envelope" | "env" => self.run_mutating(|s| s.cmd_envelope()),
            "correlate" | "cor" => self.run_mutating(|s| s.cmd_correlate(args)),
            "apk" => self.run_mutating(|s| s.cmd_apk(args)),
            "markptp" => self.run_mutating(|s| s.cmd_markptp(args)),
            "rot" | "rotate" => self.run_mutating(|s| s.cmd_rotate(args)),
            "taup" => self.run_mutating(|s| s.cmd_taup(args)),

            // --- SVG Plotting ---
            "p1" | "plot1" | "p" | "plot" => self.cmd_plot_waveforms(args, WaveformLayout::Stacked),
            "p2" | "plot2" => self.cmd_plot_waveforms(args, WaveformLayout::Overlaid),
            "prs" | "plotrecordsection" => self.cmd_plot_record_section(args),
            "spectrogram" => self.cmd_plot_spectrogram(args),
            "beachball" => self.cmd_plot_beachball(args),
            "plottaup" => self.cmd_plot_taup(args),

            unknown => Err(EarthSnailError::Format(format!(
                "unknown SAC command '{unknown}' (type 'help' to list supported commands)"
            ))),
        }
    }

    fn cmd_help(&mut self) {
        self.output_log.push_str(
            "EarthSnail SAC> Interactive Shell & Macro Engine Commands:\n\
             I/O & Headers : r/read [more], w/write [sac|le|be|alpha|mseed|ms3|over], lh/listhdr,\n\
                             ch/chnhdr, wh/writehdr, cut, cutim, merge, sort, dc/deletechannel, fg/funcgen\n\
             Preprocess    : rmean, rtrend, rglitches, taper, decimate, interpolate, dif, int\n\
             Waveform Math : add, sub, mul, div, sqr, sqrt, abs, log10, exp, reverse, addf, subf, mulf, divf\n\
             Filter/Resp   : bp/bandpass, lp/lowpass, hp/highpass, br/bandrej, transfer\n\
             Analysis      : fft, ifft, hilbert, envelope, correlate, apk, markptp, rot, taup\n\
             SVG Plotting  : p1/plot1, p2/plot2, prs/plotrecordsection, spectrogram, beachball, plottaup\n\
             Scripting     : setbb, getbb, unsetbb, m/macro, do ... enddo, undo, echo, pwd, cd, quit\n",
        );
    }

    fn cmd_undo(&mut self, args: &[String]) -> Result<()> {
        let steps = match args.first() {
            Some(s) => s.parse::<usize>().map_err(|_| {
                EarthSnailError::InvalidParameter(format!("invalid undo step count '{s}'"))
            })?,
            None => 1,
        };
        if steps == 0 {
            return Ok(());
        }
        if self.undo_stack.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'undo': undo history stack is empty".into(),
            ));
        }
        let actual = steps.min(self.undo_stack.len());
        let mut snap = None;
        for _ in 0..actual {
            snap = self.undo_stack.pop();
        }
        if let Some(s) = snap {
            self.traces = s.traces;
            self.filenames = s.filenames;
            self.spectra = s.spectra;
            self.cut_enabled = s.cut_enabled;
            self.cut_spec = s.cut_spec;
        }
        Ok(())
    }

    fn cmd_setbb(&mut self, args: &[String]) -> Result<()> {
        if args.len() < 2 {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'setbb' expects: setbb <variable> <value>".into(),
            ));
        }
        let name = args[0].trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(EarthSnailError::InvalidParameter(format!(
                "invalid blackboard variable name '{name}'"
            )));
        }
        let val = args[1..].join(" ");
        self.blackboard.insert(name.to_ascii_uppercase(), val);
        Ok(())
    }

    fn cmd_getbb(&mut self, args: &[String]) -> Result<()> {
        if args.is_empty() || (args.len() == 1 && args[0].eq_ignore_ascii_case("all")) {
            for (k, v) in &self.blackboard {
                self.output_log.push_str(&format!(" {k} = {v}\n"));
            }
            return Ok(());
        }
        for name in args {
            let key = name.to_ascii_uppercase();
            match self.blackboard.get(&key) {
                Some(v) => self.output_log.push_str(&format!(" {key} = {v}\n")),
                None => {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "undefined blackboard variable '{name}'"
                    )));
                }
            }
        }
        Ok(())
    }

    fn cmd_unsetbb(&mut self, args: &[String]) -> Result<()> {
        if args.len() == 1 && args[0].eq_ignore_ascii_case("all") {
            self.blackboard.clear();
            return Ok(());
        }
        for name in args {
            self.blackboard.remove(&name.to_ascii_uppercase());
        }
        Ok(())
    }

    // ========================================================================
    // I/O & Header Commands
    // ========================================================================

    fn cmd_read(&mut self, args: &[String]) -> Result<()> {
        if args.is_empty() {
            // Re-read current filenames:
            if self.filenames.is_empty() {
                return Err(EarthSnailError::InvalidParameter(
                    "SAC 'read' requires at least one file path or wildcard pattern".into(),
                ));
            }
            let prev = self.filenames.clone();
            return self.load_file_patterns(&prev, false);
        }
        let mut append = false;
        let mut patterns: Vec<String> = Vec::new();
        for tok in args {
            if tok.eq_ignore_ascii_case("more") && patterns.is_empty() {
                append = true;
            } else {
                patterns.push(tok.clone());
            }
        }
        if patterns.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'read more' requires at least one file path or wildcard pattern".into(),
            ));
        }
        self.load_file_patterns(&patterns, append)
    }

    fn load_file_patterns(&mut self, patterns: &[String], append: bool) -> Result<()> {
        let mut expanded_paths: Vec<String> = Vec::new();
        for pat in patterns {
            let mut matched = self.expand_file_glob(pat)?;
            if matched.is_empty() {
                return Err(EarthSnailError::Io(format!(
                    "SAC 'read': no files matched '{pat}'"
                )));
            }
            expanded_paths.append(&mut matched);
        }

        let mut new_traces = Vec::new();
        let mut new_names = Vec::new();
        let mut new_specs = Vec::new();

        for path_str in expanded_paths {
            let bytes = self.read_bytes_any(&path_str)?;
            let stream = read_waveforms_bytes(&bytes)?;
            for mut tr in stream.traces {
                ensure_sac_header_synced(&mut tr);
                if self.cut_enabled {
                    if let Some(spec) = self.cut_spec {
                        tr = apply_sac_cut_to_trace(&tr, spec)?;
                    }
                }
                new_traces.push(tr);
                new_names.push(path_str.clone());
                new_specs.push(None);
            }
        }

        if !append {
            self.traces.clear();
            self.filenames.clear();
            self.spectra.clear();
        }
        self.traces.extend(new_traces);
        self.filenames.extend(new_names);
        self.spectra.extend(new_specs);
        Ok(())
    }

    fn expand_file_glob(&self, pattern: &str) -> Result<Vec<String>> {
        if !pattern.contains('*') && !pattern.contains('?') {
            return Ok(vec![String::from(pattern)]);
        }

        // Check virtual_files first:
        let mut virt_matches: Vec<String> = self
            .virtual_files
            .keys()
            .filter(|k| wildcard_match(pattern, k))
            .cloned()
            .collect();
        virt_matches.sort();
        if !virt_matches.is_empty() {
            return Ok(virt_matches);
        }

        #[cfg(miri)]
        {
            Ok(Vec::new())
        }
        #[cfg(not(miri))]
        {
            let full_pat = self.resolve_path(pattern);
            let dir = full_pat
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(&self.working_dir);
            let file_pat = full_pat
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    EarthSnailError::InvalidParameter(format!("invalid glob pattern '{pattern}'"))
                })?;

            let entries = fs::read_dir(dir).map_err(|e| {
                EarthSnailError::Io(format!(
                    "cannot scan directory '{}' for '{}': {e}",
                    dir.display(),
                    pattern
                ))
            })?;
            let mut hits: Vec<PathBuf> = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| EarthSnailError::Io(format!("{e}")))?;
                let path = entry.path();
                if path.is_file() {
                    if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                        if wildcard_match(file_pat, fname) {
                            hits.push(path);
                        }
                    }
                }
            }
            hits.sort();
            Ok(hits
                .into_iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect())
        }
    }

    fn cmd_write(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("write")?;
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum WriteFmt {
            Auto,
            SacLe,
            SacBe,
            SacAlpha,
            Mseed2,
            Mseed3,
            GeoCsv,
            SList,
            TsPair,
        }

        let mut fmt = WriteFmt::Auto;
        let mut over = false;
        let mut out_files: Vec<String> = Vec::new();

        for tok in args {
            let lc = tok.to_ascii_lowercase();
            match lc.as_str() {
                "sac" | "le" => fmt = WriteFmt::SacLe,
                "be" => fmt = WriteFmt::SacBe,
                "alpha" => fmt = WriteFmt::SacAlpha,
                "mseed" | "mseed2" | "ms2" => fmt = WriteFmt::Mseed2,
                "ms3" | "mseed3" => fmt = WriteFmt::Mseed3,
                "geocsv" | "csv" => fmt = WriteFmt::GeoCsv,
                "slist" => fmt = WriteFmt::SList,
                "tspair" => fmt = WriteFmt::TsPair,
                "over" => over = true,
                _ => out_files.push(tok.clone()),
            }
        }

        if over {
            if self.filenames.len() != self.traces.len()
                || self.filenames.iter().any(|f| f.is_empty())
            {
                return Err(EarthSnailError::InvalidParameter(
                    "SAC 'write over': one or more traces in memory have no associated disk filename".into(),
                ));
            }
            out_files = self.filenames.clone();
        }

        if out_files.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'write' requires 'over' or output filename(s)".into(),
            ));
        }

        // Multi-trace container formats (`mseed`, `ms3`, `geocsv`, `slist`, `tspair`) allow writing
        // all traces into a single file when `out_files.len() == 1`!
        let infer_fmt = |path: &str, explicit: WriteFmt| -> WriteFmt {
            if explicit != WriteFmt::Auto {
                return explicit;
            }
            let lc = path.to_ascii_lowercase();
            if lc.ends_with(".mseed") || lc.ends_with(".msd") {
                WriteFmt::Mseed2
            } else if lc.ends_with(".ms3") || lc.ends_with(".mseed3") {
                WriteFmt::Mseed3
            } else if lc.ends_with(".alpha") || lc.ends_with(".sac.alpha") {
                WriteFmt::SacAlpha
            } else if lc.ends_with(".geocsv") || lc.ends_with(".csv") {
                WriteFmt::GeoCsv
            } else if lc.ends_with(".slist") {
                WriteFmt::SList
            } else if lc.ends_with(".tspair") {
                WriteFmt::TsPair
            } else {
                WriteFmt::SacLe
            }
        };

        if out_files.len() == 1 && self.traces.len() > 1 {
            let single_fmt = infer_fmt(&out_files[0], fmt);
            match single_fmt {
                WriteFmt::Mseed2 => {
                    let st = self.to_stream();
                    let bytes = write_mseed_stream_bytes(&st, MseedWriteOptions::default())?;
                    return self.write_bytes_any(&out_files[0], &bytes);
                }
                WriteFmt::Mseed3 => {
                    let st = self.to_stream();
                    let bytes = write_mseed3_stream_bytes(&st, &Mseed3WriteOptions::default())?;
                    return self.write_bytes_any(&out_files[0], &bytes);
                }
                WriteFmt::GeoCsv => {
                    let st = self.to_stream();
                    let txt = write_geocsv_stream(&st, true)?;
                    return self.write_bytes_any(&out_files[0], txt.as_bytes());
                }
                WriteFmt::SList => {
                    let st = self.to_stream();
                    let txt = write_slist_stream(&st)?;
                    return self.write_bytes_any(&out_files[0], txt.as_bytes());
                }
                WriteFmt::TsPair => {
                    let st = self.to_stream();
                    let txt = write_tspair_stream(&st)?;
                    return self.write_bytes_any(&out_files[0], txt.as_bytes());
                }
                _ => {}
            }
        }

        if out_files.len() != self.traces.len() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "SAC 'write': {} traces in memory, but {} output filenames specified",
                self.traces.len(),
                out_files.len()
            )));
        }

        for (idx, target) in out_files.iter().enumerate().take(self.traces.len()) {
            let f_mode = infer_fmt(target, fmt);
            let tr = self.traces[idx].clone();
            let bytes = match f_mode {
                WriteFmt::SacLe | WriteFmt::Auto => write_sac_bytes(&tr, SacEndian::Little)?,
                WriteFmt::SacBe => write_sac_bytes(&tr, SacEndian::Big)?,
                WriteFmt::SacAlpha => write_sac_alpha_bytes(&tr)?,
                WriteFmt::Mseed2 => write_mseed_stream_bytes(
                    &Stream::from_traces(vec![tr]),
                    MseedWriteOptions::default(),
                )?,
                WriteFmt::Mseed3 => write_mseed3_stream_bytes(
                    &Stream::from_traces(vec![tr]),
                    &Mseed3WriteOptions::default(),
                )?,
                WriteFmt::GeoCsv => {
                    write_geocsv_stream(&Stream::from_traces(vec![tr]), true)?.into_bytes()
                }
                WriteFmt::SList => write_slist_stream(&Stream::from_traces(vec![tr]))?.into_bytes(),
                WriteFmt::TsPair => {
                    write_tspair_stream(&Stream::from_traces(vec![tr]))?.into_bytes()
                }
            };
            self.write_bytes_any(target, &bytes)?;
            if self.filenames[idx].is_empty()
                && matches!(f_mode, WriteFmt::SacLe | WriteFmt::SacBe | WriteFmt::Auto)
            {
                self.filenames[idx] = target.clone();
            }
        }
        Ok(())
    }

    fn cmd_writehdr(&mut self) -> Result<()> {
        self.require_traces("writehdr")?;
        if self.filenames.len() != self.traces.len() || self.filenames.iter().any(|f| f.is_empty())
        {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'writehdr': traces have no associated SAC files".into(),
            ));
        }
        for idx in 0..self.traces.len() {
            let path = self.filenames[idx].clone();
            let existing = self.read_bytes_any(&path)?;
            let endian = SacHeader::detect_endian(&existing).unwrap_or(SacEndian::Little);
            let updated = write_sac_bytes(&self.traces[idx], endian)?;
            self.write_bytes_any(&path, &updated)?;
        }
        Ok(())
    }

    fn cmd_listhdr(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("listhdr")?;
        let selected: Vec<String> = args
            .iter()
            .filter(|a| !a.eq_ignore_ascii_case("default") && !a.eq_ignore_ascii_case("inclusive"))
            .map(|a| a.to_ascii_lowercase())
            .collect();

        for (idx, tr) in self.traces.iter().enumerate() {
            let fname = self
                .filenames
                .get(idx)
                .map(String::as_str)
                .unwrap_or("<memory>");
            self.output_log.push_str(&format!(
                "\n  FILE: {} - {} (#{})\n ------------------------------------------\n",
                fname,
                tr.stats.channel.seed_id(),
                idx + 1
            ));
            let Some(ref sac) = tr.stats.sac else {
                continue;
            };
            if selected.is_empty() {
                for &fnm in &SAC_FLOAT_NAMES {
                    if let Some(v) = sac.get_float(fnm) {
                        self.output_log.push_str(&format!(
                            "  {:<10} = {:.6e}\n",
                            fnm.to_ascii_uppercase(),
                            v
                        ));
                    }
                }
                for &inm in &SAC_INT_NAMES {
                    if let Some(v) = sac.get_int(inm) {
                        self.output_log.push_str(&format!(
                            "  {:<10} = {}\n",
                            inm.to_ascii_uppercase(),
                            v
                        ));
                    }
                }
                for &snm in &SAC_STR_NAMES {
                    if let Some(v) = sac.get_str(snm) {
                        self.output_log.push_str(&format!(
                            "  {:<10} = {}\n",
                            snm.to_ascii_uppercase(),
                            v
                        ));
                    }
                }
            } else {
                for field in &selected {
                    if let Some(v) = sac.get_float(field) {
                        self.output_log.push_str(&format!(
                            "  {:<10} = {:.6e}\n",
                            field.to_ascii_uppercase(),
                            v
                        ));
                    } else if let Some(v) = sac.get_int(field) {
                        self.output_log.push_str(&format!(
                            "  {:<10} = {}\n",
                            field.to_ascii_uppercase(),
                            v
                        ));
                    } else if let Some(v) = sac.get_str(field) {
                        self.output_log.push_str(&format!(
                            "  {:<10} = {}\n",
                            field.to_ascii_uppercase(),
                            v
                        ));
                    } else {
                        self.output_log.push_str(&format!(
                            "  {:<10} = UNDEFINED\n",
                            field.to_ascii_uppercase()
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn cmd_chnhdr(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("chnhdr")?;
        if args.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'chnhdr' requires <field> <value> pairs".into(),
            ));
        }

        let mut target_file: Option<usize> = None; // None = all traces
        let mut i = 0usize;
        while i < args.len() {
            let key = args[i].to_ascii_lowercase();
            if key == "file" {
                let f_tok = args.get(i + 1).ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "SAC 'chnhdr file' expects index or 'all'".into(),
                    )
                })?;
                if f_tok.eq_ignore_ascii_case("all") {
                    target_file = None;
                } else {
                    let idx1 = f_tok.parse::<usize>().map_err(|_| {
                        EarthSnailError::InvalidParameter(format!(
                            "invalid file number '{f_tok}' in chnhdr"
                        ))
                    })?;
                    if idx1 == 0 || idx1 > self.traces.len() {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "file index {idx1} out of range 1..={}",
                            self.traces.len()
                        )));
                    }
                    target_file = Some(idx1 - 1);
                }
                i += 2;
                continue;
            }
            if key == "allhdr" {
                i += 1;
                continue;
            }

            let val_str = args.get(i + 1).ok_or_else(|| {
                EarthSnailError::InvalidParameter(format!(
                    "SAC 'chnhdr': missing value for header field '{}'",
                    args[i]
                ))
            })?;

            let indices: Vec<usize> = match target_file {
                Some(k) => vec![k],
                None => (0..self.traces.len()).collect(),
            };
            for t_idx in indices {
                ensure_sac_header_synced(&mut self.traces[t_idx]);
                apply_single_chnhdr(&mut self.traces[t_idx], &key, val_str)?;
            }
            i += 2;
        }
        Ok(())
    }

    fn cmd_cut(&mut self, args: &[String], immediate_only: bool) -> Result<()> {
        if args.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'cut' expects 'on', 'off', or '<ref1> <off1> <ref2> <off2>'".into(),
            ));
        }
        if !immediate_only && args.len() == 1 {
            if args[0].eq_ignore_ascii_case("on") {
                self.cut_enabled = true;
                return Ok(());
            }
            if args[0].eq_ignore_ascii_case("off") {
                self.cut_enabled = false;
                return Ok(());
            }
        }

        let spec = parse_cut_spec(args)?;
        if !immediate_only {
            self.cut_spec = Some(spec);
            self.cut_enabled = true;
        }
        if !self.traces.is_empty() {
            let mut cut_traces = Vec::with_capacity(self.traces.len());
            for tr in &self.traces {
                cut_traces.push(apply_sac_cut_to_trace(tr, spec)?);
            }
            self.traces = cut_traces;
            self.spectra = vec![None; self.traces.len()];
        }
        Ok(())
    }

    fn cmd_merge(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("merge")?;
        let mut strat = MergeStrategy::FillZero;
        let mut i = 0;
        while i < args.len() {
            if args[i].eq_ignore_ascii_case("gap") {
                if let Some(mode) = args.get(i + 1) {
                    if mode.eq_ignore_ascii_case("interp")
                        || mode.eq_ignore_ascii_case("interpolate")
                    {
                        strat = MergeStrategy::Interpolate;
                    } else if mode.eq_ignore_ascii_case("zero") {
                        strat = MergeStrategy::FillZero;
                    }
                    i += 2;
                    continue;
                }
            } else if args[i].eq_ignore_ascii_case("interp") {
                strat = MergeStrategy::Interpolate;
            }
            i += 1;
        }
        let merged = merge_stream(&self.to_stream(), strat)?;
        self.traces = merged.traces;
        for tr in &mut self.traces {
            ensure_sac_header_synced(tr);
        }
        self.filenames = vec![String::new(); self.traces.len()];
        self.spectra = vec![None; self.traces.len()];
        Ok(())
    }

    fn cmd_sort(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("sort")?;
        let key = args
            .first()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| String::from("default"));
        let descend = args
            .iter()
            .any(|a| a.eq_ignore_ascii_case("descend") || a.eq_ignore_ascii_case("rev"));

        let mut perm: Vec<usize> = (0..self.traces.len()).collect();
        perm.sort_by(|&ia, &ib| {
            let a = &self.traces[ia];
            let b = &self.traces[ib];
            let ord = if key == "default" {
                a.stats
                    .channel
                    .cmp(&b.stats.channel)
                    .then_with(|| a.stats.starttime.cmp(&b.stats.starttime))
            } else if let (Some(sa), Some(sb)) = (&a.stats.sac, &b.stats.sac) {
                if let (Some(fa), Some(fb)) = (sa.get_float(&key), sb.get_float(&key)) {
                    fa.total_cmp(&fb)
                } else if let (Some(ka), Some(kb)) = (sa.get_str(&key), sb.get_str(&key)) {
                    ka.cmp(kb)
                } else if let (Some(na), Some(nb)) = (sa.get_int(&key), sb.get_int(&key)) {
                    na.cmp(&nb)
                } else {
                    a.stats.channel.cmp(&b.stats.channel)
                }
            } else {
                a.stats.channel.cmp(&b.stats.channel)
            };
            if descend {
                ord.reverse()
            } else {
                ord
            }
        });

        self.traces = perm.iter().map(|&k| self.traces[k].clone()).collect();
        self.filenames = perm
            .iter()
            .map(|&k| self.filenames.get(k).cloned().unwrap_or_default())
            .collect();
        self.spectra = perm
            .iter()
            .map(|&k| self.spectra.get(k).cloned().unwrap_or(None))
            .collect();
        Ok(())
    }

    fn cmd_deletechannel(&mut self, args: &[String]) -> Result<()> {
        if args.is_empty() {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'deletechannel' expects 'all', 1-based channel indices, or wildcard patterns"
                    .into(),
            ));
        }
        if args.len() == 1 && args[0].eq_ignore_ascii_case("all") {
            self.traces.clear();
            self.filenames.clear();
            self.spectra.clear();
            return Ok(());
        }

        let mut remove_mask = vec![false; self.traces.len()];
        for tok in args {
            if let Ok(idx1) = tok.parse::<usize>() {
                if idx1 == 0 || idx1 > self.traces.len() {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "SAC 'deletechannel': index {idx1} out of bounds 1..={}",
                        self.traces.len()
                    )));
                }
                remove_mask[idx1 - 1] = true;
            } else {
                for (k, tr) in self.traces.iter().enumerate() {
                    if tr.stats.channel.matches_pattern(tok)
                        || wildcard_match(tok, &tr.stats.channel.channel)
                    {
                        remove_mask[k] = true;
                    }
                }
            }
        }

        let mut next_tr = Vec::new();
        let mut next_fn = Vec::new();
        let mut next_sp = Vec::new();
        for (k, &removed) in remove_mask.iter().enumerate().take(self.traces.len()) {
            if !removed {
                next_tr.push(self.traces[k].clone());
                next_fn.push(self.filenames.get(k).cloned().unwrap_or_default());
                next_sp.push(self.spectra.get(k).cloned().unwrap_or(None));
            }
        }
        self.traces = next_tr;
        self.filenames = next_fn;
        self.spectra = next_sp;
        Ok(())
    }

    fn cmd_funcgen(&mut self, args: &[String]) -> Result<()> {
        let mut kind = "impulse";
        let mut delta = 0.01_f64;
        let mut npts = 1000_usize;
        let mut b_start = 0.0_f64;
        let mut sine_freq = 1.0_f64;
        let mut sine_phase_deg = 0.0_f64;

        let mut i = 0usize;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "impulse" => kind = "impulse",
                "sine" => {
                    kind = "sine";
                    if let Some(f) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                        sine_freq = f;
                        i += 1;
                        if let Some(ph) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                            sine_phase_deg = ph;
                            i += 1;
                        }
                    }
                }
                "boxcar" => kind = "boxcar",
                "triangle" => kind = "triangle",
                "step" => kind = "step",
                "random" => kind = "random",
                "line" => kind = "line",
                "delta" => {
                    let v = parse_next_f64(args, i, "funcgen delta")?;
                    if v <= 0.0 {
                        return Err(EarthSnailError::InvalidParameter(
                            "funcgen delta must be > 0".into(),
                        ));
                    }
                    delta = v;
                    i += 1;
                }
                "npts" => {
                    let v = parse_next_usize(args, i, "funcgen npts")?;
                    if v == 0 || v > MAX_FUNCGEN_NPTS {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "funcgen npts must lie in 1..={MAX_FUNCGEN_NPTS}"
                        )));
                    }
                    npts = v;
                    i += 1;
                }
                "begin" | "b" => {
                    b_start = parse_next_f64(args, i, "funcgen begin")?;
                    i += 1;
                }
                other => {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "unrecognized funcgen option '{other}'"
                    )));
                }
            }
            i += 1;
        }

        let mut data = vec![0.0_f64; npts];
        match kind {
            "impulse" => {
                data[npts / 2] = 1.0;
            }
            "sine" => {
                let ph_rad = sine_phase_deg.to_radians();
                for (k, slot) in data.iter_mut().enumerate() {
                    let t = (k as f64) * delta;
                    *slot = math::sin(TAU * sine_freq * t + ph_rad);
                }
            }
            "boxcar" => {
                let lo = npts / 4;
                let hi = (3 * npts / 4).max(lo + 1).min(npts);
                data[lo..hi].fill(1.0);
            }
            "triangle" => {
                let mid = (npts / 2).max(1) as f64;
                for (k, slot) in data.iter_mut().enumerate() {
                    *slot = (1.0 - ((k as f64) - mid).abs() / mid).max(0.0);
                }
            }
            "step" => {
                data[(npts / 2)..].fill(1.0);
            }
            "line" => {
                for (k, slot) in data.iter_mut().enumerate() {
                    *slot = b_start + (k as f64) * delta;
                }
            }
            "random" => {
                // Deterministic xorshift64* PRNG for reproducible synthetic noise across platforms & Miri:
                let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
                for slot in &mut data {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let u = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
                    *slot = ((u >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0;
                }
            }
            _ => unreachable!(),
        }

        let reftime: UtcTimestamp = "2000-01-01T00:00:00Z".parse()?;
        let starttime = reftime.add_secs_f64(b_start)?;
        let ch = ChannelId::new("XX", "FUNCGEN", "00", "BHZ");
        let mut stats = TraceStats::new(ch, starttime, 1.0 / delta);
        let mut sac = SacHeader::new();
        sac.set_reference_time(reftime);
        sac.b = b_start;
        sac.delta = delta;
        sac.npts = npts as i32;
        sac.e = b_start + ((npts.saturating_sub(1)) as f64) * delta;
        let _ = sac.set_str("knetwk", "XX");
        let _ = sac.set_str("kstnm", "FUNCGEN");
        let _ = sac.set_str("khole", "00");
        let _ = sac.set_str("kcmpnm", "BHZ");
        sac.update_dep_stats(&data);
        stats.sac = Some(sac);

        self.traces = vec![Trace::new(stats, data)];
        self.filenames = vec![String::new()];
        self.spectra = vec![None];
        Ok(())
    }

    // ========================================================================
    // Preprocessing & Waveform Math
    // ========================================================================

    fn cmd_rmean(&mut self) -> Result<()> {
        self.require_traces("rmean")?;
        for tr in &mut self.traces {
            rmean_in_place(&mut tr.data);
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_rtrend(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("rtrend")?;
        if let Some(ord) = args.first().and_then(|a| a.parse::<usize>().ok()) {
            for tr in &mut self.traces {
                earthsnail_dsp::detrend_polynomial_in_place(&mut tr.data, ord)?;
                refresh_dep_stats(tr);
            }
            return Ok(());
        }
        for tr in &mut self.traces {
            rtrend_in_place(&mut tr.data);
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_rglitches(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("rglitches")?;
        let mut sigma_thresh = 6.0_f64;
        let mut i = 0;
        while i < args.len() {
            if args[i].eq_ignore_ascii_case("threshold") || args[i].eq_ignore_ascii_case("thr") {
                sigma_thresh = parse_next_f64(args, i, "rglitches threshold")?;
                i += 2;
                continue;
            } else if let Ok(v) = args[i].parse::<f64>() {
                sigma_thresh = v;
            }
            i += 1;
        }
        if sigma_thresh <= 0.0 || !sigma_thresh.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "rglitches threshold must be finite and > 0".into(),
            ));
        }

        // Glitch-immune 3-point Discrete Morse local persistence spike excision:
        // Zero-persistence isolated 1-sample critical pairs satisfy |x[k] - median3(x[k-1], x[k], x[k+1])| >> sigma_mad!
        for tr in &mut self.traces {
            remove_morse_glitches_in_place(&mut tr.data, sigma_thresh);
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_taper(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("taper")?;
        let mut ttype = TaperType::Hann;
        let mut width = 0.05_f64;
        let mut i = 0;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "type" | "t" => {
                    let tn = args.get(i + 1).ok_or_else(|| {
                        EarthSnailError::InvalidParameter(
                            "taper 'type' expects hanning|hamming|cosine".into(),
                        )
                    })?;
                    ttype = parse_sac_taper_type(tn)?;
                    i += 1;
                }
                "hanning" | "hann" => ttype = TaperType::Hann,
                "hamming" => ttype = TaperType::Hamming,
                "cosine" => ttype = TaperType::SacCosine,
                "width" | "w" => {
                    width = parse_next_f64(args, i, "taper width")?;
                    i += 1;
                }
                other => {
                    if let Ok(w) = other.parse::<f64>() {
                        width = w;
                    } else {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown taper option '{other}'"
                        )));
                    }
                }
            }
            i += 1;
        }
        for tr in &mut self.traces {
            taper_in_place(&mut tr.data, width, ttype, TaperSide::Both)?;
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_decimate(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("decimate")?;
        let factor = args
            .first()
            .ok_or_else(|| {
                EarthSnailError::InvalidParameter(
                    "SAC 'decimate' requires an integer factor".into(),
                )
            })?
            .parse::<usize>()
            .map_err(|_| EarthSnailError::InvalidParameter("invalid decimate factor".into()))?;
        let mut anti_alias = true;
        for w in args.windows(2) {
            if w[0].eq_ignore_ascii_case("filter") && w[1].eq_ignore_ascii_case("off") {
                anti_alias = false;
            }
        }
        if args.iter().any(|a| a.eq_ignore_ascii_case("no")) {
            anti_alias = false;
        }

        let mut next = Vec::with_capacity(self.traces.len());
        for tr in &self.traces {
            let old_reftime = tr.stats.sac.as_ref().and_then(SacHeader::reference_time);
            let old_b = tr.stats.sac.as_ref().and_then(|s| s.get_float("b"));
            let mut dtr = decimate_trace(tr, factor, anti_alias)?;
            restore_reftime_and_b(&mut dtr, old_reftime, old_b);
            next.push(dtr);
        }
        self.traces = next;
        Ok(())
    }

    fn cmd_interpolate(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("interpolate")?;
        let mut target_dt: Option<f64> = None;
        let mut target_npts: Option<usize> = None;
        let mut i = 0;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "delta" => {
                    target_dt = Some(parse_next_f64(args, i, "interpolate delta")?);
                    i += 1;
                }
                "npts" => {
                    target_npts = Some(parse_next_usize(args, i, "interpolate npts")?);
                    i += 1;
                }
                other => {
                    if target_dt.is_none() {
                        if let Ok(dt) = other.parse::<f64>() {
                            target_dt = Some(dt);
                            i += 1;
                            continue;
                        }
                    }
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "unknown interpolate option '{other}'"
                    )));
                }
            }
            i += 1;
        }

        for tr in &mut self.traces {
            if tr.npts() < 2 {
                continue;
            }
            let old_dt = tr.delta();
            let span_s = ((tr.npts() - 1) as f64) * old_dt;
            let (new_dt, new_npts) = if let Some(dt) = target_dt {
                if !dt.is_finite() || dt <= 0.0 {
                    return Err(EarthSnailError::InvalidParameter(
                        "interpolate delta must be finite and > 0".into(),
                    ));
                }
                let n = (math::floor(span_s / dt + 1e-12) as usize) + 1;
                (dt, n)
            } else if let Some(n) = target_npts {
                if !(2..=MAX_FUNCGEN_NPTS).contains(&n) {
                    return Err(EarthSnailError::InvalidParameter(
                        "interpolate npts must be >= 2".into(),
                    ));
                }
                (span_s / ((n - 1) as f64), n)
            } else {
                return Err(EarthSnailError::InvalidParameter(
                    "SAC 'interpolate' requires 'delta <dt>' or 'npts <n>'".into(),
                ));
            };

            let resampled = lanczos_resample(
                &tr.data,
                0.0,
                old_dt,
                0.0,
                new_dt,
                new_npts,
                20,
                LanczosWindow::Lanczos,
            )?;
            tr.data = resampled;
            tr.stats.sampling_rate = 1.0 / new_dt;
            if let Some(ref mut sac) = tr.stats.sac {
                let b_val = sac.get_float("b").unwrap_or(0.0);
                sac.delta = new_dt;
                sac.npts = tr.data.len() as i32;
                sac.e = b_val + ((tr.data.len().saturating_sub(1)) as f64) * new_dt;
                sac.update_dep_stats(&tr.data);
            }
        }
        Ok(())
    }

    fn cmd_dif(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("dif")?;
        let spectral = args.iter().any(|a| a.eq_ignore_ascii_case("spectral"));
        for tr in &mut self.traces {
            let dt = tr.delta();
            tr.data = if spectral {
                differentiate_spectral(&tr.data, dt)?
            } else {
                differentiate_gradient(&tr.data, dt)?
            };
            if let Some(ref mut sac) = tr.stats.sac {
                sac.idep = match SacDepType::from_i32(sac.idep) {
                    Some(SacDepType::DisplacementNm) => SacDepType::VelocityNmPerSec as i32,
                    Some(SacDepType::VelocityNmPerSec) => SacDepType::AccelerationNmPerSec2 as i32,
                    _ => sac.idep,
                };
            }
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_int(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("int")?;
        let spectral = args.iter().any(|a| a.eq_ignore_ascii_case("spectral"));
        for tr in &mut self.traces {
            let dt = tr.delta();
            tr.data = if spectral {
                integrate_spectral(&tr.data, dt)?
            } else {
                integrate_trapezoidal(&tr.data, dt)?
            };
            if let Some(ref mut sac) = tr.stats.sac {
                sac.idep = match SacDepType::from_i32(sac.idep) {
                    Some(SacDepType::AccelerationNmPerSec2) => SacDepType::VelocityNmPerSec as i32,
                    Some(SacDepType::VelocityNmPerSec) => SacDepType::DisplacementNm as i32,
                    _ => sac.idep,
                };
            }
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_scalar_math(&mut self, op: &str, args: &[String]) -> Result<()> {
        self.require_traces(op)?;
        let default_val = match op {
            "mul" | "div" => 1.0,
            _ => 0.0,
        };
        if args.is_empty() {
            for tr in &mut self.traces {
                apply_scalar_op(&mut tr.data, op, default_val)?;
                refresh_dep_stats(tr);
            }
            return Ok(());
        }
        if args.len() == 1 {
            let c = args[0].parse::<f64>().map_err(|_| {
                EarthSnailError::InvalidParameter(format!(
                    "SAC '{op}': invalid number '{}'",
                    args[0]
                ))
            })?;
            for tr in &mut self.traces {
                apply_scalar_op(&mut tr.data, op, c)?;
                refresh_dep_stats(tr);
            }
            return Ok(());
        }
        if args.len() != self.traces.len() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "SAC '{op}': expected 1 constant or {} constants (matching trace count), got {}",
                self.traces.len(),
                args.len()
            )));
        }
        for (tr, s_val) in self.traces.iter_mut().zip(args.iter()) {
            let c = s_val.parse::<f64>().map_err(|_| {
                EarthSnailError::InvalidParameter(format!("SAC '{op}': invalid number '{s_val}'"))
            })?;
            apply_scalar_op(&mut tr.data, op, c)?;
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_unary_math(&mut self, op: &str) -> Result<()> {
        self.require_traces(op)?;
        for tr in &mut self.traces {
            match op {
                "sqr" => {
                    for v in &mut tr.data {
                        *v = (*v) * (*v);
                    }
                }
                "sqrt" => {
                    for v in &mut tr.data {
                        if *v < 0.0 {
                            return Err(EarthSnailError::Numerical(
                                "SAC 'sqrt': cannot take square root of negative sample".into(),
                            ));
                        }
                        *v = math::sqrt(*v);
                    }
                }
                "abs" => {
                    for v in &mut tr.data {
                        *v = v.abs();
                    }
                }
                "log10" => {
                    for v in &mut tr.data {
                        if *v <= 0.0 {
                            return Err(EarthSnailError::Numerical(
                                "SAC 'log10': sample must be strictly positive".into(),
                            ));
                        }
                        *v = math::log10(*v);
                    }
                }
                "exp" => {
                    for v in &mut tr.data {
                        let ev = math::exp(*v);
                        if !ev.is_finite() {
                            return Err(EarthSnailError::Numerical(
                                "SAC 'exp': floating-point overflow".into(),
                            ));
                        }
                        *v = ev;
                    }
                }
                "reverse" => {
                    tr.data.reverse();
                }
                _ => unreachable!(),
            }
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_file_math(&mut self, op: &str, args: &[String]) -> Result<()> {
        self.require_traces(op)?;
        if args.is_empty() {
            return Err(EarthSnailError::InvalidParameter(format!(
                "SAC '{op}' requires at least one waveform file"
            )));
        }
        let mut rhs_traces = Vec::new();
        for pat in args {
            for f in self.expand_file_glob(pat)? {
                let bytes = self.read_bytes_any(&f)?;
                let st = read_waveforms_bytes(&bytes)?;
                rhs_traces.extend(st.traces);
            }
        }
        if rhs_traces.is_empty() {
            return Err(EarthSnailError::EmptyData(format!(
                "SAC '{op}': no traces read from operand files"
            )));
        }
        if rhs_traces.len() != 1 && rhs_traces.len() != self.traces.len() {
            return Err(EarthSnailError::IncompatibleTraces(format!(
                "SAC '{op}': {} traces in memory vs {} operand traces",
                self.traces.len(),
                rhs_traces.len()
            )));
        }

        for (i, tr) in self.traces.iter_mut().enumerate() {
            let rhs = if rhs_traces.len() == 1 {
                &rhs_traces[0]
            } else {
                &rhs_traces[i]
            };
            let n = tr.data.len().min(rhs.data.len());
            for k in 0..n {
                let b = rhs.data[k];
                match op {
                    "addf" => tr.data[k] += b,
                    "subf" => tr.data[k] -= b,
                    "mulf" => tr.data[k] *= b,
                    "divf" => {
                        if b == 0.0 {
                            return Err(EarthSnailError::Numerical(
                                "SAC 'divf': division by zero sample".into(),
                            ));
                        }
                        tr.data[k] /= b;
                    }
                    _ => unreachable!(),
                }
            }
            tr.data.truncate(n);
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_chpf(&mut self, _args: &[String]) -> Result<()> {
        // Classic SAC 'chpf' closes an open HYPO71 pick file (no-op success when none is open).
        Ok(())
    }

    // ========================================================================
    // Filtering & Instrument Deconvolution
    // ========================================================================

    fn cmd_filter(&mut self, kind: &str, args: &[String]) -> Result<()> {
        self.require_traces(kind)?;
        let mut corners: Vec<f64> = Vec::new();
        let mut order: usize = 4;
        let mut passes: usize = 1;
        let mut use_filtfilt = false;
        let mut fam_tag = "bu";
        let mut tranbw = 0.3_f64;
        let mut atten_db = 30.0_f64;

        let mut i = 0usize;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "c" | "co" | "cor" | "corner" | "corners" => {
                    i += 1;
                    while i < args.len() {
                        if let Ok(f) = args[i].parse::<f64>() {
                            corners.push(f);
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    continue;
                }
                "n" | "np" | "npoles" => {
                    order = parse_next_usize(args, i, "filter npoles")?;
                    i += 1;
                }
                "p" | "passes" => {
                    passes = parse_next_usize(args, i, "filter passes")?;
                    if passes != 1 && passes != 2 {
                        return Err(EarthSnailError::InvalidParameter(
                            "SAC filter 'passes' must be 1 or 2".into(),
                        ));
                    }
                    i += 1;
                }
                "filtfilt" => use_filtfilt = true,
                "bu" | "butter" | "butterworth" => fam_tag = "bu",
                "be" | "bessel" => fam_tag = "be",
                "c1" | "cheby1" => fam_tag = "c1",
                "c2" | "cheby2" => fam_tag = "c2",
                "tranbw" | "tb" => {
                    tranbw = parse_next_f64(args, i, "filter tranbw")?;
                    i += 1;
                }
                "attenuation" | "a" | "atten" => {
                    atten_db = parse_next_f64(args, i, "filter attenuation")?;
                    i += 1;
                }
                other => {
                    if let Ok(f) = other.parse::<f64>() {
                        corners.push(f);
                    } else {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "unknown SAC {kind} option '{other}'"
                        )));
                    }
                }
            }
            i += 1;
        }

        let _ = tranbw;
        let family = match fam_tag {
            "bu" => FilterFamily::Butterworth,
            "be" => FilterFamily::Bessel {
                norm: BesselNorm::Phase,
            },
            "c1" => FilterFamily::Chebyshev1 { rp_db: 1.0 },
            "c2" => FilterFamily::Chebyshev2 {
                rs_db: atten_db.max(10.0),
            },
            _ => FilterFamily::Butterworth,
        };

        let band = match kind {
            "lp" => {
                if corners.len() != 1 {
                    return Err(EarthSnailError::InvalidParameter(
                        "SAC 'lowpass' requires 1 corner frequency (e.g. 'lp co 5.0')".into(),
                    ));
                }
                FilterBand::Lowpass(corners[0])
            }
            "hp" => {
                if corners.len() != 1 {
                    return Err(EarthSnailError::InvalidParameter(
                        "SAC 'highpass' requires 1 corner frequency (e.g. 'hp co 0.05')".into(),
                    ));
                }
                FilterBand::Highpass(corners[0])
            }
            "bp" => {
                if corners.len() != 2 {
                    return Err(EarthSnailError::InvalidParameter(
                        "SAC 'bandpass' requires 2 corner frequencies (e.g. 'bp co 0.02 2.0')"
                            .into(),
                    ));
                }
                FilterBand::Bandpass(corners[0].min(corners[1]), corners[0].max(corners[1]))
            }
            "br" => {
                if corners.len() != 2 {
                    return Err(EarthSnailError::InvalidParameter(
                        "SAC 'bandrej' requires 2 corner frequencies (e.g. 'br co 0.12 0.25')"
                            .into(),
                    ));
                }
                FilterBand::Bandstop(corners[0].min(corners[1]), corners[0].max(corners[1]))
            }
            _ => unreachable!(),
        };

        for tr in &mut self.traces {
            let sos = design_iir_sos(order, band, tr.stats.sampling_rate, family)?;
            tr.data = if use_filtfilt {
                sosfiltfilt(&sos, &tr.data)?
            } else if passes == 2 {
                sosfilt_zerophase_obspy(&sos, &tr.data)
            } else {
                sosfilt(&sos, &tr.data)
            };
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_transfer(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("transfer")?;
        #[derive(Clone)]
        enum RespSpec {
            None,
            SacPz(String),
            StationXml(String),
            WoodAnderson,
        }

        let mut from_spec = RespSpec::None;
        let mut to_spec = RespSpec::None;
        let mut out_unit = KinematicUnit::Displacement;
        let mut pre_filt: Option<[f64; 4]> = None;
        let mut water_level = 60.0_f64;

        let mut i = 0usize;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "from" => {
                    let kind = args.get(i + 1).ok_or_else(|| {
                        EarthSnailError::InvalidParameter(
                            "SAC 'transfer from' expects polezero|stationxml|none".into(),
                        )
                    })?;
                    let klc = kind.to_ascii_lowercase();
                    match klc.as_str() {
                        "none" => {
                            from_spec = RespSpec::None;
                            i += 2;
                        }
                        "polezero" | "pz" | "sacpz" => {
                            let mut p_idx = i + 2;
                            if args.get(p_idx).is_some_and(|s| {
                                s.eq_ignore_ascii_case("subtype") || s.eq_ignore_ascii_case("s")
                            }) {
                                p_idx += 1;
                            }
                            let f = args.get(p_idx).ok_or_else(|| {
                                EarthSnailError::InvalidParameter(
                                    "SAC 'transfer from polezero' requires a SACPZ filename".into(),
                                )
                            })?;
                            from_spec = RespSpec::SacPz(f.clone());
                            i = p_idx + 1;
                        }
                        "stationxml" | "evalresp" | "xml" => {
                            let mut p_idx = i + 2;
                            if args.get(p_idx).is_some_and(|s| {
                                s.eq_ignore_ascii_case("subtype")
                                    || s.eq_ignore_ascii_case("fname")
                                    || s.eq_ignore_ascii_case("s")
                            }) {
                                p_idx += 1;
                            }
                            let f = args.get(p_idx).ok_or_else(|| {
                                EarthSnailError::InvalidParameter(
                                    "SAC 'transfer from stationxml' requires a StationXML filename"
                                        .into(),
                                )
                            })?;
                            from_spec = RespSpec::StationXml(f.clone());
                            i = p_idx + 1;
                        }
                        other => {
                            return Err(EarthSnailError::InvalidParameter(format!(
                                "unsupported 'transfer from {other}' type"
                            )));
                        }
                    }
                    continue;
                }
                "to" => {
                    let kind = args.get(i + 1).ok_or_else(|| {
                        EarthSnailError::InvalidParameter(
                            "SAC 'transfer to' expects none|disp|vel|acc|wa|polezero".into(),
                        )
                    })?;
                    let klc = kind.to_ascii_lowercase();
                    match klc.as_str() {
                        "none" | "disp" | "displacement" => {
                            to_spec = RespSpec::None;
                            out_unit = KinematicUnit::Displacement;
                            i += 2;
                        }
                        "vel" | "velocity" => {
                            to_spec = RespSpec::None;
                            out_unit = KinematicUnit::Velocity;
                            i += 2;
                        }
                        "acc" | "acceleration" => {
                            to_spec = RespSpec::None;
                            out_unit = KinematicUnit::Acceleration;
                            i += 2;
                        }
                        "wa" | "woodanderson" => {
                            to_spec = RespSpec::WoodAnderson;
                            out_unit = KinematicUnit::Displacement;
                            i += 2;
                        }
                        "polezero" | "pz" | "sacpz" => {
                            let mut p_idx = i + 2;
                            if args.get(p_idx).is_some_and(|s| {
                                s.eq_ignore_ascii_case("subtype") || s.eq_ignore_ascii_case("s")
                            }) {
                                p_idx += 1;
                            }
                            let f = args.get(p_idx).ok_or_else(|| {
                                EarthSnailError::InvalidParameter(
                                    "SAC 'transfer to polezero' requires a SACPZ filename".into(),
                                )
                            })?;
                            to_spec = RespSpec::SacPz(f.clone());
                            out_unit = KinematicUnit::Displacement;
                            i = p_idx + 1;
                        }
                        other => {
                            return Err(EarthSnailError::InvalidParameter(format!(
                                "unsupported 'transfer to {other}' target"
                            )));
                        }
                    }
                    continue;
                }
                "freqlimits" | "freq" => {
                    if i + 4 >= args.len() {
                        return Err(EarthSnailError::InvalidParameter(
                            "SAC 'transfer freqlimits' requires 4 corner frequencies: f1 f2 f3 f4"
                                .into(),
                        ));
                    }
                    let f1 = parse_next_f64(args, i, "freqlimits f1")?;
                    let f2 = parse_next_f64(args, i + 1, "freqlimits f2")?;
                    let f3 = parse_next_f64(args, i + 2, "freqlimits f3")?;
                    let f4 = parse_next_f64(args, i + 3, "freqlimits f4")?;
                    pre_filt = Some([f1, f2, f3, f4]);
                    i += 5;
                    continue;
                }
                "waterlevel" | "wl" => {
                    water_level = parse_next_f64(args, i, "transfer waterlevel")?;
                    i += 2;
                    continue;
                }
                other => {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "unrecognized SAC 'transfer' keyword '{other}'"
                    )));
                }
            }
        }

        // Pre-load metadata inventories / SACPZ records:
        let from_pz = if let RespSpec::SacPz(ref f) = from_spec {
            let txt = String::from_utf8(self.read_bytes_any(f)?)
                .map_err(|_| EarthSnailError::Format("SACPZ file is not valid UTF-8".into()))?;
            Some(read_sacpz_str(&txt)?.response)
        } else {
            None
        };
        let from_inv = if let RespSpec::StationXml(ref f) = from_spec {
            Some(read_stationxml_bytes(&self.read_bytes_any(f)?)?)
        } else {
            None
        };
        let to_resp: Option<ChannelResponse> = match to_spec {
            RespSpec::None => None,
            RespSpec::WoodAnderson => Some(wood_anderson_response_iaspei()),
            RespSpec::SacPz(ref f) => {
                let txt = String::from_utf8(self.read_bytes_any(f)?)
                    .map_err(|_| EarthSnailError::Format("SACPZ file is not valid UTF-8".into()))?;
                Some(read_sacpz_str(&txt)?.response)
            }
            RespSpec::StationXml(_) => None,
        };

        for tr in &mut self.traces {
            let resolved_from: Option<&ChannelResponse> = if let Some(ref r) = from_pz {
                Some(r)
            } else if let Some(ref inv) = from_inv {
                Some(match_inventory_response(inv, tr)?)
            } else {
                None
            };

            let deconv = sac_transfer(
                &tr.data,
                tr.stats.sampling_rate,
                resolved_from,
                to_resp.as_ref(),
                out_unit,
                pre_filt,
                Some(water_level),
            )?;
            tr.data = deconv;
            tr.stats.calib = 1.0;
            if let Some(ref mut sac) = tr.stats.sac {
                sac.idep = match out_unit {
                    KinematicUnit::Displacement => SacDepType::DisplacementNm as i32,
                    KinematicUnit::Velocity => SacDepType::VelocityNmPerSec as i32,
                    KinematicUnit::Acceleration => SacDepType::AccelerationNmPerSec2 as i32,
                    KinematicUnit::Other => SacDepType::Unknown as i32,
                };
            }
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    // ========================================================================
    // Spectral, Picking, 3C Rotation & Native TauP
    // ========================================================================

    fn cmd_fft(&mut self, _args: &[String]) -> Result<()> {
        self.require_traces("fft")?;
        if self.spectra.len() < self.traces.len() {
            self.spectra.resize(self.traces.len(), None);
        }
        for (idx, tr) in self.traces.iter_mut().enumerate() {
            let npts = tr.npts();
            if npts < 2 {
                return Err(EarthSnailError::InvalidParameter(
                    "SAC 'fft' requires at least 2 samples".into(),
                ));
            }
            let fs = tr.stats.sampling_rate;
            let dt = 1.0 / fs;
            let orig_b = tr
                .stats
                .sac
                .as_ref()
                .and_then(|s| s.get_float("b"))
                .unwrap_or(0.0);
            let bins = rfft(&tr.data, Some(npts))?;
            self.spectra[idx] = Some(SavedSpectrum {
                rfft_bins: bins.clone(),
                nfft: npts,
                orig_npts: npts,
                orig_sampling_rate: fs,
                orig_b,
            });
            let df = fs / (npts as f64);
            // Continuous-time scaled amplitude spectrum |X(f)| * dt:
            tr.data = bins.iter().map(|c| c.abs() * dt).collect();
            tr.stats.sampling_rate = 1.0 / df;
            if let Some(ref mut sac) = tr.stats.sac {
                sac.iftype = SacFileType::SpectralAmpPhase as i32;
                sac.b = 0.0;
                sac.delta = df;
                sac.npts = tr.data.len() as i32;
                sac.e = ((tr.data.len().saturating_sub(1)) as f64) * df;
                sac.update_dep_stats(&tr.data);
            }
        }
        Ok(())
    }

    fn cmd_ifft(&mut self) -> Result<()> {
        self.require_traces("ifft")?;
        for (idx, tr) in self.traces.iter_mut().enumerate() {
            let saved = self
                .spectra
                .get_mut(idx)
                .and_then(Option::take)
                .ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "SAC 'ifft': trace has no stored FFT spectrum (run 'fft' first)".into(),
                    )
                })?;
            let td = irfft(&saved.rfft_bins, saved.nfft)?;
            tr.data = td.into_iter().take(saved.orig_npts).collect();
            tr.stats.sampling_rate = saved.orig_sampling_rate;
            if let Some(ref mut sac) = tr.stats.sac {
                let dt = 1.0 / saved.orig_sampling_rate;
                sac.iftype = SacFileType::TimeSeries as i32;
                sac.b = saved.orig_b;
                sac.delta = dt;
                sac.npts = tr.data.len() as i32;
                sac.e = saved.orig_b + ((tr.data.len().saturating_sub(1)) as f64) * dt;
                sac.update_dep_stats(&tr.data);
            }
        }
        Ok(())
    }

    fn cmd_hilbert(&mut self) -> Result<()> {
        self.require_traces("hilbert")?;
        for tr in &mut self.traces {
            let ana = analytic_signal(&tr.data)?;
            tr.data = ana.into_iter().map(|c| c.im).collect();
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_envelope(&mut self) -> Result<()> {
        self.require_traces("envelope")?;
        for tr in &mut self.traces {
            tr.data = envelope(&tr.data)?;
            refresh_dep_stats(tr);
        }
        Ok(())
    }

    fn cmd_correlate(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("correlate")?;
        let mut master_idx = 0usize;
        let mut norm = XcorrNormalize::Naive;
        let mut max_len_s: Option<f64> = None;

        let mut i = 0;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "master" | "m" => {
                    let m1 = parse_next_usize(args, i, "correlate master")?;
                    if m1 == 0 || m1 > self.traces.len() {
                        return Err(EarthSnailError::InvalidParameter(format!(
                            "correlate master {m1} out of range 1..={}",
                            self.traces.len()
                        )));
                    }
                    master_idx = m1 - 1;
                    i += 1;
                }
                "length" | "l" => {
                    max_len_s = Some(parse_next_f64(args, i, "correlate length")?);
                    i += 1;
                }
                "normalized" | "norm" => norm = XcorrNormalize::Naive,
                "unnormalized" => norm = XcorrNormalize::None,
                _ => {}
            }
            i += 1;
        }

        let master_data = self.traces[master_idx].data.clone();
        for tr in &mut self.traces {
            let dt = tr.delta();
            let max_shift = match max_len_s {
                Some(sec) if sec > 0.0 => ((sec / dt) as usize).max(1),
                _ => tr.npts().min(master_data.len()).saturating_sub(1).max(1),
            };
            let cc = correlate(&master_data, &tr.data, max_shift, true, norm)?;
            let half = ((cc.len().saturating_sub(1)) as f64) * 0.5 * dt;
            tr.data = cc;
            if let Some(ref mut sac) = tr.stats.sac {
                sac.b = -half;
                sac.npts = tr.data.len() as i32;
                sac.e = -half + ((tr.data.len().saturating_sub(1)) as f64) * dt;
                sac.update_dep_stats(&tr.data);
            }
        }
        Ok(())
    }

    fn cmd_apk(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("apk")?;
        let mut method = "morse";
        let mut thr_override: Option<f64> = None;
        let mut nsta_override: Option<usize> = None;
        let mut nlta_override: Option<usize> = None;

        let mut i = 0;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "morse" => method = "morse",
                "baer" => method = "baer",
                "stalta" => method = "stalta",
                "aic" => method = "aic",
                "thr" | "threshold" => {
                    thr_override = Some(parse_next_f64(args, i, "apk thr")?);
                    i += 1;
                }
                "nsta" => {
                    nsta_override = Some(parse_next_usize(args, i, "apk nsta")?);
                    i += 1;
                }
                "nlta" => {
                    nlta_override = Some(parse_next_usize(args, i, "apk nlta")?);
                    i += 1;
                }
                other => {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "unknown SAC 'apk' option '{other}' (expected morse|baer|stalta|aic)"
                    )));
                }
            }
            i += 1;
        }

        for tr in &mut self.traces {
            ensure_sac_header_synced(tr);
            let fs = tr.stats.sampling_rate;
            let dt = tr.delta();
            let n = tr.npts();
            let (pick_idx, pick_label) = match method {
                "morse" => {
                    let nlta_def = ((6.0 * fs) as usize).clamp(16, (n / 3).max(16));
                    let nsta_def = ((1.0 * fs) as usize).clamp(4, nlta_def / 2);
                    let nlta = nlta_override.unwrap_or(nlta_def);
                    let nsta = nsta_override.unwrap_or(nsta_def).min(nlta);
                    if n <= nlta + 4 {
                        (aic_pick(&tr.data), String::from("MORSE"))
                    } else {
                        let cfg = MorsePickConfig {
                            nsta,
                            nlta,
                            thr_on: thr_override.unwrap_or(3.5),
                            prominence_sigma: 4.0,
                            confirm_samples: 4,
                        };
                        let res = morse_pick(&tr.data, &cfg)?;
                        (res.pick_sample, String::from("MORSE"))
                    }
                }
                "baer" => {
                    let samp_int = (math::round(fs) as usize).max(1);
                    let preset = samp_int.min(n.saturating_sub(2)).max(1);
                    let params = BaerPickParams {
                        samp_int,
                        tdownmax: (samp_int / 5).max(2),
                        tupevent: (samp_int / 2).max(4),
                        thr1: thr_override.unwrap_or(7.0),
                        thr2: 12.0,
                        preset_len: preset,
                        p_dur: samp_int,
                    };
                    let res = pk_baer(&tr.data, &params)?;
                    let lbl = if res.first_motion.is_empty() {
                        String::from("BAER")
                    } else {
                        res.first_motion
                    };
                    (res.pick_sample, lbl)
                }
                "stalta" => {
                    let nlta = nlta_override
                        .unwrap_or_else(|| ((10.0 * fs) as usize).clamp(10, (n / 2).max(10)));
                    let nsta =
                        nsta_override.unwrap_or_else(|| ((1.0 * fs) as usize).clamp(2, nlta / 2));
                    let cft = classic_sta_lta(&tr.data, nsta, nlta)?;
                    let thr = thr_override.unwrap_or(3.0);
                    let trigs = trigger_onset(&cft, thr, (0.4 * thr).max(0.8));
                    (trigs.first().map(|pair| pair[0]), String::from("STALTA"))
                }
                "aic" => (aic_pick(&tr.data), String::from("AIC")),
                _ => unreachable!(),
            };

            if let Some(idx) = pick_idx {
                let sac = tr.stats.sac.as_mut().unwrap();
                let b_val = sac.get_float("b").unwrap_or(0.0);
                let a_time = b_val + (idx as f64) * dt;
                sac.set_float("a", a_time)?;
                sac.set_str("ka", &pick_label)?;
                self.output_log.push_str(&format!(
                    " apk [{}]: {} -> A = {:.4} s (sample {idx}, KA = {})\n",
                    method,
                    tr.stats.channel.seed_id(),
                    a_time,
                    pick_label
                ));
            } else {
                self.output_log.push_str(&format!(
                    " apk [{}]: {} -> no arrival triggered\n",
                    method,
                    tr.stats.channel.seed_id()
                ));
            }
        }
        Ok(())
    }

    fn cmd_markptp(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("markptp")?;
        let mut win_s = 5.0_f64;
        let mut i = 0;
        while i < args.len() {
            if args[i].eq_ignore_ascii_case("length") || args[i].eq_ignore_ascii_case("l") {
                win_s = parse_next_f64(args, i, "markptp length")?;
                i += 2;
                continue;
            }
            i += 1;
        }
        for tr in &mut self.traces {
            if tr.is_empty() {
                continue;
            }
            ensure_sac_header_synced(tr);
            let dt = tr.delta();
            let wlen = ((win_s / dt) as usize).clamp(2, tr.npts());
            let mut best_ptp = -1.0_f64;
            let mut best_min_i = 0usize;
            let mut best_max_i = 0usize;
            for start in 0..=(tr.npts() - wlen) {
                let mut mn = f64::INFINITY;
                let mut mx = f64::NEG_INFINITY;
                let mut mn_i = start;
                let mut mx_i = start;
                for (k, &v) in tr.data.iter().enumerate().skip(start).take(wlen) {
                    if v < mn {
                        mn = v;
                        mn_i = k;
                    }
                    if v > mx {
                        mx = v;
                        mx_i = k;
                    }
                }
                if mx - mn > best_ptp {
                    best_ptp = mx - mn;
                    best_min_i = mn_i;
                    best_max_i = mx_i;
                }
            }
            let sac = tr.stats.sac.as_mut().unwrap();
            let b_val = sac.get_float("b").unwrap_or(0.0);
            sac.set_float("user0", best_ptp)?;
            sac.set_float("t0", b_val + (best_min_i as f64) * dt)?;
            sac.set_str("kt0", "PTP_MIN")?;
            sac.set_float("t1", b_val + (best_max_i as f64) * dt)?;
            sac.set_str("kt1", "PTP_MAX")?;
        }
        Ok(())
    }

    fn cmd_rotate(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("rotate")?;
        let target = if args.len() >= 2 && args[0].eq_ignore_ascii_case("to") {
            args[1].to_ascii_lowercase()
        } else if let Some(first) = args.first() {
            first.to_ascii_lowercase()
        } else {
            String::from("gcp")
        };

        if target == "gcp" {
            let reversed = args.iter().any(|a| a.eq_ignore_ascii_case("reversed"));
            return self.rotate_horizontals_to_gcp(reversed);
        }
        if target == "zne" {
            return self.rotate_triplets_to_zne();
        }
        if let Ok(angle_deg) = target.parse::<f64>() {
            // Rotate horizontal pairs by explicit azimuth `angle_deg`:
            return self.rotate_horizontals_by_angle(angle_deg);
        }
        Err(EarthSnailError::InvalidParameter(format!(
            "unsupported SAC 'rotate' target '{target}' (expected 'to gcp', 'to zne', or 'to <deg>')"
        )))
    }

    fn rotate_horizontals_to_gcp(&mut self, reversed: bool) -> Result<()> {
        for tr in &mut self.traces {
            ensure_sac_header_synced(tr);
        }

        // Pair horizontal channels (`cmpinc` near 90 deg, or channel ending in N/E/1/2) sharing
        // `(network, station, location, 2-char band+inst prefix, npts)`:
        let mut used = vec![false; self.traces.len()];
        let mut rotated_pairs = 0usize;

        for i in 0..self.traces.len() {
            if used[i] || !is_horizontal_trace(&self.traces[i]) {
                continue;
            }
            let mut partner: Option<usize> = None;
            for (j, &is_used) in used.iter().enumerate().take(self.traces.len()).skip(i + 1) {
                if !is_used
                    && is_horizontal_trace(&self.traces[j])
                    && same_station_instrument(&self.traces[i], &self.traces[j])
                {
                    partner = Some(j);
                    break;
                }
            }
            let Some(j) = partner else {
                continue;
            };
            used[i] = true;
            used[j] = true;

            let az_i = infer_horizontal_cmpaz(&self.traces[i])?;
            let az_j = infer_horizontal_cmpaz(&self.traces[j])?;

            // Check orthogonality (`sin(az_j - az_i)` != 0):
            let det = math::sin((az_j - az_i).to_radians());
            if det.abs() < 1e-3 {
                return Err(EarthSnailError::SingularMatrix(format!(
                    "SAC 'rot to gcp': horizontal components '{}' (cmpaz={az_i}) and '{}' (cmpaz={az_j}) are collinear",
                    self.traces[i].stats.channel, self.traces[j].stats.channel
                )));
            }

            // Reconstruct exact (North, East) from general horizontal basis (az_i, az_j):
            // d_i = N cos(az_i) + E sin(az_i)
            // d_j = N cos(az_j) + E sin(az_j)
            let (si, ci) = math::sin_cos(az_i.to_radians());
            let (sj, cj) = math::sin_cos(az_j.to_radians());
            let inv_det = 1.0 / (ci * sj - si * cj);
            let len = self.traces[i].npts();
            let mut north = Vec::with_capacity(len);
            let mut east = Vec::with_capacity(len);
            for k in 0..len {
                let di = self.traces[i].data[k];
                let dj = self.traces[j].data[k];
                north.push((sj * di - si * dj) * inv_det);
                east.push((-cj * di + ci * dj) * inv_det);
            }

            // Obtain back-azimuth BAZ (recomputing WGS84 geodesics if needed):
            let back_az = {
                let sac_i = self.traces[i].stats.sac.as_mut().unwrap();
                if sac_i.get_float("baz").is_none() {
                    sac_i.force_update_dist_az_baz_gcarc();
                }
                sac_i.get_float("baz").ok_or_else(|| {
                    EarthSnailError::Header(format!(
                        "SAC 'rot to gcp': BAZ (or STLA/STLO/EVLA/EVLO) is undefined on '{}'",
                        self.traces[i].stats.channel
                    ))
                })?
            };

            let (mut radial, mut transverse) = rotate_ne_rt(&north, &east, back_az)?;
            if reversed {
                for v in &mut radial {
                    *v = -*v;
                }
                for v in &mut transverse {
                    *v = -*v;
                }
            }

            let r_az = math::rem_euclid(back_az + if reversed { 0.0 } else { 180.0 }, 360.0);
            let t_az = math::rem_euclid(back_az + if reversed { 90.0 } else { 270.0 }, 360.0);

            self.traces[i].data = radial;
            set_rotated_component(&mut self.traces[i], 'R', r_az, 90.0);

            self.traces[j].data = transverse;
            set_rotated_component(&mut self.traces[j], 'T', t_az, 90.0);

            rotated_pairs += 1;
        }

        if rotated_pairs == 0 {
            return Err(EarthSnailError::IncompatibleTraces(
                "SAC 'rot to gcp': found no matching horizontal component pairs (N/E or 1/2) in memory".into(),
            ));
        }
        Ok(())
    }

    fn rotate_horizontals_by_angle(&mut self, angle_deg: f64) -> Result<()> {
        let baz_equiv = math::rem_euclid(angle_deg + 180.0, 360.0);
        for tr in &mut self.traces {
            ensure_sac_header_synced(tr);
            if let Some(ref mut sac) = tr.stats.sac {
                if sac.get_float("baz").is_none() {
                    let _ = sac.set_float("baz", baz_equiv);
                }
            }
        }
        self.rotate_horizontals_to_gcp(false)
    }

    fn rotate_triplets_to_zne(&mut self) -> Result<()> {
        if self.traces.len() < 3 {
            return Err(EarthSnailError::IncompatibleTraces(
                "SAC 'rot to zne' requires at least 3 traces in memory".into(),
            ));
        }
        for tr in &mut self.traces {
            ensure_sac_header_synced(tr);
        }

        let mut chunk_start = 0usize;
        while chunk_start + 3 <= self.traces.len() {
            let n = self.traces[chunk_start].npts();
            if self.traces[chunk_start + 1].npts() != n || self.traces[chunk_start + 2].npts() != n
            {
                return Err(EarthSnailError::IncompatibleTraces(
                    "SAC 'rot to zne': 3-component triplet traces must have equal NPTS".into(),
                ));
            }
            let c0 = self.traces[chunk_start].stats.channel.component();
            let c1 = self.traces[chunk_start + 1].stats.channel.component();
            let c2 = self.traces[chunk_start + 2].stats.channel.component();
            let (z, north, east) = if matches!((c0, c1, c2), (Some('U'), Some('V'), Some('W'))) {
                rotate_uvw_zne(
                    &self.traces[chunk_start].data,
                    &self.traces[chunk_start + 1].data,
                    &self.traces[chunk_start + 2].data,
                )?
            } else {
                let (az1, dip1) = extract_seed_az_dip(&self.traces[chunk_start])?;
                let (az2, dip2) = extract_seed_az_dip(&self.traces[chunk_start + 1])?;
                let (az3, dip3) = extract_seed_az_dip(&self.traces[chunk_start + 2])?;
                rotate_to_zne(
                    &self.traces[chunk_start].data,
                    az1,
                    dip1,
                    &self.traces[chunk_start + 1].data,
                    az2,
                    dip2,
                    &self.traces[chunk_start + 2].data,
                    az3,
                    dip3,
                    false,
                )?
            };

            self.traces[chunk_start].data = z;
            set_rotated_component(&mut self.traces[chunk_start], 'Z', 0.0, 0.0);
            self.traces[chunk_start + 1].data = north;
            set_rotated_component(&mut self.traces[chunk_start + 1], 'N', 0.0, 90.0);
            self.traces[chunk_start + 2].data = east;
            set_rotated_component(&mut self.traces[chunk_start + 2], 'E', 90.0, 90.0);

            chunk_start += 3;
        }
        Ok(())
    }

    fn cmd_taup(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("taup")?;
        let (model_name, phases, force_evdp_km) = parse_taup_cmd_args(args)?;

        // Ensure engine is compiled & cached:
        let _ = self.get_or_compile_tau_model(&model_name)?;
        let engine = self.tau_engines.get_mut(&model_name).unwrap();

        for tr in &mut self.traces {
            ensure_sac_header_synced(tr);
            let sac = tr.stats.sac.as_mut().unwrap();
            let stamped = stamp_taup_into_sac_header(sac, engine, &phases, force_evdp_km)?;
            let summary: Vec<String> = stamped
                .iter()
                .map(|(slot, name, t_val)| format!("{slot}={name}({t_val:.2}s)"))
                .collect();
            self.output_log.push_str(&format!(
                " taup [{}]: {} -> {}\n",
                model_name,
                tr.stats.channel.seed_id(),
                if summary.is_empty() {
                    String::from("no arrivals at epicentral distance")
                } else {
                    summary.join(", ")
                }
            ));
        }
        Ok(())
    }

    // ========================================================================
    // Plotting Commands
    // ========================================================================

    fn cmd_plot_waveforms(&mut self, args: &[String], layout: WaveformLayout) -> Result<()> {
        self.require_traces("plot")?;
        let mut out_path: Option<String> = None;
        let mut opts = WaveformPlotOptions {
            layout,
            ..WaveformPlotOptions::default()
        };
        for tok in args {
            if tok.eq_ignore_ascii_case("utc") {
                opts.time_axis = crate::plot::TimeAxisMode::UtcClock;
            } else if out_path.is_none() {
                out_path = Some(tok.clone());
            }
        }
        let svg = plot_waveforms_svg(&self.to_stream(), &opts);
        if let Some(ref p) = out_path {
            self.write_bytes_any(p, svg.as_bytes())?;
        }
        self.last_svg = Some(svg);
        Ok(())
    }

    fn cmd_plot_record_section(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("plotrecordsection")?;
        let mut out_path: Option<String> = None;
        let mut model_name = String::from("iasp91");
        let mut phases: Vec<String> = vec![String::from("P"), String::from("S")];
        let mut opts = RecordSectionOptions::default();

        let mut i = 0;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "km" => opts.distance_unit = DistanceUnit::Kilometers,
                "deg" => opts.distance_unit = DistanceUnit::Degrees,
                "reduce" | "vred" => {
                    opts.reduction_velocity_km_s = Some(parse_next_f64(args, i, "prs reduce")?);
                    i += 1;
                }
                "model" => {
                    if let Some(m) = args.get(i + 1) {
                        model_name = m.clone();
                        i += 1;
                    }
                }
                "phases" => {
                    phases.clear();
                    i += 1;
                    while i < args.len() && !args[i].ends_with(".svg") {
                        for p in args[i].split(',').filter(|s| !s.is_empty()) {
                            phases.push(String::from(p));
                        }
                        i += 1;
                    }
                    continue;
                }
                _ => {
                    if out_path.is_none() {
                        out_path = Some(args[i].clone());
                    }
                }
            }
            i += 1;
        }

        let evdp = self
            .traces
            .first()
            .and_then(|t| t.stats.sac.as_ref())
            .and_then(|s| s.get_float("evdp"))
            .map(|d| if d > 1000.0 { d * 0.001 } else { d })
            .unwrap_or(10.0)
            .max(0.0);

        let curves = if cfg!(miri) {
            Vec::new()
        } else {
            let eng = self.get_or_compile_tau_model(&model_name)?;
            eng.base_model()
                .get_travel_time_curves(evdp, &phases, 64)
                .unwrap_or_default()
        };

        let svg = plot_record_section_svg(&self.to_stream(), &curves, &opts);
        if let Some(ref p) = out_path {
            self.write_bytes_any(p, svg.as_bytes())?;
        }
        self.last_svg = Some(svg);
        Ok(())
    }

    fn cmd_plot_spectrogram(&mut self, args: &[String]) -> Result<()> {
        self.require_traces("spectrogram")?;
        let mut out_path: Option<String> = None;
        let mut opts = SpectrogramOptions::default();
        for tok in args {
            let lc = tok.to_ascii_lowercase();
            match lc.as_str() {
                "cwt" => opts.method = SpectrogramMethod::MorletCwt { omega0: 6.0 },
                "stft" => opts.method = SpectrogramMethod::Stft { win_samples: 0 },
                "linear" => opts.freq_scale = FrequencyScale::Linear,
                "log" => opts.freq_scale = FrequencyScale::Log10,
                _ => {
                    if out_path.is_none() {
                        out_path = Some(tok.clone());
                    }
                }
            }
        }
        let svg = plot_spectrogram_svg(&self.traces[0], &opts);
        if let Some(ref p) = out_path {
            self.write_bytes_any(p, svg.as_bytes())?;
        }
        self.last_svg = Some(svg);
        Ok(())
    }

    fn cmd_plot_beachball(&mut self, args: &[String]) -> Result<()> {
        let mut out_path: Option<String> = None;
        let mut nums: Vec<f64> = Vec::new();
        for tok in args {
            if let Ok(v) = tok.parse::<f64>() {
                nums.push(v);
            } else if out_path.is_none() {
                out_path = Some(tok.clone());
            }
        }
        let fm_in = if nums.len() == 3 {
            FocalMechanismInput::StrikeDipRake(NodalPlane {
                strike_deg: nums[0],
                dip_deg: nums[1],
                rake_deg: nums[2],
            })
        } else if nums.len() == 6 {
            FocalMechanismInput::MomentTensor(MomentTensor6::new(
                nums[0], nums[1], nums[2], nums[3], nums[4], nums[5],
            ))
        } else {
            return Err(EarthSnailError::InvalidParameter(
                "SAC 'beachball' expects [<out.svg>] <strike> <dip> <rake> OR 6 moment-tensor components".into(),
            ));
        };
        let svg = plot_beachball_svg(fm_in, &BeachballPlotOptions::default());
        if let Some(ref p) = out_path {
            self.write_bytes_any(p, svg.as_bytes())?;
        }
        self.last_svg = Some(svg);
        Ok(())
    }

    fn cmd_plot_taup(&mut self, args: &[String]) -> Result<()> {
        let mut out_path: Option<String> = None;
        let mut model_name = String::from("iasp91");
        let mut evdp = 29.0_f64;
        let mut deg = 74.0_f64;
        let mut phases = vec![
            String::from("P"),
            String::from("S"),
            String::from("PP"),
            String::from("PcP"),
            String::from("SKS"),
        ];
        let mut i = 0;
        while i < args.len() {
            let lc = args[i].to_ascii_lowercase();
            match lc.as_str() {
                "model" => {
                    if let Some(m) = args.get(i + 1) {
                        model_name = m.clone();
                        i += 1;
                    }
                }
                "evdp" | "h" => {
                    evdp = parse_next_f64(args, i, "plottaup evdp")?;
                    i += 1;
                }
                "deg" | "d" => {
                    deg = parse_next_f64(args, i, "plottaup deg")?;
                    i += 1;
                }
                "phases" => {
                    phases.clear();
                    i += 1;
                    while i < args.len() && !args[i].ends_with(".svg") {
                        for p in args[i].split(',').filter(|s| !s.is_empty()) {
                            phases.push(String::from(p));
                        }
                        i += 1;
                    }
                    continue;
                }
                _ => {
                    if out_path.is_none() {
                        out_path = Some(args[i].clone());
                    }
                }
            }
            i += 1;
        }
        let paths = if cfg!(miri) {
            Vec::new()
        } else {
            let eng = self.get_or_compile_tau_model(&model_name)?;
            eng.base_model().get_ray_paths(evdp, deg, &phases)?
        };
        let svg = plot_taup_paths_svg(&paths, &model_name, &TaupPathsPlotOptions::default());
        if let Some(ref p) = out_path {
            self.write_bytes_any(p, svg.as_bytes())?;
        }
        self.last_svg = Some(svg);
        Ok(())
    }
}

// ============================================================================
// Public TauP-to-SAC Header Stamping Helper (shared by `SAC> taup` & `earthsnail taup setsac`)
// ============================================================================

/// Computes theoretical `TauP` travel times from `sac` header geometry (`EVDP`, `GCARC` / `DIST` /
/// `EVLA,EVLO,STLA,STLO`, and `O`) and stamps up to 10 arrivals into `T0..T9` and `KT0..KT9`.
/// Returns `Vec<(slot_name, phase_name, stamped_marker_s)>`.
pub fn stamp_taup_into_sac_header(
    sac: &mut SacHeader,
    engine: &mut TauEngine,
    phases: &[String],
    force_evdp_km: bool,
) -> Result<Vec<(String, String, f64)>> {
    // Ensure geodesic gcarc/dist/az/baz are populated when station & event lat/lon exist:
    if sac.get_float("gcarc").is_none() {
        sac.force_update_dist_az_baz_gcarc();
    }
    let gcarc_deg = if let Some(deg) = sac.get_float("gcarc") {
        deg
    } else if let Some(km) = sac.get_float("dist") {
        kilometer2degrees(km)
    } else {
        return Err(EarthSnailError::Header(
            "TauP header stamping requires GCARC, DIST, or (EVLA, EVLO, STLA, STLO) in SAC header"
                .into(),
        ));
    };

    let raw_evdp = sac.get_float("evdp").unwrap_or(0.0).max(0.0);
    // Classic SAC compatibility: legacy RDSEED files sometimes stored EVDP in meters (> 1000.0):
    let evdp_km = if !force_evdp_km && raw_evdp > 1_000.0 {
        raw_evdp * 0.001
    } else {
        raw_evdp
    };

    let o_offset = sac.get_float("o").unwrap_or(0.0);
    let arrivals = engine
        .base_model()
        .get_travel_times(evdp_km, gcarc_deg, phases)?;

    // Keep earliest arrival per distinct phase name (so triplications don't crowd out later phases),
    // ordered by arrival time:
    let mut chosen = Vec::new();
    for arr in arrivals {
        if !chosen.iter().any(|(nm, _): &(String, f64)| nm == &arr.name) {
            chosen.push((arr.name, o_offset + arr.time_s));
            if chosen.len() == 10 {
                break;
            }
        }
    }

    // Clear T0..T9 / KT0..KT9 before stamping fresh arrivals:
    for slot in 0..10 {
        let _ = sac.clear_float(&format!("t{slot}"));
        let _ = sac.clear_str(&format!("kt{slot}"));
    }

    let mut stamped = Vec::with_capacity(chosen.len());
    for (slot, (ph_name, t_marker)) in chosen.into_iter().enumerate() {
        let tf = format!("t{slot}");
        let ktf = format!("kt{slot}");
        sac.set_float(&tf, t_marker)?;
        sac.set_str(&ktf, &ph_name)?;
        stamped.push((tf.to_ascii_uppercase(), ph_name, t_marker));
    }
    Ok(stamped)
}

// ============================================================================
// Parsing, Cut, Rotation, Inventory Matching & Morse Glitch Helpers
// ============================================================================

fn strip_inline_hash_comment(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_ws = true;

    for ch in line.chars() {
        if ch == '\'' && !in_double {
            in_single = !in_single;
            out.push(ch);
            prev_ws = false;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            out.push(ch);
            prev_ws = false;
            continue;
        }
        if ch == '#' && !in_single && !in_double && prev_ws {
            break;
        }
        prev_ws = ch.is_whitespace() || ch == ';';
        out.push(ch);
    }
    out
}

fn split_semicolon_statements(line: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut start = 0usize;

    for (byte_idx, ch) in line.char_indices() {
        if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == ';' && !in_single && !in_double {
            parts.push(&line[start..byte_idx]);
            start = byte_idx + 1;
        }
    }
    if start <= line.len() {
        parts.push(&line[start..]);
    }
    parts
}

fn tokenize_sac_command(stmt: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;

    for ch in stmt.chars() {
        if ch == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
            continue;
        }
        if ch.is_whitespace() && !in_single && !in_double {
            if !cur.is_empty() {
                tokens.push(core::mem::take(&mut cur));
            }
        } else {
            cur.push(ch);
        }
    }
    if in_single || in_double {
        return Err(EarthSnailError::Format(format!(
            "SAC syntax error: unterminated quote in '{stmt}'"
        )));
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    Ok(tokens)
}

fn find_matching_enddo(stmts: &[String], do_idx: usize) -> Result<usize> {
    let mut depth = 0usize;
    for (k, st) in stmts.iter().enumerate().skip(do_idx) {
        let first = st
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        if first == "do" {
            depth += 1;
        } else if first == "enddo" {
            depth -= 1;
            if depth == 0 {
                return Ok(k);
            }
        }
    }
    Err(EarthSnailError::Format(
        "SAC syntax error: 'do' block is missing matching 'enddo'".into(),
    ))
}

fn parse_do_header(header: &str) -> Result<(String, Vec<String>)> {
    let cleaned = header.replace(',', " ");
    let toks = tokenize_sac_command(&cleaned)?;
    // Supported forms:
    // 1. `do var = start [to] end [by step]`
    // 2. `do var list item1 item2 ...`
    // 3. `do var = item1 item2 ...` (when non-numeric list)
    if toks.len() < 3 {
        return Err(EarthSnailError::Format(format!(
            "invalid SAC 'do' loop syntax: '{header}'"
        )));
    }
    let var_name = toks[1].clone();
    if !var_name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(EarthSnailError::InvalidParameter(format!(
            "invalid loop variable name '{var_name}'"
        )));
    }

    if toks[2].eq_ignore_ascii_case("list") {
        return Ok((var_name, toks[3..].to_vec()));
    }

    let rest = if toks[2] == "=" {
        &toks[3..]
    } else {
        &toks[2..]
    };
    if rest.is_empty() {
        return Err(EarthSnailError::Format(format!(
            "empty SAC 'do' loop range in '{header}'"
        )));
    }

    // Attempt numeric range `start [to] end [by step]`:
    if let Ok(start_v) = rest[0].parse::<f64>() {
        let mut p = 1usize;
        if rest.get(p).is_some_and(|s| s.eq_ignore_ascii_case("to")) {
            p += 1;
        }
        if let Some(Ok(end_v)) = rest.get(p).map(|s| s.parse::<f64>()) {
            p += 1;
            let mut step_v = if end_v >= start_v { 1.0 } else { -1.0 };
            if rest.get(p).is_some_and(|s| s.eq_ignore_ascii_case("by")) {
                step_v = rest
                    .get(p + 1)
                    .ok_or_else(|| {
                        EarthSnailError::InvalidParameter("missing step value after 'by'".into())
                    })?
                    .parse::<f64>()
                    .map_err(|_| {
                        EarthSnailError::InvalidParameter("invalid numeric step in 'do'".into())
                    })?;
                p += 2;
            }
            if p == rest.len() {
                if !start_v.is_finite()
                    || !end_v.is_finite()
                    || !step_v.is_finite()
                    || step_v == 0.0
                {
                    return Err(EarthSnailError::InvalidParameter(
                        "SAC 'do' loop bounds must be finite and step != 0".into(),
                    ));
                }
                let is_int = (start_v - math::round(start_v)).abs() == 0.0
                    && (end_v - math::round(end_v)).abs() == 0.0
                    && (step_v - math::round(step_v)).abs() == 0.0;
                let mut vals = Vec::new();
                let mut cur = start_v;
                let eps = 1e-11 * step_v.abs();
                while (step_v > 0.0 && cur <= end_v + eps) || (step_v < 0.0 && cur >= end_v - eps) {
                    if vals.len() > MAX_LOOP_ITERATIONS {
                        break;
                    }
                    if is_int {
                        vals.push(format!("{}", math::round(cur) as i64));
                    } else {
                        vals.push(format!("{cur}"));
                    }
                    cur += step_v;
                }
                return Ok((var_name, vals));
            }
        }
    }

    Ok((var_name, rest.to_vec()))
}

fn ensure_sac_header_synced(tr: &mut Trace) {
    if tr.stats.sac.is_none() {
        tr.stats.sync_to_sac_header(&tr.data);
    } else {
        refresh_dep_stats(tr);
    }
}

fn refresh_dep_stats(tr: &mut Trace) {
    let delta_s = tr.delta();
    if let Some(ref mut sac) = tr.stats.sac {
        sac.npts = tr.data.len() as i32;
        sac.delta = delta_s;
        let b_val = sac.get_float("b").unwrap_or(0.0);
        sac.e = if tr.data.is_empty() {
            b_val
        } else {
            b_val + ((tr.data.len() - 1) as f64) * sac.delta
        };
        sac.update_dep_stats(&tr.data);
    }
}

fn restore_reftime_and_b(tr: &mut Trace, old_reftime: Option<UtcTimestamp>, old_b: Option<f64>) {
    let delta_s = tr.delta();
    if let (Some(ref mut sac), Some(rt), Some(b0)) = (&mut tr.stats.sac, old_reftime, old_b) {
        sac.set_reference_time(rt);
        sac.b = b0;
        sac.delta = delta_s;
        sac.npts = tr.data.len() as i32;
        sac.e = b0 + ((tr.data.len().saturating_sub(1)) as f64) * sac.delta;
        sac.update_dep_stats(&tr.data);
    }
}

fn apply_single_chnhdr(tr: &mut Trace, field_lc: &str, val_str: &str) -> Result<()> {
    let sac = tr.stats.sac.as_mut().unwrap();
    let is_undef = val_str.eq_ignore_ascii_case("undef") || val_str == "-12345";

    if SAC_FLOAT_NAMES.contains(&field_lc) || matches!(field_lc, "fmt" | "sb" | "sdelta") {
        if is_undef {
            sac.clear_float(field_lc)?;
        } else {
            let v = val_str.parse::<f64>().map_err(|_| {
                EarthSnailError::Header(format!(
                    "expected float for SAC header '{field_lc}', got '{val_str}'"
                ))
            })?;
            sac.set_float(field_lc, v)?;
            if field_lc == "delta" && v > 0.0 {
                tr.stats.sampling_rate = 1.0 / v;
                refresh_dep_stats(tr);
            } else if field_lc == "b" {
                if let Some(rt) = sac.reference_time() {
                    if let Ok(st) = rt.add_secs_f64(v) {
                        tr.stats.starttime = st;
                    }
                }
                refresh_dep_stats(tr);
            }
        }
        return Ok(());
    }

    if SAC_INT_NAMES.contains(&field_lc) || matches!(field_lc, "nsnpts" | "ibody") {
        if is_undef {
            sac.clear_int(field_lc)?;
            return Ok(());
        }
        if matches!(
            field_lc,
            "leven" | "lpspol" | "lovrok" | "lcalda" | "unused27"
        ) {
            let b = match val_str.to_ascii_lowercase().as_str() {
                "true" | "t" | "1" | "on" | "yes" => true,
                "false" | "f" | "0" | "off" | "no" => false,
                _ => {
                    return Err(EarthSnailError::Header(format!(
                        "expected boolean (true/false/1/0) for '{field_lc}', got '{val_str}'"
                    )));
                }
            };
            sac.set_bool(field_lc, b)?;
            return Ok(());
        }
        let code = parse_sac_enum_or_int(field_lc, val_str)?;
        sac.set_int(field_lc, code)?;
        if matches!(
            field_lc,
            "nzyear" | "nzjday" | "nzhour" | "nzmin" | "nzsec" | "nzmsec"
        ) {
            if let Some(st) = sac.starttime() {
                tr.stats.starttime = st;
            }
        }
        return Ok(());
    }

    if SAC_STR_NAMES.contains(&field_lc) || field_lc == "kevnm1" {
        if is_undef {
            if field_lc != "kevnm1" {
                sac.clear_str(field_lc)?;
            }
        } else {
            sac.set_str(field_lc, val_str)?;
        }
        tr.stats.channel = ChannelId::new(
            sac.get_str("knetwk").unwrap_or(""),
            sac.get_str("kstnm").unwrap_or(""),
            sac.get_str("khole").unwrap_or(""),
            sac.get_str("kcmpnm").unwrap_or(""),
        );
        return Ok(());
    }

    let _ = is_sac_int_undef(-12345);
    let _ = is_sac_str_undef("-12345");
    Err(EarthSnailError::Header(format!(
        "unknown SAC header field '{field_lc}'"
    )))
}

fn parse_sac_enum_or_int(field_lc: &str, val_str: &str) -> Result<i32> {
    if let Ok(v) = val_str.parse::<i32>() {
        return Ok(v);
    }
    let vlc = val_str.to_ascii_lowercase();
    match (field_lc, vlc.as_str()) {
        ("idep", "iunkn" | "unknown") => Ok(SacDepType::Unknown as i32),
        ("idep", "idisp" | "disp") => Ok(SacDepType::DisplacementNm as i32),
        ("idep", "ivel" | "vel") => Ok(SacDepType::VelocityNmPerSec as i32),
        ("idep", "iacc" | "acc") => Ok(SacDepType::AccelerationNmPerSec2 as i32),
        ("idep", "ivolts" | "volts") => Ok(SacDepType::Volts as i32),
        ("iftype", "itime") => Ok(SacFileType::TimeSeries as i32),
        ("iftype", "irlim") => Ok(SacFileType::SpectralRealImag as i32),
        ("iftype", "iamph") => Ok(SacFileType::SpectralAmpPhase as i32),
        ("iftype", "ixy") => Ok(SacFileType::GeneralXY as i32),
        ("iztype", "iunkn") => Ok(5),
        ("iztype", "ib") => Ok(9),
        ("iztype", "iday") => Ok(10),
        ("iztype", "io") => Ok(11),
        ("iztype", "ia") => Ok(12),
        _ => Err(EarthSnailError::Header(format!(
            "invalid integer or enum '{val_str}' for SAC header '{field_lc}'"
        ))),
    }
}

fn parse_cut_spec(args: &[String]) -> Result<CutSpec> {
    let parse_one = |pos: usize| -> Result<((CutAnchor, f64), usize)> {
        let tok = args.get(pos).ok_or_else(|| {
            EarthSnailError::InvalidParameter("incomplete SAC 'cut' specification".into())
        })?;
        if let Ok(num) = tok.parse::<f64>() {
            return Ok(((CutAnchor::Zero, num), pos + 1));
        }
        let anchor = match tok.to_ascii_lowercase().as_str() {
            "b" => CutAnchor::Begin,
            "e" => CutAnchor::End,
            "o" => CutAnchor::Origin,
            "a" => CutAnchor::Arrival,
            "f" => CutAnchor::Finale,
            "z" => CutAnchor::Zero,
            "t0" => CutAnchor::Tn(0),
            "t1" => CutAnchor::Tn(1),
            "t2" => CutAnchor::Tn(2),
            "t3" => CutAnchor::Tn(3),
            "t4" => CutAnchor::Tn(4),
            "t5" => CutAnchor::Tn(5),
            "t6" => CutAnchor::Tn(6),
            "t7" => CutAnchor::Tn(7),
            "t8" => CutAnchor::Tn(8),
            "t9" => CutAnchor::Tn(9),
            other => {
                return Err(EarthSnailError::InvalidParameter(format!(
                    "unknown SAC 'cut' reference '{other}' (expected b, e, o, a, f, z, t0..t9, or seconds)"
                )));
            }
        };
        if let Some(Ok(off)) = args.get(pos + 1).map(|s| s.parse::<f64>()) {
            Ok(((anchor, off), pos + 2))
        } else {
            Ok(((anchor, 0.0), pos + 1))
        }
    };

    let ((a1, o1), next_pos) = parse_one(0)?;
    let ((a2, o2), end_pos) = parse_one(next_pos)?;
    if end_pos < args.len() {
        return Err(EarthSnailError::InvalidParameter(format!(
            "unexpected trailing token '{}' in SAC 'cut'",
            args[end_pos]
        )));
    }
    if a1 == a2 && o2 < o1 {
        return Err(EarthSnailError::InvalidParameter(format!(
            "SAC 'cut': end offset ({o2}) precedes start offset ({o1}) on {a1:?}"
        )));
    }
    Ok(CutSpec {
        start_anchor: a1,
        start_offset_s: o1,
        end_anchor: a2,
        end_offset_s: o2,
    })
}

fn eval_cut_anchor(sac: &SacHeader, anchor: CutAnchor) -> Result<f64> {
    let v = match anchor {
        CutAnchor::Zero => Some(0.0),
        CutAnchor::Begin => sac.get_float("b"),
        CutAnchor::End => sac.get_float("e"),
        CutAnchor::Origin => sac.get_float("o"),
        CutAnchor::Arrival => sac.get_float("a"),
        CutAnchor::Finale => sac.get_float("f"),
        CutAnchor::Tn(n) => sac.get_float(&format!("t{n}")),
    };
    v.ok_or_else(|| {
        EarthSnailError::Header(format!(
            "SAC 'cut': requested reference marker {anchor:?} is undefined in trace header"
        ))
    })
}

fn apply_sac_cut_to_trace(tr: &Trace, spec: CutSpec) -> Result<Trace> {
    if tr.is_empty() {
        return Ok(tr.clone());
    }
    let mut work = tr.clone();
    ensure_sac_header_synced(&mut work);
    let sac = work.stats.sac.as_ref().unwrap();
    let b_orig = sac.get_float("b").unwrap_or(0.0);
    let dt = work.delta();
    let t1_sac = eval_cut_anchor(sac, spec.start_anchor)? + spec.start_offset_s;
    let t2_sac = eval_cut_anchor(sac, spec.end_anchor)? + spec.end_offset_s;
    if t2_sac < t1_sac {
        return Err(EarthSnailError::InvalidParameter(format!(
            "SAC 'cut': end time ({t2_sac:.3}s) precedes start time ({t1_sac:.3}s)"
        )));
    }

    let n = work.npts() as isize;
    let i0_raw = math::round((t1_sac - b_orig) / dt) as isize;
    let i1_raw = math::round((t2_sac - b_orig) / dt) as isize;
    if i1_raw < 0 || i0_raw >= n {
        return Err(EarthSnailError::EmptyData(
            "SAC 'cut' window falls completely outside trace time span".into(),
        ));
    }
    let i0 = i0_raw.clamp(0, n - 1) as usize;
    let i1 = i1_raw.clamp(0, n - 1) as usize;
    let sub_data = work.data[i0..=i1].to_vec();
    let new_start = work.sample_time(i0)?;
    let new_b = b_orig + (i0 as f64) * dt;

    let mut new_stats = work.stats.clone();
    new_stats.starttime = new_start;
    if let Some(ref mut new_sac) = new_stats.sac {
        // Preserve original SAC reference_time + O/A/T0..T9 markers, updating B, E, NPTS, DEPMIN/MAX/MEN!
        new_sac.b = new_b;
        new_sac.npts = sub_data.len() as i32;
        new_sac.e = new_b + ((sub_data.len().saturating_sub(1)) as f64) * dt;
        new_sac.update_dep_stats(&sub_data);
    }
    Ok(Trace::new(new_stats, sub_data))
}

fn match_inventory_response<'a>(
    inv: &'a earthsnail_core::StationInventory,
    tr: &Trace,
) -> Result<&'a ChannelResponse> {
    if let Some(ch_meta) = inv
        .find_channel(&tr.stats.channel, Some(tr.stats.starttime))
        .or_else(|| inv.find_channel(&tr.stats.channel, None))
    {
        if let Some(ref resp) = ch_meta.response {
            return Ok(resp);
        }
    }
    // Fallback: match (station, channel) ignoring blank vs "00" location mismatches:
    if let Some(ch_meta) = inv.channels().find(|c| {
        c.channel
            .station
            .eq_ignore_ascii_case(&tr.stats.channel.station)
            && c.channel
                .channel
                .eq_ignore_ascii_case(&tr.stats.channel.channel)
            && c.response.is_some()
    }) {
        return Ok(ch_meta.response.as_ref().unwrap());
    }
    // Single-channel StationXML convenience fallback:
    let all_with_resp: Vec<&ChannelResponse> =
        inv.channels().filter_map(|c| c.response.as_ref()).collect();
    if all_with_resp.len() == 1 {
        return Ok(all_with_resp[0]);
    }
    Err(EarthSnailError::ResponseError(format!(
        "no matching channel response found in StationXML for '{}'",
        tr.stats.channel.seed_id()
    )))
}

fn is_horizontal_trace(tr: &Trace) -> bool {
    if let Some(ref sac) = tr.stats.sac {
        if let Some(inc) = sac.get_float("cmpinc") {
            return (inc - 90.0).abs() <= 10.0;
        }
    }
    matches!(
        tr.stats.channel.component().map(|c| c.to_ascii_uppercase()),
        Some('N' | 'E' | '1' | '2' | 'R' | 'T')
    )
}

fn same_station_instrument(a: &Trace, b: &Trace) -> bool {
    if a.npts() != b.npts()
        || (a.stats.sampling_rate - b.stats.sampling_rate).abs() > 1e-4 * a.stats.sampling_rate
    {
        return false;
    }
    let prefix = |ch: &str| -> String {
        ch.chars()
            .take(ch.chars().count().saturating_sub(1))
            .collect::<String>()
            .to_ascii_uppercase()
    };
    a.stats
        .channel
        .network
        .eq_ignore_ascii_case(&b.stats.channel.network)
        && a.stats
            .channel
            .station
            .eq_ignore_ascii_case(&b.stats.channel.station)
        && a.stats
            .channel
            .location
            .eq_ignore_ascii_case(&b.stats.channel.location)
        && prefix(&a.stats.channel.channel) == prefix(&b.stats.channel.channel)
}

fn infer_horizontal_cmpaz(tr: &Trace) -> Result<f64> {
    if let Some(ref sac) = tr.stats.sac {
        if let Some(az) = sac.get_float("cmpaz") {
            return Ok(az);
        }
    }
    match tr.stats.channel.component().map(|c| c.to_ascii_uppercase()) {
        Some('N' | '1') => Ok(0.0),
        Some('E' | '2') => Ok(90.0),
        _ => Err(EarthSnailError::Header(format!(
            "CMPAZ undefined on horizontal channel '{}'",
            tr.stats.channel
        ))),
    }
}

fn extract_seed_az_dip(tr: &Trace) -> Result<(f64, f64)> {
    if let Some(ref sac) = tr.stats.sac {
        if let (Some(az), Some(inc)) = (sac.get_float("cmpaz"), sac.get_float("cmpinc")) {
            // SAC cmpinc is [0..180] from upward vertical -> SEED dip = cmpinc - 90!
            return Ok((az, inc - 90.0));
        }
    }
    match tr.stats.channel.component().map(|c| c.to_ascii_uppercase()) {
        Some('Z') => Ok((0.0, -90.0)),
        Some('N' | '1') => Ok((0.0, 0.0)),
        Some('E' | '2') => Ok((90.0, 0.0)),
        _ => Err(EarthSnailError::Header(format!(
            "missing CMPAZ/CMPINC on '{}'",
            tr.stats.channel
        ))),
    }
}

fn set_rotated_component(tr: &mut Trace, comp_char: char, cmpaz_deg: f64, cmpinc_deg: f64) {
    let old_cha = &tr.stats.channel.channel;
    let new_cha = if old_cha.is_empty() {
        format!("BH{comp_char}")
    } else {
        let mut chs: Vec<char> = old_cha.chars().collect();
        if let Some(last) = chs.last_mut() {
            *last = comp_char;
        }
        chs.into_iter().collect()
    };
    tr.stats.channel.channel = new_cha.clone();
    if let Some(ref mut sac) = tr.stats.sac {
        let _ = sac.set_str("kcmpnm", &new_cha);
        let _ = sac.set_float("cmpaz", cmpaz_deg);
        let _ = sac.set_float("cmpinc", cmpinc_deg);
    }
    refresh_dep_stats(tr);
}

fn remove_morse_glitches_in_place(data: &mut [f64], sigma_thresh: f64) {
    let n = data.len();
    if n < 3 {
        return;
    }
    let med3 = |a: f64, b: f64, c: f64| -> f64 {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if c <= lo {
            lo
        } else if c >= hi {
            hi
        } else {
            c
        }
    };

    let mut residuals = Vec::with_capacity(n);
    residuals.push((data[0] - med3(data[0], data[1], data[2])).abs());
    for k in 1..(n - 1) {
        residuals.push((data[k] - med3(data[k - 1], data[k], data[k + 1])).abs());
    }
    residuals.push((data[n - 1] - med3(data[n - 3], data[n - 2], data[n - 1])).abs());

    let mut sorted_res = residuals.clone();
    sorted_res.sort_by(|a, b| a.total_cmp(b));
    let mad = sorted_res[sorted_res.len() / 2];
    let scale = (1.4826 * mad).max(1e-12);
    let cutoff = sigma_thresh * scale;

    let orig = data.to_vec();
    if residuals[0] > cutoff && residuals[1] <= cutoff {
        data[0] = orig[1];
    }
    for k in 1..(n - 1) {
        if residuals[k] > cutoff {
            data[k] = med3(orig[k - 1], orig[k], orig[k + 1]);
        }
    }
    if residuals[n - 1] > cutoff && residuals[n - 2] <= cutoff {
        data[n - 1] = orig[n - 2];
    }
}

fn parse_sac_taper_type(s: &str) -> Result<TaperType> {
    match s.to_ascii_lowercase().as_str() {
        "hanning" | "hann" => Ok(TaperType::Hann),
        "hamming" => Ok(TaperType::Hamming),
        "cosine" | "saccosine" => Ok(TaperType::SacCosine),
        "blackman" => Ok(TaperType::Blackman),
        other => Err(EarthSnailError::InvalidParameter(format!(
            "unsupported SAC taper type '{other}' (expected hanning|hamming|cosine)"
        ))),
    }
}

fn apply_scalar_op(data: &mut [f64], op: &str, c: f64) -> Result<()> {
    if !c.is_finite() {
        return Err(EarthSnailError::InvalidParameter(format!(
            "SAC '{op}': non-finite operand"
        )));
    }
    if op == "div" && c == 0.0 {
        return Err(EarthSnailError::Numerical(
            "SAC 'div': division by zero".into(),
        ));
    }
    for v in data.iter_mut() {
        match op {
            "mul" => *v *= c,
            "div" => *v /= c,
            "add" => *v += c,
            "sub" => *v -= c,
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn parse_next_f64(args: &[String], idx: usize, ctx: &str) -> Result<f64> {
    let s = args.get(idx + 1).ok_or_else(|| {
        EarthSnailError::InvalidParameter(format!("SAC '{ctx}': missing numeric argument"))
    })?;
    let v = s.parse::<f64>().map_err(|_| {
        EarthSnailError::InvalidParameter(format!("SAC '{ctx}': invalid float '{s}'"))
    })?;
    if !v.is_finite() {
        return Err(EarthSnailError::InvalidParameter(format!(
            "SAC '{ctx}': non-finite float '{s}'"
        )));
    }
    Ok(v)
}

fn parse_next_usize(args: &[String], idx: usize, ctx: &str) -> Result<usize> {
    let s = args.get(idx + 1).ok_or_else(|| {
        EarthSnailError::InvalidParameter(format!("SAC '{ctx}': missing integer argument"))
    })?;
    s.parse::<usize>().map_err(|_| {
        EarthSnailError::InvalidParameter(format!("SAC '{ctx}': invalid positive integer '{s}'"))
    })
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_pi, mut star_ti) = (None::<usize>, 0usize);

    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi].eq_ignore_ascii_case(&t[ti])) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_pi = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn parse_taup_cmd_args(args: &[String]) -> Result<(String, Vec<String>, bool)> {
    let mut model_name = String::from("iasp91");
    let mut phases: Vec<String> = Vec::new();
    let mut force_evdp_km = false;

    let mut i = 0usize;
    while i < args.len() {
        let lc = args[i].to_ascii_lowercase();
        match lc.as_str() {
            "model" | "mod" | "m" => {
                let m = args.get(i + 1).ok_or_else(|| {
                    EarthSnailError::InvalidParameter(
                        "SAC 'taup model' expects a model name".into(),
                    )
                })?;
                model_name = m.clone();
                i += 2;
                continue;
            }
            "iasp91" | "ak135" | "prem" if phases.is_empty() => {
                model_name = lc;
                i += 1;
                continue;
            }
            "evdpkm" => {
                force_evdp_km = true;
                i += 1;
                continue;
            }
            "phases" | "phase" | "ph" => {
                i += 1;
                while i < args.len() {
                    let tok_lc = args[i].to_ascii_lowercase();
                    if matches!(tok_lc.as_str(), "model" | "mod" | "evdpkm") {
                        break;
                    }
                    for p in args[i].split(',').map(str::trim).filter(|s| !s.is_empty()) {
                        phases.push(String::from(p));
                    }
                    i += 1;
                }
                continue;
            }
            _ => {
                for p in args[i].split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    phases.push(String::from(p));
                }
                i += 1;
            }
        }
    }

    if phases.is_empty() {
        phases = vec![
            String::from("P"),
            String::from("S"),
            String::from("pP"),
            String::from("PP"),
            String::from("PcP"),
            String::from("SKS"),
            String::from("PKIKP"),
        ];
    }

    Ok((model_name, phases, force_evdp_km))
}

#[cfg(test)]
mod tests {
    use super::*;
    use earthsnail_core::gps2dist_azimuth;

    #[test]
    fn sac_session_250_traces_blackboard_macros_undo_and_dsp_under_miri() {
        let mut sess = SacSession::new();

        // 1. Prove legacy SAC's 200-file ceiling is dead by loading 250 traces simultaneously!
        let n_files = if cfg!(miri) { 205 } else { 260 };
        let n_samp = if cfg!(miri) { 16 } else { 128 };
        sess.execute_line(&format!("fg sine 2.0 delta 0.05 npts {n_samp}"))
            .unwrap();
        sess.execute_line("w mem://base.sac").unwrap();
        let b = sess.get_virtual_file("mem://base.sac").unwrap().to_vec();
        for k in 0..n_files {
            sess.put_virtual_file(format!("mem://tr_{k:03}.sac"), b.clone());
        }
        sess.execute_line("r mem://tr_*.sac").unwrap();
        assert_eq!(sess.traces.len(), n_files);
        assert!(sess.traces.len() > 200);

        // 2. Blackboard variables ($var and %var%), do-loops, and positional macro arguments ($1, $2):
        sess.put_virtual_macro(
            "proc.m",
            "* Comment line at top\n\
             setbb FMIN $1\n\
             setbb FMAX $2\n\
             rmean; rtrend; taper type cosine width 0.05\n\
             bp co $FMIN %FMAX% n 4 p 2\n",
        );
        sess.execute_line("dc all").unwrap();
        assert_eq!(sess.traces.len(), 0);
        sess.execute_line("undo").unwrap();
        assert_eq!(sess.traces.len(), n_files);

        // Trim down to 1 synthetic trace for fast DSP/FFT/undo check under Miri:
        sess.execute_line(&format!(
            "fg sine 1.5 delta 0.05 npts {}",
            if cfg!(miri) { 64 } else { 256 }
        ))
        .unwrap();
        sess.execute_line("ch stla 34.9459 stlo -106.4572 evla 38.297 evlo 142.373 evdp 29.0")
            .unwrap();
        let gcarc = sess.traces[0]
            .stats
            .sac
            .as_ref()
            .unwrap()
            .get_float("gcarc")
            .expect("LCALDA auto-populated gcarc");
        let expected_geo = gps2dist_azimuth(38.297, 142.373, 34.9459, -106.4572);
        assert!((gcarc - expected_geo.gcarc_deg).abs() < 1e-6);

        sess.execute_line("m proc.m 0.5 5.0").unwrap();
        assert_eq!(sess.blackboard.get("FMIN").map(String::as_str), Some("0.5"));

        // 3. rglitches spike excision:
        sess.traces[0].data[10] = 5.0e8;
        sess.execute_line("rglitches threshold 6.0").unwrap();
        assert!(sess.traces[0].data[10].abs() < 5.0);

        // 4. FFT -> IFFT round-trip + undo:
        let before_fft = sess.traces[0].data.clone();
        sess.execute_line("fft; ifft").unwrap();
        for (a, b) in before_fft.iter().zip(sess.traces[0].data.iter()) {
            assert!((a - b).abs() < 1e-11);
        }

        // 5. Do-loop accumulation:
        sess.execute_line("setbb ACC 0; do k = 1 to 4 by 1; add 1.0; enddo")
            .unwrap();
        sess.execute_line("undo 4").unwrap();
        assert!((sess.traces[0].data[5] - before_fft[5]).abs() < 1e-11);
    }

    #[test]
    fn malformed_sac_commands_never_panic() {
        let bad_inputs = [
            "unknown_cmd 1 2 3",
            "bp co -5 999999",
            "ch nonexistent_field 42",
            "do x = 1 to 10 by 0; rmean; enddo",
            "do unclosed = 1 to 3; rmean",
            "enddo",
            "echo \"unterminated string",
            "setbb $bad! 1",
            "echo $MISSING_VAR",
            "div 0",
            "fg npts 0",
            "cut 50.0 10.0",
            "fg sine; cut e 10 b -10",
            "undo 99",
        ];
        for bad in bad_inputs {
            let mut fresh = SacSession::new();
            assert!(
                fresh.execute_line(bad).is_err(),
                "expected error on '{bad}'"
            );
        }
    }

    #[test]
    fn taup_bare_phases_case_sensitive_bp_c_and_utf8_bom_macros() {
        // 1. Bare `taup P pP S` preserves leading `P` (not swallowed as keyword `p`) and exact `pP` case:
        let args = vec![String::from("P"), String::from("pP"), String::from("S")];
        let (model_name, phases, force_km) = parse_taup_cmd_args(&args).unwrap();
        assert_eq!(model_name, "iasp91");
        assert!(!force_km);
        assert_eq!(
            phases,
            vec![String::from("P"), String::from("pP"), String::from("S")]
        );

        // 2. UTF-8 BOM `.m` macro + classic SAC `bp c <f1> <f2> n 4 p 2` abbreviation:
        let mut sess = SacSession::new();
        sess.put_virtual_macro(
            "bom_filter.m",
            "\u{feff}* UTF-8 BOM macro header\n\
             fg sine 1.5 delta 0.05 npts 64\n\
             ch stla -35.3187 stlo 148.9963 evla 38.297 evlo 142.373 evdp 29.0 o 0.0\n\
             bp c 0.2 5.0 n 4 p 2\n",
        );
        sess.execute_line("\u{feff}m bom_filter.m").unwrap();
        assert_eq!(sess.traces.len(), 1);

        // 3. Full end-to-end `taup P pP S` header stamping into T0..T2 / KT0..KT2:
        #[cfg(not(miri))]
        {
            sess.execute_line("taup P pP S").unwrap();
            let sac = sess.traces[0].stats.sac.as_ref().unwrap();
            assert_eq!(sac.get_str("kt0"), Some("P"));
            assert_eq!(sac.get_str("kt1"), Some("pP"));
            assert_eq!(sac.get_str("kt2"), Some("S"));
            let t0 = sac.get_float("t0").unwrap();
            let t1 = sac.get_float("t1").unwrap();
            let t2 = sac.get_float("t2").unwrap();
            assert!(t0 > 600.0 && t1 > t0 && t2 > t1);
        }
    }
}
