//! Zero-dependency RFC-8259 JSON builder (`JsonObj`, `JsonArr`, `json_escape`) guaranteeing
//! strictly finite numeric serialization (`null` fallback for non-finite `f64`) and proper
//! Unicode/control-character escaping for `--json` CLI output.

use core::fmt::Write as _;

/// Escapes a string into a valid double-quoted JSON string literal (`"..."`).
pub fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Formats an `f64` as a valid JSON number, emitting `null` if `NaN` or `Infinity`.
pub fn json_f64(v: f64) -> String {
    if !v.is_finite() {
        String::from("null")
    } else if v == 0.0 {
        String::from("0.0")
    } else {
        let av = v.abs();
        if !(1e-6..1e15).contains(&av) {
            format!("{v:.12e}")
        } else {
            let mut s = format!("{v:.10}");
            if s.contains('.') {
                while s.ends_with('0') && !s.ends_with(".0") {
                    s.pop();
                }
            }
            s
        }
    }
}

/// Formats an `Option<f64>` as a JSON number or `null`.
pub fn json_opt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) => json_f64(x),
        None => String::from("null"),
    }
}

/// Formats a slice of `f64` values as a compact JSON array `[...]`.
pub fn json_f64_slice(vals: &[f64]) -> String {
    let mut out = String::with_capacity(vals.len() * 10 + 2);
    out.push('[');
    for (i, &v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&json_f64(v));
    }
    out.push(']');
    out
}

/// Zero-dependency JSON object builder (`{ "k": v, ... }`).
#[derive(Debug, Clone, Default)]
pub struct JsonObj {
    buf: String,
    count: usize,
}

impl JsonObj {
    /// Creates an empty JSON object builder.
    pub fn new() -> Self {
        let mut buf = String::with_capacity(128);
        buf.push('{');
        Self { buf, count: 0 }
    }

    fn key_prefix(&mut self, key: &str) {
        if self.count > 0 {
            self.buf.push(',');
        }
        self.buf.push_str(&json_quote(key));
        self.buf.push(':');
        self.count += 1;
    }

    /// Inserts a raw pre-formatted JSON value expression.
    pub fn raw(&mut self, key: &str, raw_json: &str) -> &mut Self {
        self.key_prefix(key);
        self.buf.push_str(raw_json);
        self
    }

    /// Inserts a string field.
    pub fn str(&mut self, key: &str, val: &str) -> &mut Self {
        self.key_prefix(key);
        self.buf.push_str(&json_quote(val));
        self
    }

    /// Inserts an `Option<&str>` field (`"..."` or `null`).
    pub fn opt_str(&mut self, key: &str, val: Option<&str>) -> &mut Self {
        match val {
            Some(s) => self.str(key, s),
            None => self.raw(key, "null"),
        }
    }

    /// Inserts a boolean field.
    pub fn bool(&mut self, key: &str, val: bool) -> &mut Self {
        self.raw(key, if val { "true" } else { "false" })
    }

    /// Inserts an integer `i64` field.
    pub fn int(&mut self, key: &str, val: i64) -> &mut Self {
        self.key_prefix(key);
        let _ = write!(self.buf, "{val}");
        self
    }

    /// Inserts an unsigned integer `usize` field.
    pub fn uint(&mut self, key: &str, val: usize) -> &mut Self {
        self.key_prefix(key);
        let _ = write!(self.buf, "{val}");
        self
    }

    /// Inserts a finite `f64` field (`null` if non-finite).
    pub fn num(&mut self, key: &str, val: f64) -> &mut Self {
        self.key_prefix(key);
        self.buf.push_str(&json_f64(val));
        self
    }

    /// Inserts an `Option<f64>` field.
    pub fn opt_num(&mut self, key: &str, val: Option<f64>) -> &mut Self {
        self.key_prefix(key);
        self.buf.push_str(&json_opt_f64(val));
        self
    }

    /// Finishes and returns the JSON object string.
    pub fn finish(mut self) -> String {
        self.buf.push('}');
        self.buf
    }
}

/// Zero-dependency JSON array builder (`[ v1, v2, ... ]`).
#[derive(Debug, Clone, Default)]
pub struct JsonArr {
    buf: String,
    count: usize,
}

impl JsonArr {
    /// Creates an empty JSON array builder.
    pub fn new() -> Self {
        let mut buf = String::with_capacity(128);
        buf.push('[');
        Self { buf, count: 0 }
    }

    fn sep(&mut self) {
        if self.count > 0 {
            self.buf.push(',');
        }
        self.count += 1;
    }

    /// Appends a raw pre-formatted JSON value.
    pub fn push_raw(&mut self, raw_json: &str) -> &mut Self {
        self.sep();
        self.buf.push_str(raw_json);
        self
    }

    /// Appends a string element.
    pub fn push_str(&mut self, val: &str) -> &mut Self {
        self.sep();
        self.buf.push_str(&json_quote(val));
        self
    }

    /// Appends an `f64` element.
    pub fn push_num(&mut self, val: f64) -> &mut Self {
        self.sep();
        self.buf.push_str(&json_f64(val));
        self
    }

    /// Finishes and returns the JSON array string.
    pub fn finish(mut self) -> String {
        self.buf.push(']');
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_builder_escapes_and_handles_non_finite() {
        let mut arr = JsonArr::new();
        arr.push_str("P\"S\\wave\n");
        arr.push_num(12.5);
        arr.push_num(f64::NAN);

        let mut obj = JsonObj::new();
        obj.str("status", "ok")
            .bool("valid", true)
            .uint("count", 3)
            .opt_num("missing", None)
            .raw("items", &arr.finish());
        let out = obj.finish();
        assert_eq!(
            out,
            r#"{"status":"ok","valid":true,"count":3,"missing":null,"items":["P\"S\\wave\n",12.5,null]}"#
        );
    }
}
