//! Allocation-bounded, zero-dependency XML 1.0 pull scanner (`XmlReader`) supporting namespaces,
//! attributes, `CDATA` blocks, predefined & numeric character entities (`&amp;`, `&lt;`, `&gt;`,
//! `&quot;`, `&apos;`, `&#...;`, `&#x...;`), and strict open/close tag validation with zero panics.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use earthsnail_core::{EarthSnailError, Result};

/// Maximum XML element nesting depth accepted before returning `CapacityExceeded`.
pub const MAX_XML_DEPTH: usize = 128;

/// Maximum attributes accepted on a single start element.
pub const MAX_XML_ATTRS: usize = 128;

/// Maximum decoded text/attribute length (`16 MiB`) to prevent entity expansion bombs.
pub const MAX_XML_TEXT_BYTES: usize = 16 * 1024 * 1024;

/// A single XML pull-parser event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XmlEvent<'a> {
    /// Opening tag `<prefix:name attr="val">` (or `<prefix:name .../>` when `self_closing` is true;
    /// when `self_closing == true`, `XmlReader` automatically emits a matching `EndElement` next).
    StartElement {
        raw_name: &'a str,
        local_name: &'a str,
        attributes: Vec<XmlAttr<'a>>,
    },
    /// Closing tag `</prefix:name>`.
    EndElement {
        raw_name: &'a str,
        local_name: &'a str,
    },
    /// Decoded character data (including decoded entities and `CDATA` sections).
    Text(String),
    /// End of XML document (all open tags verified closed).
    Eof,
}

/// Decoded XML attribute key-value pair on an opening tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmlAttr<'a> {
    /// Full attribute name including optional namespace prefix (e.g. `"xmlns:q"`, `"publicID"`).
    pub raw_name: &'a str,
    /// Namespace-stripped local attribute name (e.g. `"publicID"`).
    pub local_name: &'a str,
    /// Entity-unescaped attribute value.
    pub value: String,
}

impl<'a> XmlEvent<'a> {
    /// Looks up an attribute value by local name on a `StartElement`.
    pub fn attr(&self, local: &str) -> Option<&str> {
        match self {
            Self::StartElement { attributes, .. } => attributes
                .iter()
                .find(|a| a.local_name == local)
                .map(|a| a.value.as_str()),
            _ => None,
        }
    }
}

/// Strips any namespace prefix (`"ns:Tag" -> "Tag"`).
#[inline]
pub fn strip_ns(name: &str) -> &str {
    match name.split_once(':') {
        Some((_, local)) => local,
        None => name,
    }
}

/// Unescapes XML predefined entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`) and numeric
/// character references (`&#10;`, `&#x0A;`), rejecting malformed/unknown entities cleanly.
pub fn xml_unescape(raw: &str) -> Result<String> {
    if raw.len() > MAX_XML_TEXT_BYTES {
        return Err(EarthSnailError::CapacityExceeded(
            "XML text segment exceeds maximum safety length".into(),
        ));
    }
    if !raw.contains('&') {
        return Ok(String::from(raw));
    }

    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(amp_idx) = rest.find('&') {
        out.push_str(&rest[..amp_idx]);
        let after_amp = &rest[amp_idx + 1..];
        let semi_idx = after_amp.find(';').ok_or_else(|| {
            EarthSnailError::Format("unterminated XML entity reference (missing ';')".into())
        })?;
        let entity = &after_amp[..semi_idx];
        let decoded_ch = match entity {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                let hex = &entity[2..];
                if hex.is_empty() || hex.len() > 6 {
                    return Err(EarthSnailError::Format(format!(
                        "invalid hex XML character reference '&{entity};'"
                    )));
                }
                let code = u32::from_str_radix(hex, 16).map_err(|_| {
                    EarthSnailError::Format(format!(
                        "invalid hex XML character reference '&{entity};'"
                    ))
                })?;
                validate_xml_char(code)?
            }
            _ if entity.starts_with('#') => {
                let dec = &entity[1..];
                if dec.is_empty() || dec.len() > 7 {
                    return Err(EarthSnailError::Format(format!(
                        "invalid decimal XML character reference '&{entity};'"
                    )));
                }
                let code: u32 = dec.parse().map_err(|_| {
                    EarthSnailError::Format(format!(
                        "invalid decimal XML character reference '&{entity};'"
                    ))
                })?;
                validate_xml_char(code)?
            }
            _ => {
                return Err(EarthSnailError::Format(format!(
                    "unknown or unsupported XML entity '&{entity};'"
                )));
            }
        };
        if out.len() + decoded_ch.len_utf8() > MAX_XML_TEXT_BYTES {
            return Err(EarthSnailError::CapacityExceeded(
                "XML entity expansion exceeded safety limit".into(),
            ));
        }
        out.push(decoded_ch);
        rest = &after_amp[semi_idx + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[inline]
fn validate_xml_char(code: u32) -> Result<char> {
    // XML 1.0 valid Char production (#x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF])
    let ok = matches!(
        code,
        0x09 | 0x0A | 0x0D | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF
    );
    if !ok {
        return Err(EarthSnailError::Format(format!(
            "disallowed XML character code point U+{code:04X}"
        )));
    }
    char::from_u32(code)
        .ok_or_else(|| EarthSnailError::Format(format!("invalid Unicode scalar U+{code:04X}")))
}

/// Streaming zero-dependency XML 1.0 pull reader.
pub struct XmlReader<'a> {
    src: &'a str,
    pos: usize,
    stack: Vec<&'a str>,
    pending_close: Option<&'a str>,
    seen_root: bool,
}

impl<'a> XmlReader<'a> {
    /// Creates an `XmlReader` over a UTF-8 string slice (stripping any leading UTF-8 BOM).
    pub fn new(input: &'a str) -> Self {
        let src = input.strip_prefix('\u{feff}').unwrap_or(input);
        Self {
            src,
            pos: 0,
            stack: Vec::new(),
            pending_close: None,
            seen_root: false,
        }
    }

    /// Creates an `XmlReader` over a byte slice after validating UTF-8.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let s = core::str::from_utf8(bytes)
            .map_err(|_| EarthSnailError::Format("XML document is not valid UTF-8".into()))?;
        Ok(Self::new(s))
    }

    /// Current open-element nesting depth.
    #[inline]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Pulls the next `XmlEvent` from the stream.
    pub fn next_event(&mut self) -> Result<XmlEvent<'a>> {
        if let Some(raw_name) = self.pending_close.take() {
            return Ok(XmlEvent::EndElement {
                raw_name,
                local_name: strip_ns(raw_name),
            });
        }

        let bytes = self.src.as_bytes();
        let mut text_acc = String::new();

        loop {
            if self.pos >= bytes.len() {
                if !self.stack.is_empty() {
                    let unclosed = self.stack.last().copied().unwrap_or("?");
                    return Err(EarthSnailError::Format(format!(
                        "unexpected EOF with unclosed XML element <{unclosed}>"
                    )));
                }
                if !text_acc.trim().is_empty() {
                    return Err(EarthSnailError::Format(
                        "stray non-whitespace character data outside XML root element".into(),
                    ));
                }
                if !self.seen_root {
                    return Err(EarthSnailError::Format("empty XML document".into()));
                }
                return Ok(XmlEvent::Eof);
            }

            let rem = &self.src[self.pos..];
            if let Some(lt_rel) = rem.find('<') {
                if lt_rel > 0 {
                    let raw_text = &rem[..lt_rel];
                    if raw_text.contains("]]>") {
                        return Err(EarthSnailError::Format(
                            "illegal ']]>' sequence in XML character data".into(),
                        ));
                    }
                    self.pos += lt_rel;
                    if self.stack.is_empty() {
                        if !raw_text.trim().is_empty() {
                            return Err(EarthSnailError::Format(
                                "character data outside XML root element".into(),
                            ));
                        }
                    } else {
                        let decoded = xml_unescape(raw_text)?;
                        if text_acc.len() + decoded.len() > MAX_XML_TEXT_BYTES {
                            return Err(EarthSnailError::CapacityExceeded(
                                "XML element text exceeds safety limit".into(),
                            ));
                        }
                        text_acc.push_str(&decoded);
                    }
                }

                // Now `self.src[self.pos..]` starts with `<`.
                let tag_rem = &self.src[self.pos..];
                if let Some(after_cdata) = tag_rem.strip_prefix("<![CDATA[") {
                    if self.stack.is_empty() {
                        return Err(EarthSnailError::Format(
                            "CDATA section outside XML root element".into(),
                        ));
                    }
                    let end_cdata = after_cdata.find("]]>").ok_or_else(|| {
                        EarthSnailError::Format("unterminated <![CDATA[ section".into())
                    })?;
                    let cdata_body = &after_cdata[..end_cdata];
                    if text_acc.len() + cdata_body.len() > MAX_XML_TEXT_BYTES {
                        return Err(EarthSnailError::CapacityExceeded(
                            "XML CDATA exceeds safety limit".into(),
                        ));
                    }
                    text_acc.push_str(cdata_body);
                    self.pos += 9 + end_cdata + 3;
                    continue;
                }

                if let Some(after_comment) = tag_rem.strip_prefix("<!--") {
                    let end_comment = after_comment.find("-->").ok_or_else(|| {
                        EarthSnailError::Format("unterminated XML comment".into())
                    })?;
                    let comment_body = &after_comment[..end_comment];
                    if comment_body.contains("--") {
                        return Err(EarthSnailError::Format(
                            "illegal '--' inside XML comment".into(),
                        ));
                    }
                    self.pos += 4 + end_comment + 3;
                    continue;
                }

                if let Some(after_pi) = tag_rem.strip_prefix("<?") {
                    let end_pi = after_pi.find("?>").ok_or_else(|| {
                        EarthSnailError::Format(
                            "unterminated XML declaration or processing instruction".into(),
                        )
                    })?;
                    self.pos += 2 + end_pi + 2;
                    continue;
                }

                if tag_rem.starts_with("<!DOCTYPE") || tag_rem.starts_with("<!doctype") {
                    // Reject internal DTD `<!ENTITY ...>` declarations to eliminate Billion Laughs;
                    // skip simple external/root DOCTYPE without `[` subset.
                    let gt = tag_rem.find('>').ok_or_else(|| {
                        EarthSnailError::Format("unterminated <!DOCTYPE declaration".into())
                    })?;
                    if tag_rem[..gt].contains('[') {
                        return Err(EarthSnailError::Unsupported(
                            "XML internal DTD subsets / custom <!ENTITY> declarations are disallowed"
                                .into(),
                        ));
                    }
                    self.pos += gt + 1;
                    continue;
                }

                if tag_rem.starts_with("<!") {
                    return Err(EarthSnailError::Format(
                        "malformed or unsupported XML '<!' construct".into(),
                    ));
                }

                // Before emitting a start or end tag, flush any accumulated non-empty text!
                if !text_acc.is_empty() {
                    return Ok(XmlEvent::Text(text_acc));
                }

                // Closing tag `</name>`
                if let Some(after_close) = tag_rem.strip_prefix("</") {
                    let gt = after_close
                        .find('>')
                        .ok_or_else(|| EarthSnailError::Format("unclosed XML end tag".into()))?;
                    let close_name = after_close[..gt].trim();
                    if close_name.is_empty() || close_name.contains('<') {
                        return Err(EarthSnailError::Format(format!(
                            "invalid XML closing tag '</{close_name}>'"
                        )));
                    }
                    let expected = self.stack.pop().ok_or_else(|| {
                        EarthSnailError::Format(format!(
                            "unexpected closing tag '</{close_name}>' with empty open-tag stack"
                        ))
                    })?;
                    if expected != close_name {
                        return Err(EarthSnailError::Format(format!(
                            "mismatched XML closing tag: expected '</{expected}>', got '</{close_name}>'"
                        )));
                    }
                    self.pos += 2 + gt + 1;
                    return Ok(XmlEvent::EndElement {
                        raw_name: close_name,
                        local_name: strip_ns(close_name),
                    });
                }

                // Opening tag `<name attr1="v1" ...>` or `<name .../>`
                if self.seen_root && self.stack.is_empty() {
                    return Err(EarthSnailError::Format(
                        "multiple root elements in XML document".into(),
                    ));
                }
                let (raw_name, attributes, self_closing, consumed) =
                    parse_start_tag(&tag_rem[1..])?;
                self.pos += 1 + consumed;
                self.seen_root = true;

                if self_closing {
                    self.pending_close = Some(raw_name);
                } else {
                    if self.stack.len() >= MAX_XML_DEPTH {
                        return Err(EarthSnailError::CapacityExceeded(format!(
                            "XML nesting depth exceeds {MAX_XML_DEPTH}"
                        )));
                    }
                    self.stack.push(raw_name);
                }

                return Ok(XmlEvent::StartElement {
                    raw_name,
                    local_name: strip_ns(raw_name),
                    attributes,
                });
            } else {
                // Remaining tail has no `<`
                if !self.stack.is_empty() {
                    let unclosed = self.stack.last().copied().unwrap_or("?");
                    return Err(EarthSnailError::Format(format!(
                        "unexpected EOF inside unclosed XML element <{unclosed}>"
                    )));
                }
                if !rem.trim().is_empty() {
                    return Err(EarthSnailError::Format(
                        "trailing non-whitespace text after XML root element".into(),
                    ));
                }
                self.pos = bytes.len();
            }
        }
    }

    /// Reads the trimmed text body of the currently open element and consumes its closing `EndElement`.
    /// Any nested child elements inside are skipped cleanly.
    pub fn read_text_to_end(&mut self, expected_raw_name: &str) -> Result<String> {
        if self.pending_close == Some(expected_raw_name) {
            let _ = self.next_event()?;
            return Ok(String::new());
        }
        let mut out = String::new();
        let target_depth = self.stack.len().saturating_sub(1);
        loop {
            match self.next_event()? {
                XmlEvent::Text(t) => {
                    if self.stack.len() == target_depth + 1 {
                        out.push_str(&t);
                    }
                }
                XmlEvent::StartElement { .. } => {
                    // Nested tag inside value wrapper; ignore its text unless at our level
                }
                XmlEvent::EndElement { raw_name, .. } => {
                    if self.stack.len() == target_depth && raw_name == expected_raw_name {
                        return Ok(String::from(out.trim()));
                    }
                }
                XmlEvent::Eof => {
                    return Err(EarthSnailError::Format(format!(
                        "unexpected EOF while reading <{expected_raw_name}>"
                    )));
                }
            }
        }
    }

    /// Skips all tokens until the matching `EndElement` for the just-opened `expected_raw_name` is consumed.
    pub fn skip_element(&mut self, expected_raw_name: &str) -> Result<()> {
        // Note: if the start tag was self-closing (`<Foo/>`), `pending_close` holds `expected_raw_name`
        // while `self.stack` was not pushed!
        if self.pending_close == Some(expected_raw_name) {
            let _ = self.next_event()?;
            return Ok(());
        }
        let target_depth = self.stack.len().saturating_sub(1);
        loop {
            match self.next_event()? {
                XmlEvent::EndElement { raw_name, .. }
                    if self.stack.len() == target_depth && raw_name == expected_raw_name =>
                {
                    return Ok(());
                }
                XmlEvent::Eof => {
                    return Err(EarthSnailError::Format(format!(
                        "unexpected EOF while skipping <{expected_raw_name}>"
                    )));
                }
                _ => {}
            }
        }
    }
}

fn parse_start_tag(s: &str) -> Result<(&str, Vec<XmlAttr<'_>>, bool, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0;

    // Parse tag name
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b'/'
    {
        if bytes[i] == b'<' || bytes[i] == b'=' || bytes[i] == b'"' || bytes[i] == b'\'' {
            return Err(EarthSnailError::Format(
                "illegal character in XML element name".into(),
            ));
        }
        i += 1;
    }
    if i == 0 {
        return Err(EarthSnailError::Format(
            "empty XML element name '<>'".into(),
        ));
    }
    let raw_name = &s[..i];
    if !is_valid_xml_name(raw_name) {
        return Err(EarthSnailError::Format(format!(
            "invalid XML element name '{raw_name}'"
        )));
    }

    let mut attrs = Vec::new();
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return Err(EarthSnailError::Format(format!(
                "unterminated XML start tag <{raw_name}"
            )));
        }
        if bytes[i] == b'>' {
            return Ok((raw_name, attrs, false, i + 1));
        }
        if bytes[i] == b'/' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                return Ok((raw_name, attrs, true, i + 2));
            }
            return Err(EarthSnailError::Format(format!(
                "expected '/>' in XML start tag <{raw_name}>"
            )));
        }

        // Parse attribute name
        let attr_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'='
            && bytes[i] != b'>'
            && bytes[i] != b'/'
        {
            i += 1;
        }
        let attr_name = &s[attr_start..i];
        if !is_valid_xml_name(attr_name) {
            return Err(EarthSnailError::Format(format!(
                "invalid XML attribute name '{attr_name}' in <{raw_name}>"
            )));
        }
        if attrs.len() >= MAX_XML_ATTRS {
            return Err(EarthSnailError::CapacityExceeded(
                "too many attributes on XML element".into(),
            ));
        }

        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            return Err(EarthSnailError::Format(format!(
                "missing '=' for XML attribute '{attr_name}' in <{raw_name}>"
            )));
        }
        i += 1; // skip '='
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || (bytes[i] != b'"' && bytes[i] != b'\'') {
            return Err(EarthSnailError::Format(format!(
                "unquoted XML attribute value for '{attr_name}' in <{raw_name}>"
            )));
        }
        let quote = bytes[i];
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != quote {
            if bytes[i] == b'<' {
                return Err(EarthSnailError::Format(format!(
                    "unescaped '<' inside XML attribute '{attr_name}'"
                )));
            }
            i += 1;
        }
        if i >= bytes.len() {
            return Err(EarthSnailError::Format(format!(
                "unterminated quoted attribute '{attr_name}' in <{raw_name}>"
            )));
        }
        let raw_val = &s[val_start..i];
        i += 1; // skip closing quote

        let value = xml_unescape(raw_val)?;
        attrs.push(XmlAttr {
            raw_name: attr_name,
            local_name: strip_ns(attr_name),
            value,
        });
    }
}

fn is_valid_xml_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_alphabetic() || first == '_' || first == ':') {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
}
