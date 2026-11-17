//! Pure-Rust **SeedLink v3 & v4** real-time seismic streaming wire protocol codec and client
//! (`"SL"` 8-byte v3 header + 512-byte miniSEED v2 payload, `"SLINFO"` multi-frame XML/ASCII
//! metadata packets, AND `"SE"` 17-byte v4 extended header + station ID + variable-length
//! miniSEED v2 / miniSEED v3 / JSON `INFO` payload), plus a generic `Read + Write` handshake
//! state machine (`HELLO`, `USERAGENT`, `STATION`, `SELECT`, `DATA`, `TIME`, `INFO`, `END`).

use crate::mseed::{read_mseed_bytes, read_mseed_records, DEFAULT_MSEED_RECORD_LEN};
use crate::mseed3::read_mseed3_bytes;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;
use earthsnail_core::{EarthSnailError, Result, Stream, UtcTimestamp};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};

/// Length of a SeedLink v3 packet header (`"SL"` + 6-char hex sequence or `"INFO *"` = `8` bytes).
pub const SEEDLINK_V3_HEADER_BYTES: usize = 8;

/// Standard SeedLink v3 miniSEED payload size (`512` bytes).
pub const SEEDLINK_V3_PAYLOAD_BYTES: usize = DEFAULT_MSEED_RECORD_LEN;

/// Total wire length of a standard SeedLink v3 packet (`8 + 512 = 520` bytes).
pub const SEEDLINK_V3_PACKET_BYTES: usize = SEEDLINK_V3_HEADER_BYTES + SEEDLINK_V3_PAYLOAD_BYTES;

/// Fixed header length of a SeedLink v4 `"SE"` extended packet (`17` bytes before `station_id`).
pub const SEEDLINK_V4_FIXED_HEADER_BYTES: usize = 17;

/// Maximum SeedLink v4 payload accepted (`4 MiB`).
pub const MAX_SEEDLINK_V4_PAYLOAD: usize = 4 * 1024 * 1024;

/// SeedLink v4 payload format code (`byte 2` of `"SE"` header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedLinkV4Format {
    /// `'2'`: miniSEED v2 record.
    MiniSeed2,
    /// `'3'`: FDSN miniSEED v3 record.
    MiniSeed3,
    /// `'I'`: SeedLink v4 JSON `INFO` document.
    InfoJson,
    /// `'E'`: SeedLink v4 `INFO` error payload.
    InfoError,
    /// Other ASCII format code.
    Other(u8),
}

impl SeedLinkV4Format {
    pub const fn from_byte(b: u8) -> Self {
        match b {
            b'2' => Self::MiniSeed2,
            b'3' => Self::MiniSeed3,
            b'I' => Self::InfoJson,
            b'E' => Self::InfoError,
            other => Self::Other(other),
        }
    }

    pub const fn to_byte(self) -> u8 {
        match self {
            Self::MiniSeed2 => b'2',
            Self::MiniSeed3 => b'3',
            Self::InfoJson => b'I',
            Self::InfoError => b'E',
            Self::Other(b) => b,
        }
    }
}

/// Decoded SeedLink v3 (`"SL"`) or v4 (`"SE"`) wire packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedLinkPacket {
    /// SeedLink v3 data packet (`"SL"` + 24-bit hex sequence number + 512B miniSEED v2 payload).
    V3Data { sequence: u32, payload: Vec<u8> },
    /// SeedLink v3 `INFO` packet (`"SLINFO *"` when `has_more == true`, `"SLINFO  "` on final frame).
    V3Info { has_more: bool, payload: Vec<u8> },
    /// SeedLink v4 extended packet (`"SE"` + format + subformat + `u32` len + `u64` seq + `station_id` + payload).
    V4Packet {
        format: SeedLinkV4Format,
        subformat: char,
        sequence: u64,
        station_id: String,
        payload: Vec<u8>,
    },
    /// Server `"END"` end-of-time-window / end-of-stream marker.
    End,
}

impl SeedLinkPacket {
    /// Serializes this `SeedLinkPacket` into its exact wire byte representation (`"SL..."`, `"SE..."`, or `"END"`).
    pub fn encode_frame(&self) -> Result<Vec<u8>> {
        match self {
            Self::V3Data { sequence, payload } => {
                if *sequence > 0x00FF_FFFF {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "SeedLink v3 sequence 0x{sequence:X} exceeds 24-bit hex limit 0xFFFFFF"
                    )));
                }
                if payload.len() != SEEDLINK_V3_PAYLOAD_BYTES {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "SeedLink v3 payload must be exactly 512 bytes, got {}",
                        payload.len()
                    )));
                }
                let mut out = Vec::with_capacity(SEEDLINK_V3_PACKET_BYTES);
                let hdr = format!("SL{sequence:06X}");
                out.extend_from_slice(hdr.as_bytes());
                out.extend_from_slice(payload);
                Ok(out)
            }
            Self::V3Info { has_more, payload } => {
                if payload.len() != SEEDLINK_V3_PAYLOAD_BYTES {
                    return Err(EarthSnailError::InvalidParameter(format!(
                        "SeedLink v3 INFO payload must be 512 bytes, got {}",
                        payload.len()
                    )));
                }
                let mut out = Vec::with_capacity(SEEDLINK_V3_PACKET_BYTES);
                out.extend_from_slice(if *has_more { b"SLINFO *" } else { b"SLINFO  " });
                out.extend_from_slice(payload);
                Ok(out)
            }
            Self::V4Packet {
                format,
                subformat,
                sequence,
                station_id,
                payload,
            } => {
                let sid_bytes = station_id.as_bytes();
                if sid_bytes.len() > 255 {
                    return Err(EarthSnailError::InvalidParameter(
                        "SeedLink v4 station_id exceeds 255 bytes".into(),
                    ));
                }
                if payload.len() > MAX_SEEDLINK_V4_PAYLOAD {
                    return Err(EarthSnailError::CapacityExceeded(
                        "SeedLink v4 payload exceeds maximum safety limit".into(),
                    ));
                }
                if !subformat.is_ascii() {
                    return Err(EarthSnailError::InvalidParameter(
                        "SeedLink v4 subformat must be ASCII".into(),
                    ));
                }
                let total = SEEDLINK_V4_FIXED_HEADER_BYTES + sid_bytes.len() + payload.len();
                let mut out = Vec::with_capacity(total);
                out.extend_from_slice(b"SE");
                out.push(format.to_byte());
                out.push(*subformat as u8);
                out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                out.extend_from_slice(&sequence.to_le_bytes());
                out.push(sid_bytes.len() as u8);
                debug_assert_eq!(out.len(), SEEDLINK_V4_FIXED_HEADER_BYTES);
                out.extend_from_slice(sid_bytes);
                out.extend_from_slice(payload);
                Ok(out)
            }
            Self::End => Ok(Vec::from(&b"END"[..])),
        }
    }

    /// Parses a single `SeedLinkPacket` from the front of `bytes`, returning `(packet, bytes_consumed)`.
    pub fn decode_frame(bytes: &[u8]) -> Result<(Self, usize)> {
        if bytes.len() < 3 {
            return Err(EarthSnailError::Format(
                "truncated SeedLink frame (< 3 bytes)".into(),
            ));
        }
        if bytes.starts_with(b"END") {
            return Ok((Self::End, 3));
        }
        if bytes.starts_with(b"ERROR") {
            return Err(EarthSnailError::Network(
                "SeedLink server returned ERROR frame during data phase".into(),
            ));
        }

        if bytes.starts_with(b"SL") {
            if bytes.len() < SEEDLINK_V3_PACKET_BYTES {
                return Err(EarthSnailError::Format(format!(
                    "truncated SeedLink v3 packet ({} < 520 bytes)",
                    bytes.len()
                )));
            }
            let hdr_tail = &bytes[2..8];
            let payload = bytes[8..SEEDLINK_V3_PACKET_BYTES].to_vec();
            if hdr_tail.starts_with(b"INFO") {
                let has_more = hdr_tail[5] == b'*';
                return Ok((Self::V3Info { has_more, payload }, SEEDLINK_V3_PACKET_BYTES));
            }
            let hex_str = core::str::from_utf8(hdr_tail).map_err(|_| {
                EarthSnailError::Format("non-ASCII SeedLink v3 sequence header".into())
            })?;
            let sequence = u32::from_str_radix(hex_str.trim(), 16).map_err(|_| {
                EarthSnailError::Format(format!("invalid SeedLink v3 hex sequence '{hex_str}'"))
            })?;
            return Ok((Self::V3Data { sequence, payload }, SEEDLINK_V3_PACKET_BYTES));
        }

        if bytes.starts_with(b"SE") {
            if bytes.len() < SEEDLINK_V4_FIXED_HEADER_BYTES {
                return Err(EarthSnailError::Format(format!(
                    "truncated SeedLink v4 fixed header ({} < 17 bytes)",
                    bytes.len()
                )));
            }
            let format = SeedLinkV4Format::from_byte(bytes[2]);
            let subformat_byte = bytes[3];
            if !subformat_byte.is_ascii() {
                return Err(EarthSnailError::Format(
                    "non-ASCII SeedLink v4 subformat byte".into(),
                ));
            }
            let payload_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
            if payload_len > MAX_SEEDLINK_V4_PAYLOAD {
                return Err(EarthSnailError::CapacityExceeded(format!(
                    "SeedLink v4 payload length {payload_len} exceeds safety limit {MAX_SEEDLINK_V4_PAYLOAD}"
                )));
            }
            let sequence = u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]);
            let sid_len = bytes[16] as usize;
            let total_len = SEEDLINK_V4_FIXED_HEADER_BYTES + sid_len + payload_len;
            if bytes.len() < total_len {
                return Err(EarthSnailError::Format(format!(
                    "truncated SeedLink v4 packet: expected {total_len} bytes, got {}",
                    bytes.len()
                )));
            }
            let sid_slice =
                &bytes[SEEDLINK_V4_FIXED_HEADER_BYTES..SEEDLINK_V4_FIXED_HEADER_BYTES + sid_len];
            let station_id =
                String::from(core::str::from_utf8(sid_slice).map_err(|_| {
                    EarthSnailError::Format("non-UTF8 SeedLink v4 station ID".into())
                })?);
            let payload = bytes[SEEDLINK_V4_FIXED_HEADER_BYTES + sid_len..total_len].to_vec();
            return Ok((
                Self::V4Packet {
                    format,
                    subformat: subformat_byte as char,
                    sequence,
                    station_id,
                    payload,
                },
                total_len,
            ));
        }

        Err(EarthSnailError::Format(format!(
            "unrecognized SeedLink frame magic [0x{:02X}, 0x{:02X}]",
            bytes[0], bytes[1]
        )))
    }

    /// Decodes the embedded miniSEED v2 or miniSEED v3 waveform payload into a `Stream`
    /// (`Ok(None)` for `INFO` or `End` packets).
    pub fn decode_waveforms(&self) -> Result<Option<Stream>> {
        match self {
            Self::V3Data { payload, .. } => Ok(Some(read_mseed_bytes(payload)?)),
            Self::V4Packet {
                format: SeedLinkV4Format::MiniSeed2,
                payload,
                ..
            } => Ok(Some(read_mseed_bytes(payload)?)),
            Self::V4Packet {
                format: SeedLinkV4Format::MiniSeed3,
                payload,
                ..
            } => Ok(Some(read_mseed3_bytes(payload)?)),
            _ => Ok(None),
        }
    }
}

/// Server banner returned by `HELLO`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedLinkHello {
    /// Protocol capabilities line (e.g. `"SeedLink v4.0 (2024.100) :: SLPROTO:4.0 CAP EXTREPLY"` or `"SeedLink v3.1"`).
    pub software_line: String,
    /// Organization / data center description line (e.g. `"EarthScope Ringserver"`).
    pub organization_line: String,
    /// `true` if server advertised `SLPROTO:4` or `SeedLink v4`.
    pub supports_v4: bool,
}

/// Stateful SeedLink v3/v4 client operating over any `Read + Write` byte stream (`TcpStream` in
/// production or an in-memory duplex mock stream in deterministic tests).
pub struct SeedLinkClient<S: Read + Write> {
    stream: S,
    protocol_v4: bool,
}

impl SeedLinkClient<TcpStream> {
    /// Connects to a SeedLink server over TCP (`host:18000` typical) with `timeout`.
    pub fn connect(addr: &str, timeout: Duration) -> Result<Self> {
        let mut resolved = addr.to_socket_addrs().map_err(|e| {
            EarthSnailError::Network(format!("SeedLink DNS resolution failed for '{addr}': {e}"))
        })?;
        let sock_addr = resolved.next().ok_or_else(|| {
            EarthSnailError::Network(format!("no socket addresses resolved for '{addr}'"))
        })?;
        let tcp = TcpStream::connect_timeout(&sock_addr, timeout).map_err(|e| {
            EarthSnailError::Network(format!("SeedLink TCP connect to {sock_addr} failed: {e}"))
        })?;
        let _ = tcp.set_read_timeout(Some(timeout));
        let _ = tcp.set_write_timeout(Some(timeout));
        Ok(Self::from_stream(tcp))
    }
}

impl<S: Read + Write> SeedLinkClient<S> {
    /// Wraps an existing `Read + Write` transport into a `SeedLinkClient`.
    pub fn from_stream(stream: S) -> Self {
        Self {
            stream,
            protocol_v4: false,
        }
    }

    /// Returns `true` if `HELLO` negotiated SeedLink v4 mode.
    #[inline]
    pub fn is_v4(&self) -> bool {
        self.protocol_v4
    }

    /// Borrows the underlying transport.
    pub fn inner(&self) -> &S {
        &self.stream
    }

    /// Consumes the client and returns the underlying transport.
    pub fn into_inner(self) -> S {
        self.stream
    }

    /// Sends `HELLO\r\n` and parses the two-line server banner. Automatically negotiates
    /// `SLPROTO 4.0` + `USERAGENT` if the server advertises `SLPROTO:4.0`!
    pub fn say_hello(&mut self) -> Result<SeedLinkHello> {
        self.send_raw_cmd("HELLO")?;
        let software_line = self.read_crlf_line()?;
        let organization_line = self.read_crlf_line()?;
        let up = software_line.to_ascii_uppercase();
        let supports_v4 = up.contains("SLPROTO:4") || up.contains("SEEDLINK V4");

        if up.contains("SLPROTO:4") {
            // Negotiate v4 protocol explicitly:
            self.send_raw_cmd("SLPROTO 4.0")?;
            let reply = self.read_crlf_line()?;
            if reply.starts_with("OK") {
                self.protocol_v4 = true;
            }
        } else if supports_v4 {
            self.protocol_v4 = true;
        }

        Ok(SeedLinkHello {
            software_line,
            organization_line,
            supports_v4,
        })
    }

    /// Sends `USERAGENT <client_name>\r\n` (SeedLink v4) and verifies `OK`.
    pub fn send_user_agent(&mut self, agent_name: &str) -> Result<()> {
        self.send_cmd_expect_ok(&format!("USERAGENT {agent_name}"))
    }

    /// Selects a station in either SeedLink v3 (`STATION <STA> <NET>`) or SeedLink v4 (`STATION <NET>_<STA>`).
    pub fn select_station(&mut self, network: &str, station: &str) -> Result<()> {
        let cmd = if self.protocol_v4 {
            format!("STATION {}_{}", network.trim(), station.trim())
        } else {
            format!("STATION {} {}", station.trim(), network.trim())
        };
        self.send_cmd_expect_ok(&cmd)
    }

    /// Selects a channel pattern on the active station (`SELECT <LOC><CHA>.D` in v3, or
    /// `SELECT <LOC>_<B>_<S>_<SS>` in v4).
    pub fn select_stream(&mut self, location: &str, channel: &str) -> Result<()> {
        let loc = location.trim();
        let cha = channel.trim();
        let cmd = if self.protocol_v4 {
            let formatted_cha = if cha.len() == 3 && !cha.contains('_') {
                let mut chs = cha.chars();
                let b = chs.next().unwrap_or('?');
                let s = chs.next().unwrap_or('?');
                let ss = chs.next().unwrap_or('?');
                format!("{b}_{s}_{ss}")
            } else {
                String::from(cha)
            };
            let loc_v4 = if loc.is_empty() || loc == "--" {
                "*"
            } else {
                loc
            };
            format!("SELECT {loc_v4}_{formatted_cha}")
        } else {
            let loc_v3 = if loc == "--" { "" } else { loc };
            format!("SELECT {loc_v3}{cha}.D")
        };
        self.send_cmd_expect_ok(&cmd)
    }

    /// Requests real-time/resumed data (`DATA` / `DATA <HEX_SEQ>` in v3; `DATA ALL` / `DATA ALL <DEC_SEQ>` in v4).
    pub fn request_data(&mut self, from_sequence: Option<u64>) -> Result<()> {
        let cmd = if self.protocol_v4 {
            match from_sequence {
                Some(seq) => format!("DATA ALL {seq}"),
                None => String::from("DATA ALL"),
            }
        } else {
            match from_sequence {
                Some(seq) => format!("DATA {:06X}", seq & 0x00FF_FFFF),
                None => String::from("DATA"),
            }
        };
        self.send_cmd_expect_ok(&cmd)
    }

    /// Requests a bounded historical/real-time time window (`TIME <START> [<END>]`).
    pub fn request_time_window(
        &mut self,
        start: UtcTimestamp,
        end: Option<UtcTimestamp>,
    ) -> Result<()> {
        let cmd = if self.protocol_v4 {
            match end {
                Some(e) => format!(
                    "DATA ALL {} {}",
                    format_sl_v4_time(start),
                    format_sl_v4_time(e)
                ),
                None => format!("DATA ALL {}", format_sl_v4_time(start)),
            }
        } else {
            match end {
                Some(e) => format!("TIME {} {}", format_sl_v3_time(start), format_sl_v3_time(e)),
                None => format!("TIME {}", format_sl_v3_time(start)),
            }
        };
        self.send_cmd_expect_ok(&cmd)
    }

    /// Ends the SeedLink handshaking phase (`END\r\n`) and transitions into streaming packet mode.
    pub fn end_handshake(&mut self) -> Result<()> {
        self.send_raw_cmd("END")
    }

    /// Sends `INFO <level>\r\n` (`ID`, `CAPABILITIES`, `STATIONS`, `STREAMS`) and reassembles
    /// the returned SeedLink v3 multi-packet XML/ASCII payload or SeedLink v4 JSON `INFO` document.
    pub fn request_info(&mut self, level: &str) -> Result<String> {
        self.send_raw_cmd(&format!("INFO {}", level.trim()))?;
        let mut combined = String::new();
        while let Some(pkt) = self.read_next_packet()? {
            match pkt {
                SeedLinkPacket::V3Info { has_more, payload } => {
                    let recs = read_mseed_records(&payload)?;
                    for r in recs {
                        if let Some(txt) = r.ascii_text {
                            combined.push_str(&txt);
                        }
                    }
                    if !has_more {
                        break;
                    }
                }
                SeedLinkPacket::V4Packet {
                    format: SeedLinkV4Format::InfoJson,
                    payload,
                    ..
                } => {
                    let s = core::str::from_utf8(&payload).map_err(|_| {
                        EarthSnailError::Format("non-UTF8 SeedLink v4 INFO JSON".into())
                    })?;
                    combined.push_str(s);
                    break;
                }
                SeedLinkPacket::V4Packet {
                    format: SeedLinkV4Format::InfoError,
                    payload,
                    ..
                } => {
                    let msg = String::from_utf8_lossy(&payload);
                    return Err(EarthSnailError::Network(format!(
                        "SeedLink v4 INFO error: {}",
                        msg.trim()
                    )));
                }
                SeedLinkPacket::End => break,
                _ => break,
            }
        }
        Ok(combined)
    }

    /// Reads the next streaming `SeedLinkPacket` (`Ok(None)` on clean EOF or `"END"`).
    pub fn read_next_packet(&mut self) -> Result<Option<SeedLinkPacket>> {
        let mut prefix = [0_u8; 3];
        if !read_exact_or_eof(&mut self.stream, &mut prefix)? {
            return Ok(None);
        }

        if &prefix == b"END" {
            return Ok(Some(SeedLinkPacket::End));
        }
        if prefix.starts_with(b"ER") {
            return Err(EarthSnailError::Network(
                "SeedLink server returned ERROR response".into(),
            ));
        }

        if prefix.starts_with(b"SL") {
            let mut full = alloc::vec![0_u8; SEEDLINK_V3_PACKET_BYTES];
            full[0..3].copy_from_slice(&prefix);
            self.stream.read_exact(&mut full[3..]).map_err(|e| {
                EarthSnailError::Network(format!("truncated SeedLink v3 packet read: {e}"))
            })?;
            let (pkt, _) = SeedLinkPacket::decode_frame(&full)?;
            return Ok(Some(pkt));
        }

        if prefix.starts_with(b"SE") {
            let mut fixed = [0_u8; SEEDLINK_V4_FIXED_HEADER_BYTES];
            fixed[0..3].copy_from_slice(&prefix);
            self.stream.read_exact(&mut fixed[3..]).map_err(|e| {
                EarthSnailError::Network(format!("truncated SeedLink v4 fixed header read: {e}"))
            })?;

            let payload_len = u32::from_le_bytes([fixed[4], fixed[5], fixed[6], fixed[7]]) as usize;
            if payload_len > MAX_SEEDLINK_V4_PAYLOAD {
                return Err(EarthSnailError::CapacityExceeded(format!(
                    "SeedLink v4 payload length {payload_len} exceeds safety limit"
                )));
            }
            let sid_len = fixed[16] as usize;
            let total = SEEDLINK_V4_FIXED_HEADER_BYTES + sid_len + payload_len;
            let mut full = alloc::vec![0_u8; total];
            full[..SEEDLINK_V4_FIXED_HEADER_BYTES].copy_from_slice(&fixed);
            self.stream
                .read_exact(&mut full[SEEDLINK_V4_FIXED_HEADER_BYTES..])
                .map_err(|e| {
                    EarthSnailError::Network(format!("truncated SeedLink v4 payload read: {e}"))
                })?;
            let (pkt, _) = SeedLinkPacket::decode_frame(&full)?;
            return Ok(Some(pkt));
        }

        Err(EarthSnailError::Format(format!(
            "unexpected SeedLink wire bytes [0x{:02X}, 0x{:02X}, 0x{:02X}]",
            prefix[0], prefix[1], prefix[2]
        )))
    }

    fn send_raw_cmd(&mut self, cmd: &str) -> Result<()> {
        let line = format!("{cmd}\r\n");
        self.stream
            .write_all(line.as_bytes())
            .and_then(|_| self.stream.flush())
            .map_err(|e| EarthSnailError::Network(format!("SeedLink command write failed: {e}")))
    }

    fn send_cmd_expect_ok(&mut self, cmd: &str) -> Result<()> {
        self.send_raw_cmd(cmd)?;
        let reply = self.read_crlf_line()?;
        if reply == "OK" || reply.starts_with("OK ") {
            Ok(())
        } else {
            Err(EarthSnailError::Network(format!(
                "SeedLink command '{cmd}' rejected by server: '{reply}'"
            )))
        }
    }

    fn read_crlf_line(&mut self) -> Result<String> {
        let mut buf = Vec::with_capacity(128);
        let mut byte = [0_u8; 1];
        loop {
            let n = self.stream.read(&mut byte).map_err(|e| {
                EarthSnailError::Network(format!("SeedLink handshake read failed: {e}"))
            })?;
            if n == 0 {
                return Err(EarthSnailError::Network(
                    "unexpected EOF during SeedLink handshake".into(),
                ));
            }
            if byte[0] == b'\n' {
                if buf.last() == Some(&b'\r') {
                    buf.pop();
                }
                let s = core::str::from_utf8(&buf).map_err(|_| {
                    EarthSnailError::Format("non-UTF8 SeedLink handshake response".into())
                })?;
                return Ok(String::from(s.trim()));
            }
            if buf.len() >= 4096 {
                return Err(EarthSnailError::CapacityExceeded(
                    "SeedLink handshake line exceeded 4 KiB".into(),
                ));
            }
            buf.push(byte[0]);
        }
    }
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0_usize;
    while filled < buf.len() {
        let n = r
            .read(&mut buf[filled..])
            .map_err(|e| EarthSnailError::Network(format!("SeedLink stream read failed: {e}")))?;
        if n == 0 {
            if filled == 0 {
                return Ok(false);
            }
            return Err(EarthSnailError::Format(
                "truncated SeedLink frame at EOF".into(),
            ));
        }
        filled += n;
    }
    Ok(true)
}

fn format_sl_v3_time(ts: UtcTimestamp) -> String {
    let (y, m, d, hh, mm, ss, _) = ts.to_ymd_hms_nano();
    format!("{y:04},{m:02},{d:02},{hh:02},{mm:02},{ss:02}")
}

fn format_sl_v4_time(ts: UtcTimestamp) -> String {
    ts.to_iso8601()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mseed::{write_mseed_trace_bytes, MseedWriteOptions};
    use crate::mseed3::{write_mseed3_trace_bytes, Mseed3WriteOptions};
    use earthsnail_core::{ChannelId, Trace, TraceStats};
    use std::collections::VecDeque;

    /// Deterministic in-memory mock transport recording client commands and replaying scripted server bytes.
    struct MockDuplex {
        rx: VecDeque<u8>,
        tx: Vec<u8>,
    }

    impl MockDuplex {
        fn new(server_script: Vec<u8>) -> Self {
            Self {
                rx: server_script.into_iter().collect(),
                tx: Vec::new(),
            }
        }
    }

    impl Read for MockDuplex {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.rx.len());
            for slot in &mut buf[..n] {
                *slot = self.rx.pop_front().unwrap();
            }
            Ok(n)
        }
    }

    impl Write for MockDuplex {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.tx.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn seedlink_v3_and_v4_state_machine_and_wire_roundtrip() {
        let t0: UtcTimestamp = "2024-05-01T12:00:00.123456Z".parse().unwrap();
        let ch = ChannelId::new("IU", "ANMO", "00", "BHZ");
        let tr = Trace::new(
            TraceStats::new(ch.clone(), t0, 40.0),
            (0..80).map(|i| (i * 11 - 200) as f64).collect(),
        );

        // 1. SeedLink v3 session with 512B MS2 packet + END
        let ms2 = write_mseed_trace_bytes(&tr, MseedWriteOptions::default()).unwrap();
        assert_eq!(ms2.len(), 512);
        let pkt_v3 = SeedLinkPacket::V3Data {
            sequence: 0x001A2B,
            payload: ms2,
        };
        let encoded_v3 = pkt_v3.encode_frame().unwrap();
        assert_eq!(encoded_v3.len(), 520);
        assert_eq!(&encoded_v3[0..8], b"SL001A2B");

        let mut script_v3 = Vec::new();
        script_v3.extend_from_slice(b"SeedLink v3.1 (2023.001)\r\nEarthScope DMC\r\n");
        script_v3.extend_from_slice(b"OK\r\nOK\r\nOK\r\n"); // STATION, SELECT, DATA
        script_v3.extend_from_slice(&encoded_v3);
        script_v3.extend_from_slice(b"END");

        let mut sl3 = SeedLinkClient::from_stream(MockDuplex::new(script_v3));
        let hello3 = sl3.say_hello().unwrap();
        assert!(!hello3.supports_v4);
        sl3.select_station("IU", "ANMO").unwrap();
        sl3.select_stream("00", "BHZ").unwrap();
        sl3.request_data(Some(0x1A2B)).unwrap();
        sl3.end_handshake().unwrap();

        let got_pkt3 = sl3.read_next_packet().unwrap().unwrap();
        assert_eq!(got_pkt3, pkt_v3);
        let decoded_st3 = got_pkt3.decode_waveforms().unwrap().unwrap();
        assert_eq!(decoded_st3[0].data, tr.data);
        assert_eq!(sl3.read_next_packet().unwrap(), Some(SeedLinkPacket::End));

        let sent3 = String::from_utf8(sl3.into_inner().tx).unwrap();
        assert_eq!(
            sent3,
            "HELLO\r\nSTATION ANMO IU\r\nSELECT 00BHZ.D\r\nDATA 001A2B\r\nEND\r\n"
        );

        // 2. SeedLink v4 session with `"SE"` MS3 packet + INFO JSON
        let ms3 = write_mseed3_trace_bytes(&tr, &Mseed3WriteOptions::default()).unwrap();
        let pkt_v4 = SeedLinkPacket::V4Packet {
            format: SeedLinkV4Format::MiniSeed3,
            subformat: 'D',
            sequence: 9_876_543_210,
            station_id: String::from("IU_ANMO"),
            payload: ms3,
        };
        let encoded_v4 = pkt_v4.encode_frame().unwrap();
        assert_eq!(&encoded_v4[0..4], b"SE3D");

        let mut script_v4 = Vec::new();
        script_v4.extend_from_slice(
            b"SeedLink v4.0 (2025.100) :: SLPROTO:4.0 CAP\r\nEarthScope Ringserver\r\n",
        );
        script_v4.extend_from_slice(b"OK\r\n"); // SLPROTO 4.0
        script_v4.extend_from_slice(b"OK\r\nOK\r\nOK\r\nOK\r\n"); // USERAGENT, STATION, SELECT, DATA ALL
        script_v4.extend_from_slice(&encoded_v4);

        let mut sl4 = SeedLinkClient::from_stream(MockDuplex::new(script_v4));
        let hello4 = sl4.say_hello().unwrap();
        assert!(hello4.supports_v4);
        assert!(sl4.is_v4());
        sl4.send_user_agent("EarthSnail/0.1").unwrap();
        sl4.select_station("IU", "ANMO").unwrap();
        sl4.select_stream("00", "BHZ").unwrap();
        sl4.request_data(Some(9_876_543_210)).unwrap();
        sl4.end_handshake().unwrap();

        let got_pkt4 = sl4.read_next_packet().unwrap().unwrap();
        assert_eq!(got_pkt4, pkt_v4);
        let decoded_st4 = got_pkt4.decode_waveforms().unwrap().unwrap();
        assert_eq!(decoded_st4[0].data, tr.data);
        assert_eq!(decoded_st4[0].stats.starttime, t0);
        assert_eq!(sl4.read_next_packet().unwrap(), None);

        let sent4 = String::from_utf8(sl4.into_inner().tx).unwrap();
        assert_eq!(
            sent4,
            "HELLO\r\nSLPROTO 4.0\r\nUSERAGENT EarthSnail/0.1\r\nSTATION IU_ANMO\r\nSELECT 00_B_H_Z\r\nDATA ALL 9876543210\r\nEND\r\n"
        );
    }
}
