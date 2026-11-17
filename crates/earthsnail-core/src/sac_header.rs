//! Complete 158-word (632-byte) SAC header (`SacHeader`) with 70 `f64` floats (with exact `f32`
//! `-12345.0` sentinel handling), 40 `i32` integers/enums/logicals, 24 8-byte character slots
//! (23 `K*` string fields with 16-char `KEVNM`), case-insensitive field lookup/mutation,
//! automatic `LCALDA` WGS84 `DIST`/`AZ`/`BAZ`/`GCARC` updates, and v6/v7 binary layout helpers.

use crate::error::{EarthSnailError, Result};
use crate::geodesics::gps2dist_azimuth;
use crate::math;
use crate::time::UtcTimestamp;
use alloc::format;
use alloc::string::String;

/// Total number of 32-bit words in a SAC binary header (`70 + 40 + 48 = 158`).
pub const SAC_HEADER_WORDS: usize = 158;
/// Total byte length of a standard SAC binary header (`158 * 4 = 632`).
pub const SAC_HEADER_BYTES: usize = 632;
/// Number of floating-point header words (`0..70`).
pub const SAC_FLOAT_COUNT: usize = 70;
/// Number of 32-bit integer/enum/logical header words (`0..40`).
pub const SAC_INT_COUNT: usize = 40;
/// Number of 8-byte character slots (`24 * 8 = 192` bytes; `KEVNM` spans slots `1..=2`).
pub const SAC_CHAR_SLOTS: usize = 24;
/// Number of `f64` words in a SAC v7 (`NVHDR == 7`) precision footer (`22 * 8 = 176` bytes).
pub const SAC_V7_FOOTER_DOUBLES: usize = 22;

/// SAC undefined float sentinel in 32-bit binary format (`-12345.0_f32`).
pub const SAC_FLOAT_UNDEF_F32: f32 = -12345.0_f32;
/// SAC undefined float sentinel in 64-bit memory representation (`-12345.0_f64`).
pub const SAC_FLOAT_UNDEF: f64 = -12345.0_f64;
/// SAC undefined integer/enum/logical sentinel (`-12345`).
pub const SAC_INT_UNDEF: i32 = -12345;
/// SAC undefined character string sentinel (`"-12345"`).
pub const SAC_STR_UNDEF: &str = "-12345";

/// Canonical lowercase SAC names for all 70 float words in binary layout order (`0..70`).
pub const SAC_FLOAT_NAMES: [&str; SAC_FLOAT_COUNT] = [
    "delta",
    "depmin",
    "depmax",
    "scale",
    "odelta",
    "b",
    "e",
    "o",
    "a",
    "internal1",
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "t9",
    "f",
    "resp0",
    "resp1",
    "resp2",
    "resp3",
    "resp4",
    "resp5",
    "resp6",
    "resp7",
    "resp8",
    "resp9",
    "stla",
    "stlo",
    "stel",
    "stdp",
    "evla",
    "evlo",
    "evel",
    "evdp",
    "mag",
    "user0",
    "user1",
    "user2",
    "user3",
    "user4",
    "user5",
    "user6",
    "user7",
    "user8",
    "user9",
    "dist",
    "az",
    "baz",
    "gcarc",
    "internal2",
    "internal3",
    "depmen",
    "cmpaz",
    "cmpinc",
    "xminimum",
    "xmaximum",
    "yminimum",
    "ymaximum",
    "unused6",
    "unused7",
    "unused8",
    "unused9",
    "unused10",
    "unused11",
    "unused12",
];

/// Canonical lowercase SAC names for all 40 integer/enum/logical words in binary layout order (`0..40`).
pub const SAC_INT_NAMES: [&str; SAC_INT_COUNT] = [
    "nzyear",
    "nzjday",
    "nzhour",
    "nzmin",
    "nzsec",
    "nzmsec",
    "nvhdr",
    "norid",
    "nevid",
    "npts",
    "internal4",
    "nwfid",
    "nxsize",
    "nysize",
    "unused15",
    "iftype",
    "idep",
    "iztype",
    "unused16",
    "iinst",
    "istreg",
    "ievreg",
    "ievtyp",
    "iqual",
    "isynth",
    "imagtyp",
    "imagsrc",
    "unused19",
    "unused20",
    "unused21",
    "unused22",
    "unused23",
    "unused24",
    "unused25",
    "unused26",
    "leven",
    "lpspol",
    "lovrok",
    "lcalda",
    "unused27",
];

/// Canonical lowercase SAC names for all 23 logical `K*` string fields (`kevnm` is 16 chars; others 8 chars).
pub const SAC_STR_NAMES: [&str; 23] = [
    "kstnm", "kevnm", "khole", "ko", "ka", "kt0", "kt1", "kt2", "kt3", "kt4", "kt5", "kt6", "kt7",
    "kt8", "kt9", "kf", "kuser0", "kuser1", "kuser2", "kcmpnm", "knetwk", "kdatrd", "kinst",
];

/// Endianness of a binary SAC file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SacEndian {
    /// Little-endian (x86_64, ARM64, WASM32).
    Little,
    /// Big-endian (historic SPARC / PowerPC / network SAC archives).
    Big,
}

/// SAC `IFTYPE` file type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SacFileType {
    /// Time series file (`ITIME = 1`).
    TimeSeries = 1,
    /// Spectral real-imaginary file (`IRLIM = 2`).
    SpectralRealImag = 2,
    /// Spectral amplitude-phase file (`IAMPH = 3`).
    SpectralAmpPhase = 3,
    /// General X-Y data (`IXY = 4`).
    GeneralXY = 4,
    /// General X-Y-Z 3D data (`IXYZ = 51`).
    GeneralXYZ = 51,
}

impl SacFileType {
    /// Converts a raw SAC integer code to `SacFileType`.
    pub const fn from_i32(code: i32) -> Option<Self> {
        match code {
            1 => Some(Self::TimeSeries),
            2 => Some(Self::SpectralRealImag),
            3 => Some(Self::SpectralAmpPhase),
            4 => Some(Self::GeneralXY),
            51 => Some(Self::GeneralXYZ),
            _ => None,
        }
    }
}

/// SAC `IDEP` dependent-variable physical unit enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SacDepType {
    /// Unknown units (`IUNKN = 5`).
    Unknown = 5,
    /// Displacement in nanometers (`IDISP = 6`).
    DisplacementNm = 6,
    /// Velocity in nm/s (`IVEL = 7`).
    VelocityNmPerSec = 7,
    /// Acceleration in nm/s^2 (`IACC = 8`).
    AccelerationNmPerSec2 = 8,
    /// Volts (`IVOLTS = 50`).
    Volts = 50,
}

impl SacDepType {
    /// Converts a raw SAC integer code to `SacDepType`.
    pub const fn from_i32(code: i32) -> Option<Self> {
        match code {
            5 => Some(Self::Unknown),
            6 => Some(Self::DisplacementNm),
            7 => Some(Self::VelocityNmPerSec),
            8 => Some(Self::AccelerationNmPerSec2),
            50 => Some(Self::Volts),
            _ => None,
        }
    }
}

/// SAC `IZTYPE` reference zero-time equivalence enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SacZeroTime {
    /// Unknown (`IUNKN = 5`).
    Unknown = 5,
    /// Begin time `B` (`IB = 9`).
    Begin = 9,
    /// Midnight of GMT day (`IDAY = 10`).
    Day = 10,
    /// Event origin time `O` (`IO = 11`).
    Origin = 11,
    /// First arrival time `A` (`IA = 12`).
    FirstArrival = 12,
    /// User pick `T0` (`IT0 = 13`).
    T0 = 13,
    /// User pick `T1` (`IT1 = 14`).
    T1 = 14,
    /// User pick `T2` (`IT2 = 15`).
    T2 = 15,
    /// User pick `T3` (`IT3 = 16`).
    T3 = 16,
    /// User pick `T4` (`IT4 = 17`).
    T4 = 17,
    /// User pick `T5` (`IT5 = 18`).
    T5 = 18,
    /// User pick `T6` (`IT6 = 19`).
    T6 = 19,
    /// User pick `T7` (`IT7 = 20`).
    T7 = 20,
    /// User pick `T8` (`IT8 = 21`).
    T8 = 21,
    /// User pick `T9` (`IT9 = 22`).
    T9 = 22,
}

/// Returns `true` if `v` matches the SAC `-12345.0` float undefined sentinel (in `f64` or `f32`).
#[inline]
pub fn is_sac_float_undef(v: f64) -> bool {
    !v.is_finite() || (v as f32) == SAC_FLOAT_UNDEF_F32 || (v - SAC_FLOAT_UNDEF).abs() < 1e-4
}

/// Returns `true` if `v` equals the SAC `-12345` integer undefined sentinel.
#[inline]
pub const fn is_sac_int_undef(v: i32) -> bool {
    v == SAC_INT_UNDEF
}

/// Returns `true` if `s` represents an undefined SAC character field (`""`, `"-12345"`, or `"-12345  -12345"`).
#[inline]
pub fn is_sac_str_undef(s: &str) -> bool {
    let t = s.trim_matches([' ', '\0']);
    t.is_empty() || t == SAC_STR_UNDEF || t == "-12345  -12345"
}

/// Complete 158-word SAC header storing all 70 float fields (promoted to `f64`), 40 `i32` fields,
/// and all `K*` string fields.
#[derive(Debug, Clone, PartialEq)]
pub struct SacHeader {
    // --- 70 Float fields (words 0..69) ---
    pub delta: f64,
    pub depmin: f64,
    pub depmax: f64,
    pub scale: f64,
    pub odelta: f64,
    pub b: f64,
    pub e: f64,
    pub o: f64,
    pub a: f64,
    pub internal1: f64,
    pub t0: f64,
    pub t1: f64,
    pub t2: f64,
    pub t3: f64,
    pub t4: f64,
    pub t5: f64,
    pub t6: f64,
    pub t7: f64,
    pub t8: f64,
    pub t9: f64,
    pub f: f64,
    pub resp0: f64,
    pub resp1: f64,
    pub resp2: f64,
    pub resp3: f64,
    pub resp4: f64,
    pub resp5: f64,
    pub resp6: f64,
    pub resp7: f64,
    pub resp8: f64,
    pub resp9: f64,
    pub stla: f64,
    pub stlo: f64,
    pub stel: f64,
    pub stdp: f64,
    pub evla: f64,
    pub evlo: f64,
    pub evel: f64,
    pub evdp: f64,
    pub mag: f64,
    pub user0: f64,
    pub user1: f64,
    pub user2: f64,
    pub user3: f64,
    pub user4: f64,
    pub user5: f64,
    pub user6: f64,
    pub user7: f64,
    pub user8: f64,
    pub user9: f64,
    pub dist: f64,
    pub az: f64,
    pub baz: f64,
    pub gcarc: f64,
    pub internal2: f64,
    pub internal3: f64,
    pub depmen: f64,
    pub cmpaz: f64,
    pub cmpinc: f64,
    pub xminimum: f64,
    pub xmaximum: f64,
    pub yminimum: f64,
    pub ymaximum: f64,
    pub unused6: f64,
    pub unused7: f64,
    pub unused8: f64,
    pub unused9: f64,
    pub unused10: f64,
    pub unused11: f64,
    pub unused12: f64,

    // --- 40 Integer / Enum / Logical fields (words 70..109) ---
    pub nzyear: i32,
    pub nzjday: i32,
    pub nzhour: i32,
    pub nzmin: i32,
    pub nzsec: i32,
    pub nzmsec: i32,
    pub nvhdr: i32,
    pub norid: i32,
    pub nevid: i32,
    pub npts: i32,
    pub internal4: i32,
    pub nwfid: i32,
    pub nxsize: i32,
    pub nysize: i32,
    pub unused15: i32,
    pub iftype: i32,
    pub idep: i32,
    pub iztype: i32,
    pub unused16: i32,
    pub iinst: i32,
    pub istreg: i32,
    pub ievreg: i32,
    pub ievtyp: i32,
    pub iqual: i32,
    pub isynth: i32,
    pub imagtyp: i32,
    pub imagsrc: i32,
    pub unused19: i32,
    pub unused20: i32,
    pub unused21: i32,
    pub unused22: i32,
    pub unused23: i32,
    pub unused24: i32,
    pub unused25: i32,
    pub unused26: i32,
    pub leven: i32,
    pub lpspol: i32,
    pub lovrok: i32,
    pub lcalda: i32,
    pub unused27: i32,

    // --- 23 Named String fields (24 8-byte slots, words 110..157) ---
    pub kstnm: String,
    /// 16-character event name spanning character slots 1 and 2 (`bytes 448..464`).
    pub kevnm: String,
    pub khole: String,
    pub ko: String,
    pub ka: String,
    pub kt0: String,
    pub kt1: String,
    pub kt2: String,
    pub kt3: String,
    pub kt4: String,
    pub kt5: String,
    pub kt6: String,
    pub kt7: String,
    pub kt8: String,
    pub kt9: String,
    pub kf: String,
    pub kuser0: String,
    pub kuser1: String,
    pub kuser2: String,
    pub kcmpnm: String,
    pub knetwk: String,
    pub kdatrd: String,
    pub kinst: String,
}

impl Default for SacHeader {
    fn default() -> Self {
        Self::new()
    }
}

impl SacHeader {
    /// Creates a new SAC header with standard SAC v6 defaults (`nvhdr = 6`, `iftype = ITIME`,
    /// `leven = 1`, `lovrok = 1`, `lcalda = 1`) and all other fields set to `-12345` sentinels.
    pub fn new() -> Self {
        let mut hdr = Self::empty_raw();
        hdr.nvhdr = 6;
        hdr.iftype = SacFileType::TimeSeries as i32;
        hdr.idep = SacDepType::Unknown as i32;
        hdr.iztype = SacZeroTime::Begin as i32;
        hdr.leven = 1;
        hdr.lovrok = 1;
        hdr.lcalda = 1;
        hdr
    }

    /// Creates a completely raw SAC header where every single word is initialized to `-12345`.
    pub fn empty_raw() -> Self {
        let u = SAC_FLOAT_UNDEF;
        let iu = SAC_INT_UNDEF;
        let su = || String::from(SAC_STR_UNDEF);
        Self {
            delta: u,
            depmin: u,
            depmax: u,
            scale: u,
            odelta: u,
            b: u,
            e: u,
            o: u,
            a: u,
            internal1: u,
            t0: u,
            t1: u,
            t2: u,
            t3: u,
            t4: u,
            t5: u,
            t6: u,
            t7: u,
            t8: u,
            t9: u,
            f: u,
            resp0: u,
            resp1: u,
            resp2: u,
            resp3: u,
            resp4: u,
            resp5: u,
            resp6: u,
            resp7: u,
            resp8: u,
            resp9: u,
            stla: u,
            stlo: u,
            stel: u,
            stdp: u,
            evla: u,
            evlo: u,
            evel: u,
            evdp: u,
            mag: u,
            user0: u,
            user1: u,
            user2: u,
            user3: u,
            user4: u,
            user5: u,
            user6: u,
            user7: u,
            user8: u,
            user9: u,
            dist: u,
            az: u,
            baz: u,
            gcarc: u,
            internal2: u,
            internal3: u,
            depmen: u,
            cmpaz: u,
            cmpinc: u,
            xminimum: u,
            xmaximum: u,
            yminimum: u,
            ymaximum: u,
            unused6: u,
            unused7: u,
            unused8: u,
            unused9: u,
            unused10: u,
            unused11: u,
            unused12: u,

            nzyear: iu,
            nzjday: iu,
            nzhour: iu,
            nzmin: iu,
            nzsec: iu,
            nzmsec: iu,
            nvhdr: iu,
            norid: iu,
            nevid: iu,
            npts: iu,
            internal4: iu,
            nwfid: iu,
            nxsize: iu,
            nysize: iu,
            unused15: iu,
            iftype: iu,
            idep: iu,
            iztype: iu,
            unused16: iu,
            iinst: iu,
            istreg: iu,
            ievreg: iu,
            ievtyp: iu,
            iqual: iu,
            isynth: iu,
            imagtyp: iu,
            imagsrc: iu,
            unused19: iu,
            unused20: iu,
            unused21: iu,
            unused22: iu,
            unused23: iu,
            unused24: iu,
            unused25: iu,
            unused26: iu,
            leven: iu,
            lpspol: iu,
            lovrok: iu,
            lcalda: iu,
            unused27: iu,

            kstnm: su(),
            kevnm: su(),
            khole: su(),
            ko: su(),
            ka: su(),
            kt0: su(),
            kt1: su(),
            kt2: su(),
            kt3: su(),
            kt4: su(),
            kt5: su(),
            kt6: su(),
            kt7: su(),
            kt8: su(),
            kt9: su(),
            kf: su(),
            kuser0: su(),
            kuser1: su(),
            kuser2: su(),
            kcmpnm: su(),
            knetwk: su(),
            kdatrd: su(),
            kinst: su(),
        }
    }

    /// Case-insensitive lookup of a float header field, returning `None` if unknown or `-12345.0`.
    pub fn get_float(&self, name: &str) -> Option<f64> {
        let idx = float_field_index(name)?;
        let val = self.float_by_index(idx);
        if is_sac_float_undef(val) {
            None
        } else {
            Some(val)
        }
    }

    /// Case-insensitive mutation of a float header field. Automatically recomputes `dist`, `az`,
    /// `baz`, and `gcarc` when `lcalda == 1` and `stla`/`stlo`/`evla`/`evlo` are modified.
    pub fn set_float(&mut self, name: &str, val: f64) -> Result<()> {
        if !val.is_finite() {
            return Err(EarthSnailError::Header(format!(
                "non-finite value for SAC float header '{name}'"
            )));
        }
        let idx = float_field_index(name)
            .ok_or_else(|| EarthSnailError::Header(format!("unknown SAC float header '{name}'")))?;
        *self.float_by_index_mut(idx) = val;
        if matches!(idx, 31 | 32 | 35 | 36) && self.lcalda == 1 {
            self.update_dist_az_baz_gcarc();
        }
        Ok(())
    }

    /// Resets a float header field to the SAC `-12345.0` undefined sentinel.
    pub fn clear_float(&mut self, name: &str) -> Result<()> {
        let idx = float_field_index(name)
            .ok_or_else(|| EarthSnailError::Header(format!("unknown SAC float header '{name}'")))?;
        *self.float_by_index_mut(idx) = SAC_FLOAT_UNDEF;
        Ok(())
    }

    /// Case-insensitive lookup of an integer/enum/logical header field, returning `None` if unknown or `-12345`.
    pub fn get_int(&self, name: &str) -> Option<i32> {
        let idx = int_field_index(name)?;
        let val = self.int_by_index(idx);
        if is_sac_int_undef(val) {
            None
        } else {
            Some(val)
        }
    }

    /// Case-insensitive mutation of an integer/enum/logical header field. Toggling `lcalda` to `1`
    /// triggers `update_dist_az_baz_gcarc()`.
    pub fn set_int(&mut self, name: &str, val: i32) -> Result<()> {
        let idx = int_field_index(name).ok_or_else(|| {
            EarthSnailError::Header(format!("unknown SAC integer header '{name}'"))
        })?;
        *self.int_by_index_mut(idx) = val;
        if idx == 38 && val == 1 {
            self.update_dist_az_baz_gcarc();
        }
        Ok(())
    }

    /// Resets an integer header field to the SAC `-12345` undefined sentinel.
    pub fn clear_int(&mut self, name: &str) -> Result<()> {
        let idx = int_field_index(name).ok_or_else(|| {
            EarthSnailError::Header(format!("unknown SAC integer header '{name}'"))
        })?;
        *self.int_by_index_mut(idx) = SAC_INT_UNDEF;
        Ok(())
    }

    /// Reads one of the 5 logical SAC flags (`leven`, `lpspol`, `lovrok`, `lcalda`, `unused27`) as `Option<bool>`.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        let v = self.get_int(name)?;
        match v {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    /// Sets a logical SAC flag (`true -> 1`, `false -> 0`), triggering `update_dist_az_baz_gcarc()` if `lcalda` becomes `true`.
    pub fn set_bool(&mut self, name: &str, val: bool) -> Result<()> {
        self.set_int(name, i32::from(val))
    }

    /// Case-insensitive lookup of a `K*` string header field, returning `None` if unknown or `"-12345"`.
    pub fn get_str(&self, name: &str) -> Option<&str> {
        if name.eq_ignore_ascii_case("kevnm1") {
            if is_sac_str_undef(&self.kevnm) || self.kevnm.len() <= 8 {
                return None;
            }
            let tail = self.kevnm[8..].trim_end_matches([' ', '\0']);
            return if is_sac_str_undef(tail) {
                None
            } else {
                Some(tail)
            };
        }
        let idx = str_field_index(name)?;
        let s = self.str_by_index(idx);
        if is_sac_str_undef(s) {
            None
        } else {
            Some(s.trim_end_matches([' ', '\0']))
        }
    }

    /// Case-insensitive mutation of a `K*` string header field (up to 16 ASCII chars for `kevnm`,
    /// 8 ASCII chars for all other `K*` fields).
    pub fn set_str(&mut self, name: &str, val: &str) -> Result<()> {
        if name.eq_ignore_ascii_case("kevnm1") {
            let mut prefix = if is_sac_str_undef(&self.kevnm) {
                String::from("        ")
            } else {
                let mut p = truncate_ascii(&self.kevnm, 8);
                while p.len() < 8 {
                    p.push(' ');
                }
                p
            };
            let suffix = truncate_ascii(val.trim_end_matches([' ', '\0']), 8);
            prefix.push_str(&suffix);
            self.kevnm = prefix.trim_end_matches(' ').into();
            if self.kevnm.is_empty() {
                self.kevnm = String::from(SAC_STR_UNDEF);
            }
            return Ok(());
        }
        let idx = str_field_index(name).ok_or_else(|| {
            EarthSnailError::Header(format!("unknown SAC string header '{name}'"))
        })?;
        let max_len = if idx == 1 { 16 } else { 8 };
        let cleaned = val.trim_end_matches([' ', '\0']);
        let stored = if is_sac_str_undef(cleaned) {
            String::from(SAC_STR_UNDEF)
        } else {
            truncate_ascii(cleaned, max_len)
        };
        *self.str_by_index_mut(idx) = stored;
        Ok(())
    }

    /// Resets a `K*` string header field to `"-12345"`.
    pub fn clear_str(&mut self, name: &str) -> Result<()> {
        let idx = str_field_index(name).ok_or_else(|| {
            EarthSnailError::Header(format!("unknown SAC string header '{name}'"))
        })?;
        *self.str_by_index_mut(idx) = String::from(SAC_STR_UNDEF);
        Ok(())
    }

    /// Automatically recalculates `dist` (km), `az` (deg), `baz` (deg), and `gcarc` (deg) using
    /// exact WGS84 geodesics when `lcalda == 1` and `stla`, `stlo`, `evla`, `evlo` are all defined
    /// and within valid geographic latitude bounds `[-90, 90]`.
    pub fn update_dist_az_baz_gcarc(&mut self) -> bool {
        if self.lcalda != 1 {
            return false;
        }
        self.force_update_dist_az_baz_gcarc()
    }

    /// Recalculates `dist`, `az`, `baz`, and `gcarc` from `(evla, evlo)` -> `(stla, stlo)`
    /// regardless of `lcalda`, returning `false` if any coordinate is undefined or out of bounds.
    pub fn force_update_dist_az_baz_gcarc(&mut self) -> bool {
        if is_sac_float_undef(self.stla)
            || is_sac_float_undef(self.stlo)
            || is_sac_float_undef(self.evla)
            || is_sac_float_undef(self.evlo)
        {
            return false;
        }
        if self.stla.abs() > 90.0 || self.evla.abs() > 90.0 {
            return false;
        }
        let geo = gps2dist_azimuth(self.evla, self.evlo, self.stla, self.stlo);
        self.dist = geo.distance_m * 1e-3;
        self.az = geo.azimuth_deg;
        self.baz = geo.back_azimuth_deg;
        self.gcarc = geo.gcarc_deg;
        true
    }

    /// Returns the SAC GMT reference zero-time (`NZYEAR`, `NZJDAY`, `NZHOUR`, `NZMIN`, `NZSEC`, `NZMSEC`)
    /// if all 6 integer time components are defined and valid.
    pub fn reference_time(&self) -> Option<UtcTimestamp> {
        if is_sac_int_undef(self.nzyear)
            || is_sac_int_undef(self.nzjday)
            || is_sac_int_undef(self.nzhour)
            || is_sac_int_undef(self.nzmin)
            || is_sac_int_undef(self.nzsec)
            || is_sac_int_undef(self.nzmsec)
        {
            return None;
        }
        if self.nzjday < 1 || self.nzhour < 0 || self.nzmin < 0 || self.nzsec < 0 {
            return None;
        }
        // Note: nzmsec in SAC can occasionally be negative or >= 1000 in unnormalized files;
        // add it via signed millisecond offset from NZYEAR..NZSEC!
        let base = UtcTimestamp::from_year_jday_hms_nano(
            self.nzyear,
            self.nzjday as u16,
            self.nzhour as u8,
            self.nzmin as u8,
            self.nzsec as u8,
            0,
        )
        .ok()?;
        Some(base.add_nanos((self.nzmsec as i128) * UtcTimestamp::NANOS_PER_MILLI))
    }

    /// Sets `NZYEAR..NZMSEC` to the millisecond-floored `ts` and returns any sub-millisecond
    /// remainder in seconds (`[0.0, 0.001)`), allowing lossless nanosecond round-trip via `B`.
    pub fn set_reference_time(&mut self, ts: UtcTimestamp) -> f64 {
        let (year, jday, hour, min, sec, nano) = ts.to_year_jday_hms_nano();
        let msec = nano / 1_000_000;
        let sub_ms_nanos = nano % 1_000_000;
        self.nzyear = year;
        self.nzjday = jday as i32;
        self.nzhour = hour as i32;
        self.nzmin = min as i32;
        self.nzsec = sec as i32;
        self.nzmsec = msec as i32;
        (sub_ms_nanos as f64) * 1e-9
    }

    /// Returns the absolute UTC timestamp of the first sample (`reference_time + B`, or `reference_time` if `B` is undefined).
    pub fn starttime(&self) -> Option<UtcTimestamp> {
        let reftime = self.reference_time()?;
        if is_sac_float_undef(self.b) {
            Some(reftime)
        } else {
            reftime.add_secs_f64(self.b).ok()
        }
    }

    /// Sets `NZYEAR..NZMSEC`, `IZTYPE = IB`, `B` (absorbing sub-millisecond nanoseconds!), and `E`.
    pub fn set_starttime_and_span(&mut self, starttime: UtcTimestamp, npts: usize, delta_s: f64) {
        let sub_ms_s = self.set_reference_time(starttime);
        self.iztype = SacZeroTime::Begin as i32;
        self.b = sub_ms_s;
        self.npts = npts as i32;
        self.delta = delta_s;
        if npts > 0 && delta_s > 0.0 && delta_s.is_finite() {
            self.e = self.b + ((npts - 1) as f64) * delta_s;
        } else {
            self.e = self.b;
        }
    }

    /// Accesses float word `0..70` by binary table index.
    pub fn float_by_index(&self, idx: usize) -> f64 {
        match idx {
            0 => self.delta,
            1 => self.depmin,
            2 => self.depmax,
            3 => self.scale,
            4 => self.odelta,
            5 => self.b,
            6 => self.e,
            7 => self.o,
            8 => self.a,
            9 => self.internal1,
            10 => self.t0,
            11 => self.t1,
            12 => self.t2,
            13 => self.t3,
            14 => self.t4,
            15 => self.t5,
            16 => self.t6,
            17 => self.t7,
            18 => self.t8,
            19 => self.t9,
            20 => self.f,
            21 => self.resp0,
            22 => self.resp1,
            23 => self.resp2,
            24 => self.resp3,
            25 => self.resp4,
            26 => self.resp5,
            27 => self.resp6,
            28 => self.resp7,
            29 => self.resp8,
            30 => self.resp9,
            31 => self.stla,
            32 => self.stlo,
            33 => self.stel,
            34 => self.stdp,
            35 => self.evla,
            36 => self.evlo,
            37 => self.evel,
            38 => self.evdp,
            39 => self.mag,
            40 => self.user0,
            41 => self.user1,
            42 => self.user2,
            43 => self.user3,
            44 => self.user4,
            45 => self.user5,
            46 => self.user6,
            47 => self.user7,
            48 => self.user8,
            49 => self.user9,
            50 => self.dist,
            51 => self.az,
            52 => self.baz,
            53 => self.gcarc,
            54 => self.internal2,
            55 => self.internal3,
            56 => self.depmen,
            57 => self.cmpaz,
            58 => self.cmpinc,
            59 => self.xminimum,
            60 => self.xmaximum,
            61 => self.yminimum,
            62 => self.ymaximum,
            63 => self.unused6,
            64 => self.unused7,
            65 => self.unused8,
            66 => self.unused9,
            67 => self.unused10,
            68 => self.unused11,
            69 => self.unused12,
            _ => panic!("SAC float index {idx} out of bounds 0..70"),
        }
    }

    /// Mutably accesses float word `0..70` by binary table index.
    pub fn float_by_index_mut(&mut self, idx: usize) -> &mut f64 {
        match idx {
            0 => &mut self.delta,
            1 => &mut self.depmin,
            2 => &mut self.depmax,
            3 => &mut self.scale,
            4 => &mut self.odelta,
            5 => &mut self.b,
            6 => &mut self.e,
            7 => &mut self.o,
            8 => &mut self.a,
            9 => &mut self.internal1,
            10 => &mut self.t0,
            11 => &mut self.t1,
            12 => &mut self.t2,
            13 => &mut self.t3,
            14 => &mut self.t4,
            15 => &mut self.t5,
            16 => &mut self.t6,
            17 => &mut self.t7,
            18 => &mut self.t8,
            19 => &mut self.t9,
            20 => &mut self.f,
            21 => &mut self.resp0,
            22 => &mut self.resp1,
            23 => &mut self.resp2,
            24 => &mut self.resp3,
            25 => &mut self.resp4,
            26 => &mut self.resp5,
            27 => &mut self.resp6,
            28 => &mut self.resp7,
            29 => &mut self.resp8,
            30 => &mut self.resp9,
            31 => &mut self.stla,
            32 => &mut self.stlo,
            33 => &mut self.stel,
            34 => &mut self.stdp,
            35 => &mut self.evla,
            36 => &mut self.evlo,
            37 => &mut self.evel,
            38 => &mut self.evdp,
            39 => &mut self.mag,
            40 => &mut self.user0,
            41 => &mut self.user1,
            42 => &mut self.user2,
            43 => &mut self.user3,
            44 => &mut self.user4,
            45 => &mut self.user5,
            46 => &mut self.user6,
            47 => &mut self.user7,
            48 => &mut self.user8,
            49 => &mut self.user9,
            50 => &mut self.dist,
            51 => &mut self.az,
            52 => &mut self.baz,
            53 => &mut self.gcarc,
            54 => &mut self.internal2,
            55 => &mut self.internal3,
            56 => &mut self.depmen,
            57 => &mut self.cmpaz,
            58 => &mut self.cmpinc,
            59 => &mut self.xminimum,
            60 => &mut self.xmaximum,
            61 => &mut self.yminimum,
            62 => &mut self.ymaximum,
            63 => &mut self.unused6,
            64 => &mut self.unused7,
            65 => &mut self.unused8,
            66 => &mut self.unused9,
            67 => &mut self.unused10,
            68 => &mut self.unused11,
            69 => &mut self.unused12,
            _ => panic!("SAC float index {idx} out of bounds 0..70"),
        }
    }

    /// Accesses integer word `0..40` by binary table index.
    pub fn int_by_index(&self, idx: usize) -> i32 {
        match idx {
            0 => self.nzyear,
            1 => self.nzjday,
            2 => self.nzhour,
            3 => self.nzmin,
            4 => self.nzsec,
            5 => self.nzmsec,
            6 => self.nvhdr,
            7 => self.norid,
            8 => self.nevid,
            9 => self.npts,
            10 => self.internal4,
            11 => self.nwfid,
            12 => self.nxsize,
            13 => self.nysize,
            14 => self.unused15,
            15 => self.iftype,
            16 => self.idep,
            17 => self.iztype,
            18 => self.unused16,
            19 => self.iinst,
            20 => self.istreg,
            21 => self.ievreg,
            22 => self.ievtyp,
            23 => self.iqual,
            24 => self.isynth,
            25 => self.imagtyp,
            26 => self.imagsrc,
            27 => self.unused19,
            28 => self.unused20,
            29 => self.unused21,
            30 => self.unused22,
            31 => self.unused23,
            32 => self.unused24,
            33 => self.unused25,
            34 => self.unused26,
            35 => self.leven,
            36 => self.lpspol,
            37 => self.lovrok,
            38 => self.lcalda,
            39 => self.unused27,
            _ => panic!("SAC int index {idx} out of bounds 0..40"),
        }
    }

    /// Mutably accesses integer word `0..40` by binary table index.
    pub fn int_by_index_mut(&mut self, idx: usize) -> &mut i32 {
        match idx {
            0 => &mut self.nzyear,
            1 => &mut self.nzjday,
            2 => &mut self.nzhour,
            3 => &mut self.nzmin,
            4 => &mut self.nzsec,
            5 => &mut self.nzmsec,
            6 => &mut self.nvhdr,
            7 => &mut self.norid,
            8 => &mut self.nevid,
            9 => &mut self.npts,
            10 => &mut self.internal4,
            11 => &mut self.nwfid,
            12 => &mut self.nxsize,
            13 => &mut self.nysize,
            14 => &mut self.unused15,
            15 => &mut self.iftype,
            16 => &mut self.idep,
            17 => &mut self.iztype,
            18 => &mut self.unused16,
            19 => &mut self.iinst,
            20 => &mut self.istreg,
            21 => &mut self.ievreg,
            22 => &mut self.ievtyp,
            23 => &mut self.iqual,
            24 => &mut self.isynth,
            25 => &mut self.imagtyp,
            26 => &mut self.imagsrc,
            27 => &mut self.unused19,
            28 => &mut self.unused20,
            29 => &mut self.unused21,
            30 => &mut self.unused22,
            31 => &mut self.unused23,
            32 => &mut self.unused24,
            33 => &mut self.unused25,
            34 => &mut self.unused26,
            35 => &mut self.leven,
            36 => &mut self.lpspol,
            37 => &mut self.lovrok,
            38 => &mut self.lcalda,
            39 => &mut self.unused27,
            _ => panic!("SAC int index {idx} out of bounds 0..40"),
        }
    }

    /// Accesses string field `0..23` by table index.
    pub fn str_by_index(&self, idx: usize) -> &str {
        match idx {
            0 => &self.kstnm,
            1 => &self.kevnm,
            2 => &self.khole,
            3 => &self.ko,
            4 => &self.ka,
            5 => &self.kt0,
            6 => &self.kt1,
            7 => &self.kt2,
            8 => &self.kt3,
            9 => &self.kt4,
            10 => &self.kt5,
            11 => &self.kt6,
            12 => &self.kt7,
            13 => &self.kt8,
            14 => &self.kt9,
            15 => &self.kf,
            16 => &self.kuser0,
            17 => &self.kuser1,
            18 => &self.kuser2,
            19 => &self.kcmpnm,
            20 => &self.knetwk,
            21 => &self.kdatrd,
            22 => &self.kinst,
            _ => panic!("SAC str index {idx} out of bounds 0..23"),
        }
    }

    /// Mutably accesses string field `0..23` by table index.
    pub fn str_by_index_mut(&mut self, idx: usize) -> &mut String {
        match idx {
            0 => &mut self.kstnm,
            1 => &mut self.kevnm,
            2 => &mut self.khole,
            3 => &mut self.ko,
            4 => &mut self.ka,
            5 => &mut self.kt0,
            6 => &mut self.kt1,
            7 => &mut self.kt2,
            8 => &mut self.kt3,
            9 => &mut self.kt4,
            10 => &mut self.kt5,
            11 => &mut self.kt6,
            12 => &mut self.kt7,
            13 => &mut self.kt8,
            14 => &mut self.kt9,
            15 => &mut self.kf,
            16 => &mut self.kuser0,
            17 => &mut self.kuser1,
            18 => &mut self.kuser2,
            19 => &mut self.kcmpnm,
            20 => &mut self.knetwk,
            21 => &mut self.kdatrd,
            22 => &mut self.kinst,
            _ => panic!("SAC str index {idx} out of bounds 0..23"),
        }
    }

    /// Exports all 70 float words as `f32` values, preserving exact `-12345.0_f32` sentinels.
    pub fn to_float_words(&self) -> [f32; SAC_FLOAT_COUNT] {
        let mut out = [SAC_FLOAT_UNDEF_F32; SAC_FLOAT_COUNT];
        for (i, slot) in out.iter_mut().enumerate() {
            let v = self.float_by_index(i);
            *slot = if is_sac_float_undef(v) {
                SAC_FLOAT_UNDEF_F32
            } else {
                v as f32
            };
        }
        out
    }

    /// Populates all 70 float words from raw `f32` values, promoting `-12345.0_f32` to `SAC_FLOAT_UNDEF`.
    pub fn from_float_words(&mut self, words: &[f32; SAC_FLOAT_COUNT]) {
        for (i, &w) in words.iter().enumerate() {
            *self.float_by_index_mut(i) = if w == SAC_FLOAT_UNDEF_F32 {
                SAC_FLOAT_UNDEF
            } else {
                w as f64
            };
        }
    }

    /// Exports all 40 integer/enum/logical words.
    pub fn to_int_words(&self) -> [i32; SAC_INT_COUNT] {
        let mut out = [SAC_INT_UNDEF; SAC_INT_COUNT];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.int_by_index(i);
        }
        out
    }

    /// Populates all 40 integer/enum/logical words.
    pub fn from_int_words(&mut self, words: &[i32; SAC_INT_COUNT]) {
        for (i, &w) in words.iter().enumerate() {
            *self.int_by_index_mut(i) = w;
        }
    }

    /// Encodes all 23 `K*` string fields into the 192-byte ASCII space-padded SAC character block.
    pub fn to_string_bytes(&self) -> [u8; 192] {
        let mut buf = [b' '; 192];
        write_padded_ascii(&mut buf[0..8], &self.kstnm);
        write_padded_ascii(&mut buf[8..24], &self.kevnm);
        for field_idx in 2..23 {
            let byte_off = 24 + (field_idx - 2) * 8;
            write_padded_ascii(
                &mut buf[byte_off..byte_off + 8],
                self.str_by_index(field_idx),
            );
        }
        buf
    }

    /// Decodes the 192-byte SAC character block into the 23 `K*` string fields.
    pub fn from_string_bytes(&mut self, bytes: &[u8; 192]) {
        self.kstnm = decode_sac_ascii(&bytes[0..8]);
        self.kevnm = decode_sac_ascii(&bytes[8..24]);
        for field_idx in 2..23 {
            let byte_off = 24 + (field_idx - 2) * 8;
            *self.str_by_index_mut(field_idx) = decode_sac_ascii(&bytes[byte_off..byte_off + 8]);
        }
    }

    /// Auto-detects endianness of a 632-byte SAC binary header by inspecting `NVHDR` (`bytes 304..308`).
    pub fn detect_endian(bytes: &[u8]) -> Option<SacEndian> {
        if bytes.len() < SAC_HEADER_BYTES {
            return None;
        }
        let nvhdr_bytes: [u8; 4] = [bytes[304], bytes[305], bytes[306], bytes[307]];
        let le = i32::from_le_bytes(nvhdr_bytes);
        let be = i32::from_be_bytes(nvhdr_bytes);
        if (1..=7).contains(&le) {
            Some(SacEndian::Little)
        } else if (1..=7).contains(&be) {
            Some(SacEndian::Big)
        } else {
            None
        }
    }

    /// Encodes the 632-byte SAC binary header in `endian` byte order.
    pub fn to_bytes(&self, endian: SacEndian) -> [u8; SAC_HEADER_BYTES] {
        let mut out = [0_u8; SAC_HEADER_BYTES];
        let floats = self.to_float_words();
        for (i, &f) in floats.iter().enumerate() {
            let b = match endian {
                SacEndian::Little => f.to_le_bytes(),
                SacEndian::Big => f.to_be_bytes(),
            };
            out[i * 4..i * 4 + 4].copy_from_slice(&b);
        }
        let ints = self.to_int_words();
        for (i, &v) in ints.iter().enumerate() {
            let b = match endian {
                SacEndian::Little => v.to_le_bytes(),
                SacEndian::Big => v.to_be_bytes(),
            };
            let off = 280 + i * 4;
            out[off..off + 4].copy_from_slice(&b);
        }
        let chars = self.to_string_bytes();
        out[440..632].copy_from_slice(&chars);
        out
    }

    /// Decodes a 632-byte SAC binary header in `endian` byte order, rejecting `NaN`/`Inf` floats.
    pub fn from_bytes(bytes: &[u8; SAC_HEADER_BYTES], endian: SacEndian) -> Result<Self> {
        let mut floats = [0.0_f32; SAC_FLOAT_COUNT];
        for (i, slot) in floats.iter_mut().enumerate() {
            let arr = [
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ];
            let val = match endian {
                SacEndian::Little => f32::from_le_bytes(arr),
                SacEndian::Big => f32::from_be_bytes(arr),
            };
            if !val.is_finite() {
                return Err(EarthSnailError::Header(format!(
                    "non-finite float in SAC header word {} ({})",
                    i, SAC_FLOAT_NAMES[i]
                )));
            }
            *slot = val;
        }
        let mut ints = [0_i32; SAC_INT_COUNT];
        for (i, slot) in ints.iter_mut().enumerate() {
            let off = 280 + i * 4;
            let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
            *slot = match endian {
                SacEndian::Little => i32::from_le_bytes(arr),
                SacEndian::Big => i32::from_be_bytes(arr),
            };
        }
        let mut chars = [0_u8; 192];
        chars.copy_from_slice(&bytes[440..632]);

        let mut hdr = Self::empty_raw();
        hdr.from_float_words(&floats);
        hdr.from_int_words(&ints);
        hdr.from_string_bytes(&chars);
        Ok(hdr)
    }

    /// Exports the 22 double-precision `f64` fields of a SAC v7 (`NVHDR == 7`) footer:
    /// `[delta, b, e, o, a, t0..t9, f, evlo, evla, stlo, stla, sb, sdelta]`.
    pub fn to_v7_footer(&self) -> [f64; SAC_V7_FOOTER_DOUBLES] {
        [
            self.delta,
            self.b,
            self.e,
            self.o,
            self.a,
            self.t0,
            self.t1,
            self.t2,
            self.t3,
            self.t4,
            self.t5,
            self.t6,
            self.t7,
            self.t8,
            self.t9,
            self.f,
            self.evlo,
            self.evla,
            self.stlo,
            self.stla,
            self.internal2,
            self.internal3,
        ]
    }

    /// Applies the 22 double-precision `f64` fields of a SAC v7 (`NVHDR == 7`) footer.
    pub fn apply_v7_footer(&mut self, footer: &[f64; SAC_V7_FOOTER_DOUBLES]) -> Result<()> {
        for &v in footer {
            if !v.is_finite() {
                return Err(EarthSnailError::Header(
                    "non-finite double in SAC v7 footer".into(),
                ));
            }
        }
        self.delta = footer[0];
        self.b = footer[1];
        self.e = footer[2];
        self.o = footer[3];
        self.a = footer[4];
        self.t0 = footer[5];
        self.t1 = footer[6];
        self.t2 = footer[7];
        self.t3 = footer[8];
        self.t4 = footer[9];
        self.t5 = footer[10];
        self.t6 = footer[11];
        self.t7 = footer[12];
        self.t8 = footer[13];
        self.t9 = footer[14];
        self.f = footer[15];
        self.evlo = footer[16];
        self.evla = footer[17];
        self.stlo = footer[18];
        self.stla = footer[19];
        self.internal2 = footer[20];
        self.internal3 = footer[21];
        Ok(())
    }

    /// Recomputes `depmin`, `depmax`, and `depmen` from a sample slice.
    pub fn update_dep_stats(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            self.depmin = SAC_FLOAT_UNDEF;
            self.depmax = SAC_FLOAT_UNDEF;
            self.depmen = SAC_FLOAT_UNDEF;
            return;
        }
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut comp = 0.0; // Kahan compensated summation
        for &x in samples {
            if x < mn {
                mn = x;
            }
            if x > mx {
                mx = x;
            }
            let y = x - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        self.depmin = mn;
        self.depmax = mx;
        self.depmen = sum / (samples.len() as f64);
        let _ = math::trunc(0.0);
    }
}

fn float_field_index(name: &str) -> Option<usize> {
    if name.eq_ignore_ascii_case("fmt") {
        return Some(9);
    }
    if name.eq_ignore_ascii_case("sb") {
        return Some(54);
    }
    if name.eq_ignore_ascii_case("sdelta") {
        return Some(55);
    }
    SAC_FLOAT_NAMES
        .iter()
        .position(|&candidate| candidate.eq_ignore_ascii_case(name))
}

fn int_field_index(name: &str) -> Option<usize> {
    if name.eq_ignore_ascii_case("nsnpts") {
        return Some(10);
    }
    if name.eq_ignore_ascii_case("ibody") {
        return Some(27);
    }
    SAC_INT_NAMES
        .iter()
        .position(|&candidate| candidate.eq_ignore_ascii_case(name))
}

fn str_field_index(name: &str) -> Option<usize> {
    SAC_STR_NAMES
        .iter()
        .position(|&candidate| candidate.eq_ignore_ascii_case(name))
}

fn truncate_ascii(s: &str, max_bytes: usize) -> String {
    let mut out = String::with_capacity(s.len().min(max_bytes));
    for ch in s.chars() {
        let c = if ch.is_ascii() { ch } else { '?' };
        if out.len() + 1 > max_bytes {
            break;
        }
        out.push(c);
    }
    out
}

fn write_padded_ascii(slot: &mut [u8], val: &str) {
    slot.fill(b' ');
    let src = if is_sac_str_undef(val) {
        SAC_STR_UNDEF.as_bytes()
    } else {
        val.as_bytes()
    };
    let n = src.len().min(slot.len());
    slot[..n].copy_from_slice(&src[..n]);
}

fn decode_sac_ascii(slot: &[u8]) -> String {
    // Strip trailing ASCII spaces and NUL bytes cleanly without failing on non-UTF8 bytes.
    let mut end = slot.len();
    while end > 0 && (slot[end - 1] == b' ' || slot[end - 1] == 0) {
        end -= 1;
    }
    let mut s = String::with_capacity(end);
    for &b in &slot[..end] {
        if b == 0 {
            break;
        }
        s.push(if b.is_ascii() { b as char } else { '?' });
    }
    if is_sac_str_undef(&s) {
        String::from(SAC_STR_UNDEF)
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sac_header_158_words_and_case_insensitive_lookup() {
        assert_eq!(SAC_FLOAT_NAMES.len(), 70);
        assert_eq!(SAC_INT_NAMES.len(), 40);
        assert_eq!(SAC_STR_NAMES.len(), 23);
        assert_eq!(70 * 4 + 40 * 4 + 192, SAC_HEADER_BYTES);
        assert_eq!(SAC_HEADER_BYTES / 4, SAC_HEADER_WORDS);

        let mut hdr = SacHeader::new();
        assert_eq!(hdr.get_float("DELTA"), None);
        assert_eq!(hdr.get_int("nvhdr"), Some(6));
        assert_eq!(hdr.get_bool("LCALDA"), Some(true));
        assert_eq!(hdr.get_str("KSTNM"), None);

        hdr.set_float("DeLtA", 0.01).unwrap();
        assert_eq!(hdr.get_float("delta"), Some(0.01));

        hdr.set_str("KSTNM", "ANMO").unwrap();
        hdr.set_str("KEVNM", "TOHOKU_2011_MW91").unwrap();
        assert_eq!(hdr.get_str("kstnm"), Some("ANMO"));
        assert_eq!(hdr.get_str("kevnm"), Some("TOHOKU_2011_MW91"));
        assert_eq!(hdr.get_str("kevnm1"), Some("011_MW91"));

        // Binary LE & BE round-trip
        for endian in [SacEndian::Little, SacEndian::Big] {
            let bytes = hdr.to_bytes(endian);
            assert_eq!(SacHeader::detect_endian(&bytes), Some(endian));
            let decoded = SacHeader::from_bytes(&bytes, endian).unwrap();
            assert!((decoded.delta - 0.01).abs() < 1e-8);
            assert_eq!(decoded.get_str("kstnm"), Some("ANMO"));
            assert_eq!(decoded.get_str("kevnm"), Some("TOHOKU_2011_MW91"));
            assert_eq!(decoded.get_float("t0"), None);
        }
    }

    #[test]
    fn lcalda_auto_recalculates_dist_az_baz_gcarc() {
        let mut hdr = SacHeader::new();
        // Setting coordinates one by one while others are still -12345 must NOT panic or set garbage
        hdr.set_float("evla", 38.297).unwrap();
        hdr.set_float("evlo", 142.373).unwrap();
        hdr.set_float("stla", 34.9459).unwrap();
        assert_eq!(hdr.get_float("dist"), None);

        // Once the 4th coordinate is populated and lcalda == 1, dist/az/baz/gcarc populate automatically!
        hdr.set_float("stlo", -106.4572).unwrap();
        let dist_km = hdr.get_float("dist").expect("dist computed");
        let az = hdr.get_float("az").expect("az computed");
        let baz = hdr.get_float("baz").expect("baz computed");
        let gcarc = hdr.get_float("gcarc").expect("gcarc computed");

        let ref_geo = gps2dist_azimuth(38.297, 142.373, 34.9459, -106.4572);
        let tol = if cfg!(miri) { 1e-9 } else { 1e-12 };
        assert!((dist_km - ref_geo.distance_m * 1e-3).abs() < tol);
        assert!((az - ref_geo.azimuth_deg).abs() < tol);
        assert!((baz - ref_geo.back_azimuth_deg).abs() < tol);
        assert!((gcarc - ref_geo.gcarc_deg).abs() < tol);

        // Toggling lcalda off then back on also triggers update
        let mut hdr2 = SacHeader::new();
        hdr2.set_bool("lcalda", false).unwrap();
        hdr2.set_float("evla", 38.297).unwrap();
        hdr2.set_float("evlo", 142.373).unwrap();
        hdr2.set_float("stla", 34.9459).unwrap();
        hdr2.set_float("stlo", -106.4572).unwrap();
        assert_eq!(hdr2.get_float("dist"), None);
        hdr2.set_bool("lcalda", true).unwrap();
        assert!((hdr2.get_float("dist").unwrap() - dist_km).abs() < tol);
    }
}
