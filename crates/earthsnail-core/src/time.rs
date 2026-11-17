//! Exact nanosecond UTC timestamps (`UtcTimestamp(i128)`), proleptic Gregorian & ordinal (`NZJDAY`)
//! calendar conversions, ISO-8601/FDSN parsing & formatting, and drift-free sample-index arithmetic.

use crate::error::{EarthSnailError, Result};
use crate::math;
use alloc::format;
use alloc::string::String;
use core::fmt;
use core::ops::{Add, Sub};
use core::str::FromStr;
use core::time::Duration;

/// Signed nanoseconds since the Unix epoch (`1970-01-01T00:00:00.000000000Z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct UtcTimestamp(pub i128);

impl UtcTimestamp {
    /// Unix epoch `1970-01-01T00:00:00Z`.
    pub const UNIX_EPOCH: Self = Self(0);
    /// Nanoseconds per microsecond ($10^3$).
    pub const NANOS_PER_MICRO: i128 = 1_000;
    /// Nanoseconds per millisecond ($10^6$).
    pub const NANOS_PER_MILLI: i128 = 1_000_000;
    /// Nanoseconds per second ($10^9$).
    pub const NANOS_PER_SEC: i128 = 1_000_000_000;
    /// Nanoseconds per minute ($6 \times 10^{10}$).
    pub const NANOS_PER_MIN: i128 = 60 * Self::NANOS_PER_SEC;
    /// Nanoseconds per hour ($3.6 \times 10^{12}$).
    pub const NANOS_PER_HOUR: i128 = 3_600 * Self::NANOS_PER_SEC;
    /// Nanoseconds per civil day ($8.64 \times 10^{13}$).
    pub const NANOS_PER_DAY: i128 = 86_400 * Self::NANOS_PER_SEC;

    /// Constructs a timestamp directly from signed epoch nanoseconds.
    #[inline]
    pub const fn from_epoch_nanos(nanos: i128) -> Self {
        Self(nanos)
    }

    /// Returns signed epoch nanoseconds.
    #[inline]
    pub const fn as_epoch_nanos(self) -> i128 {
        self.0
    }

    /// Constructs a timestamp from floating-point epoch seconds, preserving sub-second precision.
    pub fn from_epoch_secs_f64(secs: f64) -> Result<Self> {
        if !secs.is_finite() {
            return Err(EarthSnailError::TimeParse(
                "non-finite epoch seconds".into(),
            ));
        }
        let whole = math::trunc(secs);
        let frac = secs - whole;
        if whole.abs() > 1e25 {
            return Err(EarthSnailError::TimeParse(
                "epoch seconds out of range".into(),
            ));
        }
        let nanos = (whole as i128) * Self::NANOS_PER_SEC
            + (math::round(frac * (Self::NANOS_PER_SEC as f64)) as i128);
        Ok(Self(nanos))
    }

    /// Returns epoch seconds as `f64`.
    #[inline]
    pub fn as_epoch_secs_f64(self) -> f64 {
        let secs = self.0.div_euclid(Self::NANOS_PER_SEC);
        let rem = self.0.rem_euclid(Self::NANOS_PER_SEC);
        (secs as f64) + (rem as f64) * 1e-9
    }

    /// Proleptic Gregorian leap-year test (`divisible by 4, except centuries not divisible by 400`).
    #[inline]
    pub const fn is_leap_year(year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    /// Number of days (`365` or `366`) in proleptic Gregorian `year`.
    #[inline]
    pub const fn days_in_year(year: i32) -> u16 {
        if Self::is_leap_year(year) {
            366
        } else {
            365
        }
    }

    /// Number of days in `(year, month)` for `month` in `1..=12`.
    pub fn days_in_month(year: i32, month: u8) -> Result<u8> {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => Ok(31),
            4 | 6 | 9 | 11 => Ok(30),
            2 => Ok(if Self::is_leap_year(year) { 29 } else { 28 }),
            _ => Err(EarthSnailError::TimeParse(format!(
                "month {month} out of range 1..=12"
            ))),
        }
    }

    /// Converts `(year, month, day)` to 1-based ordinal day-of-year `jday` (`1..=366`).
    pub fn ymd_to_jday(year: i32, month: u8, day: u8) -> Result<u16> {
        let dim = Self::days_in_month(year, month)?;
        if day == 0 || day > dim {
            return Err(EarthSnailError::TimeParse(format!(
                "day {day} out of range 1..={dim} for {year}-{month:02}"
            )));
        }
        const CUM_COMMON: [u16; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let mut doy = CUM_COMMON[(month - 1) as usize] + (day as u16);
        if month > 2 && Self::is_leap_year(year) {
            doy += 1;
        }
        Ok(doy)
    }

    /// Converts `(year, jday)` (`1..=365/366`) to `(month, day)`.
    pub fn jday_to_md(year: i32, jday: u16) -> Result<(u8, u8)> {
        let max_doy = Self::days_in_year(year);
        if jday == 0 || jday > max_doy {
            return Err(EarthSnailError::TimeParse(format!(
                "Julian day {jday} out of range 1..={max_doy} for year {year}"
            )));
        }
        let leap = u16::from(Self::is_leap_year(year));
        let month_lengths: [u16; 12] = [31, 28 + leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut rem = jday;
        for (idx, &mlen) in month_lengths.iter().enumerate() {
            if rem <= mlen {
                return Ok(((idx + 1) as u8, rem as u8));
            }
            rem -= mlen;
        }
        unreachable!()
    }

    /// Constructs a `UtcTimestamp` from proleptic Gregorian `(year, month, day, hour, min, sec, nano)`.
    pub fn from_ymd_hms_nano(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        min: u8,
        sec: u8,
        nano: u32,
    ) -> Result<Self> {
        let dim = Self::days_in_month(year, month)?;
        if day == 0 || day > dim {
            return Err(EarthSnailError::TimeParse(format!(
                "day {day} out of range 1..={dim} for {year}-{month:02}"
            )));
        }
        if hour > 23 || min > 59 || sec > 59 || nano >= 1_000_000_000 {
            return Err(EarthSnailError::TimeParse(format!(
                "invalid time-of-day {hour:02}:{min:02}:{sec:02}.{nano:09}"
            )));
        }
        let days = days_from_civil(year, month, day);
        let day_nanos = (hour as i128) * Self::NANOS_PER_HOUR
            + (min as i128) * Self::NANOS_PER_MIN
            + (sec as i128) * Self::NANOS_PER_SEC
            + (nano as i128);
        Ok(Self((days as i128) * Self::NANOS_PER_DAY + day_nanos))
    }

    /// Constructs a `UtcTimestamp` from `(year, jday, hour, min, sec, nano)` (`jday` in `1..=366`).
    pub fn from_year_jday_hms_nano(
        year: i32,
        jday: u16,
        hour: u8,
        min: u8,
        sec: u8,
        nano: u32,
    ) -> Result<Self> {
        let (month, day) = Self::jday_to_md(year, jday)?;
        Self::from_ymd_hms_nano(year, month, day, hour, min, sec, nano)
    }

    /// Constructs a `UtcTimestamp` from SEED v2 `BTime` fields, gracefully advancing leap-second `sec == 60`.
    pub fn from_seed_btime(
        year: i32,
        jday: u16,
        hour: u8,
        min: u8,
        sec: u8,
        tenth_millis: u16,
    ) -> Result<Self> {
        if tenth_millis >= 10_000 {
            return Err(EarthSnailError::TimeParse(format!(
                "SEED BTime 0.1ms field {tenth_millis} >= 10000"
            )));
        }
        let (base_sec, leap_extra_s) = if sec == 60 {
            (59_u8, 1_i128)
        } else {
            (sec, 0_i128)
        };
        let ts = Self::from_year_jday_hms_nano(
            year,
            jday,
            hour,
            min,
            base_sec,
            (tenth_millis as u32) * 100_000,
        )?;
        Ok(ts.add_nanos(leap_extra_s * Self::NANOS_PER_SEC))
    }

    /// Decomposes into proleptic Gregorian `(year, month, day, hour, min, sec, nano)`.
    pub fn to_ymd_hms_nano(self) -> (i32, u8, u8, u8, u8, u8, u32) {
        let days = self.0.div_euclid(Self::NANOS_PER_DAY) as i64;
        let rem_day = self.0.rem_euclid(Self::NANOS_PER_DAY);
        let (year, month, day) = civil_from_days(days);
        let nano = (rem_day % Self::NANOS_PER_SEC) as u32;
        let total_secs = (rem_day / Self::NANOS_PER_SEC) as u32;
        let hour = (total_secs / 3_600) as u8;
        let min = ((total_secs % 3_600) / 60) as u8;
        let sec = (total_secs % 60) as u8;
        (year, month, day, hour, min, sec, nano)
    }

    /// Decomposes into `(year, jday, hour, min, sec, nano)` (`jday` in `1..=366`).
    pub fn to_year_jday_hms_nano(self) -> (i32, u16, u8, u8, u8, u32) {
        let (year, month, day, hour, min, sec, nano) = self.to_ymd_hms_nano();
        let jday = Self::ymd_to_jday(year, month, day).unwrap_or(1);
        (year, jday, hour, min, sec, nano)
    }

    /// Returns 1-based ordinal day of year (`1..=366`) for SAC `NZJDAY` and miniSEED headers.
    #[inline]
    pub fn jday(self) -> u16 {
        self.to_year_jday_hms_nano().1
    }

    /// Returns calendar year.
    #[inline]
    pub fn year(self) -> i32 {
        self.to_ymd_hms_nano().0
    }

    /// Returns calendar month (`1..=12`).
    #[inline]
    pub fn month(self) -> u8 {
        self.to_ymd_hms_nano().1
    }

    /// Returns calendar day of month (`1..=31`).
    #[inline]
    pub fn day(self) -> u8 {
        self.to_ymd_hms_nano().2
    }

    /// Returns hour (`0..=23`).
    #[inline]
    pub fn hour(self) -> u8 {
        self.to_ymd_hms_nano().3
    }

    /// Returns minute (`0..=59`).
    #[inline]
    pub fn minute(self) -> u8 {
        self.to_ymd_hms_nano().4
    }

    /// Returns second (`0..=59`).
    #[inline]
    pub fn second(self) -> u8 {
        self.to_ymd_hms_nano().5
    }

    /// Returns sub-second nanosecond (`0..1_000_000_000`).
    #[inline]
    pub fn nanosecond(self) -> u32 {
        self.to_ymd_hms_nano().6
    }

    /// Adds signed nanoseconds.
    #[inline]
    pub const fn add_nanos(self, delta_ns: i128) -> Self {
        Self(self.0.saturating_add(delta_ns))
    }

    /// Subtracts signed nanoseconds.
    #[inline]
    pub const fn sub_nanos(self, delta_ns: i128) -> Self {
        Self(self.0.saturating_sub(delta_ns))
    }

    /// Adds floating-point seconds while splitting integer & fractional parts to avoid mantissa loss.
    pub fn add_secs_f64(self, secs: f64) -> Result<Self> {
        if !secs.is_finite() {
            return Err(EarthSnailError::InvalidParameter(
                "non-finite seconds offset".into(),
            ));
        }
        let whole = math::trunc(secs);
        let frac = secs - whole;
        let delta_ns = (whole as i128) * Self::NANOS_PER_SEC
            + (math::round(frac * (Self::NANOS_PER_SEC as f64)) as i128);
        Ok(self.add_nanos(delta_ns))
    }

    /// Signed nanosecond difference `self - earlier`.
    #[inline]
    pub const fn duration_nanos_since(self, earlier: Self) -> i128 {
        self.0 - earlier.0
    }

    /// Signed second difference `(self - earlier)` in `f64` without losing precision on close timestamps.
    #[inline]
    pub fn duration_secs_f64_since(self, earlier: Self) -> f64 {
        let diff = self.0 - earlier.0;
        let whole_s = diff / Self::NANOS_PER_SEC;
        let rem_ns = diff % Self::NANOS_PER_SEC;
        (whole_s as f64) + (rem_ns as f64) * 1e-9
    }

    /// Drift-free conversion from sample `index` (may be negative or huge) at `sampling_rate` Hz
    /// to exact `UtcTimestamp` using rational `i128` arithmetic.
    pub fn sample_timestamp(self, index: i64, sampling_rate: f64) -> Result<Self> {
        let spec = RateSpec::from_hz(sampling_rate)?;
        let offset_ns = spec.index_to_nanos(index as i128);
        Ok(self.add_nanos(offset_ns))
    }

    /// Drift-free conversion from `timestamp` back to the nearest integer sample index relative to `self`.
    pub fn sample_index_nearest(self, timestamp: Self, sampling_rate: f64) -> Result<i64> {
        let spec = RateSpec::from_hz(sampling_rate)?;
        let delta_ns = timestamp.0 - self.0;
        let idx = spec.nanos_to_index_nearest(delta_ns);
        i64::try_from(idx)
            .map_err(|_| EarthSnailError::CapacityExceeded("sample index overflows i64".into()))
    }

    /// Fractional sample offset `(timestamp - self) * sampling_rate` computed from exact `i128` nanosecond delta.
    pub fn sample_offset_f64(self, timestamp: Self, sampling_rate: f64) -> Result<f64> {
        if !sampling_rate.is_finite() || sampling_rate <= 0.0 {
            return Err(EarthSnailError::InvalidParameter(format!(
                "invalid sampling_rate {sampling_rate}"
            )));
        }
        Ok(timestamp.duration_secs_f64_since(self) * sampling_rate)
    }

    /// Formats as fixed 9-digit nanosecond ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`).
    pub fn to_iso8601_nanos(self) -> String {
        let (y, m, d, hh, mm, ss, ns) = self.to_ymd_hms_nano();
        if y < 0 {
            format!(
                "-{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
                -(y as i64),
                m,
                d,
                hh,
                mm,
                ss,
                ns
            )
        } else {
            format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{ns:09}Z")
        }
    }

    /// Formats as clean ISO-8601 / FDSN UTC string, trimming trailing fractional zeros while remaining
    /// 100% lossless to 1-nanosecond precision.
    pub fn to_iso8601(self) -> String {
        let (y, m, d, hh, mm, ss, ns) = self.to_ymd_hms_nano();
        let prefix = if y < 0 {
            format!(
                "-{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                -(y as i64),
                m,
                d,
                hh,
                mm,
                ss
            )
        } else {
            format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}")
        };
        if ns == 0 {
            format!("{prefix}Z")
        } else {
            let frac_full = format!("{ns:09}");
            let trimmed = frac_full.trim_end_matches('0');
            format!("{prefix}.{trimmed}Z")
        }
    }

    /// Formats for FDSNWS URL query parameters (`YYYY-MM-DDTHH:MM:SS.ffffff` without trailing `Z`).
    pub fn to_fdsn_query(self) -> String {
        let s = self.to_iso8601();
        String::from(s.strip_suffix('Z').unwrap_or(&s))
    }

    /// Parses ISO-8601, RFC-3339, FDSN calendar (`YYYY-MM-DD[THH:MM:SS[.fff]][Z|±HH:MM]`),
    /// space-separated (`YYYY-MM-DD HH:MM:SS`), or ordinal (`YYYY-DDD[THH:MM:SS[.fff]]`) strings.
    pub fn parse_iso8601(input: &str) -> Result<Self> {
        let s = input.trim();
        if s.is_empty() {
            return Err(EarthSnailError::TimeParse("empty timestamp string".into()));
        }

        let (neg_year, rest_after_sign) = if let Some(r) = s.strip_prefix('-') {
            (true, r)
        } else if let Some(r) = s.strip_prefix('+') {
            (false, r)
        } else {
            (false, s)
        };

        let dash_pos = rest_after_sign.find('-').ok_or_else(|| {
            EarthSnailError::TimeParse(format!("missing year separator in '{input}'"))
        })?;
        let year_mag: i32 = rest_after_sign[..dash_pos]
            .parse()
            .map_err(|_| EarthSnailError::TimeParse(format!("invalid year in '{input}'")))?;
        let year = if neg_year { -year_mag } else { year_mag };
        let after_year = &rest_after_sign[dash_pos + 1..];

        // Split date and optional time at 'T', 't', or ' '.
        let (date_rem, time_part_opt) = if let Some(idx) = after_year.find(['T', 't', ' ']) {
            (&after_year[..idx], Some(after_year[idx + 1..].trim()))
        } else {
            // Date-only string could still carry trailing 'Z'
            let d = after_year
                .strip_suffix('Z')
                .or_else(|| after_year.strip_suffix('z'))
                .unwrap_or(after_year);
            (d, None)
        };

        // Determine calendar `MM-DD` vs ordinal `DDD`.
        let (month, day) = if let Some((m_str, d_str)) = date_rem.split_once('-') {
            let m: u8 = m_str
                .parse()
                .map_err(|_| EarthSnailError::TimeParse(format!("invalid month in '{input}'")))?;
            let d: u8 = d_str
                .parse()
                .map_err(|_| EarthSnailError::TimeParse(format!("invalid day in '{input}'")))?;
            (m, d)
        } else {
            let jday: u16 = date_rem.parse().map_err(|_| {
                EarthSnailError::TimeParse(format!("invalid ordinal day in '{input}'"))
            })?;
            Self::jday_to_md(year, jday)?
        };

        let Some(time_raw) = time_part_opt else {
            return Self::from_ymd_hms_nano(year, month, day, 0, 0, 0, 0);
        };
        if time_raw.is_empty() {
            return Err(EarthSnailError::TimeParse(format!(
                "empty time component in '{input}'"
            )));
        }

        // Extract timezone suffix ('Z', 'z', '+HH:MM', '-HH:MM', '+HHMM', '-HHMM', '+HH', '-HH').
        let (hms_part, tz_offset_secs): (&str, i128) = if let Some(base) = time_raw
            .strip_suffix('Z')
            .or_else(|| time_raw.strip_suffix('z'))
        {
            (base, 0)
        } else if let Some(tz_idx) = time_raw.rfind(['+', '-']) {
            let base = &time_raw[..tz_idx];
            let tz_str = &time_raw[tz_idx..];
            let offset = parse_tz_offset_secs(tz_str, input)?;
            (base, offset)
        } else {
            (time_raw, 0)
        };

        let mut parts = hms_part.split(':');
        let hh_str = parts
            .next()
            .ok_or_else(|| EarthSnailError::TimeParse(format!("missing hour in '{input}'")))?;
        let mm_str = parts.next().unwrap_or("0");
        let ss_frac_str = parts.next().unwrap_or("0");
        if parts.next().is_some() {
            return Err(EarthSnailError::TimeParse(format!(
                "too many ':' fields in '{input}'"
            )));
        }

        let hour: u8 = hh_str
            .parse()
            .map_err(|_| EarthSnailError::TimeParse(format!("invalid hour in '{input}'")))?;
        let min: u8 = mm_str
            .parse()
            .map_err(|_| EarthSnailError::TimeParse(format!("invalid minute in '{input}'")))?;

        let (sec_str, nano) = if let Some((s_part, f_part)) = ss_frac_str.split_once('.') {
            if f_part.is_empty() || !f_part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(EarthSnailError::TimeParse(format!(
                    "invalid fractional seconds in '{input}'"
                )));
            }
            let mut ns_acc: u32 = 0;
            let mut scale: u32 = 100_000_000;
            let mut round_up = false;
            for (i, b) in f_part.bytes().enumerate() {
                let digit = (b - b'0') as u32;
                if i < 9 {
                    ns_acc += digit * scale;
                    scale /= 10;
                } else if i == 9 {
                    round_up = digit >= 5;
                }
            }
            if round_up {
                ns_acc += 1;
            }
            (s_part, ns_acc)
        } else {
            (ss_frac_str, 0_u32)
        };

        let sec_raw: u8 = sec_str
            .parse()
            .map_err(|_| EarthSnailError::TimeParse(format!("invalid second in '{input}'")))?;
        // Handle leap second `:60` or sub-nanosecond rounding carry into 1_000_000_000.
        let (sec, extra_secs, adj_nano) = if sec_raw == 60 {
            if nano >= 1_000_000_000 {
                (59_u8, 2_i128, nano - 1_000_000_000)
            } else {
                (59_u8, 1_i128, nano)
            }
        } else if nano >= 1_000_000_000 {
            (sec_raw, 1_i128, nano - 1_000_000_000)
        } else {
            (sec_raw, 0_i128, nano)
        };

        let local_ts = Self::from_ymd_hms_nano(year, month, day, hour, min, sec, adj_nano)?;
        Ok(local_ts.add_nanos((extra_secs - tz_offset_secs) * Self::NANOS_PER_SEC))
    }
}

fn parse_tz_offset_secs(tz_str: &str, full_input: &str) -> Result<i128> {
    let sign: i128 = if tz_str.starts_with('-') { -1 } else { 1 };
    let digits = &tz_str[1..];
    let (hh_str, mm_str) = if let Some((h, m)) = digits.split_once(':') {
        (h, m)
    } else if digits.len() == 4 {
        (&digits[..2], &digits[2..])
    } else if digits.len() == 2 {
        (digits, "00")
    } else {
        return Err(EarthSnailError::TimeParse(format!(
            "invalid timezone offset '{tz_str}' in '{full_input}'"
        )));
    };
    let hh: u8 = hh_str
        .parse()
        .map_err(|_| EarthSnailError::TimeParse(format!("invalid tz hour in '{full_input}'")))?;
    let mm: u8 = mm_str
        .parse()
        .map_err(|_| EarthSnailError::TimeParse(format!("invalid tz minute in '{full_input}'")))?;
    if hh > 23 || mm > 59 {
        return Err(EarthSnailError::TimeParse(format!(
            "timezone offset out of range in '{full_input}'"
        )));
    }
    Ok(sign * ((hh as i128) * 3_600 + (mm as i128) * 60))
}

#[derive(Debug, Clone, Copy)]
enum RateSpec {
    /// Exact integer Hz (`1..=10^9`), `offset_ns = round(idx * 10^9 / hz)`.
    IntegerHz(i128),
    /// Exact integer nanosecond sample interval (`dt_ns`), `offset_ns = idx * dt_ns`.
    ExactPeriodNs(i128),
    /// General rational rate in pico-Hertz (`round(fs * 10^12)`), `offset_ns = round(idx * 10^21 / phz)`.
    PicoHz(i128),
}

impl RateSpec {
    const PICO_NANOS_SCALE: i128 = 1_000_000_000_000_000_000_000; // 10^21

    fn from_hz(sampling_rate: f64) -> Result<Self> {
        if !sampling_rate.is_finite() || sampling_rate <= 0.0 || sampling_rate > 1e12 {
            return Err(EarthSnailError::InvalidParameter(format!(
                "invalid sampling_rate {sampling_rate} Hz"
            )));
        }
        let rounded_hz = math::round(sampling_rate);
        if sampling_rate >= 1.0 && (sampling_rate - rounded_hz).abs() <= 1e-11 * sampling_rate {
            return Ok(Self::IntegerHz(rounded_hz as i128));
        }
        let dt_ns = 1e9 / sampling_rate;
        let rounded_dt_ns = math::round(dt_ns);
        if (1.0..=1e18).contains(&rounded_dt_ns) {
            let implied_rate = 1e9 / rounded_dt_ns;
            if (implied_rate - sampling_rate).abs() <= 1e-11 * sampling_rate {
                return Ok(Self::ExactPeriodNs(rounded_dt_ns as i128));
            }
        }
        let phz = math::round(sampling_rate * 1_000_000_000_000.0) as i128;
        if phz <= 0 {
            return Err(EarthSnailError::InvalidParameter(format!(
                "sampling_rate {sampling_rate} Hz too small"
            )));
        }
        Ok(Self::PicoHz(phz))
    }

    #[inline]
    fn index_to_nanos(self, idx: i128) -> i128 {
        match self {
            Self::IntegerHz(hz) => round_div_i128(idx * UtcTimestamp::NANOS_PER_SEC, hz),
            Self::ExactPeriodNs(dt_ns) => idx * dt_ns,
            Self::PicoHz(phz) => round_div_i128(idx * Self::PICO_NANOS_SCALE, phz),
        }
    }

    #[inline]
    fn nanos_to_index_nearest(self, delta_ns: i128) -> i128 {
        match self {
            Self::IntegerHz(hz) => round_div_i128(delta_ns * hz, UtcTimestamp::NANOS_PER_SEC),
            Self::ExactPeriodNs(dt_ns) => round_div_i128(delta_ns, dt_ns),
            Self::PicoHz(phz) => round_div_i128(delta_ns * phz, Self::PICO_NANOS_SCALE),
        }
    }
}

#[inline]
fn round_div_i128(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0);
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

/// Howard Hinnant's exact days-from-civil (`1970-01-01 -> 0`) for proleptic Gregorian dates.
#[inline]
fn days_from_civil(year: i32, month: u8, day: u8) -> i64 {
    let y = (year as i64) - if month <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = (y - era * 400) as u32;
    let m = month as u32;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + (day as u32) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + (doe as i64) - 719_468
}

/// Howard Hinnant's exact civil-from-days (`0 -> 1970-01-01`) for proleptic Gregorian dates.
#[inline]
fn civil_from_days(days_since_epoch: i64) -> (i32, u8, u8) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let m = if mp < 10 {
        (mp + 3) as u8
    } else {
        (mp - 9) as u8
    };
    let year = (y + if m <= 2 { 1 } else { 0 }) as i32;
    (year, m, d)
}

impl fmt::Display for UtcTimestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_iso8601())
    }
}

impl FromStr for UtcTimestamp {
    type Err = EarthSnailError;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse_iso8601(s)
    }
}

impl Add<Duration> for UtcTimestamp {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Duration) -> Self {
        self.add_nanos(rhs.as_nanos() as i128)
    }
}

impl Sub<Duration> for UtcTimestamp {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Duration) -> Self {
        self.sub_nanos(rhs.as_nanos() as i128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leap_years_and_jday_roundtrip() {
        assert!(!UtcTimestamp::is_leap_year(1900));
        assert!(UtcTimestamp::is_leap_year(2000));
        assert!(UtcTimestamp::is_leap_year(2024));
        assert!(!UtcTimestamp::is_leap_year(2100));
        assert!(UtcTimestamp::is_leap_year(0)); // 1 BCE in ISO proleptic Gregorian

        // Reject 1900-02-29, accept 2000-02-29 & 2024-02-29
        assert!(UtcTimestamp::from_ymd_hms_nano(1900, 2, 29, 0, 0, 0, 0).is_err());
        let leap_2000 =
            UtcTimestamp::from_ymd_hms_nano(2000, 2, 29, 12, 34, 56, 789_012_345).unwrap();
        assert_eq!(leap_2000.jday(), 60);
        assert_eq!(
            leap_2000.to_ymd_hms_nano(),
            (2000, 2, 29, 12, 34, 56, 789_012_345)
        );

        // Exhaustive check across an entire leap year (2024) and common year (2023)
        for &(year, total_days) in &[
            (1969, 365_u16),
            (1970, 365),
            (2000, 366),
            (2023, 365),
            (2024, 366),
        ] {
            for jday in 1..=total_days {
                let ts = UtcTimestamp::from_year_jday_hms_nano(year, jday, 23, 59, 59, 999_999_999)
                    .unwrap();
                let (ry, rj, rh, rmin, rs, rns) = ts.to_year_jday_hms_nano();
                assert_eq!(
                    (ry, rj, rh, rmin, rs, rns),
                    (year, jday, 23, 59, 59, 999_999_999)
                );
            }
        }
    }

    #[test]
    fn iso8601_and_fdsn_parsing() {
        let ts: UtcTimestamp = "2011-03-11T05:46:24.120Z".parse().unwrap();
        assert_eq!(ts.to_iso8601(), "2011-03-11T05:46:24.12Z");
        assert_eq!(ts.parse_check_roundtrip(), ts);

        // Ordinal ISO-8601 format (YYYY-DDDTHH:MM:SS)
        let ord: UtcTimestamp = "2011-070T05:46:24.12Z".parse().unwrap();
        assert_eq!(ord, ts);

        // Timezone offset
        let tz: UtcTimestamp = "2011-03-11T14:46:24.12+09:00".parse().unwrap();
        assert_eq!(tz, ts);

        // Pre-1970 historical event (1960 Valdivia M9.5)
        let valdivia: UtcTimestamp = "1960-05-22T19:11:20.123456789Z".parse().unwrap();
        assert!(valdivia.as_epoch_nanos() < 0);
        assert_eq!(valdivia.to_iso8601(), "1960-05-22T19:11:20.123456789Z");
        assert_eq!(
            valdivia.to_ymd_hms_nano(),
            (1960, 5, 22, 19, 11, 20, 123_456_789)
        );
    }

    #[test]
    fn zero_drift_sample_arithmetic() {
        let t0: UtcTimestamp = "2020-01-01T00:00:00Z".parse().unwrap();

        // 100 Hz over 100,000,000,000 samples (1 billion seconds): exact 0 ns drift!
        let n_100hz: i64 = 100_000_000_000;
        let t_end = t0.sample_timestamp(n_100hz, 100.0).unwrap();
        assert_eq!(
            t_end.duration_nanos_since(t0),
            1_000_000_000 * UtcTimestamp::NANOS_PER_SEC
        );
        assert_eq!(t0.sample_index_nearest(t_end, 100.0).unwrap(), n_100hz);

        // Non-terminating decimal period 30 Hz over 3,000,000,000 samples (100,000,000 s): exact 0 ns drift!
        let n_30hz: i64 = 3_000_000_000;
        let t_30 = t0.sample_timestamp(n_30hz, 30.0).unwrap();
        assert_eq!(
            t_30.duration_nanos_since(t0),
            100_000_000 * UtcTimestamp::NANOS_PER_SEC
        );
        for k in [-101_i64, -1, 0, 1, 2, 3, 29, 30, 31, 999_999, n_30hz] {
            let tk = t0.sample_timestamp(k, 30.0).unwrap();
            assert_eq!(t0.sample_index_nearest(tk, 30.0).unwrap(), k);
        }

        // Long-period 1/3 Hz (dt = 3.0 s)
        let t_lp = t0.sample_timestamp(12_345_678, 1.0 / 3.0).unwrap();
        assert_eq!(
            t_lp.duration_nanos_since(t0),
            12_345_678_i128 * 3 * UtcTimestamp::NANOS_PER_SEC
        );
    }

    impl UtcTimestamp {
        fn parse_check_roundtrip(self) -> Self {
            self.to_iso8601().parse().unwrap()
        }
    }
}
