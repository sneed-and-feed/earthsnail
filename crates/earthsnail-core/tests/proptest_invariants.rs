//! Property-based physical & algebraic invariants for `earthsnail-core` (Gate 3).
//! Gated under `not(miri)` so `cargo miri test` avoids slow OS `getrandom` loops while unit tests
//! verify deterministic edge cases under Miri.

#![cfg(not(miri))]

use earthsnail_core::{
    direct_geodesic, geocentric_to_geographic_lat, geographic_to_geocentric_lat, gps2dist_azimuth,
    Complex64, SacEndian, SacHeader, UtcTimestamp,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_calendar_iso8601_and_jday_roundtrip(
        year in -999_i32..=3500_i32,
        jday_frac in 0.0_f64..1.0_f64,
        hour in 0_u8..=23_u8,
        min in 0_u8..=59_u8,
        sec in 0_u8..=59_u8,
        nano in 0_u32..1_000_000_000_u32,
    ) {
        let max_doy = UtcTimestamp::days_in_year(year);
        let jday = 1 + ((jday_frac * (max_doy as f64)) as u16).min(max_doy - 1);
        let ts = UtcTimestamp::from_year_jday_hms_nano(year, jday, hour, min, sec, nano).unwrap();
        prop_assert_eq!(ts.to_year_jday_hms_nano(), (year, jday, hour, min, sec, nano));

        if year >= 0 {
            let s = ts.to_iso8601();
            let parsed: UtcTimestamp = s.parse().unwrap();
            prop_assert_eq!(parsed, ts);
        }
    }

    #[test]
    fn prop_sample_index_timestamp_bijection(
        start_nanos in -1_000_000_000_000_000_000_i128..1_000_000_000_000_000_000_i128,
        index in -50_000_000_i64..50_000_000_i64,
        rate_idx in 0_usize..8_usize,
    ) {
        let rates = [1.0, 20.0, 30.0, 40.0, 100.0, 125.0, 200.0, 250.0];
        let fs = rates[rate_idx];
        let t0 = UtcTimestamp::from_epoch_nanos(start_nanos);
        let tk = t0.sample_timestamp(index, fs).unwrap();
        let back = t0.sample_index_nearest(tk, fs).unwrap();
        prop_assert_eq!(back, index);
    }

    #[test]
    fn prop_complex64_field_invariants(
        ar in -1e6_f64..1e6_f64,
        ai in -1e6_f64..1e6_f64,
        br in -1e6_f64..1e6_f64,
        bi in -1e6_f64..1e6_f64,
    ) {
        let a = Complex64::new(ar, ai);
        let b = Complex64::new(br, bi);
        if b.abs() > 1e-6 {
            let q = (a * b) / b;
            let tol = 1e-10 * (1.0 + a.abs());
            prop_assert!((q - a).abs() <= tol);
        }
        let c = a * a.conj();
        prop_assert!(c.im.abs() <= 1e-9 * (1.0 + c.re.abs()));
    }

    #[test]
    fn prop_sac_header_le_be_lossless_roundtrip(
        delta in 0.001_f32..10.0_f32,
        stla in -89.9_f32..89.9_f32,
        stlo in -179.9_f32..179.9_f32,
        npts in 1_i32..1_000_000_i32,
    ) {
        let mut hdr = SacHeader::new();
        hdr.set_bool("lcalda", false).unwrap();
        hdr.set_float("delta", delta as f64).unwrap();
        hdr.set_float("stla", stla as f64).unwrap();
        hdr.set_float("stlo", stlo as f64).unwrap();
        hdr.set_int("npts", npts).unwrap();
        hdr.set_str("kstnm", "TESTSTA").unwrap();
        hdr.set_str("kevnm", "EVENT_0123456789").unwrap();

        let le = hdr.to_bytes(SacEndian::Little);
        let be = hdr.to_bytes(SacEndian::Big);
        let d_le = SacHeader::from_bytes(&le, SacEndian::Little).unwrap();
        let d_be = SacHeader::from_bytes(&be, SacEndian::Big).unwrap();
        prop_assert_eq!(&d_le, &d_be);
        prop_assert_eq!(d_le.npts, npts);
        prop_assert_eq!(d_le.get_str("kevnm"), Some("EVENT_0123456789"));
    }

    #[test]
    fn prop_wgs84_inverse_direct_roundtrip_and_symmetry(
        lat1 in -89.999_f64..89.999_f64,
        lon1 in -180.0_f64..180.0_f64,
        lat2 in -89.999_f64..89.999_f64,
        lon2 in -180.0_f64..180.0_f64,
    ) {
        let g_lat = geocentric_to_geographic_lat(geographic_to_geocentric_lat(lat1));
        prop_assert!((g_lat - lat1).abs() < 1e-11);

        let inv = gps2dist_azimuth(lat1, lon1, lat2, lon2);
        let rev = gps2dist_azimuth(lat2, lon2, lat1, lon1);
        prop_assert!((inv.distance_m - rev.distance_m).abs() < 1e-6);

        if inv.distance_m > 1e-3 {
            let dir = direct_geodesic(lat1, lon1, inv.azimuth_deg, inv.distance_m);
            prop_assert!((dir.lat_deg - lat2).abs() < 1e-9);
            let mut dlon = (dir.lon_deg - lon2) % 360.0;
            if dlon > 180.0 { dlon -= 360.0; }
            if dlon < -180.0 { dlon += 360.0; }
            prop_assert!(dlon.abs() * lat2.to_radians().cos() < 1e-8);
        }
    }
}
