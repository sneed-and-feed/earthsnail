//! `earthsnail-io`: Zero-dependency pure-Rust waveform, metadata, and streaming network I/O
//! crate for the EarthSnail seismological engine (`#![no_std] + alloc` compatible when default
//! `std` feature is disabled; `#![forbid(unsafe_code)]`).
//!
//! ### Modules
//! - [`sac`]: SAC Binary (`v6` & `v7`, auto-sensing Little/Big-Endian) and SAC Alphanumeric (`SACXY`).
//! - [`steim`]: Lossless Steim-1 & Steim-2 64-byte frame codecs (all 7 Steim-2 nibble widths, `X0`/`Xn` verified).
//! - [`mseed`]: SEED / miniSEED v2 (`2.4`) multi-record reader & writer (`256B..=65536B`, B100/B1000/B1001).
//! - [`mseed3`]: FDSN miniSEED v3 (`MS3`) reader & writer with Castagnoli CRC-32C & JSON extra headers.
//! - [`ascii`]: EarthScope GeoCSV 2.0 (`1`-col & `2`-col) + ObsPy `SLIST` & `TSPAIR` readers/writers.
//! - [`sniff`]: Universal `read_waveforms_bytes` format auto-detector.
//! - [`xml`]: Allocation-bounded zero-dependency XML 1.0 pull scanner.
//! - [`sacpz`]: SAC PoleZero (`SACPZ`) reader & writer.
//! - [`stationxml`]: FDSN `StationXML` (`1.0`–`1.2`) parser & serializer.
//! - [`quakeml`]: FDSN `QuakeML` (`1.2`) event catalog parser & serializer.
//! - [`fdsn`] (`std`): Pure-Rust FDSNWS 1.x query builders, `HttpTransport`, and `TcpHttpTransport`.
//! - [`seedlink`] (`std`): Pure-Rust SeedLink v3 & v4 wire frame codec and streaming state machine.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub use earthsnail_core as core_types;

pub mod ascii;
pub mod mseed;
pub mod mseed3;
pub mod quakeml;
pub mod sac;
pub mod sacpz;
pub mod sniff;
pub mod stationxml;
pub mod steim;
pub mod xml;

#[cfg(feature = "std")]
pub mod fdsn;
#[cfg(feature = "std")]
pub mod seedlink;

pub use ascii::{
    read_geocsv_bytes, read_geocsv_str, read_obspy_ascii_str, read_slist_str, read_tspair_str,
    write_geocsv_stream, write_geocsv_trace, write_slist_stream, write_tspair_stream,
};
pub use mseed::{
    read_mseed_bytes, read_mseed_records, write_mseed_bytes, write_mseed_stream_bytes,
    write_mseed_trace_bytes, MseedByteOrder, MseedEncoding, MseedRecord, MseedWriteOptions,
};
pub use mseed3::{
    crc32c, ms3_record_crc32c, read_mseed3_bytes, read_mseed3_records, write_mseed3_stream_bytes,
    write_mseed3_trace_bytes, Mseed3Record, Mseed3WriteOptions,
};
pub use quakeml::{read_quakeml_bytes, read_quakeml_str, write_quakeml_str};
pub use sac::{
    read_sac_alpha_bytes, read_sac_alpha_str, read_sac_bytes, read_sac_bytes_with_endian,
    write_sac_alpha_bytes, write_sac_alpha_str, write_sac_bytes,
};
pub use sacpz::{
    parse_sacpz_response, read_sacpz_bytes, read_sacpz_str, write_sacpz_str, SacPzRecord,
};
pub use sniff::{detect_waveform_format, read_waveforms_bytes, WaveformFormat};
pub use stationxml::{
    parse_stationxml, read_stationxml, read_stationxml_bytes, read_stationxml_str,
    write_stationxml_str,
};
pub use steim::{decode_steim1, decode_steim2, encode_steim1, encode_steim2};

#[cfg(feature = "std")]
pub use fdsn::{
    BulkChannelSelection, EventQuery, FdsnClient, FdsnProvider, HttpResponse, HttpTransport,
    StationLevel, StationQuery, TcpHttpTransport, WaveformQuery,
};
#[cfg(feature = "std")]
pub use mseed::{read_mseed_file, write_mseed_file};
#[cfg(feature = "std")]
pub use mseed3::{read_mseed3_file, write_mseed3_file};
#[cfg(feature = "std")]
pub use quakeml::{read_quakeml_file, write_quakeml_file};
#[cfg(feature = "std")]
pub use sac::{read_sac_alpha_file, read_sac_file, write_sac_alpha_file, write_sac_file};
#[cfg(feature = "std")]
pub use sacpz::{read_sacpz_file, write_sacpz_file};
#[cfg(feature = "std")]
pub use seedlink::{SeedLinkClient, SeedLinkHello, SeedLinkPacket, SeedLinkV4Format};
#[cfg(feature = "std")]
pub use sniff::read_waveforms_file;
#[cfg(feature = "std")]
pub use stationxml::{read_stationxml_file, write_stationxml_file};

#[cfg(test)]
mod no_std_unit_tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use earthsnail_core::{ChannelId, SacEndian, Stream, Trace, TraceStats, UtcTimestamp};

    #[test]
    fn crc32c_standard_check_vector() {
        // RFC 3720 / Castagnoli standard check value for ASCII "123456789" is 0xE3069283:
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn steim1_and_steim2_all_regimes_roundtrip_and_xn_tamper_detection() {
        // Exercises 0, 4-bit, 5-bit, 6-bit, 8-bit, 10-bit, 15-bit, 16-bit, 30-bit diffs:
        let mut samples: Vec<i32> = Vec::new();
        let mut cur = 50_000_i32;
        let step_schedule = [
            0,
            0,
            0,
            0,
            0,
            0,
            0, // 4-bit
            -7,
            7,
            -8,
            6,
            -5,
            4,
            -3, // 4-bit boundary [-8, 7]
            -16,
            15,
            -14,
            13,
            -12,
            11, // 5-bit boundary [-16, 15]
            -32,
            31,
            -30,
            29,
            -28, // 6-bit boundary [-32, 31]
            -128,
            127,
            -100,
            99, // 8-bit boundary [-128, 127]
            -512,
            511,
            -300, // 10-bit boundary [-512, 511]
            -16_384,
            16_383, // 15-bit boundary [-16384, 16383]
            -30_000,
            30_000, // 30-bit in Steim-2, 16-bit in Steim-1
            -536_870_912,
            536_870_911, // exact 30-bit boundary!
        ];
        samples.push(cur);
        for &d in &step_schedule {
            cur = cur.wrapping_add(d);
            samples.push(cur);
        }

        let s1 = encode_steim1(&samples, 0);
        assert_eq!(decode_steim1(&s1, samples.len()).unwrap(), samples);

        let mut s2 = encode_steim2(&samples, 0).unwrap();
        assert_eq!(decode_steim2(&s2, samples.len()).unwrap(), samples);

        // Corrupting Xn (bytes 8..12) MUST be caught!
        s2[11] ^= 0x01;
        assert!(decode_steim2(&s2, samples.len()).is_err());

        // True 32-bit impulse (> 30 bits) succeeds in Steim-1 and returns typed Err in Steim-2:
        let huge = vec![0_i32, 1_200_000_000, -1_200_000_000, 42];
        let s1_huge = encode_steim1(&huge, 0);
        assert_eq!(decode_steim1(&s1_huge, huge.len()).unwrap(), huge);
        assert!(encode_steim2(&huge, 0).is_err());
    }

    #[test]
    fn sac_v6_v7_alpha_ms2_ms3_geocsv_slist_tspair_roundtrips_under_miri() {
        let t0: UtcTimestamp = "2022-07-04T08:15:30.125000Z".parse().unwrap();
        let ch = ChannelId::new("IU", "ANMO", "10", "BHZ");
        let data: Vec<f64> = (0..35).map(|i| (i * i - 10 * i) as f64).collect();
        let mut tr = Trace::new(TraceStats::new(ch.clone(), t0, 40.0), data.clone());

        // SAC Binary LE & BE + Auto-sniffer
        for endian in [SacEndian::Little, SacEndian::Big] {
            let bin = write_sac_bytes(&tr, endian).unwrap();
            let auto_st = read_waveforms_bytes(&bin).unwrap();
            assert_eq!(auto_st.len(), 1);
            assert_eq!(auto_st[0].data, data);
            assert_eq!(auto_st[0].stats.channel, ch);
        }

        // SAC v7 22-double footer round-trip with sub-f32 precision timestamp/coordinates
        let mut hdr_v7 = earthsnail_core::SacHeader::new();
        hdr_v7.nvhdr = 7;
        hdr_v7.set_float("stla", 34.945_912_345_678_9).unwrap();
        hdr_v7.set_float("stlo", -106.457_198_765_432_1).unwrap();
        tr.stats.sac = Some(hdr_v7);
        let v7_bytes = write_sac_bytes(&tr, SacEndian::Little).unwrap();
        let tr_v7 = read_sac_bytes(&v7_bytes).unwrap();
        let stla_rt = tr_v7.stats.sac.as_ref().unwrap().stla;
        assert!((stla_rt - 34.945_912_345_678_9).abs() < 1e-14);

        // SAC Alpha round-trip + sniffer
        let alpha_bytes = write_sac_alpha_bytes(&tr).unwrap();
        assert_eq!(
            detect_waveform_format(&alpha_bytes),
            Some(WaveformFormat::SacAlpha)
        );
        let alpha_st = read_waveforms_bytes(&alpha_bytes).unwrap();
        assert_eq!(alpha_st[0].npts(), data.len());

        // miniSEED v2 with gap preservation (`get_gaps` compatible)
        let t_gap = t0.add_nanos(5_000_000_000); // +5.0s > 35/40s -> ~4.125s gap!
        let tr2 = Trace::new(
            TraceStats::new(ch.clone(), t_gap, 40.0),
            vec![7.0, 8.0, 9.0],
        );
        let gapped_stream = Stream::from_traces(vec![tr.clone(), tr2]);
        let ms2_bytes =
            write_mseed_stream_bytes(&gapped_stream, MseedWriteOptions::default()).unwrap();
        let ms2_rt = read_waveforms_bytes(&ms2_bytes).unwrap();
        assert_eq!(ms2_rt.len(), 2);
        assert_eq!(ms2_rt.get_gaps().len(), 1);

        // miniSEED v3 with JSON extra headers & CRC-32C tamper rejection
        let ms3_opts = Mseed3WriteOptions {
            extra_headers_json: Some(alloc::string::String::from("{\"agency\":\"USGS\"}")),
            ..Mseed3WriteOptions::default()
        };
        let mut ms3_bytes = write_mseed3_stream_bytes(&gapped_stream, &ms3_opts).unwrap();
        assert_eq!(
            detect_waveform_format(&ms3_bytes),
            Some(WaveformFormat::MiniSeed3)
        );
        let ms3_recs = read_mseed3_records(&ms3_bytes).unwrap();
        assert_eq!(
            ms3_recs[0].extra_headers_json.as_deref(),
            Some("{\"agency\":\"USGS\"}")
        );
        let ms3_rt = read_waveforms_bytes(&ms3_bytes).unwrap();
        assert_eq!(ms3_rt.len(), 2);
        let last_idx = ms3_bytes.len() - 1;
        ms3_bytes[last_idx] ^= 0x80;
        assert!(read_mseed3_bytes(&ms3_bytes).is_err());

        // GeoCSV 1-col & 2-col, SLIST, TSPAIR round-trips via sniffer
        for two_col in [false, true] {
            let csv = write_geocsv_trace(&tr, two_col).unwrap();
            let csv_st = read_waveforms_bytes(csv.as_bytes()).unwrap();
            assert_eq!(csv_st[0].data, data);
        }
        let slist = write_slist_stream(&Stream::from_traces(vec![tr.clone()])).unwrap();
        assert_eq!(
            detect_waveform_format(slist.as_bytes()),
            Some(WaveformFormat::SList)
        );
        assert_eq!(
            read_waveforms_bytes(slist.as_bytes()).unwrap()[0].data,
            data
        );

        let tspair = write_tspair_stream(&Stream::from_traces(vec![tr])).unwrap();
        assert_eq!(
            detect_waveform_format(tspair.as_bytes()),
            Some(WaveformFormat::TsPair)
        );
        assert_eq!(
            read_waveforms_bytes(tspair.as_bytes()).unwrap()[0].data,
            data
        );
    }

    #[test]
    fn xml_sacpz_stationxml_quakeml_and_mseed_encodings_under_miri() {
        use earthsnail_core::{
            Arrival, ChannelMetadata, ChannelResponse, Complex64, EvaluationMode, Event,
            EventCatalog, FirStage, FirSymmetry, FocalMechanism, KinematicUnit, Magnitude,
            MomentTensor6, NodalPlane, Origin, OriginQuality, Pick, PickOnset, PickPolarity,
            PolesZerosStage, PzTransferFunctionType, ResponseStage, StageGain, StationInventory,
        };

        // 1. Classic SACPZ with implicit omitted origin zeros (`ZEROS 3` with only 1 explicit zero!)
        let sacpz_implicit = "\
* NETWORK     : IU
* STATION     : ANMO
* LOCATION    : 00
* CHANNEL     : BHZ
* INPUT UNIT  : M
ZEROS 3
 -31.63 0.0
POLES 2
 -0.037004 +0.037016
 -0.037004 -0.037016
CONSTANT 1.5e9
";
        let pz_rec = read_sacpz_str(sacpz_implicit).unwrap();
        assert_eq!(pz_rec.zeros.len(), 3);
        assert_eq!(pz_rec.zeros[1], Complex64::ZERO);
        assert_eq!(pz_rec.zeros[2], Complex64::ZERO);
        let sacpz_out = write_sacpz_str(&pz_rec.response, pz_rec.metadata.as_ref()).unwrap();
        let pz_rt = read_sacpz_str(&sacpz_out).unwrap();
        assert!((pz_rt.constant - 1.5e9).abs() < 1e3);

        // 2. Multi-stage StationXML (PolesZeros + Odd FIR) round-trip
        let mut pz_st = PolesZerosStage {
            pz_transfer_function_type: PzTransferFunctionType::RadPerSec,
            a0: 1.0,
            normalization_freq: 1.0,
            zeros: vec![Complex64::ZERO, Complex64::ZERO],
            poles: vec![
                Complex64::new(-0.037, 0.037),
                Complex64::new(-0.037, -0.037),
            ],
            stage_gain: 1500.0,
            input_sample_rate_hz: Some(80.0),
        };
        pz_st.recompute_a0();
        let fir_st = FirStage {
            symmetry: FirSymmetry::Odd,
            coefficients: vec![0.25, 0.5], // expands to [0.25, 0.5, 0.25]
            input_sample_rate_hz: 80.0,
            decimation_factor: 2,
            delay_s: 0.0125,
            correction_s: 0.0125,
            stage_gain: 1_000_000.0,
            gain_frequency_hz: 0.0,
        };
        let resp = ChannelResponse {
            input_units: KinematicUnit::Velocity,
            output_units: alloc::string::String::from("COUNTS"),
            instrument_sensitivity: Some(StageGain {
                value: 1.5e9,
                frequency_hz: 1.0,
            }),
            stages: vec![ResponseStage::PolesZeros(pz_st), ResponseStage::Fir(fir_st)],
        };
        let mut inv = StationInventory::new();
        inv.source = Some(alloc::string::String::from("EarthScope & <Test>"));
        inv.add_channel(ChannelMetadata {
            channel: ChannelId::new("IU", "ANMO", "00", "BHZ"),
            latitude: 34.9459,
            longitude: -106.4572,
            elevation_m: 1820.0,
            depth_m: 100.0,
            azimuth_deg: Some(0.0),
            dip_deg: Some(-90.0),
            sample_rate_hz: Some(40.0),
            start_date: Some("2020-01-01T00:00:00Z".parse().unwrap()),
            end_date: None,
            response: Some(resp.clone()),
        });
        let xml_str = write_stationxml_str(&inv).unwrap();
        let inv_rt = read_stationxml_str(&xml_str).unwrap();
        assert_eq!(inv_rt.source.as_deref(), Some("EarthScope & <Test>"));
        let h_orig = resp.eval_freq(1.0);
        let h_rt = inv_rt
            .channels()
            .next()
            .unwrap()
            .response
            .as_ref()
            .unwrap()
            .eval_freq(1.0);
        assert!((h_orig - h_rt).abs() / h_orig.abs() < 1e-12);

        // 3. QuakeML full event + arrival + pick + moment tensor round-trip
        let t_ev: UtcTimestamp = "2024-04-03T23:58:11.250Z".parse().unwrap();
        let mut org = Origin::from_depth_km(t_ev, 23.819, 121.562, 34.8);
        org.public_id = Some(alloc::string::String::from("smi:local/orig/1"));
        org.quality = Some(OriginQuality {
            associated_phase_count: Some(12),
            used_phase_count: Some(10),
            standard_error_s: Some(0.18),
            azimuthal_gap_deg: Some(45.0),
            minimum_distance_deg: Some(0.2),
            maximum_distance_deg: Some(12.5),
        });
        org.arrivals.push(Arrival {
            pick_id: Some(alloc::string::String::from("smi:local/pick/1")),
            phase: alloc::string::String::from("P"),
            azimuth_deg: Some(112.0),
            distance_deg: Some(1.4),
            takeoff_angle_deg: Some(52.0),
            time_residual_s: Some(-0.04),
            time_weight: Some(1.0),
        });
        let mut cat = EventCatalog::new();
        cat.description = Some(alloc::string::String::from("Hualien <Mw 7.4> & Sequence"));
        cat.push(Event {
            public_id: alloc::string::String::from("smi:local/ev/1"),
            event_type: Some(alloc::string::String::from("earthquake")),
            description: Some(alloc::string::String::from("TAIWAN REGION")),
            preferred_origin_id: Some(alloc::string::String::from("smi:local/orig/1")),
            preferred_magnitude_id: Some(alloc::string::String::from("smi:local/mag/1")),
            preferred_focal_mechanism_id: None,
            origins: vec![org],
            magnitudes: vec![Magnitude {
                public_id: Some(alloc::string::String::from("smi:local/mag/1")),
                mag: 7.4,
                mag_type: alloc::string::String::from("Mww"),
                origin_id: Some(alloc::string::String::from("smi:local/orig/1")),
                station_count: Some(84),
                uncertainty: Some(0.05),
            }],
            picks: vec![Pick {
                public_id: Some(alloc::string::String::from("smi:local/pick/1")),
                time: t_ev.add_nanos(21_500_000_000),
                channel: ChannelId::new("TW", "NACB", "00", "BHZ"),
                phase_hint: Some(alloc::string::String::from("P")),
                polarity: Some(PickPolarity::Positive),
                onset: Some(PickOnset::Impulsive),
                time_uncertainty_s: Some(0.02),
                evaluation_mode: Some(EvaluationMode::Manual),
            }],
            focal_mechanisms: vec![FocalMechanism {
                public_id: Some(alloc::string::String::from("smi:local/fm/1")),
                nodal_plane_1: Some(NodalPlane {
                    strike_deg: 26.0,
                    dip_deg: 49.0,
                    rake_deg: 69.0,
                }),
                nodal_plane_2: None,
                moment_tensor: Some(MomentTensor6::new(
                    1.2e20, -0.5e20, -0.7e20, 0.4e20, -0.3e20, 0.2e20,
                )),
                principal_axes: None,
            }],
        });
        let qml_txt = write_quakeml_str(&cat).unwrap();
        let cat_rt = read_quakeml_str(&qml_txt).unwrap();
        assert_eq!(cat_rt, cat);

        // 4. miniSEED v2 & v3 across Int16, Float32, Float64 + non-integer Blockette 100 sample rate
        let tr_frac = Trace::new(
            TraceStats::new(
                ChannelId::new("XX", "TEST", "01", "HHZ"),
                t_ev,
                33.333_332_061_767_58,
            ),
            vec![-100.0, 0.0, 250.0, -32768.0, 32767.0],
        );
        for enc in [
            MseedEncoding::Int16,
            MseedEncoding::Float32,
            MseedEncoding::Float64,
        ] {
            for bo in [MseedByteOrder::Big, MseedByteOrder::Little] {
                let b2 = write_mseed_trace_bytes(
                    &tr_frac,
                    MseedWriteOptions {
                        record_length: 256,
                        encoding: enc,
                        byte_order: bo,
                        quality_indicator: 'D',
                    },
                )
                .unwrap();
                let rt2 = read_mseed_bytes(&b2).unwrap();
                assert_eq!(rt2[0].data, tr_frac.data);
                assert!((rt2[0].stats.sampling_rate - tr_frac.stats.sampling_rate).abs() < 1e-5);
            }
            let b3 = write_mseed3_trace_bytes(
                &tr_frac,
                &Mseed3WriteOptions {
                    encoding: enc,
                    ..Mseed3WriteOptions::default()
                },
            )
            .unwrap();
            let rt3 = read_mseed3_bytes(&b3).unwrap();
            assert_eq!(rt3[0].data, tr_frac.data);
        }
    }
}
