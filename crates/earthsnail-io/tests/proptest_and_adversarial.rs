//! Gate 3 (`proptest` Lossless Codec Invariants) & Gate 4 (Adversarial Malformed-Binary Torture
//! Suite — `0` panics, `0` OOM) for `earthsnail-io`.

use earthsnail_core::{ChannelId, SacEndian, SacHeader, Trace, TraceStats, UtcTimestamp};
#[cfg(feature = "std")]
use earthsnail_io::SeedLinkPacket;
use earthsnail_io::{
    decode_steim1, decode_steim2, encode_steim1, encode_steim2, read_geocsv_bytes,
    read_mseed3_bytes, read_mseed_bytes, read_quakeml_bytes, read_sac_alpha_bytes, read_sac_bytes,
    read_sacpz_bytes, read_stationxml_bytes, read_waveforms_bytes, write_mseed3_trace_bytes,
    write_mseed_trace_bytes, write_sac_bytes, Mseed3WriteOptions, MseedByteOrder, MseedEncoding,
    MseedWriteOptions,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Gate 3: Property-Based Lossless Codec Round-Trip Suites
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Arbitrary `i32` waveforms (including `i32::MIN`, `i32::MAX`, and >30-bit impulses) survive
    /// `Steim-1`, `miniSEED v2` (Steim-1 & Int32), and `miniSEED v3` (Steim-1 & Int32 + CRC-32C)
    /// with 0 bit error.
    #[test]
    fn prop_steim1_ms2_ms3_arbitrary_i32_lossless(
        samples in prop::collection::vec(any::<i32>(), 1..600),
        prev in any::<i32>(),
    ) {
        let frames = encode_steim1(&samples, prev);
        let decoded = decode_steim1(&frames, samples.len()).unwrap();
        prop_assert_eq!(&decoded, &samples);

        let t0 = UtcTimestamp::from_ymd_hms_nano(2023, 8, 19, 14, 22, 10, 456_000_000).unwrap();
        let ch = ChannelId::new("IU", "ANMO", "00", "BHZ");
        let f64_data: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        let tr = Trace::new(TraceStats::new(ch, t0, 100.0), f64_data.clone());

        for enc in [MseedEncoding::Steim1, MseedEncoding::Int32] {
            let ms2 = write_mseed_trace_bytes(
                &tr,
                MseedWriteOptions {
                    record_length: 512,
                    encoding: enc,
                    byte_order: MseedByteOrder::Big,
                    quality_indicator: 'D',
                },
            ).unwrap();
            let st2 = read_mseed_bytes(&ms2).unwrap();
            prop_assert_eq!(st2.len(), 1);
            prop_assert_eq!(&st2[0].data, &f64_data);
            prop_assert_eq!(st2[0].stats.starttime, t0);

            let ms3 = write_mseed3_trace_bytes(
                &tr,
                &Mseed3WriteOptions {
                    encoding: enc,
                    max_samples_per_record: 180,
                    ..Mseed3WriteOptions::default()
                },
            ).unwrap();
            let st3 = read_mseed3_bytes(&ms3).unwrap();
            prop_assert_eq!(st3.len(), 1);
            prop_assert_eq!(&st3[0].data, &f64_data);
            prop_assert_eq!(st3[0].stats.starttime, t0);
        }
    }

    /// Arbitrary waveforms exercising all 7 Steim-2 difference widths (`4, 5, 6, 8, 10, 15, 30`-bit)
    /// survive `Steim-2`, `miniSEED v2` across variable record lengths `{256, 512, 1024, 4096}`,
    /// and `miniSEED v3` with 0 sample difference.
    #[test]
    fn prop_steim2_all_seven_widths_and_variable_reclen_lossless(
        x0 in -500_000_000_i32..=500_000_000_i32,
        raw_diffs in prop::collection::vec(
            prop_oneof![
                -8_i32..=7_i32,                    // 4-bit (x7)
                -16_i32..=15_i32,                  // 5-bit (x6)
                -32_i32..=31_i32,                  // 6-bit (x5)
                -128_i32..=127_i32,                // 8-bit (x4)
                -512_i32..=511_i32,                // 10-bit (x3)
                -16_384_i32..=16_383_i32,          // 15-bit (x2)
                -536_870_912_i32..=536_870_911_i32 // 30-bit (x1)
            ],
            1..500
        ),
        rec_exp in 8_u32..=12_u32, // 256, 512, 1024, 2048, 4096
    ) {
        let mut samples = Vec::with_capacity(raw_diffs.len() + 1);
        let mut acc = x0 as i64;
        samples.push(x0);
        for d in raw_diffs {
            let next = acc + (d as i64);
            let clamped = next.clamp(i32::MIN as i64, i32::MAX as i64);
            // Verify clamped step remains within 30-bit Steim-2 range:
            let eff_d = clamped - acc;
            if (-536_870_912..=536_870_911).contains(&eff_d) {
                acc = clamped;
                samples.push(acc as i32);
            }
        }

        let frames = encode_steim2(&samples, 0).unwrap();
        let decoded = decode_steim2(&frames, samples.len()).unwrap();
        prop_assert_eq!(&decoded, &samples);

        let rec_len = 1_usize << rec_exp;
        let t0 = UtcTimestamp::from_ymd_hms_nano(2024, 2, 29, 23, 59, 58, 123_456_000).unwrap();
        let ch = ChannelId::new("G", "CAN", "00", "BHN");
        let f64_data: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        let tr = Trace::new(TraceStats::new(ch, t0, 40.0), f64_data.clone());

        let ms2 = write_mseed_trace_bytes(
            &tr,
            MseedWriteOptions {
                record_length: rec_len,
                encoding: MseedEncoding::Steim2,
                byte_order: MseedByteOrder::Big,
                quality_indicator: 'D',
            },
        ).unwrap();
        prop_assert_eq!(ms2.len() % rec_len, 0);
        let st2 = read_mseed_bytes(&ms2).unwrap();
        prop_assert_eq!(&st2[0].data, &f64_data);

        let ms3 = write_mseed3_trace_bytes(&tr, &Mseed3WriteOptions::default()).unwrap();
        let st3 = read_mseed3_bytes(&ms3).unwrap();
        prop_assert_eq!(&st3[0].data, &f64_data);
    }

    /// Arbitrary finite `f32` waveforms & coordinates round-trip losslessly across SAC Binary LE <-> BE.
    #[test]
    fn prop_sac_le_be_roundtrip(
        samples_f32 in prop::collection::vec(-1.0e20_f32..1.0e20_f32, 0..400),
        stla in -89.9_f32..89.9_f32,
        stlo in -179.9_f32..179.9_f32,
        evla in -89.9_f32..89.9_f32,
        evlo in -179.9_f32..179.9_f32,
    ) {
        let t0 = UtcTimestamp::from_ymd_hms_nano(2021, 5, 1, 6, 0, 0, 50_000_000).unwrap();
        let ch = ChannelId::new("US", "CCM", "00", "BHZ");
        let data: Vec<f64> = samples_f32.iter().map(|&x| x as f64).collect();
        let mut hdr = SacHeader::new();
        hdr.set_float("stla", stla as f64).unwrap();
        hdr.set_float("stlo", stlo as f64).unwrap();
        hdr.set_float("evla", evla as f64).unwrap();
        hdr.set_float("evlo", evlo as f64).unwrap();

        let mut stats = TraceStats::new(ch.clone(), t0, 50.0);
        stats.sac = Some(hdr);
        let tr = Trace::new(stats, data.clone());

        let le = write_sac_bytes(&tr, SacEndian::Little).unwrap();
        let be = write_sac_bytes(&tr, SacEndian::Big).unwrap();
        let tr_le = read_sac_bytes(&le).unwrap();
        let tr_be = read_sac_bytes(&be).unwrap();

        prop_assert_eq!(&tr_le.data, &data);
        prop_assert_eq!(&tr_be.data, &data);
        prop_assert_eq!(&tr_le.stats.channel, &ch);
        prop_assert_eq!(&tr_be.stats.channel, &ch);
    }

    /// Fuzzes all byte-oriented entrypoints with arbitrary random byte buffers (`0..2048` bytes)
    /// verifying 0 panics and bounded memory.
    #[test]
    fn prop_fuzz_untrusted_bytes_never_panic(
        blob in prop::collection::vec(any::<u8>(), 0..2048)
    ) {
        let _ = read_waveforms_bytes(&blob);
        let _ = read_sac_bytes(&blob);
        let _ = read_sac_alpha_bytes(&blob);
        let _ = read_mseed_bytes(&blob);
        let _ = read_mseed3_bytes(&blob);
        let _ = read_geocsv_bytes(&blob);
        let _ = read_sacpz_bytes(&blob);
        let _ = read_stationxml_bytes(&blob);
        let _ = read_quakeml_bytes(&blob);
        #[cfg(feature = "std")]
        {
            let _ = SeedLinkPacket::decode_frame(&blob);
        }
    }
}

// ---------------------------------------------------------------------------
// Gate 4: Deterministic Structure-Aware Adversarial Malformed Torture Corpus
// ---------------------------------------------------------------------------

#[test]
fn adversarial_malformed_binary_and_xml_torture_zero_panics_zero_oom() {
    let t0 = UtcTimestamp::from_ymd_hms_nano(2020, 1, 1, 0, 0, 0, 0).unwrap();
    let base_tr = Trace::new(
        TraceStats::new(ChannelId::new("IU", "ANMO", "00", "BHZ"), t0, 40.0),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );

    // 1. SAC Binary with bogus `NPTS = i32::MAX` (`2_147_483_647` -> 8 GiB allocation bomb attempt!)
    let mut sac_bomb = write_sac_bytes(&base_tr, SacEndian::Little).unwrap();
    sac_bomb[316..320].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(read_sac_bytes(&sac_bomb).is_err());
    assert!(read_waveforms_bytes(&sac_bomb).is_err());

    // 2. SAC Binary with negative `NPTS = -500`
    let mut sac_neg = write_sac_bytes(&base_tr, SacEndian::Little).unwrap();
    sac_neg[316..320].copy_from_slice(&(-500_i32).to_le_bytes());
    assert!(read_sac_bytes(&sac_neg).is_err());

    // 3. SAC Binary with `NaN` in `DELTA` (word 0) and `Inf` in sample 0
    let mut sac_nan_hdr = write_sac_bytes(&base_tr, SacEndian::Little).unwrap();
    sac_nan_hdr[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(read_sac_bytes(&sac_nan_hdr).is_err());

    let mut sac_inf_sample = write_sac_bytes(&base_tr, SacEndian::Little).unwrap();
    sac_inf_sample[632..636].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(read_sac_bytes(&sac_inf_sample).is_err());

    // 4. SAC Binary claiming `NVHDR = 7` without 176-byte v7 footer
    let mut sac_trunc_v7 = write_sac_bytes(&base_tr, SacEndian::Little).unwrap();
    sac_trunc_v7[304..308].copy_from_slice(&7_i32.to_le_bytes());
    assert!(read_sac_bytes(&sac_trunc_v7).is_err());

    // 5. Steim-2 reserved nibble combinations (`c=10 dnib=00` and `c=11 dnib=11`)
    let mut bad_s2_frame = [0_u8; 64];
    // W0: word 3 has c = 0b10 -> shift = 30 - 6 = 24
    bad_s2_frame[0..4].copy_from_slice(&(0b10_u32 << 24).to_be_bytes());
    // W3 has dnib = 0b00
    bad_s2_frame[12..16].copy_from_slice(&0_u32.to_be_bytes());
    assert!(decode_steim2(&bad_s2_frame, 1).is_err());

    bad_s2_frame[0..4].copy_from_slice(&(0b11_u32 << 24).to_be_bytes());
    bad_s2_frame[12..16].copy_from_slice(&(0b11_u32 << 30).to_be_bytes());
    assert!(decode_steim2(&bad_s2_frame, 1).is_err());

    // 6. miniSEED v2 with self-referential Blockette pointer loop (`next_off = 48` at offset `48`!)
    let mut ms2_loop = write_mseed_trace_bytes(&base_tr, MseedWriteOptions::default()).unwrap();
    ms2_loop[50..52].copy_from_slice(&48_u16.to_be_bytes());
    assert!(read_mseed_bytes(&ms2_loop).is_err());

    // 7. miniSEED v2 with bogus Blockette 1000 record length exponent `25` (`32 MiB`)
    let mut ms2_bad_exp = write_mseed_trace_bytes(&base_tr, MseedWriteOptions::default()).unwrap();
    ms2_bad_exp[54] = 25;
    assert!(read_mseed_bytes(&ms2_bad_exp).is_err());

    // 8. miniSEED v3 with single-bit flip in header/payload caught by CRC-32C, and huge `data_len = u32::MAX`
    let valid_ms3 = write_mseed3_trace_bytes(&base_tr, &Mseed3WriteOptions::default()).unwrap();
    for flip_pos in [3_usize, 10, 20, 35, valid_ms3.len() - 1] {
        let mut mutated = valid_ms3.clone();
        mutated[flip_pos] ^= 0x04;
        assert!(read_mseed3_bytes(&mutated).is_err());
    }
    let mut ms3_bomb = valid_ms3;
    ms3_bomb[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(read_mseed3_bytes(&ms3_bomb).is_err());

    // 9. XML Billion Laughs DTD entity bomb, deep nesting (>128), unclosed/mismatched tags, invalid char refs
    let billion_laughs = br#"<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<FDSNStationXML>&lol2;</FDSNStationXML>"#;
    assert!(read_stationxml_bytes(billion_laughs).is_err());

    let mismatched_xml = b"<FDSNStationXML><Network code=\"IU\"></Station></FDSNStationXML>";
    assert!(read_stationxml_bytes(mismatched_xml).is_err());

    let unclosed_xml = b"<q:quakeml><eventParameters><event publicID=\"1\">";
    assert!(read_quakeml_bytes(unclosed_xml).is_err());

    let bad_entity_xml = b"<FDSNStationXML><Source>&nonexistent_entity;</Source></FDSNStationXML>";
    assert!(read_stationxml_bytes(bad_entity_xml).is_err());

    let bad_char_ref_xml = b"<FDSNStationXML><Source>&#xD800;</Source></FDSNStationXML>";
    assert!(read_stationxml_bytes(bad_char_ref_xml).is_err());

    let mut deep_xml = String::new();
    for _ in 0..140 {
        deep_xml.push_str("<a>");
    }
    for _ in 0..140 {
        deep_xml.push_str("</a>");
    }
    assert!(read_stationxml_bytes(deep_xml.as_bytes()).is_err());

    // 10. SACPZ with `ZEROS 99999999` allocation bomb attempt
    assert!(read_sacpz_bytes(b"ZEROS 99999999\nCONSTANT 1.0\n").is_err());

    // 11. SeedLink v4 packet with `payload_len = u32::MAX`
    #[cfg(feature = "std")]
    {
        let mut sl4_bomb = vec![0_u8; 24];
        sl4_bomb[0..4].copy_from_slice(b"SE3D");
        sl4_bomb[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(SeedLinkPacket::decode_frame(&sl4_bomb).is_err());
    }
}
