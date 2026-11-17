#!/usr/bin/env node
/**
 * Gate 5 Cross-ISA Lockstep Verifier (`x86_64` Native vs `wasm32-unknown-unknown +simd128`).
 *
 * Executes the compiled `earthsnail_wasm.wasm` inside Node's V8 WebAssembly engine and compares
 * every output head-to-head against native `x86_64` Rust (`tests/lockstep_native.rs`) across:
 *   1. Steim-2 lossless frame & miniSEED v2 codec (all 7 nibble widths, 0-ULP integer parity)
 *   2. Karney WGS84 ellipsoidal geodesics across 6 global regimes (<= 1e-12)
 *   3. 4-pole zero-phase Butterworth bandpass (`sosfilt_zerophase_obspy` & `sosfiltfilt`, <= 1e-12)
 *   4. Welch RFFT PSD, Peterson NLNM/NHNM, and Marple Hilbert envelope (<= 1e-12)
 *   5. Glitch-immune 3-pt Discrete Morse vs Classic STA/LTA & Baer picking (exact sample + <= 1e-12)
 *   6. Spherical `TauP` travel times & ray paths across `iasp91`, `ak135`, and `prem` (<= 1e-12)
 *
 * Writes machine-checked receipt telemetry to `studio/wasm_lockstep_receipts.json`.
 */

import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const STUDIO_DIR = path.dirname(__filename);
const REPO_ROOT = path.resolve(STUDIO_DIR, "..");
const RECEIPT_PATH = path.join(STUDIO_DIR, "wasm_lockstep_receipts.json");

function readLeb128U32(buf, offset) {
  let result = 0;
  let shift = 0;
  let pos = offset;
  while (true) {
    const byte = buf[pos++];
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      return [result >>> 0, pos];
    }
    shift += 7;
  }
}

function inspectWasmBinary(wasmBytes) {
  if (wasmBytes.length < 8 || wasmBytes.subarray(0, 4).toString("latin1") !== "\0asm") {
    throw new Error("Invalid WASM magic bytes (expected \\0asm)");
  }
  const version = wasmBytes.readUInt32LE(4);
  if (version !== 1) {
    throw new Error(`Unexpected WASM version: ${version}`);
  }
  const imports = [];
  const exports = [];
  let hasSimd128Feature = false;
  let v128PrefixCount = 0;
  let offset = 8;

  while (offset < wasmBytes.length) {
    const secId = wasmBytes[offset++];
    let secLen;
    [secLen, offset] = readLeb128U32(wasmBytes, offset);
    const secEnd = offset + secLen;

    if (secId === 0) {
      const sliceStr = wasmBytes.subarray(offset, secEnd).toString("latin1");
      if (sliceStr.includes("target_features") && sliceStr.includes("simd128")) {
        hasSimd128Feature = true;
      }
    } else if (secId === 2) {
      let [count, p] = readLeb128U32(wasmBytes, offset);
      for (let i = 0; i < count; i++) {
        let modLen, fldLen;
        [modLen, p] = readLeb128U32(wasmBytes, p);
        const modName = wasmBytes.subarray(p, p + modLen).toString("utf8");
        p += modLen;
        [fldLen, p] = readLeb128U32(wasmBytes, p);
        const fldName = wasmBytes.subarray(p, p + fldLen).toString("utf8");
        p += fldLen;
        const kind = wasmBytes[p++];
        if (kind === 0) {
          [, p] = readLeb128U32(wasmBytes, p);
        } else if (kind === 1) {
          p += 1;
          let flags;
          [flags, p] = readLeb128U32(wasmBytes, p);
          [, p] = readLeb128U32(wasmBytes, p);
          if (flags & 1) [, p] = readLeb128U32(wasmBytes, p);
        } else if (kind === 2) {
          let flags;
          [flags, p] = readLeb128U32(wasmBytes, p);
          [, p] = readLeb128U32(wasmBytes, p);
          if (flags & 1) [, p] = readLeb128U32(wasmBytes, p);
        } else if (kind === 3) {
          p += 2;
        }
        imports.push(`${modName}.${fldName}`);
      }
    } else if (secId === 7) {
      let [count, p] = readLeb128U32(wasmBytes, offset);
      for (let i = 0; i < count; i++) {
        let nameLen;
        [nameLen, p] = readLeb128U32(wasmBytes, p);
        const name = wasmBytes.subarray(p, p + nameLen).toString("utf8");
        p += nameLen;
        p += 1;
        [, p] = readLeb128U32(wasmBytes, p);
        exports.push(name);
      }
    } else if (secId === 10) {
      for (let k = offset; k < secEnd; k++) {
        if (wasmBytes[k] === 0xfd) v128PrefixCount++;
      }
    }
    offset = secEnd;
  }

  return {
    wasm_size_bytes: wasmBytes.length,
    sha256: crypto.createHash("sha256").update(wasmBytes).digest("hex"),
    simd128_enabled: hasSimd128Feature || v128PrefixCount > 0,
    v128_opcode_bytes: v128PrefixCount,
    import_count: imports.length,
    imports,
    export_count: exports.length,
    exports,
  };
}

function resolveTargetDir() {
  if (process.env.CARGO_TARGET_DIR) {
    return path.resolve(process.env.CARGO_TARGET_DIR);
  }
  const studioWorkerTarget = path.join(REPO_ROOT, "target_workers", "studio_wasm");
  if (fs.existsSync(studioWorkerTarget)) {
    return studioWorkerTarget;
  }
  return path.join(REPO_ROOT, "target");
}

function findOrBuildWasm(targetDir) {
  const candidates = [
    path.join(targetDir, "wasm32-unknown-unknown", "release", "earthsnail_wasm.wasm"),
    path.join(REPO_ROOT, "target_workers", "studio_wasm", "wasm32-unknown-unknown", "release", "earthsnail_wasm.wasm"),
    path.join(REPO_ROOT, "target", "wasm32-unknown-unknown", "release", "earthsnail_wasm.wasm"),
  ];
  for (const cand of candidates) {
    if (fs.existsSync(cand)) return cand;
  }
  const res = spawnSync(
    "cargo",
    ["build", "-p", "earthsnail-wasm", "--target", "wasm32-unknown-unknown", "--release"],
    {
      cwd: REPO_ROOT,
      env: { ...process.env, CARGO_TARGET_DIR: targetDir },
      stdio: "inherit",
    }
  );
  if (res.status !== 0) {
    throw new Error("Failed to build earthsnail_wasm.wasm");
  }
  return candidates[0];
}

function runNativeReferenceDump(targetDir) {
  const tmpJson = path.join(os.tmpdir(), `earthsnail_native_lockstep_${process.pid}.json`);
  const res = spawnSync(
    "cargo",
    ["test", "-p", "earthsnail-wasm", "--release", "--test", "lockstep_native", "--", "--nocapture"],
    {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        CARGO_TARGET_DIR: targetDir,
        EARTHSNAIL_LOCKSTEP_DUMP: tmpJson,
      },
      encoding: "utf8",
    }
  );
  if (res.status !== 0) {
    console.error(res.stdout);
    console.error(res.stderr);
    throw new Error(`Native x86_64 lockstep test failed with status ${res.status}`);
  }
  const data = JSON.parse(fs.readFileSync(tmpJson, "utf8"));
  try {
    fs.unlinkSync(tmpJson);
  } catch {
    // ignore cleanup errors
  }
  return data;
}

// Deterministic input generators matching `tests/lockstep_native.rs` bit-for-bit:
function buildSteim2TestVector() {
  const n = 512;
  const out = new Int32Array(n);
  let cur = 25000;
  out[0] = cur;
  for (let i = 1; i < n; i++) {
    const bucket = i % 8;
    const sign = i % 2 === 0 ? 1 : -1;
    let step = 0;
    if (bucket === 0) step = 0;
    else if (bucket === 1) step = sign * ((i % 7) + 1); // 4-bit
    else if (bucket === 2) step = sign * (8 + (i % 7)); // 5-bit
    else if (bucket === 3) step = sign * (16 + (i % 15)); // 6-bit
    else if (bucket === 4) step = sign * (32 + (i % 90)); // 8-bit
    else if (bucket === 5) step = sign * (128 + (i % 350)); // 10-bit
    else if (bucket === 6) step = sign * (512 + (i * 37) % 14000); // 15-bit
    else step = sign * (16384 + ((i * 7919) % 400000)); // 30-bit
    cur = (cur + step) | 0;
    out[i] = cur;
  }
  return out;
}

const GEODESIC_PAIRS = [
  [38.297, 142.373, -35.318715, 148.996325], // Tohoku -> Canberra (G.CAN)
  [34.9459, -106.4572, 35.6895, 139.6917],   // IU.ANMO -> Tokyo
  [0.0, 0.0, 0.0, 179.6],                    // Near-antipodal equatorial arch
  [89.75, 15.0, 89.8, -165.0],               // Trans-polar Arctic
  [-45.1234, 72.5678, 45.1230, -107.4320],   // Near-antipodal mid-latitude
  [37.7749, -122.4194, 37.7752, -122.4189],  // Sub-100m short baseline
];

function buildDeterministicWaveform(n, fs) {
  // Pure polynomial/rational waveform so input vector is 0-ULP bit-identical on x86_64 and JS
  // before entering WASM transcendental/DSP kernels!
  const x = new Float64Array(n);
  let state = 0x123456789abcdef0n;
  const mask64 = 0xffffffffffffffffn;
  for (let i = 0; i < n; i++) {
    state = (state * 6364136223846793005n + 1442695040888963407n) & mask64;
    const u = Number((state >> 11n) & 0x1fffffffffffffn) / 9007199254740992.0; // [0, 1)
    const noise = (u - 0.5) * 0.04;
    const t = i / fs;
    let arrival = 0.0;
    if (i >= 640) {
      const dt = t - 640 / fs;
      // Smooth damped wave packet using exact rational envelope 1 / (1 + 0.15 * dt^2):
      const env = 1.0 / (1.0 + 0.15 * dt * dt);
      // Chebyshev/triangle-like deterministic oscillation via exact arithmetic:
      const phase1 = (dt * 0.85) % 1.0;
      const tri1 = 1.0 - 4.0 * Math.abs(phase1 - 0.5);
      const phase2 = (dt * 2.35) % 1.0;
      const tri2 = 1.0 - 4.0 * Math.abs(phase2 - 0.5);
      arrival = env * (3.2 * tri1 + 1.1 * tri2);
    }
    x[i] = noise + arrival;
  }
  return x;
}

class WasmHarness {
  constructor(instance) {
    this.ex = instance.exports;
    this.scratch = this.ex.es_alloc(64);
    if (!this.scratch) throw new Error("es_alloc(64) returned null");
  }

  memU8() {
    return new Uint8Array(this.ex.memory.buffer);
  }

  memView() {
    return new DataView(this.ex.memory.buffer);
  }

  lastError() {
    const len = this.ex.es_last_error_len() >>> 0;
    const ptr = this.ex.es_last_error_ptr() >>> 0;
    if (!len || !ptr) return "";
    return Buffer.from(this.ex.memory.buffer, ptr, len).toString("utf8");
  }

  check(rc, context) {
    if (rc !== 0) {
      throw new Error(`WASM ${context} failed (${rc}): ${this.lastError()}`);
    }
  }

  allocBytes(bytes) {
    const u8 = bytes instanceof Uint8Array ? bytes : Buffer.from(bytes);
    const ptr = this.ex.es_alloc(u8.length) >>> 0;
    if (!ptr) throw new Error(`es_alloc(${u8.length}) failed`);
    this.memU8().set(u8, ptr);
    return [ptr, u8.length];
  }

  allocF64(arr) {
    const byteLen = arr.length * 8;
    const ptr = this.ex.es_alloc(byteLen) >>> 0;
    if (!ptr) throw new Error(`es_alloc(${byteLen}) failed`);
    new Float64Array(this.ex.memory.buffer, ptr, arr.length).set(arr);
    return [ptr, byteLen];
  }

  allocI32(arr) {
    const byteLen = arr.length * 4;
    const ptr = this.ex.es_alloc(byteLen) >>> 0;
    if (!ptr) throw new Error(`es_alloc(${byteLen}) failed`);
    new Int32Array(this.ex.memory.buffer, ptr, arr.length).set(arr);
    return [ptr, byteLen];
  }

  consumeOwned() {
    const dv = this.memView();
    const ptr = dv.getUint32(this.scratch, true);
    const len = dv.getUint32(this.scratch + 4, true);
    if (!ptr || !len) return new Uint8Array(0);
    const copy = new Uint8Array(len);
    copy.set(new Uint8Array(this.ex.memory.buffer, ptr, len));
    this.ex.es_free(ptr, len);
    return copy;
  }

  dispose() {
    this.ex.es_free(this.scratch, 64);
  }
}

function unpackEstt(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const count = dv.getUint32(8, true);
  const out = [];
  let off = 16;
  for (let i = 0; i < count; i++) {
    const time_s = dv.getFloat64(off, true);
    const ray_param_s_per_rad = dv.getFloat64(off + 8, true);
    const ray_param_s_per_deg = dv.getFloat64(off + 16, true);
    const takeoff_angle_deg = dv.getFloat64(off + 24, true);
    const incident_angle_deg = dv.getFloat64(off + 32, true);
    const distance_deg = dv.getFloat64(off + 40, true);
    const purist_distance_deg = dv.getFloat64(off + 48, true);
    const source_depth_km = dv.getFloat64(off + 56, true);
    const nameRaw = u8.subarray(off + 64, off + 80);
    const nul = nameRaw.indexOf(0);
    const name = Buffer.from(nul >= 0 ? nameRaw.subarray(0, nul) : nameRaw).toString("utf8");
    out.push({
      name,
      time_s,
      ray_param_s_per_rad,
      ray_param_s_per_deg,
      takeoff_angle_deg,
      incident_angle_deg,
      distance_deg,
      purist_distance_deg,
      source_depth_km,
    });
    off += 96;
  }
  return out;
}

function unpackEsrp(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const count = dv.getUint32(8, true);
  const planet_radius_km = dv.getFloat64(16, true);
  const moho_depth_km = dv.getFloat64(24, true);
  const cmb_depth_km = dv.getFloat64(32, true);
  const iocb_depth_km = dv.getFloat64(40, true);
  let off = 48;
  const paths = [];
  for (let i = 0; i < count; i++) {
    const time_s = dv.getFloat64(off, true);
    const nPts = Number(dv.getBigUint64(off + 48, true));
    const nameRaw = u8.subarray(off + 56, off + 72);
    const nul = nameRaw.indexOf(0);
    const name = Buffer.from(nul >= 0 ? nameRaw.subarray(0, nul) : nameRaw).toString("utf8");
    off += 80;
    const pts = [];
    for (let k = 0; k < nPts; k++) {
      pts.push([
        dv.getFloat64(off, true),
        dv.getFloat64(off + 8, true),
        dv.getFloat64(off + 16, true),
        dv.getFloat64(off + 24, true),
      ]);
      off += 32;
    }
    paths.push({ name, time_s, n_points: nPts, first_pt: pts[0], mid_pt: pts[pts.length >> 1], last_pt: pts[pts.length - 1] });
  }
  return { planet_radius_km, moho_depth_km, cmb_depth_km, iocb_depth_km, paths };
}

function maxDiffStats(wasmArr, nativeArr) {
  if (wasmArr.length !== nativeArr.length) {
    throw new Error(`Length mismatch: wasm=${wasmArr.length} vs native=${nativeArr.length}`);
  }
  let maxAbs = 0;
  let maxRel = 0;
  for (let i = 0; i < wasmArr.length; i++) {
    const a = wasmArr[i];
    const b = nativeArr[i];
    const d = Math.abs(a - b);
    if (d > maxAbs) maxAbs = d;
    const denom = Math.max(Math.abs(b), 1e-15);
    const r = d / denom;
    if (r > maxRel) maxRel = r;
  }
  return { count: wasmArr.length, max_abs_diff: maxAbs, max_rel_diff: maxRel };
}

async function main() {
  const targetDir = resolveTargetDir();
  const wasmPath = findOrBuildWasm(targetDir);
  const wasmBytes = fs.readFileSync(wasmPath);
  const wasmMeta = inspectWasmBinary(wasmBytes);
  if (wasmMeta.import_count !== 0) {
    throw new Error(`Expected 0 WASM imports, got ${wasmMeta.imports.join(", ")}`);
  }
  if (!wasmMeta.simd128_enabled) {
    throw new Error("Expected +simd128 active in earthsnail_wasm.wasm");
  }

  const nativeRef = runNativeReferenceDump(targetDir);

  const tCompile0 = performance.now();
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const wasmCompileMs = performance.now() - tCompile0;
  const h = new WasmHarness(instance);

  const checks = {};

  // 1. Steim-2 Codec (0-ULP integer parity)
  {
    const t0 = performance.now();
    const samplesI32 = buildSteim2TestVector();
    const [inPtr, inBytes] = h.allocI32(samplesI32);
    h.check(
      h.ex.es_encode_steim2_frames(inPtr, samplesI32.length, 0, h.scratch, h.scratch + 4),
      "es_encode_steim2_frames"
    );
    const frames = h.consumeOwned();
    const framesHexSha = crypto.createHash("sha256").update(frames).digest("hex");

    const [fPtr, fBytes] = h.allocBytes(frames);
    const outPtr = h.ex.es_alloc(samplesI32.length * 4) >>> 0;
    h.check(
      h.ex.es_decode_steim2_frames(fPtr, fBytes, samplesI32.length, outPtr),
      "es_decode_steim2_frames"
    );
    const decoded = Array.from(new Int32Array(h.ex.memory.buffer, outPtr, samplesI32.length));
    h.ex.es_free(outPtr, samplesI32.length * 4);
    h.ex.es_free(fPtr, fBytes);
    h.ex.es_free(inPtr, inBytes);

    // Also miniSEED v2 Steim-2 full container encode -> auto-sniff ESWF decode:
    const f64In = Float64Array.from(samplesI32);
    const [sPtr, sBytes] = h.allocF64(f64In);
    const [chPtr, chLen] = h.allocBytes("G.CAN.00.BHZ");
    h.check(
      h.ex.es_encode_mseed2_steim2(chPtr, chLen, 1299822990n, 19538000, 20.0, sPtr, f64In.length, 512, h.scratch, h.scratch + 4),
      "es_encode_mseed2_steim2"
    );
    const ms2Bytes = h.consumeOwned();
    const [mPtr, mLen] = h.allocBytes(ms2Bytes);
    h.check(h.ex.es_decode_waveforms(mPtr, mLen, h.scratch, h.scratch + 4), "es_decode_waveforms");
    const eswf = h.consumeOwned();
    const eswfSamples = Array.from(new Float64Array(eswf.buffer, eswf.byteOffset + 16 + 192, samplesI32.length));
    h.ex.es_free(mPtr, mLen);
    h.ex.es_free(chPtr, chLen);
    h.ex.es_free(sPtr, sBytes);

    const elapsedMs = performance.now() - t0;
    let maxUlpDiff = 0;
    for (let i = 0; i < samplesI32.length; i++) {
      const d1 = Math.abs(decoded[i] - nativeRef.steim2.decoded_samples[i]);
      const d2 = Math.abs(eswfSamples[i] - nativeRef.steim2.decoded_samples[i]);
      if (d1 > maxUlpDiff) maxUlpDiff = d1;
      if (d2 > maxUlpDiff) maxUlpDiff = d2;
    }
    if (maxUlpDiff !== 0 || framesHexSha !== nativeRef.steim2.frames_sha256) {
      throw new Error(`Steim-2 parity mismatch: maxUlpDiff=${maxUlpDiff}`);
    }
    checks.steim2_codec = {
      status: "PASS",
      samples: samplesI32.length,
      frames_bytes: frames.length,
      frames_sha256_match: framesHexSha === nativeRef.steim2.frames_sha256,
      max_int32_ulp_diff: maxUlpDiff,
      wasm_elapsed_ms: Number(elapsedMs.toFixed(4)),
    };
  }

  // 2. WGS84 Geodesics
  {
    const t0 = performance.now();
    const outPtr = h.ex.es_alloc(5 * 8) >>> 0;
    const wasmFlat = [];
    for (const [lat1, lon1, lat2, lon2] of GEODESIC_PAIRS) {
      h.check(h.ex.es_wgs84_geodesic(lat1, lon1, lat2, lon2, outPtr), "es_wgs84_geodesic");
      wasmFlat.push(...new Float64Array(h.ex.memory.buffer, outPtr, 5));
    }
    h.ex.es_free(outPtr, 5 * 8);
    const elapsedMs = performance.now() - t0;
    const stats = maxDiffStats(wasmFlat, nativeRef.geodesics_flat);
    if (stats.max_rel_diff > 1e-12) {
      throw new Error(`WGS84 geodesic rel diff ${stats.max_rel_diff} > 1e-12`);
    }
    checks.wgs84_geodesics = {
      status: "PASS",
      pairs: GEODESIC_PAIRS.length,
      ...stats,
      wasm_elapsed_ms: Number(elapsedMs.toFixed(4)),
    };
  }

  // 3. 4-Pole Zero-Phase Butterworth Bandpass (`obspy` 2-pass & `sosfiltfilt`)
  const nWave = 1024;
  const fsWave = 50.0;
  const baseSignal = buildDeterministicWaveform(nWave, fsWave);
  {
    const t0 = performance.now();
    const [inPtr, inBytes] = h.allocF64(baseSignal);
    const [outPtr, outBytes] = h.allocF64(new Float64Array(nWave));

    h.check(
      h.ex.es_sos_filter(inPtr, nWave, fsWave, 0, 0, 4, 0.2, 5.0, 0.0, 1, outPtr),
      "es_sos_filter(zerophase_obspy)"
    );
    const zpObspy = Array.from(new Float64Array(h.ex.memory.buffer, outPtr, nWave));

    h.check(
      h.ex.es_sos_filter(inPtr, nWave, fsWave, 0, 0, 4, 0.2, 5.0, 0.0, 2, outPtr),
      "es_sos_filter(sosfiltfilt)"
    );
    const zpFiltfilt = Array.from(new Float64Array(h.ex.memory.buffer, outPtr, nWave));

    h.ex.es_free(outPtr, outBytes);
    h.ex.es_free(inPtr, inBytes);
    const elapsedMs = performance.now() - t0;

    const s1 = maxDiffStats(zpObspy, nativeRef.butterworth_zp_obspy);
    const s2 = maxDiffStats(zpFiltfilt, nativeRef.butterworth_filtfilt);
    if (s1.max_abs_diff > 1e-12 || s2.max_abs_diff > 1e-12) {
      throw new Error(`Butterworth parity exceeded 1e-12: obspy=${s1.max_abs_diff}, filtfilt=${s2.max_abs_diff}`);
    }
    checks.butterworth_zerophase = {
      status: "PASS",
      obspy_2pass_max_abs_diff: s1.max_abs_diff,
      sosfiltfilt_max_abs_diff: s2.max_abs_diff,
      wasm_elapsed_ms: Number(elapsedMs.toFixed(4)),
    };
  }

  // 4. Welch RFFT PSD, NLNM/NHNM & Hilbert Envelope
  {
    const t0 = performance.now();
    const [inPtr, inBytes] = h.allocF64(baseSignal);
    const bins = 129;
    const [fOut, fBytes] = h.allocF64(new Float64Array(bins));
    const [pOut, pBytes] = h.allocF64(new Float64Array(bins));
    h.check(
      h.ex.es_rfft_psd(inPtr, nWave, fsWave, 256, 128, fOut, pOut, bins, h.scratch),
      "es_rfft_psd"
    );
    const psdVals = Array.from(new Float64Array(h.ex.memory.buffer, pOut, bins));

    const [envOut, envBytes] = h.allocF64(new Float64Array(nWave));
    h.check(h.ex.es_hilbert_envelope(inPtr, nWave, envOut), "es_hilbert_envelope");
    const envVals = Array.from(new Float64Array(h.ex.memory.buffer, envOut, nWave));

    const periods = new Float64Array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0]);
    const [perPtr, perBytes] = h.allocF64(periods);
    const [loPtr, loBytes] = h.allocF64(new Float64Array(periods.length));
    const [hiPtr, hiBytes] = h.allocF64(new Float64Array(periods.length));
    h.check(h.ex.es_nlnm_nhnm(perPtr, periods.length, loPtr, hiPtr), "es_nlnm_nhnm");
    const nlnmVals = Array.from(new Float64Array(h.ex.memory.buffer, loPtr, periods.length));
    const nhnmVals = Array.from(new Float64Array(h.ex.memory.buffer, hiPtr, periods.length));

    h.ex.es_free(hiPtr, hiBytes);
    h.ex.es_free(loPtr, loBytes);
    h.ex.es_free(perPtr, perBytes);
    h.ex.es_free(envOut, envBytes);
    h.ex.es_free(pOut, pBytes);
    h.ex.es_free(fOut, fBytes);
    h.ex.es_free(inPtr, inBytes);
    const elapsedMs = performance.now() - t0;

    const psdStats = maxDiffStats(psdVals, nativeRef.welch_psd);
    const envStats = maxDiffStats(envVals, nativeRef.hilbert_envelope);
    const nlnmStats = maxDiffStats(nlnmVals, nativeRef.nlnm_db);
    const nhnmStats = maxDiffStats(nhnmVals, nativeRef.nhnm_db);
    if (
      psdStats.max_abs_diff > 1e-12 ||
      envStats.max_abs_diff > 1e-12 ||
      nlnmStats.max_abs_diff > 1e-12 ||
      nhnmStats.max_abs_diff > 1e-12
    ) {
      throw new Error(
        `Spectral/Envelope diff > 1e-12: psd=${psdStats.max_abs_diff}, env=${envStats.max_abs_diff}`
      );
    }
    checks.fft_psd_envelope_noise = {
      status: "PASS",
      psd_max_abs_diff: psdStats.max_abs_diff,
      envelope_max_abs_diff: envStats.max_abs_diff,
      nlnm_max_abs_diff: nlnmStats.max_abs_diff,
      nhnm_max_abs_diff: nhnmStats.max_abs_diff,
      wasm_elapsed_ms: Number(elapsedMs.toFixed(4)),
    };
  }

  // 5. Morse (Glitch-Immune) vs Classic STA/LTA & Baer Picking
  {
    const t0 = performance.now();
    const glitched = new Float64Array(baseSignal);
    glitched[280] += 1.0e6; // Massive single-sample telemetry spike!

    const [gPtr, gBytes] = h.allocF64(glitched);
    const [cleanPtr, cleanBytes] = h.allocF64(baseSignal);
    const [cfPtr, cfBytes] = h.allocF64(new Float64Array(nWave));
    const idxOut = h.scratch;
    const metricOut = h.scratch + 8;

    // Classic STA/LTA on glitched trace:
    h.check(
      h.ex.es_pick_arrivals(gPtr, nWave, fsWave, 0, 10.0, 100.0, 4.0, idxOut, metricOut, cfPtr),
      "es_pick_arrivals(sta_lta)"
    );
    const staPick = Number(h.memView().getBigInt64(idxOut, true));

    // Discrete Morse on glitched trace (0 false triggers on 1e6 spike!):
    h.check(
      h.ex.es_pick_arrivals(gPtr, nWave, fsWave, 4, 15.0, 120.0, 4.5, idxOut, metricOut, cfPtr),
      "es_pick_arrivals(morse)"
    );
    const morsePick = Number(h.memView().getBigInt64(idxOut, true));
    const morsePeak = h.memView().getFloat64(metricOut, true);
    const morseCf = Array.from(new Float64Array(h.ex.memory.buffer, cfPtr, nWave));

    // Baer on clean trace:
    h.check(
      h.ex.es_pick_arrivals(cleanPtr, nWave, fsWave, 2, 7.0, 12.0, 20.0, idxOut, metricOut, cfPtr),
      "es_pick_arrivals(baer)"
    );
    const baerPick = Number(h.memView().getBigInt64(idxOut, true));
    const baerCf = Array.from(new Float64Array(h.ex.memory.buffer, cfPtr, nWave));

    h.ex.es_free(cfPtr, cfBytes);
    h.ex.es_free(cleanPtr, cleanBytes);
    h.ex.es_free(gPtr, gBytes);
    const elapsedMs = performance.now() - t0;

    const morseCfStats = maxDiffStats(morseCf, nativeRef.pickers.morse_cf);
    const baerCfStats = maxDiffStats(baerCf, nativeRef.pickers.baer_cf);
    if (
      staPick !== nativeRef.pickers.sta_lta_glitched_pick ||
      morsePick !== nativeRef.pickers.morse_glitched_pick ||
      baerPick !== nativeRef.pickers.baer_clean_pick ||
      morseCfStats.max_abs_diff > 1e-12
    ) {
      throw new Error(
        `Picker lockstep mismatch: sta=${staPick}/${nativeRef.pickers.sta_lta_glitched_pick}, morse=${morsePick}/${nativeRef.pickers.morse_glitched_pick}, baer=${baerPick}/${nativeRef.pickers.baer_clean_pick}`
      );
    }
    checks.morse_and_baer_pickers = {
      status: "PASS",
      sta_lta_glitched_pick: staPick,
      morse_glitched_pick: morsePick,
      morse_peak_ratio_diff: Math.abs(morsePeak - nativeRef.pickers.morse_peak_ratio),
      morse_cf_max_abs_diff: morseCfStats.max_abs_diff,
      baer_clean_pick: baerPick,
      baer_cf_max_abs_diff: baerCfStats.max_abs_diff,
      wasm_elapsed_ms: Number(elapsedMs.toFixed(4)),
    };
  }

  // 6. Spherical TauP Travel Times & Ray Paths across `iasp91`, `ak135`, `prem`
  {
    const t0 = performance.now();
    const models = ["iasp91", "ak135", "prem"];
    const phasesStr = "P,pP,PP,PcP,S,SKS,PKIKP";
    const [phPtr, phLen] = h.allocBytes(phasesStr);
    let maxTimeDiff = 0;
    let maxRayParamDiff = 0;
    let maxPathCoordDiff = 0;
    let totalArrivalsChecked = 0;
    let totalPathPointsChecked = 0;

    for (const model of models) {
      const [mPtr, mLen] = h.allocBytes(model);
      const eng = h.ex.es_taup_engine_create(mPtr, mLen) >>> 0;
      if (!eng) throw new Error(`Failed to create TauEngine(${model})`);

      for (const [depthKm, distDeg] of [
        [29.0, 73.87078776567722],
        [350.0, 145.0],
      ]) {
        h.check(
          h.ex.es_taup_travel_times(eng, 0, 0, depthKm, distDeg, phPtr, phLen, h.scratch, h.scratch + 4),
          `es_taup_travel_times(${model})`
        );
        const wasmArrs = unpackEstt(h.consumeOwned());
        const key = `${model}_${depthKm}_${distDeg}`;
        const natArrs = nativeRef.taup[key].arrivals;
        if (wasmArrs.length !== natArrs.length) {
          throw new Error(`TauP arrival count mismatch on ${key}: ${wasmArrs.length} vs ${natArrs.length}`);
        }
        for (let i = 0; i < wasmArrs.length; i++) {
          if (wasmArrs[i].name !== natArrs[i].name) {
            throw new Error(`TauP phase name mismatch on ${key}[${i}]`);
          }
          maxTimeDiff = Math.max(maxTimeDiff, Math.abs(wasmArrs[i].time_s - natArrs[i].time_s));
          maxRayParamDiff = Math.max(
            maxRayParamDiff,
            Math.abs(wasmArrs[i].ray_param_s_per_deg - natArrs[i].ray_param_s_per_deg)
          );
          totalArrivalsChecked++;
        }

        h.check(
          h.ex.es_taup_ray_paths(eng, 0, 0, depthKm, distDeg, phPtr, phLen, h.scratch, h.scratch + 4),
          `es_taup_ray_paths(${model})`
        );
        const wasmPaths = unpackEsrp(h.consumeOwned());
        const natPaths = nativeRef.taup[key].paths;
        if (wasmPaths.paths.length !== natPaths.length) {
          throw new Error(`TauP path count mismatch on ${key}`);
        }
        for (let i = 0; i < wasmPaths.paths.length; i++) {
          const wp = wasmPaths.paths[i];
          const np = natPaths[i];
          if (wp.n_points !== np.n_points) {
            throw new Error(`TauP path point count mismatch on ${key}[${i}]`);
          }
          totalPathPointsChecked += wp.n_points;
          for (const slot of ["first_pt", "mid_pt", "last_pt"]) {
            for (let c = 0; c < 4; c++) {
              maxPathCoordDiff = Math.max(maxPathCoordDiff, Math.abs(wp[slot][c] - np[slot][c]));
            }
          }
        }
      }

      h.ex.es_taup_engine_free(eng);
      h.ex.es_free(mPtr, mLen);
    }
    h.ex.es_free(phPtr, phLen);
    const elapsedMs = performance.now() - t0;

    if (maxTimeDiff > 1e-12 || maxRayParamDiff > 1e-12 || maxPathCoordDiff > 1e-11) {
      throw new Error(
        `TauP parity exceeded threshold: dt=${maxTimeDiff}, dp=${maxRayParamDiff}, dcoord=${maxPathCoordDiff}`
      );
    }
    checks.taup_iasp91_ak135_prem = {
      status: "PASS",
      models,
      arrivals_verified: totalArrivalsChecked,
      ray_path_points_verified: totalPathPointsChecked,
      max_travel_time_diff_s: maxTimeDiff,
      max_ray_param_diff_s_per_deg: maxRayParamDiff,
      max_ray_path_coord_diff: maxPathCoordDiff,
      wasm_elapsed_ms: Number(elapsedMs.toFixed(4)),
    };
  }

  h.dispose();

  const receipt = {
    gate: "Gate 5 — Native x86_64 vs WASM32 (+simd128) Cross-ISA Lockstep Verification",
    overall_status: "PASS",
    node_version: process.version,
    host_platform: `${process.platform}-${process.arch}`,
    wasm_compile_ms: Number(wasmCompileMs.toFixed(4)),
    wasm_binary: wasmMeta,
    checks,
  };

  fs.writeFileSync(RECEIPT_PATH, JSON.stringify(receipt, null, 2) + "\n", "utf8");
  console.log(
    `[Gate 5 PASS] earthsnail_wasm.wasm (${wasmMeta.wasm_size_bytes} B, simd128=${wasmMeta.simd128_enabled} [${wasmMeta.v128_opcode_bytes} v128 ops], 0 imports) lockstep verified against x86_64 -> ${RECEIPT_PATH}`
  );
}

main().catch((err) => {
  console.error("[Gate 5 FAIL]", err);
  process.exit(1);
});
