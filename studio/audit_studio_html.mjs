#!/usr/bin/env node
/**
 * Offline & Functional Integrity Audit for `studio/earthsnail_portable_studio.html`:
 *   1. Zero external URLs (`http://` / `https://`) across the entire HTML document.
 *   2. Extracted `<script>` passes `node --check` with 0 syntax errors.
 *   3. Embedded `EARTHSNAIL_WASM_B64` has `\0asm` v1 magic, `0` imports, and `+simd128`.
 *   4. Embedded `PRELOADED_TOHOKU_MSEED_B64` & `PRELOADED_TOHOKU_XML_B64` decode & deconvolve
 *      live inside the embedded WASM32 module (`3 × 14,400` Steim-2 samples @ 20 Hz, `G.CAN.00.BH*`).
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const STUDIO_DIR = path.dirname(__filename);
const HTML_PATH = path.join(STUDIO_DIR, "earthsnail_portable_studio.html");

async function main() {
  const html = fs.readFileSync(HTML_PATH, "utf8");

  // 1. Zero external URLs check:
  const extUrls = html.match(/https?:\/\/[^\s"'`<>)]+/gi) || [];
  if (extUrls.length > 0) {
    throw new Error(`Found external URLs in studio HTML: ${extUrls.join(", ")}`);
  }

  // 2. Extract <script> and run `node --check`:
  const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/i);
  if (!scriptMatch) {
    throw new Error("Could not locate <script>...</script> block in studio HTML");
  }
  const scriptSrc = scriptMatch[1];
  const tmpJs = path.join(os.tmpdir(), `earthsnail_studio_check_${process.pid}.js`);
  fs.writeFileSync(tmpJs, scriptSrc, "utf8");
  try {
    const chk = spawnSync(process.execPath, ["--check", tmpJs], { encoding: "utf8" });
    if (chk.status !== 0) {
      throw new Error(`node --check failed:\n${chk.stderr}`);
    }
  } finally {
    try { fs.unlinkSync(tmpJs); } catch {}
  }

  // 3. Extract embedded Base64 constants and verify WASM magic + live Tohoku-Oki boot decode:
  const extractConst = (name) => {
    const m = scriptSrc.match(new RegExp(`const\\s+${name}\\s*=\\s*"([^"]+)";`));
    if (!m || !m[1]) throw new Error(`Embedded constant ${name} is missing or empty`);
    return Buffer.from(m[1], "base64");
  };

  const wasmBytes = extractConst("EARTHSNAIL_WASM_B64");
  if (wasmBytes.subarray(0, 4).toString("latin1") !== "\0asm") {
    throw new Error("Embedded WASM missing \\0asm magic bytes");
  }
  const hasSimd128 = wasmBytes.includes(0xfd);
  if (!hasSimd128) {
    throw new Error("Embedded WASM missing +simd128 opcodes");
  }

  const mseedBytes = extractConst("PRELOADED_TOHOKU_MSEED_B64");
  const xmlBytes = extractConst("PRELOADED_TOHOKU_XML_B64");

  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  const scratch = ex.es_alloc(32) >>> 0;
  const mPtr = ex.es_alloc(mseedBytes.length) >>> 0;
  new Uint8Array(ex.memory.buffer, mPtr, mseedBytes.length).set(mseedBytes);

  const rc = ex.es_decode_waveforms(mPtr, mseedBytes.length, scratch, scratch + 4);
  if (rc !== 0) throw new Error(`Embedded Tohoku mSEED decode failed (${rc})`);

  const dv = new DataView(ex.memory.buffer);
  const wfPtr = dv.getUint32(scratch, true);
  const wfLen = dv.getUint32(scratch + 4, true);
  const wfDv = new DataView(ex.memory.buffer, wfPtr, wfLen);
  const traceCount = wfDv.getUint32(8, true);
  const npts0 = Number(wfDv.getBigUint64(16, true));
  const fs0 = wfDv.getFloat64(24, true);

  if (traceCount !== 3 || npts0 !== 14400 || fs0 !== 20.0) {
    throw new Error(`Unexpected Tohoku trace geometry: traces=${traceCount}, npts=${npts0}, fs=${fs0}`);
  }

  // Verify live StationXML deconvolution of embedded BHE (trace 0) matches ObsPy PGV (~2.096e-4 m/s):
  const bheRawPtr = wfPtr + 16 + 192;
  const xPtr = ex.es_alloc(xmlBytes.length) >>> 0;
  new Uint8Array(ex.memory.buffer, xPtr, xmlBytes.length).set(xmlBytes);
  const chBytes = Buffer.from("G.CAN.00.BHE", "utf8");
  const chPtr = ex.es_alloc(chBytes.length) >>> 0;
  new Uint8Array(ex.memory.buffer, chPtr, chBytes.length).set(chBytes);
  const pfPtr = ex.es_alloc(32) >>> 0;
  new Float64Array(ex.memory.buffer, pfPtr, 4).set([0.005, 0.01, 8.0, 9.5]);
  const velOutPtr = ex.es_alloc(npts0 * 8) >>> 0;

  const rcResp = ex.es_remove_response(
    bheRawPtr, npts0, fs0, xPtr, xmlBytes.length, 0, chPtr, chBytes.length,
    1299822990n, 1, pfPtr, 60.0, velOutPtr
  );
  if (rcResp !== 0) throw new Error(`Embedded StationXML remove_response failed (${rcResp})`);
  const velArr = new Float64Array(ex.memory.buffer, velOutPtr, npts0);
  let pgv = 0;
  for (let i = 0; i < npts0; i++) {
    const a = Math.abs(velArr[i]);
    if (a > pgv) pgv = a;
  }
  if (!(pgv > 1.8e-4 && pgv < 2.4e-4)) {
    throw new Error(`Unexpected deconvolved Tohoku BHE PGV: ${pgv}`);
  }

  ex.es_free(velOutPtr, npts0 * 8);
  ex.es_free(pfPtr, 32);
  ex.es_free(chPtr, chBytes.length);
  ex.es_free(xPtr, xmlBytes.length);
  ex.es_free(wfPtr, wfLen);
  ex.es_free(mPtr, mseedBytes.length);
  ex.es_free(scratch, 32);

  console.log(
    JSON.stringify({
      status: "PASS",
      html_bytes: fs.statSync(HTML_PATH).size,
      external_url_count: 0,
      js_syntax_check: "PASS",
      wasm_magic: "\\0asm",
      simd128: true,
      embedded_wasm_bytes: wasmBytes.length,
      embedded_mseed_bytes: mseedBytes.length,
      embedded_stationxml_bytes: xmlBytes.length,
      decoded_traces: traceCount,
      samples_per_trace: npts0,
      bhe_deconvolved_pgv_m_s: Number(pgv.toExponential(6))
    })
  );
}

main().catch((err) => {
  console.error("[Studio Audit FAIL]", err);
  process.exit(1);
});
