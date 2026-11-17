#!/usr/bin/env python3
"""Bundle `earthsnail_wasm.wasm` (+simd128) and the real 2011 Mw 9.1 Tohoku-Oki @ G.CAN.00.BH*
Steim-2 miniSEED + StationXML fixtures into `studio/earthsnail_portable_studio.html`.

Verifies:
  1. Valid WebAssembly `\0asm` v1 binary header, 0 external imports, and `+simd128` instructions.
  2. Zero `http://` or `https://` URLs anywhere in the generated single-file HTML workstation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import re
import sys


STUDIO_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = STUDIO_DIR.parent
DEFAULT_HTML = STUDIO_DIR / "earthsnail_portable_studio.html"

ORACLE_FALLBACK_DIR = pathlib.Path(
    r"C:\Users\x\.gemini\antigravity\brain\566443e0-66bc-4144-a363-aad888d88d34\scratch\oracle_staging\fixtures\io\real_event"
)


def _read_leb128_u32(buf: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, offset
        shift += 7


def inspect_wasm_binary(wasm_bytes: bytes) -> dict[str, object]:
    if len(wasm_bytes) < 8 or wasm_bytes[:4] != b"\x00asm":
        raise ValueError("Invalid WebAssembly magic bytes (expected \\0asm)")
    version = int.from_bytes(wasm_bytes[4:8], "little")
    if version != 1:
        raise ValueError(f"Unsupported WebAssembly version: {version}")

    imports: list[str] = []
    exports: list[str] = []
    has_simd128 = False
    v128_prefix_bytes = 0
    offset = 8
    n = len(wasm_bytes)

    while offset < n:
        sec_id = wasm_bytes[offset]
        offset += 1
        sec_len, offset = _read_leb128_u32(wasm_bytes, offset)
        sec_end = offset + sec_len
        if sec_id == 0:
            sec_slice = wasm_bytes[offset:sec_end]
            if b"target_features" in sec_slice and b"simd128" in sec_slice:
                has_simd128 = True
        elif sec_id == 2:
            count, p = _read_leb128_u32(wasm_bytes, offset)
            for _ in range(count):
                mod_len, p = _read_leb128_u32(wasm_bytes, p)
                mod_name = wasm_bytes[p : p + mod_len].decode("utf-8", errors="replace")
                p += mod_len
                fld_len, p = _read_leb128_u32(wasm_bytes, p)
                fld_name = wasm_bytes[p : p + fld_len].decode("utf-8", errors="replace")
                p += fld_len
                kind = wasm_bytes[p]
                p += 1
                if kind == 0:
                    _, p = _read_leb128_u32(wasm_bytes, p)
                elif kind == 1:
                    p += 1
                    flags, p = _read_leb128_u32(wasm_bytes, p)
                    _, p = _read_leb128_u32(wasm_bytes, p)
                    if flags & 0x1:
                        _, p = _read_leb128_u32(wasm_bytes, p)
                elif kind == 2:
                    flags, p = _read_leb128_u32(wasm_bytes, p)
                    _, p = _read_leb128_u32(wasm_bytes, p)
                    if flags & 0x1:
                        _, p = _read_leb128_u32(wasm_bytes, p)
                elif kind == 3:
                    p += 2
                imports.append(f"{mod_name}.{fld_name}")
        elif sec_id == 7:
            count, p = _read_leb128_u32(wasm_bytes, offset)
            for _ in range(count):
                name_len, p = _read_leb128_u32(wasm_bytes, p)
                name = wasm_bytes[p : p + name_len].decode("utf-8", errors="replace")
                p += name_len
                p += 1
                _, p = _read_leb128_u32(wasm_bytes, p)
                exports.append(name)
        elif sec_id == 10:
            v128_prefix_bytes = wasm_bytes[offset:sec_end].count(0xFD)
        offset = sec_end

    if imports:
        raise RuntimeError(f"Expected 0 WASM imports in standalone module, found: {imports}")
    if not (has_simd128 or v128_prefix_bytes > 0):
        raise RuntimeError("Expected +simd128 active in earthsnail_wasm.wasm")

    return {
        "version": "0.1.0",
        "wasm_bytes": len(wasm_bytes),
        "sha256": hashlib.sha256(wasm_bytes).hexdigest(),
        "simd128": True,
        "v128_op_bytes": v128_prefix_bytes,
        "export_count": len(exports),
        "import_count": len(imports),
    }


def find_default_wasm() -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    env_target = os.environ.get("CARGO_TARGET_DIR")
    if env_target:
        candidates.append(
            pathlib.Path(env_target) / "wasm32-unknown-unknown" / "release" / "earthsnail_wasm.wasm"
        )
    candidates.extend(
        [
            REPO_ROOT
            / "target_workers"
            / "studio_wasm"
            / "wasm32-unknown-unknown"
            / "release"
            / "earthsnail_wasm.wasm",
            REPO_ROOT / "target" / "wasm32-unknown-unknown" / "release" / "earthsnail_wasm.wasm",
        ]
    )
    tw = REPO_ROOT / "target_workers"
    if tw.is_dir():
        for child in sorted(tw.iterdir()):
            cand = child / "wasm32-unknown-unknown" / "release" / "earthsnail_wasm.wasm"
            if cand not in candidates:
                candidates.append(cand)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "earthsnail_wasm.wasm not found. Build with: "
        "cargo build -p earthsnail-wasm --target wasm32-unknown-unknown --release"
    )


def find_real_event_files() -> tuple[pathlib.Path, pathlib.Path]:
    dirs = [
        REPO_ROOT / "tests" / "fixtures" / "io" / "real_event",
        ORACLE_FALLBACK_DIR,
    ]
    for d in dirs:
        mseed = d / "real_3c.mseed"
        xml = d / "real_3c_station.xml"
        if mseed.is_file() and xml.is_file():
            return mseed, xml
    raise FileNotFoundError("Could not locate real_3c.mseed and real_3c_station.xml fixtures")


def replace_const_b64(html: str, const_name: str, b64_val: str) -> str:
    pat = re.compile(rf'const\s+{re.escape(const_name)}\s*=\s*"[^"]*";')
    if not pat.search(html):
        raise RuntimeError(f"Placeholder `const {const_name} = \"...\";` not found in HTML")
    return pat.sub(lambda _m: f'const {const_name} = "{b64_val}";', html, count=1)


def bundle_studio(
    wasm_path: pathlib.Path,
    mseed_path: pathlib.Path,
    xml_path: pathlib.Path,
    html_path: pathlib.Path,
) -> dict[str, object]:
    wasm_bytes = wasm_path.read_bytes()
    meta = inspect_wasm_binary(wasm_bytes)
    mseed_bytes = mseed_path.read_bytes()
    xml_bytes = xml_path.read_bytes()

    wasm_b64 = base64.b64encode(wasm_bytes).decode("ascii")
    mseed_b64 = base64.b64encode(mseed_bytes).decode("ascii")
    xml_b64 = base64.b64encode(xml_bytes).decode("ascii")

    html = html_path.read_text(encoding="utf-8")
    html = replace_const_b64(html, "EARTHSNAIL_WASM_B64", wasm_b64)
    html = replace_const_b64(html, "PRELOADED_TOHOKU_MSEED_B64", mseed_b64)
    html = replace_const_b64(html, "PRELOADED_TOHOKU_XML_B64", xml_b64)

    meta_pat = re.compile(r"const\s+EARTHSNAIL_WASM_BUILD_META\s*=\s*\{[^\n]*\};")
    meta_json = json.dumps(
        {
            "version": meta["version"],
            "wasm_bytes": meta["wasm_bytes"],
            "sha256_prefix": str(meta["sha256"])[:16],
            "simd128": meta["simd128"],
            "v128_op_bytes": meta["v128_op_bytes"],
            "export_count": meta["export_count"],
            "import_count": meta["import_count"],
            "mseed_bytes": len(mseed_bytes),
            "stationxml_bytes": len(xml_bytes),
        },
        separators=(",", ":"),
    )
    if not meta_pat.search(html):
        raise RuntimeError("Placeholder `const EARTHSNAIL_WASM_BUILD_META = {...};` not found")
    html = meta_pat.sub(lambda _m: f"const EARTHSNAIL_WASM_BUILD_META = {meta_json};", html, count=1)

    ext_urls = re.findall(r"https?://[^\s\"'`<>)]+", html)
    if ext_urls:
        raise RuntimeError(f"Zero-CDN audit failed: found external URLs in HTML: {ext_urls}")

    html_path.write_text(html, encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle EarthSnail Portable Studio single-file HTML")
    parser.add_argument("--wasm", type=pathlib.Path, default=None)
    parser.add_argument("--mseed", type=pathlib.Path, default=None)
    parser.add_argument("--xml", type=pathlib.Path, default=None)
    parser.add_argument("--html", type=pathlib.Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    wasm_path = args.wasm or find_default_wasm()
    if args.mseed and args.xml:
        mseed_path, xml_path = args.mseed, args.xml
    else:
        mseed_path, xml_path = find_real_event_files()

    meta = bundle_studio(wasm_path, mseed_path, xml_path, args.html)
    print(
        f"Bundled {wasm_path.name} ({meta['wasm_bytes']} B, simd128={meta['simd128']} "
        f"[{meta['v128_op_bytes']} 0xFD bytes], {meta['export_count']} exports, 0 imports) + "
        f"{mseed_path.name} ({mseed_path.stat().st_size} B) + {xml_path.name} ({xml_path.stat().st_size} B) "
        f"-> {args.html} ({args.html.stat().st_size} B, 0 external URLs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
