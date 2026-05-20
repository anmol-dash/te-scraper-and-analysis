#!/usr/bin/env python3
"""Generate placeholder icons under src-tauri/icons for Tauri bundles."""
from __future__ import annotations

import platform
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "src-tauri" / "icons"


def png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(path: Path, size: int, rgba: tuple[int, int, int, int]) -> None:
    w = h = size
    r, g, b, a = rgba
    raw = bytearray()
    row = bytes([0, r, g, b, a] * w)
    for _ in range(h):
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", compressed)
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def write_ico_with_png(path: Path, rgba: tuple[int, int, int, int]) -> None:
    w = h = 16
    raw = bytearray()
    row = bytes([0, *rgba] * w)
    for _ in range(h):
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), 9)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png_payload = (
        png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", compressed) + png_chunk(b"IEND", b"")
    )
    header = struct.pack("<HHH", 0, 1, 22)
    entry = struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(png_payload), 22 + 16)
    path.write_bytes(header + entry + png_payload)


def try_mac_icns(base_png: Path, dest: Path) -> bool:
    if platform.system() != "Darwin":
        return False
    iconset = OUT / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    sizes = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
    ]
    for name, dim in sizes:
        target = iconset / name
        subprocess.run(
            ["sips", "-z", str(dim), str(dim), str(base_png), "--out", str(target)],
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(dest)],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(iconset, ignore_errors=True)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    color = (0x3B, 0x82, 0xF6, 0xFF)
    p32 = OUT / "32x32.png"
    p128 = OUT / "128x128.png"
    p256 = OUT / "128x128@2x.png"
    write_png(p32, 32, color)
    write_png(p128, 128, color)
    write_png(p256, 256, color)
    write_ico_with_png(OUT / "icon.ico", color)

    icns = OUT / "icon.icns"
    if not try_mac_icns(p128, icns):
        # Non-mac builders: reuse a PNG; `tauri build` on macOS still prefers a real .icns.
        shutil.copy(p128, icns)


if __name__ == "__main__":
    main()
