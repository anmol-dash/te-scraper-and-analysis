#!/usr/bin/env python3
"""Sync semver from src-tauri/tauri.conf.json into frontend/package.json and backend/pyproject.toml."""

from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "src-tauri" / "tauri.conf.json"
    version = json.loads(cfg_path.read_text(encoding="utf-8"))["version"]

    pkg_path = repo_root / "frontend" / "package.json"
    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    pkg["version"] = version
    pkg_path.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")

    py_path = repo_root / "backend" / "pyproject.toml"
    text = py_path.read_text(encoding="utf-8")
    patched, replaced = replace_project_version(text, version)
    if replaced != 1:
        raise RuntimeError("Expected exactly one [project].version entry in backend/pyproject.toml")
    py_path.write_text(patched, encoding="utf-8")

    init_path = repo_root / "backend" / "yourtool" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    init_patched = re.sub(r'__version__\s*=\s*"[^"]*"', f'__version__ = "{version}"', init_text)
    init_path.write_text(init_patched, encoding="utf-8")


def replace_project_version(toml_text: str, version: str) -> tuple[str, int]:
    header_re = re.compile(r'^\[(?P<section>[^\]]+)\]\s*$')

    lines = toml_text.splitlines(keepends=True)
    header_for = ""

    replacements = 0
    rebuilt: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = header_re.match(stripped)
        if m:
            header_for = m.group("section").strip()

        newline = ""
        bare = line
        if bare.endswith("\r\n"):
            newline = "\r\n"
            bare = bare[:-2]
        elif bare.endswith("\n"):
            newline = "\n"
            bare = bare[:-1]

        if header_for == "project" and re.match(r"^\s*version\s*=\s*\"", bare):
            indent = bare[: len(bare) - len(bare.lstrip())]
            rebuilt.append(f'{indent}version = "{version}"{newline}')
            replacements += 1
        else:
            rebuilt.append(line)

    return "".join(rebuilt), replacements


if __name__ == "__main__":
    main()
