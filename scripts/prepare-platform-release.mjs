#!/usr/bin/env node
/**
 * Picks the user-facing installer for a Tauri platform key, copies it into
 * <out>/assets/ and writes <out>/fragment.json for merge-latest-json.mjs.
 *
 * Installers shipped: macOS -> .dmg, Windows -> NSIS setup .exe, Linux -> .AppImage.
 * The Tauri updater is not wired into this app, so detached .sig signatures are
 * optional: if a matching .sig exists it is shipped alongside, otherwise it is
 * silently skipped (and the fragment signature is left empty).
 */
import fs from "node:fs";
import path from "node:path";

const platformKey = process.argv[2];
const bundleRoot = process.argv[3];
const outRoot = process.argv[4];

if (!platformKey || !bundleRoot || !outRoot) {
  console.error(
    "Usage: node prepare-platform-release.mjs <platform-key> <bundle-dir> <out-dir>",
  );
  process.exit(1);
}

function walk(dir) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

if (!fs.existsSync(bundleRoot)) {
  console.error(`Bundle root not found: ${bundleRoot}`);
  process.exit(1);
}

const files = walk(bundleRoot);

// Pick exactly one installer matching `match`, plus its optional sidecar .sig.
function pickOne(match, label) {
  const hits = files.filter((f) => !f.endsWith(".sig") && match(f));
  if (hits.length !== 1) {
    console.error(
      `Expected exactly one ${label} under ${bundleRoot}, found ${hits.length}` +
        (hits.length ? `:\n  ${hits.join("\n  ")}` : ""),
    );
    process.exit(1);
  }
  const base = hits[0];
  const sigFile = `${base}.sig`;
  return {
    bundle: base,
    sig: fs.existsSync(sigFile) ? sigFile : null,
    basename: path.basename(base),
  };
}

function pickDarwin() {
  return pickOne((f) => f.endsWith(".dmg"), ".dmg");
}

function pickLinux() {
  return pickOne((f) => f.endsWith(".AppImage"), ".AppImage");
}

function pickWindows() {
  const nsis = (f) =>
    f.includes(`${path.sep}nsis${path.sep}`) && f.endsWith(".exe");
  if (files.some(nsis)) return pickOne(nsis, "NSIS setup .exe");
  return pickOne(
    (f) => f.endsWith(".exe") && (f.includes("setup") || f.includes("-setup")),
    "NSIS setup .exe",
  );
}

let picked;
if (platformKey.startsWith("darwin-")) picked = pickDarwin();
else if (platformKey === "linux-x86_64") picked = pickLinux();
else if (platformKey === "windows-x86_64") picked = pickWindows();
else {
  console.error(`Unknown platform key: ${platformKey}`);
  process.exit(1);
}

const assetsDir = path.join(outRoot, "assets");
fs.mkdirSync(assetsDir, { recursive: true });
fs.copyFileSync(picked.bundle, path.join(assetsDir, picked.basename));
if (picked.sig) {
  fs.copyFileSync(picked.sig, path.join(assetsDir, `${picked.basename}.sig`));
}

const fragment = {
  [platformKey]: {
    signature: picked.sig ? fs.readFileSync(picked.sig, "utf8").trim() : "",
    basename: picked.basename,
  },
};

fs.writeFileSync(
  path.join(outRoot, "fragment.json"),
  `${JSON.stringify(fragment, null, 2)}\n`,
);
