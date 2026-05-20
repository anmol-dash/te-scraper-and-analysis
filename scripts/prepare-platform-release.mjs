#!/usr/bin/env node
/**
 * Picks updater bundle + .sig for a Tauri platform key, copies into <out>/assets/
 * and writes <out>/fragment.json for merge-latest-json.mjs
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
const readSig = (p) => fs.readFileSync(p, "utf8").trim();

function pickDarwin() {
  const tgz = files.filter(
    (f) => f.endsWith(".app.tar.gz") && !f.endsWith(".sig"),
  );
  if (tgz.length !== 1) {
    console.error(
      `Expected exactly one .app.tar.gz under ${bundleRoot}, found ${tgz.length}`,
    );
    process.exit(1);
  }
  const base = tgz[0];
  const sigFile = `${base}.sig`;
  if (!fs.existsSync(sigFile)) {
    console.error(`Missing signature: ${sigFile}`);
    process.exit(1);
  }
  return { bundle: base, sig: sigFile, basename: path.basename(base) };
}

function pickLinux() {
  const imgs = files.filter(
    (f) => f.endsWith(".AppImage") && !f.endsWith(".sig"),
  );
  if (imgs.length !== 1) {
    console.error(
      `Expected exactly one .AppImage under ${bundleRoot}, found ${imgs.length}`,
    );
    process.exit(1);
  }
  const base = imgs[0];
  const sigFile = `${base}.sig`;
  if (!fs.existsSync(sigFile)) {
    console.error(`Missing signature: ${sigFile}`);
    process.exit(1);
  }
  return { bundle: base, sig: sigFile, basename: path.basename(base) };
}

function pickWindows() {
  const inNsis = files.filter((f) => {
    if (!f.includes(`${path.sep}nsis${path.sep}`)) return false;
    return f.endsWith(".exe") && !f.endsWith(".sig");
  });
  const exes = inNsis.length
    ? inNsis
    : files.filter(
        (f) =>
          f.endsWith(".exe") &&
          !f.endsWith(".sig") &&
          (f.includes("setup") || f.includes("-setup")),
      );
  if (exes.length !== 1) {
    console.error(
      `Expected exactly one NSIS setup .exe under ${bundleRoot}, found ${exes.length}`,
    );
    process.exit(1);
  }
  const base = exes[0];
  const sigFile = `${base}.sig`;
  if (!fs.existsSync(sigFile)) {
    console.error(`Missing signature: ${sigFile}`);
    process.exit(1);
  }
  return { bundle: base, sig: sigFile, basename: path.basename(base) };
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
fs.copyFileSync(picked.sig, path.join(assetsDir, `${picked.basename}.sig`));

const fragment = {
  [platformKey]: {
    signature: readSig(picked.sig),
    basename: picked.basename,
  },
};

fs.writeFileSync(
  path.join(outRoot, "fragment.json"),
  `${JSON.stringify(fragment, null, 2)}\n`,
);
