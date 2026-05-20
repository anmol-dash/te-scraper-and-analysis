#!/usr/bin/env node
/**
 * Merge platform fragments into latest.json for the Tauri 2 updater.
 * Env: RELEASE_VERSION (semver without v), RELEASE_NOTES, PUB_DATE (RFC3339),
 *      GITHUB_REPOSITORY (owner/repo), TAG_NAME (v1.2.3)
 * Args: fragment.json files (one per platform build)
 */
import fs from "node:fs";

const version = process.env.RELEASE_VERSION;
const notes = process.env.RELEASE_NOTES ?? "";
const pubDate = process.env.PUB_DATE;
const repo = process.env.GITHUB_REPOSITORY;
const tag = process.env.TAG_NAME;

if (!version || !pubDate || !repo || !tag) {
  console.error(
    "Missing env: RELEASE_VERSION, PUB_DATE, GITHUB_REPOSITORY, TAG_NAME",
  );
  process.exit(1);
}

const [owner, name] = repo.split("/");
if (!owner || !name) {
  console.error(`Invalid GITHUB_REPOSITORY: ${repo}`);
  process.exit(1);
}

const baseUrl = `https://github.com/${owner}/${name}/releases/download/${tag}`;

const platforms = {};
for (const f of process.argv.slice(2)) {
  const raw = JSON.parse(fs.readFileSync(f, "utf8"));
  for (const [key, val] of Object.entries(raw)) {
    if (platforms[key]) {
      console.error(`Duplicate platform ${key}`);
      process.exit(1);
    }
    platforms[key] = {
      signature: val.signature,
      url: `${baseUrl}/${val.basename}`,
    };
  }
}

const out = {
  version,
  notes,
  pub_date: pubDate,
  platforms,
};

process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
