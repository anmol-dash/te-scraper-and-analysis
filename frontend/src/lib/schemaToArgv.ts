/**
 * Convert a shallow "schema-like" record into CLI argv flags (demo / IPC payloads).
 *
 * - Boolean `true` → `--key`
 * - Boolean `false` → omitted
 * - String / number → `--key`, `String(value)`
 */
export function schemaToArgv(
  schema: Record<string, string | number | boolean | undefined | null>,
): string[] {
  const out: string[] = [];
  for (const [key, raw] of Object.entries(schema)) {
    if (raw === undefined || raw === null || raw === false) {
      continue;
    }
    const flag = key.length === 1 ? `-${key}` : `--${key}`;
    if (raw === true) {
      out.push(flag);
      continue;
    }
    out.push(flag, String(raw));
  }
  return out;
}
