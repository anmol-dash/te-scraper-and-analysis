import type { Field } from "./cliSchema";

export type FormValueMap = Record<
  string,
  string | number | boolean | undefined
>;

export function formValuesToArgv(
  fields: Field[],
  values: FormValueMap,
): string[] {
  const argv: string[] = [];
  for (const field of fields) {
    const raw = values[field.flag];
    switch (field.kind) {
      case "file": {
        const path = String(raw ?? "").trim();
        if (path) {
          argv.push(field.flag, path);
        }
        break;
      }
      case "text": {
        const text = String(raw ?? "").trim();
        if (text !== "") {
          argv.push(field.flag, text);
        }
        break;
      }
      case "number": {
        if (raw === undefined || raw === "") {
          break;
        }
        const n = typeof raw === "number" ? raw : Number(raw);
        if (!Number.isFinite(n)) {
          break;
        }
        argv.push(field.flag, String(n));
        break;
      }
      case "bool": {
        const b =
          typeof raw === "boolean"
            ? raw
            : raw === "true"
              ? true
              : raw === "false"
                ? false
                : Boolean(raw);
        if (b) {
          argv.push(field.flag);
        }
        break;
      }
      case "select": {
        const opt = String(raw ?? "").trim();
        if (opt !== "") {
          argv.push(field.flag, opt);
        }
        break;
      }
    }
  }
  return argv;
}
