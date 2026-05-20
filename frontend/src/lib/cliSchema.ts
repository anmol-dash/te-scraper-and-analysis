export type Field =
  | { kind: "text"; flag: string; label: string; default?: string }
  | {
      kind: "number";
      flag: string;
      label: string;
      default?: number;
      min?: number;
      max?: number;
    }
  | { kind: "bool"; flag: string; label: string; default?: boolean }
  | {
      kind: "select";
      flag: string;
      label: string;
      options: string[];
      default?: string;
    }
  | {
      kind: "file";
      flag: string;
      label: string;
      filters?: { name: string; ext: string[] }[];
    };

/**
 * Replace fields in this array to match your CLI.
 * These three fields demonstrate file, text, and select controls.
 */
export const EXAMPLE_CLI_SCHEMA: Field[] = [
  {
    kind: "file",
    flag: "--input",
    label: "Input file",
    filters: [
      { name: "Sequences", ext: ["fa", "fasta", "fna", "txt"] },
      { name: "CSV tables", ext: ["csv", "tsv"] },
    ],
  },
  {
    kind: "text",
    flag: "--label",
    label: "Run label",
    default: "demo-run",
  },
  {
    kind: "select",
    flag: "--profile",
    label: "Profile",
    options: ["fast", "standard", "careful"],
    default: "standard",
  },
];
