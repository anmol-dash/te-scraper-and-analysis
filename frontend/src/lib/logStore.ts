import { createStore } from "zustand";

const ringAppend = (lines: string[], next: string, cap: number): string[] => {
  const merged = lines.concat(next);
  if (merged.length <= cap) {
    return merged;
  }
  return merged.slice(merged.length - cap);
};

export type LogStore = {
  lines: string[];
  max: number;
  append: (line: string) => void;
  clear: () => void;
  setMax: (n: number) => void;
};

export const createLogStore = (initialMax = 500) =>
  createStore<LogStore>()((set, get) => ({
    lines: [],
    max: initialMax,
    append: (line) => {
      const { lines, max } = get();
      set({ lines: ringAppend(lines, line, max) });
    },
    clear: () => set({ lines: [] }),
    setMax: (n) =>
      set((s) => {
        const max = Math.max(1, n);
        const lines =
          s.lines.length > max ? s.lines.slice(s.lines.length - max) : s.lines;
        return { max, lines };
      }),
  }));
