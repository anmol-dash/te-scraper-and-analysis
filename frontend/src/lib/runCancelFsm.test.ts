import { describe, expect, it } from "vitest";

import {
  cancelButtonDisabled,
  cancelButtonLabel,
  reduceRunCancel,
} from "./runCancelFsm";

describe("runCancelFsm", () => {
  it("enables cancel only while running", () => {
    expect(cancelButtonDisabled("idle")).toBe(true);
    expect(cancelButtonDisabled("running")).toBe(false);
    expect(cancelButtonDisabled("cancelling")).toBe(true);
    expect(cancelButtonDisabled("stopped")).toBe(true);
  });

  it("walks start → cancel → stopped", () => {
    let p = reduceRunCancel("idle", { type: "START" });
    expect(p).toBe("running");
    p = reduceRunCancel(p, { type: "CANCEL_CLICK" });
    expect(p).toBe("cancelling");
    expect(cancelButtonLabel(p)).toBe("Cancelling…");
    p = reduceRunCancel(p, { type: "CANCEL_DONE" });
    expect(p).toBe("stopped");
    p = reduceRunCancel(p, { type: "RESET" });
    expect(p).toBe("idle");
  });
});
