export type RunCancelPhase = "idle" | "running" | "cancelling" | "stopped";

export type RunCancelEvent =
  | { type: "START" }
  | { type: "FINISH" }
  | { type: "CANCEL_CLICK" }
  | { type: "CANCEL_DONE" }
  | { type: "RESET" };

/**
 * Minimal cancel button state machine for a single fire-and-cancel run.
 *
 * - idle: Run not active; Cancel disabled
 * - running: Run active; Cancel enabled
 * - cancelling: Cancel in flight; Cancel disabled (busy)
 * - stopped: Run finished or aborted; Cancel disabled until RESET
 */
export function reduceRunCancel(
  phase: RunCancelPhase,
  event: RunCancelEvent,
): RunCancelPhase {
  switch (phase) {
    case "idle":
      if (event.type === "START") {
        return "running";
      }
      return phase;
    case "running":
      if (event.type === "CANCEL_CLICK") {
        return "cancelling";
      }
      if (event.type === "FINISH") {
        return "stopped";
      }
      if (event.type === "RESET") {
        return "idle";
      }
      return phase;
    case "cancelling":
      if (event.type === "CANCEL_DONE") {
        return "stopped";
      }
      if (event.type === "FINISH") {
        return "stopped";
      }
      if (event.type === "RESET") {
        return "idle";
      }
      return phase;
    case "stopped":
      if (event.type === "RESET") {
        return "idle";
      }
      return phase;
    default:
      return phase;
  }
}

export function cancelButtonDisabled(phase: RunCancelPhase): boolean {
  return phase !== "running";
}

export function cancelButtonLabel(phase: RunCancelPhase): string {
  if (phase === "cancelling") {
    return "Cancelling…";
  }
  return "Cancel";
}
