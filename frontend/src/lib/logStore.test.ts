import { describe, expect, it } from "vitest";

import { createLogStore } from "./logStore";

describe("logStore cap", () => {
  it("drops oldest entries when exceeding max", () => {
    const store = createLogStore(3);
    store.getState().append("a");
    store.getState().append("b");
    store.getState().append("c");
    store.getState().append("d");
    expect(store.getState().lines).toEqual(["b", "c", "d"]);
  });

  it("shrinks when max decreases", () => {
    const store = createLogStore(10);
    for (let i = 0; i < 5; i += 1) {
      store.getState().append(`L${i}`);
    }
    store.getState().setMax(2);
    expect(store.getState().lines).toEqual(["L3", "L4"]);
    expect(store.getState().max).toBe(2);
  });
});
