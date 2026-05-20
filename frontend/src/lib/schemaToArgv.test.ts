import { describe, expect, it } from "vitest";

import { schemaToArgv } from "./schemaToArgv";

describe("schemaToArgv", () => {
  it("maps booleans and scalars", () => {
    expect(
      schemaToArgv({
        verbose: true,
        q: false,
        profile: "fast",
        n: 3,
      }),
    ).toEqual(["--verbose", "--profile", "fast", "-n", "3"]);
  });

  it("supports short flags as single hyphen", () => {
    expect(schemaToArgv({ j: true, o: "out.txt" })).toEqual([
      "-j",
      "-o",
      "out.txt",
    ]);
  });
});
