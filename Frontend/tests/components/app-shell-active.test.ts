import { describe, it, expect } from "vitest";
import { isNavItemActive } from "@/components/dashboard/app-shell";

describe("isNavItemActive", () => {
  it("matches a plain path item", () => {
    expect(isNavItemActive("/dashboard", "/dashboard", null)).toBe(true);
  });

  it("does not match a different plain path", () => {
    expect(isNavItemActive("/dashboard", "/chat", null)).toBe(false);
  });

  it("matches /test?tab=analysis when tab=analysis", () => {
    expect(isNavItemActive("/test?tab=analysis", "/test", "analysis")).toBe(true);
  });

  it("does not match /test?tab=mock when tab=analysis", () => {
    expect(isNavItemActive("/test?tab=mock", "/test", "analysis")).toBe(false);
  });

  it("does not match a tab item against a different base path", () => {
    expect(isNavItemActive("/test?tab=analysis", "/chat", "analysis")).toBe(false);
  });
});