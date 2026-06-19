import { describe, it, expect } from "vitest";
import { getErrorMessage } from "@/lib/errors";
import { AxiosError } from "axios";

describe("getErrorMessage", () => {
  it("returns the backend detail string from an axios error", () => {
    const error = new AxiosError("bad", "ERR", undefined, undefined, {
      status: 400,
      data: { detail: "Email already registered" },
    } as any);
    expect(getErrorMessage(error)).toBe("Email already registered");
  });

  it("joins validation detail arrays", () => {
    const error = new AxiosError("bad", "ERR", undefined, undefined, {
      status: 422,
      data: { detail: [{ message: "invalid email" }, { message: "short password" }] },
    } as any);
    expect(getErrorMessage(error)).toBe("invalid email, short password");
  });

  it("falls back to the error message for a plain Error", () => {
    expect(getErrorMessage(new Error("boom"))).toBe("boom");
  });

  it("returns a generic message for unknown shapes", () => {
    expect(getErrorMessage("oops")).toBe("Something went wrong. Please try again.");
  });
});