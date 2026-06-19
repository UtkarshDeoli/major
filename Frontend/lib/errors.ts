import { isAxiosError } from "axios";

/**
 * Extract a human-readable message from any thrown value. Handles axios
 * errors (FastAPI `detail` string or validation array), plain Errors, and
 * unknown shapes.
 */
export function getErrorMessage(error: unknown): string {
  if (isAxiosError(error)) {
    const detail = (error.response?.data as any)?.detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail)) {
      const joined = detail
        .map((d: any) => (typeof d?.message === "string" ? d.message : undefined))
        .filter(Boolean)
        .join(", ");
      if (joined) return joined;
    }
    if (error.message) return error.message;
  }
  if (error instanceof Error) return error.message;
  return "Something went wrong. Please try again.";
}