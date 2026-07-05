# Orbit Frontend — Phase 0: Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the correctness bugs and broken/misleading features in the Orbit frontend that
actively harm users today (hydration mismatches, dead links, no-op Save, broken sign-out, no quiz
submit confirmation, volatile results, stub features, silent errors, and the missing `/api/*`
routes that break onboarding and exam creation).

**Architecture:** Mostly surgical edits to existing files plus two small new modules
(`lib/errors.ts`, `components/auth/auth-split-layout.tsx`) and three small static pages. One
backend fix (register the missing `onboarding_router`). TDD is applied to the two pure-logic
helpers; visual/CSS fixes are verified by build + browser. All `fetch("/api/...")` calls are
routed through the shared axios `api` instance so the Bearer token is attached.

**Tech Stack:** Next.js 16 (App Router) · React 19 · TypeScript · Tailwind · Radix/shadcn · axios ·
Vitest (added in Task 0.0 for the two unit-tested helpers).

**Branch:** `phase0/critical-fixes` (create in Task 0.0).

**Backend route facts (confirmed by reading `Backend/src/`):**
- `exam_router` prefix `/api/exams`: `GET /api/exams/`, `POST /api/exams/`, `PATCH /api/exams/{exam_id}/active`
- `subject_router` prefix `/api/exams`: `GET/POST /api/exams/{exam_id}/subjects`
- `onboarding_router` (file exists) — paths `/api/onboarding/` (GET/POST), `/api/onboarding/complete` (POST) — **but is NOT registered in `Backend/src/main.py`** (Task 0.12 fixes this).
- The axios `api` instance (`lib/api.ts`) `baseURL = http://localhost:8001`, so `api.get("/api/exams/")` → `http://localhost:8001/api/exams/`.

---

## Task 0.0: Branch, install Vitest, add the `getErrorMessage` helper

**Files:**
- Modify: `Frontend/package.json` (add `vitest`, `@testing-library/react`, `jsdom` devDeps)
- Create: `Frontend/vitest.config.ts`
- Create: `Frontend/lib/errors.ts`
- Create: `Frontend/tests/lib/errors.test.ts`
- Modify: `Frontend/.gitignore`

- [ ] **Step 1: Create the branch**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit
git checkout -b phase0/critical-fixes
```

- [ ] **Step 2: Write the failing test for `getErrorMessage`**

Create `Frontend/tests/lib/errors.test.ts`:

```ts
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
```

- [ ] **Step 3: Install Vitest + jsdom**

```bash
cd Frontend
npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 4: Add `vitest.config.ts`**

Create `Frontend/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(__dirname, "./") },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
  },
});
```

Create `Frontend/tests/setup.ts`:

```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 5: Add the `test` script to `package.json`**

In `Frontend/package.json`, add to `scripts`:

```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 6: Run the test to verify it fails**

```bash
cd Frontend
npx vitest run tests/lib/errors.test.ts
```
Expected: FAIL — `getErrorMessage` is not defined (module doesn't exist).

- [ ] **Step 7: Implement `lib/errors.ts`**

Create `Frontend/lib/errors.ts`:

```ts
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
```

- [ ] **Step 8: Run the test to verify it passes**

```bash
cd Frontend
npx vitest run tests/lib/errors.test.ts
```
Expected: PASS (4 tests).

- [ ] **Step 9: Update `.gitignore`**

Append to `Frontend/.gitignore` (create if missing):

```
node_modules
.next
out
coverage
tsconfig.tsbuildinfo
.DS_Store
```

- [ ] **Step 10: Commit**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit
git add Frontend/package.json Frontend/package-lock.json Frontend/vitest.config.ts \
        Frontend/tests Frontend/lib/errors.ts Frontend/.gitignore
git commit -m "chore(frontend): add vitest + getErrorMessage error helper"
```

---

## Task 0.1: Fix auth-page hydration mismatch + de-duplicate login/signup

**Files:**
- Create: `Frontend/components/auth/auth-split-layout.tsx`
- Modify: `Frontend/app/(auth)/login/page.tsx`
- Modify: `Frontend/app/(auth)/signup/page.tsx`

**Problem:** `login/page.tsx` and `signup/page.tsx` generate star positions with `Math.random()`
inside render → server/client differ → hydration warnings + flicker. The two files are also
near-identical copy-paste.

- [ ] **Step 1: Create the shared `AuthSplitLayout` (server component, deterministic starfield)**

Create `Frontend/components/auth/auth-split-layout.tsx`:

```tsx
import Link from "next/link";
import Image from "next/image";
import { AuthForm } from "@/components/auth/auth-form";

interface AuthSplitLayoutProps {
  type: "login" | "signup";
  formTitle: string;
  formSubtitle: string;
}

/**
 * Deterministic PRNG (mulberry32). Using a fixed seed per star means the
 * server-rendered markup and the client-hydrated markup are identical — no
 * Math.random() hydration mismatch.
 */
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const STAR_COUNT = 80;
const SHOOTING_COUNT = 5;

// Computed once with fixed seeds → identical on server and client.
const stars = Array.from({ length: STAR_COUNT }, (_, i) => {
  const rand = mulberry32(i + 1);
  return {
    width: rand() * 3 + 1,
    height: rand() * 3 + 1,
    top: rand() * 100,
    left: rand() * 100,
    opacity: rand() * 0.8 + 0.2,
    animationDelay: rand() * 3,
    animationDuration: rand() * 2 + 2,
  };
});

const shootingStars = Array.from({ length: SHOOTING_COUNT }, (_, i) => {
  const rand = mulberry32(1000 + i);
  return { top: rand() * 50, left: rand() * 60, animationDelay: rand() * 10 + i * 5 };
});

export function AuthSplitLayout({ type, formTitle, formSubtitle }: AuthSplitLayoutProps) {
  return (
    <div className="min-h-screen flex">
      {/* Left: form */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center p-8 lg:p-16 bg-[#0D1520] relative overflow-hidden">
        <div className="absolute inset-0" aria-hidden="true">
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: "50px 50px",
          }} />
          <div className="absolute inset-0" style={{
            background: "radial-gradient(ellipse at center, transparent 0%, #0D1520 100%)",
          }} />
        </div>

        <div className="relative z-10 w-full max-w-md mx-auto">
          <Link href="/" className="flex items-center gap-3 mb-12">
            <Image src="/logo.png" alt="Orbit Logo" width={48} height={48} className="shrink-0" />
            <span className="font-bold text-2xl text-white">Orbit</span>
          </Link>

          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-3">{formTitle}</h1>
            <p className="text-gray-400 text-lg">{formSubtitle}</p>
          </div>

          <AuthForm type={type} />
        </div>
      </div>

      {/* Right: starfield + brand */}
      <div className="hidden lg:flex w-1/2 relative overflow-hidden bg-gradient-to-br from-[#15202B] to-[#0D1520]">
        <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
          {stars.map((s, i) => (
            <div
              key={i}
              className="absolute bg-white rounded-full animate-pulse"
              style={{
                width: `${s.width}px`,
                height: `${s.height}px`,
                top: `${s.top}%`,
                left: `${s.left}%`,
                opacity: s.opacity,
                animationDelay: `${s.animationDelay}s`,
                animationDuration: `${s.animationDuration}s`,
              }}
            />
          ))}
          {shootingStars.map((s, i) => (
            <div
              key={`shooting-${i}`}
              className="absolute w-1 h-1 bg-gradient-to-r from-transparent via-white to-transparent rounded-full opacity-0"
              style={{
                top: `${s.top}%`,
                left: `${s.left}%`,
                animationDelay: `${s.animationDelay}s`,
                animationDuration: "2s",
                animationIterationCount: "infinite",
              }}
            />
          ))}
        </div>

        <div className="absolute inset-0 flex flex-col items-center justify-center p-16 z-10">
          <div className="text-center">
            <Image src="/logo.png" alt="Orbit Logo" width={128} height={128} className="mx-auto mb-8" />
            <h1 className="text-5xl font-bold text-white mb-6">Orbit</h1>
            <p className="text-xl text-gray-300 max-w-md mx-auto leading-relaxed">
              Unlock your real potential by practicing with Orbit&apos;s mock test generator
            </p>
          </div>
        </div>
      </div>

      {/* Mobile logo */}
      <div className="lg:hidden absolute top-0 left-0 right-0 p-4 z-20">
        <Link href="/" className="flex items-center gap-2">
          <Image src="/logo.png" alt="Orbit Logo" width={40} height={40} />
          <span className="font-bold text-xl text-white">Orbit</span>
        </Link>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Replace `login/page.tsx`**

Replace the entire contents of `Frontend/app/(auth)/login/page.tsx` with:

```tsx
import { AuthSplitLayout } from "@/components/auth/auth-split-layout";

export default function LoginPage() {
  return (
    <AuthSplitLayout
      type="login"
      formTitle="Welcome back"
      formSubtitle="Enter your credentials to access your account"
    />
  );
}
```

- [ ] **Step 3: Replace `signup/page.tsx`**

Replace the entire contents of `Frontend/app/(auth)/signup/page.tsx` with:

```tsx
import { AuthSplitLayout } from "@/components/auth/auth-split-layout";

export default function SignupPage() {
  return (
    <AuthSplitLayout
      type="signup"
      formTitle="Create your account"
      formSubtitle="Start your journey to exam success"
    />
  );
}
```

- [ ] **Step 4: Verify build + no hydration warning**

```bash
cd Frontend
npm run build
```
Expected: build succeeds. Then run `npm run dev`, open `/login` and `/signup`, and confirm the
browser console shows **no** hydration warnings (previously: "did not match").

- [ ] **Step 5: Commit**

```bash
git add Frontend/components/auth/auth-split-layout.tsx "Frontend/app/(auth)/login/page.tsx" "Frontend/app/(auth)/signup/page.tsx"
git commit -m "fix(auth): eliminate Math.random hydration mismatch, de-dup login/signup"
```

---

## Task 0.2: Remove/implement dead auth links

**Files:**
- Create: `Frontend/app/(auth)/forgot-password/page.tsx`
- Create: `Frontend/app/(auth)/terms/page.tsx`
- Create: `Frontend/app/(auth)/privacy/page.tsx`

**Problem:** `auth-form.tsx` links to `/forgot-password`, `/terms`, `/privacy` — none exist → 404.

- [ ] **Step 1: Create the forgot-password page (coming-soon state)**

Create `Frontend/app/(auth)/forgot-password/page.tsx`:

```tsx
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function ForgotPasswordPage() {
  return (
    <div className="min-h-screen flex items-center justify-center p-8 bg-background">
      <div className="max-w-md w-full text-center space-y-4">
        <h1 className="text-2xl font-semibold tracking-tight">Forgot your password?</h1>
        <p className="text-sm text-muted-foreground">
          Password reset is coming soon. For now, please contact support to reset your account.
        </p>
        <Button asChild>
          <Link href="/login">Back to login</Link>
        </Button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create the Terms page**

Create `Frontend/app/(auth)/terms/page.tsx`:

```tsx
export const metadata = { title: "Terms of Service — Orbit" };

export default function TermsPage() {
  return (
    <div className="min-h-screen p-8 lg:p-16 bg-background">
      <article className="max-w-2xl mx-auto prose prose-sm dark:prose-invert">
        <h1>Terms of Service</h1>
        <p>
          By using Orbit, you agree to use the platform for lawful study purposes only. You are
          responsible for the content you upload and for maintaining the security of your account.
        </p>
        <p>
          Orbit provides AI-generated study aids that may be inaccurate. You are responsible for
          verifying any guidance before relying on it for examinations.
        </p>
        <p>
          We may update these terms periodically. Continued use after changes constitutes
          acceptance of the revised terms.
        </p>
      </article>
    </div>
  );
}
```

- [ ] **Step 3: Create the Privacy page**

Create `Frontend/app/(auth)/privacy/page.tsx`:

```tsx
export const metadata = { title: "Privacy Policy — Orbit" };

export default function PrivacyPage() {
  return (
    <div className="min-h-screen p-8 lg:p-16 bg-background">
      <article className="max-w-2xl mx-auto prose prose-sm dark:prose-invert">
        <h1>Privacy Policy</h1>
        <p>
          Orbit stores the documents you upload and the tests you generate to provide you with
          study aids. We do not sell your data.
        </p>
        <p>
          You may request deletion of your account and associated data at any time from Settings.
        </p>
        <p>
          Authentication tokens are stored locally in your browser. We use them only to keep you
          signed in.
        </p>
      </article>
    </div>
  );
}
```

- [ ] **Step 4: Verify the links resolve**

```bash
cd Frontend
npm run dev
```
Open `/forgot-password`, `/terms`, `/privacy` — each renders, none 404. Click the links from
`/login` and `/signup` to confirm navigation.

- [ ] **Step 5: Commit**

```bash
git add "Frontend/app/(auth)/forgot-password" "Frontend/app/(auth)/terms" "Frontend/app/(auth)/privacy"
git commit -m "feat(auth): add forgot-password, terms, and privacy pages"
```

---

## Task 0.3: Stop the misleading Settings "Save Changes"

**Files:**
- Modify: `Frontend/app/(dashboard)/settings/page.tsx`

**Problem:** `handleSave` only toasts success — nothing persists, so profile/password/notification
toggles silently revert on reload. The Save button also wrongly shows a `Lock` icon.

- [ ] **Step 1: Replace `handleSave` to persist client-side prefs honestly + fix the icon**

In `Frontend/app/(dashboard)/settings/page.tsx`:

Replace the `handleSave` function (lines 60–65):

```tsx
  const handleSave = () => {
    // Phase 0 interim: persist the preferences we can store client-side.
    // Full profile/account sync (name, email, password) arrives in Phase 3.
    localStorage.setItem(
      "orbit:preferences",
      JSON.stringify({
        emailNotifications,
        browserNotifications,
        soundNotifications,
        publicProfile,
        fullName: fullName || user?.name || "",
      })
    );
    toast({
      title: "Preferences saved",
      description:
        "Notification and profile preferences saved to this browser. Account sync coming soon.",
    });
  };
```

Replace the Save button block (lines 219–224) — swap `Lock` for `Save` and import it:

First, update the import on line 12:

```tsx
import { User, Mail, Lock, Bell, Palette, Save } from "lucide-react";
```
(`Lock` is still used by the password field rows? No — `Lock` is unused after this change except
the password inputs use `type="password"`. Keep `Lock` only if still referenced; otherwise remove
it. Verify by grep below.)

Then replace the button:

```tsx
        <div className="flex justify-end">
          <Button onClick={handleSave} className="rounded-md gap-2">
            <Save className="h-4 w-4" />
            Save Changes
          </Button>
        </div>
```

- [ ] **Step 2: Restore saved preferences on mount**

Add a second `useEffect` (after the accent-color effect, ~line 58) so toggles reflect stored prefs:

```tsx
  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = localStorage.getItem("orbit:preferences");
    if (!stored) return;
    try {
      const prefs = JSON.parse(stored) as {
        emailNotifications: boolean;
        browserNotifications: boolean;
        soundNotifications: boolean;
        publicProfile: boolean;
        fullName?: string;
      };
      setEmailNotifications(prefs.emailNotifications);
      setBrowserNotifications(prefs.browserNotifications);
      setSoundNotifications(prefs.soundNotifications);
      setPublicProfile(prefs.publicProfile);
      if (prefs.fullName) setFullName(prefs.fullName);
    } catch {
      // ignore malformed prefs
    }
  }, []);
```

- [ ] **Step 3: Verify `Lock` is no longer referenced (remove if unused)**

```bash
cd Frontend
grep -n "Lock" app/\(dashboard\)/settings/page.tsx
```
If no matches remain, drop `Lock` from the import. If matches remain, keep it.

- [ ] **Step 4: Verify build**

```bash
npm run build
```
Expected: succeeds. Run dev, open `/settings`, toggle notifications, click Save, reload — toggles
persist; toast reads "saved to this browser. Account sync coming soon."

- [ ] **Step 5: Commit**

```bash
git add "Frontend/app/(dashboard)/settings/page.tsx"
git commit -m "fix(settings): persist client prefs, stop misleading Save toast, fix Save icon"
```

---

## Task 0.4: Fix the `/test` double-active sidebar item (with unit test)

**Files:**
- Modify: `Frontend/components/dashboard/app-shell.tsx`
- Create: `Frontend/tests/components/app-shell-active.test.ts`

**Problem:** Analysis (`/test?tab=analysis`) and Mock Tests (`/test?tab=mock`) both strip to
basePath `/test`; `isActive = pathname === basePath` lights both up.

- [ ] **Step 1: Write the failing test for the active-state helper**

Create `Frontend/tests/components/app-shell-active.test.ts`:

```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd Frontend
npx vitest run tests/components/app-shell-active.test.ts
```
Expected: FAIL — `isNavItemActive` is not exported.

- [ ] **Step 3: Export `isNavItemActive` and use it in `SidebarNavItem`**

In `Frontend/components/dashboard/app-shell.tsx`:

Add `useSearchParams` to the next/navigation import (line 6):

```tsx
import { usePathname, useRouter, useSearchParams } from "next/navigation"
```

Add the exported helper above `SidebarNavItem` (before line 48):

```tsx
/**
 * Determine if a nav item is active. Tab items (/test?tab=analysis) only
 * match when the current ?tab= matches; plain items match their path only.
 */
export function isNavItemActive(href: string, pathname: string, tab: string | null): boolean {
  const [itemPath, query] = href.split("?");
  if (itemPath !== pathname) return false;
  if (!query) return true;
  const itemTab = new URLSearchParams(query).get("tab");
  return itemTab === tab;
}
```

Replace the `SidebarNavItem` active-state lines (60–61):

```tsx
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const tab = searchParams.get("tab")
  const isActive = isNavItemActive(item.href, pathname, tab)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
npx vitest run tests/components/app-shell-active.test.ts
```
Expected: PASS (5 tests).

- [ ] **Step 5: Verify in browser**

```bash
npm run dev
```
Open `/test?tab=analysis` — only Analysis is highlighted. Open `/test?tab=mock` — only Mock Tests.
Open `/dashboard` — only Dashboard.

- [ ] **Step 6: Commit**

```bash
git add Frontend/components/dashboard/app-shell.tsx Frontend/tests/components/app-shell-active.test.ts
git commit -m "fix(shell): only highlight the active /test tab in the sidebar"
```

---

## Task 0.5: Sign out via `useAuth().logout()`; delete the unused legacy header

**Files:**
- Modify: `Frontend/components/dashboard/app-shell.tsx`
- Delete: `Frontend/components/dashboard/header.tsx`

**Problem:** `handleSignOut` only clears `localStorage["token"]` + `router.push("/")`, leaving
`user` state stale and landing on the marketing page. The legacy `components/dashboard/header.tsx`
duplicates this and is imported nowhere.

- [ ] **Step 1: Confirm `header.tsx` is unused**

```bash
cd Frontend
grep -rn "dashboard/header" --include="*.tsx" --include="*.ts" . | grep -v "components/dashboard/header.tsx:"
```
Expected: no output.

- [ ] **Step 2: Use `logout()` from `useAuth`**

In `Frontend/components/dashboard/app-shell.tsx`:

Change the `useAuth` destructure (line 87) to include `logout`:

```tsx
  const { user, hasRole, logout } = useAuth()
```

Replace `handleSignOut` (lines 113–116):

```tsx
  const handleSignOut = () => {
    logout()
  }
```

- [ ] **Step 3: Delete the legacy header**

```bash
cd Frontend
git rm components/dashboard/header.tsx
```

- [ ] **Step 4: Verify build**

```bash
npm run build
```
Expected: succeeds (no dangling imports). Run dev, sign out from the user menu — you land on
`/login` (not `/`) and the sidebar does not flash stale state.

- [ ] **Step 5: Commit**

```bash
git add Frontend/components/dashboard/app-shell.tsx
git commit -m "fix(auth): sign out via useAuth().logout(); delete unused legacy header"
```

---

## Task 0.6: Quiz submit confirmation, timer warnings, unload guard

**Files:**
- Modify: `Frontend/app/(dashboard)/test/quiz/page.tsx`

**Problem:** "Submit Test" fires immediately with no confirmation and no unanswered-count; timer
auto-submit gives no 5-min/1-min warning; no `beforeunload` guard so navigating away loses progress.

- [ ] **Step 1: Add a confirmation dialog + timer-warning + unload-guard state**

In `Frontend/app/(dashboard)/test/quiz/page.tsx`:

Add `Dialog` imports (after the existing ui imports, ~line 7):

```tsx
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
```

Add state inside `TestPage` (after `const { toast } = useToast()`, line 27):

```tsx
  const [showSubmitConfirm, setShowSubmitConfirm] = useState(false)
  const [warnedAt5, setWarnedAt5] = useState(false)
  const [warnedAt1, setWarnedAt1] = useState(false)
```

- [ ] **Step 2: Add timer warnings inside the timer effect**

Replace the timer `useEffect` (lines 66–81) with:

```tsx
  // Timer effect
  useEffect(() => {
    if (!mockTest || timeRemaining <= 0) return

    const timer = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(timer)
          handleSubmit() // Auto-submit when time runs out
          return 0
        }
        return prev - 1
      })
    }, 1000)

    return () => clearInterval(timer)
  }, [mockTest, timeRemaining])

  // Time-limit warnings (5 min and 1 min remaining)
  useEffect(() => {
    if (!mockTest) return
    const fiveMin = 5 * 60
    const oneMin = 1 * 60
    if (!warnedAt5 && timeRemaining <= fiveMin && timeRemaining > oneMin) {
      setWarnedAt5(true)
      toast({ title: "5 minutes remaining", description: "Your test will auto-submit soon." })
    }
    if (!warnedAt1 && timeRemaining <= oneMin && timeRemaining > 0) {
      setWarnedAt1(true)
      toast({
        title: "1 minute remaining!",
        description: "Submitting your answers now is recommended.",
        variant: "destructive",
      })
    }
  }, [timeRemaining, mockTest, warnedAt5, warnedAt1, toast])

  // Guard against accidental navigation away while a test is in progress
  useEffect(() => {
    if (!mockTest || submitting) return
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ""
    }
    window.addEventListener("beforeunload", onBeforeUnload)
    return () => window.removeEventListener("beforeunload", onBeforeUnload)
  }, [mockTest, submitting])
```

- [ ] **Step 3: Make the submit button open the confirmation dialog**

Change the `handleSubmit` signature so it can be called from both the dialog and the timer.
Replace `handleSubmit` (lines 103–131) with two functions:

```tsx
  const performSubmit = async () => {
    if (!mockTest || submitting) return
    setSubmitting(true)
    try {
      const timeTaken = (mockTest.time_limit * 60) - timeRemaining
      const result = await mockTestAPI.submitMockTest(mockTest.test_id, answers, timeTaken)

      toast({ title: "Test submitted successfully", description: "Your answers have been analyzed." })

      // Cache the analysis keyed by submissionId so refresh/back still works (see Task 0.7).
      sessionStorage.setItem(
        `testAnalysis:${result.submission_id}`,
        JSON.stringify(result)
      )

      router.push(`/test/results?testId=${mockTest.test_id}&submissionId=${result.submission_id}`)
    } catch (error) {
      console.error("Error submitting test:", error)
      toast({
        title: "Error",
        description: getErrorMessage(error),
        variant: "destructive",
      })
    } finally {
      setSubmitting(false)
    }
  }

  const handleSubmit = () => {
    setShowSubmitConfirm(true)
  }

  const confirmSubmit = () => {
    setShowSubmitConfirm(false)
    void performSubmit()
  }
```

Add `getErrorMessage` to the imports at the top of the file:

```tsx
import { getErrorMessage } from "@/lib/errors"
```

- [ ] **Step 4: Add the confirmation dialog to the JSX**

Add this just before the closing `</div>` of the outer container (after the Question Palette
`Card`, before line 272 `</div>`):

```tsx
      {/* Submit confirmation */}
      <Dialog open={showSubmitConfirm} onOpenChange={setShowSubmitConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Submit test?</DialogTitle>
            <DialogDescription>
              You have answered{" "}
              {Object.keys(answers).length} of {mockTest.questions.length} questions.
              {Object.keys(answers).length < mockTest.questions.length &&
                ` ${mockTest.questions.length - Object.keys(answers).length} unanswered question${
                  mockTest.questions.length - Object.keys(answers).length > 1 ? "s" : ""
                } will be marked blank.`}{" "}
              You cannot change your answers after submitting.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setShowSubmitConfirm(false)}>
              Keep working
            </Button>
            <Button onClick={confirmSubmit} disabled={submitting} className="gap-2">
              {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              Submit now
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
```

- [ ] **Step 5: Verify build + behavior**

```bash
npm run build
npm run dev
```
Take a mock test: click Submit Test → dialog appears with answered/total counts → Cancel keeps
you in the test → Submit now submits. Let the timer run near 5 min and 1 min (or temporarily lower
thresholds) — warning toasts appear. Closing the tab triggers the browser's "leave site?" prompt.

- [ ] **Step 6: Commit**

```bash
git add "Frontend/app/(dashboard)/test/quiz/page.tsx"
git commit -m "feat(quiz): submit confirmation, timer warnings, beforeunload guard"
```

---

## Task 0.7: Make test results survive refresh (sessionStorage as cache, API-first)

**Files:**
- Modify: `Frontend/app/(dashboard)/test/results/page.tsx`

**Problem:** `testAnalysis` is read from `sessionStorage` then immediately deleted; refresh / new
tab / back button loses it. (Task 0.6 already keys the write by `submissionId`.)

- [ ] **Step 1: Read the keyed cache without deleting it; fall back to the API**

In `Frontend/app/(dashboard)/test/results/page.tsx`, replace the `fetchResults` body (lines 47–61):

```tsx
      try {
        // 1. Try the per-submission sessionStorage cache (fast path; not deleted on read).
        const cached = submissionId ? sessionStorage.getItem(`testAnalysis:${submissionId}`) : null
        if (cached) {
          setAnalysis(JSON.parse(cached) as MockTestAnalysis)
          setLoading(false)
          return
        }

        // 2. Otherwise fetch the authoritative copy from the backend.
        const analysisData = await mockTestAPI.getAnalysisBySubmissionId(submissionId)
        setAnalysis(analysisData)
        sessionStorage.setItem(`testAnalysis:${submissionId}`, JSON.stringify(analysisData))
      } catch (error) {
        console.error("Error fetching results:", error)
        toast({
          title: "Error",
          description: getErrorMessage(error),
          variant: "destructive",
        })
      } finally {
        setLoading(false)
      }
```

Add the import at the top:

```tsx
import { getErrorMessage } from "@/lib/errors"
```

- [ ] **Step 2: Remove the stale global-key write (Task 0.6 already uses the keyed write)**

Confirm no remaining `sessionStorage.setItem('testAnalysis', ...)` (without key) or
`sessionStorage.getItem('testAnalysis')` / `removeItem('testAnalysis')` exists:

```bash
cd Frontend
grep -n "testAnalysis'" "app/(dashboard)/test/results/page.tsx" "app/(dashboard)/test/quiz/page.tsx"
```
Expected: no matches (only the keyed `testAnalysis:` form should remain).

- [ ] **Step 3: Verify build + behavior**

```bash
npm run build
npm run dev
```
Complete a test → results render. Refresh the results page → it still renders (cache retained, not
deleted). Open `/test/results?testId=...&submissionId=...` in a new tab → falls back to the API and
renders.

- [ ] **Step 4: Commit**

```bash
git add "Frontend/app/(dashboard)/test/results/page.tsx"
git commit -m "fix(results): keep keyed sessionStorage cache; fall back to API on miss"
```

---

## Task 0.8: Mark flashcards as beta (real impl is Phase 3.1)

**Files:**
- Modify: `Frontend/app/(dashboard)/flashcards/page.tsx`

**Problem:** "Generate Deck" toasts "coming soon" but the page presents demo decks as if real.

- [ ] **Step 1: Add a visible beta banner and disable the generate action**

In `Frontend/app/(dashboard)/flashcards/page.tsx`, add a beta banner near the top of the returned
JSX (immediately inside the page's root element) and ensure the "Generate Deck" button is disabled
with a tooltip. Read the file first to find the exact button:

```bash
cd Frontend
grep -n "Generate Deck\|coming soon\|DEMO_DECKS" "app/(dashboard)/flashcards/page.tsx"
```

Then:
- Add this banner block near the top of the page content:

```tsx
        <div className="rounded-md border border-primary/30 bg-primary/5 p-3 text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Flashcards are in beta.</span>{" "}
          AI-generated decks arrive in a future update. The decks below are sample content.
        </div>
```

- Change the "Generate Deck" button to `disabled` and keep its onClick showing the toast (so the
  disabled state is self-explanatory). If it is a `<Button>`, add `disabled` and a `title`:

```tsx
          <Button disabled title="AI flashcard generation coming soon" ...>
```

- [ ] **Step 2: Verify build + browser**

```bash
npm run build
npm run dev
```
Open `/flashcards` — the beta banner shows, Generate Deck is visibly disabled.

- [ ] **Step 3: Commit**

```bash
git add "Frontend/app/(dashboard)/flashcards/page.tsx"
git commit -m "feat(flashcards): mark page as beta, disable generate action"
```

---

## Task 0.9: Hide the admin scaffold for now (real impl is Phase 3.2)

**Files:**
- Modify: `Frontend/components/dashboard/app-shell.tsx`

**Problem:** `admin/page.tsx` is an empty scaffold with `users: []`; stat cards always 0.

- [ ] **Step 1: Stop surfacing the Admin nav item until the page is real**

In `Frontend/components/dashboard/app-shell.tsx`, change line 119 so Admin is not added even for
admins (keep the route accessible by direct URL for now):

```tsx
  const allNav = nav
```

(Leave `hasRole` imported if still used elsewhere; if it becomes unused, the build will warn —
remove it from the destructure. Verify in Step 3.)

- [ ] **Step 2: (Optional) Guard the admin route so a direct visit explains it's not ready**

In `Frontend/app/(dashboard)/admin/page.tsx`, add a banner to the top of the returned JSX:

```tsx
        <div className="rounded-md border border-primary/30 bg-primary/5 p-3 text-sm text-muted-foreground mb-6">
          Admin tooling is under construction. The figures below are placeholders.
        </div>
```

- [ ] **Step 3: Verify build + that `hasRole` is still used (or remove it)**

```bash
cd Frontend
npm run build
grep -n "hasRole" components/dashboard/app-shell.tsx
```
If `hasRole` is no longer referenced, remove it from the `useAuth()` destructure to avoid an
unused-var lint error.

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/dashboard/app-shell.tsx "Frontend/app/(dashboard)/admin/page.tsx"
git commit -m "fix(admin): hide nav entry and mark scaffold as under construction"
```

---

## Task 0.10: Point landing CTAs at `/signup`

**Files:**
- Modify: `Frontend/components/marketing/navbar.tsx`
- Modify: `Frontend/components/marketing/hero.tsx`

**Problem:** "Get Started" and "Start Learning Free" send logged-out users to `/dashboard` →
`AuthProtection` → `/login` (a confusing double hop).

- [ ] **Step 1: Navbar "Get Started" → `/signup`**

In `Frontend/components/marketing/navbar.tsx`, change line 39 from `href="/dashboard"` to:

```tsx
                        <Link href="/signup" className="hidden lg:block">
```

- [ ] **Step 2: Hero "Start Learning Free" → `/signup`**

In `Frontend/components/marketing/hero.tsx`, change line 78 from `href="/dashboard"` to:

```tsx
                            <Link href="/signup" className="flex items-center gap-2 group">
```

- [ ] **Step 3: Verify build + browser**

```bash
npm run build
npm run dev
```
On the landing page (logged out), both CTAs go straight to `/signup`.

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/marketing/navbar.tsx Frontend/components/marketing/hero.tsx
git commit -m "fix(landing): point primary CTAs at /signup instead of /dashboard"
```

---

## Task 0.11: Surface silent errors via toast (exam setup, onboarding, dashboard, analytics)

**Files:**
- Modify: `Frontend/components/dashboard/exam-setup-dialog.tsx`
- Modify: `Frontend/components/onboarding/onboarding-container.tsx`
- Modify: `Frontend/app/(dashboard)/dashboard/page.tsx`
- Modify: `Frontend/app/(dashboard)/analytics/page.tsx`

**Problem:** Failures in these flows are `console.error` only — the user clicks and nothing visibly
happens. (Task 0.12 rewrites the actual fetch calls in exam-setup/onboarding; this task adds the
toast feedback and uses `getErrorMessage`.)

- [ ] **Step 1: Add toast feedback to `exam-setup-dialog.tsx`**

In `Frontend/components/dashboard/exam-setup-dialog.tsx`:

Add imports (after existing imports):

```tsx
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
```

Inside `ExamSetupDialog`, add:

```tsx
  const { toast } = useToast();
```

In `handlePresetClick`'s catch (lines 60–62), replace with:

```tsx
    } catch (error) {
      console.error("Error creating exam:", error);
      toast({
        title: "Couldn't create exam",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
```

Same for `handleCustomCreate`'s catch (lines 84–86):

```tsx
    } catch (error) {
      console.error("Error creating custom exam:", error);
      toast({
        title: "Couldn't create exam",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
```

- [ ] **Step 2: Add toast feedback to `onboarding-container.tsx`**

In `Frontend/components/onboarding/onboarding-container.tsx`:

Add imports:

```tsx
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
```

Add inside `OnboardingContainer`:

```tsx
  const { toast } = useToast();
```

Replace the three `catch` blocks (lines 30–34, 43–45, 82–85) with toasting versions:

```tsx
    } catch (error) {
      console.error("Error saving onboarding data:", error);
      toast({
        title: "Couldn't save your details",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
```

```tsx
    } catch (error) {
      console.error("Error completing onboarding:", error);
      // non-fatal: continue to dashboard
    }
```

```tsx
    } catch (error) {
      console.error("Error in step 2:", error);
      toast({
        title: "Couldn't set up your study goal",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      setIsLoading(false);
    }
```

- [ ] **Step 3: Add toast feedback to `dashboard/page.tsx` stats fetch**

In `Frontend/app/(dashboard)/dashboard/page.tsx`, find the `Promise.allSettled` block and the
existing `console.error` at lines ~80–82. Read the file to locate it:

```bash
cd Frontend
grep -n "console.error\|Promise.allSettled\|allSettled" "app/(dashboard)/dashboard/page.tsx"
```

Ensure `useToast` is imported (add if missing) and add a single toast when any of the settled
results rejected. Replace the existing silent `console.error` block with:

```tsx
    const results = await Promise.allSettled([
      pdfAPI.listPDFs(),
      chatAPI.listChatSessions(),
      mockTestAPI.listMockTests(),
    ]);
    const failed = results.filter((r) => r.status === "rejected");
    if (failed.length > 0) {
      console.error("Dashboard stats fetch failures:", failed);
      toast({
        title: "Some stats failed to load",
        description: "Parts of your dashboard may appear incomplete.",
        variant: "destructive",
      });
    }
```

(Keep the existing fulfillment extraction logic that follows; only replace the silent error path.)

- [ ] **Step 4: Add toast feedback to `analytics/page.tsx`**

In `Frontend/app/(dashboard)/analytics/page.tsx`, replace the silent `catch` (lines ~70–72) with:

```tsx
      } catch (error) {
        console.error("Error fetching analytics:", error);
        toast({
          title: "Couldn't load analytics",
          description: getErrorMessage(error),
          variant: "destructive",
        });
      }
```

Add imports for `useToast` (if missing) and `getErrorMessage`.

- [ ] **Step 5: Verify build**

```bash
npm run build
```
Expected: succeeds. (Full failure-path browser testing happens after Task 0.12 wires the calls.)

- [ ] **Step 6: Commit**

```bash
git add Frontend/components/dashboard/exam-setup-dialog.tsx \
        Frontend/components/onboarding/onboarding-container.tsx \
        "Frontend/app/(dashboard)/dashboard/page.tsx" \
        "Frontend/app/(dashboard)/analytics/page.tsx"
git commit -m "fix: surface silent errors via toast across exam setup, onboarding, dashboard, analytics"
```

---

## Task 0.12: Route the missing `/api/*` calls through the axios client + register onboarding router

**Files:**
- Modify: `Backend/src/main.py` (register `onboarding_router`)
- Modify: `Frontend/lib/api.ts` (add `examAPI`, `onboardingAPI`)
- Modify: `Frontend/lib/context/dashboard-context.tsx`
- Modify: `Frontend/components/onboarding/onboarding-container.tsx`
- Modify: `Frontend/components/dashboard/exam-setup-dialog.tsx`
- Modify: `Frontend/app/(dashboard)/layout.tsx`

**Problem:** `dashboard-context`, `onboarding-container`, `exam-setup-dialog`, and
`(dashboard)/layout` call `fetch("/api/...")`. There is no `app/api/` directory and no `rewrites`
proxy in `next.config.js`, so these 404 against the Next server — and raw `fetch` does **not**
attach the Bearer token the way the axios `api` instance does. Separately, `onboarding_router`
exists in the backend but is **not registered** in `main.py`, so even the backend has no
`/api/onboarding` route.

**Strategy:** Route every one of these calls through the shared axios `api` instance (which uses the
correct `baseURL` and attaches the token). Add typed `examAPI` + `onboardingAPI` exports to
`lib/api.ts`. Register `onboarding_router` in the backend.

- [ ] **Step 1: Register `onboarding_router` in the backend**

In `Backend/src/main.py`, add `onboarding_router` to the import block (line 6–19):

```python
from src.routers import (
    auth_router,
    pdf_router,
    document_router,
    question_router,
    analysis_router,
    mock_test_router,
    teacher_router,
    analytics_router,
    exam_router,
    subject_router,
    collection_router,
    material_router,
    onboarding_router,
)
```

And register it (after `app.include_router(material_router)`, line 44):

```python
app.include_router(onboarding_router)
```

- [ ] **Step 2: Add `examAPI` and `onboardingAPI` to `lib/api.ts`**

In `Frontend/lib/api.ts`, add these exports (anywhere after the `api` instance and existing API
objects, e.g. at the end of the file). Read the file first to confirm the export style:

```bash
cd Frontend
grep -n "^export const .*API" lib/api.ts
```

Then append:

```ts
// ─── Exams / Subjects ─────────────────────────────────────────────────────────
export const examAPI = {
  async listExams(): Promise<any> {
    const res = await api.get("/api/exams/");
    return res.data;
  },
  async createExam(payload: { name: string; icon?: string; is_active?: boolean }): Promise<any> {
    const res = await api.post("/api/exams/", payload);
    return res.data;
  },
  async setActiveExam(examId: string): Promise<any> {
    const res = await api.patch(`/api/exams/${examId}/active`);
    return res.data;
  },
  async createSubject(examId: string, name: string): Promise<any> {
    const res = await api.post(`/api/exams/${examId}/subjects`, { name });
    return res.data;
  },
};

// ─── Onboarding ───────────────────────────────────────────────────────────────
export const onboardingAPI = {
  async getStatus(): Promise<{ onboarding_completed: boolean }> {
    const res = await api.get("/api/onboarding/");
    return res.data;
  },
  async saveStep1(data: {
    name: string;
    role: string;
    institute: string;
    language: string;
  }): Promise<any> {
    const res = await api.post("/api/onboarding/", data);
    return res.data;
  },
  async complete(): Promise<any> {
    const res = await api.post("/api/onboarding/complete");
    return res.data;
  },
};
```

> Note: `any` return types are intentional for Phase 0 — they are replaced with real types in
> Phase 1 (`lib/types/api.ts`). This task only fixes correctness (right URL + token attached).

- [ ] **Step 3: Replace `fetch` in `dashboard-context.tsx`**

In `Frontend/lib/context/dashboard-context.tsx`:

Add the import (after the React import):

```tsx
import api from "@/lib/api";
```

Replace `refreshExams` (lines 79–98) with:

```tsx
  const refreshExams = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = (await api.get("/api/exams/")).data as Exam[];
      setExams(data);
      setActiveExam((prev) => {
        if (prev) {
          const stillActive = data.find((e) => e.id === prev.id);
          return stillActive ?? (data.find((e) => e.isActive) || data[0] || null);
        }
        return data.find((e) => e.isActive) || data[0] || null;
      });
    } catch (error) {
      console.error("Failed to fetch exams:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);
```

- [ ] **Step 4: Replace `fetch` in `(dashboard)/layout.tsx`**

In `Frontend/app/(dashboard)/layout.tsx`, add the import:

```tsx
import { onboardingAPI } from "@/lib/api";
```

Replace the `checkOnboarding` body (lines 22–38) with:

```tsx
    const checkOnboarding = async () => {
      try {
        const data = await onboardingAPI.getStatus();
        if (data.onboarding_completed === false) {
          setShouldRedirect(true);
        }
      } catch (error) {
        console.error("Error checking onboarding status:", error);
      } finally {
        setOnboardingChecked(true);
      }
    };
```

- [ ] **Step 5: Replace `fetch` in `onboarding-container.tsx`**

In `Frontend/components/onboarding/onboarding-container.tsx`, add imports:

```tsx
import { examAPI, onboardingAPI } from "@/lib/api";
```

Replace `handleStep1Next` (lines 15–35):

```tsx
  const handleStep1Next = async (data: {
    name: string;
    role: string;
    institute: string;
    language: string;
  }) => {
    setIsLoading(true);
    try {
      await onboardingAPI.saveStep1(data);
      setStep(2);
    } catch (error) {
      console.error("Error saving onboarding data:", error);
      toast({
        title: "Couldn't save your details",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };
```

Replace `completeOnboarding` (lines 37–46):

```tsx
  const completeOnboarding = async () => {
    try {
      await onboardingAPI.complete();
    } catch (error) {
      console.error("Error completing onboarding:", error);
    }
  };
```

Replace the exam/subject creation inside `handleStep2Next` (lines 51–78):

```tsx
      if (presetId) {
        const preset = PRESET_EXAMS.find((p) => p.id === presetId);
        if (!preset) throw new Error("Preset not found");

        // 1. Create exam
        const exam = await examAPI.createExam({
          name: preset.name,
          icon: preset.icon,
          is_active: true,
        });

        // 2. Create subjects
        await Promise.all(
          preset.subjects.map((subjectName) => examAPI.createSubject(exam.id, subjectName))
        );
      }

      await completeOnboarding();
      router.push("/dashboard");
```

- [ ] **Step 6: Replace `fetch` in `exam-setup-dialog.tsx`**

In `Frontend/components/dashboard/exam-setup-dialog.tsx`, add imports:

```tsx
import { examAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
```

Add `const { toast } = useToast();` inside the component.

Replace `handlePresetClick`'s exam/subject creation (lines 34–56):

```tsx
    setIsLoading(true);
    try {
      const exam = await examAPI.createExam({
        name: preset.name,
        icon: preset.icon,
        is_active: true,
      });

      await Promise.all(
        preset.subjects.map((subjectName) => examAPI.createSubject(exam.id, subjectName))
      );

      onExamCreated(exam.id);
      onOpenChange(false);
    } catch (error) {
      console.error("Error creating exam:", error);
      toast({
        title: "Couldn't create exam",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
```

Replace `handleCustomCreate`'s exam creation (lines 71–80):

```tsx
    setIsLoading(true);
    try {
      const exam = await examAPI.createExam({
        name: customName.trim(),
        is_active: true,
      });

      onExamCreated(exam.id);
      onOpenChange(false);
    } catch (error) {
      console.error("Error creating custom exam:", error);
      toast({
        title: "Couldn't create exam",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
```

- [ ] **Step 7: Verify there are no remaining `fetch("/api/...")` calls**

```bash
cd Frontend
grep -rn 'fetch("/api/\|fetch(`/api/' lib components app --include="*.tsx" --include="*.ts"
```
Expected: no output.

- [ ] **Step 8: Start backend + frontend and run the full onboarding + exam-creation flow**

```bash
# Terminal 1 (backend)
cd /Users/utkarsh/Developer/Projects/Orbit/Backend
source venv/bin/activate
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
```
Confirm `http://localhost:8001/api/onboarding/` returns a JSON response (not 404).

```bash
# Terminal 2 (frontend)
cd /Users/utkarsh/Developer/Projects/Orbit/Frontend
npm run dev
```
Sign up as a new student → onboarding step 1 saves → step 2 creates an exam + subjects → lands on
`/dashboard` with the new exam visible. From the dashboard, open the Exam Setup dialog and create
a preset exam → it appears in the sidebar. On failure (stop the backend), a destructive toast
appears instead of silent failure.

- [ ] **Step 9: Run the unit tests**

```bash
cd Frontend
npx vitest run
```
Expected: all tests pass (errors.test.ts + app-shell-active.test.ts).

- [ ] **Step 10: Commit**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit
git add Backend/src/main.py Frontend/lib/api.ts Frontend/lib/context/dashboard-context.tsx \
        Frontend/components/onboarding/onboarding-container.tsx \
        Frontend/components/dashboard/exam-setup-dialog.tsx \
        "Frontend/app/(dashboard)/layout.tsx"
git commit -m "fix: route /api calls through axios client; register onboarding router"
```

---

## Phase 0 — Final verification

- [ ] **Step 1: Full build + lint**

```bash
cd /Users/utkarsh/Developer/Projects/Orbit/Frontend
npm run lint
npm run build
npx vitest run
```
Expected: lint clean (allow existing warnings unrelated to these changes), build succeeds, all
unit tests pass.

- [ ] **Step 2: Smoke-test the critical flows in the browser**

With the backend running:
1. `/login` and `/signup` — no console hydration warnings.
2. Click "Forgot password?", "Terms", "Privacy" from the forms — pages render, no 404.
3. Landing CTAs → `/signup`.
4. New-user signup → onboarding → exam + subjects created → dashboard.
5. Dashboard Exam Setup dialog → preset exam created → sidebar updates.
6. Take a mock test → Submit → confirmation dialog → results. Refresh results → still renders.
7. `/settings` → toggle notifications → Save → reload → toggles persist.
8. Sign out → lands on `/login`.
9. `/test?tab=analysis` and `/test?tab=mock` — only the correct sidebar item is active.
10. `/flashcards` and `/admin` — beta/under-construction banners shown; Admin not in sidebar.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin phase0/critical-fixes
gh pr create --title "Phase 0: critical frontend fixes" --body "Implements docs/superpowers/plans/2026-06-19-orbit-frontend-phase0-critical-fixes.md"
```

---

## Self-review

**Spec coverage (Phase 0 items from `Frontend/IMPROVEMENT_PLAN.md`):**
- 0.1 hydration mismatch → Task 0.1 ✓
- 0.2 dead auth links → Task 0.2 ✓
- 0.3 Settings Save → Task 0.3 ✓
- 0.4 double-active sidebar → Task 0.4 ✓
- 0.5 sign-out → Task 0.5 ✓
- 0.6 quiz confirmation/warnings/unload → Task 0.6 ✓
- 0.7 results volatility → Task 0.7 ✓
- 0.8 flashcards stub → Task 0.8 ✓
- 0.9 admin stub → Task 0.9 ✓
- 0.10 landing CTAs → Task 0.10 ✓
- 0.11 silent errors → Task 0.11 ✓
- 0.12 missing /api routes → Task 0.12 ✓

**Placeholder scan:** No "TBD"/"implement later"/"add error handling" placeholders. Every code
step contains complete code. The two `any` return types in Task 0.12 Step 2 are explicitly scoped
to Phase 0 with a note that Phase 1 replaces them — a documented interim, not a placeholder.

**Type/name consistency:** `isNavItemActive(href, pathname, tab)` is defined in Task 0.4 and used
there only. `getErrorMessage(error)` is defined in Task 0.0 and used in Tasks 0.6, 0.7, 0.11, 0.12
with the same signature. `examAPI.createExam`/`createSubject` and `onboardingAPI.saveStep1`/
`complete`/`getStatus` are defined in Task 0.12 Step 2 and used in Steps 3–6 with matching names.
The keyed sessionStorage key `testAnalysis:${submissionId}` is written in Task 0.6 Step 3 and read
in Task 0.7 Step 1 — consistent.