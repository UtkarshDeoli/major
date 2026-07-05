# Test Page Split — Implementation Spec

## Goal
Split the 1075-line `test/page.tsx` monolith into two independent route pages (`/analysis` and `/mock-tests`) with a shared `PdfSelector` component, eliminating the `?tab=` query-param pattern and the double-active sidebar bug.

## Routes

| Route | What | Notes |
|---|---|---|
| `/analysis` | Full analysis screen | New page, extracted from `test/page.tsx` |
| `/mock-tests` | Full mock test screen | New page, extracted from `test/page.tsx` |
| `/test` | 301 redirect → `/analysis` | Keep file as redirect |
| `/test/quiz` | Quiz flow | Unchanged |
| `/test/results` | Results | Unchanged |

## Sidebar nav changes (`app-shell.tsx`)
- `Analysis` → `/analysis` (was `/test?tab=analysis`)
- `Mock Tests` → `/mock-tests` (was `/test?tab=mock`)

## Component split

**New files:**
- `app/(dashboard)/analysis/page.tsx` — analysis screen (full page, own state, imports PdfSelector)
- `app/(dashboard)/mock-tests/page.tsx` — mock tests screen (full page, own state, imports PdfSelector)
- `components/dashboard/test/pdf-selector.tsx` — reusable PDF syllabus + QP picker

**Modified files:**
- `app/(dashboard)/test/page.tsx` — simplified to redirect to `/analysis`
- `components/dashboard/app-shell.tsx` — update nav hrefs

**Unchanged:**
- `test/quiz/page.tsx`
- `test/results/page.tsx`

## PdfSelector component

Props:
```ts
interface PdfSelectorProps {
  onSelectionChange: (selection: PdfSelection) => void;
  initialSelection?: Partial<PdfSelection>;
}

interface PdfSelection {
  syllabusId: string | null;
  questionPaperIds: string[];
  notesId: string | null;
}
```

Fetches `pdfAPI.listPDFs()` on mount. Renders:
- Syllabus dropdown (filtered to tagged PDFs)
- Question paper multi-select (checkboxes)
- Optional notes select (for mock test screen)

## State management

Each page manages its own state independently:
- `analysis/page.tsx` — PdfSelection, analysis result, loading/error states
- `mock-tests/page.tsx` — PdfSelection, mock config, test list, teacher controls

No shared state middleware. Both are client components with `useEffect` fetches.

## Redirect handling

`/test/page.tsx` becomes:
```tsx
import { redirect } from "next/navigation";

export default function TestPage() {
  redirect("/analysis");
}
```

## Build verification

- `npm run build` succeeds
- `/analysis` and `/mock-tests` render their respective UIs
- `/test` redirects to `/analysis`
- Sidebar nav items link to correct routes
- No double-active sidebar state
