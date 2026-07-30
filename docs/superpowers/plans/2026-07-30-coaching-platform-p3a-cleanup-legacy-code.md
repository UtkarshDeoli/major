# Orbit Coaching-Platform Reshape — Phase 3a (Legacy Code Cleanup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unused and superseded legacy code from the coaching-platform reshape. Reduce surface area, confusion, and maintenance burden before moving to Phase 4.

**Principles:**
- Delete only code that is truly unused or fully superseded by the new coaching-platform flows.
- Keep backward-compatible data fields in MongoDB documents; do not run destructive migrations.
- Do not remove code that is still referenced by other active features.
- Each deletion must be accompanied by passing backend tests and a clean frontend build.

**Tech Stack:** FastAPI, Motor (MongoDB), Pydantic v2, pytest; Next.js 16 App Router, TypeScript, Tailwind.

## Scope of this plan

Phase 3a covers cleanup after Phases 1–3:
- Remove legacy `POST /classes/enroll` and `GET /classes/enroll/{code}` (superseded by `/classes/join`).
- Remove legacy class detail endpoints/routes that are no longer used (`/classes/enroll/{code}`, `/classes/enroll`).
- Remove `class_ids` / legacy `teacher_id` updates from `POST /classes/enroll` behavior (we will instead stop writing them; fields remain in existing docs).
- Remove dead UI components/pages related to old teacher/student class integration (e.g., `teacher-classes-panel.tsx` if unused, legacy roster dialog already removed in P2).
- Remove unused API helpers in `Frontend/lib/api.ts` (e.g., `previewEnroll`, `enroll` if no longer used).
- Remove unused backend service/router files if any were created during earlier plans and are now fully superseded.

## File Structure

**Backend — modify:**
- `Backend/src/routers/class_router.py` — remove `/enroll` and `/enroll/{code}` endpoints and related Pydantic models.
- `Backend/src/core/data_store.py` — remove helpers only used by deleted endpoints if any.

**Frontend — modify:**
- `Frontend/lib/api.ts` — remove `previewEnroll` / `enroll` if no longer referenced.
- `Frontend/components/dashboard/teacher/teacher-classes-panel.tsx` — confirm it is still used on `/teacher` dashboard; if it remains as a useful summary, keep it but remove any dead code missed in P2.

**Tests — modify:**
- Remove tests that assert deleted behavior.
- Keep and update class tests so the focused Phase 2/3 suites still pass.

---

### Task 1: Identify legacy candidates

**Files:**
- Search across backend/frontend/tests.

**Interfaces:**
- Input: current codebase.
- Output: a definitive list of legacy endpoints, functions, components, and tests that are safe to remove.

- [ ] **Step 1: Search for references to legacy endpoints**

```bash
cd Backend
grep -rn "/classes/enroll" src/ tests/
grep -rn "preview_enroll\|enroll_in_class" src/ tests/
cd ../Frontend
grep -rn "previewEnroll\|classAPI.enroll" app/ components/ lib/
```

- [ ] **Step 2: List candidates**

Expected candidates:
- `Backend/src/routers/class_router.py`:
  - `EnrollRequest`, `EnrollPreview`, `preview_enroll`, `enroll_in_class`
- `Frontend/lib/api.ts`:
  - `classAPI.previewEnroll`, `classAPI.enroll`
- `Backend/tests/test_class_router.py` or other test files that test `/classes/enroll` (if any exist).
- Dead imports of `get_current_user` in `class_router.py` if no longer needed.

- [ ] **Step 3: Confirm no active frontend page uses the legacy endpoints**

Check:
- `Frontend/app/(dashboard)/classes/page.tsx` uses `/classes/join`, not `/classes/enroll`.
- No other page/dialog calls `classAPI.enroll` or `classAPI.previewEnroll`.

- [ ] **Step 4: Document the cleanup list**

Write a brief list to the task report.

---

### Task 2: Remove legacy backend endpoints

**Files:**
- Modify: `Backend/src/routers/class_router.py`

**Interfaces:**
- Consumes: legacy enrollment endpoints.
- Produces: cleaner router with only `/`, `/{id}`, `/{id}/teachers`, `/join`, `/me`, `/{id}/content`, `/{id}/students`, `/{id}/tests`, and `/{id}/students/{email}` (remove).

- [ ] **Step 1: Delete legacy models and endpoints**

In `Backend/src/routers/class_router.py`:

1. Remove `class EnrollRequest(BaseModel):` and `class EnrollPreview(BaseModel):`.
2. Remove `preview_enroll` endpoint (`@router.get("/enroll/{code}", ...)`).
3. Remove `enroll_in_class` endpoint (`@router.post("/enroll", ...)`).

- [ ] **Step 2: Remove unused imports**

After deletion, `get_current_user` may no longer be needed in `class_router.py`. Verify with `grep` and remove if unused.

- [ ] **Step 3: Run backend tests**

```bash
cd Backend && source venv/bin/activate
pytest tests/test_class_router_p2.py tests/test_class_subjects.py tests/test_class_materials.py tests/test_class_material_generation.py tests/test_class_student_flows.py -v
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add Backend/src/routers/class_router.py
git commit -m "refactor(backend): remove legacy /classes/enroll endpoints"
```

---

### Task 3: Remove legacy frontend API helpers

**Files:**
- Modify: `Frontend/lib/api.ts`

**Interfaces:**
- Consumes: legacy `classAPI.previewEnroll` and `classAPI.enroll`.
- Produces: `classAPI` without those methods.

- [ ] **Step 1: Confirm no references**

```bash
cd Frontend
grep -rn "previewEnroll\|classAPI.enroll" app/ components/ lib/
```

- [ ] **Step 2: Delete the methods**

Remove from `classAPI`:
```ts
  async previewEnroll(code: string): Promise<any> { ... }
  async enroll(enrollCode: string): Promise<any> { ... }
```

- [ ] **Step 3: Lint + build**

```bash
cd Frontend && npm run lint && npm run build
```
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add Frontend/lib/api.ts
git commit -m "refactor(frontend): remove legacy class enrollment API helpers"
```

---

### Task 4: Remove unused tests

**Files:**
- Search and delete/update test files.

**Interfaces:**
- Remove tests that reference deleted endpoints.

- [ ] **Step 1: Search for legacy endpoint tests**

```bash
cd Backend
grep -rn "/classes/enroll" tests/
grep -rn "preview_enroll\|enroll_in_class" tests/
```

- [ ] **Step 2: Delete or update affected tests**

If a dedicated `test_class_router.py` exists and only tests legacy endpoints, delete it. If it contains still-valid tests, keep and update them.

- [ ] **Step 3: Run backend tests**

```bash
pytest tests/test_class_router_p2.py tests/test_class_subjects.py tests/test_class_materials.py tests/test_class_material_generation.py tests/test_class_student_flows.py -v
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add Backend/tests/
git commit -m "test(backend): remove/update legacy class enrollment tests"
```

---

### Task 5: Clean up dead frontend code

**Files:**
- `Frontend/components/dashboard/teacher/teacher-classes-panel.tsx`
- `Frontend/app/(dashboard)/teacher/page.tsx`

**Interfaces:**
- Remove any remaining dead roster dialog state/handlers if they survived P2 cleanup.
- Confirm the panel is still used as a summary on `/teacher`.

- [ ] **Step 1: Inspect teacher-classes-panel.tsx**

Check for unused state like `openDetail`, `detail`, `loadingDetail`, `openClassDetail`, `handleRemoveStudent`, or the roster Dialog. If present, remove them.

- [ ] **Step 2: Inspect teacher page**

Confirm `TeacherClassesPanel` is still imported/rendered. If it is no longer used at all, remove the import and the component file.

- [ ] **Step 3: Lint + build**

```bash
cd Frontend && npm run lint && npm run build
```
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/dashboard/teacher/teacher-classes-panel.tsx Frontend/app/(dashboard)/teacher/page.tsx
git commit -m "refactor(frontend): remove dead teacher class panel code"
```

---

### Task 6: Final review + merge

- [ ] **Step 1: Run full backend focused test suite**

```bash
cd Backend && source venv/bin/activate
pytest tests/test_class_router_p2.py tests/test_class_subjects.py tests/test_class_materials.py tests/test_class_material_generation.py tests/test_class_student_flows.py -v
```
Expected: all pass.

- [ ] **Step 2: Run frontend lint + build**

```bash
cd Frontend && npm run lint && npm run build
```
Expected: 0 errors.

- [ ] **Step 3: Generate review package and final review**

```bash
SK=/Users/utkarsh/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.0/skills/subagent-driven-development
"$SK/scripts/review-package" $(git merge-base master HEAD) HEAD
```

- [ ] **Step 4: Merge and push**

```bash
git checkout master
git merge feat/coaching-p3a-cleanup-legacy-code --no-ff -m "Merge Phase 3a: remove legacy class enrollment code"
git push origin master
```

---

## Phase 3a completion checklist

- [ ] Legacy `/classes/enroll` and `/classes/enroll/{code}` endpoints removed.
- [ ] Legacy frontend API helpers removed.
- [ ] Unused tests removed/updated.
- [ ] Dead frontend teacher class panel code removed.
- [ ] Backend tests pass.
- [ ] Frontend lint + build pass.
- [ ] Final review approved and branch merged.

## Non-blocking follow-ups for later phases

- Audit remaining `teacher_id` / `class_ids` fields on User documents for potential future removal or migration.
- Consolidate class enrollment into a single code path (`/classes/join`) across the product.
