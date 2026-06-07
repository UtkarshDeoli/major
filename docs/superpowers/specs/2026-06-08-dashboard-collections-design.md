# Dashboard & Collections Redesign — Design Spec

**Date:** 2026-06-08  
**Project:** Orbit — AI-Powered Study Platform  
**Scope:** Frontend Dashboard UI/UX overhaul + Study Collections workspace feature  
**Status:** Approved

---

## 1. Problem Statement

The current dashboard uses a dated neomorphic design language (`neo-card`, `neo-button`) that does not match the landing page's premium "Cosmic Productivity" aesthetic. The UI is static, lacks visual feedback, and does not support a student's primary mental model: **preparing for one exam at a time** with organized study materials.

Students need a system to:
1. Define an exam goal (e.g., JEE Mains)
2. Organize subjects under that exam
3. Group materials into named **Collections** (e.g., "Mechanics", "Organic Chemistry")
4. Upload PDFs and chat with AI scoped to those materials

---

## 2. Goals

1. **Visual parity**: Dashboard must match the landing page's aesthetic quality.
2. **Personalized study hub**: Center the dashboard around the student's active exam.
3. **Collections system**: Hierarchical workspace — Exam → Subject → Collection → Materials.
4. **Better UX**: Meaningful animations, clear hierarchy, intuitive interactions, strong feedback.
5. **Usability**: Accessibility-compliant, responsive, performance-conscious.
6. **Onboarding**: First-time users must complete a 3-step setup before accessing the dashboard.

---

## 3. Aesthetic Direction: "Cosmic Productivity"

Borrowed directly from the landing page. Dark, premium, alive with motion. Gradient accents, generous spacing, staggered reveals.

### 3.1 Tokens & Patterns

| Element | Pattern | Usage |
|---------|---------|-------|
| **Cards** | `MagicCard` with animated gradient border (`#38bdf8` → `#3b82f6`) | Major cards (stats, collections, active study) |
| **Utility Cards** | `bg-card` + subtle `border` | Secondary items (PDF list rows, settings) |
| **Animations** | `Container` component with `delay={0.1 + index * 0.1}` | Staggered entrance on mount and scroll |
| **Typography** | `font-heading` headings, `font-subheading italic` accents | Page titles, exam names, decorative text |
| **Buttons** | shadcn `Button` with `bg-primary` | All buttons. Keep `neo-button` only if dark theme demands |
| **Background** | `bg-background` with optional `bg-glow-blue` | Hero sections, active study zone |
| **Icons** | Lucide React `h-5 w-5` for headers, `h-4 w-4` inline | Consistent across dashboard |
| **Radius** | `rounded-2xl` cards, `rounded-xl` buttons, `rounded-lg` inputs | Generous rounding throughout |
| **Shadows** | Tailwind `shadow-lg`, `shadow-xl` on hover | Replace neomorphic dual-shadow |

### 3.2 Interaction Patterns

- **Hover**: Cards lift with `shadow-lg` transition (`duration-300`). Buttons show arrow-right translate (`group-hover:translate-x-1`).
- **Active**: `bg-primary/10 text-primary` with `ring-2 ring-primary/20`.
- **Loading**: Pulse-glow animation (`animate-pulse-glow` from `globals.css`), not generic spinners.
- **Empty states**: `Container` entrance + centered illustration + CTA.

---

## 4. User Onboarding Flow

New users must complete a 3-step onboarding before accessing the dashboard. This replaces the current "blank dashboard" experience.

### 4.1 Flow Overview

```
┌──────────────────────────────────────────────┐
│  STEP 1: About You                           │
│  • Name (auto-filled from signup)             │
│  • Role: Student / Teacher                    │
│  • Institute: School / Coaching / College /   │
│    None (text input)                         │
│  • Preferred Language for AI chat              │
├──────────────────────────────────────────────┤
│  STEP 2: Your Study Goal                      │
│  Option A: Choose from preset exams           │
│    • JEE Mains / NEET / UPSC / CAT / etc.    │
│    • Auto-populates subjects                  │
│  Option B: Custom setup                       │
│    • "I'll set up my own subjects"           │
│    • Goes to dashboard with empty state       │
├──────────────────────────────────────────────┤
│  STEP 3: Quick Tour (optional skip)          │
│  • 3-slide tooltip tour of dashboard          │
│  • "How Collections work" mini-demo           │
│  • "Start Studying" CTA                       │
└──────────────────────────────────────────────┘
```

### 4.2 Step 1: About You

**Layout**: Centered card on dark background, `max-w-md`, `MagicCard` gradient border.
**Fields**:
- **Name**: Pre-filled from auth, editable.
- **Role**: Segmented control (Student / Teacher). Default: Student.
  - If Teacher: Onboarding completes here, dashboard shows teacher-oriented view (future feature).
- **Institute**: Text input with optional autocomplete for known coaching centers.
  - Label: "Where do you study? (Optional)"
  - Placeholder: "e.g., Allen Kota, FIITJEE, Delhi Public School..."
- **Language**: Dropdown/select.
  - Options: English (default), Hindi, Tamil, Telugu, Marathi, Bengali, Gujarati, Kannada, Malayalam, Punjabi, Urdu.
  - This sets the default language for AI chat responses. Can be changed later in Settings.

**Buttons**:
- "Skip for now" (ghost button, bottom-left)
- "Next" (primary, bottom-right, disabled until role selected)

**Progress indicator**: Step dots (1 of 3) at top.

### 4.3 Step 2: Your Study Goal

**Layout**: Same centered card. Height expands based on content.

**Two clear options**:

**Option A: I have an exam to prepare for**
- Grid of preset exam cards (bento style, 2 columns)
- Each card: Exam icon (Lucide), exam name, brief tagline
- Examples: JEE Mains ("Engineering"), NEET ("Medical"), UPSC ("Civil Services"), CAT ("MBA"), CBSE 12th ("Board Exams"), State PSC, GATE, etc.
- Clicking a card:
  - Sets this as the active exam
  - Auto-creates default subjects (e.g., JEE → Physics, Chemistry, Mathematics)
  - Shows a loading state: "Setting up your study workspace..."
  - Proceeds to Step 3

**Option B: I'll organize my own way**
- Single card with `FolderOpen` icon
- Text: "Start with a blank workspace and add your own subjects."
- Clicking:
  - No exam created yet
  - Goes to dashboard with Active Study empty state
  - User adds subjects manually via `ExamSetupDialog`

**Buttons**:
- "← Back" (ghost)
- "Next" (hidden until Option A or B selected)

### 4.4 Step 3: Quick Tour (Optional)

**Layout**: Same card, but content is a carousel/swipeable.

**Slide 1**: "This is your Study Hub"
- Screenshot/mock of the Active Study section
- Text: "Your exam and subjects, all in one place."

**Slide 2**: "Organize with Collections"
- Screenshot/mock of Collections Panel
- Text: "Group your PDFs into Collections — like 'Mechanics' or 'Organic Chemistry'."

**Slide 3**: "Chat with AI, your way"
- Screenshot/mock of chat interface
- Text: "Ask questions in {language}. AI answers using your uploaded materials."

**Buttons**:
- "Skip tour" (ghost, always available)
- "← Back" (ghost)
- "Start Studying" (primary, on last slide)

### 4.5 Technical Details

- **Route**: `/onboarding` — only accessible if `onboardingCompleted: false` in user profile.
- **Guard**: If authenticated user with `onboardingCompleted: true` visits `/onboarding`, redirect to `/dashboard`.
- **Guard**: If unauthenticated user visits `/onboarding`, redirect to `/login`.
- **Data persistence**: Each step saves to backend as user progresses (not just at the end). If user drops off, they resume where they left off.
- **Backend field**: Extend `User` model:
  ```typescript
  interface User {
    // ...existing fields
    onboardingCompleted: boolean;
    role: 'student' | 'teacher';
    institute?: string;
    preferredLanguage: string;  // ISO code, default 'en'
    activeExamId?: string;
  }
  ```
- **Animation**: Each step uses `Container` fade-up entrance. Transition between steps: horizontal slide (`translateX` with `opacity`), 300ms.
- **Mobile**: Same 3-step flow, cards become full-width with safe area padding.

---

## 4. Dashboard Page Architecture

### 4.1 Layout Zones (top to bottom)

```
┌─────────────────────────────────────────────────────────────┐
│  ZONE 1: Welcome + Quick Actions (compact, ~60px tall)     │
├─────────────────────────────────────────────────────────────┤
│  ZONE 2: Active Study (prominent, ~40% viewport)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ Subject A│ │ Subject B│ │ Subject C│  Bento grid        │
│  │ + last   │ │ + last   │ │ + last   │                    │
│  │ collection│ │ collection│ │ collection│                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
├─────────────────────────────────────────────────────────────┤
│  ZONE 3: Stats Row (compact, 4 cards)                       │
├─────────────────────────────────────────────────────────────┤
│  ZONE 4: My Collections (bento grid, all exams)            │
├─────────────────────────────────────────────────────────────┤
│  ZONE 5: Recent Activity + Study Streak (bottom)           │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Zone Descriptions

**Zone 1: Welcome Bar**
- Shrink from current hero banner to a compact bar
- Text: "Welcome back, {name}!" in `font-heading text-lg`
- Two quick-action pills: `New Chat`, `Take Test`
- Uses `Container` with `delay={0}`

**Zone 2: Active Study (NEW)**
- The visual and functional center of the dashboard
- Answers: "What am I studying right now?"
- See Section 5 for full details

**Zone 3: Stats**
- 4 small `MagicCard`s: Documents, Chat Sessions, Mock Tests, Avg Score
- Compact: `p-4` instead of `p-6`, smaller icon containers
- Staggered entrance: `delay={0.4 + index * 0.1}`

**Zone 4: My Collections**
- Bento grid of all exams the student has created
- Each card: Exam name, subject count, collection count, progress ring, "active" badge
- Uses the same `MagicCard` pattern as landing page Features section
- Staggered: `delay={0.6 + index * 0.1}`

**Zone 5: Bottom Row**
- Left: Recent Activity (timeline, keep current data)
- Right: Study Streak (heatmap, keep current)
- Both wrapped in `MagicCard` or plain `bg-card` with border

---

## 5. Active Study Section (Centerpiece)

### 5.1 Header
- Label: "Currently Preparing For" in `text-xs uppercase tracking-wider text-muted-foreground`
- Exam name: `font-subheading italic text-2xl text-foreground` (e.g., *JEE Mains*)
- Progress ring: `ProgressRing` component showing overall readiness
- Quick action: "Continue Last Session" → opens Collections Panel pre-filtered

### 5.2 Subjects Grid
- Horizontal scroll on mobile, 3-column bento on desktop
- Each `SubjectCard`:
  - Subject icon (colored circle with Lucide icon)
  - Subject name: `font-medium text-sm`
  - Collection count: "3 collections"
  - Last studied: "2 hours ago"
  - Mini horizontal progress bar (thin, `h-1`)
- Uses `MagicCard` with subtle subject-specific tint (optional)

### 5.3 Empty State
- Full-width `MagicCard` with prominent gradient
- Icon: `Target` or `GraduationCap` in `h-12 w-12 text-primary`
- Text: "Set your exam goal to get started" in `font-heading text-xl`
- Subtext: "Add an exam and start organizing your study materials."
- Button: "Add Exam" → opens `ExamSetupDialog`
- `Container` entrance animation

---

## 6. Collections Panel (Slide-Over)

A `Sheet` component slides in from the right when any subject or exam card is clicked.

### 6.1 Panel Specs

- **Width**: `w-[480px]` on desktop, `w-full` on mobile
- **Background**: `bg-background` with `border-l`
- **Z-index**: Above all dashboard content, below toasts
- **Animation**: `slide-in-from-right` with `duration-300`

### 6.2 Panel Header
- Back button: `← Back to Dashboard` (closes panel)
- Exam name: `font-heading text-xl` + optional icon
- Close `X` button top-right
- Sticky header with `border-b` separator

### 6.3 Panel Body (Hierarchical)

```
▼ Physics                          ← Collapsible (default: expanded if active)
  ├── Collection: Mechanics       ← Collection row
  │   ├── kinematics.pdf          ← PDF row
  │   ├── forces.pdf              ← PDF row
  │   └── [+ Upload Material]     ← Upload button
  ├── Collection: Electricity
  │   ├── circuits.pdf
  │   └── [+ Upload Material]
  └── [+ New Collection]          ← Add collection button
▼ Chemistry
▼ Mathematics
┌─────────────────────────────────┐
│ 💬 Chat with this Exam          ← Sticky FAB at bottom
│   (uses all materials as RAG)  │
└─────────────────────────────────┘
```

### 6.4 Interaction Details

**Subjects** — `Collapsible` from shadcn/ui:
- `ChevronDown` / `ChevronRight` icon toggle
- Single-expand mode (only one subject open at a time) or multi-expand
- Default: active subject expanded, others collapsed

**Collections** — Indented list:
- Folder icon (`Folder` or `BookOpen`) + collection name
- Collection count badge: "2 materials"
- Clicking collection expands to show PDF list

**PDF List** — File rows:
- `FileText` icon + filename + file size (human-readable)
- Hover shows action buttons: Download, Delete (with `Tooltip`)
- Upload button: `Button variant="outline"` with `border-dashed`

**Chat FAB** — Fixed at panel bottom:
- `MessageSquare` icon + "Chat with this Exam"
- Click behavior (configurable):
  - **Option A**: Panel expands to full width, splits Collection (left) / Chat (right)
  - **Option B**: Navigate to `/chat?examId=xxx` with pre-filtered context
  - **Option C**: Inline chat drawer within panel

---

## 7. Data Model

### 7.1 TypeScript Interfaces

```typescript
interface Exam {
  id: string;
  name: string;                    // e.g., "JEE Mains"
  description?: string;
  icon?: string;                   // Lucide icon name
  color?: string;                  // HSL value for theming
  subjects: Subject[];
  isActive: boolean;               // Only one active at a time
  createdAt: Date;
  updatedAt: Date;
}

interface Subject {
  id: string;
  examId: string;
  name: string;                    // e.g., "Physics"
  icon?: string;
  collections: Collection[];
  progress: number;                // 0-100 (aggregated)
  lastStudiedAt?: Date;
}

interface Collection {
  id: string;
  subjectId: string;
  name: string;                    // user-defined, e.g., "Mechanics"
  description?: string;
  materials: Material[];
  createdAt: Date;
}

interface Material {
  id: string;
  collectionId: string;
  name: string;
  type: 'pdf' | 'image' | 'text';
  size: number;                    // bytes
  url: string;
  uploadedAt: Date;
  ragIndexed: boolean;
}
```

### 7.2 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/exams` | GET | List all exams for user |
| `/api/exams` | POST | Create new exam |
| `/api/exams/:id` | PATCH | Update exam |
| `/api/exams/:id/active` | PATCH | Set as active exam |
| `/api/exams/:id/subjects` | GET | List subjects |
| `/api/exams/:id/subjects` | POST | Add subject |
| `/api/subjects/:id` | PATCH | Update subject |
| `/api/subjects/:id/collections` | GET | List collections |
| `/api/subjects/:id/collections` | POST | Create collection |
| `/api/collections/:id/materials` | GET | List materials |
| `/api/collections/:id/materials` | POST | Upload material |
| `/api/materials/:id` | DELETE | Remove material |
| `/api/chat?collectionId=:id` | POST | Chat with RAG scoped to collection |
| `/api/onboarding` | POST | Save onboarding step data |
| `/api/onboarding` | GET | Get current onboarding status |
| `/api/onboarding/complete` | POST | Mark onboarding as complete |

### 7.3 State Management

- **Exam/Subject/Collection tree**: React Context (read-heavy, stable during session)
- **Data fetching**: SWR for caching, deduping, revalidation
- **Chat state**: Local to chat components (existing pattern)
- **Panel state**: Local state in `CollectionsPanel` (open/closed, expanded subjects)

---

## 8. Component Inventory

### 8.1 New Components

| Component | File | Purpose |
|-----------|------|---------|
| `ActiveStudyCard` | `components/dashboard/active-study-card.tsx` | Centerpiece exam card with progress ring |
| `SubjectCard` | `components/dashboard/subject-card.tsx` | Subject tile for bento grid |
| `CollectionsPanel` | `components/dashboard/collections-panel.tsx` | The `Sheet` slide-over container |
| `SubjectAccordion` | `components/dashboard/subject-accordion.tsx` | Collapsible subject section |
| `CollectionItem` | `components/dashboard/collection-item.tsx` | Collection row with expand/collapse |
| `MaterialList` | `components/dashboard/material-list.tsx` | PDF list + upload button |
| `ExamSetupDialog` | `components/dashboard/exam-setup-dialog.tsx` | Dialog to create/select first exam |
| `ProgressRing` | `components/ui/progress-ring.tsx` | Circular progress (reusable) |
| `BentoGrid` | `components/ui/bento-grid.tsx` | Layout helper for the collections grid |

### 8.2 Modified Components

| Component | Changes |
|-----------|---------|
| `dashboard/page.tsx` | Full rewrite: new zones, Active Study, collections grid |
| `app-shell.tsx` | Keep structure, add subtle `Container` entrance to sidebar |
| `globals.css` | Keep animations, **deprecate** `.neo-card` / `.neo-button` (remove from new code, keep for backwards compat) |

### 8.3 Reused from Landing Page

| Component | Source | Usage |
|---|---|---|
| `MagicCard` | `components/ui/magic-card.tsx` | All major dashboard cards |
| `Container` | `components/global/container.tsx` | Staggered entrance animations |
| `Wrapper` | `components/global/wrapper.tsx` | Section wrappers |

---

## 9. Responsive Behavior

### 9.1 Desktop (≥1024px)
- Full 3-column bento grid for subjects and collections
- Collections Panel: `w-[480px]` slide-over
- Chat split-view: panel expands to `w-[800px]` with two panes

### 9.2 Tablet (768–1023px)
- 2-column bento grid
- Collections Panel: `w-[400px]`
- Subjects become 2-column grid

### 9.3 Mobile (<768px)
- Single column, full-width cards
- Subjects: horizontal scrollable row (`overflow-x-auto`)
- Collections Panel: `w-full`, covers entire screen
- Chat FAB becomes sticky bottom bar (`h-14`)
- Upload: native file picker, progress in toast
- Touch targets: minimum `48px` height for all interactive rows

---

## 10. Accessibility

- **Keyboard**: All cards focusable with `tabindex="0"` and visible `ring-2 ring-primary` focus state.
- **Panel**: `aria-modal="true"`, focus trap, `Escape` to close.
- **Accordion**: `Collapsible` with `aria-expanded`, `aria-controls`.
- **PDF list**: Each item has `aria-label="{filename}, {size}"`.
- **Color**: Never rely on color alone. Icons + text always paired.
- **Motion**: Respect `prefers-reduced-motion`. Disable `Container` stagger, instant transitions.
- **Screen readers**: Exam name in `h2`, subject names in `h3`, collections in `h4`.

---

## 11. Animation Spec

### 11.1 Entrance Animations

| Zone | Animation | Delay | Duration |
|------|-----------|-------|----------|
| Welcome | `Container` fade-up | 0s | 0.5s |
| Active Study header | `Container` fade-up | 0.1s | 0.5s |
| Subject cards | `Container` stagger fade-up | 0.2s + 0.1s per card | 0.5s |
| Stats | `Container` stagger fade-up | 0.4s + 0.1s per card | 0.5s |
| My Collections | `Container` stagger fade-up | 0.6s + 0.1s per card | 0.5s |
| Bottom row | `Container` fade-up | 0.8s | 0.5s |

### 11.2 Interaction Animations

| Interaction | Animation | Duration |
|-------------|-----------|----------|
| Card hover | `translateY(-4px)` + `shadow-lg` | 300ms |
| Button hover | Icon `translateX(4px)` | 300ms |
| Panel open | Slide from right (`translateX(100%)` → `0`) | 300ms |
| Panel close | Slide to right (`0` → `translateX(100%)`) | 200ms |
| Accordion expand | `height: 0` → `auto` with opacity | 200ms |
| Upload progress | Pulse-glow on upload button | continuous |
| Loading state | `animate-pulse-glow` on card | continuous |

---

## 12. Backend Requirements

### 12.1 New Database Collections

- `exams` — stores exam documents with `userId`, `isActive`
- `subjects` — stores subject documents with `examId`
- `collections` — stores collection documents with `subjectId`
- `materials` — stores material metadata with `collectionId`, `ragIndexed`
- Extend existing `users` collection with `activeExamId` field

### 12.2 RAG Integration

- When a material is uploaded and `ragIndexed` becomes `true`, it is added to the vector store.
- Chat queries scoped to a `collectionId` should only retrieve chunks from materials in that collection.
- Chat queries scoped to an `examId` should retrieve from all materials across all subjects/collections under that exam.

### 12.3 File Storage

- Store uploaded files in a dedicated directory (e.g., `/uploads/{userId}/{examId}/{subjectId}/{collectionId}/`)
- Generate presigned URLs for download
- Limit file size (e.g., 10MB per file)

---

## 13. Migration Path

### 13.1 Existing Users
- On first load after update, show `ExamSetupDialog` if no exam exists.
- Pre-populate with a default exam based on their existing mock test history (if available).
- Existing documents in the old "flat" structure should be migrated into a default "Uncategorized" collection.
- Set `onboardingCompleted: true` for all existing users so they are not forced through onboarding.

### 13.2 New Users
- After signup/login, redirect to `/onboarding`.
- Complete all 3 steps before accessing `/dashboard`.
- If user skips onboarding, set `onboardingCompleted: true` but `activeExamId: null`. Dashboard shows empty state.

### 13.3 Gradual Adoption
- Phase 1: Visual redesign only (new dashboard page, no collections feature)
- Phase 2: Onboarding flow + Collections feature (new API + panel)
- Phase 3: RAG scoping (chat integration)

---

## 14. Success Criteria

1. **Visual parity**: Dashboard looks and feels like the landing page.
2. **Task completion**: A student can complete onboarding → create an exam → add subjects → add collections → upload PDFs within 3 minutes.
3. **Performance**: Dashboard TTI < 2s, panel open < 300ms, upload feedback < 1s, onboarding step transition < 300ms.
4. **Accessibility**: Passes axe-core audit with 0 violations.
5. **Mobile**: All core tasks work on a 375px-wide device.

---

## 15. Open Questions

1. Should the chat FAB open inline in the panel (Option C) or navigate to the chat page (Option B)?
2. Should we auto-create default collections (e.g., "Chapter 1", "Chapter 2") when a subject is created, or let users name everything?
3. Should the "My Collections" bento grid on the dashboard show all exams, or only non-active exams (with the active one featured separately)?

---

*Spec written on 2026-06-08. Approved by user. Ready for implementation planning.*
