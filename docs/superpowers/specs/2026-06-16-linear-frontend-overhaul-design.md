# Linear-Inspired Frontend Overhaul Design

## Scope

Rewrite the dashboard-area frontend (everything behind auth) to a Linear-inspired design language using the DESIGN.md Vercel system as a reference. Landing page, login, and signup remain unchanged.

## Design System

### Colors

**Dark mode (primary default):**
- Background: `#0a0a0a`
- Foreground: `#fafafa`
- Card: `#141414`
- Secondary: `#1a1a1a`
- Muted: `#262626`
- Muted foreground: `#888888`
- Border/hairline: `rgba(255,255,255,0.06)`
- Accent: `#5E6AD2` (Linear purple-blue)
- Link: `#0070f3`
- Error: `#ee0000` / Error soft: `#f7d4d6`
- Success: `#10b981`
- Warning: `#f5a623`

**Light mode:**
- Background: `#fafafa`
- Foreground: `#171717`
- Card: `#ffffff`
- Secondary: `#f5f5f5`
- Muted: `#ebebeb`
- Muted foreground: `#888888`
- Border/hairline: `rgba(0,0,0,0.06)`
- Accent: `#5E6AD2`
- Semantic colors same as dark

### Typography

- **Primary font**: Inter (400/500/600) — closest open-source Geist substitute
- **Mono font**: JetBrains Mono (400) — for technical labels, code
- Display sizes: aggressive negative letter-spacing (-2.4px at 48px scaling down)
- Body: neutral tracking
- Headlines: sentence-case, period-terminated
- Weight ceiling: 600 (no 700+)
- Font feature settings: `"ss01", "ss02"` for geometric alternates

### Elevation (Stacked Shadows)

Remove all neomorphic shadows and glow utilities. Use DESIGN.md elevation levels:

| Level | Shadow | Use |
|---|---|---|
| 0 | None | Full-bleed sections |
| 1 | `0 0 0 1px rgba(0,0,0,0.06)` inset | Default card chrome |
| 2 | `0px 1px 1px rgba(0,0,0,0.02), 0px 2px 2px rgba(0,0,0,0.04)` + hairline | Slightly elevated cards |
| 3 | `0px 2px 2px rgba(0,0,0,0.04), 0px 8px 8px -8px rgba(0,0,0,0.04)` + hairline | Feature cards |
| 4 | `0px 2px 2px rgba(0,0,0,0.04), 0px 8px 16px -4px rgba(0,0,0,0.04)` + hairline | Pricing/float cards |
| 5 | `0px 1px 1px rgba(0,0,0,0.02), 0px 8px 16px -4px rgba(0,0,0,0.04), 0px 24px 32px -8px rgba(0,0,0,0.06)` + hairline | Modals/dialogs |

### Shapes (Border Radius)

- `6px` — buttons, inputs, dropdowns (base UI radius)
- `8px` — cards, feature containers
- `12px` — large containers, pricing cards
- No pill shapes in dashboard (marketing only)

## App Shell & Sidebar

Replace current collapsible sidebar with Linear pattern:

- **Width**: 240px expanded, 48px collapsed (icon-only rail)
- **Background**: Matches theme background
- **Active indicator**: 2px left-edge accent bar (not glow/shadow)
- **Nav items**: Subtle hover backgrounds (`rgba(255,255,255,0.04)` dark, `rgba(0,0,0,0.04)` light). Text 14px Inter 500
- **Workspace switcher**: Exam selector dropdown at top of sidebar
- **Bottom section**: User avatar + settings link
- **Header**: Minimal 44px height. Breadcrumb trail left, theme toggle + cmd+k trigger + avatar right
- Remove `backdrop-blur-xl` and `bg-card/50` — use solid backgrounds
- Remove `shadow-[0_0_12px_rgba(56,189,248,0.15)]` active glow on nav items

## Student Dashboard

- Welcome bar: "Good morning, {name}." — clean, no emoji, period-terminated
- Stats grid: wire to real API data (document count, chat sessions, tests taken, avg score)
- Subject cards: clean 1px hairline border, simple hover lift (no BentoGrid animation)
- Assigned tests: table-style list with status badges
- Collections: wire to actual API data, remove empty-state stub
- **New: Flashcard tab** — card-based flip interface for AI-generated flashcards
- **New: Analytics section** — recharts-powered subject performance, weakness breakdown

## Teacher Dashboard

- Remove nested header (AppShell provides navigation)
- Stats: metric cards with `caption-mono` labels, large display numbers
- Student list: data table with columns (name, tests, avg score, weaknesses)
- Weakness badges: clean pill badges with subtle background tint
- "Create Test" primary CTA button (accent color, 6px radius)
- **New: Student detail drawer** — click student row to see detailed analytics in slide-over

## Chat Page

- Switch from `addMessageToChat` to `addMessageToChatStream` for streaming
- Clean message bubbles: no glow, subtle 1px border for AI messages
- Sidebar: 240px, clean tree view for collections/subjects/materials
- Input: clean textarea with 6px radius, no neomorphic pressed style
- **New: Real file attachments** — actual file picker instead of mock filenames

## Test Page

- Tab bar: clean pill-style tabs, subtle underline indicator
- Form inputs: 40px height, 6px radius, hairline border
- Analysis results: clean cards with left-edge colored indicators
- Mock test config: two-column layout, summary card with mono labels
- **New: Sortable test list** — table with status, date, score columns

## Settings Page

- Remove `neo-card` classes, use clean card chrome
- Wire profile fields to actual auth context data
- Wire save to API
- Accent color picker: functional (set CSS variable)
- Remove "this is a demo" messages

## Onboarding

- Clean step indicators (not MagicCard)
- Form inputs matching DESIGN.md form-input spec
- Subtle progress indicator

## New Features (Previously Out-of-Scope)

### Flashcard UI
- Card flip component with front (question) / back (answer)
- Deck browser with progress tracking
- AI-generated deck creation flow (select material → generate → study)
- Keyboard shortcuts for flip/next

### Student Analytics
- Subject performance bar chart (recharts)
- Weakness radar chart
- Study progress over time (line chart)
- Per-subject score trend

### Admin Dashboard
- User list table (email, name, role, status)
- Role assignment dropdown
- License status view (frontend scaffold — backend not ready)

### Streaming Chat
- SSE-based streaming in ChatInterface
- Token-by-token rendering
- Abort controller for cancellation

### Real File Uploads
- Wire document-uploader to actual API
- Replace mock chat attachments with real file picker
- Upload progress indicator

## Unchanged

- Landing page (`/`)
- Login (`/login`), signup (`/signup`)
- Marketing components
- Auth form component (used in auth routes only)

## Files to Modify

### Design System
- `app/globals.css` — new color tokens, remove neomorphic utilities, add stacked shadows
- `tailwind.config.ts` — new font families, radius scale, remove neomorphic animations
- `app/layout.tsx` — switch fonts to Inter + JetBrains Mono

### App Shell
- `components/dashboard/app-shell.tsx` — Linear-style sidebar, 44px header

### Pages
- `app/(dashboard)/dashboard/page.tsx` — clean student dashboard, real stats
- `app/(dashboard)/teacher/page.tsx` — clean teacher dashboard, data table
- `app/(dashboard)/chat/page.tsx` — streaming, clean layout
- `app/(dashboard)/test/page.tsx` — clean forms, better test list
- `app/(dashboard)/test/quiz/page.tsx` — clean quiz UI
- `app/(dashboard)/test/results/page.tsx` — clean results, fix hardcoded colors
- `app/(dashboard)/settings/page.tsx` — wire to API, remove neo-card
- `app/onboarding/page.tsx` — clean steps

### Components
- `components/dashboard/sidebar.tsx` — Linear sidebar
- `components/dashboard/active-study-card.tsx` — clean card
- `components/dashboard/subject-card.tsx` — clean card, no glow
- `components/dashboard/collections-panel.tsx` — clean panel
- `components/dashboard/chat/chat-interface.tsx` — streaming
- `components/dashboard/chat/chat-input.tsx` — clean input, real attachments
- `components/dashboard/chat/message-item.tsx` — clean bubbles
- `components/dashboard/chat/collections-chat-sidebar.tsx` — clean tree

### New Files
- `app/(dashboard)/flashcards/page.tsx` — flashcard study page
- `components/dashboard/flashcard/flashcard-deck.tsx` — deck browser
- `components/dashboard/flashcard/flashcard-card.tsx` — flip card
- `app/(dashboard)/analytics/page.tsx` — student analytics page
- `components/dashboard/analytics/subject-chart.tsx` — bar chart
- `components/dashboard/analytics/weakness-radar.tsx` — radar chart
- `app/(dashboard)/admin/page.tsx` — admin dashboard page
- `components/dashboard/admin/user-table.tsx` — user list