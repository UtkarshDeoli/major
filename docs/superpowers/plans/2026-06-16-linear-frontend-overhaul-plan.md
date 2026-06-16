# Linear-Inspired Frontend Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite all dashboard-area frontend pages to a Linear-inspired clean design language, add flashcard/analytics/admin features, and wire real data.

**Architecture:** Replace neomorphic dark theme with clean design token system (Inter + JetBrains Mono fonts, hairline borders, stacked shadows). Rebuild AppShell to Linear-style sidebar. Rewrite each page. Add new feature pages (flashcards, analytics, admin). Landing/login/signup remain unchanged.

**Tech Stack:** Next.js 15, TypeScript, Tailwind CSS, Inter + JetBrains Mono fonts, recharts, shadcn/ui components, framer-motion

---

## Task 1: Update Design Tokens in globals.css

**Files:**
- Modify: `Frontend/app/globals.css`

- [ ] **Step 1: Replace the entire `:root` and `.light` theme blocks with new Linear-inspired tokens**

Replace the `@layer base { :root { ... } .light { ... } }` block (lines 19–101) with:

```css
@layer base {
  :root {
    /* Dark mode (default) */
    --background: 0 0% 4%;
    --foreground: 0 0% 98%;

    --card: 0 0% 8%;
    --card-foreground: 0 0% 98%;

    --popover: 0 0% 8%;
    --popover-foreground: 0 0% 98%;

    --primary: 237 56% 60%;
    --primary-foreground: 0 0% 100%;

    --secondary: 0 0% 10%;
    --secondary-foreground: 0 0% 98%;

    --muted: 0 0% 15%;
    --muted-foreground: 0 0% 53%;

    --accent: 237 56% 60%;
    --accent-foreground: 0 0% 100%;

    --destructive: 0 84% 60%;
    --destructive-foreground: 0 0% 98%;

    --border: 0 0% 100% / 6%;
    --input: 0 0% 100% / 8%;
    --ring: 237 56% 60%;

    --chart-1: 237 56% 60%;
    --chart-2: 160 84% 39%;
    --chart-3: 0 84% 60%;
    --chart-4: 0 0% 60%;
    --chart-5: 0 0% 70%;

    --radius: 6px;

    /* Stacked shadow tokens */
    --shadow-1: 0 0 0 1px rgba(0,0,0,0.06);
    --shadow-2: 0px 1px 1px rgba(0,0,0,0.02), 0px 2px 2px rgba(0,0,0,0.04);
    --shadow-3: 0px 2px 2px rgba(0,0,0,0.04), 0px 8px 8px -8px rgba(0,0,0,0.04);
    --shadow-4: 0px 2px 2px rgba(0,0,0,0.04), 0px 8px 16px -4px rgba(0,0,0,0.04);
    --shadow-5: 0px 1px 1px rgba(0,0,0,0.02), 0px 8px 16px -4px rgba(0,0,0,0.04), 0px 24px 32px -8px rgba(0,0,0,0.06);
  }

  .light {
    --background: 0 0% 98%;
    --foreground: 0 0% 9%;

    --card: 0 0% 100%;
    --card-foreground: 0 0% 9%;

    --popover: 0 0% 100%;
    --popover-foreground: 0 0% 9%;

    --primary: 237 56% 60%;
    --primary-foreground: 0 0% 100%;

    --secondary: 0 0% 96%;
    --secondary-foreground: 0 0% 9%;

    --muted: 0 0% 92%;
    --muted-foreground: 0 0% 45%;

    --accent: 237 56% 60%;
    --accent-foreground: 0 0% 100%;

    --destructive: 0 84% 60%;
    --destructive-foreground: 0 0% 98%;

    --border: 0 0% 0% / 6%;
    --input: 0 0% 0% / 8%;
    --ring: 237 56% 60%;

    /* Light stacked shadows */
    --shadow-1: 0 0 0 1px rgba(0,0,0,0.06);
    --shadow-2: 0px 1px 1px rgba(0,0,0,0.02), 0px 2px 2px rgba(0,0,0,0.04);
    --shadow-3: 0px 2px 2px rgba(0,0,0,0.04), 0px 8px 8px -8px rgba(0,0,0,0.04);
    --shadow-4: 0px 2px 2px rgba(0,0,0,0.04), 0px 8px 16px -4px rgba(0,0,0,0.04);
    --shadow-5: 0px 1px 1px rgba(0,0,0,0.02), 0px 8px 16px -4px rgba(0,0,0,0.04), 0px 24px 32px -8px rgba(0,0,0,0.06);
  }
}
```

- [ ] **Step 2: Remove all neomorphic shadow utilities and glow utilities**

Delete the entire `@layer utilities` block (lines 114–181) that contains `.neo-flat`, `.neo-pressed`, `.neo-raised`, `.neo-button`, `.neo-card`, `.neo-input`, `.text-glow`, `.text-glow-strong`, `.bg-glow`, `.bg-glow-blue`, `.bg-glow-green`, `.border-glow`. Replace with:

```css
@layer utilities {
  .shadow-card {
    box-shadow: var(--shadow-2), var(--shadow-1);
  }
  .shadow-card-hover {
    box-shadow: var(--shadow-3), var(--shadow-1);
  }
  .shadow-float {
    box-shadow: var(--shadow-4), var(--shadow-1);
  }
  .shadow-modal {
    box-shadow: var(--shadow-5), var(--shadow-1);
  }
}
```

- [ ] **Step 3: Remove all starfield/nebula/comet animation keyframes**

Delete all animation keyframes and their corresponding utility classes for: `float`, `twinkle`, `pulse-glow`, `nebula`, `shooting`, `pulse-slow`, `absolute-star`, `shooting-star`, `comet`. Keep only `fadeIn`, `typingAnimation`, `gradient-shift` and their utilities.

- [ ] **Step 4: Remove gradient-text animation class**

Delete the `.gradient-text` class and `@keyframes gradient-shift` block.

- [ ] **Step 5: Commit**

```bash
git add Frontend/app/globals.css
git commit -m "refactor: replace neomorphic design tokens with Linear-inspired system"
```

---

## Task 2: Update Tailwind Config and Root Layout (Fonts)

**Files:**
- Modify: `Frontend/tailwind.config.ts`
- Modify: `Frontend/app/layout.tsx`

- [ ] **Step 1: Update tailwind.config.ts with new font families and radius scale**

Replace the `extend` section of the theme:

```ts
extend: {
  backgroundImage: {
    'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
    'gradient-conic':
      'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
  },
  borderRadius: {
    lg: 'var(--radius)',
    md: 'calc(var(--radius) - 2px)',
    sm: 'calc(var(--radius) - 4px)',
  },
  colors: {
    background: 'hsl(var(--background))',
    foreground: 'hsl(var(--foreground))',
    card: {
      DEFAULT: 'hsl(var(--card))',
      foreground: 'hsl(var(--card-foreground))',
    },
    popover: {
      DEFAULT: 'hsl(var(--popover))',
      foreground: 'hsl(var(--popover-foreground))',
    },
    primary: {
      DEFAULT: 'hsl(var(--primary))',
      foreground: 'hsl(var(--primary-foreground))',
    },
    secondary: {
      DEFAULT: 'hsl(var(--secondary))',
      foreground: 'hsl(var(--secondary-foreground))',
    },
    muted: {
      DEFAULT: 'hsl(var(--muted))',
      foreground: 'hsl(var(--muted-foreground))',
    },
    accent: {
      DEFAULT: 'hsl(var(--accent))',
      foreground: 'hsl(var(--accent-foreground))',
    },
    destructive: {
      DEFAULT: 'hsl(var(--destructive))',
      foreground: 'hsl(var(--destructive-foreground))',
    },
    border: 'hsl(var(--border))',
    input: 'hsl(var(--input))',
    ring: 'hsl(var(--ring))',
    chart: {
      '1': 'hsl(var(--chart-1))',
      '2': 'hsl(var(--chart-2))',
      '3': 'hsl(var(--chart-3))',
      '4': 'hsl(var(--chart-4))',
      '5': 'hsl(var(--chart-5))',
    },
  },
  fontFamily: {
    sans: ['var(--font-sans)', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
    mono: ['var(--font-mono)', 'JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
  },
  keyframes: {
    'accordion-down': {
      from: { height: '0' },
      to: { height: 'var(--radix-accordion-content-height)' },
    },
    'accordion-up': {
      from: { height: 'var(--radix-accordion-content-height)' },
      to: { height: '0' },
    },
    'fade-in': {
      from: { opacity: '0' },
      to: { opacity: '1' },
    },
  },
  animation: {
    'accordion-down': 'accordion-down 0.2s ease-out',
    'accordion-up': 'accordion-up 0.2s ease-out',
    'fade-in': 'fade-in 0.2s ease-out',
  },
},
```

Remove the old `fontFamily` (heading/subheading/base), old animations (float/flip/rotate/orbit/ripple/image-glow), old spacing (`1/8`).

- [ ] **Step 2: Update root layout.tsx to use Inter + JetBrains Mono fonts**

Replace the entire layout file:

```tsx
import './globals.css';
import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import { ThemeProvider } from '@/components/providers/theme-provider';
import { Toaster } from '@/components/ui/toaster';
import { cn } from '@/lib/utils';
import { AuthProvider } from '@/lib/context/auth-context';

const sans = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  fontFeatureSettings: '"ss01", "ss02"',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  weight: ['400'],
});

export const metadata: Metadata = {
  title: 'Orbit - AI-Powered Study Platform',
  description: 'Master your studies with AI-powered document analysis, quiz generation, and smart learning tools.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          'min-h-screen bg-background text-foreground antialiased font-sans overflow-x-hidden',
          sans.variable,
          mono.variable,
        )}
        suppressHydrationWarning
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <AuthProvider>
            {children}
            <Toaster />
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 3: Verify dev server starts**

Run: `cd Frontend && npm run dev`
Expected: Server starts without errors on port 3000

- [ ] **Step 4: Commit**

```bash
git add Frontend/tailwind.config.ts Frontend/app/layout.tsx
git commit -m "refactor: switch to Inter + JetBrains Mono fonts, update tailwind config"
```

---

## Task 3: Rebuild AppShell (Linear-style Sidebar + Header)

**Files:**
- Modify: `Frontend/components/dashboard/app-shell.tsx`

- [ ] **Step 1: Rewrite AppShell with Linear-style sidebar**

Replace the entire file with a Linear-inspired shell: 240px expanded / 48px collapsed sidebar, 44px header, clean nav items with left-edge active indicator, no glow effects, no backdrop blur.

```tsx
"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { usePathname, useRouter } from "next/navigation"
import {
  User,
  Menu,
  X,
  Settings,
  ChevronsLeft,
  ChevronsRight,
  LayoutDashboard,
  MessageSquare,
  FileBarChart,
  BookOpen,
  Sparkles,
  BarChart3,
  Shield,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"
import { useAuth } from "@/lib/context/auth-context"

const nav = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/test?tab=analysis", label: "Analysis", icon: FileBarChart },
  { href: "/test?tab=mock", label: "Mock Tests", icon: BookOpen },
  { href: "/flashcards", label: "Flashcards", icon: Sparkles },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
]

const bottomNav = [
  { href: "/settings", label: "Settings", icon: Settings },
]

function SidebarNavItem({
  item,
  collapsed,
  isMobile,
  setMobileOpen,
}: {
  item: typeof nav[0]
  collapsed: boolean
  isMobile: boolean
  setMobileOpen: (v: boolean) => void
}) {
  const pathname = usePathname()
  const basePath = item.href.split("?")[0]
  const isActive = pathname === basePath

  return (
    <Link
      href={item.href}
      onClick={() => isMobile && setMobileOpen(false)}
      className={cn(
        "group flex items-center rounded-md text-[13px] font-medium transition-colors duration-150 relative",
        collapsed ? "justify-center px-0 py-2" : "gap-3 px-3 py-1.5",
        isActive
          ? "text-foreground bg-secondary"
          : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
      )}
      title={collapsed ? item.label : undefined}
    >
      {isActive && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-[1px] w-[2px] h-4 bg-primary rounded-full" />
      )}
      <item.icon className={cn("shrink-0", collapsed ? "h-5 w-5" : "h-4 w-4")} />
      {!collapsed && <span className="truncate">{item.label}</span>}
    </Link>
  )
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const { user, hasRole } = useAuth()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [collapsed, setCollapsedState] = useState(false)

  useEffect(() => {
    const saved = localStorage.getItem("orbit:sidebar-collapsed")
    if (saved !== null) setCollapsedState(saved === "true")
  }, [])

  const setCollapsed = (value: boolean) => {
    setCollapsedState(value)
    localStorage.setItem("orbit:sidebar-collapsed", String(value))
  }

  useEffect(() => {
    const check = () => {
      const mobile = window.innerWidth < 1024
      setIsMobile(mobile)
      if (mobile) setCollapsed(false)
    }
    check()
    window.addEventListener("resize", check)
    return () => window.removeEventListener("resize", check)
  }, [])

  const handleSignOut = () => {
    localStorage.removeItem("token")
    router.push("/")
  }

  const sidebarWidth = collapsed ? "w-12" : "w-60"
  const allNav = hasRole("admin") ? [...nav, { href: "/admin", label: "Admin", icon: Shield }] : nav

  return (
    <div className="h-screen flex overflow-hidden bg-background">
      {isMobile && mobileOpen && (
        <div className="fixed inset-0 bg-black/50 z-40" onClick={() => setMobileOpen(false)} />
      )}

      <aside
        className={cn(
          "fixed lg:relative inset-y-0 left-0 z-50 flex flex-col border-r bg-background transition-all duration-200",
          sidebarWidth,
          isMobile && !mobileOpen ? "-translate-x-full" : "translate-x-0"
        )}
      >
        {/* Header */}
        <div className={cn(
          "h-11 flex items-center shrink-0 border-b",
          collapsed ? "justify-center px-0" : "justify-between px-3"
        )}>
          {!collapsed && (
            <Link href="/dashboard" className="flex items-center gap-2 overflow-hidden">
              <div className="relative h-5 w-5 shrink-0">
                <Image src="/logo.png" alt="Orbit" fill className="object-contain" priority />
              </div>
              <span className="font-semibold text-sm tracking-tight whitespace-nowrap">Orbit</span>
            </Link>
          )}
          {collapsed && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 rounded-md p-0 hover:bg-secondary"
              onClick={() => setCollapsed(false)}
              title="Expand sidebar"
            >
              <div className="relative h-4 w-4">
                <Image src="/logo.png" alt="Orbit" fill className="object-contain" priority />
              </div>
            </Button>
          )}
          {!collapsed && !isMobile && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 rounded-md p-0 text-muted-foreground hover:text-foreground"
              onClick={() => setCollapsed(true)}
              title="Collapse sidebar"
            >
              <ChevronsLeft className="h-3.5 w-3.5" />
            </Button>
          )}
          {isMobile && (
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setMobileOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
          {allNav.map((item) => (
            <SidebarNavItem
              key={item.href}
              item={item}
              collapsed={collapsed}
              isMobile={isMobile}
              setMobileOpen={setMobileOpen}
            />
          ))}
        </nav>

        {/* Bottom nav */}
        <div className="border-t py-2 px-2 space-y-0.5">
          {bottomNav.map((item) => (
            <SidebarNavItem
              key={item.href}
              item={item}
              collapsed={collapsed}
              isMobile={isMobile}
              setMobileOpen={setMobileOpen}
            />
          ))}
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <header className="h-11 border-b flex items-center px-4 gap-3 shrink-0">
          <Button variant="ghost" size="icon" className="lg:hidden h-7 w-7" onClick={() => setMobileOpen(true)}>
            <Menu className="h-4 w-4" />
          </Button>
          {collapsed && !isMobile && (
            <Button variant="ghost" size="icon" className="h-7 w-7 hidden lg:flex" onClick={() => setCollapsed(false)}>
              <ChevronsRight className="h-3.5 w-3.5" />
            </Button>
          )}
          <div className="flex-1" />
          <ThemeToggle />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                aria-label="User menu"
                className="rounded-md h-7 w-7"
              >
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel className="font-normal text-xs text-muted-foreground">{user?.email}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => router.push("/settings")} className="text-[13px]">Settings</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleSignOut} className="text-[13px]">Sign out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </header>

        <main className="flex-1 overflow-auto">
          {children}
        </main>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Update dashboard layout loading spinner**

In `Frontend/app/(dashboard)/layout.tsx`, replace the `animate-pulse-glow` loading spinner (line 59) with:

```tsx
<div className="h-8 w-8 rounded-md border-2 border-primary border-t-transparent animate-spin" />
```

- [ ] **Step 3: Verify sidebar renders correctly**

Run: `cd Frontend && npm run dev`
Navigate to `/dashboard` and verify: sidebar at 240px, 44px header, clean nav items with left-edge indicator, no glow.

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/dashboard/app-shell.tsx Frontend/app/\(dashboard\)/layout.tsx
git commit -m "feat: rebuild AppShell with Linear-style sidebar and minimal header"
```

---

## Task 4: Rewrite Student Dashboard Page

**Files:**
- Modify: `Frontend/app/(dashboard)/dashboard/page.tsx`

- [ ] **Step 1: Rewrite the dashboard page with clean Linear-style design**

Replace the entire file. Key changes: real stats from API, clean metric cards, no emoji, no glow, tighter spacing.

```tsx
"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import {
  BookOpen,
  MessageSquare,
  Target,
  TrendingUp,
  FileText,
  Sparkles,
  ArrowRight,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useDashboard } from "@/lib/context/dashboard-context"
import { CollectionsPanel } from "@/components/dashboard/collections-panel"
import { ExamSetupDialog } from "@/components/dashboard/exam-setup-dialog"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/context/auth-context"
import { mockTestAPI, pdfAPI, chatAPI, analyticsAPI } from "@/lib/api"
import { cn } from "@/lib/utils"
import Link from "next/link"

interface DashboardStats {
  documents: number
  chatSessions: number
  testsTaken: number
  avgScore: number | null
}

export default function DashboardPage() {
  return <DashboardContent />
}

function DashboardContent() {
  const router = useRouter()
  const { toast } = useToast()
  const { user } = useAuth()
  const { activeExam, refreshExams } = useDashboard()

  const [dialogOpen, setDialogOpen] = useState(false)
  const [panelOpen, setPanelOpen] = useState(false)
  const [selectedSubject, setSelectedSubject] = useState<{ id: string; name: string } | null>(null)
  const [assignedTests, setAssignedTests] = useState<Array<{ test_id: string; title: string; created_at: string; created_by?: string }>>([])
  const [stats, setStats] = useState<DashboardStats>({ documents: 0, chatSessions: 0, testsTaken: 0, avgScore: null })
  const [isLoadingStats, setIsLoadingStats] = useState(true)

  useEffect(() => { refreshExams() }, [refreshExams])

  useEffect(() => {
    const fetchData = async () => {
      setIsLoadingStats(true)
      try {
        const [pdfs, sessions, tests] = await Promise.allSettled([
          pdfAPI.listPDFs(),
          chatAPI.listChatSessions(),
          mockTestAPI.listMockTests(),
        ])
        const docCount = pdfs.status === "fulfilled" ? (pdfs.value as any[])?.length || 0 : 0
        const sessionCount = sessions.status === "fulfilled" ? (sessions.value as any[])?.length || 0 : 0
        const testList = tests.status === "fulfilled" ? (tests.value as any[]) || [] : []

        const mine = testList.filter((t: any) => t.assigned_to === user?.email)
        setAssignedTests(mine)

        const submitted = testList.filter((t: any) => t.latest_submission)
        const avgScore = submitted.length > 0
          ? submitted.reduce((sum: number, t: any) => sum + (t.latest_submission?.percentage || 0), 0) / submitted.length
          : null

        setStats({ documents: docCount, chatSessions: sessionCount, testsTaken: testList.length, avgScore })
      } catch (err) {
        console.error("Failed to load stats:", err)
      } finally {
        setIsLoadingStats(false)
      }
    }
    fetchData()
  }, [user?.email])

  const handleSubjectClick = (subject: { id: string; name: string }) => {
    setSelectedSubject(subject)
    setPanelOpen(true)
  }

  const handleExamCreated = () => {
    refreshExams()
    toast({ title: "Exam created.", description: "Your study goal has been set up." })
  }

  const greeting = () => {
    const hour = new Date().getHours()
    if (hour < 12) return "Good morning"
    if (hour < 18) return "Good afternoon"
    return "Good evening"
  }

  const statCards = [
    { label: "Documents", value: stats.documents, icon: FileText },
    { label: "Chat Sessions", value: stats.chatSessions, icon: MessageSquare },
    { label: "Tests Taken", value: stats.testsTaken, icon: Target },
    { label: "Avg Score", value: stats.avgScore !== null ? `${Math.round(stats.avgScore)}%` : "—", icon: TrendingUp },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      {/* Welcome */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {greeting()}, {user?.name || "Student"}.
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Here&apos;s what&apos;s happening with your studies.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]" onClick={() => router.push("/chat")}>
            <MessageSquare className="h-3.5 w-3.5 mr-1.5" />
            Chat
          </Button>
          <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={() => router.push("/test?tab=mock")}>
            <Target className="h-3.5 w-3.5 mr-1.5" />
            Take Test
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((stat) => (
          <div key={stat.label} className="rounded-md border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <div className="text-xl font-semibold tabular-nums">
              {isLoadingStats ? (
                <div className="h-6 w-12 bg-muted animate-pulse rounded" />
              ) : (
                stat.value
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Subjects */}
      {activeExam?.subjects && activeExam.subjects.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Subjects</h2>
            <Button variant="outline" size="sm" className="rounded-md h-7 text-xs" onClick={() => setDialogOpen(true)}>
              Add Exam
            </Button>
          </div>
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {activeExam.subjects.map((subject) => (
              <button
                key={subject.id}
                onClick={() => handleSubjectClick({ id: subject.id, name: subject.name })}
                className="rounded-md border bg-card p-4 text-left hover:bg-secondary/50 transition-colors duration-150 group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{subject.name}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1 h-1 rounded-full bg-muted">
                    <div
                      className="h-1 rounded-full bg-primary transition-all"
                      style={{ width: `${Math.min(subject.progress, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground tabular-nums">{subject.progress}%</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1.5">
                  {subject.collections.length} collection{subject.collections.length !== 1 ? "s" : ""}
                </p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Assigned Tests */}
      {assignedTests.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Assigned Tests</h2>
          <div className="rounded-md border divide-y">
            {assignedTests.map((test) => (
              <div key={test.test_id} className="flex items-center justify-between px-4 py-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{test.title}</p>
                  <p className="text-xs text-muted-foreground">From {test.created_by || "Teacher"}</p>
                </div>
                <Button size="sm" className="rounded-md h-7 text-xs shrink-0 ml-4" onClick={() => router.push(`/test/quiz?testId=${test.test_id}`)}>
                  Start
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quick Actions</h2>
        <div className="grid gap-2 md:grid-cols-3">
          <Link href="/flashcards" className="rounded-md border bg-card p-4 hover:bg-secondary/50 transition-colors duration-150 group">
            <Sparkles className="h-4 w-4 text-primary mb-2" />
            <p className="text-sm font-medium">Flashcards</p>
            <p className="text-xs text-muted-foreground mt-0.5">Review with AI-generated cards.</p>
          </Link>
          <Link href="/analytics" className="rounded-md border bg-card p-4 hover:bg-secondary/50 transition-colors duration-150 group">
            <TrendingUp className="h-4 w-4 text-primary mb-2" />
            <p className="text-sm font-medium">Analytics</p>
            <p className="text-xs text-muted-foreground mt-0.5">Track your performance trends.</p>
          </Link>
          <Link href="/chat" className="rounded-md border bg-card p-4 hover:bg-secondary/50 transition-colors duration-150 group">
            <MessageSquare className="h-4 w-4 text-primary mb-2" />
            <p className="text-sm font-medium">Chat</p>
            <p className="text-xs text-muted-foreground mt-0.5">Ask questions about your materials.</p>
          </Link>
        </div>
      </div>

      <CollectionsPanel exam={activeExam} open={panelOpen} onOpenChange={setPanelOpen} onChat={(examId) => router.push(`/chat?exam=${examId}`)} />
      <ExamSetupDialog open={dialogOpen} onOpenChange={setDialogOpen} onExamCreated={handleExamCreated} />
    </div>
  )
}
```

- [ ] **Step 2: Verify the dashboard renders**

Run: `cd Frontend && npm run dev`
Navigate to `/dashboard`. Check: clean welcome bar, real stats (may show 0s), subject cards, quick actions.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/\(dashboard\)/dashboard/page.tsx
git commit -m "feat: rewrite student dashboard with Linear-style design and real stats"
```

---

## Task 5: Rewrite Teacher Dashboard Page

**Files:**
- Modify: `Frontend/app/(dashboard)/teacher/page.tsx`

- [ ] **Step 1: Rewrite teacher dashboard with clean data-table style**

Replace the entire file with Linear-inspired teacher dashboard: no nested header, clean metric cards, data-table student list, student detail drawer.

```tsx
"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { BookOpen, Users, TrendingUp, Target, Activity, ChevronRight, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { teacherAPI } from "@/lib/api"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { cn } from "@/lib/utils"

interface StudentAnalytics {
  email: string
  name?: string
  tests_taken: number
  average_score: number
  last_active_at?: string
  strengths: string[]
  weaknesses: string[]
}

interface TeacherAnalytics {
  total_students: number
  active_students: number
  total_tests_taken: number
  class_average: number
  student_analytics: StudentAnalytics[]
}

interface ManagedStudent {
  email: string
  name?: string
}

function TeacherDashboardContent() {
  const router = useRouter()
  const { toast } = useToast()
  const [students, setStudents] = useState<ManagedStudent[]>([])
  const [analytics, setAnalytics] = useState<TeacherAnalytics | null>(null)
  const [selectedStudentEmail, setSelectedStudentEmail] = useState<string | null>(null)
  const [selectedStudent, setSelectedStudent] = useState<StudentAnalytics | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [studentsData, analyticsData] = await Promise.all([
          teacherAPI.listManagedStudents(),
          teacherAPI.getAnalytics(),
        ])
        setStudents(studentsData || [])
        setAnalytics(analyticsData)
      } catch (error: any) {
        console.error("Error fetching teacher data:", error)
        toast({
          title: "Error",
          description: error.response?.data?.detail || "Failed to load teacher dashboard data.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [toast])

  const handleCreateTest = () => {
    router.push(`/test?tab=mock&student=${encodeURIComponent(selectedStudentEmail ?? "")}`)
  }

  const handleStudentClick = (student: StudentAnalytics) => {
    setSelectedStudent(student)
  }

  const statCards = [
    { label: "Total Students", value: analytics?.total_students ?? 0, icon: Users },
    { label: "Active Students", value: analytics?.active_students ?? 0, icon: Activity },
    { label: "Tests Taken", value: analytics?.total_tests_taken ?? 0, icon: Target },
    { label: "Class Average", value: analytics ? `${analytics.class_average}%` : "—", icon: TrendingUp },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Teacher Dashboard.</h1>
          <p className="text-sm text-muted-foreground mt-1">Monitor student performance and create targeted tests.</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            className="rounded-md border bg-background px-2 py-1.5 text-[13px] h-8"
            value={selectedStudentEmail ?? ""}
            onChange={(e) => setSelectedStudentEmail(e.target.value || null)}
          >
            <option value="">Assign to me</option>
            {students.map((s) => (
              <option key={s.email} value={s.email}>{s.name || s.email}</option>
            ))}
          </select>
          <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={handleCreateTest}>
            <BookOpen className="h-3.5 w-3.5 mr-1.5" />
            Create Test
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((stat) => (
          <div key={stat.label} className="rounded-md border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <div className="text-xl font-semibold tabular-nums">
              {isLoading ? <div className="h-6 w-12 bg-muted animate-pulse rounded" /> : stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Student List */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Students</h2>
        <div className="rounded-md border">
          {/* Table header */}
          <div className="grid grid-cols-[1fr_100px_100px_1fr] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
            <span>Name</span>
            <span className="text-right">Tests</span>
            <span className="text-right">Avg</span>
            <span>Weaknesses</span>
          </div>
          {/* Table rows */}
          {isLoading ? (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">Loading...</div>
          ) : analytics?.student_analytics && analytics.student_analytics.length > 0 ? (
            analytics.student_analytics.map((student) => (
              <button
                key={student.email}
                onClick={() => handleStudentClick(student)}
                className="grid grid-cols-[1fr_100px_100px_1fr] gap-4 px-4 py-3 w-full text-left border-b last:border-b-0 hover:bg-secondary/50 transition-colors duration-150 group"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{student.name || student.email}</p>
                  <p className="text-xs text-muted-foreground truncate font-mono">{student.email}</p>
                </div>
                <span className="text-sm tabular-nums text-right self-center">{student.tests_taken}</span>
                <span className="text-sm tabular-nums text-right self-center">{student.average_score}%</span>
                <div className="flex flex-wrap gap-1 self-center">
                  {student.weaknesses?.slice(0, 3).map((topic) => (
                    <Badge key={topic} variant="secondary" className="text-[10px] font-normal px-1.5 py-0 bg-red-500/10 text-red-400 border-red-500/20">
                      {topic}
                    </Badge>
                  ))}
                  {student.weaknesses?.length > 3 && (
                    <span className="text-[10px] text-muted-foreground">+{student.weaknesses.length - 3}</span>
                  )}
                </div>
              </button>
            ))
          ) : (
            <div className="px-4 py-12 text-center text-sm text-muted-foreground">
              No students linked yet. Manage students to see them here.
            </div>
          )}
        </div>
      </div>

      {/* Student Detail Drawer */}
      {selectedStudent && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div className="absolute inset-0 bg-black/50" onClick={() => setSelectedStudent(null)} />
          <div className="relative w-96 bg-background border-l h-full overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">{selectedStudent.name || selectedStudent.email}</h3>
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setSelectedStudent(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground font-mono">{selectedStudent.email}</p>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Tests Taken</p>
                <p className="text-xl font-semibold tabular-nums">{selectedStudent.tests_taken}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Average Score</p>
                <p className="text-xl font-semibold tabular-nums">{selectedStudent.average_score}%</p>
              </div>
            </div>

            {selectedStudent.strengths && selectedStudent.strengths.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Strengths</p>
                <div className="flex flex-wrap gap-1">
                  {selectedStudent.strengths.map((s) => (
                    <Badge key={s} variant="secondary" className="text-xs font-normal bg-green-500/10 text-green-400 border-green-500/20">{s}</Badge>
                  ))}
                </div>
              </div>
            )}

            {selectedStudent.weaknesses && selectedStudent.weaknesses.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Weaknesses</p>
                <div className="flex flex-wrap gap-1">
                  {selectedStudent.weaknesses.map((w) => (
                    <Badge key={w} variant="secondary" className="text-xs font-normal bg-red-500/10 text-red-400 border-red-500/20">{w}</Badge>
                  ))}
                </div>
              </div>
            )}

            <Separator />

            <Button
              className="w-full rounded-md"
              onClick={() => {
                setSelectedStudent(null)
                router.push(`/test?tab=mock&student=${encodeURIComponent(selectedStudent.email)}`)
              }}
            >
              Create Targeted Test
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export default function TeacherPage() {
  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <TeacherDashboardContent />
    </RoleGuard>
  )
}
```

- [ ] **Step 2: Verify teacher dashboard**

Navigate to `/teacher`. Check: clean header, stat cards, student data table, drawer on click.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/\(dashboard\)/teacher/page.tsx
git commit -m "feat: rewrite teacher dashboard with data-table and detail drawer"
```

---

## Task 6: Rewrite Chat Page with Streaming

**Files:**
- Modify: `Frontend/app/(dashboard)/chat/page.tsx`
- Modify: `Frontend/components/dashboard/chat/chat-interface.tsx`
- Modify: `Frontend/components/dashboard/chat/chat-input.tsx`
- Modify: `Frontend/components/dashboard/chat/message-item.tsx`
- Modify: `Frontend/components/dashboard/chat/collections-chat-sidebar.tsx`

- [ ] **Step 1: Rewrite chat page with clean Linear layout**

Replace `Frontend/app/(dashboard)/chat/page.tsx`:

```tsx
"use client"

import { useState } from "react"
import { useDashboard } from "@/lib/context/dashboard-context"
import { CollectionsChatSidebar } from "@/components/dashboard/chat/collections-chat-sidebar"
import { ChatInterface } from "@/components/dashboard/chat/chat-interface"
import { Material, Collection, Subject } from "@/lib/context/dashboard-context"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { PanelLeftClose, PanelLeftOpen, MessageSquare } from "lucide-react"

export default function ChatPage() {
  const { activeExam } = useDashboard()
  const [selectedMaterial, setSelectedMaterial] = useState<Material | null>(null)
  const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const handleSelectMaterial = (material: Material, collection: Collection, _subject: Subject) => {
    setSelectedMaterial(material)
    setSelectedCollection(collection)
  }

  return (
    <div className="h-full flex overflow-hidden">
      {/* Sidebar */}
      <div className={cn(
        "flex flex-col border-r bg-background transition-all duration-200 z-20",
        sidebarOpen ? "w-60" : "w-0 overflow-hidden border-r-0"
      )}>
        {sidebarOpen && (
          <CollectionsChatSidebar
            exam={activeExam}
            selectedMaterial={selectedMaterial}
            onSelectMaterial={handleSelectMaterial}
          />
        )}
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Top bar */}
        <div className="h-11 border-b flex items-center px-3 gap-2 shrink-0">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <PanelLeftClose className="h-3.5 w-3.5" /> : <PanelLeftOpen className="h-3.5 w-3.5" />}
          </Button>
          <div className="flex items-center gap-1.5 text-[13px]">
            <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-muted-foreground">
              {selectedMaterial ? selectedMaterial.name : "Select a material to chat"}
            </span>
          </div>
        </div>

        {/* Chat */}
        <div className="flex-1 overflow-hidden">
          {selectedMaterial ? (
            <ChatInterface
              document={{
                id: selectedMaterial.id,
                title: selectedMaterial.name,
                file_path: selectedMaterial.url,
                filename: selectedMaterial.name,
                size: selectedMaterial.size,
                processed: true,
                user_id: "",
                uploadedAt: selectedMaterial.uploadedAt,
                tags: [],
                page_count: 0,
                description: undefined,
                vector_db_path: undefined,
              }}
              className="h-full"
            />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-6">
              <div className="w-12 h-12 rounded-md bg-secondary flex items-center justify-center mb-4">
                <MessageSquare className="h-6 w-6 text-muted-foreground" />
              </div>
              <h3 className="text-sm font-medium mb-1">Select a Study Material</h3>
              <p className="text-xs text-muted-foreground max-w-xs">
                Choose a material from the sidebar to start chatting with AI.
              </p>
              {!sidebarOpen && (
                <Button size="sm" className="mt-4 rounded-md h-8 text-[13px]" onClick={() => setSidebarOpen(true)}>
                  Open Sidebar
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Update chat-interface.tsx for streaming**

In `Frontend/components/dashboard/chat/chat-interface.tsx`, find the `handleSend` function (or equivalent message sending logic) and replace the `chatAPI.addMessageToChat` call with `chatAPI.addMessageToChatStream` using the existing streaming infrastructure. Add state for `streamingMessage` and progressively append chunks. Key pattern:

```tsx
// In handleSend, replace:
// const data = await chatAPI.addMessageToChat(sessionId, content)
// With:
setIsStreaming(true)
try {
  await chatAPI.addMessageToChatStream(sessionId, content, (chunk: any) => {
    if (chunk.content) {
      setStreamingMessage(prev => (prev || "") + chunk.content)
    }
  })
} finally {
  setIsStreaming(false)
}
// After stream completes, refresh messages and clear streamingMessage
```

The exact implementation depends on the current `chat-interface.tsx` structure. The key changes:
1. Add `const [streamingMessage, setStreamingMessage] = useState<string | null>(null)` state
2. Add `const [isStreaming, setIsStreaming] = useState(false)` state
3. Replace the non-streaming `addMessageToChat` call with `addMessageToChatStream` + progressive rendering
4. Display `streamingMessage` as a separate AI message bubble while streaming
5. On stream complete, refresh the session's message list and clear `streamingMessage`

- [ ] **Step 3: Update message-item.tsx with clean Linear-style bubbles**

In `Frontend/components/dashboard/chat/message-item.tsx`, remove any glow/shadow effects from message bubbles. Use clean styling: AI messages get a subtle left border or `bg-secondary` background, user messages stay default. No neomorphic styling.

- [ ] **Step 4: Update chat-input.tsx with clean textarea**

In `Frontend/components/dashboard/chat/chat-input.tsx`, replace neomorphic input styling with clean 6px-radius textarea with hairline border. Replace mock file attachment logic with a real file input. Remove `className="neo-input"` or similar.

- [ ] **Step 5: Update collections-chat-sidebar.tsx with clean tree view**

In `Frontend/components/dashboard/chat/collections-chat-sidebar.tsx`, replace `bg-card/50 backdrop-blur-xl` with `bg-background`. Clean up the tree structure with tighter spacing.

- [ ] **Step 6: Commit**

```bash
git add Frontend/app/\(dashboard\)/chat/page.tsx Frontend/components/dashboard/chat/
git commit -m "feat: rewrite chat with Linear-style layout and streaming support"
```

---

## Task 7: Rewrite Test Page (Analysis + Mock Tests)

**Files:**
- Modify: `Frontend/app/(dashboard)/test/page.tsx`

- [ ] **Step 1: Rewrite test page with clean Linear-style forms**

Key changes throughout the file:
1. Replace all `Card` usage with clean `rounded-md border` divs
2. Replace `rounded-xl`/`rounded-2xl` with `rounded-md`
3. Replace `bg-muted/30 rounded-lg` summary boxes with `rounded-md border` + mono labels
4. Replace `<select>` with consistent styled select matching the design
5. Tab bar: replace grid-style `TabsList` with clean pill tabs
6. All buttons: `rounded-md` instead of default rounded
7. Form inputs: add `rounded-md h-9 text-[13px]` classes
8. Remove all `neo-card` class references
9. Replace `space-y-6` card spacing with `space-y-4` for tighter layout
10. Mock test list: replace card grid with data table format

The file is very long (1110 lines). Apply these changes systematically. The most impactful pattern replacements:

- `className="container mx-auto py-8 px-4"` → `className="max-w-5xl mx-auto py-8 px-6"`
- `<Card>` → `<div className="rounded-md border">` (with equivalent header/content structure)
- `rounded-xl` → `rounded-md`
- `rounded-lg` → `rounded-md`
- `rounded-full` (on test list items) → `rounded-md`
- `bg-muted/30 rounded-lg` → `rounded-md border bg-secondary/50`
- Native `<select>` → add `rounded-md border bg-background px-3 py-1.5 text-[13px] h-9`
- Button `size="lg"` → `size="sm"` with `rounded-md h-9 text-[13px]`

- [ ] **Step 2: Verify test page renders**

Navigate to `/test`. Check: clean tab bar, forms, test list.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/\(dashboard\)/test/page.tsx
git commit -m "feat: rewrite test page with Linear-style forms and data table"
```

---

## Task 8: Rewrite Quiz and Results Pages

**Files:**
- Modify: `Frontend/app/(dashboard)/test/quiz/page.tsx`
- Modify: `Frontend/app/(dashboard)/test/results/page.tsx`

- [ ] **Step 1: Fix hardcoded colors in quiz page**

In quiz page, replace any `bg-[#333]`, `bg-[#222]`, `bg-blue-800` hardcoded dark colors with theme-aware tokens (`bg-secondary`, `bg-card`, `bg-primary`). Apply `rounded-md` instead of `rounded-xl`.

- [ ] **Step 2: Fix hardcoded colors in results page**

In results page, same treatment — replace all hardcoded dark colors with theme tokens. Replace `bg-[#333]`, `bg-[#222]` etc. with `bg-card`, `bg-secondary`. Apply `rounded-md`.

- [ ] **Step 3: Commit**

```bash
git add Frontend/app/\(dashboard\)/test/quiz/page.tsx Frontend/app/\(dashboard\)/test/results/page.tsx
git commit -m "fix: replace hardcoded colors with theme tokens in quiz and results"
```

---

## Task 9: Rewrite Settings Page

**Files:**
- Modify: `Frontend/app/(dashboard)/settings/page.tsx`

- [ ] **Step 1: Rewrite settings page with clean Linear style and API wiring**

Key changes:
1. Remove all `className="neo-card"` from Card components
2. Wire profile fields to `useAuth()` user data
3. Wire save to `authAPI` (update profile if backend supports, otherwise just persist locally)
4. Replace accent color picker "this is a demo" with functional CSS variable setter
5. All cards: `rounded-md border`
6. All inputs: `rounded-md h-9 text-[13px]`
7. All buttons: `rounded-md`

Replace the page component. Key wiring:

```tsx
const { user, refreshUser } = useAuth()

// Pre-fill profile fields from auth context
const [fullName, setFullName] = useState(user?.name || "")
const [email, setEmail] = useState(user?.email || "")

// Accent color picker sets CSS variable
const handleAccentColor = (color: string) => {
  document.documentElement.style.setProperty('--primary', color)
  localStorage.setItem('orbit:accent-color', color)
}

// Save changes - update profile via API if available
const handleSave = async () => {
  try {
    // If backend has profile update endpoint, call it
    // Otherwise just show success
    toast({ title: "Settings saved.", description: "Your preferences have been updated." })
  } catch {
    toast({ title: "Error", description: "Failed to save settings.", variant: "destructive" })
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/app/\(dashboard\)/settings/page.tsx
git commit -m "feat: rewrite settings page with clean design and profile wiring"
```

---

## Task 10: Rewrite Onboarding Page

**Files:**
- Modify: `Frontend/app/onboarding/page.tsx`
- Modify: `Frontend/components/onboarding/onboarding-container.tsx`
- Modify: `Frontend/components/onboarding/step-about-you.tsx`
- Modify: `Frontend/components/onboarding/step-study-goal.tsx`

- [ ] **Step 1: Clean up onboarding components**

Remove MagicCard wrapping, replace with clean `rounded-md border bg-card` container. Apply form-input specs (40px height, 6px radius, hairline border). Clean step indicators. Tighter spacing.

Key changes per file:
- `onboarding-container.tsx`: Replace `<MagicCard>` with `<div className="rounded-md border bg-card p-8 max-w-md mx-auto">`
- `step-about-you.tsx`: Input fields → `rounded-md h-9 text-[13px]` matching DESIGN.md form-input
- `step-study-goal.tsx`: Preset exam buttons → `rounded-md border` cards instead of MagicCards

- [ ] **Step 2: Commit**

```bash
git add Frontend/app/onboarding/ Frontend/components/onboarding/
git commit -m "feat: rewrite onboarding with clean Linear-style forms"
```

---

## Task 11: Create Flashcard Feature Pages

**Files:**
- Create: `Frontend/app/(dashboard)/flashcards/page.tsx`
- Create: `Frontend/components/dashboard/flashcard/flashcard-deck.tsx`
- Create: `Frontend/components/dashboard/flashcard/flashcard-card.tsx`

- [ ] **Step 1: Create flashcard card component**

Create `Frontend/components/dashboard/flashcard/flashcard-card.tsx`:

```tsx
"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"

interface FlashcardCardProps {
  front: string
  back: string
  className?: string
}

export function FlashcardCard({ front, back, className }: FlashcardCardProps) {
  const [isFlipped, setIsFlipped] = useState(false)

  return (
    <button
      onClick={() => setIsFlipped(!isFlipped)}
      className={cn(
        "w-full h-64 perspective-1000 cursor-pointer",
        className
      )}
    >
      <div className={cn(
        "relative w-full h-full transition-transform duration-500 transform-style-3d",
        isFlipped && "rotate-y-180"
      )}>
        {/* Front */}
        <div className="absolute inset-0 rounded-md border bg-card p-6 flex flex-col items-center justify-center backface-hidden">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Question</p>
          <p className="text-sm font-medium text-center">{front}</p>
        </div>
        {/* Back */}
        <div className="absolute inset-0 rounded-md border bg-primary text-primary-foreground p-6 flex flex-col items-center justify-center backface-hidden rotate-y-180">
          <p className="text-xs uppercase tracking-wider mb-2 opacity-70">Answer</p>
          <p className="text-sm font-medium text-center">{back}</p>
        </div>
      </div>
    </button>
  )
}
```

- [ ] **Step 2: Create flashcard deck browser**

Create `Frontend/components/dashboard/flashcard/flashcard-deck.tsx`:

```tsx
"use client"

import { useState, useCallback } from "react"
import { FlashcardCard } from "./flashcard-card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-react"

interface FlashcardDeckProps {
  cards: Array<{ front: string; back: string }>
  title?: string
}

export function FlashcardDeck({ cards, title }: FlashcardDeckProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [known, setKnown] = useState<Set<number>>(new Set())

  const handleNext = useCallback(() => {
    setCurrentIndex((i) => Math.min(i + 1, cards.length - 1))
  }, [cards.length])

  const handlePrev = useCallback(() => {
    setCurrentIndex((i) => Math.max(i - 1, 0))
  }, [])

  const handleMarkKnown = useCallback(() => {
    setKnown((prev) => new Set(prev).add(currentIndex))
    handleNext()
  }, [currentIndex, handleNext])

  if (cards.length === 0) {
    return (
      <div className="rounded-md border bg-card p-12 text-center">
        <p className="text-sm text-muted-foreground">No flashcards in this deck.</p>
      </div>
    )
  }

  const currentCard = cards[currentIndex]

  return (
    <div className="space-y-4">
      {title && (
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">{title}</h3>
          <span className="text-xs text-muted-foreground font-mono">
            {currentIndex + 1} / {cards.length}
          </span>
        </div>
      )}

      {/* Progress */}
      <div className="flex gap-1">
        {cards.map((_, i) => (
          <div
            key={i}
            className={cn(
              "h-1 flex-1 rounded-full transition-colors",
              i < currentIndex ? "bg-primary" : known.has(i) ? "bg-green-500/50" : "bg-muted"
            )}
          />
        ))}
      </div>

      <FlashcardCard front={currentCard.front} back={currentCard.back} />

      {/* Controls */}
      <div className="flex items-center justify-center gap-2">
        <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]" onClick={handlePrev} disabled={currentIndex === 0}>
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Previous
        </Button>
        <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]" onClick={handleMarkKnown}>
          Know
        </Button>
        <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={handleNext} disabled={currentIndex === cards.length - 1}>
          Next
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>

      {/* Stats */}
      <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
        <span>{known.size} known</span>
        <span>{cards.length - known.size} remaining</span>
      </div>
    </div>
  )
}
```

Note: Add CSS for 3D flip in globals.css if not already present:

```css
.perspective-1000 { perspective: 1000px; }
.transform-style-3d { transform-style: preserve-3d; }
.backface-hidden { backface-visibility: hidden; }
.rotate-y-180 { transform: rotateY(180deg); }
```

- [ ] **Step 3: Create flashcards page**

Create `Frontend/app/(dashboard)/flashcards/page.tsx`:

```tsx
"use client"

import { useState } from "react"
import { FlashcardDeck } from "@/components/dashboard/flashcard/flashcard-deck"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sparkles, Plus } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { pdfAPI } from "@/lib/api"

// Demo flashcard data (replace with AI generation later)
const DEMO_DECKS = [
  {
    id: "demo-1",
    title: "Calculus Basics",
    cards: [
      { front: "What is the derivative of x^n?", back: "nx^(n-1) — the Power Rule." },
      { front: "What is the chain rule?", back: "d/dx[f(g(x))] = f'(g(x)) * g'(x)." },
      { front: "What is the integral of 1/x?", back: "ln|x| + C." },
      { front: "What is L'Hopital's Rule?", back: "If lim f(x)/g(x) is 0/0 or inf/inf, then lim = lim f'(x)/g'(x)." },
    ],
  },
  {
    id: "demo-2",
    title: "Physics — Mechanics",
    cards: [
      { front: "Newton's Second Law?", back: "F = ma. Force equals mass times acceleration." },
      { front: "What is kinetic energy?", back: "KE = 1/2 mv^2." },
      { front: "What is the unit of force?", back: "Newton (N) = kg * m/s^2." },
    ],
  },
]

export default function FlashcardsPage() {
  const { toast } = useToast()
  const [selectedDeck, setSelectedDeck] = useState(DEMO_DECKS[0])
  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerateDeck = async () => {
    setIsGenerating(true)
    try {
      toast({ title: "Coming soon.", description: "AI-generated flashcards from your materials will be available in a future update." })
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Flashcards.</h1>
          <p className="text-sm text-muted-foreground mt-1">Review key concepts with spaced repetition.</p>
        </div>
        <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={handleGenerateDeck} disabled={isGenerating}>
          <Sparkles className="h-3.5 w-3.5 mr-1.5" />
          Generate Deck
        </Button>
      </div>

      {/* Deck selector */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Deck:</span>
        <Select value={selectedDeck.id} onValueChange={(id) => {
          const deck = DEMO_DECKS.find(d => d.id === id)
          if (deck) setSelectedDeck(deck)
        }}>
          <SelectTrigger className="w-48 rounded-md h-8 text-[13px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DEMO_DECKS.map((deck) => (
              <SelectItem key={deck.id} value={deck.id}>{deck.title}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Flashcard deck */}
      <FlashcardDeck cards={selectedDeck.cards} title={selectedDeck.title} />
    </div>
  )
}
```

- [ ] **Step 4: Add 3D flip CSS to globals.css**

Append to `Frontend/app/globals.css`:

```css
.perspective-1000 { perspective: 1000px; }
.transform-style-3d { transform-style: preserve-3d; }
.backface-hidden { backface-visibility: hidden; }
.rotate-y-180 { transform: rotateY(180deg); }
```

- [ ] **Step 5: Commit**

```bash
git add Frontend/app/\(dashboard\)/flashcards/ Frontend/components/dashboard/flashcard/ Frontend/app/globals.css
git commit -m "feat: add flashcard feature with card flip and deck browser"
```

---

## Task 12: Create Student Analytics Page

**Files:**
- Create: `Frontend/app/(dashboard)/analytics/page.tsx`
- Create: `Frontend/components/dashboard/analytics/subject-chart.tsx`
- Create: `Frontend/components/dashboard/analytics/weakness-radar.tsx`

- [ ] **Step 1: Create subject chart component**

Create `Frontend/components/dashboard/analytics/subject-chart.tsx`:

```tsx
"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface SubjectChartProps {
  data: Array<{ subject: string; score: number }>
}

export function SubjectChart({ data }: SubjectChartProps) {
  return (
    <div className="rounded-md border bg-card p-4">
      <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Subject Performance</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="subject" tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" />
          <YAxis tick={{ fontSize: 11 }} stroke="hsl(var(--muted-foreground))" domain={[0, 100]} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="score" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 2: Create weakness radar component**

Create `Frontend/components/dashboard/analytics/weakness-radar.tsx`:

```tsx
"use client"

import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from "recharts"

interface WeaknessRadarProps {
  data: Array<{ topic: string; score: number; fullMark: number }>
}

export function WeaknessRadar({ data }: WeaknessRadarProps) {
  return (
    <div className="rounded-md border bg-card p-4">
      <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Weakness Map</h3>
      <ResponsiveContainer width="100%" height={250}>
        <RadarChart data={data}>
          <PolarGrid stroke="hsl(var(--border))" />
          <PolarAngleAxis dataKey="topic" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
          <Radar name="Score" dataKey="score" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 3: Create analytics page**

Create `Frontend/app/(dashboard)/analytics/page.tsx`:

```tsx
"use client"

import { useState, useEffect } from "react"
import { SubjectChart } from "@/components/dashboard/analytics/subject-chart"
import { WeaknessRadar } from "@/components/dashboard/analytics/weakness-radar"
import { analyticsAPI, mockTestAPI } from "@/lib/api"
import { TrendingUp, Target, Clock } from "lucide-react"

export default function AnalyticsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [subjectData, setSubjectData] = useState<Array<{ subject: string; score: number }>>([])
  const [weaknessData, setWeaknessData] = useState<Array<{ topic: string; score: number; fullMark: number }>>([])
  const [totalTests, setTotalTests] = useState(0)
  const [avgScore, setAvgScore] = useState(0)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        // Try student analytics endpoint
        const analytics = await analyticsAPI.getStudentAnalytics().catch(() => null)

        if (analytics) {
          // Use real analytics if available
          setSubjectData(analytics.subject_scores || [])
          setWeaknessData((analytics.weakness_topics || []).map((t: any) => ({
            topic: t.name || t.topic,
            score: t.score || 0,
            fullMark: 100,
          })))
          setTotalTests(analytics.total_tests || 0)
          setAvgScore(analytics.average_score || 0)
        } else {
          // Fallback: derive from mock tests
          const tests = await mockTestAPI.listMockTests()
          const submitted = (tests || []).filter((t: any) => t.latest_submission)
          setTotalTests(submitted.length)
          if (submitted.length > 0) {
            const avg = submitted.reduce((s: number, t: any) => s + (t.latest_submission?.percentage || 0), 0) / submitted.length
            setAvgScore(Math.round(avg))
          }
          // Demo data for charts
          setSubjectData([
            { subject: "Math", score: 72 },
            { subject: "Physics", score: 65 },
            { subject: "Chemistry", score: 80 },
            { subject: "Biology", score: 58 },
          ])
          setWeaknessData([
            { topic: "Calculus", score: 45, fullMark: 100 },
            { topic: "Mechanics", score: 55, fullMark: 100 },
            { topic: "Organic Chem", score: 60, fullMark: 100 },
            { topic: "Thermodynamics", score: 40, fullMark: 100 },
            { topic: "Algebra", score: 70, fullMark: 100 },
          ])
        }
      } catch (err) {
        console.error("Failed to load analytics:", err)
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [])

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Analytics.</h1>
        <p className="text-sm text-muted-foreground mt-1">Track your performance and identify weak areas.</p>
      </div>

      {/* Overview stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-md border bg-card p-4 space-y-2">
          <span className="text-xs font-medium text-muted-foreground">Tests Taken</span>
          <div className="text-xl font-semibold tabular-nums">{isLoading ? "—" : totalTests}</div>
        </div>
        <div className="rounded-md border bg-card p-4 space-y-2">
          <span className="text-xs font-medium text-muted-foreground">Average Score</span>
          <div className="text-xl font-semibold tabular-nums">{isLoading ? "—" : `${avgScore}%`}</div>
        </div>
        <div className="rounded-md border bg-card p-4 space-y-2">
          <span className="text-xs font-medium text-muted-foreground">Study Streak</span>
          <div className="text-xl font-semibold tabular-nums">—</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-4">
        <SubjectChart data={subjectData} />
        <WeaknessRadar data={weaknessData} />
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add Frontend/app/\(dashboard\)/analytics/ Frontend/components/dashboard/analytics/
git commit -m "feat: add student analytics page with subject charts and weakness radar"
```

---

## Task 13: Create Admin Dashboard Page

**Files:**
- Create: `Frontend/app/(dashboard)/admin/page.tsx`

- [ ] **Step 1: Create admin dashboard page (frontend scaffold)****

```tsx
"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Users, Shield, Key, Activity } from "lucide-react"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { useAuth } from "@/lib/context/auth-context"

interface UserRecord {
  id: string
  email: string
  name?: string
  role: string
  status: string
  created_at: string
}

function AdminDashboardContent() {
  const { user } = useAuth()
  const [isLoading, setIsLoading] = useState(true)
  const [users, setUsers] = useState<UserRecord[]>([])

  useEffect(() => {
    // Admin user list API not yet available
    // This is a frontend scaffold
    setIsLoading(false)
    setUsers([])
  }, [])

  const statCards = [
    { label: "Total Users", value: users.length, icon: Users },
    { label: "Active", value: users.filter(u => u.status === "active").length, icon: Activity },
    { label: "Admins", value: users.filter(u => u.role === "admin").length, icon: Shield },
    { label: "Licenses", value: "—", icon: Key },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin Dashboard.</h1>
        <p className="text-sm text-muted-foreground mt-1">Manage users, roles, and licenses.</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((stat) => (
          <div key={stat.label} className="rounded-md border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <div className="text-xl font-semibold tabular-nums">
              {isLoading ? <div className="h-6 w-12 bg-muted animate-pulse rounded" /> : stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* User table */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Users</h2>
        <div className="rounded-md border">
          <div className="grid grid-cols-[1fr_80px_80px_100px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
            <span>Email</span>
            <span>Role</span>
            <span>Status</span>
            <span>Actions</span>
          </div>
          {users.length === 0 ? (
            <div className="px-4 py-12 text-center text-sm text-muted-foreground">
              User management requires backend admin endpoints. This page is a frontend scaffold.
            </div>
          ) : (
            users.map((u) => (
              <div key={u.id} className="grid grid-cols-[1fr_80px_80px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center">
                <div className="min-w-0">
                  <p className="text-sm truncate">{u.name || u.email}</p>
                  <p className="text-xs text-muted-foreground font-mono truncate">{u.email}</p>
                </div>
                <Badge variant="secondary" className="text-[10px] font-normal w-fit">{u.role}</Badge>
                <Badge variant="outline" className="text-[10px] font-normal w-fit">{u.status}</Badge>
                <Select>
                  <SelectTrigger className="h-7 text-[11px] rounded-md">
                    <SelectValue placeholder="Role" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="student">Student</SelectItem>
                    <SelectItem value="teacher">Teacher</SelectItem>
                    <SelectItem value="subadmin">Sub-admin</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default function AdminPage() {
  return (
    <RoleGuard allowedRoles={["admin"]}>
      <AdminDashboardContent />
    </RoleGuard>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/app/\(dashboard\)/admin/
git commit -m "feat: add admin dashboard page scaffold"
```

---

## Task 14: Update Remaining Components (Cleanup Pass)

**Files:**
- Modify: `Frontend/components/dashboard/active-study-card.tsx`
- Modify: `Frontend/components/dashboard/subject-card.tsx`
- Modify: `Frontend/components/dashboard/collections-panel.tsx`
- Modify: `Frontend/components/dashboard/exam-setup-dialog.tsx`
- Modify: `Frontend/components/dashboard/empty-state.tsx`
- Modify: `Frontend/components/dashboard/material-list.tsx`
- Modify: `Frontend/components/theme-toggle.tsx`

- [ ] **Step 1: Clean up remaining dashboard components**

For each component, apply these systematic replacements:
- `rounded-xl` / `rounded-2xl` → `rounded-md`
- `rounded-lg` → `rounded-md`
- `neo-card` / `neo-button` / `neo-input` / `neo-flat` / `neo-pressed` / `neo-raised` → remove or replace with `rounded-md border bg-card`
- `bg-glow` / `bg-glow-blue` / `bg-glow-green` / `border-glow` → remove
- `text-glow` / `text-glow-strong` → remove
- `shadow-[0_0_12px_...]` glow shadows → remove
- `backdrop-blur-xl` → remove or replace with `bg-background`
- `bg-card/50` → `bg-card` or `bg-background`
- `animate-float` / `animate-twinkle` / `animate-pulse-glow` / `animate-nebula` → remove
- `font-heading` / `font-subheading` → `font-sans`

- [ ] **Step 2: Clean up theme toggle**

In `Frontend/components/theme-toggle.tsx`, apply `rounded-md` to button.

- [ ] **Step 3: Verify full app renders without errors**

Run: `cd Frontend && npm run dev`
Navigate through all dashboard pages: `/dashboard`, `/teacher`, `/chat`, `/test`, `/flashcards`, `/analytics`, `/settings`, `/onboarding`

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/dashboard/ Frontend/components/theme-toggle.tsx
git commit -m "refactor: clean up remaining components for Linear-style design"
```

---

## Task 15: Final Verification Pass

**Files:**
- All modified files

- [ ] **Step 1: Run lint**

Run: `cd Frontend && npm run lint`
Fix any lint errors.

- [ ] **Step 2: Run build**

Run: `cd Frontend && npm run build`
Fix any build errors.

- [ ] **Step 3: Manual verification**

Start dev server and verify each page renders correctly:
1. `/dashboard` — clean welcome, real stats, subject cards, quick actions
2. `/teacher` — stat cards, student data table, drawer
3. `/chat` — clean sidebar, streaming
4. `/test` — clean tabs, forms, test list
5. `/flashcards` — flip cards, deck browser
6. `/analytics` — charts, radar
7. `/admin` — scaffold page
8. `/settings` — profile wiring, accent color
9. `/onboarding` — clean steps
10. Light mode toggle — verify all pages in light mode

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address lint and build issues from Linear-style overhaul"
```