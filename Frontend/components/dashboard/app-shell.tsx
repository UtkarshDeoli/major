"use client"

import React, { useState, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
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
  Crown,
  Building2,
  Shield,
  Focus,
  Calendar,
  Users,
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
import { UpgradeBanner } from "@/components/billing/upgrade-banner"

const studentNav = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/analysis", label: "Analysis", icon: FileBarChart },
  { href: "/mock-tests", label: "Mock Tests", icon: BookOpen },
  { href: "/flashcards", label: "Flashcards", icon: Sparkles },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/focus", label: "Focus", icon: Focus },
  { href: "/plans", label: "Plans", icon: Calendar },
]

const teacherNav = [
  { href: "/teacher", label: "Dashboard", icon: LayoutDashboard },
  { href: "/classes", label: "Classes", icon: Users },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/analysis", label: "Analysis", icon: FileBarChart },
  { href: "/mock-tests", label: "Mock Tests", icon: BookOpen },
  { href: "/flashcards", label: "Flashcards", icon: Sparkles },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
]

const bottomNav = [
  { href: "/settings", label: "Settings", icon: Settings },
]

// Role-specific top-level pages. Teachers see billing too for self-serve upgrades.
const billingNav = { href: "/billing", label: "Billing", icon: Crown }
const orgNav = { href: "/org", label: "Organization", icon: Building2 }
const adminNav = { href: "/admin", label: "Admin", icon: Shield }

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

type NavItem = { href: string; label: string; icon: React.ComponentType<{ className?: string }> }

function SidebarNavItem({
  item,
  collapsed,
  isMobile,
  setMobileOpen,
}: {
  item: NavItem
  collapsed: boolean
  isMobile: boolean
  setMobileOpen: (v: boolean) => void
}) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const tab = searchParams.get("tab")
  const isActive = isNavItemActive(item.href, pathname, tab)

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
  const { user, logout } = useAuth()
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
    logout()
  }

  const sidebarWidth = collapsed ? "w-12" : "w-60"

  // Build the nav list based on the current user's role.
  const allNav = (() => {
    const base = user?.role === "teacher" ? teacherNav : studentNav
    const list = [...base]
    list.push(billingNav)
    if (user?.role === "subadmin") {
      list.push(orgNav)
      list.push(adminNav)
    } else if (user?.role === "admin") {
      list.push(adminNav)
    }
    return list
  })()

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
              <Button variant="ghost" size="icon" aria-label="User menu" className="rounded-md h-7 w-7">
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
        <UpgradeBanner />
      </div>
    </div>
  )
}