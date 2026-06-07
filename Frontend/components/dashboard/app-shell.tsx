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
  ChevronLeft,
  ChevronRight,
  BookOpen,
  FileBarChart,
} from "lucide-react"
import { DashboardIcon } from "@/components/icons/dashboard-icon"
import { ChatIcon } from "@/components/icons/chat-icon"
import { MockTestsIcon } from "@/components/icons/mock-tests-icon"
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

const nav = [
  { href: "/dashboard", label: "Dashboard", icon: DashboardIcon },
  { href: "/chat", label: "Chat", icon: ChatIcon },
  { href: "/test?tab=analysis", label: "Exam Analysis", icon: FileBarChart },
  { href: "/test?tab=mock", label: "Mock Tests", icon: MockTestsIcon },
  { href: "/settings", label: "Settings", icon: Settings },
]

function NavItem({
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
  const isActive = pathname === item.href.split("?")[0]

  return (
    <Link
      href={item.href}
      onClick={() => isMobile && setMobileOpen(false)}
      className={cn(
        "flex items-center rounded-xl text-sm font-medium transition-all duration-200",
        collapsed ? "justify-center px-2 py-3" : "gap-3 px-3 py-2.5",
        isActive
          ? "bg-primary/10 text-primary shadow-[0_0_12px_rgba(56,189,248,0.15)]"
          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
      )}
      title={collapsed ? item.label : undefined}
    >
      <item.icon className={cn("shrink-0", collapsed ? "h-6 w-6" : "h-5 w-5")} />
      {!collapsed && <span className="truncate">{item.label}</span>}
    </Link>
  )
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

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

  const sidebarWidth = collapsed ? "w-16" : "w-64"

  return (
    <div className="h-screen flex flex-col lg:flex-row overflow-hidden bg-background">
      {/* Mobile overlay */}
      {isMobile && mobileOpen && (
        <div className="fixed inset-0 bg-black/50 z-40" onClick={() => setMobileOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed lg:relative inset-y-0 left-0 z-50 flex flex-col bg-card/50 backdrop-blur-xl border-r transition-all duration-300 ease-in-out",
          sidebarWidth,
          isMobile && !mobileOpen ? "-translate-x-full" : "translate-x-0"
        )}
      >
        {/* Header */}
        <div className={cn(
          "h-16 border-b flex items-center shrink-0",
          collapsed ? "justify-center px-2" : "justify-between px-4"
        )}>
          {collapsed ? (
            /* Collapsed: logo is the expand button, hover reveals chevron */
            <div className="group relative">
              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10 rounded-xl p-0 hover:bg-muted/60 relative z-10"
                onClick={() => setCollapsed(false)}
                title="Expand sidebar"
              >
                <div className="relative h-8 w-8">
                  <Image
                    src="/logo.png"
                    alt="Orbit"
                    fill
                    className="object-contain"
                    priority
                  />
                </div>
              </Button>
              {/* Hover reveal expand icon */}
              <div className="absolute -right-1 -top-1 h-4 w-4 rounded-full bg-primary text-primary-foreground flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity scale-75 group-hover:scale-100">
                <ChevronRight className="h-2.5 w-2.5" />
              </div>
            </div>
          ) : (
            /* Expanded: logo + text left, collapse button right */
            <>
              <Link href="/dashboard" className="flex items-center gap-2.5 overflow-hidden">
                <div className="relative h-8 w-8 shrink-0">
                  <Image
                    src="/logo.png"
                    alt="Orbit"
                    fill
                    className="object-contain"
                    priority
                  />
                </div>
                <span className="font-bold text-lg tracking-tight whitespace-nowrap">
                  Orbit
                </span>
              </Link>

              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 rounded-lg opacity-60 hover:opacity-100 transition-opacity"
                onClick={() => setCollapsed(true)}
                title="Collapse sidebar"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </>
          )}

          {/* Mobile close */}
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(false)}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          {nav.map((item) => (
            <NavItem
              key={item.href}
              item={item}
              collapsed={collapsed}
              isMobile={isMobile}
              setMobileOpen={setMobileOpen}
            />
          ))}
        </nav>

        {/* Footer */}
        <div className={cn(
          "border-t p-2 shrink-0",
          collapsed && "flex flex-col items-center"
        )}>
          <div className="px-2 py-1.5 text-xs text-muted-foreground/60">
            <BookOpen className="inline h-3 w-3 mr-1" />
            Study smarter
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <header className="h-16 border-b bg-card/50 backdrop-blur-sm flex items-center px-4 gap-3 shrink-0">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)}>
            <Menu className="h-5 w-5" />
          </Button>
          <div className="flex-1" />
          <ThemeToggle />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                aria-label="User menu"
                className="rounded-full h-8 w-8 bg-primary/10 text-primary"
              >
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => router.push("/dashboard")}>Profile</DropdownMenuItem>
              <DropdownMenuItem onClick={() => router.push("/settings")}>Settings</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleSignOut}>Sign out</DropdownMenuItem>
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
