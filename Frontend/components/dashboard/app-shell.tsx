"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import {
  LayoutDashboard,
  MessageSquare,
  FileBarChart,
  Target,
  User,
  Menu,
  X,
  Book,
  Settings,
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

const nav = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/test?tab=analysis", label: "Exam Analysis", icon: FileBarChart },
  { href: "/test?tab=mock", label: "Mock Tests", icon: Target },
  { href: "/settings", label: "Settings", icon: Settings },
]

function NavItem({ item, isMobile, setMobileOpen }: { item: typeof nav[0]; isMobile: boolean; setMobileOpen: (v: boolean) => void }) {
  const pathname = usePathname()
  const isActive = pathname === item.href.split("?")[0]

  return (
    <Link
      href={item.href}
      onClick={() => isMobile && setMobileOpen(false)}
      className={cn(
        "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
        isActive
          ? "bg-primary/10 text-primary"
          : "text-muted-foreground hover:bg-muted hover:text-foreground"
      )}
    >
      <item.icon className="h-4 w-4" />
      {item.label}
    </Link>
  )
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 1024)
    check()
    window.addEventListener("resize", check)
    return () => window.removeEventListener("resize", check)
  }, [])

  const handleSignOut = () => {
    localStorage.removeItem("token")
    router.push("/")
  }

  return (
    <div className="h-screen flex flex-col lg:flex-row overflow-hidden bg-background">
      {/* Mobile overlay */}
      {isMobile && mobileOpen && (
        <div className="fixed inset-0 bg-black/50 z-40" onClick={() => setMobileOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "lg:relative fixed inset-y-0 left-0 z-50 flex flex-col bg-card border-r w-64 transition-transform duration-300",
          isMobile && !mobileOpen ? "-translate-x-full" : "translate-x-0"
        )}
      >
        <div className="h-14 border-b flex items-center justify-between px-4 shrink-0">
          <Link href="/dashboard" className="flex items-center gap-2">
            <Book className="h-5 w-5 text-primary" />
            <span className="font-bold text-lg tracking-tight">Padhai Whallah</span>
          </Link>
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(false)}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="pb-2">
            {nav.map((item) => (
              <NavItem key={item.href} item={item} isMobile={isMobile} setMobileOpen={setMobileOpen} />
            ))}
          </div>
        </nav>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="h-14 border-b bg-card flex items-center px-4 gap-3 shrink-0">
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
