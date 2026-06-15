"use client"

import { useEffect } from "react"
import { usePathname, useRouter } from "next/navigation"
import { Toaster } from "@/components/ui/toaster"
import AuthProtection from "@/components/auth/route-protection/auth-protection"
import { useAuth } from "@/lib/context/auth-context"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const pathname = usePathname()
  const { user, role, isLoading } = useAuth()

  useEffect(() => {
    if (isLoading) return

    // Role-aware default redirect: teachers land on their own dashboard
    if (role === "teacher" && pathname === "/dashboard") {
      router.replace("/teacher")
    }
  }, [isLoading, role, pathname, router])

  return (
    <AuthProtection>
      <div className="min-h-screen">
        {children}
        <Toaster />
      </div>
    </AuthProtection>
  )
}
