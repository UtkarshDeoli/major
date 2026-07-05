"use client"

import { useEffect } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/lib/context/auth-context'

export default function AuthProtection({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const pathname = usePathname()
  const { user, isLoading } = useAuth()

  useEffect(() => {
    // Avoid redirect loops if we're already heading to the login page.
    if (!isLoading && !user && !pathname.startsWith('/login')) {
      router.replace('/login')
    }
  }, [isLoading, user, pathname, router])

  // While loading or unauthenticated (but not yet redirected), render nothing
  // to prevent protected UI from flashing.
  if (isLoading || !user) {
    return null
  }

  return <>{children}</>
}
