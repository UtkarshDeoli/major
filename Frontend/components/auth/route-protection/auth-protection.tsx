"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

export default function AuthProtection({
  children,
}: {
  children: React.ReactNode
}) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  
  useEffect(() => {
    const token = localStorage.getItem('token')
    
    if (!token) {
      // Prevent infinite loops: only redirect if not already on login
      if (!window.location.pathname.startsWith('/login')) {
        window.location.replace('/login')
      }
    } else {
      setIsAuthenticated(true)
    }
  }, [])
  
  // Show nothing while checking authentication to prevent flashing content
  if (isAuthenticated === null) {
    return null
  }
  
  return isAuthenticated ? <>{children}</> : null
}
