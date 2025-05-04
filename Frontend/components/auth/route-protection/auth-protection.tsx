"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useToast } from '@/hooks/use-toast'

export default function AuthProtection({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const { toast } = useToast()
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  
  useEffect(() => {
    // Check if user is authenticated
    const token = localStorage.getItem('token')
    
    if (!token) {
      // No token found, redirect to login
      toast({
        title: "Authentication required",
        description: "Please log in before continuing.",
        variant: "destructive"
      })
      router.push('/login')
    } else {
      setIsAuthenticated(true)
    }
  }, [router, toast])
  
  // Show nothing while checking authentication to prevent flashing content
  if (isAuthenticated === null) {
    return null
  }
  
  // If authenticated, render the children
  return isAuthenticated ? <>{children}</> : null
}