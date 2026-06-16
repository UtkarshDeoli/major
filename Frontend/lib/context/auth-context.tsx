"use client"

import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from "react"
import { authAPI } from "@/lib/api"

export type UserRole = "student" | "teacher" | "subadmin" | "admin"

export interface AuthUser {
  email: string
  name?: string
  role: UserRole
  onboarding_completed?: boolean
  institute?: string
  preferred_language?: string
  active_exam_id?: string
  teacher_id?: string
  managed_by?: string
  license_id?: string
  subscription?: {
    plan?: string
    status?: string
    expires_at?: string
  }
}

interface AuthContextValue {
  user: AuthUser | null
  role: UserRole | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<AuthUser>
  signup: (email: string, password: string, name?: string) => Promise<AuthUser>
  logout: () => void
  refreshUser: () => Promise<AuthUser | null>
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const loadUser = useCallback(async () => {
    if (!authAPI.isAuthenticated()) {
      setIsLoading(false)
      setUser(null)
      return null
    }

    try {
      const me = await authAPI.getMe()
      setUser(me)
      return me as AuthUser
    } catch (error) {
      console.error("Failed to load current user:", error)
      authAPI.logout()
      setUser(null)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadUser()
  }, [loadUser])

  const login = useCallback(async (email: string, password: string) => {
    await authAPI.login(email, password)
    const me = await loadUser()
    if (!me) {
      throw new Error("Login succeeded but user profile could not be loaded.")
    }
    return me
  }, [loadUser])

  const signup = useCallback(async (email: string, password: string, name?: string) => {
    await authAPI.signup(email, password, name)
    const me = await loadUser()
    if (!me) {
      throw new Error("Signup succeeded but user profile could not be loaded.")
    }
    return me
  }, [loadUser])

  const logout = useCallback(() => {
    authAPI.logout()
    setUser(null)
  }, [])

  const refreshUser = useCallback(async () => {
    return loadUser()
  }, [loadUser])

  const value: AuthContextValue = {
    user,
    role: user?.role ?? null,
    isLoading,
    isAuthenticated: !!user,
    login,
    signup,
    logout,
    refreshUser,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
