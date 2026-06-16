"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";
import { useRouter } from "next/navigation";
import api from "@/lib/api";

// ─── Types ──────────────────────────────────────────────────────────────────

export type UserRole = "student" | "teacher" | "subadmin" | "admin";

export interface SubscriptionInfo {
  plan: "weekly" | "monthly";
  started_at: string;
  expires_at: string;
  status: "active" | "expired" | "cancelled";
}

export interface User {
  email: string;
  name?: string;
  role: UserRole;
  institute?: string;
  preferred_language: string;
  onboarding_completed: boolean;
  active_exam_id?: string;
  teacher_id?: string;
  managed_by?: string;
  license_id?: string;
  subscription?: SubscriptionInfo;
}

interface AuthContextValue {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (payload: SignupPayload) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  hasRole: (...roles: UserRole[]) => boolean;
}

export interface SignupPayload {
  email: string;
  password: string;
  name?: string;
}

// ─── Context ────────────────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const readToken = () => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("token");
  };

  const fetchMe = useCallback(async () => {
    const token = readToken();
    if (!token) {
      setUser(null);
      setIsLoading(false);
      return;
    }
    try {
      const response = await api.get("/auth/me");
      setUser(response.data as User);
    } catch (error) {
      console.error("Failed to fetch current user:", error);
      localStorage.removeItem("token");
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMe();
  }, [fetchMe]);

  const login = useCallback(
    async (email: string, password: string) => {
      const params = new URLSearchParams();
      params.append("username", email);
      params.append("password", password);
      const response = await api.post("/auth/login", params, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });
      const { access_token } = response.data as { access_token: string };
      localStorage.setItem("token", access_token);
      const nextUser = (await api.get("/auth/me")).data as User;
      setUser(nextUser);
      setIsLoading(false);
      if (nextUser.role === "teacher") {
        router.replace("/teacher");
      } else {
        router.replace("/dashboard");
      }
    },
    [router]
  );

  const signup = useCallback(
    async (payload: SignupPayload) => {
      const response = await api.post("/auth/signup", payload);
      const { access_token } = response.data as { access_token: string };
      if (access_token) {
        localStorage.setItem("token", access_token);
        const nextUser = (await api.get("/auth/me")).data as User;
        setUser(nextUser);
        setIsLoading(false);
        if (nextUser.role === "teacher") {
          router.replace("/teacher");
        } else {
          router.replace("/onboarding");
        }
      }
    },
    [router]
  );

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    setUser(null);
    router.replace("/login");
  }, [router]);

  const refreshUser = useCallback(async () => {
    await fetchMe();
  }, [fetchMe]);

  const hasRole = useCallback(
    (...roles: UserRole[]) => {
      if (!user) return false;
      return roles.includes(user.role);
    },
    [user]
  );

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        signup,
        logout,
        refreshUser,
        hasRole,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (ctx === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
