"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import api from "@/lib/api";
import { getPostAuthRedirect } from "@/lib/auth/redirects";

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
  teacher_ids?: string[];
  managed_by?: string;
  class_ids?: string[];
  license_id?: string;
  subscription?: SubscriptionInfo;
}

interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
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
  /**
   * Store a token and hydrate the user object from it. Used by the OAuth
   * callback after the backend redirects back with a token.
   */
  hydrateFromToken: (token: string, initialUser?: User) => Promise<User>;
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

  const readToken = useCallback(() => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("token");
  }, []);

  const writeToken = useCallback((token: string | null) => {
    if (token) {
      localStorage.setItem("token", token);
    } else {
      localStorage.removeItem("token");
    }
  }, []);

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
      // Clear the token when the server rejects it (401) or when the user
      // account no longer exists (404). Keep it for transient failures.
      if (
        axios.isAxiosError(error) &&
        (error.response?.status === 401 || error.response?.status === 404)
      ) {
        writeToken(null);
      }
      console.error("Failed to fetch current user:", error);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, [readToken, writeToken]);

  useEffect(() => {
    fetchMe();
  }, [fetchMe]);

  /**
   * Hydrate the context from a brand-new token (e.g., after login, signup,
   * or Google OAuth callback). Optionally reuse a `user` object returned
   * by the backend to avoid a second network round-trip.
   */
  const hydrateFromToken = useCallback(
    async (token: string, initialUser?: User) => {
      writeToken(token);
      if (initialUser) {
        setUser(initialUser);
        setIsLoading(false);
        return initialUser;
      }
      // Fallback: fetch /auth/me when the backend did not embed the user.
      try {
        const response = await api.get("/auth/me");
        const nextUser = response.data as User;
        setUser(nextUser);
        setIsLoading(false);
        return nextUser;
      } catch (error) {
        if (
          axios.isAxiosError(error) &&
          (error.response?.status === 401 || error.response?.status === 404)
        ) {
          writeToken(null);
        }
        setUser(null);
        setIsLoading(false);
        throw error;
      }
    },
    [writeToken]
  );

  const finalizeAuth = useCallback(
    (nextUser: User) => {
      // Keep the spinner up during the redirect so consumers don't render
      // partially-authenticated UI or fire competing navigations.
      router.replace(getPostAuthRedirect(nextUser));
    },
    [router]
  );

  const login = useCallback(
    async (email: string, password: string) => {
      setIsLoading(true);
      try {
        const params = new URLSearchParams();
        params.append("username", email);
        params.append("password", password);
        const response = await api.post("/auth/login", params, {
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
        });
        const { access_token, user: nextUser } = response.data as AuthResponse;
        await hydrateFromToken(access_token, nextUser);
        finalizeAuth(nextUser);
      } catch (error) {
        setIsLoading(false);
        throw error;
      }
    },
    [hydrateFromToken, finalizeAuth]
  );

  const signup = useCallback(
    async (payload: SignupPayload) => {
      setIsLoading(true);
      try {
        const response = await api.post("/auth/signup", payload);
        const { access_token, user: nextUser } = response.data as AuthResponse;
        if (!access_token) {
          throw new Error("No access token received from signup.");
        }
        await hydrateFromToken(access_token, nextUser);
        finalizeAuth(nextUser);
      } catch (error) {
        setIsLoading(false);
        throw error;
      }
    },
    [hydrateFromToken, finalizeAuth]
  );

  const logout = useCallback(() => {
    writeToken(null);
    setUser(null);
    setIsLoading(false);
    router.replace("/login");
  }, [router, writeToken]);

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
        hydrateFromToken,
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
