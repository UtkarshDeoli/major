"use client";

import type { User, UserRole } from "@/lib/context/auth-context";

/**
 * Default home route for each role. This is where an already-onboarded
 * user lands after authentication or when visiting a protected page directly.
 */
export const ROLE_HOME_ROUTE: Record<UserRole, string> = {
  student: "/dashboard",
  teacher: "/teacher",
  subadmin: "/admin",
  admin: "/admin",
};

/**
 * Destination immediately after a successful sign-in.
 * - Teachers / admins / subadmins always go straight to their role home.
 * - Students go to onboarding the very first time, otherwise dashboard.
 */
export function getPostAuthRedirect(user: User): string {
  if (user.role === "student") {
    return user.onboarding_completed ? "/dashboard" : "/onboarding";
  }
  return ROLE_HOME_ROUTE[user.role];
}

/**
 * Generic role home. Use when the only thing known is the role.
 */
export function getRoleHomeRoute(role: UserRole): string {
  return ROLE_HOME_ROUTE[role] || "/dashboard";
}
