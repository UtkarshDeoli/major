"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth, type UserRole } from "@/lib/context/auth-context";

interface RoleGuardProps {
  allowedRoles: UserRole[];
  fallback?: string;
  children: React.ReactNode;
}

export default function RoleGuard({
  allowedRoles,
  fallback = "/dashboard",
  children,
}: RoleGuardProps) {
  const router = useRouter();
  const { user, isLoading, isAuthenticated } = useAuth();
  const role = user?.role;

  useEffect(() => {
    if (isLoading) return;

    if (!isAuthenticated || !role) {
      router.replace("/login");
      return;
    }

    if (!allowedRoles.includes(role)) {
      router.replace(fallback);
    }
  }, [isLoading, isAuthenticated, role, allowedRoles, fallback, router]);

  if (isLoading || !isAuthenticated || !role || !allowedRoles.includes(role)) {
    return null;
  }

  return <>{children}</>;
}
