"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import AuthProtection from "@/components/auth/route-protection/auth-protection";
import AppShell from "@/components/dashboard/app-shell";
import { DashboardProvider } from "@/lib/context/dashboard-context";
import { useAuth } from "@/lib/context/auth-context";
import { getRoleHomeRoute } from "@/lib/auth/redirects";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, isLoading: authLoading } = useAuth();

  // If a student needs to finish onboarding, don't render the protected page
  // even for a single frame — redirect immediately while showing a spinner.
  const needsOnboarding =
    user?.role === "student" && !user.onboarding_completed && !pathname.startsWith("/onboarding");
  const needsRoleHome = pathname === "/dashboard" && user && getRoleHomeRoute(user.role) !== "/dashboard";

  useEffect(() => {
    if (authLoading || !user) return;

    if (needsOnboarding) {
      router.replace("/onboarding");
      return;
    }

    if (needsRoleHome) {
      router.replace(getRoleHomeRoute(user.role));
    }
  }, [authLoading, user, pathname, router, needsOnboarding, needsRoleHome]);

  if (authLoading || needsOnboarding || needsRoleHome) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="h-8 w-8 rounded-md border-2 border-primary border-t-transparent animate-spin" />
      </div>
    );
  }

  return (
    <AuthProtection>
      <DashboardProvider>
        <AppShell>{children}</AppShell>
      </DashboardProvider>
    </AuthProtection>
  );
}
