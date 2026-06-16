"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import AuthProtection from "@/components/auth/route-protection/auth-protection";
import AppShell from "@/components/dashboard/app-shell";
import { DashboardProvider } from "@/lib/context/dashboard-context";
import { useAuth } from "@/lib/context/auth-context";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, isLoading: authLoading } = useAuth();
  const [onboardingChecked, setOnboardingChecked] = useState(false);
  const [shouldRedirect, setShouldRedirect] = useState(false);

  useEffect(() => {
    const checkOnboarding = async () => {
      try {
        const res = await fetch("/api/onboarding");
        if (!res.ok) {
          setOnboardingChecked(true);
          return;
        }
        const data = await res.json();
        if (data.onboarding_completed === false) {
          setShouldRedirect(true);
        }
      } catch (error) {
        console.error("Error checking onboarding status:", error);
      } finally {
        setOnboardingChecked(true);
      }
    };

    checkOnboarding();
  }, []);

  useEffect(() => {
    if (shouldRedirect) {
      router.push("/onboarding");
    }
  }, [shouldRedirect, router]);

  useEffect(() => {
    if (!authLoading && user?.role === "teacher" && pathname === "/dashboard") {
      router.replace("/teacher");
    }
  }, [authLoading, user, pathname, router]);

  if (!onboardingChecked || shouldRedirect || authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="h-10 w-10 rounded-full bg-primary animate-pulse-glow" />
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
