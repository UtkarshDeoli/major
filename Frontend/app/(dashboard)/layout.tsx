"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import AuthProtection from "@/components/auth/route-protection/auth-protection";
import AppShell from "@/components/dashboard/app-shell";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
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

  if (!onboardingChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="h-10 w-10 rounded-full bg-primary animate-pulse-glow" />
      </div>
    );
  }

  if (shouldRedirect) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="h-10 w-10 rounded-full bg-primary animate-pulse-glow" />
      </div>
    );
  }

  return (
    <AuthProtection>
      <AppShell>{children}</AppShell>
    </AuthProtection>
  );
}
