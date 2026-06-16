"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Container from "@/components/global/container";
import { StepAboutYou } from "./step-about-you";
import { StepStudyGoal } from "./step-study-goal";
import { PRESET_EXAMS } from "@/lib/constants/exams";

export function OnboardingContainer() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);

  const handleStep1Next = async (data: {
    name: string;
    role: string;
    institute: string;
    language: string;
  }) => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to save onboarding data");
      setStep(2);
    } catch (error) {
      console.error("Error saving onboarding data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const completeOnboarding = async () => {
    try {
      await fetch("/api/onboarding/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } catch (error) {
      console.error("Error completing onboarding:", error);
    }
  };

  const handleStep2Next = async (presetId: string | null) => {
    setIsLoading(true);
    try {
      if (presetId) {
        const preset = PRESET_EXAMS.find((p) => p.id === presetId);
        if (!preset) throw new Error("Preset not found");

        // 1. Create exam
        const examRes = await fetch("/api/exams", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: preset.name,
            icon: preset.icon,
            is_active: true,
          }),
        });
        if (!examRes.ok) throw new Error("Failed to create exam");
        const exam = await examRes.json();

        // 2. Create subjects
        await Promise.all(
          preset.subjects.map((subjectName) =>
            fetch(`/api/exams/${exam.id}/subjects`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: subjectName }),
            })
          )
        );
      }

      await completeOnboarding();
      router.push("/dashboard");
    } catch (error) {
      console.error("Error in step 2:", error);
      setIsLoading(false);
    }
  };

  const handleSkip = async () => {
    setIsLoading(true);
    await completeOnboarding();
    router.push("/dashboard");
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <Container delay={0.1}>
        <div className="rounded-md border bg-card p-8 max-w-md mx-auto">
          {/* Step dots */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <div
              className={`h-2.5 w-2.5 rounded-full transition-colors ${
                step >= 1 ? "bg-primary" : "bg-muted"
              }`}
            />
            <div
              className={`h-2.5 w-2.5 rounded-full transition-colors ${
                step >= 2 ? "bg-primary" : "bg-muted"
              }`}
            />
          </div>

          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/50 rounded-md z-20">
              <div className="h-8 w-8 rounded-full bg-primary animate-pulse-glow" />
            </div>
          )}

          {step === 1 && (
            <StepAboutYou
              onNext={handleStep1Next}
              onSkip={handleSkip}
              defaultName=""
            />
          )}
          {step === 2 && (
            <StepStudyGoal onNext={handleStep2Next} onBack={() => setStep(1)} />
          )}
        </div>
      </Container>
    </div>
  );
}