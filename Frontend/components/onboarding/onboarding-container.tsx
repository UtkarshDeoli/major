"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Container from "@/components/global/container";
import { StepAboutYou } from "./step-about-you";
import { StepStudyGoal } from "./step-study-goal";
import { PRESET_EXAMS } from "@/lib/constants/exams";
import { examAPI, onboardingAPI } from "@/lib/api";
import { useAuth } from "@/lib/context/auth-context";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";

export function OnboardingContainer() {
  const router = useRouter();
  const { refreshUser } = useAuth();
  const { toast } = useToast();
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
      await onboardingAPI.saveStep1(data);
      setStep(2);
    } catch (error) {
      console.error("Error saving onboarding data:", error);
      toast({
        title: "Couldn't save your details",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const completeOnboarding = async () => {
    // Throws on failure so callers can surface the error and avoid navigating
    // to /dashboard while onboarding_completed is still false (which would
    // bounce the user back to /onboarding in a loop).
    await onboardingAPI.complete();
  };

  const handleStep2Next = async (presetId: string | null) => {
    setIsLoading(true);
    try {
      if (presetId) {
        const preset = PRESET_EXAMS.find((p) => p.id === presetId);
        if (!preset) throw new Error("Preset not found");

        // 1. Create exam
        const exam = await examAPI.createExam({
          name: preset.name,
          icon: preset.icon,
          is_active: true,
        });

        // 2. Create subjects (premade per-exam subjects, e.g. JEE -> Physics/Chemistry/Maths)
        await Promise.all(
          preset.subjects.map((subjectName) => examAPI.createSubject(exam.id, subjectName))
        );
      }

      await completeOnboarding();
      await refreshUser();
      router.push("/dashboard");
    } catch (error) {
      console.error("Error in step 2:", error);
      toast({
        title: "Couldn't set up your study goal",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      setIsLoading(false);
    }
  };

  const handleSkip = async () => {
    setIsLoading(true);
    try {
      await completeOnboarding();
      await refreshUser();
      router.push("/dashboard");
    } catch (error) {
      console.error("Error completing onboarding:", error);
      toast({
        title: "Couldn't finish onboarding",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      setIsLoading(false);
    }
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
              <div className="h-8 w-8 rounded-full bg-primary" />
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