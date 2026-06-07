"use client";

import { GraduationCap } from "lucide-react";
import { MagicCard } from "@/components/ui/magic-card";
import { ProgressRing } from "@/components/ui/progress-ring";
import { useDashboard } from "@/lib/context/dashboard-context";
import Container from "@/components/global/container";
import { Button } from "@/components/ui/button";

interface ActiveStudyCardProps {
  onAddExam: () => void;
  onContinueSession: () => void;
}

export function ActiveStudyCard({ onAddExam, onContinueSession }: ActiveStudyCardProps) {
  const { activeExam } = useDashboard();

  const overallProgress =
    activeExam?.subjects?.length
      ? activeExam.subjects.reduce((sum, s) => sum + (s.progress || 0), 0) /
        activeExam.subjects.length
      : 0;

  if (!activeExam) {
    return (
      <Container delay={0.1}>
        <MagicCard className="p-8 flex flex-col items-center justify-center text-center min-h-[200px]">
          <div className="flex flex-col items-center gap-4">
            <div className="p-3 rounded-full bg-primary/10">
              <GraduationCap className="h-8 w-8 text-primary" />
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-semibold">
                Set your exam goal to get started
              </h3>
              <p className="text-sm text-muted-foreground">
                Add an exam to organize your study materials and track your progress
              </p>
            </div>
            <Button onClick={onAddExam}>Add Exam</Button>
          </div>
        </MagicCard>
      </Container>
    );
  }

  return (
    <Container delay={0.1}>
      <MagicCard className="p-6 lg:p-8">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
          <div className="flex flex-col gap-2">
            <span className="text-xs uppercase tracking-wider text-muted-foreground">
              Currently Preparing For
            </span>
            <h2 className="font-subheading italic text-2xl">{activeExam.name}</h2>
            <Button
              variant="ghost"
              className="w-fit px-0"
              onClick={onContinueSession}
            >
              Continue Last Session &rarr;
            </Button>
          </div>
          <ProgressRing progress={overallProgress} size={100} strokeWidth={6} />
        </div>
      </MagicCard>
    </Container>
  );
}
