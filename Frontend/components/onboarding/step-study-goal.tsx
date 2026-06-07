"use client";

import { FolderOpen, ArrowLeft } from "lucide-react";
import { MagicCard } from "@/components/ui/magic-card";
import { Button } from "@/components/ui/button";
import { PRESET_EXAMS } from "@/lib/constants/exams";

interface StepStudyGoalProps {
  onNext: (presetId: string | null) => void;
  onBack: () => void;
}

export function StepStudyGoal({ onNext, onBack }: StepStudyGoalProps) {
  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      <div className="space-y-1 text-center">
        <h2 className="text-2xl font-bold">Your Study Goal</h2>
        <p className="text-sm text-muted-foreground">
          Choose an exam you are preparing for, or set up your own structure
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {PRESET_EXAMS.map((preset) => (
          <MagicCard
            key={preset.id}
            className="cursor-pointer p-4 hover:-translate-y-0.5 transition-transform"
            onClick={() => onNext(preset.id)}
          >
            <div className="flex flex-col gap-1">
              <h3 className="font-semibold">{preset.name}</h3>
              <p className="text-xs text-muted-foreground">{preset.tagline}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {preset.subjects.length} subjects
              </p>
            </div>
          </MagicCard>
        ))}

        <MagicCard
          className="cursor-pointer p-4 hover:-translate-y-0.5 transition-transform"
          gradientFrom="#9ca3af"
          gradientTo="#6b7280"
          onClick={() => onNext(null)}
        >
          <div className="flex flex-col items-center justify-center gap-2 h-full text-center">
            <FolderOpen className="h-8 w-8 text-muted-foreground" />
            <h3 className="font-semibold">I&apos;ll organize my own way</h3>
            <p className="text-xs text-muted-foreground">
              Create a custom exam without presets
            </p>
          </div>
        </MagicCard>
      </div>

      <div className="flex justify-start pt-2">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
      </div>
    </div>
  );
}
