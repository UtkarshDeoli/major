"use client";

import { useState } from "react";
import { FolderOpen } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { MagicCard } from "@/components/ui/magic-card";
import { PRESET_EXAMS } from "@/lib/constants/exams";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { examAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";

interface ExamSetupDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExamCreated: (examId: string) => void;
}

export function ExamSetupDialog({
  open,
  onOpenChange,
  onExamCreated,
}: ExamSetupDialogProps) {
  const [customName, setCustomName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handlePresetClick = async (preset: (typeof PRESET_EXAMS)[0]) => {
    setIsLoading(true);
    try {
      const exam = await examAPI.createExam({
        name: preset.name,
        icon: preset.icon,
        is_active: true,
      });

      await Promise.all(
        preset.subjects.map((subjectName) => examAPI.createSubject(exam.id, subjectName))
      );

      onExamCreated(exam.id);
      onOpenChange(false);
    } catch (error) {
      console.error("Error creating exam:", error);
      toast({
        title: "Couldn't create exam",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCustomCreate = async () => {
    if (!customName.trim()) return;
    setIsLoading(true);
    try {
      const exam = await examAPI.createExam({
        name: customName.trim(),
        is_active: true,
      });

      onExamCreated(exam.id);
      onOpenChange(false);
    } catch (error) {
      console.error("Error creating custom exam:", error);
      toast({
        title: "Couldn't create exam",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Set Your Exam Goal</DialogTitle>
          <DialogDescription>
            Choose a preset exam or create your own to get started
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
          {PRESET_EXAMS.map((preset) => (
            <MagicCard
              key={preset.id}
              className="cursor-pointer p-4 hover:-translate-y-0.5 transition-transform"
              onClick={() => !isLoading && handlePresetClick(preset)}
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
            onClick={() => {
              if (customName.trim()) {
                handleCustomCreate();
              }
            }}
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

        <div className="mt-6 space-y-2">
          <label className="text-sm font-medium">Custom Exam Name</label>
          <div className="flex gap-2">
            <Input
              placeholder="Enter exam name..."
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCustomCreate();
              }}
            />
            <Button
              onClick={handleCustomCreate}
              disabled={isLoading || !customName.trim()}
            >
              Create
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
