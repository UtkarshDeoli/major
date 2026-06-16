"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { INDIAN_LANGUAGES } from "@/lib/constants/exams";

interface StepAboutYouProps {
  onNext: (data: {
    name: string;
    role: string;
    institute: string;
    language: string;
  }) => void;
  onSkip: () => void;
  defaultName?: string;
}

export function StepAboutYou({
  onNext,
  onSkip,
  defaultName = "",
}: StepAboutYouProps) {
  const [name, setName] = useState(defaultName);
  const [role, setRole] = useState<"Student" | "Teacher">("Student");
  const [institute, setInstitute] = useState("");
  const [language, setLanguage] = useState("en");

  const isNameFilled = name.trim().length > 0;

  const handleNext = () => {
    if (!isNameFilled) return;
    onNext({
      name: name.trim(),
      role,
      institute: institute.trim(),
      language,
    });
  };

  return (
    <div className="w-full max-w-md mx-auto space-y-6">
      <div className="space-y-1 text-center">
        <h2 className="text-2xl font-bold">About You</h2>
        <p className="text-sm text-muted-foreground">
          Tell us a little about yourself to personalize your experience
        </p>
      </div>

      <div className="space-y-4">
        {/* Name */}
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input
            id="name"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && isNameFilled) handleNext();
            }}
            className="rounded-md h-9 text-[13px]"
          />
        </div>

        {/* Role */}
        <div className="space-y-2">
          <Label>Role</Label>
          <div className="flex rounded-md border overflow-hidden">
            <button
              type="button"
              onClick={() => setRole("Student")}
              className={`flex-1 py-2 text-sm font-medium transition-colors ${
                role === "Student"
                  ? "bg-primary text-primary-foreground"
                  : "bg-background text-muted-foreground hover:bg-muted"
              }`}
            >
              Student
            </button>
            <button
              type="button"
              onClick={() => setRole("Teacher")}
              className={`flex-1 py-2 text-sm font-medium transition-colors ${
                role === "Teacher"
                  ? "bg-primary text-primary-foreground"
                  : "bg-background text-muted-foreground hover:bg-muted"
              }`}
            >
              Teacher
            </button>
          </div>
        </div>

        {/* Institute */}
        <div className="space-y-2">
          <Label htmlFor="institute">Where do you study? (Optional)</Label>
          <Input
            id="institute"
            placeholder="School, college, or institution name"
            value={institute}
            onChange={(e) => setInstitute(e.target.value)}
            className="rounded-md h-9 text-[13px]"
          />
        </div>

        {/* Language */}
        <div className="space-y-2">
          <Label htmlFor="language">Preferred Language</Label>
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger id="language">
              <SelectValue placeholder="Select a language" />
            </SelectTrigger>
            <SelectContent>
              {INDIAN_LANGUAGES.map((lang) => (
                <SelectItem key={lang.code} value={lang.code}>
                  {lang.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Bottom row */}
      <div className="flex items-center justify-between pt-2">
        <Button variant="ghost" onClick={onSkip}>
          Skip for now
        </Button>
        <Button onClick={handleNext} disabled={!isNameFilled}>
          Next
        </Button>
      </div>
    </div>
  );
}
