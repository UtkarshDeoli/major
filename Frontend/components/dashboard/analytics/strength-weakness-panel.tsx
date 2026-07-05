"use client"

import { ThumbsUp, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface SubjectFeedback {
  subject: string
  strengths: string[]
  weaknesses: string[]
}

interface StrengthWeaknessPanelProps {
  data: SubjectFeedback[]
  className?: string
}

export function StrengthWeaknessPanel({ data, className }: StrengthWeaknessPanelProps) {
  const hasFeedback = data.some((s) => s.strengths.length > 0 || s.weaknesses.length > 0)

  return (
    <div className={cn("flex flex-col", className)}>
      {!hasFeedback ? (
        <p className="text-sm text-muted-foreground text-center py-10">
          Complete a few mock tests to see your strengths and focus areas.
        </p>
      ) : (
        <div className="space-y-4 max-h-[280px] overflow-y-auto pr-1">
          {data.map((subject) => (
            <div key={subject.subject} className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">{subject.subject}</h4>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  {subject.strengths.length} strength{subject.strengths.length !== 1 ? "s" : ""} ·{" "}
                  {subject.weaknesses.length} focus area{subject.weaknesses.length !== 1 ? "s" : ""}
                </span>
              </div>

              {subject.strengths.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {subject.strengths.map((strength, i) => (
                    <span
                      key={`str-${i}`}
                      className="inline-flex items-start gap-1.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 px-2.5 py-1 text-xs leading-snug text-left border border-emerald-500/20 break-words"
                    >
                      <ThumbsUp className="h-3 w-3 shrink-0 mt-0.5" />
                      {strength}
                    </span>
                  ))}
                </div>
              )}

              {subject.weaknesses.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {subject.weaknesses.map((weakness, i) => (
                    <span
                      key={`weak-${i}`}
                      className="inline-flex items-start gap-1.5 rounded-full bg-rose-500/10 text-rose-600 dark:text-rose-400 px-2.5 py-1 text-xs leading-snug text-left border border-rose-500/20 break-words"
                    >
                      <AlertCircle className="h-3 w-3 shrink-0 mt-0.5" />
                      {weakness}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
