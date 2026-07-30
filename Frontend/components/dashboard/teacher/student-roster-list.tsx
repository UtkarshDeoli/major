"use client"

import { motion } from "framer-motion"
import { ChevronRight } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export interface RosterStudent {
  email: string
  name?: string
  average_score: number
  tests_taken: number
  last_active_at?: string
  weaknesses?: string[]
  class_ids?: string[]
  class_names?: string[]
}

interface StudentRosterListProps {
  students: RosterStudent[]
  onSelect: (student: RosterStudent) => void
  emptyMessage?: string
}

export function StudentRosterList({ students, onSelect, emptyMessage = "No students yet." }: StudentRosterListProps) {
  if (students.length === 0) {
    return (
      <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">
        {emptyMessage}
      </div>
    )
  }

  return (
    <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
      {students.map((student, i) => (
        <motion.button
          key={student.email}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: i * 0.04 }}
          whileHover={{ y: -2, transition: { duration: 0.15 } }}
          onClick={() => onSelect(student)}
          className={cn(
            "rounded-xl border bg-card p-4 text-left transition-all duration-200 group",
            "hover:shadow-md hover:border-primary/30"
          )}
        >
          <div className="flex items-start justify-between">
            <div className="min-w-0">
              <p className="text-sm font-medium truncate">{student.name || student.email.split("@")[0]}</p>
              <p className="text-[11px] text-muted-foreground truncate font-mono">{student.email}</p>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity mt-0.5 shrink-0" />
          </div>

          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span>Score</span>
                <span className="font-semibold text-foreground">{student.average_score}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: student.average_score >= 70 ? "hsl(160, 84%, 39%)" : student.average_score >= 50 ? "hsl(40, 84%, 50%)" : "hsl(0, 84%, 60%)" }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(student.average_score, 100)}%` }}
                  transition={{ duration: 0.8, delay: 0.3 + i * 0.04, ease: "easeOut" }}
                />
              </div>
            </div>
            <div className="text-center shrink-0">
              <p className="text-lg font-semibold tabular-nums">{student.tests_taken}</p>
              <p className="text-[10px] text-muted-foreground">tests</p>
            </div>
          </div>

          {student.weaknesses && student.weaknesses.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2.5">
              {student.weaknesses.slice(0, 3).map((topic) => (
                <Badge key={topic} variant="secondary" className="text-[10px] font-normal px-1.5 py-0 bg-red-500/10 text-red-400 border-red-500/20 rounded-md">
                  {topic}
                </Badge>
              ))}
              {student.weaknesses.length > 3 && (
                <span className="text-[10px] text-muted-foreground">+{student.weaknesses.length - 3}</span>
              )}
            </div>
          )}
        </motion.button>
      ))}
    </div>
  )
}
