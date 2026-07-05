"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { mockTestAPI } from "@/lib/api"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Loader2, ClipboardList, CheckCircle2, Clock } from "lucide-react"

interface AttemptsProps {
  testId: string | null
  isTeacher: boolean
  open: boolean
  onOpenChange: (o: boolean) => void
}

interface Submission {
  submission_id: string
  user_id: string
  total_score: number
  max_score: number
  percentage: number
  time_taken: number
  status?: string
  grading_mode?: string
  created_at?: string
}

interface QFeedback {
  question_id: string
  question: string
  user_answer: string
  correct_answer?: string
  feedback?: string
  marks_awarded: number
  max_marks: number
}

export function MockTestAttemptsDialog({ testId, isTeacher, open, onOpenChange }: AttemptsProps) {
  const { toast } = useToast()
  const [submissions, setSubmissions] = useState<Submission[]>([])
  const [loading, setLoading] = useState(false)
  const [grading, setGrading] = useState<Submission | null>(null)
  const [questions, setQuestions] = useState<QFeedback[]>([])
  const [grades, setGrades] = useState<Record<string, { marks: number; feedback: string }>>({})
  const [loadingGrade, setLoadingGrade] = useState(false)
  const [savingGrade, setSavingGrade] = useState(false)

  const fetchSubs = async () => {
    if (!testId) return
    setLoading(true)
    try {
      const subs = await mockTestAPI.listSubmissions(testId)
      setSubmissions((subs || []) as Submission[])
    } catch (e) {
      toast({ title: "Couldn't load attempts", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open && testId) fetchSubs()
    if (!open) { setSubmissions([]); setGrading(null); setQuestions([]); setGrades({}) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, testId])

  const openGrading = async (sub: Submission) => {
    setGrading(sub)
    setLoadingGrade(true)
    try {
      const analysis = await mockTestAPI.getAnalysisBySubmissionId(sub.submission_id)
      const qf: QFeedback[] = (analysis.question_feedback || []) as QFeedback[]
      setQuestions(qf)
      const init: Record<string, { marks: number; feedback: string }> = {}
      qf.forEach((q) => { init[q.question_id] = { marks: q.marks_awarded, feedback: q.feedback || "" } })
      setGrades(init)
    } catch (e) {
      toast({ title: "Couldn't load submission", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoadingGrade(false)
    }
  }

  const submitGrades = async () => {
    if (!grading) return
    setSavingGrade(true)
    try {
      const payload = questions.map((q) => ({
        question_id: q.question_id,
        marks_awarded: grades[q.question_id]?.marks ?? q.marks_awarded,
        feedback: grades[q.question_id]?.feedback,
      }))
      await mockTestAPI.gradeSubmission(grading.submission_id, payload)
      toast({ title: "Graded", description: "Submission marked and finalized." })
      setGrading(null); setQuestions([]); setGrades({})
      fetchSubs()
    } catch (e) {
      toast({ title: "Grading failed", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setSavingGrade(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2"><ClipboardList className="h-4 w-4" /> Attempts</DialogTitle>
        </DialogHeader>

        {loading ? (
          <div className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
        ) : submissions.length === 0 ? (
          <p className="text-sm text-muted-foreground">No attempts yet.</p>
        ) : (
          <div className="space-y-2">
            {submissions.map((s) => (
              <div key={s.submission_id} className="flex items-center gap-3 rounded-md border p-2.5">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{s.percentage.toFixed(0)}% · {s.total_score}/{s.max_score}</div>
                  <div className="text-[11px] text-muted-foreground flex items-center gap-2">
                    <Clock className="h-3 w-3" />{Math.round((s.time_taken || 0) / 60)}m
                    {s.status && <span className={s.status === "graded" ? "text-emerald-600" : "text-amber-600"}>{s.status === "graded" ? <CheckCircle2 className="inline h-3 w-3" /> : null} {s.status.replace("_", " ")}</span>}
                  </div>
                </div>
                {isTeacher && s.status === "pending_review" && (
                  <Button size="sm" className="h-7 text-[12px]" onClick={() => openGrading(s)}>Grade</Button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Grading sub-dialog */}
        <Dialog open={!!grading} onOpenChange={(o) => !o && setGrading(null)}>
          <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
            <DialogHeader><DialogTitle>Grade answers</DialogTitle></DialogHeader>
            {loadingGrade ? (
              <div className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
            ) : (
              <div className="space-y-4">
                {questions.filter((q) => !q.correct_answer).map((q) => (
                  <div key={q.question_id} className="space-y-1.5 rounded-md border p-3">
                    <p className="text-sm font-medium">{q.question}</p>
                    <p className="text-xs text-muted-foreground whitespace-pre-wrap bg-muted/40 rounded p-2">Answer: {q.user_answer || "(blank)"}</p>
                    <div className="flex items-end gap-2">
                      <div className="space-y-1 flex-1">
                        <Label className="text-[11px]">Marks (max {q.max_marks})</Label>
                        <Input
                          type="number" min={0} max={q.max_marks} step="0.5"
                          value={grades[q.question_id]?.marks ?? 0}
                          onChange={(e) => setGrades((p) => ({ ...p, [q.question_id]: { ...(p[q.question_id] || { feedback: "" }), marks: Number(e.target.value) } }))}
                          className="h-8 text-[13px]"
                        />
                      </div>
                    </div>
                    <Input
                      placeholder="Feedback (optional)"
                      value={grades[q.question_id]?.feedback ?? ""}
                      onChange={(e) => setGrades((p) => ({ ...p, [q.question_id]: { ...(p[q.question_id] || { marks: 0 }), feedback: e.target.value } }))}
                      className="h-8 text-[13px]"
                    />
                  </div>
                ))}
                <Button disabled={savingGrade} onClick={submitGrades} className="w-full">
                  {savingGrade ? <Loader2 className="h-4 w-4 animate-spin" /> : "Submit grades"}
                </Button>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </DialogContent>
    </Dialog>
  )
}