"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import {
  MessageSquare,
  Target,
  TrendingUp,
  FileText,
  Sparkles,
  ArrowRight,
  BookOpen,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { useDashboard } from "@/lib/context/dashboard-context"
import { CollectionsPanel } from "@/components/dashboard/collections-panel"
import { ExamSetupDialog } from "@/components/dashboard/exam-setup-dialog"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/context/auth-context"
import { mockTestAPI, pdfAPI, chatAPI } from "@/lib/api"
import Link from "next/link"

interface DashboardStats {
  documents: number
  chatSessions: number
  testsTaken: number
  avgScore: number | null
}

export default function DashboardPage() {
  return <DashboardContent />
}

function DashboardContent() {
  const router = useRouter()
  const { toast } = useToast()
  const { user } = useAuth()
  const { activeExam, refreshExams } = useDashboard()

  const [dialogOpen, setDialogOpen] = useState(false)
  const [panelOpen, setPanelOpen] = useState(false)
  const [selectedSubject, setSelectedSubject] = useState<{ id: string; name: string } | null>(null)
  const [assignedTests, setAssignedTests] = useState<Array<{ test_id: string; title: string; created_at: string; created_by?: string }>>([])
  const [stats, setStats] = useState<DashboardStats>({ documents: 0, chatSessions: 0, testsTaken: 0, avgScore: null })
  const [isLoadingStats, setIsLoadingStats] = useState(true)

  useEffect(() => { refreshExams() }, [refreshExams])

  useEffect(() => {
    const fetchData = async () => {
      setIsLoadingStats(true)
      try {
        const [pdfs, sessions, tests] = await Promise.allSettled([
          pdfAPI.listPDFs(),
          chatAPI.listChatSessions(),
          mockTestAPI.listMockTests(),
        ])
        const docCount = pdfs.status === "fulfilled" ? (pdfs.value as any[])?.length || 0 : 0
        const sessionCount = sessions.status === "fulfilled" ? (sessions.value as any[])?.length || 0 : 0
        const testList = tests.status === "fulfilled" ? (tests.value as any[]) || [] : []

        const mine = testList.filter((t: any) => t.assigned_to === user?.email)
        setAssignedTests(mine)

        const submitted = testList.filter((t: any) => t.latest_submission)
        const avgScore = submitted.length > 0
          ? submitted.reduce((sum: number, t: any) => sum + (t.latest_submission?.percentage || 0), 0) / submitted.length
          : null

        setStats({ documents: docCount, chatSessions: sessionCount, testsTaken: testList.length, avgScore })
      } catch (err) {
        console.error("Failed to load stats:", err)
      } finally {
        setIsLoadingStats(false)
      }
    }
    fetchData()
  }, [user?.email])

  const handleSubjectClick = (subject: { id: string; name: string }) => {
    setSelectedSubject(subject)
    setPanelOpen(true)
  }

  const handleExamCreated = () => {
    refreshExams()
    toast({ title: "Exam created.", description: "Your study goal has been set up." })
  }

  const greeting = () => {
    const hour = new Date().getHours()
    if (hour < 12) return "Good morning"
    if (hour < 18) return "Good afternoon"
    return "Good evening"
  }

  const statCards = [
    { label: "Documents", value: stats.documents, icon: FileText },
    { label: "Chat Sessions", value: stats.chatSessions, icon: MessageSquare },
    { label: "Tests Taken", value: stats.testsTaken, icon: Target },
    { label: "Avg Score", value: stats.avgScore !== null ? `${Math.round(stats.avgScore)}%` : "—", icon: TrendingUp },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      {/* Welcome */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {greeting()}, {user?.name || "Student"}.
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Here&apos;s what&apos;s happening with your studies.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]" onClick={() => router.push("/chat")}>
            <MessageSquare className="h-3.5 w-3.5 mr-1.5" />
            Chat
          </Button>
          <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={() => router.push("/test?tab=mock")}>
            <Target className="h-3.5 w-3.5 mr-1.5" />
            Take Test
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((stat) => (
          <div key={stat.label} className="rounded-md border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <div className="text-xl font-semibold tabular-nums">
              {isLoadingStats ? (
                <div className="h-6 w-12 bg-muted animate-pulse rounded" />
              ) : (
                stat.value
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Subjects */}
      {activeExam?.subjects && activeExam.subjects.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Subjects</h2>
            <Button variant="outline" size="sm" className="rounded-md h-7 text-xs" onClick={() => setDialogOpen(true)}>
              Add Exam
            </Button>
          </div>
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {activeExam.subjects.map((subject) => (
              <button
                key={subject.id}
                onClick={() => handleSubjectClick({ id: subject.id, name: subject.name })}
                className="rounded-md border bg-card p-4 text-left hover:bg-secondary/50 transition-colors duration-150 group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{subject.name}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1 h-1 rounded-full bg-muted">
                    <div
                      className="h-1 rounded-full bg-primary transition-all"
                      style={{ width: `${Math.min(subject.progress, 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground tabular-nums">{subject.progress}%</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1.5">
                  {subject.collections.length} collection{subject.collections.length !== 1 ? "s" : ""}
                </p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Assigned Tests */}
      {assignedTests.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Assigned Tests</h2>
          <div className="rounded-md border divide-y">
            {assignedTests.map((test) => (
              <div key={test.test_id} className="flex items-center justify-between px-4 py-3">
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{test.title}</p>
                  <p className="text-xs text-muted-foreground">From {test.created_by || "Teacher"}</p>
                </div>
                <Button size="sm" className="rounded-md h-7 text-xs shrink-0 ml-4" onClick={() => router.push(`/test/quiz?testId=${test.test_id}`)}>
                  Start
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quick Actions</h2>
        <div className="grid gap-2 md:grid-cols-3">
          <Link href="/flashcards" className="rounded-md border bg-card p-4 hover:bg-secondary/50 transition-colors duration-150 group">
            <Sparkles className="h-4 w-4 text-primary mb-2" />
            <p className="text-sm font-medium">Flashcards</p>
            <p className="text-xs text-muted-foreground mt-0.5">Review with AI-generated cards.</p>
          </Link>
          <Link href="/analytics" className="rounded-md border bg-card p-4 hover:bg-secondary/50 transition-colors duration-150 group">
            <TrendingUp className="h-4 w-4 text-primary mb-2" />
            <p className="text-sm font-medium">Analytics</p>
            <p className="text-xs text-muted-foreground mt-0.5">Track your performance trends.</p>
          </Link>
          <Link href="/chat" className="rounded-md border bg-card p-4 hover:bg-secondary/50 transition-colors duration-150 group">
            <MessageSquare className="h-4 w-4 text-primary mb-2" />
            <p className="text-sm font-medium">Chat</p>
            <p className="text-xs text-muted-foreground mt-0.5">Ask questions about your materials.</p>
          </Link>
        </div>
      </div>

      <CollectionsPanel exam={activeExam} open={panelOpen} onOpenChange={setPanelOpen} onChat={(examId) => router.push(`/chat?exam=${examId}`)} />
      <ExamSetupDialog open={dialogOpen} onOpenChange={setDialogOpen} onExamCreated={handleExamCreated} />
    </div>
  )
}