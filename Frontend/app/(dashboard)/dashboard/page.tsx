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
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { useDashboard } from "@/lib/context/dashboard-context"
import { CollectionsPanel } from "@/components/dashboard/collections-panel"
import { ExamSetupDialog } from "@/components/dashboard/exam-setup-dialog"
import { EnrollClassDialog } from "@/components/dashboard/student/enroll-class-dialog"
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

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.07 } } },
  item: {
    initial: { opacity: 0, y: 14 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
  },
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
        const failed = [pdfs, sessions, tests].filter((r) => r.status === "rejected")
        if (failed.length > 0) {
          console.error("Dashboard stats fetch failures:", failed)
          toast({
            title: "Some stats failed to load",
            description: "Parts of your dashboard may appear incomplete.",
            variant: "destructive",
          })
        }
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
    { label: "Documents", value: stats.documents, icon: FileText, color: "hsl(221, 83%, 53%)" },
    { label: "Chat Sessions", value: stats.chatSessions, icon: MessageSquare, color: "hsl(237, 56%, 60%)" },
    { label: "Tests Taken", value: stats.testsTaken, icon: Target, color: "hsl(160, 84%, 39%)" },
    { label: "Avg Score", value: stats.avgScore !== null ? `${Math.round(stats.avgScore)}%` : "—", icon: TrendingUp, color: "hsl(280, 56%, 60%)" },
  ]

  const quickActions = [
    { icon: Sparkles, label: "Flashcards", desc: "Review with AI-generated cards.", href: "/flashcards" },
    { icon: TrendingUp, label: "Analytics", desc: "Track your performance trends.", href: "/analytics" },
    { icon: MessageSquare, label: "Chat", desc: "Ask questions about your materials.", href: "/chat" },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      {/* Welcome */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {greeting()}, {user?.name || "Student"}.
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Here&apos;s what&apos;s happening with your studies.
          </p>
        </div>
        <div className="flex gap-2">
          {user?.role === "student" && <EnrollClassDialog />}
          <Button variant="outline" size="sm" className="rounded-lg h-8 text-[13px]" onClick={() => router.push("/chat")}>
            <MessageSquare className="h-3.5 w-3.5 mr-1.5" />
            Chat
          </Button>
          <Button size="sm" className="rounded-lg h-8 text-[13px]" onClick={() => router.push("/mock-tests")}>
            <Target className="h-3.5 w-3.5 mr-1.5" />
            Take Test
          </Button>
        </div>
      </motion.div>

      {/* Stats */}
      <motion.div
        variants={stagger.container}
        initial="initial"
        animate="animate"
        className="grid grid-cols-2 md:grid-cols-4 gap-3"
      >
        {statCards.map((stat) => (
          <motion.div
            key={stat.label}
            variants={stagger.item}
            whileHover={{ y: -2, transition: { duration: 0.15 } }}
            className="rounded-xl border bg-card p-4 space-y-2.5 transition-shadow hover:shadow-md group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <div className="h-7 w-7 rounded-lg flex items-center justify-center transition-colors" style={{ backgroundColor: `${stat.color}15` }}>
                <stat.icon className="h-3.5 w-3.5" style={{ color: stat.color }} />
              </div>
            </div>
            <div className="text-xl font-semibold tabular-nums tracking-tight">
              {isLoadingStats ? (
                <div className="h-7 w-12 bg-muted animate-pulse rounded" />
              ) : (
                stat.value
              )}
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Subjects */}
      {activeExam?.subjects && activeExam.subjects.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="space-y-3"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Subjects</h2>
            <Button variant="outline" size="sm" className="rounded-lg h-7 text-xs" onClick={() => setDialogOpen(true)}>
              Add Exam
            </Button>
          </div>
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {activeExam.subjects.map((subject, i) => (
              <motion.button
                key={subject.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.25 + i * 0.05 }}
                whileHover={{ y: -2, transition: { duration: 0.15 } }}
                onClick={() => handleSubjectClick({ id: subject.id, name: subject.name })}
                className="rounded-xl border bg-card p-4 text-left hover:shadow-md hover:border-primary/30 transition-all duration-200 group"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{subject.name}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-primary"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(subject.progress, 100)}%` }}
                      transition={{ duration: 0.8, delay: 0.3 + i * 0.05, ease: "easeOut" }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground tabular-nums">{subject.progress}%</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1.5">
                  {subject.collections.length} collection{subject.collections.length !== 1 ? "s" : ""}
                </p>
              </motion.button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Assigned Tests */}
      {assignedTests.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className="space-y-3"
        >
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Assigned Tests</h2>
          <div className="rounded-xl border divide-y overflow-hidden">
            {assignedTests.map((test, i) => (
              <motion.div
                key={test.test_id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.35 + i * 0.05 }}
                className="flex items-center justify-between px-4 py-3 hover:bg-secondary/30 transition-colors"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{test.title}</p>
                  <p className="text-xs text-muted-foreground">From {test.created_by || "Teacher"}</p>
                </div>
                <Button size="sm" className="rounded-lg h-7 text-xs shrink-0 ml-4" onClick={() => router.push(`/test/quiz?testId=${test.test_id}`)}>
                  Start
                </Button>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.35 }}
        className="space-y-3"
      >
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quick Actions</h2>
        <div className="grid gap-3 md:grid-cols-3">
          {quickActions.map((action, i) => (
            <motion.div
              key={action.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.4 + i * 0.05 }}
              whileHover={{ y: -3, transition: { duration: 0.15 } }}
            >
              <Link href={action.href} className="block rounded-xl border bg-card p-5 hover:shadow-md hover:border-primary/30 transition-all duration-200 group relative overflow-hidden">
                <action.icon className="absolute -right-5 -bottom-5 h-24 w-24 text-primary opacity-[0.06] group-hover:opacity-[0.1] transition-opacity" />
                <div className="relative">
                  <p className="text-sm font-medium">{action.label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{action.desc}</p>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </motion.div>

      <CollectionsPanel exam={activeExam} open={panelOpen} onOpenChange={setPanelOpen} onChat={(examId) => router.push(`/chat?exam=${examId}`)} />
      <ExamSetupDialog open={dialogOpen} onOpenChange={setDialogOpen} onExamCreated={handleExamCreated} />
    </div>
  )
}