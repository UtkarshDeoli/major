"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import {
  BookOpen,
  FileText,
  MessageSquare,
  Target,
  TrendingUp,
  Clock,
  Award,
  Zap,
  ArrowRight,
  BarChart3,
  Brain,
  Calendar,
  CheckCircle2,
  FlaskConical,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { mockTestAPI, pdfAPI, chatAPI } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"

const mockSubjects = [
  { name: "Physics", progress: 72, color: "bg-sky-500", icon: FlaskConical },
  { name: "Mathematics", progress: 45, color: "bg-violet-500", icon: BarChart3 },
  { name: "Chemistry", progress: 58, color: "bg-emerald-500", icon: FlaskConical },
  { name: "Computer Science", progress: 88, color: "bg-amber-500", icon: Brain },
]

const recentActivities = [
  { action: "Completed Mock Test", detail: "Physics - Wave Optics", time: "2h ago", icon: CheckCircle2 },
  { action: "Analyzed Question Paper", detail: "JEE Mains 2023", time: "5h ago", icon: FileText },
  { action: "Chat session", detail: "Organic Chemistry Notes", time: "1d ago", icon: MessageSquare },
  { action: "Uploaded document", detail: "Calculus Formulas.pdf", time: "2d ago", icon: BookOpen },
]

export default function DashboardPage() {
  const router = useRouter()
  const { toast } = useToast()
  const [userEmail, setUserEmail] = useState("")
  const [stats, setStats] = useState({
    documents: 0,
    chats: 0,
    testsTaken: 0,
    avgScore: 0,
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem("token")
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]))
        setUserEmail(payload.sub || "Student")
      } catch {
        setUserEmail("Student")
      }
    }

    const fetchStats = async () => {
      try {
        const [pdfs, sessions, tests] = await Promise.all([
          pdfAPI.listPDFs().catch(() => []),
          chatAPI.listChatSessions().catch(() => []),
          mockTestAPI.listMockTests().catch(() => []),
        ])
        const totalScore = tests.reduce((acc: number, t: any) => acc + (t.latest_submission?.percentage || 0), 0)
        setStats({
          documents: pdfs.length,
          chats: sessions.length,
          testsTaken: tests.length,
          avgScore: tests.length ? Math.round(totalScore / tests.length) : 0,
        })
      } catch {
        // keep defaults
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [])

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Welcome */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Welcome back, {userEmail.split("@")[0] || "Student"}! 🎓</h1>
          <p className="text-muted-foreground mt-1">Here&apos;s what&apos;s happening with your studies today.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => router.push("/chat")}>
            <MessageSquare className="h-4 w-4 mr-2" />
            New Chat
          </Button>
          <Button onClick={() => router.push("/test?tab=mock")}>
            <Target className="h-4 w-4 mr-2" />
            Take Test
          </Button>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="neo-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Documents</CardTitle>
            <BookOpen className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? "—" : stats.documents}</div>
            <p className="text-xs text-muted-foreground">Total uploaded</p>
          </CardContent>
        </Card>

        <Card className="neo-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Chat Sessions</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? "—" : stats.chats}</div>
            <p className="text-xs text-muted-foreground">Active conversations</p>
          </CardContent>
        </Card>

        <Card className="neo-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Mock Tests</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? "—" : stats.testsTaken}</div>
            <p className="text-xs text-muted-foreground">Tests taken</p>
          </CardContent>
        </Card>

        <Card className="neo-card">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Avg Score</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loading ? "—" : `${stats.avgScore}%`}</div>
            <p className="text-xs text-muted-foreground">Across all tests</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Subjects */}
        <Card className="lg:col-span-2 neo-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              Your Subjects
            </CardTitle>
            <CardDescription>Track progress across your study areas</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {mockSubjects.map((subject) => (
              <div key={subject.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${subject.color} bg-opacity-10`}>
                      <subject.icon className={`h-4 w-4 ${subject.color.replace("bg-", "text-")}`} />
                    </div>
                    <span className="font-medium">{subject.name}</span>
                  </div>
                  <span className="text-sm text-muted-foreground">{subject.progress}%</span>
                </div>
                <Progress value={subject.progress} className="h-2" />
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Subscription + Activity */}
        <div className="space-y-6">
          <Card className="neo-card border-primary/20">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Subscription</CardTitle>
                <Badge variant="secondary">Pro Plan</Badge>
              </div>
              <CardDescription>Active until Dec 31, 2025</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Documents used</span>
                  <span className="font-medium">12 / 50</span>
                </div>
                <Progress value={24} className="h-2" />
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Mock tests</span>
                  <span className="font-medium">8 / 20</span>
                </div>
                <Progress value={40} className="h-2" />
              </div>
              <Button className="w-full" variant="outline">
                <Zap className="h-4 w-4 mr-2" />
                Upgrade Plan
              </Button>
            </CardContent>
          </Card>

          <Card className="neo-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Recent Activity</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {recentActivities.map((activity, i) => (
                <div key={i} className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-muted">
                    <activity.icon className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">{activity.action}</p>
                    <p className="text-xs text-muted-foreground truncate">{activity.detail}</p>
                  </div>
                  <span className="text-xs text-muted-foreground whitespace-nowrap">{activity.time}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="neo-card hover:border-primary/50 transition-colors">
          <CardContent className="p-6">
            <Link href="/test?tab=analysis" className="flex items-center justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  <span className="font-semibold">Analyze Papers</span>
                </div>
                <p className="text-sm text-muted-foreground">Upload syllabus & previous papers to get AI insights</p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </Link>
          </CardContent>
        </Card>

        <Card className="neo-card hover:border-primary/50 transition-colors">
          <CardContent className="p-6">
            <Link href="/test?tab=mock" className="flex items-center justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" />
                  <span className="font-semibold">Generate Mock Test</span>
                </div>
                <p className="text-sm text-muted-foreground">Create personalized tests based on your materials</p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </Link>
          </CardContent>
        </Card>

        <Card className="neo-card hover:border-primary/50 transition-colors">
          <CardContent className="p-6">
            <Link href="/chat" className="flex items-center justify-between">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5 text-primary" />
                  <span className="font-semibold">AI Tutor Chat</span>
                </div>
                <p className="text-sm text-muted-foreground">Ask questions and get explanations from your documents</p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Study Streak / Calendar Placeholder */}
      <Card className="neo-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5 text-primary" />
            Study Streak
          </CardTitle>
          <CardDescription>Keep the momentum going! 🔥</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-6">
            <div className="flex flex-col items-center">
              <span className="text-4xl font-bold">12</span>
              <span className="text-sm text-muted-foreground">Day streak</span>
            </div>
            <div className="flex-1 grid grid-cols-7 gap-2">
              {["M", "T", "W", "T", "F", "S", "S"].map((day, i) => (
                <div key={day + i} className="flex flex-col items-center gap-1">
                  <span className="text-xs text-muted-foreground">{day}</span>
                  <div
                    className={`w-8 h-8 rounded-lg ${
                      i < 5 ? "bg-primary/80" : i === 5 ? "bg-primary/40" : "bg-muted"
                    }`}
                  />
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
