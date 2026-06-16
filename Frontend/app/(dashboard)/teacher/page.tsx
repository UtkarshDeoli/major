"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { BookOpen, Users, TrendingUp, Target, Activity, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { teacherAPI } from "@/lib/api"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { cn } from "@/lib/utils"

interface StudentAnalytics {
  email: string
  name?: string
  tests_taken: number
  average_score: number
  last_active_at?: string
  strengths: string[]
  weaknesses: string[]
}

interface TeacherAnalytics {
  total_students: number
  active_students: number
  total_tests_taken: number
  class_average: number
  student_analytics: StudentAnalytics[]
}

interface ManagedStudent {
  email: string
  name?: string
}

function TeacherDashboardContent() {
  const router = useRouter()
  const { toast } = useToast()
  const [students, setStudents] = useState<ManagedStudent[]>([])
  const [analytics, setAnalytics] = useState<TeacherAnalytics | null>(null)
  const [selectedStudentEmail, setSelectedStudentEmail] = useState<string | null>(null)
  const [selectedStudent, setSelectedStudent] = useState<StudentAnalytics | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [studentsData, analyticsData] = await Promise.all([
          teacherAPI.listManagedStudents(),
          teacherAPI.getAnalytics(),
        ])
        setStudents(studentsData || [])
        setAnalytics(analyticsData)
      } catch (error: any) {
        console.error("Error fetching teacher data:", error)
        toast({
          title: "Error",
          description: error.response?.data?.detail || "Failed to load teacher dashboard data.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [toast])

  const handleCreateTest = () => {
    router.push(`/test?tab=mock&student=${encodeURIComponent(selectedStudentEmail ?? "")}`)
  }

  const handleStudentClick = (student: StudentAnalytics) => {
    setSelectedStudent(student)
  }

  const statCards = [
    { label: "Total Students", value: analytics?.total_students ?? 0, icon: Users },
    { label: "Active Students", value: analytics?.active_students ?? 0, icon: Activity },
    { label: "Tests Taken", value: analytics?.total_tests_taken ?? 0, icon: Target },
    { label: "Class Average", value: analytics ? `${analytics.class_average}%` : "—", icon: TrendingUp },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Teacher Dashboard.</h1>
          <p className="text-sm text-muted-foreground mt-1">Monitor student performance and create targeted tests.</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            className="rounded-md border bg-background px-2 py-1.5 text-[13px] h-8"
            value={selectedStudentEmail ?? ""}
            onChange={(e) => setSelectedStudentEmail(e.target.value || null)}
          >
            <option value="">Assign to me</option>
            {students.map((s) => (
              <option key={s.email} value={s.email}>{s.name || s.email}</option>
            ))}
          </select>
          <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={handleCreateTest}>
            <BookOpen className="h-3.5 w-3.5 mr-1.5" />
            Create Test
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((stat) => (
          <div key={stat.label} className="rounded-md border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <div className="text-xl font-semibold tabular-nums">
              {isLoading ? <div className="h-6 w-12 bg-muted animate-pulse rounded" /> : stat.value}
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Students</h2>
        <div className="rounded-md border">
          <div className="grid grid-cols-[1fr_100px_100px_1fr] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
            <span>Name</span>
            <span className="text-right">Tests</span>
            <span className="text-right">Avg</span>
            <span>Weaknesses</span>
          </div>
          {isLoading ? (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">Loading...</div>
          ) : analytics?.student_analytics && analytics.student_analytics.length > 0 ? (
            analytics.student_analytics.map((student) => (
              <button
                key={student.email}
                onClick={() => handleStudentClick(student)}
                className="grid grid-cols-[1fr_100px_100px_1fr] gap-4 px-4 py-3 w-full text-left border-b last:border-b-0 hover:bg-secondary/50 transition-colors duration-150 group"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate">{student.name || student.email}</p>
                  <p className="text-xs text-muted-foreground truncate font-mono">{student.email}</p>
                </div>
                <span className="text-sm tabular-nums text-right self-center">{student.tests_taken}</span>
                <span className="text-sm tabular-nums text-right self-center">{student.average_score}%</span>
                <div className="flex flex-wrap gap-1 self-center">
                  {student.weaknesses?.slice(0, 3).map((topic) => (
                    <Badge key={topic} variant="secondary" className="text-[10px] font-normal px-1.5 py-0 bg-red-500/10 text-red-400 border-red-500/20">
                      {topic}
                    </Badge>
                  ))}
                  {student.weaknesses?.length > 3 && (
                    <span className="text-[10px] text-muted-foreground">+{student.weaknesses.length - 3}</span>
                  )}
                </div>
              </button>
            ))
          ) : (
            <div className="px-4 py-12 text-center text-sm text-muted-foreground">
              No students linked yet. Manage students to see them here.
            </div>
          )}
        </div>
      </div>

      {selectedStudent && (
        <div className="fixed inset-0 z-50 flex justify-end">
          <div className="absolute inset-0 bg-black/50" onClick={() => setSelectedStudent(null)} />
          <div className="relative w-96 bg-background border-l h-full overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">{selectedStudent.name || selectedStudent.email}</h3>
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setSelectedStudent(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <p className="text-xs text-muted-foreground font-mono">{selectedStudent.email}</p>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Tests Taken</p>
                <p className="text-xl font-semibold tabular-nums">{selectedStudent.tests_taken}</p>
              </div>
              <div className="rounded-md border p-3">
                <p className="text-xs text-muted-foreground">Average Score</p>
                <p className="text-xl font-semibold tabular-nums">{selectedStudent.average_score}%</p>
              </div>
            </div>

            {selectedStudent.strengths && selectedStudent.strengths.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Strengths</p>
                <div className="flex flex-wrap gap-1">
                  {selectedStudent.strengths.map((s) => (
                    <Badge key={s} variant="secondary" className="text-xs font-normal bg-green-500/10 text-green-400 border-green-500/20">{s}</Badge>
                  ))}
                </div>
              </div>
            )}

            {selectedStudent.weaknesses && selectedStudent.weaknesses.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Weaknesses</p>
                <div className="flex flex-wrap gap-1">
                  {selectedStudent.weaknesses.map((w) => (
                    <Badge key={w} variant="secondary" className="text-xs font-normal bg-red-500/10 text-red-400 border-red-500/20">{w}</Badge>
                  ))}
                </div>
              </div>
            )}

            <Separator />

            <Button
              className="w-full rounded-md"
              onClick={() => {
                setSelectedStudent(null)
                router.push(`/test?tab=mock&student=${encodeURIComponent(selectedStudent.email)}`)
              }}
            >
              Create Targeted Test
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export default function TeacherPage() {
  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <TeacherDashboardContent />
    </RoleGuard>
  )
}