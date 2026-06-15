"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { BookOpen, Users, TrendingUp, Target, Activity, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { teacherAPI } from "@/lib/api"
import RoleGuard from "@/components/auth/route-protection/role-guard"

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

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto h-16 flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => router.push("/dashboard")}>
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <h1 className="text-xl font-bold">Teacher Dashboard</h1>
          </div>
          <div className="flex items-center gap-2">
            <select
              className="border rounded px-2 py-1 text-sm bg-background"
              value={selectedStudentEmail ?? ""}
              onChange={(e) => setSelectedStudentEmail(e.target.value || null)}
            >
              <option value="">Assign to me</option>
              {students.map((s) => (
                <option key={s.email} value={s.email}>
                  {s.name || s.email}
                </option>
              ))}
            </select>
            <Button onClick={handleCreateTest}>
              <BookOpen className="h-4 w-4 mr-2" />
              Create Test
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-primary border-t-transparent mx-auto mb-4" />
              <p className="text-muted-foreground">Loading dashboard...</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Total Students</CardDescription>
                  <CardTitle className="text-3xl flex items-center gap-2">
                    <Users className="h-6 w-6 text-primary" />
                    {analytics?.total_students ?? 0}
                  </CardTitle>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Active Students</CardDescription>
                  <CardTitle className="text-3xl flex items-center gap-2">
                    <Activity className="h-6 w-6 text-green-500" />
                    {analytics?.active_students ?? 0}
                  </CardTitle>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Tests Taken</CardDescription>
                  <CardTitle className="text-3xl flex items-center gap-2">
                    <Target className="h-6 w-6 text-blue-500" />
                    {analytics?.total_tests_taken ?? 0}
                  </CardTitle>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Class Average</CardDescription>
                  <CardTitle className="text-3xl flex items-center gap-2">
                    <TrendingUp className="h-6 w-6 text-purple-500" />
                    {analytics?.class_average ?? 0}%
                  </CardTitle>
                </CardHeader>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5" />
                  Students
                </CardTitle>
                <CardDescription>
                  {students.length === 0
                    ? "No students linked yet. Manage students to see them here."
                    : `View performance and weak topics for ${students.length} student${students.length === 1 ? "" : "s"}.`}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analytics?.student_analytics && analytics.student_analytics.length > 0 ? (
                  <div className="space-y-4">
                    {analytics.student_analytics.map((student) => (
                      <div key={student.email}>
                        <div className="flex items-start justify-between">
                          <div>
                            <p className="font-medium">{student.name || student.email}</p>
                            <p className="text-sm text-muted-foreground">{student.email}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">
                              {student.tests_taken} test{student.tests_taken === 1 ? "" : "s"} taken
                            </p>
                            <p className="text-sm text-muted-foreground">
                              Avg: {student.average_score}%
                            </p>
                          </div>
                        </div>
                        {student.weaknesses && student.weaknesses.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {student.weaknesses.map((topic) => (
                              <Badge
                                key={topic}
                                variant="secondary"
                                className="text-xs bg-red-100 text-red-700 hover:bg-red-100"
                              >
                                {topic}
                              </Badge>
                            ))}
                          </div>
                        )}
                        <Separator className="mt-4" />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Users className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">No student analytics available.</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </main>
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
