"use client"

import { useState, useEffect, useMemo } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import {
  BookOpen, Users, TrendingUp, Target, Activity,
  Search, ArrowUpDown, BarChart3, ClipboardList,
} from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { useToast } from "@/hooks/use-toast"
import { teacherAPI, mockTestAPI, classAPI } from "@/lib/api"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { StatsCard } from "@/components/dashboard/analytics/stats-card"
import { ProgressRing } from "@/components/dashboard/analytics/progress-ring"
import { ClassChart } from "@/components/dashboard/analytics/class-chart"
import { TeacherClassesPanel } from "@/components/dashboard/teacher/teacher-classes-panel"
import { TeacherAlertsPanel } from "@/components/dashboard/teacher/teacher-alerts-panel"
import { StudentRosterList, RosterStudent } from "@/components/dashboard/teacher/student-roster-list"
import { X } from "lucide-react"

interface StudentAnalytics {
  email: string
  name?: string
  tests_taken: number
  average_score: number
  last_active_at?: string
  strengths: string[]
  weaknesses: string[]
  class_ids?: string[]
}

interface TeacherAnalytics {
  total_students: number
  active_students: number
  total_tests_taken: number
  class_average: number
  student_analytics: StudentAnalytics[]
}

type SortKey = "name" | "score" | "tests"
type SortDir = "asc" | "desc"

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.06 } } },
  item: {
    initial: { opacity: 0, y: 14 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
  },
}

function StudentDetailPanel({
  student,
  onClose,
  onCreateTest,
}: {
  student: StudentAnalytics
  onClose: () => void
  onCreateTest: (email: string) => void
}) {
  return (
    <>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="absolute inset-0 bg-black/50 z-40"
        onClick={onClose}
      />
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 200 }}
        className="absolute right-0 top-0 bottom-0 w-96 bg-background border-l z-50 overflow-y-auto"
      >
        <div className="p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">{student.name || student.email.split("@")[0]}</h3>
            <Button variant="ghost" size="icon" className="h-7 w-7 rounded-lg" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground font-mono -mt-4">{student.email}</p>

          <div className="grid grid-cols-3 gap-3">
            <div className="flex flex-col items-center">
              <ProgressRing value={student.average_score} label="Score" suffix="%" size={72} strokeWidth={5} />
            </div>
            <div className="rounded-xl border bg-card p-3 text-center">
              <p className="text-xl font-semibold tabular-nums">{student.tests_taken}</p>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mt-0.5">Tests</p>
            </div>
            <div className="rounded-xl border bg-card p-3 text-center">
              <p className="text-xl font-semibold tabular-nums">
                {student.last_active_at
                  ? new Date(student.last_active_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })
                  : "—"}
              </p>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mt-0.5">Active</p>
            </div>
          </div>

          {student.strengths && student.strengths.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Strengths</p>
              <div className="flex flex-wrap gap-1.5">
                {student.strengths.map((s) => (
                  <Badge key={s} variant="secondary" className="text-xs font-normal bg-green-500/10 text-green-400 border-green-500/20 rounded-lg">
                    {s}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {student.weaknesses && student.weaknesses.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Weaknesses</p>
              <div className="flex flex-wrap gap-1.5">
                {student.weaknesses.map((w) => (
                  <Badge key={w} variant="secondary" className="text-xs font-normal bg-red-500/10 text-red-400 border-red-500/20 rounded-lg">
                    {w}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <Separator />

          <Button
            className="w-full rounded-lg"
            onClick={() => onCreateTest(student.email)}
          >
            <ClipboardList className="h-4 w-4 mr-2" />
            Create Targeted Test
          </Button>
        </div>
      </motion.div>
    </>
  )
}

function TeacherDashboardContent() {
  const router = useRouter()
  const { toast } = useToast()
  const [students, setStudents] = useState<RosterStudent[]>([])
  const [classes, setClasses] = useState<Array<{ id: string; name: string }>>([])
  const [classFilter, setClassFilter] = useState<string>("__all__")
  const [analytics, setAnalytics] = useState<TeacherAnalytics | null>(null)
  const [selectedStudentEmail, setSelectedStudentEmail] = useState<string>("")
  const [selectedStudent, setSelectedStudent] = useState<StudentAnalytics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortKey, setSortKey] = useState<SortKey>("name")
  const [sortDir, setSortDir] = useState<SortDir>("asc")

  const [teacherTests, setTeacherTests] = useState<Array<{ test_id: string; title: string; assigned_to?: string; created_at: string }>>([])
  const [isLoadingTests, setIsLoadingTests] = useState(false)
  const [assignDialogOpen, setAssignDialogOpen] = useState(false)
  const [selectedTestId, setSelectedTestId] = useState<string>("")
  const [assignStudentEmail, setAssignStudentEmail] = useState<string>("")
  const [isAssigning, setIsAssigning] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      setIsLoadingTests(true)
      try {
        const [classesData, studentsData, testsData] = await Promise.all([
          classAPI.listClasses(),
          classAPI.getTeacherStudents(),
          mockTestAPI.listMockTests().catch(() => []),
        ])
        setClasses((classesData || []) as Array<{ id: string; name: string }>)
        const list = (studentsData?.students || []) as RosterStudent[]
        setStudents(list)
        setAnalytics({
          total_students: list.length,
          active_students: list.filter((s) => s.tests_taken > 0).length,
          total_tests_taken: list.reduce((acc, s) => acc + s.tests_taken, 0),
          class_average: list.length > 0
            ? Math.round(list.reduce((acc, s) => acc + s.average_score, 0) / list.length)
            : 0,
          student_analytics: list as StudentAnalytics[],
        } as TeacherAnalytics)
        setTeacherTests((testsData as any[]) || [])
      } catch (error: any) {
        console.error("Error fetching teacher data:", error)
        toast({
          title: "Error",
          description: error.response?.data?.detail || "Failed to load teacher dashboard data.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
        setIsLoadingTests(false)
      }
    }
    fetchData()
  }, [toast])

  const handleCreateTest = () => {
    const studentParam =
      selectedStudentEmail && selectedStudentEmail !== "__me__" ? selectedStudentEmail : ""
    router.push(`/mock-tests${studentParam ? `?student=${encodeURIComponent(studentParam)}` : ""}`)
  }

  const handleCreateTestForStudent = (email: string) => {
    setSelectedStudent(null)
    router.push(`/mock-tests?student=${encodeURIComponent(email)}`)
  }

  const handleAssignTest = async () => {
    if (!selectedTestId || !assignStudentEmail) return
    setIsAssigning(true)
    try {
      await teacherAPI.assignMockTest(selectedTestId, assignStudentEmail)
      toast({ title: "Test assigned", description: "The test has been assigned to the student." })
      setTeacherTests((prev) =>
        prev.map((t) => (t.test_id === selectedTestId ? { ...t, assigned_to: assignStudentEmail } : t))
      )
      setAssignDialogOpen(false)
      setSelectedTestId("")
      setAssignStudentEmail("")
      const fresh = await classAPI.getTeacherStudents()
      const list = (fresh?.students || []) as RosterStudent[]
      setStudents(list)
      setAnalytics({
        total_students: list.length,
        active_students: list.filter((s) => s.tests_taken > 0).length,
        total_tests_taken: list.reduce((acc, s) => acc + s.tests_taken, 0),
        class_average: list.length > 0
          ? Math.round(list.reduce((acc, s) => acc + s.average_score, 0) / list.length)
          : 0,
        student_analytics: list as StudentAnalytics[],
      } as TeacherAnalytics)
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to assign test.",
        variant: "destructive",
      })
    } finally {
      setIsAssigning(false)
    }
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"))
    } else {
      setSortKey(key)
      setSortDir("asc")
    }
  }

  const filteredStudents = useMemo(() => {
    let list = [...students]
    if (classFilter !== "__all__") {
      list = list.filter((s) => (s.class_ids || []).includes(classFilter))
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      list = list.filter((s) => (s.name || s.email).toLowerCase().includes(q))
    }
    return list.sort((a, b) => {
      let cmp = 0
      if (sortKey === "name") cmp = (a.name || a.email).localeCompare(b.name || b.email)
      else if (sortKey === "score") cmp = a.average_score - b.average_score
      else if (sortKey === "tests") cmp = a.tests_taken - b.tests_taken
      return sortDir === "asc" ? cmp : -cmp
    })
  }, [students, searchQuery, sortKey, sortDir, classFilter])

  const statCards = [
    { label: "Total Students", value: analytics?.total_students ?? 0, icon: Users, color: "hsl(221, 83%, 53%)", subtitle: `${analytics?.active_students ?? 0} active` },
    { label: "Active Students", value: analytics?.active_students ?? 0, icon: Activity, color: "hsl(160, 84%, 39%)" },
    { label: "Tests Taken", value: analytics?.total_tests_taken ?? 0, icon: Target, color: "hsl(237, 56%, 60%)" },
    { label: "Class Average", value: analytics ? `${analytics.class_average}%` : "—", icon: TrendingUp, color: "hsl(280, 56%, 60%)" },
  ]

  const classChartData = useMemo(() => {
    if (!analytics?.student_analytics) return []
    return analytics.student_analytics.map((s) => ({
      name: (s.name || s.email.split("@")[0]).split(" ")[0],
      score: Math.round(s.average_score),
    }))
  }, [analytics])

  const quickActions = [
    { icon: ClipboardList, label: "Create Test", desc: "Build a targeted assessment.", href: "/mock-tests" },
    { icon: BarChart3, label: "Analytics", desc: "Detailed performance insights.", href: "/analytics" },
    { icon: Users, label: "Students", desc: "Browse students across classes.", href: "/students" },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Teacher Dashboard.</h1>
          <p className="text-sm text-muted-foreground mt-1">Monitor student performance and create targeted tests.</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedStudentEmail} onValueChange={setSelectedStudentEmail}>
            <SelectTrigger className="w-48 h-8 rounded-lg text-[13px]">
              <SelectValue placeholder="Assign to..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__me__">Assign to me</SelectItem>
              {students.map((s) => (
                <SelectItem key={s.email} value={s.email}>
                  {s.name || s.email}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" className="rounded-lg h-8 text-[13px]" onClick={handleCreateTest}>
            <BookOpen className="h-3.5 w-3.5 mr-1.5" />
            Create Test
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
        {statCards.map((stat, i) => (
          <StatsCard key={stat.label} {...stat} delay={0.05 + i * 0.05} isLoading={isLoading} />
        ))}
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.25 }}
        className="space-y-3"
      >
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Quick Actions</h2>
        <div className="grid gap-3 md:grid-cols-3">
          {quickActions.map((action, i) => (
            <motion.div
              key={action.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.3 + i * 0.05 }}
              whileHover={{ y: -3, transition: { duration: 0.15 } }}
            >
              <a
                href={action.href}
                className="block rounded-xl border bg-card p-5 hover:shadow-md hover:border-primary/30 transition-all duration-200 group relative overflow-hidden"
              >
                <action.icon className="absolute -right-5 -bottom-5 h-24 w-24 text-primary opacity-[0.06] group-hover:opacity-[0.1] transition-opacity" />
                <div className="relative">
                  <p className="text-sm font-medium">{action.label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{action.desc}</p>
                </div>
              </a>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Alerts & Insights */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.3 }}
      >
        <TeacherAlertsPanel />
      </motion.div>

      {/* Class Performance Chart */}
      {!isLoading && classChartData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.35 }}
        >
          <ClassChart data={classChartData} />
        </motion.div>
      )}

      {/* Classes / Batches */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <TeacherClassesPanel />
      </motion.div>

      <div className="flex items-center justify-between mt-4">
        <p className="text-sm text-muted-foreground">View all classes, subjects, materials, and tests.</p>
        <Button asChild variant="outline" size="sm"><Link href="/classes">Manage classes →</Link></Button>
      </div>

      {/* Assign Existing Test */}
      {!isLoadingTests && teacherTests.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.4 }}
          className="space-y-3"
        >
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Assign Existing Test</h2>
            <Button
              variant="outline"
              size="sm"
              className="rounded-lg h-8 text-[13px]"
              onClick={() => setAssignDialogOpen(true)}
            >
              <ClipboardList className="h-3.5 w-3.5 mr-1.5" />
              Assign Test
            </Button>
          </div>

          <div className="grid gap-2 md:grid-cols-2">
            {teacherTests
              .filter((t) => !t.assigned_to)
              .slice(0, 4)
              .map((test, i) => (
                <motion.div
                  key={test.test_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.45 + i * 0.05 }}
                  className="rounded-xl border bg-card p-4 flex items-center justify-between hover:border-primary/30 transition-colors"
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{test.title || "Untitled Test"}</p>
                    <p className="text-xs text-muted-foreground">
                      Created {test.created_at ? new Date(test.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "—"}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="rounded-lg h-7 text-xs shrink-0 ml-3"
                    onClick={() => {
                      setSelectedTestId(test.test_id)
                      setAssignDialogOpen(true)
                    }}
                  >
                    Assign
                  </Button>
                </motion.div>
              ))}
          </div>
        </motion.div>
      )}

      {/* Students */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.45 }}
        className="space-y-3"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Students</h2>
          <div className="flex items-center gap-2">
            <Select value={classFilter} onValueChange={setClassFilter}>
              <SelectTrigger className="w-48 h-8 rounded-lg text-[13px]">
                <SelectValue placeholder="Filter by class" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All classes</SelectItem>
                {classes.map((c) => (
                  <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder="Search students..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 w-48 rounded-lg pl-8 text-[13px]"
              />
            </div>
            <Button
              variant="outline"
              size="sm"
              className="rounded-lg h-8 text-[13px] gap-1"
              onClick={() => handleSort(sortKey)}
            >
              <ArrowUpDown className="h-3 w-3" />
              {sortKey === "name" ? "Name" : sortKey === "score" ? "Score" : "Tests"}
              {sortDir === "asc" ? "↑" : "↓"}
            </Button>
          </div>
        </div>

        {isLoading ? (
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="rounded-xl border bg-card p-4 space-y-3 animate-pulse">
                <div className="h-4 w-24 bg-muted rounded" />
                <div className="h-3 w-36 bg-muted rounded" />
                <div className="h-2 w-full bg-muted rounded" />
              </div>
            ))}
          </div>
        ) : (
          <StudentRosterList
            students={filteredStudents}
            onSelect={(s) => router.push(`/students/${encodeURIComponent(s.email)}`)}
            emptyMessage="No students in your classes yet. Create a class and share the enroll code."
          />
        )}
      </motion.div>

      {/* Student Detail Panel */}
      <AnimatePresence>
        {selectedStudent && (
          <div className="fixed inset-0 z-50">
            <StudentDetailPanel
              student={selectedStudent}
              onClose={() => setSelectedStudent(null)}
              onCreateTest={handleCreateTestForStudent}
            />
          </div>
        )}
      </AnimatePresence>

      {/* Assign Existing Test Dialog */}
      <Dialog open={assignDialogOpen} onOpenChange={setAssignDialogOpen}>
        <DialogContent className="sm:max-w-md rounded-xl">
          <DialogHeader>
            <DialogTitle>Assign Existing Test</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Select Test</label>
              <Select value={selectedTestId} onValueChange={setSelectedTestId}>
                <SelectTrigger className="rounded-lg">
                  <SelectValue placeholder="Choose a test..." />
                </SelectTrigger>
                <SelectContent>
                  {teacherTests
                    .filter((t) => !t.assigned_to)
                    .map((t) => (
                      <SelectItem key={t.test_id} value={t.test_id}>
                        {t.title || "Untitled Test"}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Assign to Student</label>
              <Select value={assignStudentEmail} onValueChange={setAssignStudentEmail}>
                <SelectTrigger className="rounded-lg">
                  <SelectValue placeholder="Choose a student..." />
                </SelectTrigger>
                <SelectContent>
                  {students.map((s) => (
                    <SelectItem key={s.email} value={s.email}>
                      {s.name || s.email}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" className="rounded-lg" onClick={() => setAssignDialogOpen(false)}>Cancel</Button>
            <Button
              className="rounded-lg"
              onClick={handleAssignTest}
              disabled={!selectedTestId || !assignStudentEmail || isAssigning}
            >
              {isAssigning ? "Assigning..." : "Assign Test"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
