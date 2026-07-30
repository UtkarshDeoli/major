"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { Loader2, Users, Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { StudentRosterList, RosterStudent } from "@/components/dashboard/teacher/student-roster-list"

interface ClassOption {
  id: string
  name: string
  enroll_code: string
  description?: string
}

export default function StudentsPage() {
  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <StudentsPageContent />
    </RoleGuard>
  )
}

function StudentsPageContent() {
  const router = useRouter()
  const { toast } = useToast()
  const [classes, setClasses] = useState<ClassOption[]>([])
  const [selectedClassId, setSelectedClassId] = useState<string>("__all__")
  const [students, setStudents] = useState<RosterStudent[]>([])
  const [loading, setLoading] = useState(true)

  const fetchClasses = useCallback(async () => {
    try {
      const list = await classAPI.listClasses()
      setClasses((list || []) as ClassOption[])
    } catch (e) {
      toast({ title: "Couldn't load classes", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [toast])

  const fetchStudents = useCallback(async () => {
    setLoading(true)
    try {
      if (selectedClassId === "__all__") {
        const res = await classAPI.getTeacherStudents()
        setStudents((res.students || []) as RosterStudent[])
      } else {
        const res = await classAPI.getClassStudentsAnalytics(selectedClassId)
        setStudents((res.students || []) as RosterStudent[])
      }
    } catch (e) {
      toast({ title: "Couldn't load students", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }, [selectedClassId, toast])

  useEffect(() => { fetchClasses() }, [fetchClasses])
  useEffect(() => { fetchStudents() }, [fetchStudents])

  const selectedClass = useMemo(() => classes.find((c) => c.id === selectedClassId), [classes, selectedClassId])

  const copyCode = (code: string) => {
    navigator.clipboard?.writeText(code)
    toast({ title: "Enroll code copied", description: code })
  }

  const handleSelect = (student: RosterStudent) => {
    router.push(`/students/${encodeURIComponent(student.email)}`)
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Students</h1>
        <p className="text-sm text-muted-foreground">View and monitor students by class.</p>
      </div>

      <div className="flex items-center gap-3">
        <Select value={selectedClassId} onValueChange={setSelectedClassId}>
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select a class" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All my classes</SelectItem>
            {classes.map((c) => (
              <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {selectedClass && (
        <div className="rounded-lg border bg-card p-4 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="font-medium">{selectedClass.name}</h2>
            <span className="text-xs text-muted-foreground flex items-center gap-1"><Users className="h-3 w-3" />{students.length}</span>
          </div>
          {selectedClass.description && <p className="text-xs text-muted-foreground">{selectedClass.description}</p>}
          <div className="flex items-center gap-2">
            <code className="rounded bg-secondary px-2 py-1 text-xs tracking-wider font-mono">{selectedClass.enroll_code}</code>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => copyCode(selectedClass.enroll_code)}><Copy className="h-3.5 w-3.5" /></Button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      ) : (
        <StudentRosterList
          students={students}
          onSelect={handleSelect}
          emptyMessage={selectedClassId === "__all__" ? "No students in any of your classes yet." : "No students in this class yet."}
        />
      )}
    </div>
  )
}
