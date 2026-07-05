"use client"

import { useCallback, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import { Copy, Loader2, Plus, Trash2, Users } from "lucide-react"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogTrigger,
} from "@/components/ui/dialog"

interface ClassItem {
  id: string
  name: string
  description?: string
  exam_preset?: string
  enroll_code: string
  student_count: number
}

interface ClassStudent {
  email: string
  name?: string
  tests_taken: number
  average_score: number
  last_active_at?: string
}

interface ClassDetail extends ClassItem {
  students: ClassStudent[]
}

export function TeacherClassesPanel() {
  const { toast } = useToast()
  const [classes, setClasses] = useState<ClassItem[]>([])
  const [loading, setLoading] = useState(true)
  const [openDetail, setOpenDetail] = useState<string | null>(null)
  const [detail, setDetail] = useState<ClassDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [newName, setNewName] = useState("")
  const [newDesc, setNewDesc] = useState("")
  const [newPreset, setNewPreset] = useState("")
  const [isCreating, setIsCreating] = useState(false)

  const fetchClasses = useCallback(async () => {
    setLoading(true)
    try {
      const list = await classAPI.listClasses()
      setClasses((list || []) as ClassItem[])
    } catch (e) {
      toast({ title: "Couldn't load classes", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { fetchClasses() }, [fetchClasses])

  const openClassDetail = async (id: string) => {
    setOpenDetail(id)
    setLoadingDetail(true)
    try {
      const d = await classAPI.getClass(id)
      setDetail(d as ClassDetail)
    } catch (e) {
      toast({ title: "Couldn't load class", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoadingDetail(false)
    }
  }

  const handleCreate = async () => {
    if (!newName.trim()) return
    setIsCreating(true)
    try {
      await classAPI.createClass({ name: newName.trim(), description: newDesc.trim() || undefined, exam_preset: newPreset.trim() || undefined })
      setCreateOpen(false); setNewName(""); setNewDesc(""); setNewPreset("")
      fetchClasses()
      toast({ title: "Class created", description: "Share the enroll code with students." })
    } catch (e) {
      toast({ title: "Couldn't create class", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setIsCreating(false)
    }
  }

  const handleRemoveStudent = async (classId: string, email: string) => {
    try {
      await classAPI.removeStudent(classId, email)
      setDetail((prev) => prev ? { ...prev, students: prev.students.filter((s) => s.email !== email) } : prev)
      fetchClasses()
    } catch (e) {
      toast({ title: "Couldn't remove student", description: getErrorMessage(e), variant: "destructive" })
    }
  }

  const copyCode = (code: string) => {
    navigator.clipboard?.writeText(code)
    toast({ title: "Enroll code copied", description: code })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Classes</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Group students into batches and share an enroll code.</p>
        </div>
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="rounded-md h-8 text-[13px]"><Plus className="h-3.5 w-3.5 mr-1.5" />New Class</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader><DialogTitle>New Class</DialogTitle></DialogHeader>
            <div className="space-y-3">
              <div className="space-y-1.5"><Label htmlFor="cn">Name</Label><Input id="cn" value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. JEE 2026 Batch" /></div>
              <div className="space-y-1.5"><Label htmlFor="cd">Description</Label><Input id="cd" value={newDesc} onChange={(e) => setNewDesc(e.target.value)} placeholder="Optional" /></div>
              <div className="space-y-1.5"><Label htmlFor="cp">Exam preset</Label><Input id="cp" value={newPreset} onChange={(e) => setNewPreset(e.target.value)} placeholder="e.g. jee-mains" /></div>
            </div>
            <DialogFooter>
              <Button disabled={!newName.trim() || isCreating} onClick={handleCreate}>{isCreating ? <Loader2 className="h-4 w-4 animate-spin" /> : "Create"}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {loading ? (
        <div className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
      ) : classes.length === 0 ? (
        <div className="rounded-md border bg-card p-6 text-center text-sm text-muted-foreground">
          No classes yet. Create one to group your students.
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {classes.map((c) => (
            <div key={c.id} className="rounded-md border bg-card p-3 space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium">{c.name}</h3>
                <span className="text-[11px] text-muted-foreground flex items-center gap-1"><Users className="h-3 w-3" />{c.student_count}</span>
              </div>
              {c.description && <p className="text-xs text-muted-foreground">{c.description}</p>}
              <div className="flex items-center gap-2">
                <code className="rounded bg-secondary px-2 py-1 text-[12px] tracking-wider font-mono">{c.enroll_code}</code>
                <Button variant="ghost" size="icon" className="h-6 w-6" title="Copy enroll code" onClick={() => copyCode(c.enroll_code)}><Copy className="h-3 w-3" /></Button>
                <Button variant="outline" size="sm" className="ml-auto h-7 text-[12px]" onClick={() => openClassDetail(c.id)}>Roster</Button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Roster dialog */}
      <Dialog open={!!openDetail} onOpenChange={(o) => !o && setOpenDetail(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader><DialogTitle>{detail?.name ?? "Class"} — Roster</DialogTitle></DialogHeader>
          {loadingDetail ? (
            <div className="flex justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
          ) : detail && detail.students.length > 0 ? (
            <div className="space-y-2">
              {detail.students.map((s) => (
                <div key={s.email} className="flex items-center gap-2 rounded-md border p-2">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{s.name || s.email}</div>
                    <div className="text-[11px] text-muted-foreground">{s.tests_taken} tests · avg {s.average_score}%</div>
                  </div>
                  <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive" onClick={() => handleRemoveStudent(detail.id, s.email)}><Trash2 className="h-3.5 w-3.5" /></Button>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No students enrolled yet. Share the enroll code <code className="font-mono">{detail?.enroll_code}</code>.</p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}