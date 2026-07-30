"use client"

import { useCallback, useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import { useAuth } from "@/lib/context/auth-context"
import { Copy, Loader2, Plus, Users } from "lucide-react"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogTrigger,
} from "@/components/ui/dialog"
import RoleGuard from "@/components/auth/route-protection/role-guard"

interface ClassItem {
  id: string
  name: string
  description?: string
  exam_preset?: string
  enroll_code: string
  student_count: number
}

export default function ClassesPage() {
  const { user } = useAuth()
  if (!user) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      </div>
    )
  }
  return user.role === "teacher" ? <TeacherClassesView /> : <StudentClassesView />
}

function TeacherClassesView() {
  const { toast } = useToast()
  const [classes, setClasses] = useState<ClassItem[]>([])
  const [loading, setLoading] = useState(true)
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

  const copyCode = (code: string) => {
    navigator.clipboard?.writeText(code)
    toast({ title: "Enroll code copied", description: code })
  }

  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Classes</h1>
            <p className="text-sm text-muted-foreground">Manage your batches, subjects, materials, and tests.</p>
          </div>
          <Dialog open={createOpen} onOpenChange={setCreateOpen}>
            <DialogTrigger asChild>
              <Button><Plus className="h-4 w-4 mr-2" />New Class</Button>
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
          <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
        ) : classes.length === 0 ? (
          <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No classes yet. Create one to group your students.</div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {classes.map((c) => (
              <div key={c.id} className="rounded-lg border bg-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">{c.name}</h3>
                  <span className="text-xs text-muted-foreground flex items-center gap-1"><Users className="h-3 w-3" />{c.student_count}</span>
                </div>
                {c.description && <p className="text-xs text-muted-foreground">{c.description}</p>}
                <div className="flex items-center gap-2">
                  <code className="rounded bg-secondary px-2 py-1 text-xs tracking-wider font-mono">{c.enroll_code}</code>
                  <Button variant="ghost" size="icon" className="h-7 w-7" title="Copy enroll code" onClick={() => copyCode(c.enroll_code)}><Copy className="h-3.5 w-3.5" /></Button>
                </div>
                <Button asChild variant="outline" className="w-full">
                  <Link href={`/classes/${c.id}`}>Open class</Link>
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </RoleGuard>
  )
}

function StudentClassesView() {
  const { toast } = useToast()
  const [classes, setClasses] = useState<ClassItem[]>([])
  const [loading, setLoading] = useState(true)
  const [code, setCode] = useState("")
  const [joining, setJoining] = useState(false)

  const fetchClasses = useCallback(async () => {
    setLoading(true)
    try {
      const res = await classAPI.listMyClasses()
      setClasses((res.classes || []) as ClassItem[])
    } catch (e) {
      toast({ title: "Couldn't load classes", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }, [toast])

  useEffect(() => { fetchClasses() }, [fetchClasses])

  const handleJoin = async () => {
    if (!code.trim()) return
    setJoining(true)
    try {
      await classAPI.joinClass(code.trim())
      setCode("")
      fetchClasses()
      toast({ title: "Joined class" })
    } catch (e) {
      toast({ title: "Couldn't join class", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setJoining(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold">My Classes</h1>
        <p className="text-sm text-muted-foreground">Join a class with an enroll code and view your study content.</p>
      </div>
      <div className="flex gap-2 max-w-md">
        <Input value={code} onChange={(e) => setCode(e.target.value)} placeholder="Enter enroll code" />
        <Button onClick={handleJoin} disabled={!code.trim() || joining}>{joining ? <Loader2 className="h-4 w-4 animate-spin" /> : "Join"}</Button>
      </div>
      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      ) : classes.length === 0 ? (
        <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">You haven&apos;t joined any classes yet. Enter an enroll code to get started.</div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {classes.map((c) => (
            <div key={c.id} className="rounded-lg border bg-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">{c.name}</h3>
                <span className="text-xs text-muted-foreground flex items-center gap-1"><Users className="h-3 w-3" />{c.student_count}</span>
              </div>
              {c.description && <p className="text-xs text-muted-foreground">{c.description}</p>}
              <Button asChild variant="outline" className="w-full">
                <Link href={`/classes/${c.id}`}>Open class</Link>
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
