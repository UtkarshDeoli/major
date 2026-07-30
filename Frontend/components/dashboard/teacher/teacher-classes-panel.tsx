"use client"

import { useCallback, useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import { Copy, Loader2, Plus, Users } from "lucide-react"
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

export function TeacherClassesPanel() {
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
                <Button variant="outline" size="sm" className="ml-auto h-7 text-[12px]" asChild>
                  <Link href={`/classes/${c.id}`}>Open</Link>
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  )
}