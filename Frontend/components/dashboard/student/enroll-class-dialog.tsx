"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog"
import { GraduationCap, Loader2 } from "lucide-react"

export function EnrollClassDialog() {
  const { toast } = useToast()
  const [open, setOpen] = useState(false)
  const [code, setCode] = useState("")
  const [preview, setPreview] = useState<{ name: string; teacher_name?: string; description?: string } | null>(null)
  const [loading, setLoading] = useState(false)
  const [enrolling, setEnrolling] = useState(false)

  const handlePreview = async () => {
    if (!code.trim()) return
    setLoading(true)
    try {
      const p = await classAPI.previewEnroll(code.trim().toUpperCase())
      setPreview(p as any)
    } catch (e) {
      setPreview(null)
      toast({ title: "Invalid code", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }

  const handleEnroll = async () => {
    setEnrolling(true)
    try {
      await classAPI.enroll(code.trim().toUpperCase())
      toast({ title: "Enrolled!", description: "You can now receive material from your teacher." })
      setOpen(false); setCode(""); setPreview(null)
    } catch (e) {
      toast({ title: "Enroll failed", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setEnrolling(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(o) => { setOpen(o); if (!o) { setCode(""); setPreview(null) } }}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]">
          <GraduationCap className="h-3.5 w-3.5 mr-1.5" />Join Class
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader><DialogTitle>Join a teacher's class</DialogTitle></DialogHeader>
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="enroll-code">Enroll code</Label>
            <div className="flex gap-2">
              <Input id="enroll-code" value={code} onChange={(e) => { setCode(e.target.value); setPreview(null) }} placeholder="6-character code" className="uppercase font-mono" />
              <Button variant="secondary" onClick={handlePreview} disabled={!code.trim() || loading}>{loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Check"}</Button>
            </div>
          </div>
          {preview && (
            <div className="rounded-md border p-3 text-sm space-y-1">
              <div className="font-medium">{preview.name}</div>
              {preview.teacher_name && <div className="text-xs text-muted-foreground">Teacher: {preview.teacher_name}</div>}
              {preview.description && <div className="text-xs text-muted-foreground">{preview.description}</div>}
            </div>
          )}
        </div>
        <DialogFooter>
          <Button disabled={!preview || enrolling} onClick={handleEnroll}>{enrolling ? <Loader2 className="h-4 w-4 animate-spin" /> : "Enroll"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}