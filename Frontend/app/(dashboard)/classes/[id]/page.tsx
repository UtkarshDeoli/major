"use client"

import { useCallback, useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI, classSubjectAPI, classMaterialAPI } from "@/lib/api"
import { Loader2, Plus, Trash2, FileText, Layers, ClipboardList } from "lucide-react"
import RoleGuard from "@/components/auth/route-protection/role-guard"

interface Subject {
  id: string
  name: string
  icon?: string
}

interface Material {
  id: string
  name: string
  type: string
  doc_id?: string
  rag_indexed: boolean
}

export default function ClassDetailPage() {
  const { id } = useParams() as { id: string }
  const { toast } = useToast()
  const [cls, setCls] = useState<any>(null)
  const [subjects, setSubjects] = useState<Subject[]>([])
  const [activeSubject, setActiveSubject] = useState<string | null>(null)
  const [materials, setMaterials] = useState<Material[]>([])
  const [loading, setLoading] = useState(true)
  const [newSubject, setNewSubject] = useState("")
  const [uploading, setUploading] = useState(false)
  const [generating, setGenerating] = useState<string | null>(null)

  const loadClass = useCallback(async () => {
    try {
      const c = await classAPI.getClass(id)
      setCls(c)
    } catch (e) {
      toast({ title: "Couldn't load class", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, toast])

  const loadSubjects = useCallback(async () => {
    try {
      const res = await classSubjectAPI.list(id)
      const list = (res.subjects || []) as Subject[]
      setSubjects(list)
      setActiveSubject((prev) => (list.some((s) => s.id === prev) ? prev : list[0]?.id ?? null))
    } catch (e) {
      toast({ title: "Couldn't load subjects", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, toast])

  const loadMaterials = useCallback(async () => {
    if (!activeSubject) return
    try {
      const res = await classMaterialAPI.list(id, activeSubject)
      setMaterials((res.materials || []) as Material[])
    } catch (e) {
      toast({ title: "Couldn't load materials", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, activeSubject, toast])

  useEffect(() => { setLoading(true); Promise.all([loadClass(), loadSubjects()]).finally(() => setLoading(false)) }, [loadClass, loadSubjects])
  useEffect(() => { loadMaterials() }, [loadMaterials])

  const handleAddSubject = async () => {
    if (!newSubject.trim()) return
    try {
      await classSubjectAPI.create(id, { name: newSubject.trim() })
      setNewSubject("")
      loadSubjects()
      toast({ title: "Subject added" })
    } catch (e) {
      toast({ title: "Couldn't add subject", description: getErrorMessage(e), variant: "destructive" })
    }
  }

  const handleDeleteSubject = async (subjectId: string) => {
    try {
      await classSubjectAPI.delete(id, subjectId)
      loadSubjects()
      toast({ title: "Subject deleted" })
    } catch (e) {
      toast({ title: "Couldn't delete subject", description: getErrorMessage(e), variant: "destructive" })
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f || !activeSubject) return
    setUploading(true)
    try {
      await classMaterialAPI.upload(id, activeSubject, f)
      loadMaterials()
      toast({ title: "Material uploaded" })
    } catch (err) {
      toast({ title: "Upload failed", description: getErrorMessage(err), variant: "destructive" })
    } finally {
      setUploading(false)
    }
  }

  const handleGenerateFlashcards = async (material: Material) => {
    setGenerating(`flash-${material.id}`)
    try {
      const res = await classMaterialAPI.generateFlashcards(id, material.id)
      toast({ title: "Flashcards ready", description: `${res.card_count} cards generated.` })
    } catch (err) {
      toast({ title: "Flashcards failed", description: getErrorMessage(err), variant: "destructive" })
    } finally {
      setGenerating(null)
    }
  }

  const handleGenerateMockTest = async (material: Material) => {
    setGenerating(`mock-${material.id}`)
    try {
      const res = await classMaterialAPI.generateMockTest(id, material.id)
      toast({ title: "Mock test ready", description: `Test ID: ${res.test_id}` })
    } catch (err) {
      toast({ title: "Mock test failed", description: getErrorMessage(err), variant: "destructive" })
    } finally {
      setGenerating(null)
    }
  }

  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        {loading ? (
          <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
        ) : (
          <>
            <div>
              <h1 className="text-xl font-semibold">{cls?.name}</h1>
              {cls?.description && <p className="text-sm text-muted-foreground">{cls.description}</p>}
            </div>

            <div className="grid gap-6 lg:grid-cols-[240px,1fr]">
              <div className="space-y-4">
                <h2 className="text-sm font-semibold">Subjects</h2>
                <div className="flex gap-2">
                  <Input value={newSubject} onChange={(e) => setNewSubject(e.target.value)} placeholder="New subject" />
                  <Button size="icon" onClick={handleAddSubject} disabled={!newSubject.trim()}><Plus className="h-4 w-4" /></Button>
                </div>
                <div className="space-y-1">
                  {subjects.map((s) => (
                    <div
                      key={s.id}
                      className={`group flex items-center justify-between rounded-md px-3 py-2 text-sm ${activeSubject === s.id ? "bg-secondary text-foreground" : "hover:bg-muted/50 text-muted-foreground"}`}
                    >
                      <button
                        type="button"
                        onClick={() => setActiveSubject(s.id)}
                        className="flex-1 text-left"
                      >
                        {s.name}
                      </button>
                      <Button
                        type="button"
                        size="icon"
                        variant="ghost"
                        aria-label="Delete subject"
                        title="Delete subject"
                        className="h-6 w-6 opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                        onClick={() => handleDeleteSubject(s.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  ))}
                  {subjects.length === 0 && <p className="text-xs text-muted-foreground px-1">No subjects yet.</p>}
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold">Materials</h2>
                  <div className="flex items-center gap-2">
                    {uploading && <Loader2 className="h-4 w-4 animate-spin" />}
                    <Label htmlFor="cm-upload" className="cursor-pointer">
                      <div className="inline-flex items-center justify-center rounded-md text-sm font-medium h-9 px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90">Upload material</div>
                      <input id="cm-upload" type="file" className="hidden" onChange={handleUpload} disabled={!activeSubject || uploading} />
                    </Label>
                  </div>
                </div>

                {!activeSubject ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Select a subject to view/upload materials.</div>
                ) : materials.length === 0 ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No materials yet for this subject. Upload one.</div>
                ) : (
                  <div className="grid gap-3">
                    {materials.map((m) => (
                      <div key={m.id} className="rounded-md border p-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <div className="text-sm font-medium">{m.name}</div>
                            <div className="text-xs text-muted-foreground">{m.rag_indexed ? "AI-ready" : "Not indexed"}</div>
                          </div>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <Button size="sm" variant="outline" disabled={!m.rag_indexed || !!generating} onClick={() => handleGenerateFlashcards(m)}>
                            {generating === `flash-${m.id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Layers className="h-3.5 w-3.5 mr-1" />}
                            Flashcards
                          </Button>
                          <Button size="sm" variant="outline" disabled={!m.rag_indexed || !!generating} onClick={() => handleGenerateMockTest(m)}>
                            {generating === `mock-${m.id}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ClipboardList className="h-3.5 w-3.5 mr-1" />}
                            Mock Test
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </RoleGuard>
  )
}
