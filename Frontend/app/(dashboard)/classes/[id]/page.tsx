"use client"

import { useCallback, useEffect, useState } from "react"
import Link from "next/link"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI, classSubjectAPI, classMaterialAPI } from "@/lib/api"
import { useAuth } from "@/lib/context/auth-context"
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
  const { user } = useAuth()
  if (!user) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      </div>
    )
  }
  return user.role === "teacher" ? <TeacherClassDetailPage id={id} /> : <StudentClassDetailPage id={id} />
}

function TeacherClassDetailPage({ id }: { id: string }) {
  const { toast } = useToast()
  const [cls, setCls] = useState<any>(null)
  const [subjects, setSubjects] = useState<Subject[]>([])
  const [activeSubject, setActiveSubject] = useState<string | null>(null)
  const [materials, setMaterials] = useState<Material[]>([])
  const [loading, setLoading] = useState(true)
  const [newSubject, setNewSubject] = useState("")
  const [uploading, setUploading] = useState(false)
  const [generating, setGenerating] = useState<string | null>(null)
  const [students, setStudents] = useState<any[]>([])
  const [tests, setTests] = useState<any[]>([])

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

  const loadStudents = useCallback(async () => {
    try {
      const res = await classAPI.getClassStudents(id)
      setStudents(res.students || [])
    } catch (e) {
      toast({ title: "Couldn't load students", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, toast])

  const loadTests = useCallback(async () => {
    try {
      const res = await classAPI.getClassTests(id)
      setTests(res.tests || [])
    } catch (e) {
      toast({ title: "Couldn't load tests", description: getErrorMessage(e), variant: "destructive" })
    }
  }, [id, toast])

  useEffect(() => { setLoading(true); Promise.all([loadClass(), loadSubjects(), loadStudents(), loadTests()]).finally(() => setLoading(false)) }, [loadClass, loadSubjects, loadStudents, loadTests])
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

            <Tabs defaultValue="subjects">
              <TabsList>
                <TabsTrigger value="subjects">Subjects & Materials</TabsTrigger>
                <TabsTrigger value="tests">Tests</TabsTrigger>
                <TabsTrigger value="students">Students</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
              </TabsList>

              <TabsContent value="subjects">
                <div className="grid gap-6 lg:grid-cols-[240px,1fr] pt-4">
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
              </TabsContent>

              <TabsContent value="tests">
                <div className="space-y-3 pt-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold">Class Tests</h2>
                    <span className="text-xs text-muted-foreground">{tests.length} generated</span>
                  </div>
                  {tests.length === 0 ? (
                    <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No mock tests yet. Generate one from a material.</div>
                  ) : (
                    <div className="grid gap-2">
                      {tests.map((t) => (
                        <Link key={t.test_id || t.id} href={`/mock-tests/${t.test_id || t.id}`} className="rounded-md border p-3 hover:bg-muted/50 flex items-center justify-between">
                          <div className="text-sm font-medium">{t.title}</div>
                          <div className="text-xs text-muted-foreground">{t.total_marks} marks</div>
                        </Link>
                      ))}
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="students">
                <div className="space-y-3 pt-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold">Roster</h2>
                    <span className="text-xs text-muted-foreground">{students.length} enrolled</span>
                  </div>
                  {students.length === 0 ? (
                    <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No students enrolled yet. Share the enroll code.</div>
                  ) : (
                    <div className="grid gap-2">
                      {students.map((s) => (
                        <div key={s.email || s.id} className="rounded-md border p-3 flex items-center justify-between">
                          <div>
                            <div className="text-sm font-medium">{s.name || s.email}</div>
                            <div className="text-xs text-muted-foreground">{s.email}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="analytics">
                <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Analytics tab — class performance will appear here.</div>
              </TabsContent>
            </Tabs>
          </>
        )}
      </div>
    </RoleGuard>
  )
}

function StudentClassDetailPage({ id }: { id: string }) {
  const { toast } = useToast()
  const [content, setContent] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [activeSubject, setActiveSubject] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    classAPI.getClassContent(id)
      .then((c) => { setContent(c); setActiveSubject(c.subjects?.[0]?.id ?? null) })
      .catch((e) => toast({ title: "Couldn't load class", description: getErrorMessage(e), variant: "destructive" }))
      .finally(() => setLoading(false))
  }, [id, toast])

  const decksForSubject = (subjectId: string) => (content?.decks || []).filter((d: any) => d.class_subject_id === subjectId)
  const testsForSubject = (subjectId: string) => (content?.tests || []).filter((t: any) => t.class_subject_id === subjectId)

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      ) : (
        <>
          <div>
            <h1 className="text-xl font-semibold">{content?.class_name || "Class"}</h1>
          </div>
          <Tabs defaultValue="subjects">
            <TabsList>
              <TabsTrigger value="subjects">Subjects</TabsTrigger>
              <TabsTrigger value="flashcards">Flashcards</TabsTrigger>
              <TabsTrigger value="mock-tests">Mock Tests</TabsTrigger>
            </TabsList>

            <TabsContent value="subjects">
              <div className="grid gap-6 lg:grid-cols-[240px,1fr] pt-4">
                <div className="space-y-4">
                  <h2 className="text-sm font-semibold">Subjects</h2>
                  <div className="space-y-1">
                    {content?.subjects?.map((s: any) => (
                      <button
                        key={s.id}
                        onClick={() => setActiveSubject(s.id)}
                        className={`w-full text-left rounded-md px-3 py-2 text-sm ${activeSubject === s.id ? "bg-secondary text-foreground" : "hover:bg-muted/50 text-muted-foreground"}`}
                      >
                        {s.name}
                      </button>
                    ))}
                    {content?.subjects?.length === 0 && <p className="text-xs text-muted-foreground px-1">No subjects yet.</p>}
                  </div>
                </div>

                <div className="space-y-4">
                  <h2 className="text-sm font-semibold">Study content</h2>
                  {!activeSubject ? (
                    <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">Select a subject.</div>
                  ) : (
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <h3 className="text-xs font-semibold uppercase text-muted-foreground">Flashcards</h3>
                        {decksForSubject(activeSubject).length === 0 ? (
                          <p className="text-sm text-muted-foreground">No flashcards yet.</p>
                        ) : (
                          <div className="grid gap-2">
                            {decksForSubject(activeSubject).map((d: any) => (
                              <Link key={d.id} href={`/flashcards/${d.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                                <div className="text-sm font-medium">{d.title}</div>
                                <div className="text-xs text-muted-foreground">{d.card_count} cards</div>
                              </Link>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="space-y-2">
                        <h3 className="text-xs font-semibold uppercase text-muted-foreground">Mock Tests</h3>
                        {testsForSubject(activeSubject).length === 0 ? (
                          <p className="text-sm text-muted-foreground">No mock tests yet.</p>
                        ) : (
                          <div className="grid gap-2">
                            {testsForSubject(activeSubject).map((t: any) => (
                              <Link key={t.test_id || t.id} href={`/mock-tests/${t.test_id || t.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                                <div className="text-sm font-medium">{t.title}</div>
                                <div className="text-xs text-muted-foreground">{t.total_marks} marks</div>
                              </Link>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="flashcards">
              <div className="grid gap-2 pt-4">
                {content?.decks?.length === 0 ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No flashcards available yet.</div>
                ) : (
                  content.decks.map((d: any) => (
                    <Link key={d.id} href={`/flashcards/${d.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                      <div className="text-sm font-medium">{d.title}</div>
                      <div className="text-xs text-muted-foreground">{d.card_count} cards</div>
                    </Link>
                  ))
                )}
              </div>
            </TabsContent>

            <TabsContent value="mock-tests">
              <div className="grid gap-2 pt-4">
                {content?.tests?.length === 0 ? (
                  <div className="rounded-md border bg-card p-8 text-center text-sm text-muted-foreground">No mock tests available yet.</div>
                ) : (
                  content.tests.map((t: any) => (
                    <Link key={t.test_id || t.id} href={`/mock-tests/${t.test_id || t.id}`} className="rounded-md border p-3 hover:bg-muted/50">
                      <div className="text-sm font-medium">{t.title}</div>
                      <div className="text-xs text-muted-foreground">{t.total_marks} marks</div>
                    </Link>
                  ))
                )}
              </div>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  )
}
