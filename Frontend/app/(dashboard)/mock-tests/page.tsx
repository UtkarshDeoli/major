"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { useToast } from "@/hooks/use-toast"
import { useRouter, useSearchParams } from "next/navigation"
import { mockTestAPI, teacherAPI } from "@/lib/api"
import { useAuth } from "@/lib/context/auth-context"
import { useDashboard } from "@/lib/context/dashboard-context"
import { getErrorMessage } from "@/lib/errors"
import { MockTest } from "@/lib/data"
import { PdfSelector, type PdfSelection } from "@/components/dashboard/test/pdf-selector"
import { MockTestAttemptsDialog } from "@/components/dashboard/mock-test-attempts-dialog"
import { Target, Clock, History } from "lucide-react"

export default function MockTestsPage() {
  const router = useRouter()
  const { toast } = useToast()
  const { user } = useAuth()
  const isTeacher = user?.role === "teacher"
  const { activeExam } = useDashboard()
  const examSubjects = activeExam?.subjects ?? []

  const [selection, setSelection] = useState<PdfSelection>({
    syllabusId: "",
    questionPaperIds: [],
    notesId: "none",
  })
  const [isGeneratingMockTest, setIsGeneratingMockTest] = useState(false)
  const [mockTestSettings, setMockTestSettings] = useState({
    numMcq: 15,
    numText: 5,
    totalMarks: 50,
    difficultyLevel: "medium",
  })
  const [adaptiveMode, setAdaptiveMode] = useState(false)
  const [mockTests, setMockTests] = useState<MockTest[]>([])
  const [isLoadingMockTests, setIsLoadingMockTests] = useState(false)
  const [linkedStudents, setLinkedStudents] = useState<Array<{ email: string; name?: string }>>([])
  const [selectedStudent, setSelectedStudent] = useState("")
  const [subject, setSubject] = useState("")
  const [selectedTopics, setSelectedTopics] = useState("")
  const [targetWeaknesses, setTargetWeaknesses] = useState(false)
  const [gradingMode, setGradingMode] = useState<"auto" | "teacher">("auto")
  const [attemptsTestId, setAttemptsTestId] = useState<string | null>(null)

  const handleSelectionChange = useCallback((s: PdfSelection) => {
    setSelection(s)
  }, [])

  // Prefill subject + focus topics from query (e.g. arriving from chat "Mock Test"
  // or analytics "practice weak areas" actions).
  const searchParams = useSearchParams()
  useEffect(() => {
    const subj = searchParams.get("subject")
    if (subj) setSubject(subj)
    const focus = searchParams.get("focus")
    if (focus) setSelectedTopics(focus)
    const weak = searchParams.get("weak")
    if (weak === "1") setTargetWeaknesses(true)
  }, [searchParams])

  // Fetch managed students for teachers
  useEffect(() => {
    if (!isTeacher) return
    teacherAPI
      .listManagedStudents()
      .then((students) => setLinkedStudents(students || []))
      .catch((err) => console.error("Failed to load students:", err))
  }, [isTeacher])

  // Fetch mock tests
  const fetchMockTests = useCallback(async () => {
    setIsLoadingMockTests(true)
    try {
      const userMockTests = await mockTestAPI.listMockTests()
      setMockTests(Array.isArray(userMockTests) ? userMockTests : [])
    } catch (error) {
      console.error("Error fetching mock tests:", error)
      setMockTests([])
      toast({
        title: "Error",
        description: "Failed to fetch mock tests",
        variant: "destructive",
      })
    } finally {
      setIsLoadingMockTests(false)
    }
  }, [toast])

  useEffect(() => {
    fetchMockTests()
  }, [fetchMockTests])

  const handleGenerateMockTest = async () => {
    if (!selection.syllabusId || selection.syllabusId === "no-pdfs" || selection.questionPaperIds.length === 0) {
      toast({
        title: "Missing Selection",
        description: "Please select a syllabus and at least one question paper",
        variant: "destructive",
      })
      return
    }

    setIsGeneratingMockTest(true)
    try {
      const topicList = selectedTopics
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
      const weakTopicsToSend = targetWeaknesses ? topicList : undefined

      const isAdaptive = adaptiveMode || mockTestSettings.difficultyLevel === "adaptive"
      const mockTest = await mockTestAPI.generateMockTest(
        selection.syllabusId,
        selection.questionPaperIds,
        selection.notesId !== "none" ? selection.notesId : undefined,
        mockTestSettings.numMcq,
        mockTestSettings.numText,
        mockTestSettings.totalMarks,
        mockTestSettings.difficultyLevel,
        topicList.length > 0 ? topicList : undefined,
        weakTopicsToSend,
        subject || undefined,
        selectedStudent || undefined,
        gradingMode,
        undefined,
        isAdaptive,
      )

      toast({
        title: "Mock Test Generated",
        description: "Your personalized mock test is ready!",
      })

      router.push(`/test/quiz?testId=${mockTest.test_id}`)
    } catch (error: unknown) {
      console.error("Mock test generation error:", error)
      toast({
        title: "Generation Failed",
        description: getErrorMessage(error),
        variant: "destructive",
      })
    } finally {
      setIsGeneratingMockTest(false)
    }
  }

  const handleStartTest = (testId: string) => {
    router.push(`/test/quiz?testId=${testId}`)
  }

  const handleViewResults = (testId: string, submissionId: string) => {
    router.push(`/test/results?testId=${testId}&submissionId=${submissionId}`)
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-6">
      <div className="grid gap-6">
        {/* Mock Test Generator */}
        <div className="rounded-md border">
          <div className="p-4 border-b">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Target className="h-5 w-5" />
              Generate Mock Test
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
              Create a personalized mock test based on your study materials
            </p>
          </div>
          <div className="p-4 space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              {/* Material Selection */}
              <div className="space-y-4">
                <h3 className="font-semibold text-base">Select Study Materials</h3>
                <PdfSelector onSelectionChange={handleSelectionChange} showNotes />
              </div>

              {/* Test Configuration */}
              <div className="space-y-4">
                <h3 className="font-semibold text-base">Test Configuration</h3>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="numMcq">MCQ Questions</Label>
                    <Input
                      id="numMcq"
                      type="number"
                      min="5"
                      max="50"
                      value={mockTestSettings.numMcq}
                      onChange={(e) =>
                        setMockTestSettings((prev) => ({
                          ...prev,
                          numMcq: parseInt(e.target.value) || 15,
                        }))
                      }
                      className="rounded-md h-9 text-[13px]"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="numText">Descriptive Questions</Label>
                    <Input
                      id="numText"
                      type="number"
                      min="1"
                      max="20"
                      value={mockTestSettings.numText}
                      onChange={(e) =>
                        setMockTestSettings((prev) => ({
                          ...prev,
                          numText: parseInt(e.target.value) || 5,
                        }))
                      }
                      className="rounded-md h-9 text-[13px]"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="totalMarks">Total Marks</Label>
                    <Input
                      id="totalMarks"
                      type="number"
                      min="20"
                      max="200"
                      value={mockTestSettings.totalMarks}
                      onChange={(e) =>
                        setMockTestSettings((prev) => ({
                          ...prev,
                          totalMarks: parseInt(e.target.value) || 50,
                        }))
                      }
                      className="rounded-md h-9 text-[13px]"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="difficulty">Difficulty Level</Label>
                    <Select
                      value={mockTestSettings.difficultyLevel}
                      onValueChange={(value) =>
                        setMockTestSettings((prev) => ({ ...prev, difficultyLevel: value }))
                      }
                    >
                      <SelectTrigger className="rounded-md h-9 text-[13px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="easy">Easy</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="hard">Hard</SelectItem>
                        <SelectItem value="adaptive">Adaptive (based on my past tests)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {isTeacher && (
                    <div className="space-y-2 sm:col-span-2">
                      <Label htmlFor="assign-student">Assign to Student</Label>
                      <select
                        id="assign-student"
                        className="w-full rounded-md border bg-background px-3 py-1.5 text-[13px] h-9"
                        value={selectedStudent}
                        onChange={(e) => setSelectedStudent(e.target.value)}
                      >
                        <option value="">Myself / No assignment</option>
                        {linkedStudents.map((s) => (
                          <option key={s.email} value={s.email}>
                            {s.name || s.email}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="subject">Subject</Label>
                    {examSubjects.length > 0 ? (
                      <Select value={subject} onValueChange={setSubject}>
                        <SelectTrigger id="subject">
                          <SelectValue placeholder="Select a subject" />
                        </SelectTrigger>
                        <SelectContent>
                          {examSubjects.map((s) => (
                            <SelectItem key={s.id} value={s.name}>
                              {s.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : (
                      <Input
                        id="subject"
                        type="text"
                        placeholder="Set up an exam with subjects first"
                        value={subject}
                        onChange={(e) => setSubject(e.target.value)}
                        className="rounded-md h-9 text-[13px]"
                      />
                    )}
                  </div>

                  {isTeacher && (
                    <div className="space-y-2">
                      <Label htmlFor="grading-mode">Grading</Label>
                      <Select value={gradingMode} onValueChange={(v) => setGradingMode(v as "auto" | "teacher")}>
                        <SelectTrigger id="grading-mode">
                          <SelectValue placeholder="Grading mode" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto (AI / MCQ graded)</SelectItem>
                          <SelectItem value="teacher">Teacher-marked (you grade answers)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  <div className="space-y-2">
                    <Label htmlFor="focus-topics">Focus Topics (comma-separated)</Label>
                    <Input
                      id="focus-topics"
                      type="text"
                      placeholder="e.g. Calculus, Algebra"
                      value={selectedTopics}
                      onChange={(e) => setSelectedTopics(e.target.value)}
                      className="rounded-md h-9 text-[13px]"
                    />
                  </div>

                  {isTeacher && (
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="target-weaknesses"
                        checked={targetWeaknesses}
                        onCheckedChange={(checked) => setTargetWeaknesses(checked === true)}
                      />
                      <Label htmlFor="target-weaknesses" className="text-sm">
                        Target weak topics
                      </Label>
                    </div>
                  )}
                </div>

                <div className="rounded-md border bg-secondary/50 p-4">
                  <h4 className="font-medium mb-2 text-sm">Test Summary</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>&bull; {mockTestSettings.numMcq} MCQ questions (2 marks each)</p>
                    <p>&bull; {mockTestSettings.numText} descriptive questions</p>
                    <p>&bull; Total marks: {mockTestSettings.totalMarks}</p>
                    <p>
                      &bull; Estimated time:{" "}
                      {Math.ceil(mockTestSettings.numMcq * 2 + mockTestSettings.numText * 10)} minutes
                    </p>
                    <p>&bull; Difficulty: {mockTestSettings.difficultyLevel}{adaptiveMode ? " (adaptive)" : ""}</p>
                  </div>
                </div>
              </div>
            </div>

            <Button
              onClick={handleGenerateMockTest}
              disabled={
                isGeneratingMockTest ||
                !selection.syllabusId ||
                selection.syllabusId === "no-pdfs" ||
                selection.questionPaperIds.length === 0
              }
              className="w-full rounded-md h-9 text-[13px]"
              size="sm"
            >
              {isGeneratingMockTest ? (
                <span className="flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Generating Test with AI...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Target className="h-4 w-4" />
                  Generate Mock Test
                </span>
              )}
            </Button>
          </div>
        </div>

        {/* Existing Mock Tests */}
        <div className="rounded-md border">
          <div className="p-4 border-b">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Your Mock Tests
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Previously generated mock tests ready to take
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={fetchMockTests}
                disabled={isLoadingMockTests}
                className="rounded-md h-9 text-[13px]"
              >
                Refresh
              </Button>
            </div>
          </div>
          <div className="p-4">
            {isLoadingMockTests ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                  <p className="text-muted-foreground">Loading mock tests...</p>
                </div>
              </div>
            ) : mockTests.length === 0 ? (
              <div className="text-center py-8">
                <div className="mx-auto w-16 h-16 rounded-md bg-secondary flex items-center justify-center mb-4">
                  <Target className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="text-muted-foreground mb-2">No mock tests generated yet</p>
                <p className="text-sm text-muted-foreground">
                  Generate your first mock test using the form above!
                </p>
              </div>
            ) : (
              <div className="rounded-md border">
                <div className="grid grid-cols-[1fr_80px_80px_80px_100px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  <span>Title</span>
                  <span className="text-right">Questions</span>
                  <span className="text-right">Marks</span>
                  <span className="text-right">Time</span>
                  <span className="text-right">Actions</span>
                </div>
                {mockTests.map((test: any) => (
                  <div
                    key={test.test_id}
                    className="grid grid-cols-[1fr_80px_80px_80px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center"
                  >
                    <div className="min-w-0">
                      <p className="text-sm font-medium truncate">{test.title}</p>
                      <p className="text-xs text-muted-foreground font-mono">
                        {new Date(test.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <span className="text-sm tabular-nums text-right">{test.questions.length}</span>
                    <span className="text-sm tabular-nums text-right">{test.total_marks}</span>
                    <span className="text-sm tabular-nums text-right">{test.time_limit}m</span>
                    <div className="flex gap-1 justify-end">
                      <Button
                        size="sm"
                        className="rounded-md h-7 text-xs"
                        onClick={() => handleStartTest(test.test_id)}
                      >
                        Start
                      </Button>
                      {test.latest_submission && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="rounded-md h-7 text-xs"
                          onClick={() =>
                            handleViewResults(test.test_id, test.latest_submission.submission_id)
                          }
                        >
                          {test.latest_submission.percentage.toFixed(0)}%
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="rounded-md h-7 text-xs"
                        title="Attempts"
                        onClick={() => setAttemptsTestId(test.test_id)}
                      >
                        <History className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <MockTestAttemptsDialog
        testId={attemptsTestId}
        isTeacher={isTeacher}
        open={!!attemptsTestId}
        onOpenChange={(o) => !o && setAttemptsTestId(null)}
      />
    </div>
  )
}