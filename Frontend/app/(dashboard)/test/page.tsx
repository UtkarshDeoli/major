"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  FileUp,
  Book,
  FileText,
  Clock,
  CheckCircle,
  Brain,
  Target,
  TrendingUp,
  Download,
  Lightbulb,
  BarChart3
} from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { useRouter, useSearchParams } from 'next/navigation'
import { pdfAPI, analysisAPI, mockTestAPI, teacherAPI } from '@/lib/api'
import { useAuth } from '@/lib/context/auth-context'

export default function TestPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const activeTabFromUrl = searchParams.get('tab') || 'analysis'
  const [activeTab, setActiveTab] = useState<'analysis' | 'mock'>(activeTabFromUrl as 'analysis' | 'mock')
  const { toast } = useToast()
  const { user } = useAuth()
  const isTeacher = user?.role === 'teacher'

  useEffect(() => {
    setActiveTab(activeTabFromUrl as 'analysis' | 'mock')
  }, [activeTabFromUrl])

  const handleTabChange = (value: string) => {
    setActiveTab(value as 'analysis' | 'mock')
    router.push(`/test?tab=${value}`)
  }
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGeneratingMockTest, setIsGeneratingMockTest] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [pdfs, setPdfs] = useState<any[]>([])
  const [selectedSyllabus, setSelectedSyllabus] = useState<string>('')
  const [selectedQuestionPapers, setSelectedQuestionPapers] = useState<string[]>([])
  const [selectedNotes, setSelectedNotes] = useState<string>('none')
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [mockTestSettings, setMockTestSettings] = useState({
    numMcq: 15,
    numText: 5,
    totalMarks: 50,
    difficultyLevel: 'medium'
  })
  const [selectedFiles, setSelectedFiles] = useState<{ [key: string]: File | null }>({
    questionPaper: null,
    syllabus: null,
    notes: null
  })
  const [mockTests, setMockTests] = useState<any[]>([])
  const [isLoadingMockTests, setIsLoadingMockTests] = useState(false)

  // Teacher-student integration controls
  const [linkedStudents, setLinkedStudents] = useState<Array<{ email: string; name?: string }>>([])
  const [selectedStudent, setSelectedStudent] = useState<string>(searchParams.get('student') || '')
  const [subject, setSubject] = useState<string>('')
  const [selectedTopics, setSelectedTopics] = useState<string>('')
  const [targetWeaknesses, setTargetWeaknesses] = useState(false)

  useEffect(() => {
    if (!isTeacher) return
    teacherAPI.listManagedStudents()
      .then((students) => setLinkedStudents(students || []))
      .catch((err) => console.error('Failed to load students:', err))
  }, [isTeacher])

  // Function to fetch mock tests (can be called from refresh button)
  const fetchMockTests = async () => {
    setIsLoadingMockTests(true)
    try {
      const userMockTests = await mockTestAPI.listMockTests()
      // userMockTests is already the tests array from the API
      setMockTests(Array.isArray(userMockTests) ? userMockTests : [])
    } catch (error) {
      console.error('Error fetching mock tests:', error)
      setMockTests([]) // Ensure mockTests is always an array
      toast({
        title: "Error",
        description: "Failed to fetch mock tests",
        variant: "destructive"
      })
    } finally {
      setIsLoadingMockTests(false)
    }
  }

  // Fetch user's PDFs and mock tests on component mount
  useEffect(() => {
    const fetchPDFs = async () => {
      try {
        const userPdfs = await pdfAPI.listPDFs()
        setPdfs(userPdfs)
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to fetch your PDFs",
          variant: "destructive"
        })
      }
    }

    fetchPDFs()
    fetchMockTests()
  }, [])

  const handleQuestionPaperToggle = (pdfId: string) => {
    setSelectedQuestionPapers(prev =>
      prev.includes(pdfId)
        ? prev.filter(id => id !== pdfId)
        : [...prev, pdfId]
    )
  }

  const handleAnalyze = async () => {
    if (!selectedSyllabus || selectedSyllabus === 'no-pdfs' || selectedQuestionPapers.length === 0) {
      toast({
        title: "Missing Selection",
        description: "Please select a syllabus and at least one question paper",
        variant: "destructive"
      })
      return
    }

    setIsAnalyzing(true)
    try {
      console.log("Selected Syllabus:", selectedSyllabus)
      console.log("Selected Question Papers:", selectedQuestionPapers)
      const result = await analysisAPI.analyzeQuestionPapers(selectedSyllabus, selectedQuestionPapers)
      console.log("Analysis Result:", result)
      setAnalysisResult(result)
      toast({
        title: "Analysis Complete",
        description: "Question paper analysis has been generated successfully"
      })
    } catch (error: any) {
      console.error("Analysis error:", error);
      let errorMessage = "Failed to analyze question papers. Please try again.";

      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }

      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive"
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleGenerateMockTest = async () => {
    if (!selectedSyllabus || selectedSyllabus === 'no-pdfs' || selectedQuestionPapers.length === 0) {
      toast({
        title: "Missing Selection",
        description: "Please select a syllabus and at least one question paper",
        variant: "destructive"
      })
      return
    }

    setIsGeneratingMockTest(true)
    try {
      const topicList = selectedTopics
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean)
      const weakTopicsToSend = targetWeaknesses ? topicList : undefined

      const mockTest = await mockTestAPI.generateMockTest(
        selectedSyllabus,
        selectedQuestionPapers,
        selectedNotes !== 'none' ? selectedNotes : undefined,
        mockTestSettings.numMcq,
        mockTestSettings.numText,
        mockTestSettings.totalMarks,
        mockTestSettings.difficultyLevel,
        topicList.length > 0 ? topicList : undefined,
        weakTopicsToSend,
        subject || undefined,
        selectedStudent || undefined
      )

      toast({
        title: "Mock Test Generated",
        description: "Your personalized mock test is ready!"
      })

      // Navigate to the quiz page with the generated test
      router.push(`/test/quiz?testId=${mockTest.test_id}`)

    } catch (error: any) {
      console.error("Mock test generation error:", error);
      let errorMessage = "Failed to generate mock test. Please try again.";

      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.message) {
        errorMessage = error.message;
      }

      toast({
        title: "Generation Failed",
        description: errorMessage,
        variant: "destructive"
      })
    } finally {
      setIsGeneratingMockTest(false)
    }
  }

  const handleFileSelect = (type: string, file: File | null) => {
    setSelectedFiles(prev => ({
      ...prev,
      [type]: file
    }))
  }

  const handleUpload = async () => {
    if (!Object.values(selectedFiles).some(file => file)) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload",
        variant: "destructive"
      })
      return
    }

    setIsUploading(true)
    try {
      const uploadPromises = Object.entries(selectedFiles)
        .filter(([_, file]) => file !== null)
        .map(async ([type, file]) => {
          if (file) {
            const title = type === 'syllabus' ? 'Syllabus' :
                         type === 'questionPaper' ? 'Question Paper' : 'Study Notes'
            return await pdfAPI.uploadPDF(file, title, `Uploaded ${title}`, [type])
          }
        })

      await Promise.all(uploadPromises)

      // Refresh the PDF list
      const userPdfs = await pdfAPI.listPDFs()
      setPdfs(userPdfs)

      // Reset selected files
      setSelectedFiles({
        questionPaper: null,
        syllabus: null,
        notes: null
      })

      toast({
        title: "Files uploaded successfully",
        description: "Your files have been processed and are now available for analysis."
      })
    } catch (error) {
      toast({
        title: "Upload failed",
        description: "Failed to upload files. Please try again.",
        variant: "destructive"
      })
    } finally {
      setIsUploading(false)
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
      <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-6">
        <TabsList className="inline-flex h-9 items-center gap-1 rounded-md bg-secondary p-1">
          <TabsTrigger value="analysis" className="rounded-md text-[13px] px-3">Analysis</TabsTrigger>
          <TabsTrigger value="mock" className="rounded-md text-[13px] px-3">Mock Tests</TabsTrigger>
        </TabsList>

        <TabsContent value="analysis">
          <div className="grid gap-6 max-w-6xl mx-auto">
            {!analysisResult ? (
              <div className="grid gap-6 lg:grid-cols-2">
                {/* PDF Selection */}
                <div className="rounded-md border">
                  <div className="p-4 border-b">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      Generate Question Paper Analysis
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Select your syllabus and previous year question papers to generate intelligent analysis using AI.
                      Make sure your PDFs contain clear, readable text for best results.
                    </p>
                  </div>
                  <div className="p-4 space-y-6">
                    {/* Syllabus Selection */}
                    <div className="space-y-3">
                      <Label className="text-sm font-medium">Select Syllabus</Label>
                      <Select value={selectedSyllabus} onValueChange={setSelectedSyllabus}>
                        <SelectTrigger className="rounded-md h-9 text-[13px]">
                          <SelectValue placeholder="Choose a syllabus PDF" />
                        </SelectTrigger>
                        <SelectContent>
                          {pdfs.length === 0 ? (
                            <SelectItem value="no-pdfs" disabled>
                              No PDFs available - upload some files first
                            </SelectItem>
                          ) : (
                            pdfs.map((pdf) => (
                              <SelectItem key={pdf.id} value={pdf.id}>
                                <div className="flex items-center gap-2">
                                  <Book className="h-4 w-4" />
                                  <span className="truncate">{pdf.title || pdf.filename}</span>
                                </div>
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Question Papers Selection */}
                    <div className="space-y-3">
                      <Label className="text-sm font-medium">
                        Select Question Papers ({selectedQuestionPapers.length} selected)
                      </Label>
                      <ScrollArea className="h-48 w-full border rounded-md p-3">
                        <div className="space-y-2">
                          {pdfs.length === 0 ? (
                            <div className="text-center text-muted-foreground py-8">
                              <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                              <p>No PDFs available</p>
                              <p className="text-sm">Upload some files first to get started</p>
                            </div>
                          ) : (
                            pdfs.map((pdf) => (
                              <div key={pdf.id} className="flex items-center space-x-2">
                                <Checkbox
                                  id={pdf.id}
                                  checked={selectedQuestionPapers.includes(pdf.id)}
                                  onCheckedChange={() => handleQuestionPaperToggle(pdf.id)}
                                />
                                <label
                                  htmlFor={pdf.id}
                                  className="flex-1 text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                                >
                                  <div className="flex items-center gap-2">
                                    <FileText className="h-4 w-4" />
                                    <span className="truncate">{pdf.title || pdf.filename}</span>
                                  </div>
                                </label>
                              </div>
                            ))
                          )}
                        </div>
                      </ScrollArea>
                    </div>

                    <Button
                      onClick={handleAnalyze}
                      disabled={isAnalyzing || !selectedSyllabus || selectedSyllabus === 'no-pdfs' || selectedQuestionPapers.length === 0}
                      className="w-full rounded-md h-9 text-[13px]"
                      size="sm"
                    >
                      {isAnalyzing ? (
                        <span className="flex items-center gap-2">
                          <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          Analyzing with AI...
                        </span>
                      ) : (
                        <span className="flex items-center gap-2">
                          <Brain className="h-4 w-4" />
                          Generate Analysis ({selectedQuestionPapers.length} papers)
                        </span>
                      )}
                    </Button>
                  </div>
                </div>

                {/* Upload New PDFs */}
                <div className="rounded-md border">
                  <div className="p-4 border-b">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <FileUp className="h-5 w-5" />
                      Upload New PDFs
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Upload additional study materials if needed
                    </p>
                  </div>
                  <div className="p-4 space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="questionPaper">Previous Question Papers</Label>
                      <div className="flex items-center gap-4">
                        <Input
                          id="questionPaper"
                          type="file"
                          accept=".pdf,.doc,.docx"
                          onChange={(e) => handleFileSelect('questionPaper', e.target.files?.[0] || null)}
                          className="rounded-md h-9 text-[13px]"
                        />
                        <FileUp className="h-5 w-5 text-muted-foreground" />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="syllabus">Syllabus</Label>
                      <div className="flex items-center gap-4">
                        <Input
                          id="syllabus"
                          type="file"
                          accept=".pdf,.doc,.docx"
                          onChange={(e) => handleFileSelect('syllabus', e.target.files?.[0] || null)}
                          className="rounded-md h-9 text-[13px]"
                        />
                        <Book className="h-5 w-5 text-muted-foreground" />
                      </div>
                    </div>

                    <Button
                      onClick={handleUpload}
                      disabled={isUploading || !Object.values(selectedFiles).some(file => file)}
                      className="w-full rounded-md"
                    >
                      {isUploading ? (
                        <span className="flex items-center gap-2">
                          <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          Uploading...
                        </span>
                      ) : (
                        'Upload Files'
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              /* Analysis Results */
              <div className="space-y-6">
                {/* Header */}
                <div className="rounded-md border p-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Question Paper Analysis Report
                    </h3>
                  </div>
                  <p className="text-sm text-muted-foreground mt-2 pl-5">
                    AI-powered analysis of your syllabus and previous year question papers
                  </p>
                </div>

                {/* Overall Summary */}
                <div className="rounded-md border p-4 space-y-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Overall Summary
                    </h3>
                  </div>
                  <p className="text-muted-foreground leading-relaxed pl-5">
                    {analysisResult.overall_summary}
                  </p>
                </div>

                {/* Focus Areas */}
                <div className="rounded-md border p-4 space-y-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Key Focus Areas
                    </h3>
                  </div>
                  <div className="flex flex-wrap gap-2 pl-5">
                    {analysisResult.focus_areas.map((area: string, index: number) => (
                      <Badge key={index} variant="secondary" className="px-3 py-1 rounded-md">
                        {area}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Unit-wise Analysis */}
                <div className="rounded-md border p-4 space-y-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Unit-wise Analysis
                    </h3>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2 pl-5">
                    {analysisResult.unit_wise_analysis.map((unit: any, index: number) => (
                      <div key={index} className="rounded-md border border-l-2 border-l-primary p-4">
                        <div className="flex justify-between items-start">
                          <h4 className="text-sm font-semibold">{unit.unit_name}</h4>
                          <Badge variant={unit.difficulty_level === 'Easy' ? 'secondary' : unit.difficulty_level === 'Medium' ? 'default' : 'destructive'} className="rounded-md">
                            {unit.difficulty_level}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 mt-2">
                          <div className="text-2xl font-bold text-primary">
                            {unit.weightage_percentage}%
                          </div>
                          <div className="text-sm text-muted-foreground">weightage</div>
                        </div>
                        <div className="space-y-3 mt-3">
                          <div>
                            <h4 className="font-medium mb-2 text-sm">Important Topics:</h4>
                            <div className="flex flex-wrap gap-1">
                              {unit.important_topics.map((topic: string, topicIndex: number) => (
                                <Badge key={topicIndex} variant="outline" className="text-xs rounded-md">
                                  {topic}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-medium mb-1 text-sm">Recommendation:</h4>
                            <p className="text-sm text-muted-foreground">{unit.recommendation}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Question Patterns */}
                <div className="rounded-md border p-4 space-y-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      Question Patterns
                    </h3>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2 pl-5">
                    {analysisResult.question_patterns.map((pattern: any, index: number) => (
                      <div key={index} className="rounded-md border p-4">
                        <h4 className="text-sm font-semibold">{pattern.question_type}</h4>
                        <div className="text-sm text-muted-foreground">
                          Frequency: {pattern.frequency} times
                        </div>
                        <div className="space-y-3 mt-3">
                          <div>
                            <h4 className="font-medium mb-2 text-sm">Marks Distribution:</h4>
                            <div className="flex gap-2 flex-wrap">
                              {Object.entries(pattern.marks_distribution).map(([marks, count]: [string, any]) => (
                                <Badge key={marks} variant="outline" className="rounded-md">
                                  {marks.replace('_', ' ')}: {count}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-medium mb-2 text-sm">Examples:</h4>
                            <ul className="text-sm text-muted-foreground space-y-1">
                              {pattern.examples.slice(0, 2).map((example: string, exIndex: number) => (
                                <li key={exIndex} className="truncate">&bull; {example}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Sample Questions */}
                <div className="rounded-md border p-4 space-y-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <Lightbulb className="h-5 w-5" />
                      Generated Sample Questions
                    </h3>
                  </div>
                  <div className="space-y-4 pl-5">
                    {analysisResult.sample_questions.map((question: string, index: number) => (
                      <div key={index} className="p-4 rounded-md border">
                        <div className="flex items-start gap-3">
                          <div className="flex-shrink-0 w-8 h-8 rounded-md bg-secondary flex items-center justify-center">
                            <span className="text-sm font-medium">{index + 1}</span>
                          </div>
                          <p className="text-sm leading-relaxed">{question}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Preparation Strategy */}
                <div className="rounded-md border p-4 space-y-4">
                  <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <CheckCircle className="h-5 w-5" />
                      Preparation Strategy
                    </h3>
                  </div>
                  <p className="text-muted-foreground leading-relaxed whitespace-pre-line pl-5">
                    {analysisResult.preparation_strategy}
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-4">
                  <Button
                    onClick={() => setAnalysisResult(null)}
                    variant="outline"
                    className="rounded-md"
                  >
                    Generate New Analysis
                  </Button>
                  <Button
                    onClick={() => {
                      const dataStr = JSON.stringify(analysisResult, null, 2)
                      const dataBlob = new Blob([dataStr], {type: 'application/json'})
                      const url = URL.createObjectURL(dataBlob)
                      const link = document.createElement('a')
                      link.href = url
                      link.download = 'question-paper-analysis.json'
                      link.click()
                    }}
                    className="rounded-md"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Report
                  </Button>
                </div>
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="mock">
          <div className="grid gap-6 max-w-6xl mx-auto">
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

                    {/* Syllabus Selection */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Syllabus (Required)</Label>
                      <Select value={selectedSyllabus} onValueChange={setSelectedSyllabus}>
                        <SelectTrigger className="rounded-md h-9 text-[13px]">
                          <SelectValue placeholder="Choose syllabus PDF" />
                        </SelectTrigger>
                        <SelectContent>
                          {pdfs.length === 0 ? (
                            <SelectItem value="no-pdfs" disabled>
                              No PDFs available - upload some files first
                            </SelectItem>
                          ) : (
                            pdfs.map((pdf) => (
                              <SelectItem key={pdf.id} value={pdf.id}>
                                <div className="flex items-center gap-2">
                                  <Book className="h-4 w-4" />
                                  <span className="truncate">{pdf.title || pdf.filename}</span>
                                </div>
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Question Papers Selection */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">
                        Previous Year Papers (Required) - {selectedQuestionPapers.length} selected
                      </Label>
                      <ScrollArea className="h-32 w-full border rounded-md p-3">
                        <div className="space-y-2">
                          {pdfs.length === 0 ? (
                            <div className="text-center text-muted-foreground py-4">
                              <FileText className="h-6 w-6 mx-auto mb-1 opacity-50" />
                              <p className="text-sm">No PDFs available</p>
                            </div>
                          ) : (
                            pdfs.map((pdf) => (
                              <div key={pdf.id} className="flex items-center space-x-2">
                                <Checkbox
                                  id={`mock-${pdf.id}`}
                                  checked={selectedQuestionPapers.includes(pdf.id)}
                                  onCheckedChange={() => handleQuestionPaperToggle(pdf.id)}
                                />
                                <label
                                  htmlFor={`mock-${pdf.id}`}
                                  className="flex-1 text-sm font-medium leading-none cursor-pointer"
                                >
                                  <div className="flex items-center gap-2">
                                    <FileText className="h-4 w-4" />
                                    <span className="truncate">{pdf.title || pdf.filename}</span>
                                  </div>
                                </label>
                              </div>
                            ))
                          )}
                        </div>
                      </ScrollArea>
                    </div>

                    {/* Notes Selection */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Study Notes (Optional)</Label>
                      <Select value={selectedNotes} onValueChange={setSelectedNotes}>
                        <SelectTrigger className="rounded-md h-9 text-[13px]">
                          <SelectValue placeholder="Choose study notes (optional)" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">No notes selected</SelectItem>
                          {pdfs.map((pdf) => (
                            <SelectItem key={pdf.id} value={pdf.id}>
                              <div className="flex items-center gap-2">
                                <FileText className="h-4 w-4" />
                                <span className="truncate">{pdf.title || pdf.filename}</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
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
                          onChange={(e) => setMockTestSettings(prev => ({
                            ...prev,
                            numMcq: parseInt(e.target.value) || 15
                          }))}
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
                          onChange={(e) => setMockTestSettings(prev => ({
                            ...prev,
                            numText: parseInt(e.target.value) || 5
                          }))}
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
                          onChange={(e) => setMockTestSettings(prev => ({
                            ...prev,
                            totalMarks: parseInt(e.target.value) || 50
                          }))}
                          className="rounded-md h-9 text-[13px]"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="difficulty">Difficulty Level</Label>
                        <Select
                          value={mockTestSettings.difficultyLevel}
                          onValueChange={(value) => setMockTestSettings(prev => ({
                            ...prev,
                            difficultyLevel: value
                          }))}
                        >
                          <SelectTrigger className="rounded-md h-9 text-[13px]">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="easy">Easy</SelectItem>
                            <SelectItem value="medium">Medium</SelectItem>
                            <SelectItem value="hard">Hard</SelectItem>
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
                        <Input
                          id="subject"
                          type="text"
                          placeholder="e.g. Mathematics"
                          value={subject}
                          onChange={(e) => setSubject(e.target.value)}
                          className="rounded-md h-9 text-[13px]"
                        />
                      </div>

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
                        <p>&bull; Estimated time: {Math.ceil((mockTestSettings.numMcq * 2 + mockTestSettings.numText * 10))} minutes</p>
                        <p>&bull; Difficulty: {mockTestSettings.difficultyLevel}</p>
                      </div>
                    </div>
                  </div>
                </div>

                <Button
                  onClick={handleGenerateMockTest}
                  disabled={isGeneratingMockTest || !selectedSyllabus || selectedSyllabus === 'no-pdfs' || selectedQuestionPapers.length === 0}
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

            {/* Upload New PDFs (if needed) */}
            <div className="rounded-md border">
              <div className="p-4 border-b">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <FileUp className="h-5 w-5" />
                  Upload New PDFs
                </h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Add new study materials to your collection
                </p>
              </div>
              <div className="p-4 space-y-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="questionPaper">Question Papers</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        id="questionPaper"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('questionPaper', e.target.files?.[0] || null)}
                        className="rounded-md h-9 text-[13px]"
                      />
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="syllabus">Syllabus</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        id="syllabus"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('syllabus', e.target.files?.[0] || null)}
                        className="rounded-md h-9 text-[13px]"
                      />
                      <Book className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="notes">Study Notes</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        id="notes"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('notes', e.target.files?.[0] || null)}
                        className="rounded-md h-9 text-[13px]"
                      />
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>
                </div>

                <Button
                  onClick={handleUpload}
                  disabled={isUploading || !Object.values(selectedFiles).some(file => file)}
                  className="w-full rounded-md"
                >
                  {isUploading ? (
                    <span className="flex items-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Uploading...
                    </span>
                  ) : (
                    'Upload Files'
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
                ) : (mockTests && mockTests.length === 0) ? (
                  <div className="text-center py-8">
                    <div className="mx-auto w-16 h-16 rounded-md bg-secondary flex items-center justify-center mb-4">
                      <Target className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground mb-2">No mock tests generated yet</p>
                    <p className="text-sm text-muted-foreground">Generate your first mock test using the form above!</p>
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
                    {(mockTests || []).map((test: any) => (
                      <div key={test.test_id} className="grid grid-cols-[1fr_80px_80px_80px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center">
                        <div className="min-w-0">
                          <p className="text-sm font-medium truncate">{test.title}</p>
                          <p className="text-xs text-muted-foreground font-mono">{new Date(test.created_at).toLocaleDateString()}</p>
                        </div>
                        <span className="text-sm tabular-nums text-right">{test.questions.length}</span>
                        <span className="text-sm tabular-nums text-right">{test.total_marks}</span>
                        <span className="text-sm tabular-nums text-right">{test.time_limit}m</span>
                        <div className="flex gap-1 justify-end">
                          <Button size="sm" className="rounded-md h-7 text-xs" onClick={() => handleStartTest(test.test_id)}>Start</Button>
                          {test.latest_submission && (
                            <Button variant="outline" size="sm" className="rounded-md h-7 text-xs" onClick={() => handleViewResults(test.test_id, test.latest_submission!.submission_id)}>
                              {test.latest_submission.percentage.toFixed(0)}%
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}