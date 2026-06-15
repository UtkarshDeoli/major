"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
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
import { pdfAPI, analysisAPI, mockTestAPI, authAPI, teacherAPI } from '@/lib/api'

export default function TestPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const preselectedStudent = searchParams.get("student")
  const { toast } = useToast()
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [user, setUser] = useState<{ email: string; name?: string; role?: string } | null>(null)
  const [linkedStudents, setLinkedStudents] = useState<Array<{ email: string; name?: string }>>([])
  const [selectedStudent, setSelectedStudent] = useState<string>(preselectedStudent || "")
  const [targetWeaknesses, setTargetWeaknesses] = useState(false)
  const [selectedTopics, setSelectedTopics] = useState<string[]>([])
  const [isGeneratingMockTest, setIsGeneratingMockTest] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [pdfs, setPdfs] = useState<any[]>([])
  const [selectedSyllabus, setSelectedSyllabus] = useState<string>('')
  const [selectedQuestionPapers, setSelectedQuestionPapers] = useState<string[]>([])
  const [selectedNotes, setSelectedNotes] = useState<string>('none')
  const [subject, setSubject] = useState<string>("")
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

  // Load current user
  useEffect(() => {
    const loadUser = async () => {
      try {
        const me = await authAPI.getMe();
        setUser(me);
      } catch (error) {
        console.error("Error loading user:", error);
      }
    };

    if (authAPI.isAuthenticated()) {
      loadUser();
    }
  }, []);

  // Fetch linked students when the current user is a teacher
  useEffect(() => {
    if (user?.role === "teacher") {
      teacherAPI.listManagedStudents().then(setLinkedStudents).catch((error) => {
        console.error("Error fetching managed students:", error);
      });
    }
  }, [user?.role]);

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
      const mockTest = await mockTestAPI.generateMockTest(
        selectedSyllabus,
        selectedQuestionPapers,
        selectedNotes !== 'none' ? selectedNotes : undefined,
        mockTestSettings.numMcq,
        mockTestSettings.numText,
        mockTestSettings.totalMarks,
        mockTestSettings.difficultyLevel,
        selectedTopics.length > 0 ? selectedTopics : undefined,
        targetWeaknesses ? [] : undefined,
        subject || undefined,
        selectedStudent || undefined,
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
    <div className="container mx-auto py-8 px-4">
      <Tabs defaultValue="analysis" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 max-w-[400px]">
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="mock">Mock Tests</TabsTrigger>
        </TabsList>

        <TabsContent value="analysis">
          <div className="grid gap-6 max-w-6xl mx-auto">
            {!analysisResult ? (
              <div className="grid gap-6 lg:grid-cols-2">
                {/* PDF Selection */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="h-5 w-5" />
                      Generate Question Paper Analysis
                    </CardTitle>
                    <CardDescription>
                      Select your syllabus and previous year question papers to generate intelligent analysis using AI. 
                      Make sure your PDFs contain clear, readable text for best results.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Syllabus Selection */}
                    <div className="space-y-3">
                      <Label className="text-sm font-medium">Select Syllabus</Label>
                      <Select value={selectedSyllabus} onValueChange={setSelectedSyllabus}>
                        <SelectTrigger>
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
                      className="w-full"
                      size="lg"
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
                  </CardContent>
                </Card>

                {/* Upload New PDFs */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileUp className="h-5 w-5" />
                      Upload New PDFs
                    </CardTitle>
                    <CardDescription>
                      Upload additional study materials if needed
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="questionPaper">Previous Question Papers</Label>
                      <div className="flex items-center gap-4">
                        <Input
                          id="questionPaper"
                          type="file"
                          accept=".pdf,.doc,.docx"
                          onChange={(e) => handleFileSelect('questionPaper', e.target.files?.[0] || null)}
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
                        />
                        <Book className="h-5 w-5 text-muted-foreground" />
                      </div>
                    </div>

                    <Button 
                      onClick={handleUpload} 
                      disabled={isUploading || !Object.values(selectedFiles).some(file => file)}
                      className="w-full"
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
                  </CardContent>
                </Card>
              </div>
            ) : (
              /* Analysis Results */
              <div className="space-y-6">
                {/* Header */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-6 w-6" />
                      Question Paper Analysis Report
                    </CardTitle>
                    <CardDescription>
                      AI-powered analysis of your syllabus and previous year question papers
                    </CardDescription>
                  </CardHeader>
                </Card>

                {/* Overall Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5" />
                      Overall Summary
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground leading-relaxed">
                      {analysisResult.overall_summary}
                    </p>
                  </CardContent>
                </Card>

                {/* Focus Areas */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="h-5 w-5" />
                      Key Focus Areas
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {analysisResult.focus_areas.map((area: string, index: number) => (
                        <Badge key={index} variant="secondary" className="px-3 py-1">
                          {area}
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Unit-wise Analysis */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Unit-wise Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4 md:grid-cols-2">
                      {analysisResult.unit_wise_analysis.map((unit: any, index: number) => (
                        <Card key={index} className="border-l-4 border-l-primary">
                          <CardHeader className="pb-2">
                            <div className="flex justify-between items-start">
                              <CardTitle className="text-lg">{unit.unit_name}</CardTitle>
                              <Badge variant={unit.difficulty_level === 'Easy' ? 'secondary' : unit.difficulty_level === 'Medium' ? 'default' : 'destructive'}>
                                {unit.difficulty_level}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="text-2xl font-bold text-primary">
                                {unit.weightage_percentage}%
                              </div>
                              <div className="text-sm text-muted-foreground">weightage</div>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div>
                              <h4 className="font-medium mb-2">Important Topics:</h4>
                              <div className="flex flex-wrap gap-1">
                                {unit.important_topics.map((topic: string, topicIndex: number) => (
                                  <Badge key={topicIndex} variant="outline" className="text-xs">
                                    {topic}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                            <div>
                              <h4 className="font-medium mb-1">Recommendation:</h4>
                              <p className="text-sm text-muted-foreground">{unit.recommendation}</p>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Question Patterns */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      Question Patterns
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4 md:grid-cols-2">
                      {analysisResult.question_patterns.map((pattern: any, index: number) => (
                        <Card key={index}>
                          <CardHeader className="pb-2">
                            <CardTitle className="text-base">{pattern.question_type}</CardTitle>
                            <div className="text-sm text-muted-foreground">
                              Frequency: {pattern.frequency} times
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div>
                              <h4 className="font-medium mb-2">Marks Distribution:</h4>
                              <div className="flex gap-2 flex-wrap">
                                {Object.entries(pattern.marks_distribution).map(([marks, count]: [string, any]) => (
                                  <Badge key={marks} variant="outline">
                                    {marks.replace('_', ' ')}: {count}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                            <div>
                              <h4 className="font-medium mb-2">Examples:</h4>
                              <ul className="text-sm text-muted-foreground space-y-1">
                                {pattern.examples.slice(0, 2).map((example: string, exIndex: number) => (
                                  <li key={exIndex} className="truncate">• {example}</li>
                                ))}
                              </ul>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Sample Questions */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Lightbulb className="h-5 w-5" />
                      Generated Sample Questions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {analysisResult.sample_questions.map((question: string, index: number) => (
                        <div key={index} className="p-4 border rounded-lg">
                          <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                              <span className="text-sm font-medium text-primary">{index + 1}</span>
                            </div>
                            <p className="text-sm leading-relaxed">{question}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Preparation Strategy */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="h-5 w-5" />
                      Preparation Strategy
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground leading-relaxed whitespace-pre-line">
                      {analysisResult.preparation_strategy}
                    </p>
                  </CardContent>
                </Card>

                {/* Actions */}
                <div className="flex gap-4">
                  <Button 
                    onClick={() => setAnalysisResult(null)}
                    variant="outline"
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
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Generate Mock Test
                </CardTitle>
                <CardDescription>
                  Create a personalized mock test based on your study materials
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-6 lg:grid-cols-2">
                  {/* Material Selection */}
                  <div className="space-y-4">
                    <h3 className="font-semibold text-base">Select Study Materials</h3>
                    
                    {/* Syllabus Selection */}
                    <div className="space-y-2">
                      <Label className="text-sm font-medium">Syllabus (Required)</Label>
                      <Select value={selectedSyllabus} onValueChange={setSelectedSyllabus}>
                        <SelectTrigger>
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
                        <SelectTrigger>
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
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="easy">Easy</SelectItem>
                            <SelectItem value="medium">Medium</SelectItem>
                            <SelectItem value="hard">Hard</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="subject">Subject</Label>
                        <Input
                          id="subject"
                          type="text"
                          placeholder="e.g. Mathematics"
                          value={subject}
                          onChange={(e) => setSubject(e.target.value)}
                        />
                      </div>
                    </div>

                    <div className="space-y-2 md:col-span-2">
                      <Label htmlFor="focusTopics">Focus Topics (optional)</Label>
                      <Input
                        id="focusTopics"
                        placeholder="Enter topics separated by commas"
                        value={selectedTopics.join(", ")}
                        onChange={(e) =>
                          setSelectedTopics(
                            e.target.value
                              .split(",")
                              .map((t) => t.trim())
                              .filter((t) => t.length > 0)
                          )
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        The test will emphasize these topics when provided.
                      </p>
                    </div>

                    {user?.role === "teacher" && (
                      <div className="space-y-4 md:col-span-2">
                        <div className="space-y-2">
                          <Label htmlFor="assignStudent" className="text-sm font-medium">
                            Assign to student
                          </Label>
                          <Select
                            value={selectedStudent}
                            onValueChange={setSelectedStudent}
                          >
                            <SelectTrigger id="assignStudent">
                              <SelectValue placeholder="Myself / No assignment" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="">Myself / No assignment</SelectItem>
                              {linkedStudents.map((s) => (
                                <SelectItem key={s.email} value={s.email}>
                                  {s.name || s.email}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="flex items-center gap-2">
                          <Checkbox
                            id="target-weaknesses"
                            checked={targetWeaknesses}
                            onCheckedChange={(checked) =>
                              setTargetWeaknesses(checked === true)
                            }
                          />
                          <Label htmlFor="target-weaknesses" className="text-sm">
                            Target weak topics
                          </Label>
                        </div>
                      </div>
                    )}

                    <div className="p-4 bg-muted/30 rounded-lg">
                      <h4 className="font-medium mb-2">Test Summary</h4>
                      <div className="text-sm text-muted-foreground space-y-1">
                        <p>• {mockTestSettings.numMcq} MCQ questions (2 marks each)</p>
                        <p>• {mockTestSettings.numText} descriptive questions</p>
                        <p>• Total marks: {mockTestSettings.totalMarks}</p>
                        <p>• Estimated time: {Math.ceil((mockTestSettings.numMcq * 2 + mockTestSettings.numText * 10))} minutes</p>
                        <p>• Difficulty: {mockTestSettings.difficultyLevel}</p>
                      </div>
                    </div>
                  </div>
                </div>

                <Button 
                  onClick={handleGenerateMockTest}
                  disabled={isGeneratingMockTest || !selectedSyllabus || selectedSyllabus === 'no-pdfs' || selectedQuestionPapers.length === 0}
                  className="w-full"
                  size="lg"
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
              </CardContent>
            </Card>

            {/* Upload New PDFs (if needed) */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileUp className="h-5 w-5" />
                  Upload New PDFs
                </CardTitle>
                <CardDescription>
                  Add new study materials to your collection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="questionPaper">Question Papers</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        id="questionPaper"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('questionPaper', e.target.files?.[0] || null)}
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
                      />
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>
                </div>

                <Button 
                  onClick={handleUpload} 
                  disabled={isUploading || !Object.values(selectedFiles).some(file => file)}
                  className="w-full"
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
              </CardContent>
            </Card>

            {/* Existing Mock Tests */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Your Mock Tests
                    </CardTitle>
                    <CardDescription>
                      Previously generated mock tests ready to take
                    </CardDescription>
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={fetchMockTests}
                    disabled={isLoadingMockTests}
                  >
                    Refresh
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {isLoadingMockTests ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
                      <p className="text-muted-foreground">Loading mock tests...</p>
                    </div>
                  </div>
                ) : (mockTests && mockTests.length === 0) ? (
                  <div className="text-center py-8">
                    <div className="mx-auto w-16 h-16 bg-muted rounded-full flex items-center justify-center mb-4">
                      <Target className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground mb-2">No mock tests generated yet</p>
                    <p className="text-sm text-muted-foreground">Generate your first mock test using the form above!</p>
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                    {(mockTests || []).map((test: any) => (
                      <Card key={test.test_id} className="hover:shadow-md transition-shadow">
                        <CardContent className="pt-6">
                          <div className="space-y-3">
                            <div>
                              <h4 className="font-semibold text-base truncate">{test.title}</h4>
                              <div className="text-xs text-muted-foreground">
                                Created: {new Date(test.created_at).toLocaleDateString()}
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <CheckCircle className="h-4 w-4" />
                                <span>{test.questions.length} questions</span>
                              </div>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Target className="h-4 w-4" />
                                <span>{test.total_marks} marks</span>
                              </div>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Clock className="h-4 w-4" />
                                <span>{test.time_limit} min</span>
                              </div>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Brain className="h-4 w-4" />
                                <span className="capitalize">{test.difficulty_level}</span>
                              </div>
                            </div>
                            
                            <Separator />
                            
                            <div className="space-y-2">
                              <Button 
                                className="w-full"
                                onClick={() => handleStartTest(test.test_id)}
                              >
                                <Target className="h-4 w-4 mr-2" />
                                Start Test
                              </Button>
                              
                              {test.latest_submission && (
                                <Button 
                                  variant="outline"
                                  className="w-full"
                                  onClick={() => handleViewResults(test.test_id, test.latest_submission!.submission_id)}
                                >
                                  <BarChart3 className="h-4 w-4 mr-2" />
                                  View Results ({test.latest_submission.percentage.toFixed(1)}%)
                                </Button>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}