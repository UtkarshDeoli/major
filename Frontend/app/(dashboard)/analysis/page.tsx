"use client"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { analysisAPI } from "@/lib/api"
import { getErrorMessage } from "@/lib/errors"
import { PdfSelector, type PdfSelection } from "@/components/dashboard/test/pdf-selector"
import {
  Brain,
  Target,
  TrendingUp,
  FileText,
  BarChart3,
  Lightbulb,
  CheckCircle,
  Download,
} from "lucide-react"

export default function AnalysisPage() {
  const { toast } = useToast()
  const [selection, setSelection] = useState<PdfSelection>({
    syllabusId: "",
    questionPaperIds: [],
    notesId: "none",
  })
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)

  const handleSelectionChange = useCallback((s: PdfSelection) => {
    setSelection(s)
  }, [])

  const handleAnalyze = async () => {
    if (!selection.syllabusId || selection.syllabusId === "no-pdfs" || selection.questionPaperIds.length === 0) {
      toast({
        title: "Missing Selection",
        description: "Please select a syllabus and at least one question paper",
        variant: "destructive",
      })
      return
    }

    setIsAnalyzing(true)
    try {
      const result = await analysisAPI.analyzeQuestionPapers(selection.syllabusId, selection.questionPaperIds)
      setAnalysisResult(result)
      toast({
        title: "Analysis Complete",
        description: "Question paper analysis has been generated successfully",
      })
    } catch (error: unknown) {
      console.error("Analysis error:", error)
      toast({
        title: "Analysis Failed",
        description: getErrorMessage(error),
        variant: "destructive",
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-6">
      <div className="grid gap-6">
        {!analysisResult ? (
          <>
            {/* Header */}
            <div className="rounded-md border p-4">
              <div className="flex items-center gap-2 border-l-2 border-l-primary pl-3">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Generate Question Paper Analysis
                </h3>
              </div>
              <p className="text-sm text-muted-foreground mt-2 pl-5">
                Select your syllabus and previous year question papers to generate intelligent analysis using AI.
                Make sure your PDFs contain clear, readable text for best results.
              </p>
            </div>

            {/* PDF Selection */}
            <div className="rounded-md border p-4">
              <PdfSelector onSelectionChange={handleSelectionChange} />
            </div>

            {/* Analyze Button */}
            <Button
              onClick={handleAnalyze}
              disabled={
                isAnalyzing ||
                !selection.syllabusId ||
                selection.syllabusId === "no-pdfs" ||
                selection.questionPaperIds.length === 0
              }
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
                  Generate Analysis ({selection.questionPaperIds.length} papers)
                </span>
              )}
            </Button>
          </>
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
              <p className="text-muted-foreground leading-relaxed pl-5">{analysisResult.overall_summary}</p>
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
                      <Badge
                        variant={
                          unit.difficulty_level === "Easy"
                            ? "secondary"
                            : unit.difficulty_level === "Medium"
                            ? "default"
                            : "destructive"
                        }
                        className="rounded-md"
                      >
                        {unit.difficulty_level}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <div className="text-2xl font-bold text-primary">{unit.weightage_percentage}%</div>
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
                              {marks.replace("_", " ")}: {count}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <h4 className="font-medium mb-2 text-sm">Examples:</h4>
                        <ul className="text-sm text-muted-foreground space-y-1">
                          {pattern.examples.slice(0, 2).map((example: string, exIndex: number) => (
                            <li key={exIndex} className="truncate">
                              &bull; {example}
                            </li>
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
              <Button onClick={() => setAnalysisResult(null)} variant="outline" className="rounded-md">
                Generate New Analysis
              </Button>
              <Button
                onClick={() => {
                  const dataStr = JSON.stringify(analysisResult, null, 2)
                  const dataBlob = new Blob([dataStr], { type: "application/json" })
                  const url = URL.createObjectURL(dataBlob)
                  const link = document.createElement("a")
                  link.href = url
                  link.download = "question-paper-analysis.json"
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
    </div>
  )
}