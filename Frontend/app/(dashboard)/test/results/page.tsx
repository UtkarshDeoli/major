"use client"

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/hooks/use-toast'
import { 
  ArrowLeft, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Target, 
  TrendingUp, 
  TrendingDown,
  BookOpen,
  Award,
  Loader2
} from 'lucide-react'
import { mockTestAPI } from '@/lib/api'
import { MockTestAnalysis, AnswerFeedback } from '@/lib/data'
import { getErrorMessage } from '@/lib/errors'

export default function TestResultsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const testId = searchParams.get('testId')
  const submissionId = searchParams.get('submissionId')
  
  const [analysis, setAnalysis] = useState<MockTestAnalysis | null>(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()
  
  useEffect(() => {
    const fetchResults = async () => {
      if (!testId || !submissionId) {
        toast({
          title: "Error",
          description: "Missing test or submission information",
          variant: "destructive"
        })
        router.push('/mock-tests')
        return
      }

      try {
        // 1. Try the per-submission sessionStorage cache (fast path; not deleted on read).
        const cached = submissionId ? sessionStorage.getItem(`testAnalysis:${submissionId}`) : null
        if (cached) {
          setAnalysis(JSON.parse(cached) as MockTestAnalysis)
          setLoading(false)
          return
        }

        // 2. Otherwise fetch the authoritative copy from the backend.
        const analysisData = await mockTestAPI.getAnalysisBySubmissionId(submissionId)
        setAnalysis(analysisData)
        sessionStorage.setItem(`testAnalysis:${submissionId}`, JSON.stringify(analysisData))
      } catch (error) {
        console.error("Error fetching results:", error)
        toast({
          title: "Error",
          description: getErrorMessage(error),
          variant: "destructive",
        })
      } finally {
        setLoading(false)
      }
    }

    fetchResults()
  }, [testId, submissionId, toast, router])

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${minutes}m ${secs}s`
  }

  const getScoreColor = (percentage: number) => {
    if (percentage >= 80) return "text-green-600"
    if (percentage >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const getGradeColor = (grade: string) => {
    const gradeColors: Record<string, string> = {
      'A+': 'bg-green-100 text-green-800',
      'A': 'bg-green-100 text-green-800',
      'B+': 'bg-primary/10 text-primary',
      'B': 'bg-primary/10 text-primary',
      'C+': 'bg-yellow-100 text-yellow-800',
      'C': 'bg-yellow-100 text-yellow-800',
      'D': 'bg-orange-100 text-orange-800',
      'F': 'bg-red-100 text-red-800'
    }
    return gradeColors[grade] || 'bg-gray-100 text-gray-800'
  }

  if (loading) {
    return (
      <div className="container mx-auto py-8 px-4 flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Analyzing your test results...</p>
        </div>
      </div>
    )
  }

  if (!analysis) {
    return (
      <div className="container mx-auto py-8 px-4 text-center">
        <p>Test results not found</p>
        <Button onClick={() => router.push('/mock-tests')} className="mt-4">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Tests
        </Button>
      </div>
    )
  }

  const getGradeFromPercentage = (percentage: number) => {
    if (percentage >= 90) return 'A+'
    if (percentage >= 80) return 'A'
    if (percentage >= 70) return 'B+'
    if (percentage >= 60) return 'B'
    if (percentage >= 50) return 'C+'
    if (percentage >= 40) return 'C'
    if (percentage >= 30) return 'D'
    return 'F'
  }

  const scorePercentage = analysis.percentage

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            onClick={() => router.push('/mock-tests')}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Tests
          </Button>
          <h1 className="text-3xl font-bold">Test Results</h1>
        </div>
        <Badge className={`text-lg px-4 py-2 ${getGradeColor(getGradeFromPercentage(scorePercentage))}`}>
          Grade: {getGradeFromPercentage(scorePercentage)}
        </Badge>
      </div>

      {/* Overall Score Card */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-6 w-6" />
            Overall Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className={`text-4xl font-bold ${getScoreColor(scorePercentage)}`}>
                {analysis.total_score}/{analysis.max_score}
              </div>
              <p className="text-muted-foreground">Total Score</p>
            </div>
            <div className="text-center">
              <div className={`text-4xl font-bold ${getScoreColor(scorePercentage)}`}>
                {scorePercentage.toFixed(1)}%
              </div>
              <p className="text-muted-foreground">Percentage</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-600">
                {formatTime(analysis.time_taken)}
              </div>
              <p className="text-muted-foreground">Time Taken</p>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-purple-600">
                {analysis.question_feedback.length}
              </div>
              <p className="text-muted-foreground">Questions</p>
            </div>
          </div>
          
          <div className="mt-6">
            <div className="flex justify-between text-sm text-muted-foreground mb-2">
              <span>Score Progress</span>
              <span>{scorePercentage.toFixed(1)}%</span>
            </div>
            <Progress value={scorePercentage} className="h-3" />
          </div>
        </CardContent>
      </Card>

      {/* Question-wise Feedback */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-6 w-6" />
            Question-wise Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {analysis.question_feedback.map((feedback: AnswerFeedback, index: number) => (
              <Card key={feedback.question_id} className="border-l-4 border-l-blue-500">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-lg">Question {index + 1}</CardTitle>
                    <div className="flex items-center gap-2">
                      {feedback.is_correct ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : (
                        <XCircle className="h-5 w-5 text-red-600" />
                      )}
                      <Badge variant={feedback.is_correct ? "default" : "destructive"}>
                        {feedback.marks_awarded}/{feedback.max_marks}
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <p className="font-medium text-muted-foreground">Question:</p>
                      <p>{feedback.question}</p>
                    </div>
                    
                    <div>
                      <p className="font-medium text-muted-foreground">Your Answer:</p>
                      <p className="bg-secondary p-3 rounded-md border">
                        {feedback.user_answer || "No answer provided"}
                      </p>
                    </div>
                    
                    {feedback.correct_answer && (
                      <div>
                        <p className="font-medium text-muted-foreground">Correct Answer:</p>
                        <p className="bg-secondary p-3 rounded-md border border-green-200">
                          {feedback.correct_answer}
                        </p>
                      </div>
                    )}
                    
                    {feedback.feedback && (
                      <div>
                        <p className="font-medium text-muted-foreground">AI Feedback:</p>
                        <p className="bg-primary text-primary-foreground p-3 rounded-md border border-primary/20">
                          {feedback.feedback}
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Strengths and Improvements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-green-600">
              <TrendingUp className="h-6 w-6" />
              Strengths
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {analysis.strengths.map((strength: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">{strength}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-orange-600">
              <TrendingDown className="h-6 w-6" />
              Areas for Improvement
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {analysis.improvements.map((improvement: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <XCircle className="h-4 w-4 text-orange-600 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">{improvement}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Study Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-6 w-6" />
            Study Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3">
            {analysis.study_recommendations.map((recommendation: string, index: number) => (
              <li key={index} className="flex items-start gap-3 p-3 bg-secondary rounded-md border">
                <Clock className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span className="text-sm">{recommendation}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex gap-4 mt-8 justify-center">
        <Button onClick={() => router.push('/mock-tests')} variant="outline">
          Take Another Test
        </Button>
        <Button onClick={() => router.push('/dashboard')}>
          Back to Dashboard
        </Button>
      </div>
    </div>
  )
}
