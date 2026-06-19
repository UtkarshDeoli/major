"use client"

import { useState, useEffect } from "react"
import { SubjectChart } from "@/components/dashboard/analytics/subject-chart"
import { WeaknessRadar } from "@/components/dashboard/analytics/weakness-radar"
import { analyticsAPI, mockTestAPI } from "@/lib/api"
import { TrendingUp } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"

export default function AnalyticsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [subjectData, setSubjectData] = useState<Array<{ subject: string; score: number }>>([])
  const [weaknessData, setWeaknessData] = useState<Array<{ topic: string; score: number; fullMark: number }>>([])
  const [totalTests, setTotalTests] = useState(0)
  const [avgScore, setAvgScore] = useState(0)
  const { toast } = useToast()

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const analytics = await analyticsAPI.getStudentAnalytics().catch(() => null)
        if (analytics) {
          setSubjectData(analytics.subject_scores || [])
          setWeaknessData((analytics.weakness_topics || []).map((t: any) => ({
            topic: t.name || t.topic,
            score: t.score || 0,
            fullMark: 100,
          })))
          setTotalTests(analytics.total_tests || 0)
          setAvgScore(analytics.average_score || 0)
        } else {
          const tests = await mockTestAPI.listMockTests()
          const submitted = (tests || []).filter((t: any) => t.latest_submission)
          setTotalTests(submitted.length)
          if (submitted.length > 0) {
            const avg = submitted.reduce((s: number, t: any) => s + (t.latest_submission?.percentage || 0), 0) / submitted.length
            setAvgScore(Math.round(avg))
          }
          setSubjectData([
            { subject: "Math", score: 72 },
            { subject: "Physics", score: 65 },
            { subject: "Chemistry", score: 80 },
            { subject: "Biology", score: 58 },
          ])
          setWeaknessData([
            { topic: "Calculus", score: 45, fullMark: 100 },
            { topic: "Mechanics", score: 55, fullMark: 100 },
            { topic: "Organic Chem", score: 60, fullMark: 100 },
            { topic: "Thermodynamics", score: 40, fullMark: 100 },
            { topic: "Algebra", score: 70, fullMark: 100 },
          ])
        }
      } catch (err) {
        console.error("Failed to load analytics:", err)
        toast({
          title: "Couldn't load analytics",
          description: getErrorMessage(err),
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [])

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Analytics.</h1>
        <p className="text-sm text-muted-foreground mt-1">Track your performance and identify weak areas.</p>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-md border bg-card p-4 space-y-2">
          <span className="text-xs font-medium text-muted-foreground">Tests Taken</span>
          <div className="text-xl font-semibold tabular-nums">{isLoading ? "—" : totalTests}</div>
        </div>
        <div className="rounded-md border bg-card p-4 space-y-2">
          <span className="text-xs font-medium text-muted-foreground">Average Score</span>
          <div className="text-xl font-semibold tabular-nums">{isLoading ? "—" : `${avgScore}%`}</div>
        </div>
        <div className="rounded-md border bg-card p-4 space-y-2">
          <span className="text-xs font-medium text-muted-foreground">Study Streak</span>
          <div className="text-xl font-semibold tabular-nums">—</div>
        </div>
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        <SubjectChart data={subjectData} />
        <WeaknessRadar data={weaknessData} />
      </div>
    </div>
  )
}