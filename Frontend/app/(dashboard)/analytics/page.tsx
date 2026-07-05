"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { SubjectChart } from "@/components/dashboard/analytics/subject-chart"
import { StrengthWeaknessPanel } from "@/components/dashboard/analytics/strength-weakness-panel"
import { ActivityChart } from "@/components/dashboard/analytics/activity-chart"
import { StatsCard } from "@/components/dashboard/analytics/stats-card"
import { ProgressRing } from "@/components/dashboard/analytics/progress-ring"
import { AnalyticsCard } from "@/components/dashboard/analytics/analytics-card"
import { analyticsAPI, mockTestAPI } from "@/lib/api"
import { TrendingUp, Target, BookOpen, Flame } from "lucide-react"
import { motion } from "framer-motion"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"

interface SectionAnalytics {
  section: string
  attempts: number
  accuracy: number
}
interface SubjectWise {
  subject: string
  tests_taken: number
  average_score: number
  strengths: string[]
  weaknesses: string[]
  sections?: SectionAnalytics[]
  weak_sections?: string[]
}

interface WeeklyActivity {
  day: string
  hours: number
  quizzes: number
}

interface TrendDelta {
  score_delta: number
  tests_delta: number
}

interface StudentAnalytics {
  tests_taken: number
  average_score: number
  documents?: number
  study_streak?: number
  weekly_activity?: WeeklyActivity[]
  completion?: number
  consistency?: number
  trend?: TrendDelta
  subject_wise?: SubjectWise[]
}

export default function AnalyticsPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [subjectData, setSubjectData] = useState<Array<{ subject: string; score: number }>>([])
  const [feedbackData, setFeedbackData] = useState<
    Array<{ subject: string; strengths: string[]; weaknesses: string[] }>
  >([])
  const [totalTests, setTotalTests] = useState(0)
  const [avgScore, setAvgScore] = useState(0)
  const [documents, setDocuments] = useState(0)
  const [studyStreak, setStudyStreak] = useState(0)
  const [weeklyActivity, setWeeklyActivity] = useState<WeeklyActivity[]>([])
  const [completion, setCompletion] = useState(0)
  const [consistency, setConsistency] = useState(0)
  const [trend, setTrend] = useState<TrendDelta | null>(null)
  const [sectionWeakness, setSectionWeakness] = useState<Array<{ subject: string; sections: SectionAnalytics[]; weak: string[] }>>([])
  const { toast } = useToast()

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const analytics = await analyticsAPI.getStudentAnalytics().catch(() => null)
        if (analytics) {
          const data = analytics as StudentAnalytics
          const subjects = (data.subject_wise || []).map((s) => ({
            subject: s.subject,
            // Clamp display score to 0-100 so the chart axis stays meaningful.
            score: Math.max(0, Math.min(100, Math.round(s.average_score))),
          }))
          setSubjectData(subjects)
          setFeedbackData(
            (data.subject_wise || []).map((s) => ({
              subject: s.subject,
              strengths: s.strengths || [],
              weaknesses: s.weaknesses || [],
            }))
          )
          setTotalTests(data.tests_taken || 0)
          setAvgScore(Math.round(data.average_score || 0))
          setDocuments(data.documents || 0)
          setStudyStreak(data.study_streak || 0)
          setWeeklyActivity(data.weekly_activity || [])
          setCompletion(Math.round(data.completion || 0))
          setConsistency(Math.round(data.consistency || 0))
          setTrend(data.trend || null)
          setSectionWeakness(
            (data.subject_wise || [])
              .filter((s) => (s.sections?.length ?? 0) > 0)
              .map((s) => ({ subject: s.subject, sections: s.sections || [], weak: s.weak_sections || [] }))
          )
        } else {
          // Fallback when the analytics endpoint is unavailable: derive what we can from mock tests.
          const tests = await mockTestAPI.listMockTests()
          const submitted = (tests || []).filter((t: any) => t.latest_submission)
          setTotalTests(submitted.length)
          if (submitted.length > 0) {
            const avg =
              submitted.reduce(
                (s: number, t: any) => s + Math.max(0, Math.min(100, t.latest_submission?.percentage || 0)),
                0
              ) / submitted.length
            setAvgScore(Math.round(avg))
          }
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

  const formatTrend = (delta: number, suffix = "") => ({
    value: `${delta > 0 ? "+" : ""}${delta}${suffix}`,
    positive: delta >= 0,
  })

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h1 className="text-2xl font-semibold tracking-tight">Analytics.</h1>
        <p className="text-sm text-muted-foreground mt-1">Track your performance and identify weak areas.</p>
      </motion.div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatsCard
          label="Tests Taken"
          value={totalTests}
          icon={Target}
          trend={trend ? formatTrend(trend.tests_delta) : undefined}
          delay={0.05}
          isLoading={isLoading}
        />
        <StatsCard
          label="Average Score"
          value={`${avgScore}%`}
          icon={TrendingUp}
          trend={trend ? formatTrend(trend.score_delta, "%") : undefined}
          delay={0.1}
          isLoading={isLoading}
        />
        <StatsCard
          label="Documents"
          value={documents}
          icon={BookOpen}
          delay={0.15}
          isLoading={isLoading}
        />
        <StatsCard
          label="Study Streak"
          value={`${studyStreak} ${studyStreak === 1 ? "day" : "days"}`}
          icon={Flame}
          delay={0.2}
          isLoading={isLoading}
        />
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <AnalyticsCard title="Subject Performance" delay={0.25}>
            <SubjectChart data={subjectData} />
          </AnalyticsCard>
        </div>

        <AnalyticsCard
          title="Strengths & Focus Areas"
          subtitle="Real feedback from your last tests"
          delay={0.3}
        >
          <StrengthWeaknessPanel data={feedbackData} />
        </AnalyticsCard>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <AnalyticsCard
          title="Progress"
          subtitle="Completion rate and weekly consistency"
          delay={0.35}
          bodyClassName="flex items-center justify-around py-4"
        >
          <ProgressRing value={completion} label="Completion" suffix="%" color="hsl(160, 84%, 39%)" />
          <ProgressRing value={consistency} label="Consistency" suffix="%" color="hsl(221, 83%, 53%)" />
        </AnalyticsCard>

        <div className="md:col-span-2">
          <AnalyticsCard title="Weekly Activity" delay={0.4}>
            <ActivityChart data={weeklyActivity} />
          </AnalyticsCard>
        </div>
      </div>

      {/* Section-level weakness within each subject */}
      {sectionWeakness.length > 0 && (
        <AnalyticsCard title="Weak Sections by Subject" subtitle="Where to focus next, per chapter/unit" delay={0.45}>
          <div className="space-y-4">
            {sectionWeakness.map((sw) => (
              <div key={sw.subject} className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium">{sw.subject}</h4>
                  {sw.weak.length > 0 && (
                    <span className="text-[10px] rounded bg-rose-500/10 text-rose-600 px-1.5 py-0.5">
                      {sw.weak.length} weak
                    </span>
                  )}
                  {sw.weak.length > 0 && (
                    <Link
                      href={`/mock-tests?subject=${encodeURIComponent(sw.subject)}&focus=${encodeURIComponent(sw.weak.join(", "))}&weak=1`}
                      className="ml-auto text-[11px] text-primary hover:underline"
                    >
                      Practice weak areas →
                    </Link>
                  )}
                </div>
                <div className="space-y-1">
                  {sw.sections.map((sec) => (
                    <div key={sec.section} className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground w-32 truncate">{sec.section}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.max(0, Math.min(100, sec.accuracy))}%`,
                            backgroundColor: sec.accuracy < 50 ? "hsl(0, 84%, 60%)" : sec.accuracy < 75 ? "hsl(38, 92%, 50%)" : "hsl(160, 84%, 39%)",
                          }}
                        />
                      </div>
                      <span className="text-[11px] text-muted-foreground w-10 text-right">{sec.accuracy}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </AnalyticsCard>
      )}
    </div>
  )
}
