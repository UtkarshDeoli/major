"use client"

import { useEffect, useMemo, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { Loader2, TrendingUp, Target, BookOpen, Flame, ClipboardList, ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"
import { classAPI } from "@/lib/api"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { StatsCard } from "@/components/dashboard/analytics/stats-card"
import { SubjectChart } from "@/components/dashboard/analytics/subject-chart"
import { StrengthWeaknessPanel } from "@/components/dashboard/analytics/strength-weakness-panel"
import { AnalyticsCard } from "@/components/dashboard/analytics/analytics-card"

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

interface StudentAnalytics {
  email: string
  name?: string
  class_ids?: string[]
  tests_taken: number
  average_score: number
  best_score: number
  documents: number
  study_streak: number
  subject_wise?: SubjectWise[]
  recent_submissions?: any[]
}

export default function StudentProfilePage() {
  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <StudentProfileContent />
    </RoleGuard>
  )
}

function StudentProfileContent() {
  const { email } = useParams() as { email: string }
  const router = useRouter()
  const { toast } = useToast()
  const [analytics, setAnalytics] = useState<StudentAnalytics | null>(null)
  const [classes, setClasses] = useState<{ id: string; name: string }[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const decoded = decodeURIComponent(email)
    const fetchData = async () => {
      setLoading(true)
      try {
        const [analyticsData, classesData] = await Promise.all([
          classAPI.getStudentAnalytics(decoded),
          classAPI.listClasses(),
        ])
        setAnalytics(analyticsData)
        setClasses((classesData || []) as { id: string; name: string }[])
      } catch (e) {
        toast({ title: "Couldn't load student profile", description: getErrorMessage(e), variant: "destructive" })
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [email, toast])

  const subjectData = useMemo(() => {
    return (analytics?.subject_wise || []).map((s) => ({
      subject: s.subject,
      score: Math.max(0, Math.min(100, Math.round(s.average_score))),
    }))
  }, [analytics])

  const feedbackData = useMemo(() => {
    return (analytics?.subject_wise || []).map((s) => ({
      subject: s.subject,
      strengths: s.strengths || [],
      weaknesses: s.weaknesses || [],
    }))
  }, [analytics])

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <Button variant="outline" onClick={() => router.push("/students")}><ArrowLeft className="h-4 w-4 mr-2" />Back to students</Button>
        <p className="mt-4 text-muted-foreground">Could not load student profile.</p>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div className="flex items-center gap-3">
        <Button variant="outline" size="sm" onClick={() => router.push("/students")}><ArrowLeft className="h-4 w-4 mr-2" />Back</Button>
      </div>

      <div>
        <h1 className="text-xl font-semibold">{analytics.name || analytics.email.split("@")[0]}</h1>
        <p className="text-sm text-muted-foreground font-mono">{analytics.email}</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {classes.filter((c) => analytics.class_ids?.includes(c.id)).map((c) => (
          <Badge key={c.id} variant="secondary">{c.name}</Badge>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatsCard label="Tests Taken" value={analytics.tests_taken || 0} icon={Target} isLoading={loading} />
        <StatsCard label="Average Score" value={`${Math.round(analytics.average_score || 0)}%`} icon={TrendingUp} isLoading={loading} />
        <StatsCard label="Best Score" value={`${Math.round(analytics.best_score || 0)}%`} icon={Flame} isLoading={loading} />
        <StatsCard label="Documents" value={analytics.documents || 0} icon={BookOpen} isLoading={loading} />
      </div>

      <Button onClick={() => router.push(`/mock-tests?student=${encodeURIComponent(analytics.email)}`)}>
        <ClipboardList className="h-4 w-4 mr-2" />Create Targeted Test
      </Button>

      <div className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <AnalyticsCard title="Subject Performance">
            <SubjectChart data={subjectData} />
          </AnalyticsCard>
        </div>
        <AnalyticsCard title="Strengths & Focus Areas">
          <StrengthWeaknessPanel data={feedbackData} />
        </AnalyticsCard>
      </div>

      {analytics.recent_submissions && analytics.recent_submissions.length > 0 && (
        <AnalyticsCard title="Recent Submissions">
          <div className="space-y-2">
            {analytics.recent_submissions.map((sub) => (
              <div key={sub.submission_id} className="flex items-center justify-between rounded-md border p-3">
                <div>
                  <p className="text-sm font-medium">{sub.subject || "General"}</p>
                  <p className="text-xs text-muted-foreground">{sub.created_at ? new Date(sub.created_at).toLocaleDateString() : "—"}</p>
                </div>
                <span className="text-sm font-semibold">{Math.round(sub.percentage || 0)}%</span>
              </div>
            ))}
          </div>
        </AnalyticsCard>
      )}
    </div>
  )
}
