"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  BookOpen,
  MessageSquare,
  Target,
  TrendingUp,
  Clock,
  Award,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Container from "@/components/global/container";
import { BentoGrid } from "@/components/ui/bento-grid";
import { useDashboard } from "@/lib/context/dashboard-context";
import { ActiveStudyCard } from "@/components/dashboard/active-study-card";
import { SubjectCard } from "@/components/dashboard/subject-card";
import { CollectionsPanel } from "@/components/dashboard/collections-panel";
import { ExamSetupDialog } from "@/components/dashboard/exam-setup-dialog";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/lib/context/auth-context";
import { mockTestAPI } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function DashboardPage() {
  return <DashboardContent />;
}

function DashboardContent() {
  const router = useRouter();
  const { toast } = useToast();
  const { user } = useAuth();
  const { activeExam, refreshExams } = useDashboard();

  const [panelOpen, setPanelOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedSubject, setSelectedSubject] = useState<
    { id: string; name: string } | null
  >(null);
  const [assignedTests, setAssignedTests] = useState<Array<{ test_id: string; title: string; created_at: string; created_by?: string }>>([]);

  useEffect(() => {
    refreshExams();
  }, [refreshExams]);

  useEffect(() => {
    if (!user?.email) return;
    mockTestAPI.listMockTests()
      .then((tests) => {
        const mine = (tests || []).filter((t: any) => t.assigned_to === user.email);
        setAssignedTests(mine);
      })
      .catch((err) => console.error("Failed to load assigned tests:", err));
  }, [user?.email]);

  const handleAddExam = () => {
    setDialogOpen(true);
  };

  const handleContinueSession = () => {
    if (activeExam) {
      router.push("/chat");
    }
  };

  const handleSubjectClick = (subject: { id: string; name: string }) => {
    setSelectedSubject(subject);
    setPanelOpen(true);
  };

  const handleExamCreated = () => {
    refreshExams();
    toast({
      title: "Exam created",
      description: "Your study goal has been set up successfully.",
    });
  };

  const handleChatWithExam = (examId: string) => {
    router.push(`/chat?exam=${examId}`);
  };

  const userName = "Student";

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-6 px-4 lg:px-8">
      {/* Zone 1: Welcome bar */}
      <Container>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              Welcome back, {userName}! 🎓
            </h1>
            <p className="text-muted-foreground mt-1">
              Here&apos;s what&apos;s happening with your studies today.
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => router.push("/chat")}>
              <MessageSquare className="h-4 w-4 mr-2" />
              New Chat
            </Button>
            <Button onClick={() => router.push("/test?tab=mock")}>
              <Target className="h-4 w-4 mr-2" />
              Take Test
            </Button>
          </div>
        </div>
      </Container>

      {/* Zone 2: Active Study */}
      <ActiveStudyCard
        onAddExam={handleAddExam}
        onContinueSession={handleContinueSession}
      />

      {/* Zone 3: Subjects */}
      {activeExam?.subjects && activeExam.subjects.length > 0 && (
        <>
          <Container delay={0.2}>
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              Subjects
            </h2>
          </Container>
          <BentoGrid columns={3}>
            {activeExam.subjects.map((s, i) => (
              <SubjectCard
                key={s.id}
                subject={s}
                index={i}
                onClick={() => handleSubjectClick({ id: s.id, name: s.name })}
              />
            ))}
          </BentoGrid>
        </>
      )}

      {/* Zone 4: Stats */}
      <Container delay={0.4}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            {
              label: "Documents",
              value: "—",
              icon: BookOpen,
              sub: "Total uploaded",
            },
            {
              label: "Chat Sessions",
              value: "—",
              icon: MessageSquare,
              sub: "Active conversations",
            },
            {
              label: "Mock Tests",
              value: "—",
              icon: Target,
              sub: "Tests taken",
            },
            {
              label: "Avg Score",
              value: "—",
              icon: TrendingUp,
              sub: "Across all tests",
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="rounded-xl border bg-card p-4 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{stat.label}</span>
                <stat.icon className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">{stat.sub}</p>
            </div>
          ))}
        </div>
      </Container>

      {/* Zone 5: Assigned Tests */}
      {assignedTests.length > 0 && (
        <Container delay={0.6}>
          <Card>
            <CardHeader>
              <CardTitle>Assigned Tests</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {assignedTests.map((test) => (
                  <div key={test.test_id} className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">{test.title}</p>
                      <p className="text-xs text-muted-foreground">From {test.created_by}</p>
                    </div>
                    <Button size="sm" onClick={() => router.push(`/test/quiz?testId=${test.test_id}`)}>
                      Start
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </Container>
      )}

      {/* Zone 6: My Collections */}
      <Container delay={0.7}>
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">My Collections</h2>
          <div className="bg-muted/50 rounded-xl p-8 text-center">
            <Zap className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">
              Collections will appear here once you add materials to your
              subjects.
            </p>
          </div>
        </div>
      </Container>

      {/* Panel + Dialog */}
      <CollectionsPanel
        exam={activeExam}
        open={panelOpen}
        onOpenChange={setPanelOpen}
        onChat={handleChatWithExam}
      />
      <ExamSetupDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onExamCreated={handleExamCreated}
      />
    </div>
  );
}
