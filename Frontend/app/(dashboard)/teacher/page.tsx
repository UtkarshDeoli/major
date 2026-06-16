"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Users, Plus, Trash2, GraduationCap, BarChart3, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuth } from "@/lib/context/auth-context";
import { teacherAPI, analyticsAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import Container from "@/components/global/container";
import RoleGuard from "@/components/auth/route-protection/role-guard";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface ManagedStudent {
  id: string;
  email: string;
  name?: string;
  tests_taken: number;
  average_score: number;
  last_active_at?: string;
  strengths: string[];
  weaknesses: string[];
}

interface TeacherAnalytics {
  total_students: number;
  active_students: number;
  total_tests_taken: number;
  class_average: number;
  student_analytics: ManagedStudent[];
}

function TeacherDashboardContent() {
  const { user } = useAuth();
  const { toast } = useToast();
  const router = useRouter();

  const [students, setStudents] = useState<ManagedStudent[]>([]);
  const [analytics, setAnalytics] = useState<TeacherAnalytics | null>(null);
  const [newStudentEmail, setNewStudentEmail] = useState("");
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [studentsData, analyticsData] = await Promise.all([
        teacherAPI.listManagedStudents(),
        analyticsAPI.getTeacherAnalytics(),
      ]);
      setStudents(studentsData as ManagedStudent[]);
      setAnalytics(analyticsData as TeacherAnalytics);
    } catch (error) {
      console.error("Failed to load teacher data:", error);
      toast({
        title: "Failed to load data",
        description: "Could not retrieve your students or analytics.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleAddStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newStudentEmail.trim()) return;
    try {
      await teacherAPI.manageStudent(newStudentEmail.trim());
      toast({ title: "Student linked", description: `${newStudentEmail} is now managed by you.` });
      setNewStudentEmail("");
      await fetchData();
    } catch (error) {
      toast({
        title: "Failed to link student",
        description: error instanceof Error ? error.message : "Please try again",
        variant: "destructive",
      });
    }
  };

  const handleRemoveStudent = async (email: string) => {
    try {
      await teacherAPI.unmanageStudent(email);
      toast({ title: "Student unlinked", description: `${email} is no longer managed by you.` });
      await fetchData();
    } catch (error) {
      toast({
        title: "Failed to unlink student",
        description: error instanceof Error ? error.message : "Please try again",
        variant: "destructive",
      });
    }
  };

  const chartData =
    analytics?.student_analytics.map((s) => ({
      name: s.name || s.email.split("@")[0],
      score: Number.isFinite(s.average_score) ? s.average_score : 0,
      tests: s.tests_taken,
    })) || [];

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-6 px-4 lg:px-8">
      <Container>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
              <GraduationCap className="h-8 w-8 text-primary" />
              Teacher Dashboard
            </h1>
            <p className="text-muted-foreground mt-1">
              Manage your students, view their progress, and create targeted assessments.
            </p>
          </div>
          <Button onClick={() => router.push("/test?tab=mock")}>
            <BookOpen className="h-4 w-4 mr-2" />
            Create Test
          </Button>
        </div>
      </Container>

      {/* Stats */}
      <Container delay={0.1}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Students</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{analytics?.total_students ?? "—"}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Active Students</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{analytics?.active_students ?? "—"}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Tests Taken</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{analytics?.total_tests_taken ?? "—"}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Class Average</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {analytics?.class_average !== undefined ? `${analytics.class_average.toFixed(1)}%` : "—"}
              </div>
            </CardContent>
          </Card>
        </div>
      </Container>

      <Tabs defaultValue="students" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="students">
            <Users className="h-4 w-4 mr-2" />
            My Students
          </TabsTrigger>
          <TabsTrigger value="analytics">
            <BarChart3 className="h-4 w-4 mr-2" />
            Class Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="students" className="space-y-6">
          <Container delay={0.2}>
            <Card>
              <CardHeader>
                <CardTitle>Link a Student</CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleAddStudent} className="flex gap-2">
                  <Input
                    type="email"
                    placeholder="student@example.com"
                    value={newStudentEmail}
                    onChange={(e) => setNewStudentEmail(e.target.value)}
                    className="flex-1"
                    required
                  />
                  <Button type="submit" disabled={!newStudentEmail.trim()}>
                    <Plus className="h-4 w-4 mr-2" />
                    Link Student
                  </Button>
                </form>
              </CardContent>
            </Card>
          </Container>

          <Container delay={0.3}>
            <div className="grid gap-4">
              {loading ? (
                <div className="flex justify-center py-12">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                </div>
              ) : students.length === 0 ? (
                <Card>
                  <CardContent className="py-12 text-center text-muted-foreground">
                    <Users className="h-10 w-10 mx-auto mb-3 opacity-50" />
                    <p>No students linked yet.</p>
                    <p className="text-sm">Add a student above to start monitoring their progress.</p>
                  </CardContent>
                </Card>
              ) : (
                students.map((student) => (
                  <Card key={student.id} className="group">
                    <CardContent className="p-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div>
                          <h3 className="font-semibold">{student.name || student.email}</h3>
                          <p className="text-sm text-muted-foreground">{student.email}</p>
                          <div className="flex flex-wrap gap-2 mt-2 text-xs text-muted-foreground">
                            <span>Tests: {student.tests_taken}</span>
                            <span>•</span>
                            <span>Avg: {Number.isFinite(student.average_score) ? `${student.average_score}%` : "—"}</span>
                            {student.last_active_at && (
                              <>
                                <span>•</span>
                                <span>Last active: {new Date(student.last_active_at).toLocaleDateString()}</span>
                              </>
                            )}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="shrink-0 text-muted-foreground hover:text-destructive"
                          onClick={() => handleRemoveStudent(student.email)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </Container>
        </TabsContent>

        <TabsContent value="analytics">
          <Container delay={0.2}>
            <Card>
              <CardHeader>
                <CardTitle>Student Performance</CardTitle>
              </CardHeader>
              <CardContent>
                {chartData.length === 0 ? (
                  <div className="py-12 text-center text-muted-foreground">
                    <BarChart3 className="h-10 w-10 mx-auto mb-3 opacity-50" />
                    <p>No test data available yet.</p>
                    <p className="text-sm">Once students take tests, their scores will appear here.</p>
                  </div>
                ) : (
                  <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="opacity-20" />
                        <XAxis dataKey="name" stroke="currentColor" className="text-xs" />
                        <YAxis stroke="currentColor" className="text-xs" domain={[0, 100]} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "var(--background)",
                            borderColor: "var(--border)",
                            color: "var(--foreground)",
                          }}
                        />
                        <Bar dataKey="score" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} name="Avg Score (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card>
          </Container>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function TeacherPage() {
  return (
    <RoleGuard allowedRoles={["teacher"]}>
      <TeacherDashboardContent />
    </RoleGuard>
  );
}
