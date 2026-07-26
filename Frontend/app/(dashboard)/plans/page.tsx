"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { studyAPI } from "@/lib/api";
import { getErrorMessage } from "@/lib/errors";
import { Loader2, Calendar, Trash2, BookOpen, CheckCircle2, Sparkles } from "lucide-react";

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

interface Task {
  subject: string;
  topic: string;
  activity: string;
  minutes: number;
  resource?: string;
  completed?: boolean;
}

interface DayPlan {
  day: string;
  tasks: Task[];
}

interface WeekPlan {
  week: number;
  focus: string;
  days: DayPlan[];
}

interface StudyPlan {
  plan_id: string;
  title: string;
  exam_date?: string;
  subjects: string[];
  weak_topics: string[];
  hours_per_day: number;
  weeks: number;
  plan: { weeks: WeekPlan[] };
  created_at: string;
}

export default function PlansPage() {
  const { toast } = useToast();
  const [plans, setPlans] = useState<StudyPlan[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<StudyPlan | null>(null);
  const [form, setForm] = useState({
    title: "",
    exam_date: "",
    subjects: "",
    weak_topics: "",
    hours_per_day: 4,
    weeks: 4,
  });

  useEffect(() => {
    loadPlans();
  }, []);

  const loadPlans = async () => {
    setIsLoading(true);
    try {
      const data = await studyAPI.listStudyPlans();
      setPlans(data.plans || []);
      if (data.plans?.length && !selectedPlan) {
        setSelectedPlan(data.plans[0]);
      }
    } catch (error) {
      toast({ title: "Could not load plans", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    try {
      const created = await studyAPI.createStudyPlan({
        title: form.title,
        exam_date: form.exam_date || undefined,
        subjects: form.subjects.split(",").map((s) => s.trim()).filter(Boolean),
        weak_topics: form.weak_topics.split(",").map((s) => s.trim()).filter(Boolean),
        hours_per_day: form.hours_per_day,
        weeks: form.weeks,
      });
      toast({ title: "Study plan created" });
      setForm({ title: "", exam_date: "", subjects: "", weak_topics: "", hours_per_day: 4, weeks: 4 });
      await loadPlans();
      setSelectedPlan(created);
    } catch (error) {
      toast({ title: "Could not create plan", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDelete = async (planId: string) => {
    if (!confirm("Delete this study plan?")) return;
    try {
      await studyAPI.deleteStudyPlan(planId);
      toast({ title: "Plan deleted" });
      await loadPlans();
      if (selectedPlan?.plan_id === planId) setSelectedPlan(null);
    } catch (error) {
      toast({ title: "Could not delete plan", description: getErrorMessage(error), variant: "destructive" });
    }
  };

  const toggleTask = async (weekIndex: number, dayName: string, taskIndex: number, completed: boolean) => {
    if (!selectedPlan) return;
    try {
      await studyAPI.updatePlanProgress(selectedPlan.plan_id, weekIndex + 1, dayName, taskIndex, completed);
      const updated = { ...selectedPlan };
      updated.plan.weeks[weekIndex].days[DAYS.indexOf(dayName)].tasks[taskIndex].completed = completed;
      setSelectedPlan(updated);
      setPlans((prev) => prev.map((p) => (p.plan_id === updated.plan_id ? updated : p)));
    } catch (error) {
      toast({ title: "Could not update progress", description: getErrorMessage(error), variant: "destructive" });
    }
  };

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Study Planner</h1>
        <p className="text-sm text-muted-foreground mt-1">
          AI-generated weekly study plans tailored to your exam date and weak topics.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              New plan
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="space-y-3">
              <div className="space-y-2">
                <Label htmlFor="title">Plan title</Label>
                <Input id="title" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="e.g. JEE Main 3-month sprint" required />
              </div>

              <div className="space-y-2">
                <Label htmlFor="exam_date">Exam date</Label>
                <Input id="exam_date" type="date" value={form.exam_date} onChange={(e) => setForm({ ...form, exam_date: e.target.value })} />
              </div>

              <div className="space-y-2">
                <Label htmlFor="subjects">Subjects (comma-separated)</Label>
                <Input id="subjects" value={form.subjects} onChange={(e) => setForm({ ...form, subjects: e.target.value })} placeholder="Physics, Chemistry, Maths" />
              </div>

              <div className="space-y-2">
                <Label htmlFor="weak_topics">Weak topics</Label>
                <Input id="weak_topics" value={form.weak_topics} onChange={(e) => setForm({ ...form, weak_topics: e.target.value })} placeholder="Calculus, Organic Chemistry" />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="hours">Hours/day</Label>
                  <Input id="hours" type="number" min={1} max={12} value={form.hours_per_day} onChange={(e) => setForm({ ...form, hours_per_day: parseInt(e.target.value) || 4 })} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="weeks">Weeks</Label>
                  <Input id="weeks" type="number" min={1} max={12} value={form.weeks} onChange={(e) => setForm({ ...form, weeks: parseInt(e.target.value) || 4 })} />
                </div>
              </div>

              <Button type="submit" className="w-full" disabled={isGenerating}>
                {isGenerating && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Generate plan
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : plans.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                <BookOpen className="h-10 w-10 mx-auto mb-3 opacity-50" />
                <p>No study plans yet. Create one on the left.</p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                {plans.map((plan) => (
                  <Button
                    key={plan.plan_id}
                    variant={selectedPlan?.plan_id === plan.plan_id ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedPlan(plan)}
                    className="shrink-0"
                  >
                    {plan.title}
                  </Button>
                ))}
              </div>

              {selectedPlan && (
                <Card>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="text-lg">{selectedPlan.title}</CardTitle>
                        <p className="text-xs text-muted-foreground mt-1">
                          {selectedPlan.subjects.join(", ")} · {selectedPlan.weeks} weeks · {selectedPlan.hours_per_day}h/day
                        </p>
                      </div>
                      <Button variant="ghost" size="icon" onClick={() => handleDelete(selectedPlan.plan_id)}>
                        <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="week1">
                      <TabsList className="mb-4 flex-wrap h-auto">
                        {selectedPlan.plan.weeks.map((w) => (
                          <TabsTrigger key={w.week} value={`week${w.week}`}>
                            Week {w.week}
                          </TabsTrigger>
                        ))}
                      </TabsList>

                      {selectedPlan.plan.weeks.map((week, weekIndex) => (
                        <TabsContent key={week.week} value={`week${week.week}`} className="space-y-4">
                          <p className="text-sm font-medium">{week.focus}</p>
                          <div className="grid gap-4">
                            {DAYS.map((dayName) => {
                              const day = week.days.find((d) => d.day === dayName) || { day: dayName, tasks: [] };
                              return (
                                <div key={dayName} className="rounded-md border p-3">
                                  <p className="text-sm font-medium mb-2">{dayName}</p>
                                  {day.tasks.length === 0 ? (
                                    <p className="text-xs text-muted-foreground">No tasks</p>
                                  ) : (
                                    <div className="space-y-2">
                                      {day.tasks.map((task, taskIndex) => (
                                        <div key={taskIndex} className="flex items-start gap-2 text-sm">
                                          <Checkbox
                                            checked={task.completed || false}
                                            onCheckedChange={(checked) => toggleTask(weekIndex, dayName, taskIndex, checked === true)}
                                          />
                                          <div className="flex-1">
                                            <p className={task.completed ? "line-through text-muted-foreground" : ""}>
                                              {task.activity} · {task.subject}: {task.topic}
                                            </p>
                                            <div className="flex items-center gap-2 mt-1">
                                              <Badge variant="outline" className="text-[10px]">
                                                {task.minutes} min
                                              </Badge>
                                              {task.resource && (
                                                <Badge variant="secondary" className="text-[10px]">
                                                  {task.resource}
                                                </Badge>
                                              )}
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </TabsContent>
                      ))}
                    </Tabs>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
