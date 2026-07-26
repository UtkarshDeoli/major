"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { studyAPI } from "@/lib/api";
import { getErrorMessage } from "@/lib/errors";
import { Play, Pause, RotateCcw, CheckCircle, Clock, Volume2, VolumeX } from "lucide-react";

const PRESETS = [15, 25, 45, 60];

export default function FocusPage() {
  const { toast } = useToast();
  const [duration, setDuration] = useState(25);
  const [task, setTask] = useState("");
  const [timeLeft, setTimeLeft] = useState(25 * 60);
  const [isRunning, setIsRunning] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [stats, setStats] = useState({ total_minutes: 0, weekly_minutes: 0, sessions_count: 0, completed_sessions: 0 });
  const [soundEnabled, setSoundEnabled] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadStats();
  }, []);

  useEffect(() => {
    if (isRunning && timeLeft > 0) {
      intervalRef.current = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            finishSession(true);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, timeLeft]);

  const loadStats = async () => {
    try {
      const data = await studyAPI.getFocusStats();
      setStats(data);
    } catch (error) {
      console.error("Focus stats error:", error);
    }
  };

  const startSession = async () => {
    if (!task.trim()) {
      toast({ title: "Enter a task", description: "What are you focusing on?", variant: "destructive" });
      return;
    }
    try {
      const data = await studyAPI.startFocusSession({ task, duration_minutes: duration });
      setSessionId(data.session_id);
      setTimeLeft(duration * 60);
      setIsRunning(true);
    } catch (error) {
      toast({ title: "Could not start session", description: getErrorMessage(error), variant: "destructive" });
    }
  };

  const finishSession = async (completed: boolean) => {
    setIsRunning(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (sessionId) {
      try {
        await studyAPI.endFocusSession(sessionId, { completed });
        toast({ title: completed ? "Focus session complete!" : "Session paused" });
      } catch (error) {
        toast({ title: "Could not save session", description: getErrorMessage(error), variant: "destructive" });
      }
    }
    setSessionId(null);
    setTimeLeft(duration * 60);
    await loadStats();
  };

  const togglePause = () => {
    setIsRunning((r) => !r);
  };

  const reset = () => {
    setIsRunning(false);
    setTimeLeft(duration * 60);
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const progress = Math.max(0, Math.min(100, ((duration * 60 - timeLeft) / (duration * 60)) * 100));

  return (
    <div className="max-w-4xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Focus Mode</h1>
        <p className="text-sm text-muted-foreground mt-1">Distraction-free Pomodoro sessions to power through your study goals.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 flex items-center gap-3">
            <Clock className="h-5 w-5 text-primary" />
            <div>
              <p className="text-xs text-muted-foreground">Total focus time</p>
              <p className="text-xl font-semibold">{stats.total_minutes} min</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 flex items-center gap-3">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <div>
              <p className="text-xs text-muted-foreground">Completed sessions</p>
              <p className="text-xl font-semibold">{stats.completed_sessions}</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 flex items-center gap-3">
            <Clock className="h-5 w-5 text-yellow-500" />
            <div>
              <p className="text-xs text-muted-foreground">This week</p>
              <p className="text-xl font-semibold">{stats.weekly_minutes} min</p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-2 border-primary/10">
        <CardHeader className="pb-2">
          <CardTitle className="text-center text-base">{isRunning ? task || "Focusing..." : "Ready to focus?"}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="text-center">
            <div className="text-7xl font-bold tabular-nums">{formatTime(timeLeft)}</div>
            <Progress value={progress} className="mt-4 max-w-md mx-auto" />
          </div>

          {!isRunning && !sessionId && (
            <div className="space-y-4 max-w-md mx-auto">
              <div className="space-y-2">
                <Label htmlFor="task">What are you working on?</Label>
                <Input
                  id="task"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="e.g. Practice Calculus integration"
                />
              </div>

              <div className="space-y-2">
                <Label>Duration</Label>
                <div className="flex gap-2">
                  {PRESETS.map((p) => (
                    <Button
                      key={p}
                      variant={duration === p ? "default" : "outline"}
                      size="sm"
                      onClick={() => {
                        setDuration(p);
                        setTimeLeft(p * 60);
                      }}
                    >
                      {p} min
                    </Button>
                  ))}
                </div>
              </div>

              <Button className="w-full" onClick={startSession}>
                <Play className="mr-2 h-4 w-4" /> Start focus session
              </Button>
            </div>
          )}

          {(isRunning || sessionId) && (
            <div className="flex items-center justify-center gap-3">
              <Button variant="outline" size="icon" onClick={togglePause}>
                {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button variant="outline" size="icon" onClick={reset}>
                <RotateCcw className="h-4 w-4" />
              </Button>
              <Button variant="destructive" onClick={() => finishSession(false)}>
                Stop
              </Button>
              <Button onClick={() => finishSession(true)}>
                <CheckCircle className="mr-2 h-4 w-4" /> Complete
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
