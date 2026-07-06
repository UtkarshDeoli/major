"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, AlertTriangle, TrendingDown, Clock, BookOpen, Target } from "lucide-react";
import { analyticsAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";

const FLAG_META: Record<string, { label: string; icon: any; color: string }> = {
  score_drop: { label: "Score drop", icon: TrendingDown, color: "text-red-500" },
  inactive: { label: "Inactive", icon: Clock, color: "text-yellow-500" },
  low_mastery: { label: "Low mastery", icon: Target, color: "text-orange-500" },
  low_average: { label: "Low average", icon: AlertTriangle, color: "text-red-500" },
  no_activity: { label: "No activity", icon: Clock, color: "text-muted-foreground" },
};

interface Alert {
  student_email: string;
  name?: string;
  flags: string[];
  last_active_at?: string;
  average_score: number;
}

export function TeacherAlertsPanel() {
  const { toast } = useToast();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [insights, setInsights] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const load = useCallback(async () => {
    setIsLoading(true);
    try {
      const [alertsData, insightsData] = await Promise.all([
        analyticsAPI.getTeacherAlerts(),
        analyticsAPI.getTeacherInsights(),
      ]);
      setAlerts(alertsData.alerts || []);
      setInsights(insightsData.insights || []);
    } catch (error) {
      toast({ title: "Could not load alerts", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
            At-risk students
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : alerts.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4">No at-risk students right now. Great work!</p>
          ) : (
            <div className="space-y-3">
              {alerts.slice(0, 6).map((alert) => (
                <div key={alert.student_email} className="rounded-md border p-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{alert.name || alert.student_email}</span>
                    <span className="text-xs text-muted-foreground">Avg {alert.average_score}%</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {alert.flags.map((flag) => {
                      const meta = FLAG_META[flag] || { label: flag, icon: AlertTriangle, color: "text-muted-foreground" };
                      const Icon = meta.icon;
                      return (
                        <Badge key={flag} variant="outline" className={`text-[10px] gap-1 ${meta.color}`}>
                          <Icon className="h-3 w-3" />
                          {meta.label}
                        </Badge>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-primary" />
            Recommended focus
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : insights.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4">No insights yet. Students need to take a few tests first.</p>
          ) : (
            <div className="space-y-3">
              {insights.slice(0, 6).map((insight) => (
                <div key={insight.student_email} className="rounded-md border p-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{insight.name || insight.student_email}</span>
                    <span className="text-xs text-muted-foreground">
                      {insight.weak_topics.length} weak topics
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{insight.recommended_action}</p>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {insight.recommended_focus.map((topic: string) => (
                      <Badge key={topic} variant="secondary" className="text-[10px]">
                        {topic}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
