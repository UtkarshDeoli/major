"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Lightbulb, ChevronRight, RotateCcw } from "lucide-react";
import { socraticAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";

interface SocraticStep {
  type: "probe" | "hint" | "partial";
  content: string;
  expectation?: string;
}

export function SocraticExplainer({
  question,
  concept,
  docIds,
}: {
  question: string;
  concept?: string;
  docIds?: string[];
}) {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{ summary: string; steps: SocraticStep[]; final_prompt: string } | null>(null);
  const [visibleSteps, setVisibleSteps] = useState(1);

  const load = async () => {
    setIsLoading(true);
    setVisibleSteps(1);
    try {
      const data = await socraticAPI.explain({ question, concept, doc_ids: docIds });
      setResult(data);
    } catch (error) {
      toast({ title: "Socratic explanation failed", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setVisibleSteps(1);
  };

  if (!result) {
    return (
      <Button variant="outline" size="sm" onClick={load} disabled={isLoading}>
        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Lightbulb className="mr-2 h-4 w-4" />}
        Explain like a tutor
      </Button>
    );
  }

  return (
    <Card className="mt-4 border-yellow-500/30 bg-yellow-500/5">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Lightbulb className="h-4 w-4 text-yellow-600" /> Socratic explanation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm">{result.summary}</p>
        <div className="space-y-2">
          {result.steps.slice(0, visibleSteps).map((step, idx) => (
            <div key={idx} className="rounded-md border bg-background p-3 text-sm">
              <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Step {idx + 1} · {step.type}
              </span>
              <p className="mt-1">{step.content}</p>
              {step.expectation && (
                <p className="mt-1 text-xs text-muted-foreground">Goal: {step.expectation}</p>
              )}
            </div>
          ))}
        </div>

        {visibleSteps < result.steps.length ? (
          <Button variant="ghost" size="sm" onClick={() => setVisibleSteps((s) => s + 1)}>
            Next hint
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        ) : (
          <div className="rounded-md border border-primary/30 bg-primary/5 p-3 text-sm">
            <span className="font-medium">Final check: {" "}</span>
            {result.final_prompt}
          </div>
        )}

        <Button variant="ghost" size="sm" onClick={reset} className="text-muted-foreground">
          <RotateCcw className="mr-1 h-3 w-3" /> Reset
        </Button>
      </CardContent>
    </Card>
  );
}
