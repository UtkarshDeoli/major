"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { X, ArrowRight, Crown } from "lucide-react";
import { cn } from "@/lib/utils";

interface UpgradePayload {
  resource?: string;
  used?: number;
  limit?: number;
  plan?: string;
  upgradeUrl?: string;
  message?: string;
}

const RESOURCE_NAMES: Record<string, string> = {
  mock_test: "mock tests",
  flashcard: "flashcards",
  ai_material: "AI summaries",
  chat_message: "chat messages",
  doc_storage: "document storage",
  class_count: "classes",
};

export function UpgradeBanner() {
  const [payload, setPayload] = useState<UpgradePayload | null>(null);

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail as UpgradePayload;
      setPayload(detail);
    };
    window.addEventListener("orbit:upgrade-required", handler);
    return () => window.removeEventListener("orbit:upgrade-required", handler);
  }, []);

  if (!payload) return null;

  const resourceName = payload.resource ? RESOURCE_NAMES[payload.resource] || payload.resource : "this feature";
  const unlimited = !Number.isFinite(payload.limit || 0);
  const pct =
    payload.limit && payload.limit > 0 && !unlimited
      ? Math.min(100, Math.round(((payload.used || 0) / payload.limit) * 100))
      : 100;

  return (
    <div className="fixed bottom-0 inset-x-0 z-50 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="max-w-6xl mx-auto px-4 py-3">
        <div className="flex items-start gap-3">
          <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
            <Crown className="h-4 w-4 text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium">You&apos;ve reached your {resourceName} limit.</p>
            {payload.used !== undefined && payload.limit !== undefined && !unlimited && (
              <div className="mt-2 max-w-md">
                <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                  <span>{payload.used.toLocaleString()} / {payload.limit.toLocaleString()} used</span>
                  <span className="capitalize">Current plan: {payload.plan}</span>
                </div>
                <Progress value={pct} />
              </div>
            )}
          </div>
          <Link href={payload.upgradeUrl || "/billing"}>
            <Button size="sm" className="shrink-0">
              Upgrade
              <ArrowRight className="ml-1 h-4 w-4" />
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0 h-8 w-8"
            onClick={() => setPayload(null)}
            aria-label="Dismiss upgrade prompt"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
