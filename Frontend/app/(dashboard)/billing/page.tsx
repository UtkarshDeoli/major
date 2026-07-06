"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Receipt, AlertTriangle, CheckCircle2, Crown, ArrowRight, Download } from "lucide-react";
import { subscriptionAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
import { useAuth } from "@/lib/context/auth-context";

const RESOURCE_LABELS: Record<string, string> = {
  mock_test: "Mock tests",
  flashcard: "Flashcards",
  ai_material: "AI summaries",
  chat_message: "Chat messages",
  doc_storage: "Document storage",
  class_count: "Classes / batches",
};

const RESOURCE_UNITS: Record<string, string> = {
  mock_test: "this month",
  flashcard: "this month",
  ai_material: "this month",
  chat_message: "this month",
  doc_storage: "total",
  class_count: "total",
};

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "Unlimited";
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatLimit(value: number | string): string {
  if (value === Infinity || value === "Infinity" || value === Number.POSITIVE_INFINITY) return "Unlimited";
  const n = typeof value === "number" ? value : parseFloat(value);
  if (Number.isNaN(n)) return String(value);
  return n.toLocaleString();
}

function formatDate(value?: string): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return value;
  }
}

export default function BillingPage() {
  const router = useRouter();
  const { toast } = useToast();
  const { user, refreshUser } = useAuth();
  const [isLoading, setIsLoading] = useState(true);
  const [billing, setBilling] = useState<any | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  const loadBilling = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await subscriptionAPI.getMe();
      setBilling(data);
    } catch (error) {
      toast({
        title: "Could not load billing",
        description: getErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadBilling();
  }, [loadBilling]);

  const handleCancel = async () => {
    if (!confirm("Are you sure you want to cancel your subscription? You will keep access until the current billing period ends.")) return;
    setIsCancelling(true);
    try {
      await subscriptionAPI.cancel();
      toast({ title: "Subscription cancelled", description: "Your plan will remain active until the end of the billing period." });
      await refreshUser();
      await loadBilling();
    } catch (error) {
      toast({ title: "Cancellation failed", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsCancelling(false);
    }
  };

  const plan = billing?.plan || user?.subscription?.plan || "starter";
  const isPaid = plan !== "starter";
  const status = billing?.status || "free";

  const usageEntries = useMemo(() => {
    const usage = billing?.usage || {};
    const limits = billing?.limits || {};
    return Object.entries(RESOURCE_LABELS)
      .filter(([key]) => usage[key] !== undefined || limits[key] !== undefined)
      .map(([key, label]) => {
        const used = usage[key] || 0;
        const limitValue = limits[key] ?? Infinity;
        const unlimited = !Number.isFinite(limitValue);
        const pct = unlimited ? 0 : Math.min(100, Math.round((used / Number(limitValue)) * 100));
        return { key, label, used, limit: limitValue, unlimited, pct };
      });
  }, [billing]);

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Billing & Plans</h1>
          <p className="text-sm text-muted-foreground mt-1">Manage your subscription, usage, and invoices.</p>
        </div>
        {!isPaid && (
          <Link href="/pricing">
            <Button>
              Upgrade plan
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        )}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          <Card>
            <CardHeader className="pb-4">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <Crown className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1">
                  <CardTitle className="capitalize">{plan} Plan</CardTitle>
                  <CardDescription>
                    {status === "free" ? "You are on the free starter plan." : `Subscription status: ${status}`}
                  </CardDescription>
                </div>
                <Badge variant={isPaid ? "default" : "secondary"}>
                  {isPaid ? "Active" : "Free"}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="rounded-md border p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Billing cycle</p>
                  <p className="text-sm font-medium mt-1 capitalize">{billing?.cycle || "—"}</p>
                </div>
                <div className="rounded-md border p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Current period</p>
                  <p className="text-sm font-medium mt-1">
                    {formatDate(billing?.current_period_start)} → {formatDate(billing?.current_period_end)}
                  </p>
                </div>
                <div className="rounded-md border p-3">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Source</p>
                  <p className="text-sm font-medium mt-1 capitalize">{billing?.source || "—"}</p>
                </div>
              </div>

              {isPaid && status !== "free" && (
                <div className="flex items-center gap-2 rounded-md border border-yellow-500/30 bg-yellow-500/10 p-3 text-sm">
                  <AlertTriangle className="h-4 w-4 text-yellow-600 shrink-0" />
                  <span>Cancelling will downgrade you to the free Starter plan at the end of the current period.</span>
                  <Button
                    variant="outline"
                    size="sm"
                    className="ml-auto"
                    onClick={handleCancel}
                    disabled={isCancelling}
                  >
                    {isCancelling && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
                    Cancel subscription
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          <Tabs defaultValue="usage" className="w-full">
            <TabsList>
              <TabsTrigger value="usage">Usage</TabsTrigger>
              <TabsTrigger value="invoices">Invoices</TabsTrigger>
            </TabsList>

            <TabsContent value="usage" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Plan usage</CardTitle>
                  <CardDescription>Your current consumption against plan limits.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  {usageEntries.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No usage data available yet.</p>
                  ) : (
                    usageEntries.map((entry) => (
                      <div key={entry.key} className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium">{entry.label}</span>
                          <span className="text-muted-foreground">
                            {entry.key === "doc_storage" ? (
                              <>
                                {formatBytes(entry.used)} / {formatBytes(Number(entry.limit))} {" "}
                                <span className="text-xs">({RESOURCE_UNITS[entry.key]})</span>
                              </>
                            ) : (
                              <>
                                {entry.used.toLocaleString()} / {entry.unlimited ? "Unlimited" : formatLimit(entry.limit)} {" "}
                                <span className="text-xs">({RESOURCE_UNITS[entry.key]})</span>
                              </>
                            )}
                          </span>
                        </div>
                        <Progress value={entry.pct} />
                      </div>
                    ))
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="invoices" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Invoices</CardTitle>
                  <CardDescription>Payment history for your Orbit subscription.</CardDescription>
                </CardHeader>
                <CardContent>
                  {billing?.invoices?.length ? (
                    <div className="rounded-md border">
                      <div className="grid grid-cols-[1fr_80px_100px_100px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        <span>Date</span>
                        <span>Plan</span>
                        <span>Amount</span>
                        <span>Status</span>
                      </div>
                      {billing.invoices.map((invoice: any) => (
                        <div
                          key={invoice.id || invoice.order_id}
                          className="grid grid-cols-[1fr_80px_100px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center text-sm"
                        >
                          <span>{formatDate(invoice.created_at || invoice.date)}</span>
                          <span className="capitalize">{invoice.plan || plan}</span>
                          <span>₹{((invoice.amount || 0) / 100).toLocaleString()}</span>
                          <span className="flex items-center gap-1.5">
                            {invoice.status === "captured" || invoice.status === "paid" ? (
                              <>
                                <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                                Paid
                              </>
                            ) : (
                              invoice.status
                            )}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                      <Receipt className="h-10 w-10 mb-3 opacity-50" />
                      <p className="text-sm">No invoices yet.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
