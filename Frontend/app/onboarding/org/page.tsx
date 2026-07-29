"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, Building2 } from "lucide-react";
import { orgAPI } from "@/lib/api";
import { useAuth } from "@/lib/context/auth-context";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";

export default function OrgOnboardingPage() {
  const router = useRouter();
  const { user, isLoading, isAuthenticated, refreshUser } = useAuth();
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [name, setName] = useState("");
  const [brandName, setBrandName] = useState("");
  const [tier, setTier] = useState<"pro" | "premium">("pro");
  const [seats, setSeats] = useState(5);
  const [billingCycle, setBillingCycle] = useState<"monthly" | "yearly">("monthly");
  const [tagline, setTagline] = useState("");
  const [logoFile, setLogoFile] = useState<File | null>(null);

  const logoPreviewUrl = useMemo(
    () => (logoFile ? URL.createObjectURL(logoFile) : null),
    [logoFile],
  );

  useEffect(() => {
    return () => {
      if (logoPreviewUrl) URL.revokeObjectURL(logoPreviewUrl);
    };
  }, [logoPreviewUrl]);

  useEffect(() => {
    if (isLoading) return;
    if (!isAuthenticated) {
      router.replace("/login");
      return;
    }
    if (user?.role !== "subadmin") {
      router.replace(getPostAuthRedirect(user));
      return;
    }
    if (user?.org_id) {
      router.replace("/admin");
    }
  }, [isLoading, isAuthenticated, user, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await orgAPI.create({
        name,
        brand_name: brandName || undefined,
        tagline: tagline || undefined,
        tier,
        seats_total: seats,
        billing_cycle: billingCycle,
      });
      if (logoFile) {
        try {
          await orgAPI.uploadLogo(logoFile);
        } catch (e) {
          // non-fatal: org is created; logo can be added later from /org
        }
      }
      toast({ title: "Organization created", description: "You can now manage seats and invites from /org." });
      await refreshUser();
      router.push("/admin");
    } catch (error) {
      toast({ title: "Could not create organization", description: getErrorMessage(error), variant: "destructive" });
      setIsSubmitting(false);
    }
  };

  if (isLoading || !isAuthenticated || !user || user.role !== "subadmin") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
              <Building2 className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle>Create your organization</CardTitle>
              <CardDescription>Set up your coaching center, school, or tuition class on Orbit.</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Organization name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Aakash Institute"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="brand">Brand name (optional)</Label>
              <Input
                id="brand"
                value={brandName}
                onChange={(e) => setBrandName(e.target.value)}
                placeholder="Display name for invites"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="tier">Plan tier</Label>
                <Select value={tier} onValueChange={(v) => setTier(v as "pro" | "premium")}>
                  <SelectTrigger id="tier">
                    <SelectValue placeholder="Select tier" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pro">Pro</SelectItem>
                    <SelectItem value="premium">Premium</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="seats">Seat licenses</Label>
                <Input
                  id="seats"
                  type="number"
                  min={1}
                  max={500}
                  value={seats}
                  onChange={(e) => setSeats(Math.max(1, parseInt(e.target.value || "1", 10)))}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="cycle">Billing cycle</Label>
              <Select value={billingCycle} onValueChange={(v) => setBillingCycle(v as "monthly" | "yearly")}>
                <SelectTrigger id="cycle">
                  <SelectValue placeholder="Select billing cycle" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="monthly">Monthly</SelectItem>
                  <SelectItem value="yearly">Yearly (save ~17%)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <label htmlFor="tagline" className="text-sm font-medium">Tagline (optional)</label>
              <input id="tagline" value={tagline} onChange={(e) => setTagline(e.target.value)}
                className="border rounded-md px-3 py-2" placeholder="e.g. Dream. Prepare. Achieve." />
            </div>
            <div className="grid gap-2">
              <label htmlFor="logo" className="text-sm font-medium">Coaching logo (optional)</label>
              <input id="logo" type="file" accept="image/*"
                onChange={(e) => setLogoFile(e.target.files?.[0] || null)}
                className="border rounded-md px-3 py-2" />
              {logoPreviewUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={logoPreviewUrl} alt="logo preview" className="h-12 w-12 object-contain" />
              )}
            </div>

            <Button type="submit" className="w-full" disabled={isSubmitting}>
              {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Create organization
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

function getPostAuthRedirect(user: any): string {
  if (!user) return "/login";
  if (user.role === "student") return user.onboarding_completed ? "/dashboard" : "/onboarding";
  if (user.role === "teacher") return "/teacher";
  return "/admin";
}
