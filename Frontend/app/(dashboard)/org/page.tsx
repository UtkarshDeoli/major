"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Loader2, Users, Copy, Trash2, UserPlus, Crown, Building2, Mail } from "lucide-react";
import { orgAPI } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
import RoleGuard from "@/components/auth/route-protection/role-guard";

interface Member {
  email: string;
  name?: string;
  role?: string;
  member_role?: string;
  joined_at?: string;
}

function OrgPageContent() {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [org, setOrg] = useState<any | null>(null);
  const [members, setMembers] = useState<Member[]>([]);
  const [seats, setSeats] = useState({ used: 0, total: 0 });

  // Invite form
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"teacher" | "student">("teacher");
  const [isInviting, setIsInviting] = useState(false);

  // Add seats form
  const [seatsToAdd, setSeatsToAdd] = useState(1);
  const [isAddingSeats, setIsAddingSeats] = useState(false);

  const loadOrg = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await orgAPI.getMe();
      setOrg(data.org);
      setMembers(data.members || []);
      setSeats(data.seats || { used: 0, total: 0 });
    } catch (error) {
      toast({ title: "Could not load organization", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadOrg();
  }, [loadOrg]);

  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsInviting(true);
    try {
      const res = await orgAPI.invite({ member_role: inviteRole, email: inviteEmail || undefined });
      toast({
        title: "Invite created",
        description: (
          <div className="space-y-2">
            <p>Share this code with your {inviteRole}:</p>
            <div className="flex items-center gap-2 rounded-md border p-2 bg-muted">
              <code className="text-xs font-mono flex-1 truncate">{res.code}</code>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-7 w-7 shrink-0"
                onClick={() => {
                  navigator.clipboard.writeText(res.code);
                  toast({ title: "Copied invite code" });
                }}
              >
                <Copy className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        ),
      });
      setInviteEmail("");
      await loadOrg();
    } catch (error) {
      toast({ title: "Invite failed", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsInviting(false);
    }
  };

  const handleRemove = async (email: string) => {
    if (!confirm(`Remove ${email} from your organization?`)) return;
    try {
      await orgAPI.removeMember(email);
      toast({ title: "Member removed" });
      await loadOrg();
    } catch (error) {
      toast({ title: "Could not remove member", description: getErrorMessage(error), variant: "destructive" });
    }
  };

  const handleAddSeats = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsAddingSeats(true);
    try {
      await orgAPI.addSeats(seatsToAdd);
      toast({ title: "Seats added", description: `${seatsToAdd} seat(s) added to your organization.` });
      setSeatsToAdd(1);
      await loadOrg();
    } catch (error) {
      toast({ title: "Could not add seats", description: getErrorMessage(error), variant: "destructive" });
    } finally {
      setIsAddingSeats(false);
    }
  };

  const seatPct = Math.min(100, Math.round((seats.used / Math.max(1, seats.total)) * 100));

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Building2 className="h-6 w-6" />
          Organization
        </h1>
        <p className="text-sm text-muted-foreground mt-1">Manage seats, invites, and members.</p>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : !org ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">No organization found. Create one first from onboarding.</p>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="md:col-span-2">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                      <Crown className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-base">{org.brand_name || org.name}</CardTitle>
                      <CardDescription className="capitalize">{org.tier} tier · {org.status}</CardDescription>
                    </div>
                  </div>
                  <Badge variant="secondary">{org.billing_cycle || "monthly"}</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">Seat usage</span>
                    <span className="text-muted-foreground">{seats.used} / {seats.total} used</span>
                  </div>
                  <Progress value={seatPct} />
                </div>

                <form onSubmit={handleAddSeats} className="flex items-end gap-3">
                  <div className="flex-1 space-y-2">
                    <Label htmlFor="add-seats">Add seats</Label>
                    <Input
                      id="add-seats"
                      type="number"
                      min={1}
                      max={500}
                      value={seatsToAdd}
                      onChange={(e) => setSeatsToAdd(Math.max(1, parseInt(e.target.value || "1", 10)))}
                    />
                  </div>
                  <Button type="submit" disabled={isAddingSeats}>
                    {isAddingSeats && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Add
                  </Button>
                </form>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <UserPlus className="h-4 w-4" /> Invite members
                </CardTitle>
                <CardDescription>Create an invite code for teachers or students.</CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleInvite} className="space-y-3">
                  <div className="space-y-2">
                    <Label htmlFor="invite-email">Email (optional)</Label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        id="invite-email"
                        type="email"
                        value={inviteEmail}
                        onChange={(e) => setInviteEmail(e.target.value)}
                        placeholder="member@example.com"
                        className="pl-10"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="invite-role">Role</Label>
                    <Select value={inviteRole} onValueChange={(v) => setInviteRole(v as "teacher" | "student")}>
                      <SelectTrigger id="invite-role">
                        <SelectValue placeholder="Select role" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="teacher">Teacher</SelectItem>
                        <SelectItem value="student">Student</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button type="submit" className="w-full" disabled={isInviting}>
                    {isInviting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Generate invite code
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Users className="h-4 w-4" /> Members
              </CardTitle>
              <CardDescription>Teachers and students enrolled under your organization.</CardDescription>
            </CardHeader>
            <CardContent>
              {members.length === 0 ? (
                <p className="text-sm text-muted-foreground py-8 text-center">No members yet. Generate an invite code to add some.</p>
              ) : (
                <div className="rounded-md border">
                  <div className="grid grid-cols-[1fr_100px_120px_60px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    <span>Email</span>
                    <span>Role</span>
                    <span>Joined</span>
                    <span />
                  </div>
                  {members.map((member) => (
                    <div
                      key={member.email}
                      className="grid grid-cols-[1fr_100px_120px_60px] gap-4 px-4 py-3 border-b last:border-b-0 items-center text-sm"
                    >
                      <div className="min-w-0">
                        <p className="truncate font-medium">{member.name || member.email}</p>
                        <p className="text-xs text-muted-foreground font-mono truncate">{member.email}</p>
                      </div>
                      <Badge variant="outline" className="w-fit text-[10px] capitalize">
                        {member.member_role || member.role || "member"}
                      </Badge>
                      <span className="text-muted-foreground text-xs">
                        {member.joined_at ? new Date(member.joined_at).toLocaleDateString() : "—"}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-muted-foreground hover:text-destructive"
                        onClick={() => handleRemove(member.email)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

export default function OrgPage() {
  return (
    <RoleGuard allowedRoles={["subadmin"]} fallback="/dashboard">
      <OrgPageContent />
    </RoleGuard>
  );
}
