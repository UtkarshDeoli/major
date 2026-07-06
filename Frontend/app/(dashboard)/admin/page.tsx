"use client"

import { useCallback, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Users, Shield, Key, Activity, Loader2, RefreshCw, Crown, Building2 } from "lucide-react"
import RoleGuard from "@/components/auth/route-protection/role-guard"
import { adminAPI } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { getErrorMessage } from "@/lib/errors"

const ROLES = ["student", "teacher", "subadmin", "admin"]

export default function AdminPage() {
  return (
    <RoleGuard allowedRoles={["admin", "subadmin"]} fallback="/dashboard">
      <AdminDashboardContent />
    </RoleGuard>
  )
}

function AdminDashboardContent() {
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState("overview")
  const [isLoading, setIsLoading] = useState(true)

  // Overview
  const [analytics, setAnalytics] = useState<any | null>(null)

  // Users
  const [users, setUsers] = useState<any[]>([])
  const [userSearch, setUserSearch] = useState("")
  const [roleFilter, setRoleFilter] = useState<string | "all">("all")

  // Orgs
  const [orgs, setOrgs] = useState<any[]>([])

  // Subscriptions / payments
  const [subs, setSubs] = useState<any[]>([])
  const [payments, setPayments] = useState<any[]>([])

  // Manual activation
  const [activateEmail, setActivateEmail] = useState("")
  const [activatePlan, setActivatePlan] = useState<"pro" | "premium">("pro")
  const [activateDays, setActivateDays] = useState(30)
  const [isActivating, setIsActivating] = useState(false)

  const loadAll = useCallback(async () => {
    setIsLoading(true)
    try {
      const [analyticsData, usersData, orgsData, subsData] = await Promise.all([
        adminAPI.getAnalytics().catch(() => null),
        adminAPI.listUsers({ limit: 200 }),
        adminAPI.listOrgs(),
        adminAPI.listSubscriptions(),
      ])
      setAnalytics(analyticsData)
      setUsers(usersData.users || [])
      setOrgs(orgsData.orgs || [])
      setSubs(subsData.subscriptions || [])
      setPayments(subsData.payments || [])
    } catch (error) {
      toast({ title: "Could not load admin data", description: getErrorMessage(error), variant: "destructive" })
    } finally {
      setIsLoading(false)
    }
  }, [toast])

  useEffect(() => {
    loadAll()
  }, [loadAll])

  const filteredUsers = users.filter((u) => {
    const matchesSearch =
      (u.email || "").toLowerCase().includes(userSearch.toLowerCase()) ||
      (u.name || "").toLowerCase().includes(userSearch.toLowerCase())
    const matchesRole = roleFilter === "all" || u.role === roleFilter
    return matchesSearch && matchesRole
  })

  const handleRoleChange = async (email: string, role: string) => {
    try {
      await adminAPI.updateRole(email, role)
      toast({ title: "Role updated" })
      setUsers((prev) => prev.map((u) => (u.email === email ? { ...u, role } : u)))
    } catch (error) {
      toast({ title: "Update failed", description: getErrorMessage(error), variant: "destructive" })
    }
  }

  const handleStatusChange = async (email: string, active: boolean) => {
    try {
      await adminAPI.updateStatus(email, active)
      toast({ title: `User ${active ? "enabled" : "disabled"}` })
      setUsers((prev) => prev.map((u) => (u.email === email ? { ...u, active } : u)))
    } catch (error) {
      toast({ title: "Update failed", description: getErrorMessage(error), variant: "destructive" })
    }
  }

  const handleActivate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!activateEmail) return
    setIsActivating(true)
    try {
      await adminAPI.activateSubscription(activateEmail, activatePlan, activateDays)
      toast({ title: "Subscription activated" })
      setActivateEmail("")
      await loadAll()
    } catch (error) {
      toast({ title: "Activation failed", description: getErrorMessage(error), variant: "destructive" })
    } finally {
      setIsActivating(false)
    }
  }

  const totalUsers = analytics?.totals?.users_by_role || {}
  const activeSubs = analytics?.totals?.active_subscriptions || 0
  const orgCount = analytics?.totals?.org_count || 0
  const mrrPaise = analytics?.totals?.mrr_paise || 0

  const statCards = [
    { label: "Total Users", value: Object.values(totalUsers as Record<string, number>).reduce((a, b) => a + b, 0), icon: Users },
    { label: "Active Subs", value: activeSubs, icon: Activity },
    { label: "Admins", value: totalUsers.admin || 0, icon: Shield },
    { label: "Organizations", value: orgCount, icon: Building2 },
  ]

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Admin Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1">Manage users, roles, organizations, and subscriptions.</p>
        </div>
        <Button variant="outline" size="sm" onClick={loadAll} disabled={isLoading}>
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          <span className="ml-2">Refresh</span>
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="users">Users</TabsTrigger>
          <TabsTrigger value="orgs">Organizations</TabsTrigger>
          <TabsTrigger value="subscriptions">Subscriptions</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {statCards.map((stat) => (
              <Card key={stat.label}>
                <CardContent className="p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
                    <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
                  </div>
                  <div className="text-xl font-semibold tabular-nums">{stat.value}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Revenue</CardTitle>
              <CardDescription>Estimated monthly recurring revenue from captured payments.</CardDescription>
            </CardHeader>
            <CardContent className="text-3xl font-semibold">
              ₹{(mrrPaise / 100).toLocaleString()}
              <span className="text-sm font-normal text-muted-foreground ml-2">MRR</span>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Manual subscription activation</CardTitle>
              <CardDescription>Grant a paid plan to a user by email. Use for support / sales.</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleActivate} className="flex flex-col sm:flex-row gap-3">
                <Input
                  placeholder="user@example.com"
                  value={activateEmail}
                  onChange={(e) => setActivateEmail(e.target.value)}
                  required
                />
                <Select value={activatePlan} onValueChange={(v) => setActivatePlan(v as "pro" | "premium")}>
                  <SelectTrigger className="w-[120px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pro">Pro</SelectItem>
                    <SelectItem value="premium">Premium</SelectItem>
                  </SelectContent>
                </Select>
                <Input
                  type="number"
                  min={1}
                  max={365}
                  value={activateDays}
                  onChange={(e) => setActivateDays(Math.max(1, parseInt(e.target.value || "1", 10)))}
                  className="w-[100px]"
                />
                <Button type="submit" disabled={isActivating}>
                  {isActivating && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Activate
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="users" className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              placeholder="Search by email or name..."
              value={userSearch}
              onChange={(e) => setUserSearch(e.target.value)}
              className="sm:max-w-sm"
            />
            <Select value={roleFilter} onValueChange={(v) => setRoleFilter(v)}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Filter role" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All roles</SelectItem>
                {ROLES.map((r) => (
                  <SelectItem key={r} value={r} className="capitalize">{r}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Card>
            <CardContent className="p-0">
              <div className="rounded-md border">
                <div className="grid grid-cols-[1fr_110px_100px_100px_80px_80px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  <span>Email / Name</span>
                  <span>Role</span>
                  <span>Status</span>
                  <span>Org</span>
                  <span className="text-center">Active</span>
                  <span />
                </div>
                {filteredUsers.length === 0 ? (
                  <div className="px-4 py-12 text-center text-sm text-muted-foreground">No users found.</div>
                ) : (
                  filteredUsers.map((u: any) => (
                    <div
                      key={u.email}
                      className="grid grid-cols-[1fr_110px_100px_100px_80px_80px] gap-4 px-4 py-3 border-b last:border-b-0 items-center text-sm"
                    >
                      <div className="min-w-0">
                        <p className="truncate">{u.name || u.email}</p>
                        <p className="text-xs text-muted-foreground font-mono truncate">{u.email}</p>
                      </div>
                      <Select value={u.role} onValueChange={(v) => handleRoleChange(u.email, v)}>
                        <SelectTrigger className="h-7 text-[11px] rounded-md">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {ROLES.map((r) => (
                            <SelectItem key={r} value={r} className="capitalize">{r}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Badge variant="outline" className="w-fit text-[10px]">{u.active === false ? "Disabled" : "Active"}</Badge>
                      <span className="text-xs text-muted-foreground truncate">{u.org_id ? "Yes" : "—"}</span>
                      <div className="flex justify-center">
                        <input
                          type="checkbox"
                          checked={u.active !== false}
                          onChange={(e) => handleStatusChange(u.email, e.target.checked)}
                          className="h-4 w-4 rounded border-gray-300"
                        />
                      </div>
                      <span className="text-right text-xs text-muted-foreground">{u.org_id || "—"}</span>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="orgs" className="space-y-4">
          <Card>
            <CardContent className="p-0">
              <div className="rounded-md border">
                <div className="grid grid-cols-[1fr_80px_120px_120px_100px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  <span>Name</span>
                  <span>Tier</span>
                  <span>Seats</span>
                  <span>Status</span>
                  <span />
                </div>
                {orgs.length === 0 ? (
                  <div className="px-4 py-12 text-center text-sm text-muted-foreground">No organizations found.</div>
                ) : (
                  orgs.map((o: any) => (
                    <div
                      key={o.org_id}
                      className="grid grid-cols-[1fr_80px_120px_120px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center text-sm"
                    >
                      <div className="min-w-0">
                        <p className="truncate font-medium">{o.name}</p>
                        <p className="text-xs text-muted-foreground font-mono truncate">{o.org_id}</p>
                      </div>
                      <Badge variant="secondary" className="w-fit text-[10px] capitalize">{o.tier}</Badge>
                      <span className="text-xs">{o.seats_used} / {o.seats_total}</span>
                      <Badge variant="outline" className="w-fit text-[10px] capitalize">{o.status}</Badge>
                      <Button variant="ghost" size="sm" className="h-7 text-[11px]">
                        Manage
                      </Button>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="subscriptions" className="space-y-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Subscriptions</CardTitle>
              <CardDescription>Active and historical self-serve subscriptions.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="rounded-md border">
                <div className="grid grid-cols-[1fr_80px_100px_120px_120px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  <span>User</span>
                  <span>Plan</span>
                  <span>Status</span>
                  <span>Start</span>
                  <span>End</span>
                </div>
                {subs.length === 0 ? (
                  <div className="px-4 py-12 text-center text-sm text-muted-foreground">No subscriptions found.</div>
                ) : (
                  subs.map((s: any) => (
                    <div
                      key={s._id || s.user_id}
                      className="grid grid-cols-[1fr_80px_100px_120px_120px] gap-4 px-4 py-3 border-b last:border-b-0 items-center text-sm"
                    >
                      <span className="truncate">{s.user_id}</span>
                      <span className="capitalize">{s.plan}</span>
                      <Badge variant="outline" className="w-fit text-[10px] capitalize">{s.status}</Badge>
                      <span className="text-xs text-muted-foreground">{formatDate(s.current_period_start)}</span>
                      <span className="text-xs text-muted-foreground">{formatDate(s.current_period_end)}</span>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Payments</CardTitle>
              <CardDescription>Captured Razorpay orders.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="rounded-md border">
                <div className="grid grid-cols-[1fr_80px_100px_100px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  <span>Date</span>
                  <span>Plan</span>
                  <span>Amount</span>
                  <span>Status</span>
                </div>
                {payments.length === 0 ? (
                  <div className="px-4 py-12 text-center text-sm text-muted-foreground">No payments found.</div>
                ) : (
                  payments.map((p: any) => (
                    <div
                      key={p._id || p.order_id}
                      className="grid grid-cols-[1fr_80px_100px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center text-sm"
                    >
                      <span className="text-xs text-muted-foreground">{formatDate(p.created_at)}</span>
                      <span className="capitalize">{p.plan}</span>
                      <span>₹{((p.amount || 0) / 100).toLocaleString()}</span>
                      <Badge variant="outline" className="w-fit text-[10px] capitalize">{p.status}</Badge>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function formatDate(value?: string): string {
  if (!value) return "—"
  try {
    return new Date(value).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
  } catch {
    return value
  }
}
