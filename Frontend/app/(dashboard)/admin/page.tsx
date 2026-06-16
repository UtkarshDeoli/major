"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Users, Shield, Key, Activity } from "lucide-react"
import RoleGuard from "@/components/auth/route-protection/role-guard"

function AdminDashboardContent() {
  const [isLoading, setIsLoading] = useState(true)
  const [users, setUsers] = useState<any[]>([])

  useEffect(() => {
    setIsLoading(false)
    setUsers([])
  }, [])

  const statCards = [
    { label: "Total Users", value: users.length, icon: Users },
    { label: "Active", value: users.filter((u: any) => u.status === "active").length, icon: Activity },
    { label: "Admins", value: users.filter((u: any) => u.role === "admin").length, icon: Shield },
    { label: "Licenses", value: "—", icon: Key },
  ]

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin Dashboard.</h1>
        <p className="text-sm text-muted-foreground mt-1">Manage users, roles, and licenses.</p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {statCards.map((stat) => (
          <div key={stat.label} className="rounded-md border bg-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">{stat.label}</span>
              <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <div className="text-xl font-semibold tabular-nums">
              {isLoading ? <div className="h-6 w-12 bg-muted animate-pulse rounded" /> : stat.value}
            </div>
          </div>
        ))}
      </div>
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Users</h2>
        <div className="rounded-md border">
          <div className="grid grid-cols-[1fr_80px_80px_100px] gap-4 px-4 py-2 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider">
            <span>Email</span>
            <span>Role</span>
            <span>Status</span>
            <span>Actions</span>
          </div>
          {users.length === 0 ? (
            <div className="px-4 py-12 text-center text-sm text-muted-foreground">
              User management requires backend admin endpoints. This page is a frontend scaffold.
            </div>
          ) : (
            users.map((u: any) => (
              <div key={u.id} className="grid grid-cols-[1fr_80px_80px_100px] gap-4 px-4 py-3 border-b last:border-b-0 items-center">
                <div className="min-w-0">
                  <p className="text-sm truncate">{u.name || u.email}</p>
                  <p className="text-xs text-muted-foreground font-mono truncate">{u.email}</p>
                </div>
                <Badge variant="secondary" className="text-[10px] font-normal w-fit">{u.role}</Badge>
                <Badge variant="outline" className="text-[10px] font-normal w-fit">{u.status}</Badge>
                <Select>
                  <SelectTrigger className="h-7 text-[11px] rounded-md">
                    <SelectValue placeholder="Role" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="student">Student</SelectItem>
                    <SelectItem value="teacher">Teacher</SelectItem>
                    <SelectItem value="subadmin">Sub-admin</SelectItem>
                    <SelectItem value="admin">Admin</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default function AdminPage() {
  return (
    <RoleGuard allowedRoles={["admin"]}>
      <AdminDashboardContent />
    </RoleGuard>
  )
}