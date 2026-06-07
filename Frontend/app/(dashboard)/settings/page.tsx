"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { User, Mail, Lock, Bell, Palette, Globe } from "lucide-react"

export default function SettingsPage() {
  const { toast } = useToast()
  const [emailNotifications, setEmailNotifications] = useState(true)
  const [browserNotifications, setBrowserNotifications] = useState(true)
  const [soundNotifications, setSoundNotifications] = useState(false)
  const [publicProfile, setPublicProfile] = useState(false)

  const handleSave = () => {
    toast({
      title: "Settings saved",
      description: "Your preferences have been updated.",
    })
  }

  return (
    <div className="p-6 lg:p-8 max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground mt-1">Manage your account preferences and application settings.</p>
      </div>

      <div className="space-y-6">
        {/* Profile */}
        <Card className="neo-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5 text-primary" />
              Profile
            </CardTitle>
            <CardDescription>Update your personal information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="fullName">Full Name</Label>
                <Input id="fullName" placeholder="John Doe" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input id="username" placeholder="johndoe" />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="bio">Bio</Label>
              <Input id="bio" placeholder="Tell us a little about yourself..." />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Public Profile</Label>
                <p className="text-muted-foreground text-xs">Make your profile visible to other students</p>
              </div>
              <Switch checked={publicProfile} onCheckedChange={setPublicProfile} />
            </div>
          </CardContent>
        </Card>

        {/* Account */}
        <Card className="neo-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-primary" />
              Account
            </CardTitle>
            <CardDescription>Manage your account credentials</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" placeholder="you@example.com" />
            </div>
            <Separator />
            <div className="space-y-2">
              <Label htmlFor="currentPassword">Current Password</Label>
              <Input id="currentPassword" type="password" />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="newPassword">New Password</Label>
                <Input id="newPassword" type="password" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm Password</Label>
                <Input id="confirmPassword" type="password" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card className="neo-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5 text-primary" />
              Notifications
            </CardTitle>
            <CardDescription>Choose how you want to be notified</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Email Notifications</Label>
                <p className="text-muted-foreground text-xs">Receive updates about your account via email</p>
              </div>
              <Switch checked={emailNotifications} onCheckedChange={setEmailNotifications} />
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Browser Notifications</Label>
                <p className="text-muted-foreground text-xs">Show desktop notifications for test results and reminders</p>
              </div>
              <Switch checked={browserNotifications} onCheckedChange={setBrowserNotifications} />
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Sound Notifications</Label>
                <p className="text-muted-foreground text-xs">Play a sound when AI responses or test results are ready</p>
              </div>
              <Switch checked={soundNotifications} onCheckedChange={setSoundNotifications} />
            </div>
          </CardContent>
        </Card>

        {/* Appearance */}
        <Card className="neo-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Palette className="h-5 w-5 text-primary" />
              Appearance
            </CardTitle>
            <CardDescription>Customize the look and feel</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Accent Color</Label>
              <div className="flex gap-3">
                {["#7C3AED", "#EC4899", "#3B82F6", "#10B981", "#F59E0B", "#6B7280"].map((color) => (
                  <div
                    key={color}
                    className="w-8 h-8 rounded-full cursor-pointer border-2 border-transparent hover:scale-110 transition-transform"
                    style={{ backgroundColor: color }}
                    onClick={() => {
                      toast({
                        title: "Accent color updated",
                        description: "This is a demo - color changes require theme integration.",
                      })
                    }}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button onClick={handleSave} className="gap-2">
            <Lock className="h-4 w-4" />
            Save Changes
          </Button>
        </div>
      </div>
    </div>
  )
}
