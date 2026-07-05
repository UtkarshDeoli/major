"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/context/auth-context"
import { User, Mail, Bell, Palette, Save } from "lucide-react"

export default function SettingsPage() {
  const { toast } = useToast()
  const { user } = useAuth()

  const [fullName, setFullName] = useState(user?.name || "")
  const [email, setEmail] = useState(user?.email || "")
  const [emailNotifications, setEmailNotifications] = useState(true)
  const [browserNotifications, setBrowserNotifications] = useState(true)
  const [soundNotifications, setSoundNotifications] = useState(false)
  const [publicProfile, setPublicProfile] = useState(false)

  // Sync profile fields if user loads after mount
  useEffect(() => {
    if (user) {
      setFullName((prev) => prev || user.name || "")
      setEmail((prev) => prev || user.email || "")
    }
  }, [user])

  const [accentColor, setAccentColor] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("orbit:accent-color") || "#7C3AED"
    }
    return "#7C3AED"
  })

  const handleAccentColor = (color: string) => {
    document.documentElement.style.setProperty("--primary", color)
    localStorage.setItem("orbit:accent-color", color)
    setAccentColor(color)
    toast({
      title: "Accent color updated",
      description: "The new accent color has been applied.",
    })
  }

  // Apply saved accent color on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("orbit:accent-color")
      if (saved) {
        document.documentElement.style.setProperty("--primary", saved)
      }
    }
  }, [])

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = localStorage.getItem("orbit:preferences");
    if (!stored) return;
    try {
      const prefs = JSON.parse(stored) as {
        emailNotifications: boolean;
        browserNotifications: boolean;
        soundNotifications: boolean;
        publicProfile: boolean;
        fullName?: string;
      };
      setEmailNotifications(prefs.emailNotifications);
      setBrowserNotifications(prefs.browserNotifications);
      setSoundNotifications(prefs.soundNotifications);
      setPublicProfile(prefs.publicProfile);
      if (prefs.fullName) setFullName(prefs.fullName);
    } catch {
      // ignore malformed prefs
    }
  }, []);

  const handleSave = () => {
    // Phase 0 interim: persist the preferences we can store client-side.
    // Full profile/account sync (name, email, password) arrives in Phase 3.
    localStorage.setItem(
      "orbit:preferences",
      JSON.stringify({
        emailNotifications,
        browserNotifications,
        soundNotifications,
        publicProfile,
        fullName: fullName || user?.name || "",
      })
    );
    toast({
      title: "Preferences saved",
      description:
        "Notification and profile preferences saved to this browser. Account sync coming soon.",
    });
  };

  return (
    <div className="p-6 lg:p-8 max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">Manage your account preferences and application settings.</p>
      </div>

      <div className="space-y-6">
        {/* Profile */}
        <Card className="rounded-md border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <User className="h-4 w-4 text-muted-foreground" />
              Profile
            </CardTitle>
            <CardDescription>Update your personal information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="fullName" className="text-xs">Full Name</Label>
                <Input
                  id="fullName"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="John Doe"
                  className="rounded-md h-9 text-[13px]"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="username" className="text-xs">Username</Label>
                <Input id="username" placeholder="johndoe" className="rounded-md h-9 text-[13px]" />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="bio" className="text-xs">Bio</Label>
              <Input id="bio" placeholder="Tell us a little about yourself..." className="rounded-md h-9 text-[13px]" />
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
        <Card className="rounded-md border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Mail className="h-4 w-4 text-muted-foreground" />
              Account
            </CardTitle>
            <CardDescription>Manage your account credentials</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email" className="text-xs">Email</Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="rounded-md h-9 text-[13px]"
              />
            </div>
            <Separator />
            <div className="space-y-2">
              <Label htmlFor="currentPassword" className="text-xs">Current Password</Label>
              <Input id="currentPassword" type="password" className="rounded-md h-9 text-[13px]" />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="newPassword" className="text-xs">New Password</Label>
                <Input id="newPassword" type="password" className="rounded-md h-9 text-[13px]" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-xs">Confirm Password</Label>
                <Input id="confirmPassword" type="password" className="rounded-md h-9 text-[13px]" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card className="rounded-md border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Bell className="h-4 w-4 text-muted-foreground" />
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
        <Card className="rounded-md border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Palette className="h-4 w-4 text-muted-foreground" />
              Appearance
            </CardTitle>
            <CardDescription>Customize the look and feel</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Accent Color</Label>
              <div className="flex gap-3">
                {["#7C3AED", "#EC4899", "#3B82F6", "#10B981", "#F59E0B", "#6B7280"].map((color) => (
                  <button
                    key={color}
                    className="w-8 h-8 rounded-full cursor-pointer border-2 transition-transform hover:scale-110"
                    style={{
                      backgroundColor: color,
                      borderColor: accentColor === color ? "currentColor" : "transparent",
                    }}
                    onClick={() => handleAccentColor(color)}
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button onClick={handleSave} className="rounded-md gap-2">
            <Save className="h-4 w-4" />
            Save Changes
          </Button>
        </div>
      </div>
    </div>
  )
}