"use client"

import { useState } from 'react'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetClose,
} from '@/components/ui/sheet'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useToast } from '@/hooks/use-toast'

interface SettingsPanelProps {
  onClose: () => void
  isMobile?: boolean
}

export function SettingsPanel({ onClose, isMobile = false }: SettingsPanelProps) {
  const { toast } = useToast()
  const [fontSizeValue, setFontSizeValue] = useState(100)
  
  const handleSaveSettings = () => {
    toast({
      title: "Settings saved",
      description: "Your preferences have been updated."
    })
    
    onClose()
  }
  
  return (
    <Sheet open={true} onOpenChange={onClose}>
      <SheetContent className={`${isMobile ? "w-full max-w-full" : "sm:max-w-md"} overflow-auto`}>
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription>
            Customize your experience and preferences
          </SheetDescription>
        </SheetHeader>
        
        <Tabs defaultValue="appearance" className="mt-6">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="appearance" className="text-xs sm:text-sm">Appearance</TabsTrigger>
            <TabsTrigger value="notifications" className="text-xs sm:text-sm">Notifications</TabsTrigger>
            <TabsTrigger value="accessibility" className="text-xs sm:text-sm">Accessibility</TabsTrigger>
          </TabsList>
          
          <TabsContent value="appearance" className="space-y-6 py-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Theme</h4>
              <RadioGroup defaultValue="system">
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="light" id="theme-light" />
                  <Label htmlFor="theme-light">Light</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="dark" id="theme-dark" />
                  <Label htmlFor="theme-dark">Dark</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="system" id="theme-system" />
                  <Label htmlFor="theme-system">System</Label>
                </div>
              </RadioGroup>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <Label htmlFor="accent-color">Accent Color</Label>
                <div className="w-6 h-6 rounded-full bg-primary border"></div>
              </div>
              <div className="grid grid-cols-6 gap-2">
                {["#7C3AED", "#EC4899", "#3B82F6", "#10B981", "#F59E0B", "#6B7280"].map(color => (
                  <div
                    key={color}
                    className="w-full aspect-square rounded-full cursor-pointer border"
                    style={{ backgroundColor: color }}
                    onClick={() => {
                      // In a real app, this would update the theme
                    }}
                  />
                ))}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="font-size">Font Size ({fontSizeValue}%)</Label>
              </div>
              <Slider
                id="font-size"
                min={75}
                max={150}
                step={5}
                value={[fontSizeValue]}
                onValueChange={(values) => setFontSizeValue(values[0])}
              />
            </div>
          </TabsContent>
          
          <TabsContent value="notifications" className="space-y-6 py-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="email-notifications">Email Notifications</Label>
                  <p className="text-muted-foreground text-xs">
                    Receive notifications about your account via email
                  </p>
                </div>
                <Switch id="email-notifications" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="browser-notifications">Browser Notifications</Label>
                  <p className="text-muted-foreground text-xs">
                    Show desktop notifications when AI responses arrive
                  </p>
                </div>
                <Switch id="browser-notifications" defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="sound-notifications">Sound Notifications</Label>
                  <p className="text-muted-foreground text-xs">
                    Play a sound when new messages arrive
                  </p>
                </div>
                <Switch id="sound-notifications" />
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="accessibility" className="space-y-6 py-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="reduce-motion">Reduce Motion</Label>
                  <p className="text-muted-foreground text-xs">
                    Minimize animations throughout the interface
                  </p>
                </div>
                <Switch id="reduce-motion" />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="high-contrast">High Contrast</Label>
                  <p className="text-muted-foreground text-xs">
                    Increase contrast for better visibility
                  </p>
                </div>
                <Switch id="high-contrast" />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="keyboard-shortcut">Customize Keyboard Shortcuts</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Send message</p>
                    <Input id="shortcut-send" defaultValue="Enter" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-1">New line</p>
                    <Input id="shortcut-newline" defaultValue="Shift + Enter" />
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="mt-6 flex justify-end gap-2">
          <SheetClose asChild>
            <Button variant="outline">Cancel</Button>
          </SheetClose>
          <Button onClick={handleSaveSettings}>Save Changes</Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}