"use client"

import { useState } from 'react'
import { Bell, Book, Menu, Search, Settings, User, TestTube } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ThemeToggle } from '@/components/theme-toggle'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useRouter } from 'next/navigation'

interface DashboardHeaderProps {
  onToggleSidebar: () => void
  onOpenSettings: () => void
  isMobile?: boolean
}

export function DashboardHeader({ onToggleSidebar, onOpenSettings, isMobile = false }: DashboardHeaderProps) {
  const router = useRouter()
  const [isSearchExpanded, setIsSearchExpanded] = useState(false)
  
  const handleSignOut = () => {
    // In a real app, you'd implement proper sign out logic
    // delete the access token from local storage or cookies
    localStorage.removeItem('token')
    router.push('/')
  }
  
  return (
    <header className="border-b h-14 flex items-center px-4 bg-background">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSidebar}
          className={isMobile ? "" : "md:hidden"}
          aria-label="Toggle sidebar"
        >
          <Menu className="h-5 w-5" />
        </Button>
        
        <div className={`${isMobile ? "hidden" : "hidden md:flex"} items-center gap-2`}>
          <Book className="h-5 w-5 text-primary" />
          <span className="font-bold text-lg">Padhai Whallah</span>
        </div>
      </div>
      
      <div className={`flex-1 flex items-center ${isSearchExpanded ? 'justify-center' : 'justify-end'} gap-2`}>
        {isSearchExpanded ? (
          <div className="w-full max-w-md flex items-center relative">
            <Input
              placeholder="Search documents..."
              className="pl-10"
              autoFocus
              onBlur={() => setIsSearchExpanded(false)}
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          </div>
        ) : (
          <>

            <Button
              variant="default"
              size="default"
              onClick={() => router.push('/test')}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Go to Test"
              title="Go to Test Page"
            >
              <TestTube className="h-5 w-5" />
              <div className=''>Tests</div>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSearchExpanded(true)}
              className={`text-muted-foreground hover:text-foreground ${isMobile ? "hidden sm:flex" : ""}`}
              aria-label="Search"
            >
              <Search className="h-5 w-5" />
            </Button>
            
            {/* <Button
              variant="ghost"
              size="icon"
              className={`text-muted-foreground hover:text-foreground ${isMobile ? "hidden sm:flex" : ""}`}
              aria-label="Notifications"
            >
              <Bell className="h-5 w-5" />
            </Button> */}
            
            <ThemeToggle />
            
            
            
            <Button
              variant="ghost"
              size="icon"
              onClick={onOpenSettings}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Settings"
            >
              <Settings className="h-5 w-5" />
            </Button>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-full h-8 w-8 bg-primary/10 text-primary"
                  aria-label="User menu"
                >
                  <User className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>Profile</DropdownMenuItem>
                <DropdownMenuItem onClick={onOpenSettings}>Settings</DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut}>Sign out</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        )}
      </div>
    </header>
  )
}