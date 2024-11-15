"use client"

import { useEffect, useState } from 'react'
import { 
  FileText, MessageSquare, Search, FileUp, BrainCircuit, Settings,
  type LucideIcon
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface FeatureCardProps {
  icon: string
  title: string
  description: string
  delay: number
  isLoaded: boolean
}

const iconMap: Record<string, LucideIcon> = {
  FileText,
  MessageSquare,
  Search,
  FileUp,
  BrainCircuit,
  Settings
}

export function LandingFeatureCard({ 
  icon, 
  title, 
  description, 
  delay,
  isLoaded
}: FeatureCardProps) {
  const [isVisible, setIsVisible] = useState(false)
  const Icon = iconMap[icon]

  useEffect(() => {
    if (isLoaded) {
      const timer = setTimeout(() => {
        setIsVisible(true)
      }, delay * 1000)
      
      return () => clearTimeout(timer)
    }
  }, [isLoaded, delay])

  return (
    <div
      className={cn(
        "group p-6 rounded-lg border bg-card transition-all duration-500 hover:shadow-md hover:-translate-y-1",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      )}
    >
      <div className="mb-4 w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
        {Icon && <Icon className="h-6 w-6 text-primary" />}
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  )
}