"use client"

import { useEffect, useState } from 'react'
import { Upload, Brain, GraduationCap, ArrowRight, type LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface HowItWorksCardProps {
  step: string
  title: string
  description: string
  icon: string
  delay: number
  isLoaded: boolean
}

const iconMap: Record<string, LucideIcon> = {
  Upload,
  Brain,
  GraduationCap
}

export function HowItWorksCard({
  step,
  title,
  description,
  icon,
  delay,
  isLoaded
}: HowItWorksCardProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
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
        "relative p-8 rounded-2xl transition-all duration-500 group",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className={cn(
        "absolute inset-0 rounded-2xl transition-all duration-500 overflow-hidden",
        isHovered
          ? "bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border border-blue-500/20"
          : "bg-[#192734] border border-white/5"
      )}>
        <div className={cn(
          "absolute inset-0 opacity-0 transition-opacity duration-500",
          isHovered ? "opacity-100" : ""
        )}>
          <div className="absolute top-[-50%] left-[-50%] w-[200%] h-[200%] bg-gradient-to-r from-transparent via-blue-500/5 to-transparent animate-spin" style={{ animationDuration: '12s' }} />
        </div>
      </div>

      <div className="absolute -top-4 left-8">
        <div className={cn(
          "px-5 py-1.5 rounded-full text-sm font-bold transition-all duration-500 shadow-lg",
          isHovered
            ? "bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-blue-500/25"
            : "bg-[#15202B] text-gray-400 border border-white/5"
        )}>
          {step}
        </div>
      </div>

      <div className="relative z-10 mt-4">
        <div className={cn(
          "mb-6 w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-500",
          isHovered
            ? "bg-gradient-to-br from-blue-500/30 to-cyan-500/20 shadow-lg shadow-blue-500/20"
            : "bg-[#15202B] border border-white/5"
        )}>
          {Icon && (
            <Icon
              className={cn(
                "h-8 w-8 transition-all duration-500",
                isHovered ? "text-blue-400 scale-110" : "text-gray-400"
              )}
            />
          )}
        </div>

        <h3 className={cn(
          "text-xl font-bold mb-3 transition-colors",
          isHovered ? "text-white" : "text-gray-200"
        )}>
          {title}
        </h3>

        <p className="text-gray-400 leading-relaxed">
          {description}
        </p>
      </div>

      <div className={cn(
        "absolute bottom-8 right-8 transition-all duration-500 opacity-0 transform translate-x-4",
        isHovered ? "opacity-100 translate-x-0" : ""
      )}>
        <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center">
          <ArrowRight className="w-5 h-5 text-blue-400" />
        </div>
      </div>

      <div className={cn(
        "absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full opacity-0 transition-all duration-500",
        isHovered ? "opacity-100 translate-y-2" : ""
      )}>
        <div className="flex items-center gap-1 text-xs text-blue-400">
          <span>Learn more</span>
          <ArrowRight className="w-3 h-3" />
        </div>
      </div>
    </div>
  )
}
