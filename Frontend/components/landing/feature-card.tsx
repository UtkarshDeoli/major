"use client"

import {
  FileText, MessageSquare, Search, FileUp, BrainCircuit, Settings, Brain, Zap,
  type LucideIcon
} from 'lucide-react'

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
  Settings,
  Brain,
  Zap
}

export function LandingFeatureCard({
  icon,
  title,
  description,
  delay,
  isLoaded
}: FeatureCardProps) {
  const Icon = iconMap[icon]

  return (
    <div
      className={`group relative p-6 rounded-2xl transition-all duration-500 ${
        isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      }`}
      style={{ transitionDelay: `${delay * 1000}ms` }}
    >
      <div className={`absolute inset-0 rounded-2xl bg-[#192734] border border-blue-500/10 transition-all duration-500 group-hover:bg-gradient-to-br group-hover:from-blue-500/10 group-hover:to-cyan-500/5 group-hover:border-blue-500/20 group-hover:shadow-lg group-hover:shadow-blue-500/10`} />

      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-all duration-500 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
        <div className="absolute bottom-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent" />
        <div className="absolute top-[-50%] left-[-50%] w-[200%] h-[200%] bg-gradient-to-r from-transparent via-blue-500/3 to-transparent animate-spin" style={{ animationDuration: '15s' }} />
      </div>

      <div className="relative z-10">
        <div className={`mb-5 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-500 ${
          isLoaded ? "opacity-100" : "opacity-0"
        } bg-[#0D1520] border border-blue-500/10 group-hover:bg-gradient-to-br group-hover:from-blue-500/30 group-hover:to-cyan-500/20 group-hover:border-blue-500/30 group-hover:shadow-lg group-hover:shadow-blue-500/20`}>
          {Icon && (
            <Icon
              className={`h-6 w-6 transition-all duration-500 ${
                isLoaded ? "opacity-100" : "opacity-0"
              } text-gray-400 group-hover:text-blue-400 group-hover:scale-110`}
            />
          )}
        </div>

        <h3 className={`text-lg font-semibold mb-2 transition-colors ${
          isLoaded ? "opacity-100" : "opacity-0"
        } text-gray-200 group-hover:text-white`}>
          {title}
        </h3>

        <p className={`text-gray-400 leading-relaxed text-sm transition-all duration-500 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}>
          {description}
        </p>
      </div>

      <div className={`absolute bottom-4 right-4 transition-all duration-500 opacity-0 transform translate-x-2 group-hover:opacity-100 group-hover:translate-x-0`}>
        <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
          <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </div>
    </div>
  )
}
