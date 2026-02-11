"use client"

import { useEffect, useState } from 'react'
import { Star, Quote } from 'lucide-react'
import { cn } from '@/lib/utils'

interface TestimonialCardProps {
  name: string
  role: string
  avatar: string
  content: string
  rating: number
  delay: number
  isLoaded: boolean
}

export function TestimonialCard({
  name,
  role,
  avatar,
  content,
  rating,
  delay,
  isLoaded
}: TestimonialCardProps) {
  const [isVisible, setIsVisible] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

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
        "relative p-6 rounded-2xl transition-all duration-500",
        isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className={cn(
        "absolute inset-0 rounded-2xl transition-all duration-500",
        isHovered
          ? "bg-gradient-to-br from-blue-500/5 to-cyan-500/5 border border-blue-500/10"
          : "bg-[#192734] border border-white/5"
      )} />

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-1">
            {Array.from({ length: rating }).map((_, i) => (
              <Star key={i} className="h-4 w-4 fill-blue-400 text-blue-400" />
            ))}
          </div>
          <div className={cn(
            "w-8 h-8 rounded-full flex items-center justify-center transition-all duration-500",
            isHovered ? "bg-blue-500/20" : "bg-white/5"
          )}>
            <Quote className={cn(
              "w-4 h-4 transition-colors duration-500",
              isHovered ? "text-blue-400" : "text-gray-500"
            )} />
          </div>
        </div>

        <p className="text-gray-300 mb-6 leading-relaxed text-sm line-clamp-4">{content}</p>

        <div className="flex items-center gap-3">
          <div className={cn(
            "w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all duration-500",
            isHovered
              ? "bg-gradient-to-br from-blue-500/30 to-cyan-500/20 text-blue-400"
              : "bg-[#15202B] text-gray-400 border border-white/5"
          )}>
            {avatar}
          </div>
          <div>
            <p className="font-semibold text-white">{name}</p>
            <p className="text-sm text-gray-500">{role}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
