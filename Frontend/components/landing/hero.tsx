"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ChevronRight, Sparkles, FileText, Zap, Users, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { StarrySky } from '@/components/ui/starry-sky'

type FloatingElement = {
  id: number
  type: 'document' | 'chat' | 'quiz' | 'search' | 'star'
  top: string
  left: string
  delay: number
  duration: number
}

export function LandingHero() {
  const [isLoaded, setIsLoaded] = useState(false)
  const [floatingElements, setFloatingElements] = useState<FloatingElement[]>([])

  useEffect(() => {
    const generateFloatingElements = (): FloatingElement[] => {
      return [
        { id: 1, type: 'document', top: '15%', left: '8%', delay: 0, duration: 15 },
        { id: 2, type: 'chat', top: '25%', left: '88%', delay: 2, duration: 18 },
        { id: 3, type: 'quiz', top: '60%', left: '5%', delay: 4, duration: 12 },
        { id: 4, type: 'search', top: '70%', left: '90%', delay: 1, duration: 20 },
        { id: 5, type: 'star', top: '10%', left: '50%', delay: 0, duration: 8 },
        { id: 6, type: 'star', top: '40%', left: '12%', delay: 3, duration: 14 },
        { id: 7, type: 'star', top: '80%', left: '75%', delay: 5, duration: 16 },
      ]
    }

    setFloatingElements(generateFloatingElements())
    setIsLoaded(true)
  }, [])

  const getFloatingIcon = (type: string) => {
    switch (type) {
      case 'document':
        return <FileText className="w-5 h-5 text-blue-400" />
      case 'chat':
        return <Zap className="w-5 h-5 text-yellow-400" />
      case 'quiz':
        return <Users className="w-5 h-5 text-green-400" />
      case 'search':
        return <Sparkles className="w-5 h-5 text-blue-400" />
      default:
        return <Sparkles className="w-5 h-5 text-blue-400" />
    }
  }

  return (
    <section className="relative pt-20 pb-32 overflow-hidden min-h-screen">
      <div className="absolute inset-0 bg-gradient-to-br from-[#0D1520] via-[#15202B] to-[#0D1520] -z-20" />

      <StarrySky starCount={150} shootingStarCount={2} className="z-0" />

      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute top-[20%] left-[10%] w-[500px] h-[500px] bg-blue-500/8 rounded-full blur-[120px]" />
        <div className="absolute bottom-[20%] right-[10%] w-[500px] h-[500px] bg-cyan-500/5 rounded-full blur-[120px]" />
        <div className="absolute top-[50%] left-[40%] w-[300px] h-[300px] bg-blue-500/3 rounded-full blur-[80px]" />
      </div>

      <div className="container px-4 mx-auto relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <div className='m-28'></div>
          {/* <div
            className={cn(
              "inline-flex items-center gap-2 px-6 py-2.5 mb-8 rounded-full neo-flat transition-all duration-500",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-blue-400 font-medium">AI-Powered Study Platform</span>
          </div> */}

          <h1
            className={cn(
              "text-4xl md:text-5xl lg:text-6xl font-bold mb-6 transition-all duration-700",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            <span className="text-white">Master Your Studies with</span>
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-blue-500 to-cyan-400 bg-clip-text text-transparent">
              Orbit AI
            </span>
          </h1>

          <p
            className={cn(
              "text-lg md:text-xl text-gray-400 mb-10 max-w-2xl mx-auto leading-relaxed transition-all duration-700 delay-100",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            Upload your syllabus and notes, generate quizzes, discuss strategy and track your progress.
            Everything you need to ace your exams in one powerful platform.
          </p>

          <div
            className={cn(
              "flex flex-col sm:flex-row justify-center gap-4 transition-all duration-700 delay-200",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            <Link href="/dashboard">
              <Button size="lg" className="w-full sm:w-auto px-8 py-6 text-lg bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 shadow-lg shadow-blue-500/25 transition-all duration-300 hover:scale-105">
                <span className="flex items-center gap-2">
                  Start Learning Free
                  <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
                </span>
              </Button>
            </Link>
            <Link href="#how-it-works">
              <Button size="lg" variant="outline" className="w-full sm:w-auto px-8 py-6 text-lg border-blue-500/30 hover:bg-blue-500/10 transition-all duration-300">
                See How It Works
              </Button>
            </Link>
          </div>

          <div
            className={cn(
              "mt-16 flex flex-wrap justify-center gap-8 md:gap-16 transition-all duration-700 delay-300",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            {[
              { value: '50K+', label: 'Students' },
              { value: '1M+', label: 'Documents' },
              { value: '95%', label: 'Success Rate' },
              { value: '24/7', label: 'AI Support' }
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-1">{stat.value}</div>
                <div className="text-sm text-blue-300/60">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        <div
          className={cn(
            "mt-20 max-w-5xl mx-auto rounded-2xl overflow-hidden border border-blue-500/10 shadow-2xl transition-all duration-1000 delay-500",
            isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
          )}
        >
          <div className="h-10 bg-[#192734] flex items-center px-4 border-b border-blue-500/10">
            <div className="flex space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500/60" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
              <div className="w-3 h-3 rounded-full bg-green-500/60" />
            </div>
            <div className="mx-auto px-4 py-1 rounded-lg bg-[#0D1520] border border-blue-500/10">
              <span className="text-xs text-blue-300/60 font-mono">orbit.app/dashboard</span>
            </div>
          </div>

          <div className="h-[380px] md:h-[450px] bg-[#0D1520] flex">
            <div className="hidden md:flex flex-col w-64 border-r border-blue-500/10 p-4">
              <div className="flex items-center gap-3 mb-6 p-3 rounded-xl bg-[#192734]">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500/30 to-cyan-500/20 flex items-center justify-center">
                  <span className="text-lg font-bold text-blue-400">O</span>
                </div>
                <span className="font-semibold text-white">Orbit</span>
              </div>

              <div className="space-y-1 flex-1">
                {['Dashboard', 'Documents', 'AI Chat', 'Quiz Generator', 'Analytics', 'Settings'].map((item, i) => (
                  <div
                    key={i}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-lg transition-colors cursor-pointer",
                      i === 0 ? "bg-blue-500/10 text-blue-400" : "hover:bg-blue-500/5 text-gray-400"
                    )}
                  >
                    <div className={cn("w-2 h-2 rounded-full", i === 0 ? "bg-blue-400" : "bg-gray-500")} />
                    <span className="text-sm">{item}</span>
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-blue-500/10">
                <div className="flex items-center gap-3 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500" />
                  <div>
                    <div className="h-3 w-20 bg-white/20 rounded mb-1" />
                    <div className="h-2 w-14 bg-white/10 rounded" />
                  </div>
                </div>
              </div>
            </div>

            <div className="flex-1 p-6 flex flex-col">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <div className="h-7 w-48 bg-blue-500/10 rounded mb-2" />
                  <div className="h-4 w-32 bg-blue-500/5 rounded" />
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                    <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                    <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                    </svg>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-6">
                {[
                  { label: 'Documents', value: '24', color: 'blue' },
                  { label: 'Study Hours', value: '48h', color: 'green' },
                  { label: 'Quizzes', value: '12', color: 'yellow' }
                ].map((stat, i) => (
                  <div key={i} className="p-4 rounded-xl bg-[#192734] border border-blue-500/10">
                    <div className="h-8 w-8 rounded-lg bg-blue-500/20 mb-2 flex items-center justify-center">
                      {stat.color === 'blue' && <FileText className="w-4 h-4 text-blue-400" />}
                      {stat.color === 'green' && <Zap className="w-4 h-4 text-green-400" />}
                      {stat.color === 'yellow' && <Users className="w-4 h-4 text-yellow-400" />}
                    </div>
                    <div className="h-6 w-12 bg-blue-500/10 rounded mb-1" />
                    <div className="h-3 w-16 bg-blue-500/5 rounded" />
                  </div>
                ))}
              </div>

              <div className="flex-1 rounded-xl bg-[#192734] border border-blue-500/10 p-4 overflow-hidden">
                <div className="flex items-center gap-2 mb-4 pb-3 border-b border-blue-500/10">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-sm text-gray-300">AI Assistant</span>
                </div>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500/30 to-cyan-500/20 flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="flex-1 p-3 rounded-2xl rounded-tl-sm bg-[#0D1520] border border-blue-500/10">
                      <div className="h-3 w-full bg-blue-500/10 rounded mb-2" />
                      <div className="h-3 w-4/5 bg-blue-500/10 rounded" />
                    </div>
                  </div>
                  <div className="flex items-start gap-3 justify-end">
                    <div className="flex-1 p-3 rounded-2xl rounded-tr-sm bg-blue-500/10 border border-blue-500/20">
                      <div className="h-3 w-full bg-blue-500/20 rounded mb-2" />
                      <div className="h-3 w-2/3 bg-blue-500/20 rounded" />
                    </div>
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0">
                      <span className="text-xs font-semibold text-white">You</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 rounded-full border-2 border-blue-500/30 flex items-start justify-center p-2">
          <div className="w-1.5 h-2 rounded-full bg-blue-500/50 animate-pulse" />
        </div>
      </div>
    </section>
  )
}
