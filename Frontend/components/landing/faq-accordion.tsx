"use client"

import { useState, useEffect } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface FAQItem {
  question: string
  answer: string
}

interface FAQAccordionProps {
  items: FAQItem[]
  isLoaded: boolean
}

export function FAQAccordion({ items, isLoaded }: FAQAccordionProps) {
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  useEffect(() => {
    if (isLoaded) {
      const timers: NodeJS.Timeout[] = []
      items.forEach((_, index) => {
        timers.push(
          setTimeout(() => {
            setOpenIndex(prev => prev === null ? 0 : prev)
          }, 100)
        )
      })
      return () => timers.forEach(clearTimeout)
    }
  }, [isLoaded, items.length])

  return (
    <div className="space-y-4">
      {items.map((item, index) => (
        <div
          key={index}
          className={cn(
            "rounded-xl border transition-all duration-500 overflow-hidden",
            isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4",
            openIndex === index
              ? "bg-[#192734] border-blue-500/20"
              : "bg-[#15202B] border-white/5 hover:border-white/10"
          )}
          style={{ transitionDelay: `${index * 50}ms` }}
        >
          <button
            className="w-full px-6 py-4 text-left flex items-center justify-between focus:outline-none"
            onClick={() => setOpenIndex(openIndex === index ? null : index)}
          >
            <span className={cn(
              "font-semibold text-lg transition-colors",
              openIndex === index ? "text-white" : "text-gray-200"
            )}>
              {item.question}
            </span>
            <ChevronDown
              className={cn(
                "h-5 w-5 transition-all duration-300",
                openIndex === index
                  ? "text-blue-400 rotate-180"
                  : "text-gray-500"
              )}
            />
          </button>
          <div
            className={cn(
              "transition-all duration-300 ease-in-out overflow-hidden",
              openIndex === index ? "max-h-48 opacity-100" : "max-h-0 opacity-0"
            )}
          >
            <div className="px-6 pb-4 text-gray-400">
              {item.answer}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
