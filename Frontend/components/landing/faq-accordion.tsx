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
            "rounded-xl transition-all duration-500 overflow-hidden",
            isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4",
            openIndex === index
              ? "bg-card"
              : "bg-card/50 hover:bg-card/70"
          )}
          style={{ transitionDelay: `${index * 50}ms` }}
        >
          <button
            className="w-full px-6 py-4 text-left flex items-center justify-between focus:outline-none"
            onClick={() => setOpenIndex(openIndex === index ? null : index)}
          >
            <span className={cn(
              "font-semibold text-lg transition-colors",
              openIndex === index ? "text-foreground" : "text-muted-foreground"
            )}>
              {item.question}
            </span>
            <ChevronDown
              className={cn(
                "h-5 w-5 transition-all duration-300",
                openIndex === index
                  ? "text-primary rotate-180"
                  : "text-muted-foreground"
              )}
            />
          </button>
          <div
            className={cn(
              "transition-all duration-300 ease-in-out overflow-hidden",
              openIndex === index ? "max-h-48 opacity-100" : "max-h-0 opacity-0"
            )}
          >
            <div className="px-6 pb-4 text-muted-foreground">
              {item.answer}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}