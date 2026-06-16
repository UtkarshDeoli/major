"use client"

import { useState, useCallback } from "react"
import { FlashcardCard } from "./flashcard-card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

interface FlashcardDeckProps {
  cards: Array<{ front: string; back: string }>
  title?: string
}

export function FlashcardDeck({ cards, title }: FlashcardDeckProps) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [known, setKnown] = useState<Set<number>>(new Set())

  const handleNext = useCallback(() => {
    setCurrentIndex((i) => Math.min(i + 1, cards.length - 1))
  }, [cards.length])

  const handlePrev = useCallback(() => {
    setCurrentIndex((i) => Math.max(i - 1, 0))
  }, [])

  const handleMarkKnown = useCallback(() => {
    setKnown((prev) => new Set(prev).add(currentIndex))
    handleNext()
  }, [currentIndex, handleNext])

  if (cards.length === 0) {
    return (
      <div className="rounded-md border bg-card p-12 text-center">
        <p className="text-sm text-muted-foreground">No flashcards in this deck.</p>
      </div>
    )
  }

  const currentCard = cards[currentIndex]

  return (
    <div className="space-y-4">
      {title && (
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">{title}</h3>
          <span className="text-xs text-muted-foreground font-mono">
            {currentIndex + 1} / {cards.length}
          </span>
        </div>
      )}
      <div className="flex gap-1">
        {cards.map((_, i) => (
          <div
            key={i}
            className={cn(
              "h-1 flex-1 rounded-full transition-colors",
              i < currentIndex ? "bg-primary" : known.has(i) ? "bg-green-500/50" : "bg-muted"
            )}
          />
        ))}
      </div>
      <FlashcardCard front={currentCard.front} back={currentCard.back} />
      <div className="flex items-center justify-center gap-2">
        <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]" onClick={handlePrev} disabled={currentIndex === 0}>
          <ChevronLeft className="h-3.5 w-3.5 mr-1" />
          Previous
        </Button>
        <Button variant="outline" size="sm" className="rounded-md h-8 text-[13px]" onClick={handleMarkKnown}>
          Know
        </Button>
        <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={handleNext} disabled={currentIndex === cards.length - 1}>
          Next
          <ChevronRight className="h-3.5 w-3.5 ml-1" />
        </Button>
      </div>
      <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
        <span>{known.size} known</span>
        <span>{cards.length - known.size} remaining</span>
      </div>
    </div>
  )
}