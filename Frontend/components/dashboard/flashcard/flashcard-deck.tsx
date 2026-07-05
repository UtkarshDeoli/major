"use client"

import { useState, useCallback } from "react"
import { FlashcardCard } from "./flashcard-card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

type Grade = "again" | "hard" | "good" | "easy"

interface FlashcardDeckProps {
  cards: Array<{ id?: string; front: string; back: string }>
  title?: string
  /** When provided, shows SRS grade buttons and calls back on each review. */
  onReview?: (cardId: string, grade: Grade) => void
}

export function FlashcardDeck({ cards, title, onReview }: FlashcardDeckProps) {
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

  const handleGrade = useCallback((grade: Grade) => {
    const card = cards[currentIndex]
    if (card?.id && onReview) onReview(card.id, grade)
    setKnown((prev) => grade === "again" ? prev : new Set(prev).add(currentIndex))
    handleNext()
  }, [cards, currentIndex, onReview, handleNext])

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
      {onReview ? (
        <div className="grid grid-cols-4 gap-2">
          <Button variant="outline" size="sm" className="rounded-md h-8 text-[12px] text-red-600" onClick={() => handleGrade("again")}>Again</Button>
          <Button variant="outline" size="sm" className="rounded-md h-8 text-[12px] text-amber-600" onClick={() => handleGrade("hard")}>Hard</Button>
          <Button variant="outline" size="sm" className="rounded-md h-8 text-[12px] text-blue-600" onClick={() => handleGrade("good")}>Good</Button>
          <Button variant="outline" size="sm" className="rounded-md h-8 text-[12px] text-green-600" onClick={() => handleGrade("easy")}>Easy</Button>
        </div>
      ) : (
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
      )}
      <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
        <span>{known.size} known</span>
        <span>{cards.length - known.size} remaining</span>
      </div>
    </div>
  )
}