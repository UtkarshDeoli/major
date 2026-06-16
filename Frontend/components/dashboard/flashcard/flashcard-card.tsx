"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"

interface FlashcardCardProps {
  front: string
  back: string
  className?: string
}

export function FlashcardCard({ front, back, className }: FlashcardCardProps) {
  const [isFlipped, setIsFlipped] = useState(false)

  return (
    <button
      onClick={() => setIsFlipped(!isFlipped)}
      className={cn("w-full h-64 [perspective:1000px] cursor-pointer", className)}
    >
      <div className={cn(
        "relative w-full h-full transition-transform duration-500 [transform-style:preserve-3d]",
        isFlipped && "[transform:rotateY(180deg)]"
      )}>
        <div className="absolute inset-0 rounded-md border bg-card p-6 flex flex-col items-center justify-center [backface-visibility:hidden]">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Question</p>
          <p className="text-sm font-medium text-center">{front}</p>
        </div>
        <div className="absolute inset-0 rounded-md border bg-primary text-primary-foreground p-6 flex flex-col items-center justify-center [backface-visibility:hidden] [transform:rotateY(180deg)]">
          <p className="text-xs uppercase tracking-wider mb-2 opacity-70">Answer</p>
          <p className="text-sm font-medium text-center">{back}</p>
        </div>
      </div>
    </button>
  )
}