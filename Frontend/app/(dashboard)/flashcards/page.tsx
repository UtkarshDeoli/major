"use client"

import { useState } from "react"
import { FlashcardDeck } from "@/components/dashboard/flashcard/flashcard-deck"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sparkles } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

const DEMO_DECKS = [
  {
    id: "demo-1",
    title: "Calculus Basics",
    cards: [
      { front: "What is the derivative of x^n?", back: "nx^(n-1) — the Power Rule." },
      { front: "What is the chain rule?", back: "d/dx[f(g(x))] = f'(g(x)) * g'(x)." },
      { front: "What is the integral of 1/x?", back: "ln|x| + C." },
      { front: "What is L'Hopital's Rule?", back: "If lim f(x)/g(x) is 0/0 or inf/inf, then lim = lim f'(x)/g'(x)." },
    ],
  },
  {
    id: "demo-2",
    title: "Physics — Mechanics",
    cards: [
      { front: "Newton's Second Law?", back: "F = ma. Force equals mass times acceleration." },
      { front: "What is kinetic energy?", back: "KE = 1/2 mv^2." },
      { front: "What is the unit of force?", back: "Newton (N) = kg * m/s^2." },
    ],
  },
]

export default function FlashcardsPage() {
  const { toast } = useToast()
  const [selectedDeck, setSelectedDeck] = useState(DEMO_DECKS[0])
  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerateDeck = async () => {
    setIsGenerating(true)
    try {
      toast({ title: "Coming soon.", description: "AI-generated flashcards from your materials will be available in a future update." })
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Flashcards.</h1>
          <p className="text-sm text-muted-foreground mt-1">Review key concepts with spaced repetition.</p>
        </div>
        <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={handleGenerateDeck} disabled title="AI flashcard generation coming soon">
          <Sparkles className="h-3.5 w-3.5 mr-1.5" />
          Generate Deck
        </Button>
      </div>
      <div className="rounded-md border border-primary/30 bg-primary/5 p-3 text-sm text-muted-foreground">
        <span className="font-medium text-foreground">Flashcards are in beta.</span>{" "}
        AI-generated decks arrive in a future update. The decks below are sample content.
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Deck:</span>
        <Select value={selectedDeck.id} onValueChange={(id) => {
          const deck = DEMO_DECKS.find(d => d.id === id)
          if (deck) setSelectedDeck(deck)
        }}>
          <SelectTrigger className="w-48 rounded-md h-8 text-[13px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DEMO_DECKS.map((deck) => (
              <SelectItem key={deck.id} value={deck.id}>{deck.title}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <FlashcardDeck cards={selectedDeck.cards} title={selectedDeck.title} />
    </div>
  )
}