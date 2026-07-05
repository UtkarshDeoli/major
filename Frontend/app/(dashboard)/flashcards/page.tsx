"use client"

import { useEffect, useState, useCallback } from "react"
import { FlashcardDeck } from "@/components/dashboard/flashcard/flashcard-deck"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sparkles, Loader2 } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useSearchParams, useRouter } from "next/navigation"
import { flashcardAPI } from "@/lib/api"
import { getErrorMessage } from "@/lib/errors"

interface Deck {
  id: string
  title: string
  subject?: string
  card_count: number
  cards: Array<{ id: string; front: string; back: string }>
}

export default function FlashcardsPage() {
  const { toast } = useToast()
  const router = useRouter()
  const searchParams = useSearchParams()
  const targetDeckId = searchParams.get("deck")

  const [decks, setDecks] = useState<Deck[]>([])
  const [selectedId, setSelectedId] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [loadingDeck, setLoadingDeck] = useState(false)

  const fetchDecks = useCallback(async () => {
    setLoading(true)
    try {
      const list = await flashcardAPI.listDecks()
      const mapped: Deck[] = (list || []).map((d: any) => ({
        id: d.id, title: d.title, subject: d.subject, card_count: d.card_count, cards: [],
      }))
      setDecks(mapped)
      if (targetDeckId && mapped.some((d) => d.id === targetDeckId)) {
        setSelectedId(targetDeckId)
      } else if (mapped.length > 0 && !selectedId) {
        setSelectedId(mapped[0].id)
      }
    } catch (e) {
      toast({ title: "Couldn't load decks", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoading(false)
    }
  }, [targetDeckId, selectedId, toast])

  useEffect(() => { fetchDecks() }, [fetchDecks])

  const selectedDeck = decks.find((d) => d.id === selectedId)

  const loadDeckCards = useCallback(async (deckId: string) => {
    if (!deckId) return
    const existing = decks.find((d) => d.id === deckId)
    if (existing && existing.cards.length > 0) return
    setLoadingDeck(true)
    try {
      const detail = await flashcardAPI.getDeck(deckId)
      const cards = (detail.cards || []).map((c: any) => ({ id: c.id, front: c.front, back: c.back }))
      setDecks((prev) => prev.map((d) => d.id === deckId ? { ...d, cards, card_count: cards.length } : d))
    } catch (e) {
      toast({ title: "Couldn't load cards", description: getErrorMessage(e), variant: "destructive" })
    } finally {
      setLoadingDeck(false)
    }
  }, [decks, toast])

  useEffect(() => {
    if (selectedId) loadDeckCards(selectedId)
  }, [selectedId, loadDeckCards])

  const handleReview = async (cardId: string, grade: "again" | "hard" | "good" | "easy") => {
    try {
      await flashcardAPI.reviewCard(cardId, grade)
    } catch (e) {
      // Non-fatal: review state still advances locally
      console.error("Review sync failed:", e)
    }
  }

  return (
    <div className="max-w-5xl mx-auto py-8 px-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Flashcards.</h1>
          <p className="text-sm text-muted-foreground mt-1">Review key concepts with spaced repetition.</p>
        </div>
        <Button size="sm" className="rounded-md h-8 text-[13px]" onClick={() => router.push("/chat")}>
          <Sparkles className="h-3.5 w-3.5 mr-1.5" />
          Generate from a material
        </Button>
      </div>

      {loading ? (
        <div className="flex justify-center py-12"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
      ) : decks.length === 0 ? (
        <div className="rounded-md border bg-card p-12 text-center">
          <p className="text-sm text-muted-foreground">No decks yet. Open a material in Chat and use the Flashcards action to generate one.</p>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Deck:</span>
            <Select value={selectedId} onValueChange={setSelectedId}>
              <SelectTrigger className="w-64 rounded-md h-8 text-[13px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {decks.map((deck) => (
                  <SelectItem key={deck.id} value={deck.id}>
                    {deck.title}{deck.subject ? ` · ${deck.subject}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {loadingDeck ? (
            <div className="flex justify-center py-8"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
          ) : selectedDeck ? (
            <FlashcardDeck
              cards={selectedDeck.cards}
              title={selectedDeck.title}
              onReview={handleReview}
            />
          ) : null}
        </>
      )}
    </div>
  )
}