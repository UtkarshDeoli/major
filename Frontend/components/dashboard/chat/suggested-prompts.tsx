import { Button } from '@/components/ui/button'
import { Document } from '@/lib/data'

interface SuggestedPromptsProps {
  onSelectPrompt: (prompt: string) => void
  document: Document
}

export function SuggestedPrompts({ onSelectPrompt, document }: SuggestedPromptsProps) {
  // Generate dynamic prompts based on document information
  const prompts = [
    `Summarize ${document.title} in 3-5 key points`,
    `Explain the main concepts in ${document.title} in simple terms`,
    `What are the most important formulas or equations in ${document.title}?`,
    `Generate 5 practice questions based on ${document.title}`,
  ]
  
  return (
    <div className="grid grid-cols-1 sm:grid-cols-1 gap-2 max-w-xl mx-auto w-full">
      {prompts.map((prompt, index) => (
        <Button
          key={index}
          variant="outline"
          className="justify-start h-auto py-3 px-4 text-left overflow-hidden"
          onClick={() => onSelectPrompt(prompt)}
        >
          {prompt}
        </Button>
      ))}
    </div>
  )
}