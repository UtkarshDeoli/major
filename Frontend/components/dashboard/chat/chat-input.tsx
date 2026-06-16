"use client"

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import {
  Paperclip,
  Send,
  X,
  FileText,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSendMessage: (content: string, attachments: string[]) => void
  isTyping: boolean
  disabled?: boolean
}

export function ChatInput({ onSendMessage, isTyping, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [attachments, setAttachments] = useState<string[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [message])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (message.trim() || attachments.length > 0) {
      onSendMessage(message.trim(), attachments)
      setMessage('')
      setAttachments([])

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files) {
      const newAttachments = Array.from(files).map(f => f.name)
      setAttachments(prev => [...prev, ...newAttachments])
    }
    // Reset the input so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleRemoveAttachment = (attachment: string) => {
    setAttachments(attachments.filter(a => a !== attachment))
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      {/* Hidden file input */}
      <input type="file" className="hidden" ref={fileInputRef} onChange={handleFileSelect} accept=".pdf,.doc,.docx,.txt" />

      {/* Attachments display */}
      {attachments.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {attachments.map((attachment, index) => (
            <div
              key={index}
              className="flex items-center gap-1 bg-muted rounded-md px-2 py-1 text-xs"
            >
              <FileText className="h-3 w-3" />
              <span className="truncate max-w-[100px]">{attachment}</span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-4 w-4 rounded-md hover:bg-muted-foreground/20"
                onClick={() => handleRemoveAttachment(attachment)}
              >
                <X className="h-2 w-2" />
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* Main input area */}
      <div className="flex items-end gap-2 rounded-md border bg-background">
        <div className="flex flex-1 items-end">
          {/* Attachment button */}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-9 w-9 rounded-md"
            onClick={() => fileInputRef.current?.click()}
            disabled={isTyping || disabled}
          >
            <Paperclip className="h-4 w-4 text-muted-foreground" />
          </Button>

          {/* Textarea for message */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isTyping ? "AI is typing..." : "Type your message..."}
            className={cn(
              "flex-1 resize-none border-0 bg-transparent p-2 text-[13px] focus-visible:ring-0 focus-visible:ring-offset-0",
              "min-h-[36px] max-h-[200px] overflow-y-auto",
              isTyping && "text-muted-foreground"
            )}
            disabled={isTyping || disabled}
          />
        </div>

        {/* Send button */}
        <Button
          type="submit"
          size="icon"
          className="h-9 w-9 rounded-md"
          disabled={isTyping || (!message.trim() && attachments.length === 0) || disabled}
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </form>
  )
}