"use client"

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { 
  Paperclip, 
  Send, 
  Smile,
  Image,
  FileText,
  Mic,
  X
} from 'lucide-react'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
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
  
  const handleAddAttachment = (type: string) => {
    // Mock attachment - in a real app, this would involve file selection
    const mockAttachment = `${type}-${Math.floor(Math.random() * 1000)}.${type === 'image' ? 'jpg' : 'pdf'}`
    setAttachments([...attachments, mockAttachment])
  }
  
  const handleRemoveAttachment = (attachment: string) => {
    setAttachments(attachments.filter(a => a !== attachment))
  }
  
  return (
    <form onSubmit={handleSubmit} className="relative">
      {/* Attachments display */}
      {attachments.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {attachments.map((attachment, index) => (
            <div 
              key={index}
              className="flex items-center gap-1 bg-muted rounded-md px-2 py-1 text-xs"
            >
              {attachment.includes('image') ? (
                <Image className="h-3 w-3" />
              ) : (
                <FileText className="h-3 w-3" />
              )}
              <span className="truncate max-w-[100px]">{attachment}</span>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-4 w-4 rounded-full hover:bg-muted-foreground/20"
                onClick={() => handleRemoveAttachment(attachment)}
              >
                <X className="h-2 w-2" />
              </Button>
            </div>
          ))}
        </div>
      )}
      
      {/* Main input area */}
      <div className="flex items-end gap-2 bg-background rounded-lg border">
        <div className="flex flex-1 items-end">
          {/* Attachment options */}
          <Popover>
            <PopoverTrigger asChild>
              <Button 
                type="button" 
                variant="ghost" 
                size="icon" 
                className="h-10 w-10 rounded-none rounded-bl-lg"
              >
                <Paperclip className="h-5 w-5 text-muted-foreground" />
              </Button>
            </PopoverTrigger>
            <PopoverContent side="top" align="start" className="w-auto p-2">
              <div className="flex gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => handleAddAttachment('image')}
                      >
                        <Image className="h-5 w-5" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Add image</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => handleAddAttachment('document')}
                      >
                        <FileText className="h-5 w-5" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Add document</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </PopoverContent>
          </Popover>
          
          {/* Textarea for message */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isTyping ? "AI is typing..." : "Type your message..."}
            className={cn(
              "flex-1 resize-none border-0 bg-transparent p-3 focus-visible:ring-0 focus-visible:ring-offset-0",
              "min-h-[40px] max-h-[200px] overflow-y-auto",
              isTyping && "text-muted-foreground"
            )}
            disabled={isTyping}
          />
          
          {/* Emoji picker */}
          <Popover>
            <PopoverTrigger asChild>
              <Button 
                type="button" 
                variant="ghost" 
                size="icon" 
                className="h-10 w-10"
                disabled={isTyping}
              >
                <Smile className="h-5 w-5 text-muted-foreground" />
              </Button>
            </PopoverTrigger>
            <PopoverContent side="top" align="end" className="w-auto p-2">
              <div className="grid grid-cols-6 gap-1">
                {["😊", "👍", "🎉", "🤔", "❤️", "😂", "🙏", "👋", "🔥", "✅", "⭐", "❓"].map(emoji => (
                  <Button
                    key={emoji}
                    type="button"
                    variant="ghost"
                    className="h-8 w-8 p-0"
                    onClick={() => setMessage(prev => prev + emoji)}
                  >
                    {emoji}
                  </Button>
                ))}
              </div>
            </PopoverContent>
          </Popover>
        </div>
        
        {/* Send button */}
        <Button 
          type="submit" 
          size="icon" 
          className="h-10 w-10 rounded-none rounded-br-lg rounded-tr-lg"
          disabled={isTyping || (!message.trim() && attachments.length === 0)}
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
    </form>
  )
}