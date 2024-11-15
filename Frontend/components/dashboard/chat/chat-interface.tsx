"use client"

import { useState, useRef, useEffect } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { ChatInput } from '@/components/dashboard/chat/chat-input'
import { MessageList } from '@/components/dashboard/chat/message-list'
import { SuggestedPrompts } from '@/components/dashboard/chat/suggested-prompts'
import { createMessage, DEFAULT_MESSAGES, Message, MessageRole } from '@/lib/data'
import { Document } from '@/lib/data'
import { useToast } from '@/hooks/use-toast'
import { nanoid, cn } from '@/lib/utils'

interface ChatInterfaceProps {
  document: Document
  initialMessages?: Message[]
  className?: string
}

export function ChatInterface({ document, initialMessages = [], className }: ChatInterfaceProps) {
  // Fix: Initialize with empty array and update in useEffect to avoid hydration mismatch
  const [messages, setMessages] = useState<Message[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()
  
  // Initialize messages on client-side only to avoid hydration mismatch
  useEffect(() => {
    setMessages(initialMessages.length > 0 ? initialMessages : DEFAULT_MESSAGES);
  }, [initialMessages]);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])
  
  const handleSendMessage = (content: string, attachments: string[] = []) => {
    // Add user message
    const userMessage: Message = createMessage({
      id: nanoid(),
      role: MessageRole.User,
      content,
      attachments,
      timestamp: new Date().toISOString(),
    })
    
    setMessages(prev => [...prev, userMessage])
    
    // Simulate AI thinking
    setIsTyping(true)
    
    // Simulate AI response after a delay
    setTimeout(() => {
      setIsTyping(false)
      
      // Generate AI response
      const aiMessage: Message = createMessage({
        id: nanoid(),
        role: MessageRole.Assistant,
        content: generateAIResponse(content, document),
        timestamp: new Date().toISOString(),
      })
      
      setMessages(prev => [...prev, aiMessage])
    }, 1500 + Math.random() * 1000)
  }
  
  const handleUsePrompt = (prompt: string) => {
    handleSendMessage(prompt)
  }
  
  const handleAddReaction = (messageId: string, reaction: string) => {
    setMessages(prev => 
      prev.map(msg => {
        if (msg.id === messageId) {
          const existingReactions = msg.reactions || {}
          return {
            ...msg,
            reactions: {
              ...existingReactions,
              [reaction]: (existingReactions[reaction] || 0) + 1
            }
          }
        }
        return msg
      })
    )
    
    toast({
      description: `You reacted with ${reaction}`
    })
  }
  
  return (
    <div className={cn("flex flex-col h-full", className)}>
      <div className="flex-1 flex flex-col">
        <ScrollArea className="flex-1 p-3 sm:p-4" ref={scrollAreaRef}>
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-3 sm:p-4">
              <h3 className="text-base sm:text-lg font-semibold mb-2">Chat about this document</h3>
              <p className="text-sm sm:text-base text-muted-foreground mb-4 sm:mb-6">
                Ask questions, get summaries, or request explanations about {document.title}
              </p>
              <SuggestedPrompts onSelectPrompt={handleUsePrompt} document={document} />
            </div>
          ) : (
            <MessageList 
              messages={messages} 
              isTyping={isTyping}
              onAddReaction={handleAddReaction}
            />
          )}
        </ScrollArea>
        
        <div className="p-2 sm:p-4 border-t">
          <ChatInput onSendMessage={handleSendMessage} isTyping={isTyping} />
        </div>
      </div>
    </div>
  )
}

// Mock AI response generator
function generateAIResponse(prompt: string, document: Document): string {
  const responses = [
    `Based on ${document.title}, I can tell you that the key concepts covered include study techniques, memory enhancement, and effective note-taking.`,
    `The document discusses various exam preparation strategies, including spaced repetition and active recall.`,
    `According to the material, the most effective study method involves breaking down complex topics into smaller, manageable chunks.`,
    `The author recommends creating mind maps to visualize connections between different concepts.`,
    `This document emphasizes the importance of understanding concepts rather than memorizing facts.`,
    `I'd suggest focusing on the sections about time management and prioritization of topics based on their importance.`,
    `The material covers both theoretical knowledge and practical applications with several examples and case studies.`
  ]
  
  return responses[Math.floor(Math.random() * responses.length)]
}