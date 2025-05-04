"use client"

import { Message, MessageRole } from '@/lib/data'
import { MessageItem } from '@/components/dashboard/chat/message-item'
import { TypingIndicator } from '@/components/dashboard/chat/typing-indicator'
import { formatMessageGroups } from '@/lib/utils'

interface MessageListProps {
  messages: Message[]
  isTyping: boolean
  onAddReaction: (messageId: string, reaction: string) => void
}

export function MessageList({ messages, isTyping, onAddReaction }: MessageListProps) {
  // Group messages by role and by time
  const messageGroups = formatMessageGroups(messages)
  
  return (
    <div className="space-y-6">
      {messageGroups.map((group, groupIndex) => (
        <div key={groupIndex} className="space-y-2">
          {group.messages.map((message: Message, messageIndex: number) => (
            <MessageItem
              key={message.id}
              message={message}
              isFirstInGroup={messageIndex === 0}
              isLastInGroup={messageIndex === group.messages.length - 1}
              onAddReaction={onAddReaction}
            />
          ))}
        </div>
      ))}
      
      {isTyping && (
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
            <span className="text-primary text-xs font-semibold">AI</span>
          </div>
          <div className="rounded-lg bg-muted p-3 max-w-[80%]">
            <TypingIndicator />
          </div>
        </div>
      )}
    </div>
  )
}