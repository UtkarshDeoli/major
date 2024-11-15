"use client"

import { useState } from 'react'
import { Message, MessageRole } from '@/lib/data'
import { formatRelativeTime } from '@/lib/utils'
import { CheckCheck, Copy, Smile, ThumbsUp } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'
import remarkGfm from 'remark-gfm'
import ReactMarkdown from 'react-markdown'

interface MessageItemProps {
  message: Message
  isFirstInGroup: boolean
  isLastInGroup: boolean
  onAddReaction: (messageId: string, reaction: string) => void
}

export function MessageItem({ 
  message, 
  isFirstInGroup,
  isLastInGroup,
  onAddReaction
}: MessageItemProps) {
  const [isCopied, setIsCopied] = useState(false)
  const { toast } = useToast()
  const isUser = message.role === MessageRole.User
  
  const handleCopyMessage = () => {
    navigator.clipboard.writeText(message.content)
    setIsCopied(true)
    
    toast({
      description: "Message copied to clipboard"
    })
    
    setTimeout(() => setIsCopied(false), 2000)
  }
  
  const handleAddReaction = (reaction: string) => {
    onAddReaction(message.id, reaction)
  }
  
  return (
    <div 
      className={cn(
        "flex items-start gap-3", 
        isUser && "justify-end"
      )}
    >
      {!isUser && isFirstInGroup && (
        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
          <span className="text-primary text-xs font-semibold">AI</span>
        </div>
      )}
      
      {!isUser && !isFirstInGroup && <div className="w-8" />}
      
      <div 
        className={cn(
          "relative group rounded-lg p-3 max-w-[80%]",
          isUser 
            ? "bg-primary/10 text-foreground" 
            : "bg-muted text-foreground"
        )}
      >
        {/* Message content */}
        <div className="prose dark:prose-invert prose-sm max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              pre: ({ node, ...props }) => (
                <div className="overflow-auto rounded-md bg-secondary p-2 my-2">
                  <pre {...props} />
                </div>
              ),
              code: ({ node, ...props }) => (
                <code className="rounded-sm bg-secondary px-1 py-0.5" {...props} />
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        
        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.attachments.map((attachment, index) => (
              <div
                key={index}
                className="bg-muted/50 rounded-md px-2 py-1 text-xs flex items-center gap-1"
              >
                <span>{attachment}</span>
              </div>
            ))}
          </div>
        )}
        
        {/* Reactions */}
        {message.reactions && Object.keys(message.reactions).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {Object.entries(message.reactions).map(([emoji, count]) => (
              <div 
                key={emoji}
                className="flex items-center bg-muted/50 rounded-full px-2 py-0.5 text-xs"
              >
                <span>{emoji}</span>
                <span className="ml-1 text-muted-foreground">{count}</span>
              </div>
            ))}
          </div>
        )}
        
        {/* Message actions - only visible on hover */}
        <div 
          className={cn(
            "absolute top-1 opacity-0 group-hover:opacity-100 transition-opacity",
            isUser ? "left-1" : "right-1"
          )}
        >
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 rounded-full hover:bg-background/80"
              onClick={handleCopyMessage}
            >
              {isCopied ? (
                <CheckCheck className="h-3 w-3" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 rounded-full hover:bg-background/80"
                >
                  <Smile className="h-3 w-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align={isUser ? "start" : "end"} className="w-auto p-2">
                <div className="grid grid-cols-6 gap-1">
                  {["👍", "❤️", "😊", "🎉", "🤔", "😂"].map(emoji => (
                    <DropdownMenuItem
                      key={emoji}
                      className="p-0 h-8 w-8 flex items-center justify-center cursor-pointer"
                      onClick={() => handleAddReaction(emoji)}
                    >
                      {emoji}
                    </DropdownMenuItem>
                  ))}
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        
        {/* Timestamp - only shown for last message in a group */}
        {isLastInGroup && (
          <div 
            className={cn(
              "text-[10px] text-muted-foreground mt-1",
              isUser ? "text-right" : "text-left"
            )}
          >
            {formatRelativeTime(message.timestamp)}
          </div>
        )}
      </div>
      
      {isUser && isFirstInGroup && (
        <div className="w-8 h-8 rounded-full bg-accent/30 flex items-center justify-center flex-shrink-0">
          <span className="text-accent-foreground text-xs font-semibold">You</span>
        </div>
      )}
      
      {isUser && !isFirstInGroup && <div className="w-8" />}
    </div>
  )
}