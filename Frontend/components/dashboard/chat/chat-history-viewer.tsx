"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { ChatHistory } from '@/lib/data'
import { Search, MessageCircle, Clock, ChevronRight, Trash2 } from 'lucide-react'
import { format } from 'date-fns'
import { cn } from '@/lib/utils'

interface ChatHistoryViewerProps {
  chatHistory: ChatHistory[]
  onSelectChat: (chat: ChatHistory) => void
  onDeleteChat: (chatId: string) => void
  className?: string
}

export function ChatHistoryViewer({
  chatHistory,
  onSelectChat,
  onDeleteChat,
  className
}: ChatHistoryViewerProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [formattedChats, setFormattedChats] = useState<(ChatHistory & { formattedDate: string })[]>([])
  
  // Format dates on client-side only to avoid hydration mismatches
  useEffect(() => {
    // Filter and format chat history based on search term
    const filtered = chatHistory.filter(chat => 
      chat.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      chat.documentTitle.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    // Sort by most recent first
    const sorted = [...filtered].sort((a, b) => 
      new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime()
    );
    
    // Add formatted dates
    const formatted = sorted.map(chat => ({
      ...chat,
      formattedDate: format(new Date(chat.lastUpdated), 'MMM d, yyyy')
    }));
    
    setFormattedChats(formatted);
  }, [chatHistory, searchTerm]);
  
  return (
    <div className={cn("border-r flex flex-col h-full", className)}>
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold mb-4">Chat History</h2>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search chats..."
            className="pl-10"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        {formattedChats.length > 0 ? (
          <div className="p-2">
            {formattedChats.map((chat) => (
              <div
                key={chat.id}
                className="rounded-md p-3 hover:bg-accent cursor-pointer group"
                onClick={() => onSelectChat(chat)}
              >
                <div className="flex justify-between items-start">
                  <h3 className="font-medium truncate">{chat.title}</h3>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteChat(chat.id);
                    }}
                  >
                    <Trash2 className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </div>
                
                <div className="flex items-center text-xs text-muted-foreground mt-1">
                  <MessageCircle className="h-3 w-3 mr-1" />
                  <span className="truncate">{chat.documentTitle}</span>
                </div>
                
                <div className="flex justify-between items-center mt-2 text-xs text-muted-foreground">
                  <div className="flex items-center">
                    <Clock className="h-3 w-3 mr-1" />
                    <span>{chat.formattedDate}</span>
                  </div>
                  <ChevronRight className="h-4 w-4" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full p-4 text-center">
            <MessageCircle className="h-10 w-10 text-muted-foreground mb-2 opacity-50" />
            <p className="text-muted-foreground">No chat history found</p>
            {searchTerm && (
              <p className="text-sm text-muted-foreground">Try adjusting your search</p>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}