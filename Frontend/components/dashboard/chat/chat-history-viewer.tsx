"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Input } from '@/components/ui/input'
import { ChatSession, convertApiSessionToSession } from '@/lib/data'
import { Search, MessageCircle, Clock, ChevronRight, Trash2 } from 'lucide-react'
import { format } from 'date-fns'
import { cn } from '@/lib/utils'
import { chatAPI } from '@/lib/api'
import { useToast } from '@/hooks/use-toast'

interface ChatHistoryViewerProps {
  chatHistory: ChatSession[]
  onSelectChat: (chat: ChatSession) => void
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
  const [formattedChats, setFormattedChats] = useState<(ChatSession & { formattedDate: string })[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()
  
  // Fetch chat sessions on mount
  useEffect(() => {
    fetchChatSessions();
  }, []);

  // Format and filter chat sessions
  useEffect(() => {
    formatChatSessions(chatHistory);
  }, [chatHistory, searchTerm]);
  
  // Fetch chat sessions from API
  const fetchChatSessions = async () => {
    try {
      setIsLoading(true);
      
      const apiSessions = await chatAPI.listChatSessions();
      
      // Convert API sessions to our format
      const convertedSessions = apiSessions.map((session: any) => 
        convertApiSessionToSession(session)
      );
      
      formatChatSessions(convertedSessions);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching chat sessions:", error);
      toast({
        title: "Failed to load chat history",
        description: "Could not retrieve your chat sessions. Using default data instead.",
        variant: "destructive"
      });
      
      formatChatSessions(chatHistory);
      setIsLoading(false);
    }
  };

  // Format chat sessions for display
  const formatChatSessions = (sessions: ChatSession[]) => {
    // Filter based on search term
    const filtered = sessions.filter(chat => 
      chat.title.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    // Sort by most recent first
    const sorted = [...filtered].sort((a, b) => 
      new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    );
    
    // Add formatted dates
    const formatted = sorted.map(chat => ({
      ...chat,
      formattedDate: format(new Date(chat.updated_at), 'MMM d, yyyy')
    }));
    
    setFormattedChats(formatted);
  };
  
  return (
    <div className={cn("border-r flex flex-col h-full bg-card", className)}>
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
        {isLoading ? (
          <div className="flex justify-center py-8">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : formattedChats.length > 0 ? (
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
                
                {/* <div className="flex items-center text-xs text-muted-foreground mt-1">
                  <MessageCircle className="h-3 w-3 mr-1" />
                  <span className="truncate">
                    {chat.messages ? `${chat.messages.length} messages` : '0 messages'}
                  </span>
                </div> */}
                
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