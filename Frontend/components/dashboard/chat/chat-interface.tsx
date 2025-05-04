"use client"

import { useState, useRef, useEffect } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { ChatInput } from '@/components/dashboard/chat/chat-input'
import { MessageList } from '@/components/dashboard/chat/message-list'
import { SuggestedPrompts } from '@/components/dashboard/chat/suggested-prompts'
import { createMessage, MessageRole, Message, Document, ChatSession, convertApiSessionToSession } from '@/lib/data'
import { useToast } from '@/hooks/use-toast'
import { nanoid } from '@/lib/utils'
import { chatAPI } from '@/lib/api'

interface ChatInterfaceProps {
  document: Document | null
  initialMessages?: Message[]
  initialChatId?: string
  className?: string
}

export function ChatInterface({ document, initialMessages = [], initialChatId, className }: ChatInterfaceProps) {
  // State for messages and chat session
  const [messages, setMessages] = useState<Message[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const [chatSession, setChatSession] = useState<ChatSession | null>(null)
  const [context, setContext] = useState<string | null>(null)
  
  // Refs
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()
  
  // Initialize messages on client-side only to avoid hydration mismatch
  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  // Fetch existing chat session if initialChatId is provided
  useEffect(() => {
    if (initialChatId) {
      fetchChatSession(initialChatId);
    }
  }, [initialChatId]);

  // Initialize a new chat session when document changes and no initialChatId
  useEffect(() => {
    if (document && !initialChatId && !chatSession) {
      createNewChatSession();
    }
  }, [document, initialChatId, chatSession]);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      // Find the scrollable viewport element within ScrollArea
      const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    }
  }, [messages])
  
  // Fetch existing chat session
  const fetchChatSession = async (sessionId: string) => {
    console.log("session id:", sessionId);

    try {
      const sessionData = await chatAPI.getChatSession(sessionId);
      const session = convertApiSessionToSession(sessionData);
      setChatSession(session);
      
      if (session.messages && session.messages.length > 0) {
        setMessages(session.messages);
      }
    } catch (error) {
      console.error("Error fetching chat session:", error);
      toast({
        title: "Error loading chat",
        description: "Could not load the chat history. Starting a new chat.",
        variant: "destructive"
      });
      
      if (document) {
        createNewChatSession();
      }
    }
  };
  
  // Create a new chat session
  const createNewChatSession = async () => {
    if (!document) return;
    try {
      const title = `Chat about ${document.title}`;
      const sessionData = await chatAPI.createChatSession(title, document.id);
      const session = convertApiSessionToSession(sessionData);
      setChatSession(session);
      
      // Add system message
      const systemMessage = createMessage({
        id: nanoid(),
        role: MessageRole.System,
        content: `You're now chatting with an AI assistant about "${document.title}". Ask any questions about the document.`,
        timestamp: new Date().toISOString()
      });
      
      setMessages([systemMessage]);
    } catch (error) {
      console.error("Error creating chat session:", error);
      toast({
        title: "Error creating chat",
        description: "Could not create a new chat session. Please try again.",
        variant: "destructive"
      });
    }
  };
  
  
  // Handle sending a message
  const handleSendMessage = async (content: string, attachments: string[] = []) => {
    // Don't allow sending messages if no document or chat session
    if (!document || !chatSession) {
      toast({
        title: "Cannot send message",
        description: "Please select a document first.",
        variant: "destructive"
      });
      return;
    }
    
    // Add user message to UI immediately
    const userMessage: Message = createMessage({
      id: nanoid(),
      role: MessageRole.User,
      content,
      attachments,
      timestamp: new Date().toISOString(),
    });
    
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);
    
    try {
      let aiMessage: Message;
      let responseContext: string | null = null;
      
      if (chatSession.id) {
        // Use the non-streaming API to add the message and get the response
        const response = await chatAPI.addMessageToChat(chatSession.id, content);
        
        // Assuming the response contains the assistant's message and context
        // Adjust based on the actual structure of the response from addMessageToChat
        aiMessage = createMessage({
          id: nanoid(), // Or use an ID from the response if available
          role: MessageRole.Assistant,
          content: response.answer || "Sorry, I couldn't get a response.", // Adjust property names as needed
          timestamp: new Date().toISOString(), // Adjust property names as needed
        });
        
        if (response.context) {
          responseContext = response.context;
        }
        
      } else {
        // Fallback to non-streaming askQuestion if no chat session ID (should ideally not happen if session is created)
        const response = await chatAPI.askQuestion(content, document.id);
        
        aiMessage = createMessage({
          id: nanoid(),
          role: MessageRole.Assistant,
          content: response.answer,
          timestamp: new Date().toISOString(),
        });
        
        if (response.context) {
          responseContext = response.context;
        }
      }
      
      // Add the AI message to the state
      setMessages(prev => [...prev, aiMessage]);
      
      // Update context if available
      if (responseContext) {
        setContext(responseContext);
      }
      
    } catch (error) {
      console.error("Error getting AI response:", error);
      toast({
        title: "Error",
        description: "Failed to get a response. Please try again.",
        variant: "destructive"
      });
      
      // Optionally remove the user message if the API call failed
      // setMessages(prev => prev.filter(msg => msg.id !== userMessage.id));
      
    } finally {
      setIsTyping(false);
    }
  };
  
  // ...existing code...
  
  const handleUsePrompt = (prompt: string) => {
    handleSendMessage(prompt);
  };
  
  const handleAddReaction = (messageId: string, reaction: string) => {
    setMessages(prev => 
      prev.map(msg => {
        if (msg.id === messageId) {
          const existingReactions = msg.reactions || {};
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
    );
    
    toast({
      description: `You reacted with ${reaction}`
    });
  };
  
  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex-1 flex flex-col overflow-auto">
          <div className='p-2 flex-1'>
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-3 sm:p-4">
              <h3 className="text-base sm:text-lg font-semibold mb-2">Chat about this document</h3>
              <p className="text-sm sm:text-base text-muted-foreground mb-4 sm:mb-6">
                Ask questions, get summaries, or request explanations about {document?.title || 'your document'}
              </p>
              {document && <SuggestedPrompts onSelectPrompt={handleUsePrompt} document={document} />}
            </div>
          ) : (
            <MessageList 
              messages={messages} 
              isTyping={isTyping}
              onAddReaction={handleAddReaction}
            />
          )}
          </div>
      
        
        {context && (
          <div className="px-4 py-2 border-t bg-muted/30 text-xs text-muted-foreground">
            <details>
              <summary className="cursor-pointer font-medium">Context from document</summary>
              <div className="mt-2 max-h-32 overflow-y-auto whitespace-pre-wrap">
                {context}
              </div>
            </details>
          </div>
        )}
        
        <div className="p-2 sm:p-4 border-t mt-auto">
          
          <ChatInput 
            onSendMessage={handleSendMessage} 
            isTyping={isTyping}
            disabled={!document || !chatSession}
          />
        </div>
      </div>
    </div>
  )
}