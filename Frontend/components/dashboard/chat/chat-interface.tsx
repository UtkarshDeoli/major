"use client"

import { useState, useRef, useEffect } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { ChatInput } from '@/components/dashboard/chat/chat-input'
import { MessageList } from '@/components/dashboard/chat/message-list'
import { SuggestedPrompts } from '@/components/dashboard/chat/suggested-prompts'
import { SocraticExplainer } from '@/components/socratic/socratic-explainer'
import { createMessage, MessageRole, Message, Document, ChatSession, convertApiSessionToSession } from '@/lib/data'
import { useToast } from '@/hooks/use-toast'
import { nanoid } from '@/lib/utils'
import { chatAPI } from '@/lib/api'

interface ChatInterfaceProps {
  document: Document | null
  initialMessages?: Message[]
  initialChatId?: string
  onChatSessionUpdate?: (chatSession: ChatSession) => void
  className?: string
}

export function ChatInterface({ document, initialMessages = [], initialChatId, onChatSessionUpdate, className }: ChatInterfaceProps) {
  // State for messages and chat session
  const [messages, setMessages] = useState<Message[]>([])
  const [isTyping, setIsTyping] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
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

  // Initialize a new chat session whenever the selected document changes.
  // This avoids reusing a session scoped to a previously selected document.
  useEffect(() => {
    if (!document || initialChatId) return;

    setChatSession(null);
    setMessages([]);
    setStreamingMessage(null);

    let cancelled = false;
    const setup = async () => {
      try {
        const title = `Chat about ${document.title}`;
        const docIds = document.doc_id ? [document.doc_id] : undefined;
        const sessionData = await chatAPI.createChatSession(title, document.id, docIds);
        if (cancelled) return;
        const session = convertApiSessionToSession(sessionData);
        setChatSession(session);

        const systemMessage = createMessage({
          id: nanoid(),
          role: MessageRole.System,
          content: `You're now chatting with an AI assistant about "${document.title}". Ask any questions about the document.`,
          timestamp: new Date().toISOString(),
        });
        setMessages([systemMessage]);
      } catch (error) {
        if (cancelled) return;
        console.error("Error creating chat session:", error);
        toast({
          title: "Error creating chat",
          description: "Could not create a new chat session. Please try again.",
          variant: "destructive",
        });
      }
    };

    setup();
    return () => {
      cancelled = true;
    };
  }, [document?.id, initialChatId]);
  
  // Scroll to bottom when messages change or streaming updates
  useEffect(() => {
    if (scrollAreaRef.current) {
      // Find the scrollable viewport element within ScrollArea
      const viewport = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    }
  }, [messages, streamingMessage])
  
  // Fetch existing chat session
  const fetchChatSession = async (sessionId: string) => {
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
        variant: "destructive",
      });

      if (document) {
        // Let the document effect create a fresh session on the next tick.
        setChatSession(null);
      }
    }
  };
  
  
  // Handle sending a message
  const handleSendMessage = async (content: string, attachments: string[] = [], imageDataUrl?: string) => {
    // Don't allow sending messages if no document or chat session
    if (!document || !chatSession) {
      toast({
        title: "Cannot send message",
        description: "Please select a document first.",
        variant: "destructive"
      });
      return;
    }

    // Check if this is the first user message (excluding system messages)
    const isFirstUserMessage = messages.filter(msg => msg.role === MessageRole.User).length === 0;
    
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
      if (chatSession.id) {
        // Use the streaming API to add the message and get the response
        setIsStreaming(true)
        setStreamingMessage("")
        await chatAPI.addMessageToChatStream(chatSession.id, content, (chunk: any) => {
          if (chunk.error) {
            console.error("Streaming error chunk:", chunk.error);
            return;
          }
          if (chunk.content && !chunk.done) {
            setStreamingMessage((prev) => (prev || "") + chunk.content);
          }
        }, imageDataUrl)
        // After stream completes, refresh messages from the session
        await fetchChatSession(chatSession.id)
        setStreamingMessage(null)

        // If this was the first user message, add the chat session to history
        if (isFirstUserMessage && onChatSessionUpdate && chatSession) {
          const updatedChatSession = {
            ...chatSession,
            messages: [...messages, userMessage],
            updated_at: new Date().toISOString()
          };
          onChatSessionUpdate(updatedChatSession);
        }
      } else {
        // Fallback to non-streaming askQuestion if no chat session ID
        const response = await chatAPI.askQuestion(content, document.id, imageDataUrl);
        const aiMessage = createMessage({
          id: nanoid(),
          role: MessageRole.Assistant,
          content: response.answer,
          timestamp: new Date().toISOString(),
        });
        setMessages(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error("Error getting AI response:", error);
      toast({
        title: "Error",
        description: "Failed to get a response. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsTyping(false);
      setIsStreaming(false);
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
              messages={
                streamingMessage
                  ? [
                      ...messages,
                      createMessage({
                        id: 'streaming',
                        role: MessageRole.Assistant,
                        content: streamingMessage,
                        timestamp: new Date().toISOString(),
                      }),
                    ]
                  : messages
              }
              isTyping={isTyping && !isStreaming}
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
        
        <div className="p-2 sm:p-4 border-t mt-auto space-y-2">
          {document && chatSession && messages.length > 0 && (
            <SocraticExplainer
              question={messages.filter((m) => m.role === MessageRole.User).slice(-1)[0]?.content || ""}
              docIds={document.doc_id ? [document.doc_id] : undefined}
            />
          )}

          <ChatInput
            onSendMessage={handleSendMessage}
            isTyping={isTyping}
            disabled={!document || !chatSession || isStreaming}
          />
        </div>
      </div>
    </div>
  )
}