"use client"

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/dashboard/sidebar'
import { ChatInterface } from '@/components/dashboard/chat/chat-interface'
import { ChatHistoryViewer } from '@/components/dashboard/chat/chat-history-viewer'
import { DocumentViewer } from '@/components/dashboard/documents/document-viewer'
import { EmptyState } from '@/components/dashboard/empty-state'
import { Document, ChatSession, DEFAULT_DOCUMENTS, DEFAULT_CHAT_HISTORY } from '@/lib/data'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'
import { pdfAPI, chatAPI } from '@/lib/api'

export default function ChatPage() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([])
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null)
  const [isMobile, setIsMobile] = useState(false)
  const [activeView, setActiveView] = useState<'document' | 'chat'>('document')
  const [activeTab, setActiveTab] = useState<'documents' | 'chats'>('documents')
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    let mounted = true
    const initializeData = async () => {
      if (!mounted) return
      setIsLoading(true)
      try {
        const [apiDocuments, apiSessions] = await Promise.all([
          pdfAPI.listPDFs().catch(() => []),
          chatAPI.listChatSessions().catch(() => []),
        ])
        if (!mounted) return
        setDocuments(apiDocuments && apiDocuments.length > 0 ? apiDocuments : DEFAULT_DOCUMENTS)
        setChatHistory(apiSessions && apiSessions.length > 0 ? apiSessions : DEFAULT_CHAT_HISTORY)
      } catch (error) {
        if (!mounted) return
        console.error("Error initializing data:", error)
        setDocuments(DEFAULT_DOCUMENTS)
        setChatHistory(DEFAULT_CHAT_HISTORY)
        toast({
          title: "Data load error",
          description: "Could not load your documents and chats. Using example data instead.",
          variant: "destructive"
        })
      } finally {
        if (mounted) setIsLoading(false)
      }
    }
    initializeData()
    return () => { mounted = false }
  }, [toast])

  useEffect(() => {
    const checkIsMobile = () => {
      const mobile = window.innerWidth < 768
      setIsMobile(mobile)
    }
    checkIsMobile()
    window.addEventListener('resize', checkIsMobile)
    return () => window.removeEventListener('resize', checkIsMobile)
  }, [])

  const handleSelectDocument = async (doc: Document) => {
    try {
      const newChatSession = await chatAPI.createChatSession(doc.title, doc.id);
      setSelectedDocument(doc);
      setSelectedChat(newChatSession);
      setActiveView('document');
      toast({
        title: "New chat session created",
        description: `Started a new chat for ${doc.title}`,
      });
    } catch (error) {
      console.error("Error creating new chat session:", error);
      setSelectedDocument(doc);
      setSelectedChat(null);
      setActiveView('document');
      toast({
        title: "Document selected",
        description: "Could not create new chat session, but document is loaded.",
        variant: "destructive"
      });
    }
  }

  const handleUploadDocument = async (doc: Document) => {
    try {
      const newChatSession = await chatAPI.createChatSession(doc.title, doc.id);
      setDocuments(prev => [...prev, doc]);
      setSelectedDocument(doc);
      setSelectedChat(newChatSession);
      setActiveView('document');
      toast({
        title: "Document uploaded & chat created",
        description: `Started a new chat for ${doc.title}`,
      });
    } catch (error) {
      console.error("Error creating chat session for uploaded document:", error);
      setDocuments(prev => [...prev, doc]);
      setSelectedDocument(doc);
      setSelectedChat(null);
      setActiveView('document');
      toast({
        title: "Document uploaded",
        description: "Could not create chat session, but document is available.",
        variant: "destructive"
      });
    }
  }

  const handleSelectChat = (chat: ChatSession) => {
    setSelectedChat(chat)
    if (chat.pdf_id) {
      const associatedDoc = documents.find(doc => doc.id === chat.pdf_id);
      if (associatedDoc) {
        setSelectedDocument(associatedDoc);
      } else {
        setSelectedDocument(null);
      }
    } else {
      setSelectedDocument(null);
    }
    setActiveView('chat')
  }

  const handleDeleteChat = async (chatId: string) => {
    setChatHistory(prev => prev.filter(chat => chat.id !== chatId))
    if (selectedChat?.id === chatId) {
      setSelectedChat(null)
    }
    toast({
      title: "Chat deleted",
      description: "The chat has been removed from your history.",
    })
  }

  const handleChatSessionUpdate = (chatSession: ChatSession) => {
    setChatHistory(prev => {
      const exists = prev.some(session => session.id === chatSession.id);
      if (!exists) {
        return [chatSession, ...prev];
      }
      return prev;
    });
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="flex flex-col items-center">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent mb-4" />
          <p className="text-muted-foreground">Loading your workspace...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex overflow-hidden">
      <div className={`${isMobile ? "absolute z-50 h-full w-72" : "relative w-72"} flex flex-col border-r bg-card`}>
        <Tabs 
          value={activeTab} 
          onValueChange={(value) => setActiveTab(value as 'documents' | 'chats')}
          className="h-full flex flex-col"
        >
          <TabsList className="grid grid-cols-2 w-full rounded-none border-b bg-transparent">
            <TabsTrigger value="documents">Documents</TabsTrigger>
            <TabsTrigger value="chats">Chat History</TabsTrigger>
          </TabsList>
          
          <TabsContent value="documents" className="flex-1 p-0 m-0 h-[calc(100%-40px)]">
            <Sidebar 
              isOpen={true} 
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={handleSelectDocument}
              onUploadDocument={handleUploadDocument}
              isMobile={isMobile}
              className="border-none h-full"
            />
          </TabsContent>
          
          <TabsContent value="chats" className="flex-1 p-0 m-0 h-[calc(100%-40px)]">
            <ChatHistoryViewer 
              chatHistory={chatHistory}
              onSelectChat={handleSelectChat}
              onDeleteChat={handleDeleteChat}
              className="border-none h-full"
            />
          </TabsContent>
        </Tabs>
      </div>
      
      <main className="flex-1 flex flex-col overflow-hidden">
        {(selectedDocument || selectedChat) ? (
          <>
            {isMobile && selectedDocument && (
              <div className="flex border-b">
                <button 
                  className={`flex-1 py-3 text-center font-medium transition-colors ${activeView === 'document' ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50'}`}
                  onClick={() => setActiveView('document')}
                >
                  Document
                </button>
                <button 
                  className={`flex-1 py-3 text-center font-medium transition-colors ${activeView === 'chat' ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50'}`}
                  onClick={() => setActiveView('chat')}
                >
                  Chat
                </button>
              </div>
            )}
            
            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
              {selectedDocument && (!isMobile || activeView === 'document') && (
                <DocumentViewer 
                  document={selectedDocument} 
                  className={`${isMobile ? "flex-1" : "md:w-1/2 lg:w-3/5"}`}
                />
              )}
              
              {(!isMobile || activeView === 'chat') && (
                <ChatInterface 
                  document={selectedDocument}
                  initialMessages={selectedChat?.messages || []}
                  initialChatId={selectedChat?.id}
                  onChatSessionUpdate={handleChatSessionUpdate}
                  className={`${isMobile ? "flex-1" : "md:w-1/2 lg:w-2/5"}`}
                />
              )}
            </div>
          </>
        ) : (
          <EmptyState onUploadDocument={handleUploadDocument} />
        )}
      </main>
    </div>
  )
}
