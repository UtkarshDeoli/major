"use client"

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/dashboard/sidebar'
import { ChatInterface } from '@/components/dashboard/chat/chat-interface'
import { ChatHistoryViewer } from '@/components/dashboard/chat/chat-history-viewer'
import { SettingsPanel } from '@/components/dashboard/settings/settings-panel'
import { DocumentViewer } from '@/components/dashboard/documents/document-viewer'
import { EmptyState } from '@/components/dashboard/empty-state'
import { DashboardHeader } from '@/components/dashboard/header'
import { Document, ChatSession, DEFAULT_DOCUMENTS, DEFAULT_CHAT_HISTORY } from '@/lib/data'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'
import { Button } from '@/components/ui/button'
import { useRouter } from 'next/navigation'
import { TestTube } from 'lucide-react'
import { pdfAPI, chatAPI, authAPI } from '@/lib/api'

export default function DashboardPage() {
  // Initialize with empty arrays and populate in useEffect to avoid hydration mismatches
  const [documents, setDocuments] = useState<Document[]>([])
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([])
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null)
  // Don't set initial state for UI elements that might differ between server/client
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [activeView, setActiveView] = useState<'document' | 'chat'>('document')
  const [activeTab, setActiveTab] = useState<'documents' | 'chats'>('documents')
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const { toast } = useToast()
  const router = useRouter()

  // Check authentication on initial load
  useEffect(() => {
    const checkAuth = () => {
      const isAuth = authAPI.isAuthenticated();
      setIsAuthenticated(isAuth);
      
      if (!isAuth) {
        // Redirect to login page if not authenticated
        router.push('/login');
      } else {
        // Load data if authenticated
        initializeData();
      }
    };
    
    checkAuth();
  }, [router]);

  // Initialize data when authenticated
  const initializeData = async () => {
    setIsLoading(true);
    
    try {
      // Fetch documents from API
      const apiDocuments = await pdfAPI.listPDFs();
      if (apiDocuments && apiDocuments.length > 0) {
        setDocuments(apiDocuments);
      } else {
        // Fall back to default documents if API returns empty
        setDocuments(DEFAULT_DOCUMENTS);
      }
      
      // Fetch chat history from API
      const apiSessions = await chatAPI.listChatSessions();
      if (apiSessions && apiSessions.length > 0) {
        setChatHistory(apiSessions);
      } else {
        // Fall back to default chat history if API returns empty
        setChatHistory(DEFAULT_CHAT_HISTORY);
      }
    } catch (error) {
      console.error("Error initializing data:", error);
      // Fall back to default data on error
      setDocuments(DEFAULT_DOCUMENTS);
      setChatHistory(DEFAULT_CHAT_HISTORY);
      
      toast({
        title: "Data load error",
        description: "Could not load your documents and chats. Using example data instead.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
      setIsSidebarOpen(!isMobile);
    }
  };

  // Detect mobile view
  useEffect(() => {
    const checkIsMobile = () => {
      const mobile = window.innerWidth < 768
      setIsMobile(mobile)
      // Auto-close sidebar on mobile
      if (mobile) {
        setIsSidebarOpen(false)
      } else {
        setIsSidebarOpen(true)
      }
    }
    
    checkIsMobile()
    window.addEventListener('resize', checkIsMobile)
    
    return () => window.removeEventListener('resize', checkIsMobile)
  }, [])

  const handleSelectDocument = (doc: Document) => {
    setSelectedDocument(doc)
    setSelectedChat(null)
    setActiveView('document')
    // Auto-close sidebar on mobile after selection
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }

  const handleUploadDocument = (doc: Document) => {
    setDocuments(prev => [...prev, doc])
    setSelectedDocument(doc)
    setSelectedChat(null)
    setActiveView('document')
    // Auto-close sidebar on mobile after upload
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }
  
  const handleSelectChat = (chat: ChatSession) => {
    setSelectedChat(chat)
    
    // If this chat is associated with a document, also select that document
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
    
    // Auto-close sidebar on mobile after selection
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }
  
  const handleDeleteChat = async (chatId: string) => {
    try {
      // In the future, we would call an API to delete the chat
      // await chatAPI.deleteSession(chatId);
      
      // For now, just update the UI
      setChatHistory(prev => prev.filter(chat => chat.id !== chatId))
      
      if (selectedChat?.id === chatId) {
        setSelectedChat(null)
      }
      
      toast({
        title: "Chat deleted",
        description: "The chat has been removed from your history.",
      })
    } catch (error) {
      console.error("Error deleting chat:", error);
      toast({
        title: "Delete failed",
        description: "Could not delete the chat. Please try again.",
        variant: "destructive"
      });
    }
  }

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="flex flex-col items-center">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent mb-4" />
          <p className="text-muted-foreground">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      <DashboardHeader 
        onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        isMobile={isMobile}
      />
      <div className="flex-1 flex overflow-hidden">
        {/* Overlay backdrop for mobile sidebar */}
        {isMobile && isSidebarOpen && (
          <div 
            className="fixed inset-0 bg-black/50 z-40"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
        
        {/* Sidebar with tabs for documents and chat history */}
        <div className={`${isSidebarOpen ? 'block' : 'hidden'} ${isMobile ? "absolute z-50 h-[calc(100%-4rem)] top-16 w-72" : "relative w-72"}`}>
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
              {/* Mobile view switcher - only for document view */}
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
                {/* Show document viewer based on mobile selection or always on desktop, and only when a document is selected */}
                {selectedDocument && (!isMobile || activeView === 'document') && (
                  <DocumentViewer 
                    document={selectedDocument} 
                    className={`${isMobile ? "flex-1" : "md:w-1/2 lg:w-3/5"}`}
                  />
                )}
                
                {/* Show chat interface based on mobile selection or always on desktop */}
                {(!isMobile || activeView === 'chat') && (
                  <ChatInterface 
                    document={selectedDocument}
                    initialMessages={selectedChat?.messages || []}
                    initialChatId={selectedChat?.id}
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
      
      {isSettingsOpen && (
        <SettingsPanel 
          onClose={() => setIsSettingsOpen(false)}
          isMobile={isMobile} 
        />
      )}
    </div>
  )
}