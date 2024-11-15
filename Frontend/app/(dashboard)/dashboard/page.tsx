"use client"

import { useState, useEffect } from 'react'
import { Sidebar } from '@/components/dashboard/sidebar'
import { ChatInterface } from '@/components/dashboard/chat/chat-interface'
import { ChatHistoryViewer } from '@/components/dashboard/chat/chat-history-viewer'
import { SettingsPanel } from '@/components/dashboard/settings/settings-panel'
import { DocumentViewer } from '@/components/dashboard/documents/document-viewer'
import { EmptyState } from '@/components/dashboard/empty-state'
import { DashboardHeader } from '@/components/dashboard/header'
import { DEFAULT_DOCUMENTS, Document, DEFAULT_CHAT_HISTORY, ChatHistory } from '@/lib/data'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'

export default function DashboardPage() {
  // Initialize with empty arrays and populate in useEffect to avoid hydration mismatches
  const [documents, setDocuments] = useState<Document[]>([])
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [selectedChat, setSelectedChat] = useState<ChatHistory | null>(null)
  // Don't set initial state for UI elements that might differ between server/client
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [activeView, setActiveView] = useState<'document' | 'chat'>('document')
  const [activeTab, setActiveTab] = useState<'documents' | 'chats'>('documents')
  const { toast } = useToast()

  // Initialize data on client-side only
  useEffect(() => {
    setDocuments(DEFAULT_DOCUMENTS);
    setChatHistory(DEFAULT_CHAT_HISTORY);
    setIsSidebarOpen(!isMobile);
  }, [isMobile]);

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
  
  const handleSelectChat = (chat: ChatHistory) => {
    setSelectedChat(chat)
    setSelectedDocument(null)
    setActiveView('chat')
    // Auto-close sidebar on mobile after selection
    if (isMobile) {
      setIsSidebarOpen(false)
    }
  }
  
  const handleDeleteChat = (chatId: string) => {
    setChatHistory(prev => prev.filter(chat => chat.id !== chatId))
    
    if (selectedChat?.id === chatId) {
      setSelectedChat(null)
    }
    
    toast({
      title: "Chat deleted",
      description: "The chat has been removed from your history.",
    })
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
                    document={selectedDocument || { 
                      id: selectedChat?.documentId || '',
                      title: selectedChat?.documentTitle || '',
                      fileName: '',
                      url: '',
                      type: 'pdf',
                      uploadedAt: '',
                      tags: []
                    }}
                    initialMessages={selectedChat?.messages || []}
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