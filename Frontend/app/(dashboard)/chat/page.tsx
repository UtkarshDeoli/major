"use client"

import { useState } from "react"
import { useDashboard } from "@/lib/context/dashboard-context"
import { CollectionsChatSidebar } from "@/components/dashboard/chat/collections-chat-sidebar"
import { ChatInterface } from "@/components/dashboard/chat/chat-interface"
import { Material, Collection, Subject } from "@/lib/context/dashboard-context"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { PanelLeftClose, PanelLeftOpen, MessageSquare } from "lucide-react"

export default function ChatPage() {
  const { activeExam } = useDashboard()
  const [selectedMaterial, setSelectedMaterial] = useState<Material | null>(null)
  const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const handleSelectMaterial = (material: Material, collection: Collection, _subject: Subject) => {
    setSelectedMaterial(material)
    setSelectedCollection(collection)
  }

  return (
    <div className="h-full flex overflow-hidden">
      <div className={cn(
        "flex flex-col border-r bg-background transition-all duration-200 z-20",
        sidebarOpen ? "w-60" : "w-0 overflow-hidden border-r-0"
      )}>
        {sidebarOpen && (
          <CollectionsChatSidebar
            exam={activeExam}
            selectedMaterial={selectedMaterial}
            onSelectMaterial={handleSelectMaterial}
          />
        )}
      </div>

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <div className="h-11 border-b flex items-center px-3 gap-2 shrink-0">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <PanelLeftClose className="h-3.5 w-3.5" /> : <PanelLeftOpen className="h-3.5 w-3.5" />}
          </Button>
          <div className="flex items-center gap-1.5 text-[13px]">
            <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-muted-foreground">
              {selectedMaterial ? selectedMaterial.name : "Select a material to chat"}
            </span>
          </div>
        </div>

        <div className="flex-1 overflow-hidden">
          {selectedMaterial ? (
            <ChatInterface
              document={{
                id: selectedMaterial.id,
                title: selectedMaterial.name,
                file_path: selectedMaterial.url,
                filename: selectedMaterial.name,
                size: selectedMaterial.size,
                processed: true,
                user_id: "",
                uploadedAt: selectedMaterial.uploadedAt,
                tags: [],
                page_count: 0,
                description: undefined,
                vector_db_path: undefined,
              }}
              className="h-full"
            />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-6">
              <div className="w-12 h-12 rounded-md bg-secondary flex items-center justify-center mb-4">
                <MessageSquare className="h-6 w-6 text-muted-foreground" />
              </div>
              <h3 className="text-sm font-medium mb-1">Select a Study Material</h3>
              <p className="text-xs text-muted-foreground max-w-xs">
                Choose a material from the sidebar to start chatting with AI.
              </p>
              {!sidebarOpen && (
                <Button size="sm" className="mt-4 rounded-md h-8 text-[13px]" onClick={() => setSidebarOpen(true)}>
                  Open Sidebar
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}