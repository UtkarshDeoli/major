"use client";

import { useState } from "react";
import { useDashboard } from "@/lib/context/dashboard-context";
import { CollectionsChatSidebar } from "@/components/dashboard/chat/collections-chat-sidebar";
import { ChatInterface } from "@/components/dashboard/chat/chat-interface";
import { Material, Collection, Subject } from "@/lib/context/dashboard-context";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Menu, X, MessageSquare } from "lucide-react";

export default function ChatPage() {
  const { activeExam } = useDashboard();
  const [selectedMaterial, setSelectedMaterial] = useState<Material | null>(null);
  const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const handleSelectMaterial = (
    material: Material,
    collection: Collection,
    _subject: Subject
  ) => {
    setSelectedMaterial(material);
    setSelectedCollection(collection);
  };

  return (
    <div className="h-full flex overflow-hidden bg-background">
      {/* Sidebar */}
      <div
        className={cn(
          "flex flex-col border-r bg-card/50 backdrop-blur-xl transition-all duration-300 ease-in-out z-20",
          sidebarOpen ? "w-72" : "w-0 overflow-hidden border-r-0"
        )}
      >
        {sidebarOpen && (
          <CollectionsChatSidebar
            exam={activeExam}
            selectedMaterial={selectedMaterial}
            onSelectMaterial={handleSelectMaterial}
          />
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Top bar */}
        <div className="h-14 border-b bg-card/50 backdrop-blur-sm flex items-center px-4 gap-3 shrink-0">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </Button>

          <div className="flex items-center gap-2 text-sm">
            <MessageSquare className="h-4 w-4 text-primary" />
            <span className="font-medium">
              {selectedMaterial ? selectedMaterial.name : "Select a material to chat"}
            </span>
          </div>
        </div>

        {/* Chat content */}
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
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                <MessageSquare className="h-8 w-8 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Select a Study Material</h3>
              <p className="text-sm text-muted-foreground max-w-sm">
                Open the sidebar and choose a material from your collections to start
                chatting with AI.
              </p>
              {!sidebarOpen && (
                <Button
                  className="mt-4"
                  onClick={() => setSidebarOpen(true)}
                >
                  Open Sidebar
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
