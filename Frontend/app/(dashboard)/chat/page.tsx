"use client"

import { useState } from "react"
import { useDashboard } from "@/lib/context/dashboard-context"
import { useAuth } from "@/lib/context/auth-context"
import { ChatBooksSidebar, BookMaterial } from "@/components/dashboard/chat/chat-books-sidebar"
import { AIMaterialsSidebar } from "@/components/dashboard/chat/ai-materials-sidebar"
import { ChatInterface } from "@/components/dashboard/chat/chat-interface"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, MessageSquare, Sparkles } from "lucide-react"

export default function ChatPage() {
  const { activeExam } = useDashboard()
  const { user } = useAuth()
  const [selectedMaterial, setSelectedMaterial] = useState<BookMaterial | null>(null)
  const [selectedSubjectName, setSelectedSubjectName] = useState<string>("")
  const [leftOpen, setLeftOpen] = useState(true)
  const [rightOpen, setRightOpen] = useState(true)

  const isUnenrolledStudent = user?.role === "student" && !(user?.teacher_ids?.length) && !(user?.teacher_id)

  const handleSelectMaterial = (material: BookMaterial, _collectionName: string, subjectName: string) => {
    setSelectedMaterial(material)
    setSelectedSubjectName(subjectName)
  }

  return (
    <div className="h-full flex overflow-hidden">
      {/* Left: books/sections tree */}
      <div className={cn(
        "flex flex-col border-r bg-background transition-all duration-200 z-20",
        leftOpen ? "w-64" : "w-0 overflow-hidden border-r-0"
      )}>
        {leftOpen && (
          <ChatBooksSidebar
            examId={activeExam?.id}
            examName={activeExam?.name}
            selectedMaterialId={selectedMaterial?.id}
            onSelectMaterial={handleSelectMaterial}
            showSampleHint={isUnenrolledStudent}
          />
        )}
      </div>

      {/* Center: chat */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <div className="h-11 border-b flex items-center px-3 gap-2 shrink-0">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setLeftOpen(!leftOpen)}>
            {leftOpen ? <PanelLeftClose className="h-3.5 w-3.5" /> : <PanelLeftOpen className="h-3.5 w-3.5" />}
          </Button>
          <div className="flex items-center gap-1.5 text-[13px] min-w-0">
            <MessageSquare className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <span className="text-muted-foreground truncate">
              {selectedMaterial ? selectedMaterial.name : "Select a material to chat"}
            </span>
          </div>
          <Button variant="ghost" size="icon" className="ml-auto h-7 w-7" onClick={() => setRightOpen(!rightOpen)} title="AI materials">
            {rightOpen ? <PanelRightClose className="h-3.5 w-3.5" /> : <PanelRightOpen className="h-3.5 w-3.5" />}
          </Button>
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
                processed: selectedMaterial.ragIndexed,
                user_id: "",
                uploadedAt: "",
                tags: [],
                page_count: 0,
                description: undefined,
                vector_db_path: undefined,
                doc_id: selectedMaterial.docId,
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
                Choose a material from your books to start chatting, then generate summaries, flashcards, or a mock test from the right panel.
              </p>
              {(!leftOpen && !selectedMaterial) && (
                <Button size="sm" className="mt-4 rounded-md h-8 text-[13px]" onClick={() => setLeftOpen(true)}>
                  Open Books
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right: AI-generated material */}
      <div className={cn(
        "flex flex-col border-l bg-background transition-all duration-200",
        rightOpen ? "w-72" : "w-0 overflow-hidden border-l-0"
      )}>
        {rightOpen && (
          <AIMaterialsSidebar
            selectedMaterial={selectedMaterial ? { id: selectedMaterial.id, name: selectedMaterial.name, docId: selectedMaterial.docId } : null}
            selectedSubjectName={selectedSubjectName}
            onClose={() => setRightOpen(false)}
          />
        )}
      </div>

      {/* Floating reopen button for right panel when closed */}
      {!rightOpen && (
        <Button
          variant="outline"
          size="icon"
          className="absolute right-3 top-14 z-20 h-7 w-7"
          onClick={() => setRightOpen(true)}
          title="AI materials"
        >
          <Sparkles className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  )
}