"use client"

import { useState } from 'react'
import { 
  Book, FileText, FolderClosed, Plus, Search, Upload, X
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { DocumentUploader } from '@/components/dashboard/documents/document-uploader'
import { cn } from '@/lib/utils'
import { Document } from '@/lib/data'

interface SidebarProps {
  isOpen: boolean
  documents: Document[]
  selectedDocument: Document | null
  onSelectDocument: (doc: Document) => void
  onUploadDocument: (doc: Document) => void
  isMobile?: boolean
  className?: string
}

export function Sidebar({ 
  isOpen, 
  documents, 
  selectedDocument,
  onSelectDocument,
  onUploadDocument,
  isMobile = false,
  className = ""
}: SidebarProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [isUploaderOpen, setIsUploaderOpen] = useState(false)
  const [activeFilter, setActiveFilter] = useState<string | null>(null)
  
  // Filter documents based on search term and active filter
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.title.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = !activeFilter || doc.tags.includes(activeFilter)
    return matchesSearch && matchesFilter
  })
  
  const tags = Array.from(new Set(documents.flatMap(doc => doc.tags)))
  
  const handleUpload = (doc: Document) => {
    onUploadDocument(doc)
    setIsUploaderOpen(false)
  }
  
  if (!isOpen) {
    return !isMobile ? (
      <div className="w-14 border-r h-full bg-card hidden md:flex flex-col items-center py-4">
        <Button variant="ghost" size="icon" className="mb-4">
          <Book className="h-5 w-5 text-primary" />
        </Button>
        <Button variant="ghost" size="icon" onClick={() => setIsUploaderOpen(true)} className="mt-auto">
          <Plus className="h-5 w-5" />
        </Button>
      </div>
    ) : null;
  }
  
  return (
    <>
      <aside 
        className={cn(
          "w-72 border-r h-full bg-card flex flex-col z-30 transition-all duration-300",
          isMobile ? "fixed left-0" : "relative",
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0 md:w-14",
          className
        )}
      >
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Book className="h-5 w-5 text-primary" />
            <span className="font-semibold">Documents</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onSelectDocument(selectedDocument as Document)}
            className={isMobile ? "" : "md:hidden"}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="p-4">
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search documents..."
              className="pl-10"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          
          <Button 
            onClick={() => setIsUploaderOpen(true)} 
            className="w-full mb-4 gap-2"
          >
            <Upload className="h-4 w-4" />
            Upload Document
          </Button>
          
          <div className="flex flex-wrap gap-2 mb-4">
            {tags.map(tag => (
              <Button
                key={tag}
                variant={activeFilter === tag ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveFilter(activeFilter === tag ? null : tag)}
                className={cn(
                  "text-xs",
                  activeFilter === tag && "bg-primary"
                )}
              >
                {tag}
              </Button>
            ))}
          </div>
        </div>
        
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-2">
            {filteredDocuments.length > 0 ? (
              filteredDocuments.map(doc => (
                <button
                  key={doc.id}
                  onClick={() => onSelectDocument(doc)}
                  className={cn(
                    "w-full flex items-start gap-3 p-3 rounded-md text-left transition-colors",
                    selectedDocument?.id === doc.id
                      ? "bg-primary/10 text-primary"
                      : "hover:bg-muted"
                  )}
                >
                  <FileText className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <div className="overflow-hidden">
                    <p className="font-medium truncate">{doc.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(doc.uploadedAt).toLocaleDateString()}
                    </p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {doc.tags.map(tag => (
                        <span
                          key={tag}
                          className="inline-block text-[10px] px-1.5 py-0.5 bg-muted text-muted-foreground rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              ))
            ) : (
              <div className="text-center py-6 text-muted-foreground">
                <FolderClosed className="h-10 w-10 mx-auto mb-2 opacity-50" />
                <p>No documents found</p>
                <p className="text-sm">Try adjusting your search or filters</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </aside>
      
      {isUploaderOpen && (
        <DocumentUploader 
          onUpload={handleUpload} 
          onCancel={() => setIsUploaderOpen(false)}
        />
      )}
    </>
  )
}