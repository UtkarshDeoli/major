"use client"

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Upload, Book, FileText, MessageSquare } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Document } from '@/lib/data'
import { DocumentUploader } from '@/components/dashboard/documents/document-uploader'

interface EmptyStateProps {
  onUploadDocument: (doc: Document) => void
}

export function EmptyState({ onUploadDocument }: EmptyStateProps) {
  const [isUploaderOpen, setIsUploaderOpen] = useState(false)
  
  const handleUpload = (doc: Document) => {
    onUploadDocument(doc)
    setIsUploaderOpen(false)
  }

  return (
    <div className="h-full flex flex-col items-center justify-center p-6 text-center">
      <div className="max-w-md mx-auto">
        <div className="mb-6 relative">
          <div className="absolute inset-0 bg-primary/10 rounded-full blur-xl opacity-50" />
          <div className="relative flex items-center justify-center w-20 h-20 mx-auto bg-primary/20 rounded-full">
            <Book className="h-10 w-10 text-primary" />
          </div>
        </div>
        
        <h2 className="text-2xl font-bold mb-2">Welcome to Padhai Whallah</h2>
        <p className="text-muted-foreground mb-8">
          Upload your first document to start studying with AI assistance
        </p>
        
        <Button 
          size="lg" 
          onClick={() => setIsUploaderOpen(true)} 
          className="gap-2 mx-auto"
        >
          <Upload className="h-4 w-4" />
          Upload Document
        </Button>
        
        {/* Features grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-12">
          <div className="p-4 border rounded-lg bg-card">
            <FileText className="h-8 w-8 text-primary/70 mb-2 mx-auto" />
            <h3 className="font-medium mb-1">Document Analysis</h3>
            <p className="text-sm text-muted-foreground">
              Extract key information from your study materials
            </p>
          </div>
          
          <div className="p-4 border rounded-lg bg-card">
            <MessageSquare className="h-8 w-8 text-primary/70 mb-2 mx-auto" />
            <h3 className="font-medium mb-1">AI Conversations</h3>
            <p className="text-sm text-muted-foreground">
              Chat about concepts and get detailed explanations
            </p>
          </div>
        </div>
      </div>
      
      {isUploaderOpen && (
        <DocumentUploader 
          onUpload={handleUpload} 
          onCancel={() => setIsUploaderOpen(false)} 
        />
      )}
    </div>
  )
}