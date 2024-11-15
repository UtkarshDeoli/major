"use client"

import { useState, useRef } from 'react'
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import { File, X, Upload } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { Document } from '@/lib/data'
import { nanoid } from '@/lib/utils'

interface DocumentUploaderProps {
  onUpload: (doc: Document) => void
  onCancel: () => void
}

export function DocumentUploader({ onUpload, onCancel }: DocumentUploaderProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [fileName, setFileName] = useState('')
  const [title, setTitle] = useState('')
  const [tags, setTags] = useState<string[]>([])
  const [currentTag, setCurrentTag] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFileName(file.name)
      setTitle(file.name.replace(/\.[^/.]+$/, ""))
    }
  }
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setFileName(file.name)
      setTitle(file.name.replace(/\.[^/.]+$/, ""))
    }
  }
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }
  
  const handleAddTag = () => {
    if (currentTag.trim() && !tags.includes(currentTag.trim())) {
      setTags([...tags, currentTag.trim()])
      setCurrentTag('')
    }
  }
  
  const handleRemoveTag = (tagToRemove: string) => {
    setTags(tags.filter(tag => tag !== tagToRemove))
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddTag()
    }
  }
  
  const handleUpload = () => {
    if (!fileName) {
      toast({
        title: "No file selected",
        description: "Please select a PDF file to upload.",
        variant: "destructive"
      })
      return
    }
    
    if (!title.trim()) {
      toast({
        title: "Title required",
        description: "Please provide a title for your document.",
        variant: "destructive"
      })
      return
    }
    
    setIsUploading(true)
    
    // Simulate upload process
    setTimeout(() => {
      setIsUploading(false)
      
      const newDocument: Document = {
        id: nanoid(),
        title: title.trim(),
        fileName: fileName,
        uploadedAt: new Date().toISOString(),
        tags: tags,
        url: `/documents/${fileName}`,
        type: 'pdf'
      }
      
      toast({
        title: "Document uploaded",
        description: `${title} has been successfully uploaded.`
      })
      
      onUpload(newDocument)
    }, 1500)
  }
  
  return (
    <Dialog open={true} onOpenChange={onCancel}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload Document</DialogTitle>
          <DialogDescription>
            Upload a PDF document to analyze with AI
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <div 
            className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-muted/50 transition-colors"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {fileName ? (
              <div className="flex items-center justify-center gap-2">
                <File className="h-8 w-8 text-primary" />
                <div className="text-left">
                  <p className="font-medium">{fileName}</p>
                  <p className="text-xs text-muted-foreground">
                    Click or drag to replace
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="h-10 w-10 text-muted-foreground mx-auto" />
                <p className="text-sm font-medium">
                  Click to select or drag and drop
                </p>
                <p className="text-xs text-muted-foreground">
                  Support for PDF, DOC, DOCX (max 10MB)
                </p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.doc,.docx"
              onChange={handleFileChange}
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="title">Document Title</Label>
            <Input
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter document title"
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="tags">Tags</Label>
            <div className="flex gap-2">
              <Input
                id="tags"
                value={currentTag}
                onChange={(e) => setCurrentTag(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Add tags and press Enter"
              />
              <Button 
                type="button" 
                size="sm"
                variant="secondary"
                onClick={handleAddTag}
              >
                Add
              </Button>
            </div>
          </div>
          
          {tags.length > 0 && (
            <ScrollArea className="max-h-24">
              <div className="flex flex-wrap gap-2 mt-2">
                {tags.map(tag => (
                  <div 
                    key={tag} 
                    className="flex items-center bg-secondary text-secondary-foreground rounded-full px-2 py-1 text-xs"
                  >
                    <span>{tag}</span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-4 w-4 ml-1 text-muted-foreground hover:text-foreground"
                      onClick={() => handleRemoveTag(tag)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </div>
        
        <div className="flex gap-2 justify-end">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button 
            onClick={handleUpload} 
            disabled={isUploading || !fileName}
          >
            {isUploading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Uploading...
              </span>
            ) : (
              "Upload"
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}