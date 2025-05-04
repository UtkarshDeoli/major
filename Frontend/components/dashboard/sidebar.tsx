"use client"

import { useState, useEffect } from 'react'
import { 
  Book, FileText, FolderClosed, Plus, Search, Upload, X
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { DocumentUploader } from '@/components/dashboard/documents/document-uploader'
import { cn } from '@/lib/utils'
import { Document, convertApiDocumentToDocument } from '@/lib/data'
import { pdfAPI } from '@/lib/api'
import { useToast } from '@/hooks/use-toast'

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
  const [activeFilters, setActiveFilters] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [userDocuments, setUserDocuments] = useState<Document[]>(documents)
  const { toast } = useToast()
  
  // Fetch documents from API on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);
  
  // Fetch documents from the API
  const fetchDocuments = async () => {
    try {
      setIsLoading(true);
      const apiDocuments = await pdfAPI.listPDFs();
      
      // Convert API documents to our Document format
      const convertedDocs = apiDocuments.map((doc: any) => {
        // Handle tags - they might be a single string containing a JSON array
        let docTags: string[] = [];
        
        if (doc.tags && Array.isArray(doc.tags)) {
          if (doc.tags.length > 0) {
            try {
              // If the tags are stored as a JSON string inside an array
              if (typeof doc.tags[0] === 'string' && doc.tags[0].startsWith('[')) {
                // Parse the JSON string to get the actual array of tags
                const parsedTags = JSON.parse(doc.tags[0]);
                if (Array.isArray(parsedTags)) {
                  docTags = parsedTags;
                }
              } else {
                // Normal array of string tags
                docTags = doc.tags;
              }
            } catch (error) {
              console.error("Error parsing tags:", error);
              docTags = doc.tags; // Use as is if parsing fails
            }
          }
        } else if (!doc.tags) {
          // Default tag if none exists
          docTags = ['document'];
        }
        
        return convertApiDocumentToDocument({
          ...doc,
          tags: docTags
        });
      });
      
      setUserDocuments(convertedDocs);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching documents:", error);
      toast({
        title: "Failed to load documents",
        description: "Could not retrieve your documents. Using default documents instead.",
        variant: "destructive"
      });
      setIsLoading(false);
    }
  };
  
  // Filter documents based on search term and active filters
  const filteredDocuments = userDocuments.filter(doc => {
    const matchesSearch = doc.title.toLowerCase().includes(searchTerm.toLowerCase())
    
    // If no filters are active, show all documents that match the search
    if (activeFilters.length === 0) {
      return matchesSearch;
    }
    
    // Check if document has tags and if any of those tags are in activeFilters
    const matchesFilters = doc.tags && doc.tags.some(tag => activeFilters.includes(tag));
    
    return matchesSearch && matchesFilters;
  })
  
  // Extract unique tags from all documents
  const tags = Array.from(
    new Set(
      userDocuments.flatMap(doc => doc.tags || [])
    )
  );
  
  const handleUpload = (doc: Document) => {
    // Add the new document to the local state
    setUserDocuments(prev => [...prev, doc]);
    onUploadDocument(doc);
    setIsUploaderOpen(false);
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
          
          {tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {tags.map(tag => (
                <Button
                  key={tag}
                  variant={activeFilters.includes(tag) ? "default" : "outline"}
                  size="sm"
                  onClick={() => setActiveFilters(prev => 
                    prev.includes(tag) 
                      ? prev.filter(filter => filter !== tag) 
                      : [...prev, tag]
                  )}
                  className={cn(
                    "text-xs",
                    activeFilters.includes(tag) && "bg-primary"
                  )}
                >
                  {tag}
                </Button>
              ))}
            </div>
          )}
        </div>
        
        <ScrollArea className="flex-1">
          <div className="p-4 space-y-2">
            {isLoading ? (
              <div className="flex justify-center py-8">
                <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              </div>
            ) : filteredDocuments.length > 0 ? (
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
                    {doc.tags && doc.tags.length > 0 && (
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
                    )}
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