"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Document } from '@/lib/data'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Download, Printer } from 'lucide-react'
import { cn } from '@/lib/utils'
import { pdfAPI } from '@/lib/api'
import { useToast } from '@/hooks/use-toast'

interface DocumentViewerProps {
  document: Document
  className?: string
}

export function DocumentViewer({ document, className }: DocumentViewerProps) {
  const [zoom, setZoom] = useState(100)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(document.page_count || 1)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()
  
  // Load PDF when document changes
  useEffect(() => {
    if (document?.id) {
      loadPdf();
    }
  }, [document?.id]);
  
  const loadPdf = async () => {
    try {
      setIsLoading(true);
      
      // Create a blob URL for the PDF
      const blob = await pdfAPI.downloadPDF(document.id);
      const url = URL.createObjectURL(blob);
      setPdfUrl(url);
      
      // Set total pages from document metadata
      if (document.page_count) {
        setTotalPages(document.page_count);
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error("Error loading PDF:", error);
      toast({
        title: "Error loading document",
        description: "There was a problem loading the PDF. Please try again.",
        variant: "destructive"
      });
      setIsLoading(false);
    }
  };
  
  const handleZoomIn = () => {
    setZoom(Math.min(zoom + 10, 200))
  }
  
  const handleZoomOut = () => {
    setZoom(Math.max(zoom - 10, 50))
  }
  
  const handleNextPage = () => {
    setCurrentPage(Math.min(currentPage + 1, totalPages))
  }
  
  const handlePrevPage = () => {
    setCurrentPage(Math.max(currentPage - 1, 1))
  }
  
  const handleDownload = async () => {
    try {
      if (pdfUrl) {
        // Use existing blob URL if available
        const a = window.document.createElement('a');
        a.href = pdfUrl;
        a.download = document.filename || 'document.pdf';
        window.document.body.appendChild(a);
        a.click();
        window.document.body.removeChild(a);
      } else {
        // Download new blob if URL not available
        const blob = await pdfAPI.downloadPDF(document.id);
        const url = URL.createObjectURL(blob);
        
        const a = window.document.createElement('a');
        a.href = url;
        a.download = document.filename || 'document.pdf';
        window.document.body.appendChild(a);
        a.click();
        
        // Clean up
        URL.revokeObjectURL(url);
        window.document.body.removeChild(a);
      }
    } catch (error) {
      console.error("Error downloading PDF:", error);
      toast({
        title: "Download failed",
        description: "There was an error downloading the document. Please try again.",
        variant: "destructive"
      });
    }
  }
  
  return (
    <div className={cn("border-r flex flex-col h-full", className)}>
      <div className="p-3 border-b flex items-center justify-between bg-muted/20">
        <div className="text-sm font-medium truncate flex-1">
          {document.title}
        </div>
        <div className="flex items-center gap-1 sm:gap-2">
          <Button variant="outline" size="icon" onClick={handleZoomOut} className="h-8 w-8">
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs w-10 text-center">{zoom}%</span>
          <Button variant="outline" size="icon" onClick={handleZoomIn} className="h-8 w-8">
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button 
            variant="outline" 
            size="icon" 
            className="h-8 w-8 hidden sm:flex"
            onClick={handleDownload}
          >
            <Download className="h-4 w-4" />
          </Button>
          {/* <Button variant="outline" size="icon" className="h-8 w-8 hidden sm:flex">
            <Printer className="h-4 w-4" />
          </Button> */}
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-4 sm:p-6 flex justify-center h-full w-full">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <span className="ml-2">Loading document...</span>
            </div>
            ) : pdfUrl ? (
            <iframe 
              src={`${pdfUrl}#page=${currentPage}`}
              className="w-full h-full border rounded-md shadow-sm"
              title={document.title}
              style={{ 
                width: `${8.5 * zoom / 100}in`,
                height: `${11 * zoom / 100}in`
              }}
            />
            ) : (
            <div 
              className="border rounded-md shadow-sm bg-white dark:bg-muted p-4 sm:p-8 transition-all max-w-full"
              style={{ 
                width: `${8.5 * zoom / 100}in`,
                height: `${11 * zoom / 100}in`
              }}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <p className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6">{document.title}</p>
                <p className="text-muted-foreground mb-4 sm:mb-8 text-sm sm:text-base">
                  Unable to load document preview.
                </p>
                <Button variant="outline" onClick={loadPdf}>
                  Retry
                </Button>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      
      <div className="p-3 border-t flex items-center justify-between bg-muted/20">
        {/* <Button
          variant="outline"
          size="sm"
          onClick={handlePrevPage}
          disabled={currentPage === 1 || isLoading}
          className="h-8"
        >
          <ChevronLeft className="h-4 w-4 mr-1" /> <span className="hidden sm:inline">Previous</span>
        </Button> */}
        
        <span className="text-sm text-muted-foreground">
          {totalPages} Pages
        </span>
        
        {/* <Button
          variant="outline"
          size="sm"
          onClick={handleNextPage}
          disabled={currentPage === totalPages || isLoading}
          className="h-8"
        >
          <span className="hidden sm:inline">Next</span> <ChevronRight className="h-4 w-4 ml-1" />
        </Button> */}
      </div>
    </div>
  )
}