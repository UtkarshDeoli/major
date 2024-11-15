"use client"

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Document } from '@/lib/data'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Download, Printer } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DocumentViewerProps {
  document: Document
  className?: string
}

export function DocumentViewer({ document, className }: DocumentViewerProps) {
  const [zoom, setZoom] = useState(100)
  const [currentPage, setCurrentPage] = useState(1)
  const totalPages = 5 // Mock data - in real app would depend on actual PDF
  
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
          <Button variant="outline" size="icon" className="h-8 w-8 hidden sm:flex">
            <Download className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" className="h-8 w-8 hidden sm:flex">
            <Printer className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-4 sm:p-6 flex justify-center min-h-full">
          {/* This would be replaced with an actual PDF viewer in a real app */}
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
                This is a placeholder for document content.
              </p>
              <p className="text-muted-foreground text-sm sm:text-base">
                In a real application, this would display the actual PDF content.
              </p>
            </div>
          </div>
        </div>
      </ScrollArea>
      
      <div className="p-3 border-t flex items-center justify-between bg-muted/20">
        <Button
          variant="outline"
          size="sm"
          onClick={handlePrevPage}
          disabled={currentPage === 1}
          className="h-8"
        >
          <ChevronLeft className="h-4 w-4 mr-1" /> <span className="hidden sm:inline">Previous</span>
        </Button>
        
        <span className="text-sm text-muted-foreground">
          {currentPage}/{totalPages}
        </span>
        
        <Button
          variant="outline"
          size="sm"
          onClick={handleNextPage}
          disabled={currentPage === totalPages}
          className="h-8"
        >
          <span className="hidden sm:inline">Next</span> <ChevronRight className="h-4 w-4 ml-1" />
        </Button>
      </div>
    </div>
  )
}