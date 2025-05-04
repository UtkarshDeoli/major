"use client"

import { useState, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCw } from 'lucide-react'

// Initialize PDF worker
if (typeof window !== 'undefined' && !pdfjs.GlobalWorkerOptions.workerSrc) {
  // Use a CDN-hosted worker file to avoid webpack issues
  pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;
}

interface PDFViewerProps {
  fileUrl: string
}

export default function PDFViewer({ fileUrl }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [scale, setScale] = useState(1.0)
  const [rotation, setRotation] = useState(0)
  const [error, setError] = useState<Error | null>(null)

  // Reset state when PDF changes
  useEffect(() => {
    setPageNumber(1)
    setScale(1.0)
    setRotation(0)
    setError(null)
  }, [fileUrl])
  
  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages)
  }
  
  function onDocumentLoadError(error: Error) {
    console.error('Error loading PDF:', error)
    setError(error)
  }

  function goToPreviousPage() {
    setPageNumber(prev => Math.max(prev - 1, 1))
  }

  function goToNextPage() {
    setPageNumber(prev => Math.min(prev + 1, numPages || 1))
  }
  
  function zoomIn() {
    setScale(prev => Math.min(prev + 0.2, 3.0))
  }
  
  function zoomOut() {
    setScale(prev => Math.max(prev - 0.2, 0.5))
  }
  
  function rotate() {
    setRotation(prev => (prev + 90) % 360)
  }

  // If there's an error loading the PDF
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-6">
        <p className="text-red-500 mb-2">Failed to load PDF</p>
        <p className="text-sm text-muted-foreground mb-4">{error.message}</p>
        <Button 
          variant="outline"
          onClick={() => window.location.reload()}
        >
          Reload page
        </Button>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-2 border-b">
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon"
            onClick={goToPreviousPage}
            disabled={pageNumber <= 1}
            className="h-8 w-8"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm px-2">
            {pageNumber} / {numPages || '-'}
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={goToNextPage}
            disabled={!numPages || pageNumber >= numPages}
            className="h-8 w-8"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="icon"
            onClick={zoomOut}
            className="h-8 w-8"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-sm px-2">
            {Math.round(scale * 100)}%
          </span>
          <Button
            variant="outline"
            size="icon"
            onClick={zoomIn}
            className="h-8 w-8"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={rotate}
            className="h-8 w-8 ml-2"
            title="Rotate"
          >
            <RotateCw className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      <div className="flex-1 overflow-auto flex justify-center p-4 bg-muted/30">
        <div className="pdf-container">
          <Document
            file={fileUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={
              <div className="flex items-center justify-center h-[600px] w-[500px]">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              </div>
            }
          >
            <Page
              pageNumber={pageNumber}
              scale={scale}
              rotate={rotation}
              renderTextLayer={false}
              renderAnnotationLayer={false}
              loading={
                <div className="flex items-center justify-center h-[600px] w-[500px]">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                </div>
              }
            />
          </Document>
        </div>
      </div>
    </div>
  )
}