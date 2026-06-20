"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useToast } from "@/hooks/use-toast"
import { pdfAPI } from "@/lib/api"
import { FileUp, Book, FileText } from "lucide-react"

export interface PdfSelection {
  syllabusId: string
  questionPaperIds: string[]
  notesId: string
}

interface PdfSelectorProps {
  onSelectionChange: (selection: PdfSelection) => void
  showNotes?: boolean
  initialSelection?: Partial<PdfSelection>
}

export function PdfSelector({ onSelectionChange, showNotes = false, initialSelection }: PdfSelectorProps) {
  const { toast } = useToast()
  const [pdfs, setPdfs] = useState<any[]>([])
  const [selectedSyllabus, setSelectedSyllabus] = useState(initialSelection?.syllabusId || "")
  const [selectedQuestionPapers, setSelectedQuestionPapers] = useState<string[]>(initialSelection?.questionPaperIds || [])
  const [selectedNotes, setSelectedNotes] = useState(initialSelection?.notesId || "none")
  const [isUploading, setIsUploading] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<{ [key: string]: File | null }>({
    questionPaper: null,
    syllabus: null,
    notes: null,
  })

  // Fetch PDFs on mount
  useEffect(() => {
    const fetchPDFs = async () => {
      try {
        const userPdfs = await pdfAPI.listPDFs()
        setPdfs(userPdfs)
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to fetch your PDFs",
          variant: "destructive",
        })
      }
    }
    fetchPDFs()
  }, [toast])

  // Notify parent of selection changes
  useEffect(() => {
    onSelectionChange({
      syllabusId: selectedSyllabus,
      questionPaperIds: selectedQuestionPapers,
      notesId: selectedNotes,
    })
  }, [selectedSyllabus, selectedQuestionPapers, selectedNotes, onSelectionChange])

  const handleQuestionPaperToggle = (pdfId: string) => {
    setSelectedQuestionPapers((prev) =>
      prev.includes(pdfId) ? prev.filter((id) => id !== pdfId) : [...prev, pdfId]
    )
  }

  const handleFileSelect = (type: string, file: File | null) => {
    setSelectedFiles((prev) => ({ ...prev, [type]: file }))
  }

  const handleUpload = async () => {
    if (!Object.values(selectedFiles).some((file) => file)) {
      toast({
        title: "No files selected",
        description: "Please select at least one file to upload",
        variant: "destructive",
      })
      return
    }

    setIsUploading(true)
    try {
      const uploadPromises = Object.entries(selectedFiles)
        .filter(([_, file]) => file !== null)
        .map(async ([type, file]) => {
          if (file) {
            const title =
              type === "syllabus"
                ? "Syllabus"
                : type === "questionPaper"
                ? "Question Paper"
                : "Study Notes"
            return await pdfAPI.uploadPDF(file, title, `Uploaded ${title}`, [type])
          }
        })

      await Promise.all(uploadPromises)

      // Refresh the PDF list
      const userPdfs = await pdfAPI.listPDFs()
      setPdfs(userPdfs)

      // Reset selected files
      setSelectedFiles({ questionPaper: null, syllabus: null, notes: null })

      toast({
        title: "Files uploaded successfully",
        description: "Your files have been processed and are now available.",
      })
    } catch (error) {
      toast({
        title: "Upload failed",
        description: "Failed to upload files. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Syllabus Selection */}
      <div className="space-y-3">
        <Label className="text-sm font-medium">Select Syllabus</Label>
        <Select value={selectedSyllabus} onValueChange={setSelectedSyllabus}>
          <SelectTrigger className="rounded-md h-9 text-[13px]">
            <SelectValue placeholder="Choose a syllabus PDF" />
          </SelectTrigger>
          <SelectContent>
            {pdfs.length === 0 ? (
              <SelectItem value="no-pdfs" disabled>
                No PDFs available - upload some files first
              </SelectItem>
            ) : (
              pdfs.map((pdf) => (
                <SelectItem key={pdf.id} value={pdf.id}>
                  <div className="flex items-center gap-2">
                    <Book className="h-4 w-4" />
                    <span className="truncate">{pdf.title || pdf.filename}</span>
                  </div>
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
      </div>

      {/* Question Papers Selection */}
      <div className="space-y-3">
        <Label className="text-sm font-medium">
          Select Question Papers ({selectedQuestionPapers.length} selected)
        </Label>
        <ScrollArea className="h-48 w-full border rounded-md p-3">
          <div className="space-y-2">
            {pdfs.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <FileText className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No PDFs available</p>
                <p className="text-sm">Upload some files first to get started</p>
              </div>
            ) : (
              pdfs.map((pdf) => (
                <div key={pdf.id} className="flex items-center space-x-2">
                  <Checkbox
                    id={pdf.id}
                    checked={selectedQuestionPapers.includes(pdf.id)}
                    onCheckedChange={() => handleQuestionPaperToggle(pdf.id)}
                  />
                  <label
                    htmlFor={pdf.id}
                    className="flex-1 text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      <span className="truncate">{pdf.title || pdf.filename}</span>
                    </div>
                  </label>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Notes Selection (optional, for mock tests) */}
      {showNotes && (
        <div className="space-y-2">
          <Label className="text-sm font-medium">Study Notes (Optional)</Label>
          <Select value={selectedNotes} onValueChange={setSelectedNotes}>
            <SelectTrigger className="rounded-md h-9 text-[13px]">
              <SelectValue placeholder="Choose study notes (optional)" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">No notes selected</SelectItem>
              {pdfs.map((pdf) => (
                <SelectItem key={pdf.id} value={pdf.id}>
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    <span className="truncate">{pdf.title || pdf.filename}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Upload New PDFs */}
      <div className="rounded-md border">
        <div className="p-4 border-b">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <FileUp className="h-5 w-5" />
            Upload New PDFs
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Add new study materials to your collection
          </p>
        </div>
        <div className="p-4 space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="qp-upload">Question Papers</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="qp-upload"
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={(e) => handleFileSelect("questionPaper", e.target.files?.[0] || null)}
                  className="rounded-md h-9 text-[13px]"
                />
                <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="syllabus-upload">Syllabus</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="syllabus-upload"
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={(e) => handleFileSelect("syllabus", e.target.files?.[0] || null)}
                  className="rounded-md h-9 text-[13px]"
                />
                <Book className="h-5 w-5 text-muted-foreground shrink-0" />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="notes-upload">Study Notes</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="notes-upload"
                  type="file"
                  accept=".pdf,.doc,.docx"
                  onChange={(e) => handleFileSelect("notes", e.target.files?.[0] || null)}
                  className="rounded-md h-9 text-[13px]"
                />
                <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
              </div>
            </div>
          </div>
          <Button
            onClick={handleUpload}
            disabled={isUploading || !Object.values(selectedFiles).some((file) => file)}
            className="w-full rounded-md"
          >
            {isUploading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Uploading...
              </span>
            ) : (
              "Upload Files"
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}