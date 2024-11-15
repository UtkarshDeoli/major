"use client"

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { FileUp, Book, FileText, Clock, CheckCircle } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { useRouter } from 'next/navigation'

export default function TestPage() {
  const router = useRouter()
  const { toast } = useToast()
  const [isUploading, setIsUploading] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<{ [key: string]: File | null }>({
    questionPaper: null,
    syllabus: null,
    notes: null
  })

  const handleFileSelect = (type: string, file: File | null) => {
    setSelectedFiles(prev => ({
      ...prev,
      [type]: file
    }))
  }

  const handleUpload = () => {
    setIsUploading(true)
    setTimeout(() => {
      setIsUploading(false)
      toast({
        title: "Files uploaded successfully",
        description: "Your test materials have been processed."
      })
    }, 2000)
  }

  const handleStartTest = (testId: string) => {
    router.push(`/test/${testId}`)
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <Tabs defaultValue="generate" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 max-w-[400px]">
          <TabsTrigger value="generate">Generate Test</TabsTrigger>
          <TabsTrigger value="mock">Mock Tests</TabsTrigger>
        </TabsList>

        <TabsContent value="generate">
          <div className="grid gap-6 max-w-2xl mx-auto">
            <Card>
              <CardHeader>
                <CardTitle>Generate Question Paper</CardTitle>
                <CardDescription>
                  Upload your materials to generate a customized question paper
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="questionPaper">Previous Question Papers</Label>
                  <div className="flex items-center gap-4">
                    <Input
                      id="questionPaper"
                      type="file"
                      accept=".pdf,.doc,.docx"
                      onChange={(e) => handleFileSelect('questionPaper', e.target.files?.[0] || null)}
                    />
                    <FileUp className="h-5 w-5 text-muted-foreground" />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="syllabus">Syllabus</Label>
                  <div className="flex items-center gap-4">
                    <Input
                      id="syllabus"
                      type="file"
                      accept=".pdf,.doc,.docx"
                      onChange={(e) => handleFileSelect('syllabus', e.target.files?.[0] || null)}
                    />
                    <Book className="h-5 w-5 text-muted-foreground" />
                  </div>
                </div>

                <Button 
                  onClick={handleUpload} 
                  disabled={isUploading || !Object.values(selectedFiles).some(file => file)}
                  className="w-full"
                >
                  {isUploading ? (
                    <span className="flex items-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      Processing...
                    </span>
                  ) : (
                    'Generate Question Paper'
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="mock">
          <div className="grid gap-6 max-w-4xl mx-auto">
            <Card>
              <CardHeader>
                <CardTitle>Mock Test Generator</CardTitle>
                <CardDescription>
                  Create practice tests using your study materials
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="mockQuestionPaper">Question Papers</Label>
                    <div className="flex items-center gap-4">
                      <Input
                        id="mockQuestionPaper"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('questionPaper', e.target.files?.[0] || null)}
                      />
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="mockSyllabus">Syllabus</Label>
                    <div className="flex items-center gap-4">
                      <Input
                        id="mockSyllabus"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('syllabus', e.target.files?.[0] || null)}
                      />
                      <Book className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="mockNotes">Study Notes</Label>
                    <div className="flex items-center gap-4">
                      <Input
                        id="mockNotes"
                        type="file"
                        accept=".pdf,.doc,.docx"
                        onChange={(e) => handleFileSelect('notes', e.target.files?.[0] || null)}
                      />
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                  </div>
                </div>

                <div className="grid gap-4 mt-6">
                  <h3 className="font-semibold">Available Mock Tests</h3>
                  
                  <div className="grid gap-4 md:grid-cols-2">
                    {[
                      { id: "test1", title: "Full Length Test", duration: "3 hours", questions: 100 },
                      { id: "test2", title: "Quick Practice", duration: "1 hour", questions: 30 },
                      { id: "test3", title: "Topic Focus: Calculus", duration: "45 minutes", questions: 25 },
                      { id: "test4", title: "Chapter Review", duration: "30 minutes", questions: 20 }
                    ].map((test) => (
                      <Card key={test.id}>
                        <CardContent className="pt-6">
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h4 className="font-semibold">{test.title}</h4>
                              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                                <span className="flex items-center gap-1">
                                  <Clock className="h-4 w-4" />
                                  {test.duration}
                                </span>
                                <span className="flex items-center gap-1">
                                  <CheckCircle className="h-4 w-4" />
                                  {test.questions} questions
                                </span>
                              </div>
                            </div>
                          </div>
                          <Button 
                            className="w-full"
                            onClick={() => handleStartTest(test.id)}
                          >
                            Start Test
                          </Button>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}