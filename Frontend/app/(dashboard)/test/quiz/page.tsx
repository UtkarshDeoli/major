"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { useToast } from '@/hooks/use-toast'
import { Clock, ChevronLeft, ChevronRight, Save } from 'lucide-react'
import { Progress } from '@/components/ui/progress'



interface Question {
  id: string
  type: 'mcq' | 'text'
  question: string
  options?: string[]
  correctAnswer?: string
  marks: number
}

const MOCK_QUESTIONS: Question[] = [
  {
    id: '1',
    type: 'mcq',
    question: 'What is the derivative of x²?',
    options: ['x', '2x', '2x²', 'x/2'],
    correctAnswer: '2x',
    marks: 2
  },
  {
    id: '2',
    type: 'text',
    question: 'Explain the concept of limits in calculus.',
    marks: 5
  },
  {
    id: '3',
    type: 'mcq',
    question: 'Which of these is not a type of function?',
    options: ['Linear', 'Quadratic', 'Circular', 'Exponential'],
    correctAnswer: 'Circular',
    marks: 2
  },
  {
    id: '4',
    type: 'text',
    question: 'Describe the applications of integration in real-world scenarios.',
    marks: 5
  }
]

export default function TestPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  // Use state + useEffect pattern for timer to avoid hydration mismatch
  const [timeRemaining, setTimeRemaining] = useState(3600)
  const { toast } = useToast()
  
  const question = MOCK_QUESTIONS[currentQuestion]
  // Fix: use a stable calculation that won't cause hydration mismatch
  const progress = Math.round(((currentQuestion + 1) / MOCK_QUESTIONS.length) * 100)
  
  // Add useEffect for timer to ensure client-side only execution
  useEffect(() => {
    const timer = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 0) {
          clearInterval(timer);
          // Only call handleSubmit if it won't trigger more rendering
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);
  
  const handleAnswer = (answer: string) => {
    setAnswers(prev => ({
      ...prev,
      [question.id]: answer
    }))
  }
  
  const handleNext = () => {
    if (currentQuestion < MOCK_QUESTIONS.length - 1) {
      setCurrentQuestion(prev => prev + 1)
    }
  }
  
  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1)
    }
  }
  
  const handleSubmit = () => {
    toast({
      title: "Test submitted successfully",
      description: "Your answers have been saved."
    })
  }
  
  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  
  return (
    <div className="container mx-auto py-8 px-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold">Mock Test</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Clock className="h-5 w-5" />
            <span className="font-mono">{formatTime(timeRemaining)}</span>
          </div>
          <Button onClick={handleSubmit} className="gap-2">
            <Save className="h-4 w-4" />
            Submit Test
          </Button>
        </div>
      </div>
      
      {/* Progress bar */}
      <div className="mb-8">
        <div className="flex justify-between text-sm text-muted-foreground mb-2">
          <span>Question {currentQuestion + 1} of {MOCK_QUESTIONS.length}</span>
          {/* Fix: Remove toFixed which can cause hydration issues */}
          <span>{progress}% Complete</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>
      
      {/* Question card */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="flex items-start gap-2">
            <span className="text-muted-foreground">Q{currentQuestion + 1}.</span>
            <span>{question.question}</span>
          </CardTitle>
          <div className="text-sm text-muted-foreground">
            Marks: {question.marks}
          </div>
        </CardHeader>
        <CardContent>
          {question.type === 'mcq' ? (
            <RadioGroup
              value={answers[question.id]}
              onValueChange={handleAnswer}
              className="space-y-3"
            >
              {question.options?.map((option, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <RadioGroupItem value={option} id={`option-${index}`} />
                  <Label htmlFor={`option-${index}`}>{option}</Label>
                </div>
              ))}
            </RadioGroup>
          ) : (
            <Textarea
              value={answers[question.id] || ''}
              onChange={(e) => handleAnswer(e.target.value)}
              placeholder="Type your answer here..."
              className="min-h-[200px]"
            />
          )}
        </CardContent>
      </Card>
      
      {/* Navigation */}
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={handlePrevious}
          disabled={currentQuestion === 0}
          className="gap-2"
        >
          <ChevronLeft className="h-4 w-4" /> Previous
        </Button>
        <Button
          onClick={handleNext}
          disabled={currentQuestion === MOCK_QUESTIONS.length - 1}
          className="gap-2"
        >
          Next <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      
      {/* Question palette */}
      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Question Palette</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-10 gap-2">
            {MOCK_QUESTIONS.map((q, index) => (
              <Button
                key={q.id}
                variant={answers[q.id] ? "default" : "outline"}
                className={`h-10 w-10 p-0 ${currentQuestion === index ? 'ring-2 ring-primary' : ''}`}
                onClick={() => setCurrentQuestion(index)}
              >
                {index + 1}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}