"use client"

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ChevronRight } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'

// Define the star type for better type safety
type Star = {
  id: number;
  size: number;
  top: string;
  left: string;
  delay: number;
  duration: number;
  opacity: number;
}

export function LandingHero() {
  const [isLoaded, setIsLoaded] = useState(false)
  // Initialize stars with empty array to avoid hydration mismatch
  const [stars, setStars] = useState<Star[]>([])

  useEffect(() => {
    // Only generate stars on the client side
    const generateStars = () => {
      return Array.from({ length: 20 }).map((_, i) => ({
        id: i,
        size: Math.floor(Math.random() * 3) + 2, // 2-4px
        top: `${Math.random() * 100}%`,
        left: `${Math.random() * 100}%`,
        delay: Math.random() * 5,
        duration: Math.random() * 10 + 15, // 15-25s
        opacity: 0.4 + (Math.random() * 0.6)
      }));
    };

    setStars(generateStars());
    setIsLoaded(true);
  }, []);

  return (
    <section className="relative pt-20 pb-32 overflow-hidden">
      {/* Purple gradient background */}
      <div className="absolute inset-0 bg-gradient-to-b from-purple-100/60 to-transparent dark:from-purple-950/20 dark:to-transparent -z-10" />
      
      {/* Animated stars */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        {stars.map((star) => (
          <div
            key={star.id}
            className={cn(
              "absolute rounded-full bg-white dark:bg-primary/60",
              isLoaded ? "animate-float" : ""
            )}
            style={{
              width: `${star.size}px`,
              height: `${star.size}px`,
              top: star.top,
              left: star.left,
              opacity: isLoaded ? star.opacity : 0,
              animationDelay: `${star.delay}s`,
              animationDuration: `${star.duration}s`,
              transition: "opacity 0.5s ease-in-out",
              boxShadow: `0 0 ${star.size * 2}px ${star.size/2}px rgba(255, 255, 255, 0.8)`,
            }}
          />
        ))}
      </div>
      
      {/* Animated circles */}
      {/* <div className="absolute inset-0 overflow-hidden -z-10">
        <div
          className={cn(
            "absolute rounded-full bg-primary/5 dark:bg-primary/10",
            isLoaded ? "animate-[pulse_8s_ease-in-out_infinite]" : ""
          )}
          style={{
            left: "50%",
            top: "30%",
            width: "300px",
            height: "300px",
            opacity: isLoaded ? 1 : 0,
            transition: "opacity 0.5s ease-in-out",
            transform: "translate(-50%, -50%)"
          }}
        />
      </div> */}

      <div className="container px-4 mx-auto relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <div 
            className={cn(
              "inline-block px-4 py-1.5 mb-6 rounded-full bg-primary/10 text-primary font-medium text-sm transition-all duration-500",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            Your Study Companion for Exam Success
          </div>
          
          <h1 
            className={cn(
              "text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600 transition-all duration-700",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            Ace Your Exams with Padhai Whallah
          </h1>
          
          <p 
            className={cn(
              "text-xl text-muted-foreground mb-8 max-w-2xl mx-auto transition-all duration-700 delay-100",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            Upload your study materials, chat with our AI assistant, and transform the way you prepare for exams.
          </p>
          
          <div 
            className={cn(
              "flex flex-col sm:flex-row justify-center gap-4 transition-all duration-700 delay-200",
              isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
            )}
          >
            <Link href="/dashboard">
              <Button size="lg" className="w-full sm:w-auto group">
                Get Started
                <ChevronRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
            <Link href="#learn-more">
              <Button size="lg" variant="outline" className="w-full sm:w-auto">
                Learn More
              </Button>
            </Link>
          </div>
        </div>

        {/* Animated UI Preview */}
        <div 
          className={cn(
            "mt-16 max-w-5xl mx-auto rounded-lg shadow-2xl border bg-card overflow-hidden transition-all duration-1000 delay-300",
            isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
          )}
        >
          <div className="h-10 bg-muted flex items-center px-4 border-b">
            <div className="flex space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
            <div className="w-1/2 mx-auto bg-background h-6 rounded-full flex items-center justify-center text-xs text-muted-foreground">
              https://phadai.utkarshdeoli.in/dashboard
            </div>
          </div>
          <div className="h-[350px] md:h-[400px] bg-card flex">
            {/* Sidebar mockup */}
            <div className="hidden md:block w-64 border-r bg-muted/30 p-4">
              <div className="flex items-center gap-2 mb-6">
                <div className="w-8 h-8 rounded bg-primary/20"></div>
                <div className="h-4 w-32 bg-muted rounded"></div>
              </div>
              {[...Array(5)].map((_, i) => (
                <div key={i} className="flex items-center gap-3 mb-4 p-2 rounded-md bg-background/80">
                  <div className="w-4 h-4 rounded-sm bg-muted"></div>
                  <div className="h-3 w-28 bg-muted rounded"></div>
                </div>
              ))}
            </div>
            
            {/* Main content mockup */}
            <div className="flex-1 p-6 flex flex-col">
              <div className="flex-1">
                {/* Document preview */}
                <div className="mb-4 h-32 rounded-md border bg-muted/20 flex items-center justify-center">
                  <p className="text-muted-foreground text-sm">Document Preview</p>
                </div>
                
                {/* Chat messages */}
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/20 flex-shrink-0"></div>
                    <div className="bg-muted/30 rounded-lg p-3 max-w-[80%]">
                      <div className="h-3 w-40 bg-muted rounded mb-2"></div>
                      <div className="h-3 w-56 bg-muted rounded"></div>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 justify-end">
                    <div className="bg-primary/10 rounded-lg p-3 max-w-[80%]">
                      <div className="h-3 w-48 bg-primary/20 rounded mb-2"></div>
                      <div className="h-3 w-32 bg-primary/20 rounded"></div>
                    </div>
                    <div className="w-8 h-8 rounded-full bg-accent/30 flex-shrink-0"></div>
                  </div>
                </div>
              </div>
              
              {/* Chat input */}
              <div className="mt-4 relative">
                <div className="h-12 rounded-md border bg-card flex items-center px-4">
                  <div className="h-3 w-full bg-muted/50 rounded"></div>
                  <div className="ml-2 w-8 h-8 rounded-full bg-primary/20 flex-shrink-0"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}