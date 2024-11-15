"use client"

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { LandingFeatureCard } from '@/components/landing/feature-card'
import { LandingHero } from '@/components/landing/hero'
import { LandingNavbar } from '@/components/landing/navbar'
import Link from 'next/link'

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <LandingNavbar />
      <main className="flex-1">
        <LandingHero />

        {/* Features Section */}
        <section
        id="features" 
        className="py-16 px-4 md:px-8 lg:px-16 bg-gradient-to-b from-purple-50 to-white dark:from-purple-950/20 dark:to-background transition-colors">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">Revolutionize Your Study Experience</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Our AI-powered platform provides everything you need to ace your exams with confidence.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <LandingFeatureCard 
                icon="FileText" 
                title="Document Management" 
                description="Upload and organize your study materials in one centralized location."
                delay={0.1}
                isLoaded={isLoaded}
              />
              <LandingFeatureCard 
                icon="MessageSquare" 
                title="AI Chat Assistant" 
                description="Get instant answers to your questions from our intelligent AI assistant."
                delay={0.2}
                isLoaded={isLoaded}
              />
              <LandingFeatureCard 
                icon="Search" 
                title="Smart Search" 
                description="Quickly find what you're looking for with our powerful search functionality."
                delay={0.3}
                isLoaded={isLoaded}
              />
              <LandingFeatureCard 
                icon="FileUp" 
                title="Easy File Uploads" 
                description="Simply drag and drop your PDFs, images, and documents."
                delay={0.4}
                isLoaded={isLoaded}
              />
              <LandingFeatureCard 
                icon="BrainCircuit" 
                title="Concept Breakdown" 
                description="Break down complex topics into easily digestible information."
                delay={0.5}
                isLoaded={isLoaded}
              />
              <LandingFeatureCard 
                icon="Settings" 
                title="Customizable" 
                description="Tailor the platform to fit your personal study preferences."
                delay={0.6}
                isLoaded={isLoaded}
              />
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section
        id="pricing"
        className="py-20 px-4 md:px-8 lg:px-16 bg-muted/20 dark:bg-muted/5">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">Choose Your Plan</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                Select the perfect plan that fits your study needs and budget.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 ">
              {/* Free Plan */}
              <div className={cn(
                "rounded-lg border bg-card shadow-sm transition-all duration-500",
                isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
              )}
              style={{ transitionDelay: "100ms" }}>
                <div className="p-6 border-b">
                  <h3 className="text-xl font-bold">Basic</h3>
                  <p className="text-muted-foreground mt-1">Get started for free</p>
                  <div className="mt-4 flex items-baseline">
                    <span className="text-3xl font-bold">₹0</span>
                    <span className="text-muted-foreground ml-1">/month</span>
                  </div>
                </div>
                <div className="p-6 space-y-4">
                  <ul className="space-y-3">
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>5 Documents Upload</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Basic AI Assistance</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Standard Support</span>
                    </li>
                    <li className="flex items-center text-muted-foreground">
                      <svg className="h-5 w-5 text-muted-foreground mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                      </svg>
                      <span>Advanced Features</span>
                    </li>
                  </ul>
                  <Link href="/dashboard">
                    <Button className="w-full mt-16" variant="outline">
                      Get Started
                    </Button>
                  </Link>
                </div>
              </div>

              {/* Pro Plan (Highlighted) */}
              <div className={cn(
                "rounded-lg border-2 border-primary bg-card shadow-md relative transition-all duration-500",
                isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
              )}
              style={{ transitionDelay: "300ms" }}>
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-medium">
                  Popular
                </div>
                <div className="p-6 border-b">
                  <h3 className="text-xl font-bold">Pro</h3>
                  <p className="text-muted-foreground mt-1">Perfect for serious students</p>
                  <div className="mt-4 flex items-baseline">
                    <span className="text-3xl font-bold">₹499</span>
                    <span className="text-muted-foreground ml-1">/month</span>
                  </div>
                </div>
                <div className="p-6 space-y-4">
                  <ul className="space-y-3">
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Unlimited Documents</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Advanced AI Assistance</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Priority Support</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Quiz Generation</span>
                    </li>
                  </ul>
                  <Link href="/dashboard">
                    <Button className="w-full mt-16">
                      Get Started
                    </Button>
                  </Link>
                </div>
              </div>

              {/* Premium Plan */}
              <div
              
               className={cn(
                "rounded-lg border bg-card shadow-sm transition-all duration-500",
                isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
              )}
              style={{ transitionDelay: "500ms" }}>
                <div className="p-6 border-b">
                  <h3 className="text-xl font-bold">Premium</h3>
                  <p className="text-muted-foreground mt-1">For advanced exam preparation</p>
                  <div className="mt-4 flex items-baseline">
                    <span className="text-3xl font-bold">₹999</span>
                    <span className="text-muted-foreground ml-1">/month</span>
                  </div>
                </div>
                <div className="p-6 space-y-4">
                  <ul className="space-y-3">
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Everything in Pro</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Custom Study Plans</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>24/7 Dedicated Support</span>
                    </li>
                    <li className="flex items-center">
                      <svg className="h-5 w-5 text-primary mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                      </svg>
                      <span>Personalized Reports</span>
                    </li>
                  </ul>
                  <Link href="/dashboard">
                    <Button className="w-full mt-16" variant="outline">
                      Get Started
                    </Button>
                  </Link>
                </div>
              </div>
            </div>

            <div className="mt-12 text-center">
              <p className="text-muted-foreground mb-4">
                Need a custom solution for your institution?
              </p>
              <Link href="#contact">
                <Button variant="link" className="gap-1">
                  Contact us for enterprise pricing
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-4 md:px-8 lg:px-16 bg-primary/5 dark:bg-primary/10">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Elevate Your Study Game?</h2>
            <p className="text-lg text-muted-foreground mb-8">
              Join thousands of students who have transformed their exam preparation with Padhai Whallah.
            </p>
            <Link href="/dashboard">
              <Button size="lg" className="group">
                Get Started
                <ChevronRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="py-8 px-4 border-t">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="font-semibold text-lg">Padhai Whallah</p>
            <p className="text-muted-foreground text-sm">© 2025 All rights reserved</p>
          </div>
          <div className="flex gap-6">
            <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">Terms</Link>
            <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">Privacy</Link>
            <Link href="#" className="text-muted-foreground hover:text-foreground transition-colors">Help</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}