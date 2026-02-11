"use client"

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ChevronRight, Upload, Brain, GraduationCap, FileText, MessageSquare, Search, Zap, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { LandingFeatureCard } from '@/components/landing/feature-card'
import { LandingHero } from '@/components/landing/hero'
import { LandingNavbar } from '@/components/landing/navbar'
import { FAQAccordion } from '@/components/landing/faq-accordion'
import { TestimonialsCarousel } from '@/components/landing/testimonials-carousel'
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
          className="min-h-screen py-24 px-4 md:px-8 lg:px-16 bg-gradient-to-b from-[#0D1520] to-[#15202B] flex items-center"
        >
          <div className="max-w-7xl mx-auto w-full">
            <div className="text-center mb-20">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
                <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                <span className="text-sm text-blue-400 font-medium">Powerful Features</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">Everything You Need to Succeed</h2>
              <p className="text-gray-400 max-w-3xl mx-auto leading-relaxed text-lg">
                Our AI-powered platform provides all the tools you need to ace your exams with confidence.
                From document management to AI-powered chat, we've got you covered.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
                icon="Brain"
                title="Concept Breakdown"
                description="Break down complex topics into easily digestible information."
                delay={0.5}
                isLoaded={isLoaded}
              />
              <LandingFeatureCard
                icon="Zap"
                title="Instant Quiz Generation"
                description="Generate quizzes instantly to test your knowledge and track progress."
                delay={0.6}
                isLoaded={isLoaded}
              />
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" className="min-h-screen py-24 px-4 md:px-8 lg:px-16 bg-[#0D1520] flex items-center">
          <div className="max-w-7xl mx-auto w-full">
            <div className="text-center mb-20">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
                <span className="text-sm text-blue-400 font-medium">Simple Process</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">How It Works</h2>
              <p className="text-gray-400 max-w-3xl mx-auto leading-relaxed text-lg">
                Get started in minutes with our simple 3-step process. Your journey to exam success begins here.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Step 1 */}
              <div className="relative group">
                <div className="bg-[#15202B] rounded-2xl p-8 h-full border border-blue-500/10 hover:border-blue-500/30 transition-all duration-500">
                  <div className="absolute -top-4 left-8">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/30">
                      01
                    </div>
                  </div>
                  <div className="mt-8 mb-6">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center mb-4">
                      <Upload className="w-8 h-8 text-blue-400" />
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-3">Upload Your Materials</h3>
                    <p className="text-gray-400 leading-relaxed mb-6">
                      Drag and drop your PDF documents, notes, or images into the platform. Our system accepts various file formats and organizes everything for you.
                    </p>
                  </div>
                  <div className="bg-[#0D1520] rounded-xl p-4 border border-blue-500/10">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center">
                        <FileText className="w-4 h-4 text-blue-400" />
                      </div>
                      <span className="text-sm text-gray-300">Supported Formats</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs">PDF</span>
                      <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs">DOCX</span>
                      <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs">Images</span>
                      <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs">Notes</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 2 */}
              <div className="relative group">
                <div className="bg-[#15202B] rounded-2xl p-8 h-full border border-blue-500/10 hover:border-blue-500/30 transition-all duration-500">
                  <div className="absolute -top-4 left-8">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/30">
                      02
                    </div>
                  </div>
                  <div className="mt-8 mb-6">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center mb-4">
                      <Brain className="w-8 h-8 text-blue-400" />
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-3">AI Processing</h3>
                    <p className="text-gray-400 leading-relaxed mb-6">
                      Our advanced AI analyzes and indexes your content, making it searchable and ready for intelligent chat interactions.
                    </p>
                  </div>
                  <div className="bg-[#0D1520] rounded-xl p-4 border border-blue-500/10">
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-green-400" />
                        <span className="text-sm text-gray-300">Content indexed</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-green-400" />
                        <span className="text-sm text-gray-300">Keywords extracted</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 rounded-full bg-green-400" />
                        <span className="text-sm text-gray-300">Concepts mapped</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 3 */}
              <div className="relative group">
                <div className="bg-[#15202B] rounded-2xl p-8 h-full border border-blue-500/10 hover:border-blue-500/30 transition-all duration-500">
                  <div className="absolute -top-4 left-8">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/30">
                      03
                    </div>
                  </div>
                  <div className="mt-8 mb-6">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center mb-4">
                      <GraduationCap className="w-8 h-8 text-blue-400" />
                    </div>
                    <h3 className="text-2xl font-bold text-white mb-3">Learn & Prepare</h3>
                    <p className="text-gray-400 leading-relaxed mb-6">
                      Chat with AI, generate quizzes, and master your exam topics. Track your progress and improve continuously.
                    </p>
                  </div>
                  <div className="bg-[#0D1520] rounded-xl p-4 border border-blue-500/10">
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div>
                        <div className="text-2xl font-bold text-blue-400">24/7</div>
                        <div className="text-xs text-gray-500">AI Support</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-blue-400">100+</div>
                        <div className="text-xs text-gray-500">Topics</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-blue-400">∞</div>
                        <div className="text-xs text-gray-500">Quizzes</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-16 bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-2xl p-8 border border-blue-500/20">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-4">Ready to transform your study experience?</h3>
                  <p className="text-gray-400 mb-6">Join thousands of students who are already using Orbit to ace their exams.</p>
                  <Link href="/dashboard">
                    <Button className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg shadow-blue-500/25">
                      Start Free Trial
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                </div>
                <div className="flex justify-center">
                  <div className="w-64 h-48 bg-[#15202B] rounded-xl border border-blue-500/20 flex items-center justify-center">
                    <span className="text-gray-500">Demo Preview</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Demo Preview Section */}
        <section className="min-h-screen py-24 px-4 md:px-8 lg:px-16 bg-gradient-to-b from-[#15202B] to-[#0D1520] flex items-center">
          <div className="max-w-7xl mx-auto w-full">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
                <span className="text-sm text-blue-400 font-medium">See It In Action</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">Powerful AI at Your Fingertips</h2>
              <p className="text-gray-400 max-w-3xl mx-auto leading-relaxed text-lg">
                Experience the future of learning with our AI-powered features designed to help you succeed.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-[#15202B] rounded-2xl p-6 border border-blue-500/10">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                    <MessageSquare className="w-5 h-5 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white">AI Chat Assistant</h3>
                </div>
                <div className="bg-[#0D1520] rounded-xl p-4 space-y-4">
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0">
                      <span className="text-xs font-bold text-white">AI</span>
                    </div>
                    <div className="bg-[#15202B] rounded-xl rounded-tl-sm p-3 border border-blue-500/20">
                      <p className="text-sm text-gray-300">Hello! I'm your AI study assistant. Ask me anything about your uploaded documents.</p>
                    </div>
                  </div>
                  <div className="flex gap-3 flex-row-reverse">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center flex-shrink-0">
                      <span className="text-xs font-bold text-white">You</span>
                    </div>
                    <div className="bg-blue-500/10 rounded-xl rounded-tr-sm p-3 border border-blue-500/20">
                      <p className="text-sm text-gray-300">Can you explain the photosynthesis process?</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-[#15202B] rounded-2xl p-6 border border-blue-500/10">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                    <Zap className="w-5 h-5 text-green-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white">Quiz Generator</h3>
                </div>
                <div className="bg-[#0D1520] rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between p-3 bg-[#15202B] rounded-lg border border-blue-500/10">
                    <span className="text-sm text-gray-300">What is photosynthesis?</span>
                    <span className="text-xs text-green-400">✓ Correct</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-[#15202B] rounded-lg border border-blue-500/10">
                    <span className="text-sm text-gray-300">Which organelle performs photosynthesis?</span>
                    <span className="text-xs text-green-400">✓ Correct</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-[#15202B] rounded-lg border border-blue-500/10">
                    <span className="text-sm text-gray-300">What is the primary pigment?</span>
                    <span className="text-xs text-green-400">✓ Correct</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Testimonials Section */}
        <section id="testimonials" className="min-h-screen py-24 px-4 md:px-8 lg:px-16 bg-gradient-to-b from-[#0D1520] to-[#15202B] flex items-center overflow-hidden">
          <div className="max-w-7xl mx-auto w-full">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
                <span className="text-sm text-blue-400 font-medium">Student Success</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">What Students Say</h2>
              <p className="text-gray-400 max-w-3xl mx-auto leading-relaxed text-lg">
                Join thousands of students who have transformed their study habits with Orbit.
              </p>
            </div>

            <TestimonialsCarousel
              testimonials={[
                { name: "Aisha Patel", role: "Medical Student", avatar: "AP", content: "Padhai Whallah helped me prepare for my medical entrance exam. The AI chat feature is incredibly helpful for clarifying complex concepts.", rating: 5 },
                { name: "Rahul Sharma", role: "Engineering Student", avatar: "RS", content: "The document upload and search functionality saves me hours of study time. I can quickly find exactly what I need.", rating: 5 },
                { name: "Priya Singh", role: "Commerce Student", avatar: "PS", content: "Quiz generation feature is a game-changer. I can test my knowledge and track my progress effectively.", rating: 5 },
                { name: "Vikram Kumar", role: "Law Student", avatar: "VK", content: "The AI-powered summaries help me review case laws quickly. Best study companion for law exams!", rating: 5 },
                { name: "Ananya Reddy", role: "Science Student", avatar: "AR", content: "Love the smart search feature! It helps me find relevant study material within seconds.", rating: 5 },
                { name: "Sanjay Gupta", role: "CA Student", avatar: "SG", content: "The progress analytics help me identify weak areas. My exam scores improved significantly!", rating: 5 }
              ]}
            />
          </div>
        </section>

        {/* Stats Section */}
        <section className="py-24 px-4 md:px-8 lg:px-16 bg-[#0D1520]">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
              {[
                { value: "50K+", label: "Active Students", icon: "Users" },
                { value: "1M+", label: "Documents Uploaded", icon: "FileText" },
                { value: "95%", label: "Success Rate", icon: "Zap" },
                { value: "24/7", label: "AI Support", icon: "Brain" }
              ].map((stat, i) => (
                <div key={i} className="text-center">
                  <div className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-2">{stat.value}</div>
                  <div className="text-gray-500">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* FAQ Section */}
        <section id="faq" className="min-h-screen py-24 px-4 md:px-8 lg:px-16 bg-gradient-to-b from-[#15202B] to-[#0D1520] flex items-center">
          <div className="max-w-4xl mx-auto w-full">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
                <span className="text-sm text-blue-400 font-medium">Help & Support</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">Frequently Asked Questions</h2>
              <p className="text-gray-400 max-w-2xl mx-auto leading-relaxed text-lg">
                Have questions? We've got answers. If you don't see your question here, feel free to contact us.
              </p>
            </div>

            <FAQAccordion
              items={[
                { question: "What types of documents can I upload?", answer: "You can upload PDF files, images (JPG, PNG), and text documents. Our AI will process and index all your study materials." },
                { question: "How does the AI chat assistance work?", answer: "Our AI assistant answers questions based on your uploaded materials. It understands context and provides accurate, helpful responses." },
                { question: "Can I use Orbit on mobile devices?", answer: "Yes! Orbit is fully responsive and works great on smartphones, tablets, and desktops." },
                { question: "What payment methods do you accept?", answer: "We accept all major credit/debit cards, UPI, and net banking. All payments are securely processed." },
                { question: "Can I cancel my subscription anytime?", answer: "Absolutely! You can cancel your subscription at any time from your account settings. You'll continue to have access until your billing period ends." },
                { question: "Is my data secure?", answer: "Yes, we take data security seriously. All your data is encrypted and stored securely. We never share your data with third parties." }
              ]}
              isLoaded={isLoaded}
            />
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="min-h-screen py-24 px-4 md:px-8 lg:px-16 bg-[#0D1520] flex items-center">
          <div className="max-w-7xl mx-auto w-full">
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
                <span className="text-sm text-blue-400 font-medium">Flexible Plans</span>
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">Choose Your Plan</h2>
              <p className="text-gray-400 max-w-3xl mx-auto leading-relaxed text-lg">
                Select the perfect plan that fits your study needs and budget. All plans include our core features.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="bg-[#15202B] rounded-2xl p-8 border border-blue-500/10 hover:border-blue-500/30 transition-all duration-500 hover:-translate-y-2">
                <h3 className="text-2xl font-bold text-white mb-2">Basic</h3>
                <p className="text-gray-400 mb-4">Get started for free</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-white">₹0</span>
                  <span className="text-gray-500">/month</span>
                </div>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>5 Documents Upload</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Basic AI Assistance</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Standard Support</li>
                </ul>
                <Link href="/dashboard" className="block">
                  <Button className="w-full bg-[#15202B] border border-blue-500/20 hover:bg-blue-500/10">Get Started</Button>
                </Link>
              </div>

              <div className="bg-[#15202B] rounded-2xl p-8 border-2 border-blue-500/50 relative hover:-translate-y-2 transition-all duration-500">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-blue-500 to-cyan-500 px-4 py-1 rounded-full text-sm font-medium text-white">Most Popular</div>
                <h3 className="text-2xl font-bold text-white mb-2">Pro</h3>
                <p className="text-gray-400 mb-4">Perfect for serious students</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-white">₹499</span>
                  <span className="text-gray-500">/month</span>
                </div>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Unlimited Documents</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Advanced AI Assistance</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Priority Support</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Quiz Generation</li>
                </ul>
                <Link href="/dashboard" className="block">
                  <Button className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg shadow-blue-500/25">Get Started</Button>
                </Link>
              </div>

              <div className="bg-[#15202B] rounded-2xl p-8 border border-blue-500/10 hover:border-blue-500/30 transition-all duration-500 hover:-translate-y-2">
                <h3 className="text-2xl font-bold text-white mb-2">Premium</h3>
                <p className="text-gray-400 mb-4">For advanced exam preparation</p>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-white">₹999</span>
                  <span className="text-gray-500">/month</span>
                </div>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Everything in Pro</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Custom Study Plans</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>24/7 Dedicated Support</li>
                  <li className="flex items-center text-gray-300"><span className="text-blue-400 mr-2">✓</span>Personalized Reports</li>
                </ul>
                <Link href="/dashboard" className="block">
                  <Button className="w-full bg-[#15202B] border border-blue-500/20 hover:bg-blue-500/10">Get Started</Button>
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="min-h-[80vh] py-24 px-4 md:px-8 lg:px-16 bg-gradient-to-b from-[#15202B] to-[#0D1520] flex items-center">
          <div className="max-w-4xl mx-auto text-center w-full">
            <h2 className="text-4xl md:text-5xl font-bold mb-6 text-white">Ready to Elevate Your Study Game?</h2>
            <p className="text-gray-400 max-w-2xl mx-auto leading-relaxed text-lg mb-10">
              Join thousands of students who have transformed their exam preparation with Orbit. Start your journey today.
            </p>
            <Link href="/dashboard">
              <Button size="lg" className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 shadow-lg shadow-blue-500/25 px-10 py-7 text-xl">
                Start Learning Free
                <ChevronRight className="ml-2 h-6 w-6" />
              </Button>
            </Link>
          </div>
        </section>
      </main>

      <footer className="py-12 px-4 border-t border-blue-500/10 bg-[#0D1520]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <p className="font-bold text-2xl text-white mb-1">Orbit</p>
            <p className="text-gray-500 text-sm">© 2025 All rights reserved</p>
          </div>
          <div className="flex gap-6">
            <Link href="#" className="text-gray-500 hover:text-blue-400 transition-colors">Terms</Link>
            <Link href="#" className="text-gray-500 hover:text-blue-400 transition-colors">Privacy</Link>
            <Link href="#" className="text-gray-500 hover:text-blue-400 transition-colors">Help</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}
