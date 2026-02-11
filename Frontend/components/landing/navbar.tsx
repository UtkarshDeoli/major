"use client"

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Book, Menu, X, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/theme-toggle'
import { cn } from '@/lib/utils'

export function LandingNavbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        isScrolled
          ? "bg-[#15202B]/90 backdrop-blur-lg border-b border-white/5"
          : "bg-transparent"
      )}
    >
      <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-8">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500/20 to-coral-500/20 flex items-center justify-center border border-white/10 group-hover:border-red-500/30 transition-colors">
              <img 
                src="/logo.png" 
                alt="Orbit Logo" 
                className="w-8 h-8 object-contain"
              />
            </div>
          </div>
          <span className="font-bold text-xl text-white group-hover:text-red-400 transition-colors">Orbit</span>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {[
            { href: '#home', label: 'Home' },
            { href: '#features', label: 'Features' },
            { href: '#how-it-works', label: 'How It Works' },
            { href: '#testimonials', label: 'Testimonials' },
            { href: '#pricing', label: 'Pricing' },
            { href: '#faq', label: 'FAQ' }
          ].map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="px-4 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all text-sm font-medium"
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="hidden md:flex items-center gap-3">
          <ThemeToggle />
          <Link href="/login">
            <Button variant="ghost" className="text-gray-400 hover:text-white hover:bg-white/5">
              Log in
            </Button>
          </Link>
          <Link href="/dashboard">
            <Button className="bg-gradient-to-r shadow-lg transition-all duration-300 hover:scale-105">
              Get Started
            </Button>
          </Link>
        </div>

        <div className="flex items-center gap-2 md:hidden">
          <ThemeToggle />
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
            className={cn(
              "text-gray-400 hover:text-white",
              isMobileMenuOpen ? "bg-white/5" : ""
            )}
          >
            {isMobileMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>

      {isMobileMenuOpen && (
        <div className="md:hidden absolute top-16 inset-x-0 bg-[#15202B]/95 backdrop-blur-lg border-b border-white/5 z-50 animate-in slide-in-from-top-2">
          <div className="flex flex-col p-4 space-y-2">
            {[
              { href: '#home', label: 'Home' },
              { href: '#features', label: 'Features' },
              { href: '#how-it-works', label: 'How It Works' },
              { href: '#testimonials', label: 'Testimonials' },
              { href: '#pricing', label: 'Pricing' },
              { href: '#faq', label: 'FAQ' }
            ].map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="px-4 py-3 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors font-medium"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}
            <hr className="my-2 border-white/5" />
            <Link
              href="/login"
              className="px-4 py-3 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Log in
            </Link>
            <Link href="/dashboard" onClick={() => setIsMobileMenuOpen(false)}>
              <Button className="w-full mt-2 bg-gradient-to-r from-red-500 to-coral-500 hover:from-red-600 hover:to-coral-600">
                Get Started
              </Button>
            </Link>
          </div>
        </div>
      )}
    </header>
  )
}
