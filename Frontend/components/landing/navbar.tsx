"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "#features", name: "Features" },
  { href: "#how-it-works", name: "How It Works" },
  { href: "#testimonials", name: "Testimonials" },
  { href: "#pricing", name: "Pricing" },
];

export function LandingNavbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 w-full h-16 bg-background/80 backdrop-blur-sm z-50 border-b border-border/30">
      <div className="w-full mx-auto lg:max-w-screen-xl px-4 md:px-12 h-full">
        <div className="flex items-center justify-between h-full">

          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center border border-primary/20">
              <img src="/logo.png" alt="Orbit Logo" className="w-6 h-6 object-contain" />
            </div>
            <span className="text-xl font-heading font-semibold text-foreground">
              Orbit
            </span>
          </Link>

          <nav className="hidden lg:flex items-center gap-8">
            <ul className="flex items-center gap-8">
              {NAV_LINKS.map((link) => (
                <li key={link.href} className="text-sm font-medium">
                  <Link
                    href={link.href}
                    className="text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          <div className="flex items-center gap-3">
            <Link href="/login" className="hidden lg:block">
              <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
                Log in
              </Button>
            </Link>
            <Link href="/dashboard" className="hidden lg:block">
              <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">
                Get Started
              </Button>
            </Link>
            <button
              className="flex lg:hidden items-center justify-center w-9 h-9 rounded-lg hover:bg-muted transition-colors"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </div>

      {isMobileMenuOpen && (
        <div className="lg:hidden absolute top-16 inset-x-0 bg-background/95 backdrop-blur-lg border-b border-border/30 z-50">
          <div className="flex flex-col p-4 space-y-1">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="px-4 py-3 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors font-medium text-sm"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                {link.name}
              </Link>
            ))}
            <hr className="my-2 border-border/30" />
            <Link href="/login" className="px-4 py-3 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors text-sm" onClick={() => setIsMobileMenuOpen(false)}>
              Log in
            </Link>
            <Link href="/dashboard" onClick={() => setIsMobileMenuOpen(false)}>
              <Button className="w-full mt-2">Get Started</Button>
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
