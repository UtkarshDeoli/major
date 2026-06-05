"use client";

import Link from "next/link";
import Container from "@/components/global/container";
import { Button } from "@/components/ui/button";
import Particles from "@/components/ui/particles";

export function CTASection() {
  return (
    <div className="relative flex flex-col items-center justify-center w-full py-20 px-4 md:px-12">
      <Container className="py-20 max-w-6xl mx-auto w-full">
        <div className="relative flex flex-col items-center justify-center py-12 lg:py-20 px-0 rounded-2xl lg:rounded-3xl bg-background/20 text-center border border-foreground/10 overflow-hidden">

          {/* Permanent ambient glows — always visible */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background:
                "radial-gradient(ellipse 60% 50% at 20% 110%, rgba(56,189,248,0.18) 0%, transparent 70%), " +
                "radial-gradient(ellipse 50% 40% at 80% 110%, rgba(99,102,241,0.14) 0%, transparent 65%)",
            }}
          />

          {/* Subtle top edge glow */}
          <div
            className="absolute top-0 left-1/2 -translate-x-1/2 w-2/3 h-px"
            style={{
              background: "linear-gradient(90deg, transparent, rgba(56,189,248,0.5), transparent)",
              boxShadow: "0 0 30px 6px rgba(56,189,248,0.15)",
            }}
          />

          <Particles
            refresh
            ease={80}
            quantity={80}
            color="#d4d4d4"
            className="hidden lg:block absolute inset-0 z-0"
          />
          <Particles
            refresh
            ease={80}
            quantity={35}
            color="#d4d4d4"
            className="block lg:hidden absolute inset-0 z-0"
          />

          <h2 className="text-3xl md:text-5xl lg:text-6xl font-heading font-medium !leading-snug relative z-10">
            Ready to elevate your <br />{" "}
            <span className="font-subheading italic">study game</span>?
          </h2>
          <p className="text-sm md:text-lg text-center text-muted-foreground max-w-2xl mx-auto mt-4 font-heading relative z-10">
            Join thousands of students who have transformed their exam preparation with Orbit.{" "}
            <span className="hidden lg:inline">Start your journey to better grades today.</span>
          </p>
          <Link href="/dashboard" className="mt-8 relative z-10">
            <Button size="lg" className="font-heading">
              Start Learning Free
            </Button>
          </Link>
        </div>
      </Container>
    </div>
  );
}
