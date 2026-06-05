"use client";

import { ArrowRightIcon } from "lucide-react";
import Link from "next/link";
import Container from "@/components/global/container";
import { Button } from "@/components/ui/button";
import { OrbitingCircles } from "@/components/ui/orbiting-circles";

// ─── Planet SVG Components ────────────────────────────────────────────────────

/** Small icy moon with surface cracks – inner orbit */
const SmallIcyMoon = () => (
  <svg width="20" height="20" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="sim-halo" cx="50%" cy="50%" r="50%">
        <stop offset="60%" stopColor="transparent" />
        <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.5" />
      </radialGradient>
      <radialGradient id="sim-base" cx="32%" cy="26%" r="72%">
        <stop offset="0%" stopColor="#f0f9ff" />
        <stop offset="28%" stopColor="#bae6fd" />
        <stop offset="62%" stopColor="#38bdf8" />
        <stop offset="100%" stopColor="#0c4a6e" />
      </radialGradient>
      <clipPath id="sim-clip"><circle cx="50" cy="50" r="43" /></clipPath>
      <radialGradient id="sim-spec" cx="28%" cy="22%" r="28%">
        <stop offset="0%" stopColor="white" stopOpacity="0.65" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#sim-halo)" />
    <circle cx="50" cy="50" r="43" fill="url(#sim-base)" />
    <g clipPath="url(#sim-clip)">
      <ellipse cx="46" cy="56" rx="13" ry="9" fill="#e0f2fe" opacity="0.22" />
      <ellipse cx="65" cy="38" rx="9" ry="6" fill="#f0f9ff" opacity="0.18" />
      <path d="M30 38 L46 54 L39 70" stroke="#e0f2fe" strokeWidth="1.5" fill="none" strokeOpacity="0.5" />
      <path d="M54 24 L60 44 L76 52" stroke="#bae6fd" strokeWidth="1" fill="none" strokeOpacity="0.4" />
      <path d="M22 60 L36 64 L28 80" stroke="#e0f2fe" strokeWidth="1" fill="none" strokeOpacity="0.35" />
      <ellipse cx="50" cy="14" rx="20" ry="9" fill="white" opacity="0.2" />
    </g>
    <circle cx="50" cy="50" r="43" fill="url(#sim-spec)" />
  </svg>
);

/** Tiny cratered rocky moon – inner orbit */
const TinyRocky = () => (
  <svg width="16" height="16" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="tr-halo" cx="50%" cy="50%" r="50%">
        <stop offset="60%" stopColor="transparent" />
        <stop offset="100%" stopColor="#92400e" stopOpacity="0.28" />
      </radialGradient>
      <radialGradient id="tr-base" cx="34%" cy="28%" r="72%">
        <stop offset="0%" stopColor="#d6d3d1" />
        <stop offset="32%" stopColor="#78716c" />
        <stop offset="68%" stopColor="#44403c" />
        <stop offset="100%" stopColor="#1c1917" />
      </radialGradient>
      <clipPath id="tr-clip"><circle cx="50" cy="50" r="43" /></clipPath>
      <radialGradient id="tr-spec" cx="28%" cy="22%" r="22%">
        <stop offset="0%" stopColor="white" stopOpacity="0.22" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
      <radialGradient id="tr-crat" cx="40%" cy="35%" r="60%">
        <stop offset="0%" stopColor="#1c1917" stopOpacity="0.7" />
        <stop offset="100%" stopColor="#1c1917" stopOpacity="0" />
      </radialGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#tr-halo)" />
    <circle cx="50" cy="50" r="43" fill="url(#tr-base)" />
    <g clipPath="url(#tr-clip)">
      <circle cx="38" cy="42" r="11" fill="url(#tr-crat)" />
      <circle cx="38" cy="42" r="11" fill="none" stroke="#a8a29e" strokeWidth="1" strokeOpacity="0.22" />
      <circle cx="63" cy="58" r="8" fill="url(#tr-crat)" />
      <circle cx="63" cy="58" r="8" fill="none" stroke="#a8a29e" strokeWidth="0.8" strokeOpacity="0.18" />
      <circle cx="54" cy="33" r="6" fill="url(#tr-crat)" />
      <circle cx="30" cy="66" r="7" fill="url(#tr-crat)" />
      <circle cx="72" cy="38" r="4" fill="url(#tr-crat)" />
    </g>
    <circle cx="50" cy="50" r="43" fill="url(#tr-spec)" />
  </svg>
);

/** Ocean world with continents and clouds – middle orbit */
const OceanWorld = () => (
  <svg width="36" height="36" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="ow-halo" cx="50%" cy="50%" r="50%">
        <stop offset="62%" stopColor="transparent" />
        <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.6" />
      </radialGradient>
      <radialGradient id="ow-base" cx="32%" cy="27%" r="73%">
        <stop offset="0%" stopColor="#7dd3fc" />
        <stop offset="26%" stopColor="#0284c7" />
        <stop offset="58%" stopColor="#075985" />
        <stop offset="100%" stopColor="#0a1f33" />
      </radialGradient>
      <clipPath id="ow-clip"><circle cx="50" cy="50" r="44" /></clipPath>
      <radialGradient id="ow-spec" cx="28%" cy="22%" r="32%">
        <stop offset="0%" stopColor="white" stopOpacity="0.5" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#ow-halo)" />
    <circle cx="50" cy="50" r="44" fill="url(#ow-base)" />
    <g clipPath="url(#ow-clip)">
      {/* Continents */}
      <ellipse cx="42" cy="37" rx="14" ry="10" fill="#047857" opacity="0.6" transform="rotate(-15 42 37)" />
      <ellipse cx="64" cy="56" rx="11" ry="14" fill="#065f46" opacity="0.55" transform="rotate(20 64 56)" />
      <ellipse cx="34" cy="66" rx="9" ry="6" fill="#047857" opacity="0.45" />
      <ellipse cx="68" cy="32" rx="6" ry="4" fill="#059669" opacity="0.4" />
      {/* Cloud wisps */}
      <ellipse cx="48" cy="30" rx="17" ry="4" fill="white" opacity="0.18" />
      <ellipse cx="70" cy="44" rx="11" ry="3" fill="white" opacity="0.15" />
      <ellipse cx="28" cy="58" rx="13" ry="3.5" fill="white" opacity="0.15" />
      <ellipse cx="55" cy="70" rx="10" ry="3" fill="white" opacity="0.12" />
      {/* Atmosphere limb */}
      <ellipse cx="50" cy="16" rx="30" ry="11" fill="#38bdf8" opacity="0.13" />
    </g>
    <circle cx="50" cy="50" r="44" fill="url(#ow-spec)" />
  </svg>
);

/** Violet gas giant with swirl bands – middle orbit */
const VioletGas = () => (
  <svg width="32" height="32" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="vg-halo" cx="50%" cy="50%" r="50%">
        <stop offset="62%" stopColor="transparent" />
        <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.55" />
      </radialGradient>
      <radialGradient id="vg-base" cx="34%" cy="27%" r="73%">
        <stop offset="0%" stopColor="#ddd6fe" />
        <stop offset="28%" stopColor="#7c3aed" />
        <stop offset="62%" stopColor="#4c1d95" />
        <stop offset="100%" stopColor="#150a2e" />
      </radialGradient>
      <clipPath id="vg-clip"><circle cx="50" cy="50" r="43" /></clipPath>
      <radialGradient id="vg-spec" cx="28%" cy="22%" r="30%">
        <stop offset="0%" stopColor="white" stopOpacity="0.38" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
      <linearGradient id="vg-bl" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#c4b5fd" stopOpacity="0" />
        <stop offset="18%" stopColor="#c4b5fd" stopOpacity="0.32" />
        <stop offset="82%" stopColor="#c4b5fd" stopOpacity="0.32" />
        <stop offset="100%" stopColor="#c4b5fd" stopOpacity="0" />
      </linearGradient>
      <linearGradient id="vg-bd" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#2e1065" stopOpacity="0" />
        <stop offset="18%" stopColor="#2e1065" stopOpacity="0.48" />
        <stop offset="82%" stopColor="#2e1065" stopOpacity="0.48" />
        <stop offset="100%" stopColor="#2e1065" stopOpacity="0" />
      </linearGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#vg-halo)" />
    <circle cx="50" cy="50" r="43" fill="url(#vg-base)" />
    <g clipPath="url(#vg-clip)">
      <rect x="0" y="27" width="100" height="9"  fill="url(#vg-bl)" />
      <rect x="0" y="40" width="100" height="12" fill="url(#vg-bd)" />
      <rect x="0" y="56" width="100" height="8"  fill="url(#vg-bl)" />
      <rect x="0" y="67" width="100" height="11" fill="url(#vg-bd)" />
      {/* Vortex storm */}
      <ellipse cx="36" cy="48" rx="9" ry="5.5" fill="#ede9fe" opacity="0.28" />
      <ellipse cx="36" cy="48" rx="5" ry="3" fill="#f5f3ff" opacity="0.2" />
    </g>
    <circle cx="50" cy="50" r="43" fill="url(#vg-spec)" />
  </svg>
);

/** Bright cyan ice giant with polar cap – middle orbit */
const IceGiant = () => (
  <svg width="28" height="28" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="ig-halo" cx="50%" cy="50%" r="50%">
        <stop offset="60%" stopColor="transparent" />
        <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.6" />
      </radialGradient>
      <radialGradient id="ig-base" cx="30%" cy="25%" r="73%">
        <stop offset="0%" stopColor="#ffffff" />
        <stop offset="18%" stopColor="#a5f3fc" />
        <stop offset="52%" stopColor="#0891b2" />
        <stop offset="100%" stopColor="#062f38" />
      </radialGradient>
      <clipPath id="ig-clip"><circle cx="50" cy="50" r="43" /></clipPath>
      <radialGradient id="ig-spec" cx="26%" cy="20%" r="28%">
        <stop offset="0%" stopColor="white" stopOpacity="0.72" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
      <linearGradient id="ig-band" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#67e8f9" stopOpacity="0" />
        <stop offset="20%" stopColor="#67e8f9" stopOpacity="0.22" />
        <stop offset="80%" stopColor="#67e8f9" stopOpacity="0.22" />
        <stop offset="100%" stopColor="#67e8f9" stopOpacity="0" />
      </linearGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#ig-halo)" />
    <circle cx="50" cy="50" r="43" fill="url(#ig-base)" />
    <g clipPath="url(#ig-clip)">
      {/* Polar ice cap */}
      <ellipse cx="50" cy="14" rx="24" ry="15" fill="white" opacity="0.38" />
      {/* Subtle bands */}
      <rect x="0" y="38" width="100" height="9" fill="url(#ig-band)" />
      <rect x="0" y="54" width="100" height="7" fill="url(#ig-band)" />
      {/* Ice surface cracks */}
      <path d="M34 34 Q50 46 42 62" stroke="#e0f2fe" strokeWidth="1.2" fill="none" strokeOpacity="0.4" />
      <path d="M60 28 Q65 50 57 66" stroke="#cffafe" strokeWidth="1" fill="none" strokeOpacity="0.32" />
      <ellipse cx="62" cy="60" rx="12" ry="8" fill="#a5f3fc" opacity="0.18" />
    </g>
    <circle cx="50" cy="50" r="43" fill="url(#ig-spec)" />
  </svg>
);

/** Large blue gas giant with storm bands – outer orbit */
const GasGiantBlue = () => (
  <svg width="48" height="48" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="ggb-halo" cx="50%" cy="50%" r="50%">
        <stop offset="63%" stopColor="transparent" />
        <stop offset="100%" stopColor="#1d4ed8" stopOpacity="0.65" />
      </radialGradient>
      <radialGradient id="ggb-base" cx="32%" cy="26%" r="75%">
        <stop offset="0%" stopColor="#93c5fd" />
        <stop offset="20%" stopColor="#3b82f6" />
        <stop offset="52%" stopColor="#1e40af" />
        <stop offset="82%" stopColor="#1e3a8a" />
        <stop offset="100%" stopColor="#06102a" />
      </radialGradient>
      <clipPath id="ggb-clip"><circle cx="50" cy="50" r="44" /></clipPath>
      <radialGradient id="ggb-spec" cx="28%" cy="21%" r="33%">
        <stop offset="0%" stopColor="white" stopOpacity="0.42" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
      <linearGradient id="ggb-bl" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#60a5fa" stopOpacity="0" />
        <stop offset="14%" stopColor="#60a5fa" stopOpacity="0.28" />
        <stop offset="86%" stopColor="#60a5fa" stopOpacity="0.28" />
        <stop offset="100%" stopColor="#60a5fa" stopOpacity="0" />
      </linearGradient>
      <linearGradient id="ggb-bd" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#172554" stopOpacity="0" />
        <stop offset="14%" stopColor="#172554" stopOpacity="0.52" />
        <stop offset="86%" stopColor="#172554" stopOpacity="0.52" />
        <stop offset="100%" stopColor="#172554" stopOpacity="0" />
      </linearGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#ggb-halo)" />
    <circle cx="50" cy="50" r="44" fill="url(#ggb-base)" />
    <g clipPath="url(#ggb-clip)">
      <rect x="0" y="23" width="100" height="8"  fill="url(#ggb-bl)" />
      <rect x="0" y="34" width="100" height="13" fill="url(#ggb-bd)" />
      <rect x="0" y="50" width="100" height="8"  fill="url(#ggb-bl)" />
      <rect x="0" y="62" width="100" height="12" fill="url(#ggb-bd)" />
      <rect x="0" y="77" width="100" height="7"  fill="url(#ggb-bl)" />
      {/* Great Storm */}
      <ellipse cx="62" cy="43" rx="13" ry="7" fill="#93c5fd" opacity="0.32" />
      <ellipse cx="62" cy="43" rx="8"  ry="4" fill="#bfdbfe" opacity="0.26" />
      <ellipse cx="62" cy="43" rx="3"  ry="2" fill="white"   opacity="0.22" />
    </g>
    <circle cx="50" cy="50" r="44" fill="url(#ggb-spec)" />
  </svg>
);

/** Reddish desert planet with craters and dust streaks – outer orbit */
const DesertPlanet = () => (
  <svg width="24" height="24" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="dp-halo" cx="50%" cy="50%" r="50%">
        <stop offset="62%" stopColor="transparent" />
        <stop offset="100%" stopColor="#b45309" stopOpacity="0.35" />
      </radialGradient>
      <radialGradient id="dp-base" cx="34%" cy="28%" r="73%">
        <stop offset="0%" stopColor="#fcd34d" />
        <stop offset="24%" stopColor="#b45309" />
        <stop offset="58%" stopColor="#7c2d12" />
        <stop offset="100%" stopColor="#2c0a00" />
      </radialGradient>
      <clipPath id="dp-clip"><circle cx="50" cy="50" r="43" /></clipPath>
      <radialGradient id="dp-spec" cx="28%" cy="22%" r="26%">
        <stop offset="0%" stopColor="white" stopOpacity="0.26" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
      <radialGradient id="dp-crat" cx="38%" cy="32%" r="65%">
        <stop offset="0%" stopColor="#2c0a00" stopOpacity="0.72" />
        <stop offset="100%" stopColor="#2c0a00" stopOpacity="0" />
      </radialGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#dp-halo)" />
    <circle cx="50" cy="50" r="43" fill="url(#dp-base)" />
    <g clipPath="url(#dp-clip)">
      {/* Dust streaks */}
      <ellipse cx="45" cy="55" rx="19" ry="5" fill="#92400e" opacity="0.42" transform="rotate(-10 45 55)" />
      <ellipse cx="64" cy="37" rx="13" ry="4" fill="#78350f" opacity="0.36" transform="rotate(14 64 37)" />
      {/* Craters */}
      <circle cx="37" cy="44" r="10" fill="url(#dp-crat)" />
      <circle cx="37" cy="44" r="10" fill="none" stroke="#fcd34d" strokeWidth="0.7" strokeOpacity="0.14" />
      <circle cx="65" cy="60" r="8" fill="url(#dp-crat)" />
      <circle cx="51" cy="29" r="6" fill="url(#dp-crat)" />
      <circle cx="72" cy="40" r="4" fill="url(#dp-crat)" />
    </g>
    <circle cx="50" cy="50" r="43" fill="url(#dp-spec)" />
  </svg>
);

/** Distant indigo/navy gas dwarf – outer orbit */
const DistantGas = () => (
  <svg width="20" height="20" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="dg-halo" cx="50%" cy="50%" r="50%">
        <stop offset="60%" stopColor="transparent" />
        <stop offset="100%" stopColor="#4f46e5" stopOpacity="0.45" />
      </radialGradient>
      <radialGradient id="dg-base" cx="32%" cy="26%" r="73%">
        <stop offset="0%" stopColor="#a5b4fc" />
        <stop offset="28%" stopColor="#4338ca" />
        <stop offset="63%" stopColor="#312e81" />
        <stop offset="100%" stopColor="#0d0a22" />
      </radialGradient>
      <clipPath id="dg-clip"><circle cx="50" cy="50" r="43" /></clipPath>
      <radialGradient id="dg-spec" cx="28%" cy="22%" r="27%">
        <stop offset="0%" stopColor="white" stopOpacity="0.32" />
        <stop offset="100%" stopColor="white" stopOpacity="0" />
      </radialGradient>
      <linearGradient id="dg-b" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stopColor="#818cf8" stopOpacity="0" />
        <stop offset="20%" stopColor="#818cf8" stopOpacity="0.3" />
        <stop offset="80%" stopColor="#818cf8" stopOpacity="0.3" />
        <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
      </linearGradient>
    </defs>
    <circle cx="50" cy="50" r="49" fill="url(#dg-halo)" />
    <circle cx="50" cy="50" r="43" fill="url(#dg-base)" />
    <g clipPath="url(#dg-clip)">
      <rect x="0" y="30" width="100" height="9"  fill="url(#dg-b)" />
      <rect x="0" y="46" width="100" height="12" fill="url(#dg-b)" />
      <rect x="0" y="62" width="100" height="8"  fill="url(#dg-b)" />
    </g>
    <circle cx="50" cy="50" r="43" fill="url(#dg-spec)" />
  </svg>
);

// ─── Hero ─────────────────────────────────────────────────────────────────────

export function LandingHero() {
  return (
    <div className="relative flex flex-col items-center justify-center w-full py-20">

      {/* Mobile glow */}
      <div className="absolute flex lg:hidden size-40 rounded-full bg-blue-500 blur-[10rem] top-0 left-1/2 -translate-x-1/2 -z-10" />

      {/* Orbit rings — full-viewport, behind content */}
      <Container className="hidden lg:flex absolute inset-0 top-0 mb-auto flex-col items-center justify-center w-full min-h-screen z-0 pointer-events-none">
        {/* Ambient glows */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 60% 40% at 72% 12%, rgba(56,189,248,0.07) 0%, transparent 65%), " +
              "radial-gradient(ellipse 50% 35% at 18% 82%, rgba(99,102,241,0.06) 0%, transparent 60%)",
          }}
        />

        <div className="relative size-[1100px]">
          {/* Orbit 1 – innermost */}
          <OrbitingCircles speed={0.55} radius={210} iconSize={20}>
            <TinyRocky />
          </OrbitingCircles>

          {/* Orbit 2 */}
          <OrbitingCircles speed={0.3} radius={310} iconSize={36} reverse>
            <OceanWorld />
          </OrbitingCircles>

          {/* Orbit 3 */}
          <OrbitingCircles speed={0.18} radius={400} iconSize={32}>
            <VioletGas />
          </OrbitingCircles>

          {/* Orbit 4 – outermost */}
          <OrbitingCircles speed={0.09} radius={500} iconSize={48}>
            <GasGiantBlue />
          </OrbitingCircles>
        </div>
      </Container>

      {/* Content — above orbit layer */}
      <div className="relative z-10 flex flex-col items-center justify-center text-center gap-y-4">

        {/* Badge */}
        <Container className="hidden lg:flex justify-center">
          <button className="group relative grid overflow-hidden rounded-full px-3 py-1.5 shadow-[0_1000px_0_0_hsl(0_0%_10%)_inset] transition-colors duration-200">
            <span>
              <span className="spark mask-gradient absolute inset-0 h-[100%] w-[100%] animate-flip overflow-hidden rounded-full [mask:linear-gradient(white,_transparent_50%)] before:absolute before:aspect-square before:w-[200%] before:rotate-[-90deg] before:animate-rotate before:bg-[conic-gradient(from_0deg,transparent_0_340deg,white_360deg)] before:content-[''] before:[inset:0_auto_auto_50%] before:[translate:-50%_-15%]" />
            </span>
            <span className="backdrop absolute inset-[1px] rounded-full bg-background transition-colors duration-200 group-hover:bg-muted" />
            <span className="z-10 py-0.5 text-sm text-neutral-200 flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-full bg-gradient-to-r from-sky-400 to-primary text-[9px] font-heading font-semibold text-white tracking-wide">
                NEW
              </span>
              AI-Powered Study Platform
            </span>
          </button>
        </Container>

        <Container delay={0.15}>
          <h1 className="text-4xl md:text-5xl lg:text-7xl font-heading font-bold !leading-tight max-w-4xl mx-auto">
            Master Your Studies{" "}
            <br className="hidden lg:block" />
            with{" "}
            <span className="font-subheading italic">Orbit AI</span>
          </h1>
        </Container>

        <Container delay={0.2}>
          <p className="max-w-xl mx-auto mt-2 text-base lg:text-lg text-center text-muted-foreground font-heading">
            Upload your syllabus and notes, generate quizzes, discuss strategy and track your progress.
            Everything you need to ace your exams in one powerful platform.
          </p>
        </Container>

        <Container delay={0.25} className="z-20">
          <div className="flex items-center justify-center mt-6 gap-x-4">
            <Link href="/dashboard">
              <Button size="lg" className="group font-heading">
                Start Learning Free
                <ArrowRightIcon className="size-4 group-hover:translate-x-1 transition-all duration-300" />
              </Button>
            </Link>
            <Link href="#how-it-works">
              <Button size="lg" variant="outline" className="font-heading">
                See How It Works
              </Button>
            </Link>
          </div>
        </Container>

        {/* Stats */}
        <Container delay={0.3}>
          <div className="mt-6 flex flex-wrap justify-center gap-10 md:gap-16">
            {[
              { value: "50K+", label: "Students" },
              { value: "1M+", label: "Documents" },
              { value: "95%", label: "Success Rate" },
              { value: "24/7", label: "AI Support" },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-3xl md:text-4xl font-heading font-bold mb-1">{stat.value}</div>
                <div className="text-xs text-muted-foreground font-heading uppercase tracking-wider">{stat.label}</div>
              </div>
            ))}
          </div>
        </Container>

        {/* Dashboard preview */}
        <Container delay={0.35} className="relative w-full mt-6">
          <div className="relative rounded-xl lg:rounded-[32px] border border-border p-2 backdrop-blur-lg max-w-5xl mx-auto">
            <div className="absolute top-1/4 left-1/2 -z-10 bg-gradient-to-r from-sky-500 to-blue-600 w-1/2 lg:w-3/4 -translate-x-1/2 h-1/4 -translate-y-1/2 blur-[4rem] lg:blur-[10rem] animate-image-glow" />
            <div className="hidden lg:block absolute -top-1/4 left-1/2 -z-20 bg-blue-600 w-1/4 -translate-x-1/2 h-1/4 -translate-y-1/2 blur-[10rem] animate-image-glow" />

            <div className="rounded-lg lg:rounded-[22px] border border-border bg-card overflow-hidden">
              <div className="w-full aspect-video flex flex-col">
                {/* Browser bar */}
                <div className="flex items-center gap-1.5 px-4 py-2.5 bg-muted/50 border-b border-border">
                  <div className="size-2.5 rounded-full bg-red-500/70" />
                  <div className="size-2.5 rounded-full bg-yellow-500/70" />
                  <div className="size-2.5 rounded-full bg-green-500/70" />
                  <div className="ml-4 flex-1 max-w-xs h-5 rounded-md bg-background/60 border border-border/50" />
                </div>
                {/* Placeholder content */}
                <div className="flex flex-1 gap-3 p-4">
                  <div className="hidden md:flex flex-col gap-2 w-36 shrink-0">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <div key={i} className="h-6 rounded-md bg-muted/40" style={{ width: `${70 + (i % 3) * 15}%` }} />
                    ))}
                  </div>
                  <div className="flex-1 flex flex-col gap-3">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {Array.from({ length: 4 }).map((_, i) => (
                        <div key={i} className="h-16 rounded-lg bg-muted/30 border border-border/30" />
                      ))}
                    </div>
                    <div className="flex gap-3 flex-1">
                      <div className="flex-1 rounded-lg bg-muted/20 border border-border/20" />
                      <div className="w-1/3 rounded-lg bg-muted/20 border border-border/20" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-gradient-to-t from-background to-transparent absolute bottom-0 inset-x-0 w-full h-1/2 pointer-events-none" />
        </Container>

      </div>
    </div>
  );
}
