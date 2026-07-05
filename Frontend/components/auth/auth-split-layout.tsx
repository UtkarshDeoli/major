import Link from "next/link";
import Image from "next/image";
import { AuthForm } from "@/components/auth/auth-form";

interface AuthSplitLayoutProps {
  type: "login" | "signup";
  formTitle: string;
  formSubtitle: string;
}

/**
 * Deterministic PRNG (mulberry32). Using a fixed seed per star means the
 * server-rendered markup and the client-hydrated markup are identical — no
 * Math.random() hydration mismatch.
 */
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const STAR_COUNT = 80;
const SHOOTING_COUNT = 5;

// Computed once with fixed seeds → identical on server and client.
const stars = Array.from({ length: STAR_COUNT }, (_, i) => {
  const rand = mulberry32(i + 1);
  return {
    width: rand() * 3 + 1,
    height: rand() * 3 + 1,
    top: rand() * 100,
    left: rand() * 100,
    opacity: rand() * 0.8 + 0.2,
    animationDelay: rand() * 3,
    animationDuration: rand() * 2 + 2,
  };
});

const shootingStars = Array.from({ length: SHOOTING_COUNT }, (_, i) => {
  const rand = mulberry32(1000 + i);
  return { top: rand() * 50, left: rand() * 60, animationDelay: rand() * 10 + i * 5 };
});

export function AuthSplitLayout({ type, formTitle, formSubtitle }: AuthSplitLayoutProps) {
  return (
    <div className="min-h-screen flex">
      {/* Left: form */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center p-8 lg:p-16 bg-[#0D1520] relative overflow-hidden">
        <div className="absolute inset-0" aria-hidden="true">
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: "50px 50px",
          }} />
          <div className="absolute inset-0" style={{
            background: "radial-gradient(ellipse at center, transparent 0%, #0D1520 100%)",
          }} />
        </div>

        <div className="relative z-10 w-full max-w-md mx-auto">
          <Link href="/" className="flex items-center gap-3 mb-12">
            <Image src="/logo.png" alt="Orbit Logo" width={48} height={48} className="shrink-0" />
            <span className="font-bold text-2xl text-white">Orbit</span>
          </Link>

          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-3">{formTitle}</h1>
            <p className="text-gray-400 text-lg">{formSubtitle}</p>
          </div>

          <AuthForm type={type} />
        </div>
      </div>

      {/* Right: starfield + brand */}
      <div className="hidden lg:flex w-1/2 relative overflow-hidden bg-gradient-to-br from-[#15202B] to-[#0D1520]">
        <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
          {stars.map((s, i) => (
            <div
              key={i}
              className="absolute bg-white rounded-full animate-pulse"
              style={{
                width: `${s.width}px`,
                height: `${s.height}px`,
                top: `${s.top}%`,
                left: `${s.left}%`,
                opacity: s.opacity,
                animationDelay: `${s.animationDelay}s`,
                animationDuration: `${s.animationDuration}s`,
              }}
            />
          ))}
          {shootingStars.map((s, i) => (
            <div
              key={`shooting-${i}`}
              className="absolute w-1 h-1 bg-gradient-to-r from-transparent via-white to-transparent rounded-full opacity-0"
              style={{
                top: `${s.top}%`,
                left: `${s.left}%`,
                animationDelay: `${s.animationDelay}s`,
                animationDuration: "2s",
                animationIterationCount: "infinite",
              }}
            />
          ))}
        </div>

        <div className="absolute inset-0 flex flex-col items-center justify-center p-16 z-10">
          <div className="text-center">
            <Image src="/logo.png" alt="Orbit Logo" width={128} height={128} className="mx-auto mb-8" />
            <h1 className="text-5xl font-bold text-white mb-6">Orbit</h1>
            <p className="text-xl text-gray-300 max-w-md mx-auto leading-relaxed">
              Unlock your real potential by practicing with Orbit&apos;s mock test generator
            </p>
          </div>
        </div>
      </div>

      {/* Mobile logo */}
      <div className="lg:hidden absolute top-0 left-0 right-0 p-4 z-20">
        <Link href="/" className="flex items-center gap-2">
          <Image src="/logo.png" alt="Orbit Logo" width={40} height={40} />
          <span className="font-bold text-xl text-white">Orbit</span>
        </Link>
      </div>
    </div>
  );
}