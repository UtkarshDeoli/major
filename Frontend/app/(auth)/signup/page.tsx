import { AuthForm } from '@/components/auth/auth-form'
import Link from 'next/link'

export default function SignupPage() {
  return (
    <div className="min-h-screen flex">
      <div className="w-full lg:w-1/2 flex flex-col justify-center p-8 lg:p-16 bg-[#0D1520] relative overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute inset-0" style={{
            backgroundImage: `
              linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px'
          }} />
          <div className="absolute inset-0" style={{
            background: 'radial-gradient(ellipse at center, transparent 0%, #0D1520 100%)'
          }} />
        </div>

        <div className="relative z-10 w-full max-w-md mx-auto">
          <Link href="/" className="flex items-center gap-3 mb-12">
            <img src="/logo.png" alt="Orbit Logo" className="w-12 h-12" />
            <span className="font-bold text-2xl text-white">Orbit</span>
          </Link>

          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-3">Create your account</h1>
            <p className="text-gray-400 text-lg">Start your journey to exam success</p>
          </div>

          <AuthForm type="signup" />
        </div>
      </div>

      <div className="hidden lg:flex w-1/2 relative overflow-hidden bg-gradient-to-br from-[#15202B] to-[#0D1520]">
        <div className="absolute inset-0 overflow-hidden">
          {Array.from({ length: 80 }).map((_, i) => (
            <div
              key={i}
              className="absolute bg-white rounded-full animate-pulse"
              style={{
                width: Math.random() * 3 + 1 + 'px',
                height: Math.random() * 3 + 1 + 'px',
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
                opacity: Math.random() * 0.8 + 0.2,
                animationDelay: Math.random() * 3 + 's',
                animationDuration: Math.random() * 2 + 2 + 's'
              }}
            />
          ))}
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={`shooting-${i}`}
              className="absolute w-1 h-1 bg-gradient-to-r from-transparent via-white to-transparent rounded-full opacity-0"
              style={{
                top: Math.random() * 50 + '%',
                left: Math.random() * 60 + '%',
                animationDelay: Math.random() * 10 + i * 5 + 's',
                animationDuration: '2s',
                animationIterationCount: 'infinite'
              }}
            />
          ))}
        </div>

        <div className="absolute inset-0 flex flex-col items-center justify-center p-16 z-10">
          <div className="text-center">
            <img src="/logo.png" alt="Orbit Logo" className="w-32 h-32 mx-auto mb-8" />
            <h1 className="text-5xl font-bold text-white mb-6">Orbit</h1>
            <p className="text-xl text-gray-300 max-w-md mx-auto leading-relaxed">
              Unlock your real potential by practicing with Orbit's mock test generator
            </p>
          </div>
        </div>
      </div>

      <div className="lg:hidden absolute top-0 left-0 right-0 p-4 z-20">
        <Link href="/" className="flex items-center gap-2">
          <img src="/logo.png" alt="Orbit Logo" className="w-10 h-10" />
          <span className="font-bold text-xl text-white">Orbit</span>
        </Link>
      </div>
    </div>
  )
}
