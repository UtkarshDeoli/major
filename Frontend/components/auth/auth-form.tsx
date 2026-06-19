"use client"

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/hooks/use-toast'
import { Eye, EyeOff, Mail, Lock, User } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { useAuth } from '@/lib/context/auth-context'
import { getErrorMessage } from '@/lib/errors'

interface AuthFormProps {
  type: 'login' | 'signup'
}

export function AuthForm({ type }: AuthFormProps) {
  const { toast } = useToast()
  const router = useRouter()
  const { login, signup } = useAuth()
  const [isLoading, setIsLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      if (type === 'login') {
        await login(email, password);
      } else {
        await signup({
          email,
          password,
          name: name || undefined,
        });
      }
      
      toast({
        title: type === 'login' ? 'Welcome back!' : 'Account created!',
        description: "You're now part of Orbit.",
      });
    } catch (error) {
      console.error('Authentication error:', error);
      toast({
        title: 'Authentication failed',
        description: getErrorMessage(error),
        variant: 'destructive'
      });
    } finally {
      setIsLoading(false)
    }
  };

  const handleGoogleSignIn = () => {
    toast({
      title: 'Google Sign In',
      description: 'Google authentication coming soon!',
    });
  };
  
  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Google Sign In Button */}
      <Button
        type="button"
        variant="outline"
        onClick={handleGoogleSignIn}
        className="w-full h-12 rounded-md bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300"
      >
        <svg className="w-5 h-5 mr-3" viewBox="0 0 24 24">
          <path
            fill="currentColor"
            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
          />
          <path
            fill="currentColor"
            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
          />
          <path
            fill="currentColor"
            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
          />
          <path
            fill="currentColor"
            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
          />
        </svg>
        <span className="text-gray-300 font-medium">Continue with Google</span>
      </Button>

      {/* Divider */}
      <div className="relative my-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-white/10" />
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-4 bg-[#0a0a1a] text-gray-500">or continue with email</span>
        </div>
      </div>

      {/* Name field for signup */}
      {type === 'signup' && (
        <div className="space-y-2">
          <Label htmlFor="name" className="text-gray-400 text-sm font-medium">Full Name</Label>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <User className="h-5 w-5 text-gray-500" />
            </div>
            <Input 
              id="name" 
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter your full name" 
              required 
              disabled={isLoading}
              className="h-12 rounded-md bg-[#1a1a30] border-white/10 pl-12 text-white placeholder:text-gray-500 focus:border-purple-500/50 focus:ring-purple-500/20"
            />
          </div>
        </div>
      )}

      {/* Email field */}
      <div className="space-y-2">
        <Label htmlFor="email" className="text-gray-400 text-sm font-medium">Email</Label>
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Mail className="h-5 w-5 text-gray-500" />
          </div>
          <Input 
            id="email" 
            type="email" 
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email" 
            required 
            disabled={isLoading}
            className="h-12 rounded-md bg-[#1a1a30] border-white/10 pl-12 text-white placeholder:text-gray-500 focus:border-purple-500/50 focus:ring-purple-500/20"
          />
        </div>
      </div>
      
      {/* Password field */}
      <div className="space-y-2">
        <Label htmlFor="password" className="text-gray-400 text-sm font-medium">Password</Label>
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Lock className="h-5 w-5 text-gray-500" />
          </div>
          <Input 
            id="password" 
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password" 
            required 
            disabled={isLoading}
            className="h-12 rounded-md bg-[#1a1a30] border-white/10 pl-12 pr-12 text-white placeholder:text-gray-500 focus:border-purple-500/50 focus:ring-purple-500/20"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-1 top-1/2 -translate-y-1/2 h-10 w-10 text-gray-500 hover:text-white hover:bg-transparent"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? (
              <EyeOff className="h-5 w-5" />
            ) : (
              <Eye className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>

      {/* Forgot password link for login */}
      {type === 'login' && (
        <div className="flex justify-end">
          <Link href="/forgot-password" className="text-sm text-purple-400 hover:text-purple-300 transition-colors">
            Forgot password?
          </Link>
        </div>
      )}
      
      {/* Submit Button */}
      <Button 
        type="submit" 
        className="w-full h-12 rounded-md bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 border-0 shadow-lg shadow-purple-500/25 text-white font-semibold text-lg transition-all duration-300"
        disabled={isLoading}
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
            {type === 'login' ? 'Signing in...' : 'Creating account...'}
          </span>
        ) : (
          type === 'login' ? 'Sign In' : 'Create Account'
        )}
      </Button>
      
      {/* Social proof */}
      <p className="text-center text-gray-500 text-sm">
        {type === 'login' ? (
          <>
            Don&apos;t have an account?{' '}
            <Link href="/signup" className="text-purple-400 hover:text-purple-300 font-medium transition-colors">
              Sign up
            </Link>
          </>
        ) : (
          <>
            Already have an account?{' '}
            <Link href="/login" className="text-purple-400 hover:text-purple-300 font-medium transition-colors">
              Sign in
            </Link>
          </>
        )}
      </p>

      {/* Terms */}
      {type === 'signup' && (
        <p className="text-center text-gray-500 text-xs">
          By creating an account, you agree to our{' '}
          <Link href="/terms" className="text-purple-400 hover:text-purple-300 transition-colors">
            Terms of Service
          </Link>{' '}
          and{' '}
          <Link href="/privacy" className="text-purple-400 hover:text-purple-300 transition-colors">
            Privacy Policy
          </Link>
        </p>
      )}
    </form>
  )
}
