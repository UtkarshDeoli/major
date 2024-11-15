import { AuthForm } from '@/components/auth/auth-form'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'

export default function SignupPage() {
  return (
    <div className="h-screen flex flex-col items-center justify-center p-4">
      <Link 
        href="/" 
        className="absolute top-4 left-4 flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Home
      </Link>
      
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">Create an account</h1>
          <p className="text-muted-foreground">Enter your email to create your account</p>
        </div>
        
        <AuthForm type="signup" />
      </div>
    </div>
  )
}