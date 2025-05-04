import { ThemeProvider } from '@/components/providers/theme-provider'
import { Toaster } from '@/components/ui/toaster'
import AuthProtection from '@/components/auth/route-protection/auth-protection'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <AuthProtection>
      <div className="min-h-screen">
        {children}
        <Toaster />
      </div>
    </AuthProtection>
  )
}