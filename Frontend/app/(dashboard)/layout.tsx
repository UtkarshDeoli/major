import AuthProtection from '@/components/auth/route-protection/auth-protection'
import AppShell from '@/components/dashboard/app-shell'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <AuthProtection>
      <AppShell>{children}</AppShell>
    </AuthProtection>
  )
}
