import './globals.css';
import type { Metadata } from 'next';
import { Inter, Instrument_Serif } from 'next/font/google';
import localFont from 'next/font/local';
import { ThemeProvider } from '@/components/providers/theme-provider';
import { Toaster } from '@/components/ui/toaster';
import { cn } from '@/lib/utils';
import { AuthProvider } from '@/lib/context/auth-context';

const heading = localFont({
  src: [
    { path: '../public/fonts/Satoshi-Light.woff2', weight: '300', style: 'normal' },
    { path: '../public/fonts/Satoshi-Regular.woff2', weight: '400', style: 'normal' },
    { path: '../public/fonts/Satoshi-Medium.woff2', weight: '500', style: 'normal' },
    { path: '../public/fonts/Satoshi-Bold.woff2', weight: '700', style: 'normal' },
    { path: '../public/fonts/Satoshi-Black.woff2', weight: '900', style: 'normal' },
  ],
  variable: '--font-heading',
});

const base = Inter({
  subsets: ['latin'],
  variable: '--font-base',
});

const subheading = Instrument_Serif({
  subsets: ['latin'],
  weight: ['400'],
  variable: '--font-subheading',
});

export const metadata: Metadata = {
  title: 'Orbit - AI-Powered Study Platform',
  description: 'Master your studies with AI-powered document analysis, quiz generation, and smart learning tools.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          'min-h-screen bg-background text-foreground antialiased font-heading overflow-x-hidden',
          heading.variable,
          base.variable,
          subheading.variable,
        )}
        suppressHydrationWarning
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <AuthProvider>
            {children}
            <Toaster />
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}