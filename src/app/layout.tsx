import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { SiteNav, SiteFooter } from '@/components/site-nav';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const jetbrains = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains' });

export const metadata: Metadata = {
  title: 'Neural Nets — Build, train, and understand neural networks',
  description:
    'An interactive guide to neural networks. Linear algebra, calculus, backpropagation, and a live training playground — all in your browser.',
  keywords: [
    'neural networks',
    'machine learning',
    'deep learning',
    'education',
    'linear algebra',
    'calculus',
    'backpropagation',
    'gradient descent',
    'visualization',
    'TypeScript',
    'Next.js',
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrains.variable} dark`}
      suppressHydrationWarning
    >
      <body className="flex min-h-screen flex-col font-sans antialiased">
        <div className="grid-bg fixed inset-0 -z-10 opacity-40" aria-hidden />
        <SiteNav />
        <div className="flex-1">{children}</div>
        <SiteFooter />
      </body>
    </html>
  );
}
