import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { LearnSidebar } from '@/components/learn/sidebar';

export default function LearnLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-4 flex items-center justify-between">
        <Link href="/" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="mr-1 h-4 w-4" /> Back to workspace
        </Link>
      </div>
      <div className="grid gap-8 lg:grid-cols-[260px_1fr]">
        <aside className="lg:sticky lg:top-6 lg:self-start">
          <LearnSidebar />
        </aside>
        <article className="prose prose-invert max-w-none [&_h1]:mb-2 [&_h1]:mt-0 [&_h2]:mt-8 [&_h3]:mt-6 [&_a]:text-sky-300">
          {children}
        </article>
      </div>
    </div>
  );
}
