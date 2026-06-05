"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { HomeIcon } from "lucide-react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background px-4">
      <div className="text-center max-w-md">
        <div className="text-7xl font-bold text-destructive mb-4">500</div>
        <h1 className="text-2xl md:text-3xl font-heading font-semibold text-foreground mb-4">
          You have encountered an error
        </h1>
        <p className="text-muted-foreground mb-8">
          Something went wrong on our end. Please try again or return to the home page.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Button variant="outline" size="lg" onClick={reset} className="gap-2">
            Try Again
          </Button>
          <Link href="/">
            <Button size="lg" className="gap-2">
              <HomeIcon className="w-4 h-4" />
              Go to Home Page
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
