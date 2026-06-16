import Link from "next/link";
import { Button } from "@/components/ui/button";
import { HomeIcon } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background px-4">
      <div className="text-center max-w-md">
        <div className="text-7xl font-bold text-primary mb-4">404</div>
        <h1 className="text-2xl md:text-3xl font-sans font-semibold text-foreground mb-4">
          This page does not exist
        </h1>
        <p className="text-muted-foreground mb-8">
          The page you are looking for might have been removed, renamed, or is temporarily unavailable.
        </p>
        <Link href="/">
          <Button size="lg" className="gap-2">
            <HomeIcon className="w-4 h-4" />
            Go to Home Page
          </Button>
        </Link>
      </div>
    </div>
  );
}
