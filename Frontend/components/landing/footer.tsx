import Link from "next/link";
import Container from "@/components/global/container";

export function LandingFooter() {
  return (
    <footer className="py-12 px-4 md:px-12 border-t border-border/30 bg-background">
      <div className="max-w-screen-xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <p className="font-heading font-bold text-2xl text-foreground mb-1">Orbit</p>
          <p className="text-muted-foreground text-sm font-heading">© 2025 All rights reserved</p>
        </div>
        <div className="flex gap-6">
          {["Terms", "Privacy", "Help"].map((item) => (
            <Link
              key={item}
              href="#"
              className="text-muted-foreground hover:text-foreground transition-colors font-heading text-sm"
            >
              {item}
            </Link>
          ))}
        </div>
      </div>
    </footer>
  );
}
