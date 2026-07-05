import React from "react";
import { cn } from "@/lib/utils";

interface BentoGridProps {
  children: React.ReactNode;
  className?: string;
  columns?: 2 | 3 | 4;
}

export function BentoGrid({
  children,
  className,
  columns = 3,
}: BentoGridProps) {
  const columnClasses = {
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
  };

  return (
    <div
      className={cn(
        "grid auto-rows-min gap-4",
        columnClasses[columns],
        className
      )}
    >
      {children}
    </div>
  );
}

interface BentoItemProps {
  children: React.ReactNode;
  className?: string;
  span?: 1 | 2;
}

export function BentoItem({
  children,
  className,
  span = 1,
}: BentoItemProps) {
  return (
    <div
      className={cn(
        "rounded-3xl border bg-card p-6 shadow-sm",
        span === 2 && "sm:col-span-2",
        className
      )}
    >
      {children}
    </div>
  );
}
