"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface MagicCardProps extends React.HTMLAttributes<HTMLDivElement> {
    gradientFrom?: string;
    gradientTo?: string;
    innerGlowColor?: string;
    gradientColor?: string; // DEPRECATED: kept for backwards compat with landing page
    gradientOpacity?: number; // DEPRECATED
}

export function MagicCard({
    children,
    className,
    gradientFrom = "#38bdf8",
    gradientTo = "#3b82f6",
    innerGlowColor = "rgba(56, 189, 248, 0.08)",
    ...props
}: MagicCardProps) {
    return (
        <div className={cn("group relative flex size-full rounded-md", className)} {...props}>
            {/* Gradient border (1px sharp line) */}
            <div
                className="absolute inset-0 rounded-md opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                style={{
                    background: `linear-gradient(135deg, ${gradientFrom}, ${gradientTo})`,
                }}
            />

            {/* Card background with uniform inner glow on hover */}
            <div
                className="absolute inset-[1px] z-10 rounded-md bg-card transition-all duration-300"
                style={{
                    boxShadow: "inset 0 0 0 0 transparent",
                }}
            />
            <div
                className="absolute inset-[1px] z-10 rounded-md opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                style={{
                    background: innerGlowColor,
                }}
            />

            {/* Content */}
            <div className="relative z-30 w-full">{children}</div>
        </div>
    );
}
