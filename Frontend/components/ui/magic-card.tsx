"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface MagicCardProps extends React.HTMLAttributes<HTMLDivElement> {
    gradientSize?: number;
    gradientColor?: string;
    gradientOpacity?: number;
    gradientFrom?: string;
    gradientTo?: string;
}

export function MagicCard({
    children,
    className,
    gradientColor = "rgba(56, 189, 248, 0.15)",
    gradientFrom = "#38bdf8",
    gradientTo = "#3b82f6",
    ...props
}: MagicCardProps) {
    return (
        <div className={cn("group relative flex size-full rounded-xl", className)} {...props}>
            {/* Base card background */}
            <div className="absolute inset-px z-10 rounded-xl bg-card" />

            {/* Content layer */}
            <div className="relative z-30 w-full">{children}</div>

            {/* Full-card felt glow - illuminates entire interior on hover */}
            <div
                className="pointer-events-none absolute inset-px z-10 rounded-xl opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                style={{
                    background: `radial-gradient(circle at 50% 50%, ${gradientColor}, transparent 70%)`,
                }}
            />

            {/* Border glow - static gradient behind card on hover */}
            <div
                className="pointer-events-none absolute -inset-0.5 rounded-xl opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                style={{
                    background: `linear-gradient(135deg, ${gradientFrom}, ${gradientTo})`,
                    filter: "blur(4px)",
                }}
            />
        </div>
    );
}
