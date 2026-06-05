import { cn } from "@/lib/utils";
import React from "react";

export interface OrbitingCirclesProps extends React.HTMLAttributes<HTMLDivElement> {
    className?: string;
    children?: React.ReactNode;
    reverse?: boolean;
    duration?: number;
    delay?: number;
    radius?: number;
    path?: boolean;
    iconSize?: number;
    speed?: number;
    stroke?: string;
}

export function OrbitingCircles({
    className,
    children,
    reverse,
    duration = 20,
    radius = 160,
    path = true,
    iconSize = 30,
    speed = 1,
    stroke = "currentColor",
    ...props
}: OrbitingCirclesProps) {
    const calculatedDuration = duration / speed;
    return (
        <>
            {path && (
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    version="1.1"
                    className="pointer-events-none absolute inset-0 size-full"
                >
                    <circle
                        style={{ stroke: stroke, strokeWidth: 2, opacity: 0.6 }}
                        strokeDasharray="6 8"
                        cx="50%"
                        cy="50%"
                        r={radius}
                        fill="none"
                    />
                </svg>
            )}
            {React.Children.map(children, (child, index) => {
                const angle = (360 / React.Children.count(children)) * index;
                return (
                    <div
                        style={
                            {
                                "--duration": calculatedDuration,
                                "--radius": radius,
                                "--angle": angle,
                                "--icon-size": `${iconSize}px`,
                                // Center the element on the orbit origin
                                marginTop: `-${iconSize / 2}px`,
                                marginLeft: `-${iconSize / 2}px`,
                            } as React.CSSProperties
                        }
                        className={cn(
                            `absolute top-1/2 left-1/2 flex size-[var(--icon-size)] transform-gpu animate-orbit items-center justify-center rounded-full`,
                            { "[animation-direction:reverse]": reverse },
                            className,
                        )}
                        {...props}
                    >
                        {child}
                    </div>
                );
            })}
        </>
    );
}
