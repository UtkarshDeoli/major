"use client"

import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface AnalyticsCardProps {
  title?: string
  subtitle?: string
  children: React.ReactNode
  className?: string
  bodyClassName?: string
  delay?: number
  noPadding?: boolean
}

export function AnalyticsCard({
  title,
  subtitle,
  children,
  className,
  bodyClassName,
  delay = 0,
  noPadding = false,
}: AnalyticsCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={cn(
        "rounded-xl border bg-card overflow-hidden transition-shadow hover:shadow-md",
        className
      )}
    >
      {(title || subtitle) && (
        <div className="px-5 pt-5 pb-2">
          {title && <h3 className="text-sm font-semibold tracking-tight">{title}</h3>}
          {subtitle && <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>}
        </div>
      )}
      <div className={cn(noPadding ? "" : "p-5", bodyClassName)}>{children}</div>
    </motion.div>
  )
}
