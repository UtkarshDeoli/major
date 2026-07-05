"use client"

import { motion } from "framer-motion"
import { LucideIcon } from "lucide-react"

interface StatsCardProps {
  label: string
  value: string | number
  icon: LucideIcon
  trend?: { value: string; positive: boolean }
  subtitle?: string
  delay?: number
  isLoading?: boolean
  color?: string
}

export function StatsCard({ label, value, icon: Icon, trend, subtitle, delay = 0, isLoading, color }: StatsCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className="rounded-xl border bg-card p-5 relative overflow-hidden transition-shadow hover:shadow-md group h-[140px] flex flex-col justify-between"
    >
      <Icon
        className="absolute -right-5 -bottom-5 h-32 w-32 opacity-[0.06] group-hover:opacity-[0.1] transition-opacity"
        style={color ? { color } : undefined}
      />

      <span className="relative text-sm font-medium text-muted-foreground uppercase tracking-wider text-right">
        {label}
      </span>

      <div className="relative flex-1 flex flex-col justify-end"
      >
        <div className="text-5xl font-bold tabular-nums tracking-tighter leading-none"
        >
          {isLoading ? <div className="h-12 w-24 bg-muted animate-pulse rounded mt-1" /> : value}
        </div>
        {trend && (
          <div className={`text-xs flex items-center gap-1 mt-1.5 ${trend.positive ? "text-green-500" : "text-red-400"}`}>
            <span>{trend.value}</span>
            <span className="text-muted-foreground">vs last week</span>
          </div>
        )}
        {subtitle && !trend && (
          <div className="text-xs text-muted-foreground mt-1.5">{subtitle}</div>
        )}
      </div>
    </motion.div>
  )
}