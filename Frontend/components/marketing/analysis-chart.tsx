"use client"

import { useEffect, useState, useRef } from "react"

interface BarData {
  label: string
  value: number
  color?: string
}

function AnimatedBar({ value, max, delay, color }: { value: number; max: number; delay: number; color: string }) {
  const [width, setWidth] = useState(0)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setWidth((value / max) * 100), delay)
        }
      },
      { threshold: 0.3 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [value, max, delay])

  return (
    <div ref={ref} className="h-2 rounded-full bg-muted overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700 ease-out"
        style={{ width: `${width}%`, background: color }}
      />
    </div>
  )
}

function MiniLineChart({ data, color }: { data: number[]; color: string }) {
  const [progress, setProgress] = useState(0)
  const ref = useRef<SVGSVGElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setProgress(1), 200)
        }
      },
      { threshold: 0.3 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // hsl(...) strings contain spaces/commas which break url(#...) references and
  // collide if reused, so slug the color into a valid, unique gradient id.
  const gradId = `grad-${color.replace(/[^a-z0-9]/gi, "")}`

  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const w = 200
  const h = 60
  const padding = 4

  const points = data.map((v, i) => {
    const x = padding + (i / (data.length - 1)) * (w - padding * 2)
    const y = h - padding - ((v - min) / range) * (h - padding * 2)
    return `${x},${y}`
  })

  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"}${p}`).join(" ")

  const areaPath = `${pathD} L${padding + ((data.length - 1) / (data.length - 1)) * (w - padding * 2)},${h - padding} L${padding},${h - padding} Z`

  return (
    <svg ref={ref} viewBox={`0 0 ${w} ${h}`} className="w-full h-16 mt-3">
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d={areaPath}
        fill={`url(#${gradId})`}
        style={{ opacity: progress, transition: "opacity 0.6s ease-out 0.3s" }}
      />
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          strokeDasharray: 300,
          strokeDashoffset: 300 * (1 - progress),
          transition: "stroke-dashoffset 1.2s ease-out",
        }}
      />
      {data.map((v, i) => {
        const x = padding + (i / (data.length - 1)) * (w - padding * 2)
        const y = h - padding - ((v - min) / range) * (h - padding * 2)
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r="3"
            fill={color}
            style={{
              opacity: progress,
              transition: `opacity 0.3s ease-out ${0.3 + i * 0.1}s`,
            }}
          />
        )
      })}
    </svg>
  )
}

export function AnalysisBarChart({ data }: { data: BarData[] }) {
  const max = Math.max(...data.map((d) => d.value))
  return (
    <div className="space-y-3 mt-4">
      {data.map((item, i) => (
        <div key={item.label} className="space-y-1.5">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">{item.label}</span>
            <span className="font-semibold tabular-nums">{item.value}%</span>
          </div>
          <AnimatedBar value={item.value} max={max} delay={i * 100 + 200} color={item.color || "hsl(237, 56%, 60%)"} />
        </div>
      ))}
    </div>
  )
}

export function AnalysisLineChart({ data, color, label }: { data: number[]; color: string; label: string }) {
  return (
    <div>
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <div className="h-2 w-2 rounded-full" style={{ background: color }} />
        {label}
      </div>
      <MiniLineChart data={data} color={color} />
    </div>
  )
}