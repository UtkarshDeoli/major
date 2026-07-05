"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { useState } from "react"

interface SubjectChartProps {
  data: Array<{ subject: string; score: number }>
}

const COLORS = [
  "hsl(221, 83%, 53%)",
  "hsl(237, 56%, 60%)",
  "hsl(200, 98%, 53%)",
  "hsl(260, 56%, 60%)",
  "hsl(280, 56%, 60%)",
  "hsl(320, 56%, 60%)",
]

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-xl border bg-card px-3 py-2 shadow-lg">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold">{payload[0].value}%</p>
    </div>
  )
}

export function SubjectChart({ data }: SubjectChartProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  // Clamp display values to the 0-100 range so the chart axis stays meaningful.
  const normalized = data.map((d) => ({ ...d, score: Math.max(0, Math.min(100, d.score)) }))

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={normalized} margin={{ top: 8, right: 8, bottom: 8, left: -8 }} barCategoryGap="22%">
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="subject"
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
            axisLine={false}
            tickLine={false}
            interval={0}
            angle={-12}
            textAnchor="end"
            height={40}
          />
          <YAxis
            tick={{ fontSize: 10 }}
            stroke="hsl(var(--muted-foreground))"
            domain={[0, 100]}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(var(--muted))", radius: 4 }} />
          <Bar dataKey="score" radius={[6, 6, 0, 0]} animationDuration={800} animationEasing="ease-out">
            {data.map((_, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
                opacity={activeIndex === null || activeIndex === index ? 1 : 0.4}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 mt-3">
        {data.map((item, index) => (
          <button
            key={item.subject}
            onMouseEnter={() => setActiveIndex(index)}
            onMouseLeave={() => setActiveIndex(null)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <span className="h-2 w-2 rounded-full" style={{ background: COLORS[index % COLORS.length] }} />
            {item.subject}
          </button>
        ))}
      </div>
    </div>
  )
}