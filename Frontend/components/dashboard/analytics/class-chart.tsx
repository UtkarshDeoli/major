"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"

interface ClassChartProps {
  data: Array<{ name: string; score: number }>
}

const COLORS = [
  "hsl(160, 84%, 39%)",
  "hsl(221, 83%, 53%)",
  "hsl(237, 56%, 60%)",
  "hsl(280, 56%, 60%)",
  "hsl(320, 56%, 60%)",
  "hsl(40, 84%, 50%)",
  "hsl(200, 98%, 53%)",
  "hsl(100, 56%, 50%)",
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

export function ClassChart({ data }: ClassChartProps) {
  return (
    <div className="rounded-xl border bg-card p-5 transition-shadow hover:shadow-md">
      <h3 className="text-sm font-semibold mb-4">Class Performance</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -8 }} barCategoryGap="20%">
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
            domain={[0, 100]}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(var(--muted))", radius: 4 }} />
          <Bar dataKey="score" radius={[6, 6, 0, 0]} animationDuration={800} animationEasing="ease-out">
            {data.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}