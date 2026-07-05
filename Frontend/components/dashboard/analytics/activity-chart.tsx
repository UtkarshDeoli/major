"use client"

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface ActivityChartProps {
  data: Array<{ day: string; hours: number; quizzes: number }>
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-xl border bg-card px-3 py-2 shadow-lg">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} className="text-sm font-semibold" style={{ color: entry.color }}>
          {entry.name}: {entry.value}
          {entry.name === "Active Time" ? " h" : ""}
        </p>
      ))}
    </div>
  )
}

export function ActivityChart({ data }: ActivityChartProps) {
  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: -8 }}>
          <defs>
            <linearGradient id="activeGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(237, 56%, 60%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(237, 56%, 60%)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="quizGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(160, 84%, 39%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(160, 84%, 39%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 11 }}
            stroke="hsl(var(--muted-foreground))"
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            yAxisId="left"
            tick={{ fontSize: 10 }}
            stroke="hsl(var(--muted-foreground))"
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v}h`}
            label={{ value: "Hours", angle: -90, position: "insideLeft", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fontSize: 10 }}
            stroke="hsl(var(--muted-foreground))"
            axisLine={false}
            tickLine={false}
            label={{ value: "Quizzes", angle: 90, position: "insideRight", fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            yAxisId="left"
            type="monotone"
            dataKey="hours"
            name="Active Time"
            stroke="hsl(237, 56%, 60%)"
            fill="url(#activeGrad)"
            strokeWidth={2}
            animationDuration={800}
            animationEasing="ease-out"
          />
          <Area
            yAxisId="right"
            type="monotone"
            dataKey="quizzes"
            name="Quizzes"
            stroke="hsl(160, 84%, 39%)"
            fill="url(#quizGrad)"
            strokeWidth={2}
            animationDuration={800}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 mt-3">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="h-2 w-2 rounded-full" style={{ background: "hsl(237, 56%, 60%)" }} />
          Active Time
        </div>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="h-2 w-2 rounded-full" style={{ background: "hsl(160, 84%, 39%)" }} />
          Quizzes
        </div>
      </div>
    </div>
  )
}
