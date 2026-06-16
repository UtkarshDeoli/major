"use client"

import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from "recharts"

interface WeaknessRadarProps {
  data: Array<{ topic: string; score: number; fullMark: number }>
}

export function WeaknessRadar({ data }: WeaknessRadarProps) {
  return (
    <div className="rounded-md border bg-card p-4">
      <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Weakness Map</h3>
      <ResponsiveContainer width="100%" height={250}>
        <RadarChart data={data}>
          <PolarGrid stroke="hsl(var(--border))" />
          <PolarAngleAxis dataKey="topic" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
          <Radar name="Score" dataKey="score" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}