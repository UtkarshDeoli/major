"use client"

import { useEffect, useState } from 'react'

type Star = {
  id: number
  size: number
  top: string
  left: string
  delay: number
  duration: number
  opacity: number
  twinkleDelay: number
}

type ShootingStar = {
  id: number
  top: string
  left: string
  delay: number
}

interface StarrySkyProps {
  starCount?: number
  shootingStarCount?: number
  className?: string
  showShootingStars?: boolean
}

export function StarrySky({
  starCount = 100,
  shootingStarCount = 3,
  className = '',
  showShootingStars = true
}: StarrySkyProps) {
  const [isLoaded, setIsLoaded] = useState(false)
  const [stars, setStars] = useState<Star[]>([])
  const [shootingStars, setShootingStars] = useState<ShootingStar[]>([])

  useEffect(() => {
    const generateStars = (): Star[] => {
      return Array.from({ length: starCount }).map((_, i) => ({
        id: i,
        size: Math.random() * 2.5 + 0.5,
        top: `${Math.random() * 100}%`,
        left: `${Math.random() * 100}%`,
        delay: Math.random() * 5,
        duration: Math.random() * 15 + 15,
        opacity: Math.random() * 0.8 + 0.2,
        twinkleDelay: Math.random() * 3
      }))
    }

    const generateShootingStars = (): ShootingStar[] => {
      return Array.from({ length: shootingStarCount }).map((_, i) => ({
        id: i,
        top: `${Math.random() * 50}%`,
        left: `${Math.random() * 60}%`,
        delay: Math.random() * 15 + i * 8
      }))
    }

    setStars(generateStars())
    setShootingStars(generateShootingStars())
    setIsLoaded(true)
  }, [starCount, shootingStarCount])

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {stars.map((star) => (
        <div
          key={`static-${star.id}`}
          className={`absolute rounded-full bg-white ${isLoaded ? 'opacity-100' : 'opacity-0'} absolute-star`}
          style={{
            width: `${star.size}px`,
            height: `${star.size}px`,
            top: star.top,
            left: star.left,
            opacity: star.opacity,
            boxShadow: `0 0 ${star.size * 2}px rgba(255, 255, 255, ${star.opacity * 0.8})`
          }}
        />
      ))}

      {stars.filter((_, i) => i % 3 === 0).map((star) => (
        <div
          key={`twinkle-${star.id}`}
          className={`absolute rounded-full bg-white ${isLoaded ? 'animate-twinkle' : ''}`}
          style={{
            width: `${star.size * 1.5}px`,
            height: `${star.size * 1.5}px`,
            top: star.top,
            left: star.left,
            opacity: isLoaded ? star.opacity * 0.6 : 0,
            animationDelay: `${star.twinkleDelay}s`,
            animationDuration: '3s',
            boxShadow: `0 0 ${star.size * 3}px rgba(255, 255, 255, 0.9)`
          }}
        />
      ))}

      {stars.filter((_, i) => i % 5 === 0).map((star) => (
        <div
          key={`glow-${star.id}`}
          className={`absolute rounded-full bg-white ${isLoaded ? 'animate-pulse-slow' : ''}`}
          style={{
            width: `${star.size * 2}px`,
            height: `${star.size * 2}px`,
            top: star.top,
            left: star.left,
            opacity: isLoaded ? star.opacity * 0.4 : 0,
            animationDelay: `${star.delay}s`,
            animationDuration: '4s',
            boxShadow: `0 0 ${star.size * 6}px rgba(29, 161, 242, 0.8), 0 0 ${star.size * 10}px rgba(29, 161, 242, 0.4)`
          }}
        />
      ))}

      {showShootingStars && shootingStars.map((star) => (
        <div
          key={`shooting-${star.id}`}
          className={`absolute w-1 h-1 bg-gradient-to-r from-transparent via-white to-transparent rounded-full opacity-0 ${isLoaded ? 'animate-shooting-star' : ''}`}
          style={{
            top: star.top,
            left: star.left,
            animationDelay: `${star.delay}s`,
            animationDuration: '2.5s',
            animationIterationCount: 'infinite',
            boxShadow: '0 0 4px rgba(255, 255, 255, 0.8), 0 0 8px rgba(29, 161, 242, 0.6)'
          }}
        />
      ))}

      <div className="absolute inset-0" style={{
        background: 'radial-gradient(ellipse at center, transparent 0%, rgba(13, 21, 32, 0.4) 100%)'
      }} />
    </div>
  )
}
