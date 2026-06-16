"use client"

import { Star, Quote } from 'lucide-react'

interface Testimonial {
  name: string
  role: string
  avatar: string
  content: string
  rating: number
}

interface TestimonialsCarouselProps {
  testimonials: Testimonial[]
}

export function TestimonialsCarousel({ testimonials }: TestimonialsCarouselProps) {
  return (
    <div className="relative w-full overflow-hidden">
      {/* Fade gradients on both edges */}
      <div className="absolute left-0 top-0 bottom-0 w-16 md:w-32 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none" />
      <div className="absolute right-0 top-0 bottom-0 w-16 md:w-32 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none" />

      <div className="flex animate-scroll hover:pause">
        {testimonials.map((testimonial, index) => (
          <div
            key={`${testimonial.name}-${index}`}
            className="flex-shrink-0 w-[350px] md:w-[400px] px-4"
          >
            <div className="relative p-6 rounded-2xl bg-card/50 transition-all duration-300 h-full">
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-1">
                    {Array.from({ length: testimonial.rating }).map((_, i) => (
                      <Star key={i} className="h-4 w-4 fill-amber-400 text-amber-400" />
                    ))}
                  </div>
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                    <Quote className="w-4 h-4 text-muted-foreground" />
                  </div>
                </div>
                <p className="text-muted-foreground mb-6 leading-relaxed text-sm line-clamp-4">
                  {testimonial.content}
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center font-semibold text-primary">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <p className="font-semibold text-foreground">{testimonial.name}</p>
                    <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
        {testimonials.map((testimonial, index) => (
          <div
            key={`duplicate-${testimonial.name}-${index}`}
            className="flex-shrink-0 w-[350px] md:w-[400px] px-4"
          >
            <div className="relative p-6 rounded-2xl bg-card/50 transition-all duration-300 h-full">
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-1">
                    {Array.from({ length: testimonial.rating }).map((_, i) => (
                      <Star key={i} className="h-4 w-4 fill-amber-400 text-amber-400" />
                    ))}
                  </div>
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                    <Quote className="w-4 h-4 text-muted-foreground" />
                  </div>
                </div>
                <p className="text-muted-foreground mb-6 leading-relaxed text-sm line-clamp-4">
                  {testimonial.content}
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary/20 to-primary/10 flex items-center justify-center font-semibold text-primary">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <p className="font-semibold text-foreground">{testimonial.name}</p>
                    <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
