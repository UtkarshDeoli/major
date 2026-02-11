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
      <div className="flex animate-scroll hover:pause">
        {testimonials.map((testimonial, index) => (
          <div
            key={`${testimonial.name}-${index}`}
            className="flex-shrink-0 w-[350px] md:w-[400px] px-4"
          >
            <div className="relative p-6 rounded-2xl bg-[#192734] border border-blue-500/10 hover:border-blue-500/30 transition-all duration-300 h-full">
              <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-1">
                    {Array.from({ length: testimonial.rating }).map((_, i) => (
                      <Star key={i} className="h-4 w-4 fill-blue-400 text-blue-400" />
                    ))}
                  </div>
                  <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center">
                    <Quote className="w-4 h-4 text-blue-400" />
                  </div>
                </div>
                <p className="text-gray-300 mb-6 leading-relaxed text-sm line-clamp-4">
                  {testimonial.content}
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500/30 to-cyan-500/20 flex items-center justify-center font-semibold text-blue-400">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <p className="font-semibold text-white">{testimonial.name}</p>
                    <p className="text-sm text-blue-300/60">{testimonial.role}</p>
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
            <div className="relative p-6 rounded-2xl bg-[#192734] border border-blue-500/10 hover:border-blue-500/30 transition-all duration-300 h-full">
              <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />
              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-1">
                    {Array.from({ length: testimonial.rating }).map((_, i) => (
                      <Star key={i} className="h-4 w-4 fill-blue-400 text-blue-400" />
                    ))}
                  </div>
                  <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center">
                    <Quote className="w-4 h-4 text-blue-400" />
                  </div>
                </div>
                <p className="text-gray-300 mb-6 leading-relaxed text-sm line-clamp-4">
                  {testimonial.content}
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500/30 to-cyan-500/20 flex items-center justify-center font-semibold text-blue-400">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <p className="font-semibold text-white">{testimonial.name}</p>
                    <p className="text-sm text-blue-300/60">{testimonial.role}</p>
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
