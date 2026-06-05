"use client";

import { useRef } from "react";
import { ArrowRightIcon, FileText, MessageSquare, CircleCheck, BarChart3, FileCheck, Brain } from "lucide-react";
import { motion, useInView } from "framer-motion";
import Link from "next/link";
import Container from "@/components/global/container";
import { Button } from "@/components/ui/button";
import Ripple from "@/components/ui/ripple";

/* ─── Data with sizes for perspective effect ─── */
const TOOLS = [
  { Icon: FileText, label: "PDF", size: "sm" as const, wave: 3 },
  { Icon: MessageSquare, label: "Chat", size: "md" as const, wave: 2 },
  { Icon: CircleCheck, label: "Quiz", size: "lg" as const, wave: 1 },
  { Icon: BarChart3, label: "Analytics", size: "lg" as const, wave: 1 },
  { Icon: FileCheck, label: "Mock", size: "md" as const, wave: 2 },
  { Icon: Brain, label: "AI", size: "sm" as const, wave: 3 },
];

/* ─── Size configs — dramatic perspective gap ─── */
const SIZE_MAP = {
  sm: { bubble: "w-9 h-9", icon: "w-3.5 h-3.5" },
  md: { bubble: "w-12 h-12", icon: "w-5 h-5" },
  lg: { bubble: "w-16 h-16", icon: "w-7 h-7" },
} as const;

/* ─── Animation variants ─── */
const containerVariants = {
  hidden: {},
  visible: {},
};

const logoVariants = {
  hidden: { opacity: 0, scale: 0.5 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.6, ease: "easeOut" as const },
  },
};

const iconVariants = {
  hidden: { opacity: 0, scale: 0.3, y: 20 },
  visible: (wave: number) => ({
    opacity: 1,
    scale: 1,
    y: 0,
    transition: {
      delay: wave * 0.35 + 0.3,
      duration: 0.5,
      ease: "backOut" as const,
    },
  }),
};

/* ─── Tool bubble with perspective sizing & label ─── */
const ToolBubble = ({
  Icon,
  label,
  size,
}: {
  Icon: React.ElementType;
  label: string;
  size: "sm" | "md" | "lg";
}) => {
  const s = SIZE_MAP[size];
  return (
    <div className="relative flex flex-col items-center cursor-default">
      <div
        className={`${s.bubble} rounded-full bg-[#1a1a24] border border-white/[0.12] flex items-center justify-center transition-all duration-300 hover:scale-110 hover:border-white/30 hover:bg-[#252532]`}
      >
        <Icon
          className={`${s.icon} text-white/80 hover:text-white transition-colors`}
          strokeWidth={1.5}
        />
      </div>
      <span className="absolute top-full mt-1.5 text-[10px] font-medium text-white/50 uppercase tracking-wider whitespace-nowrap">
        {label}
      </span>
    </div>
  );
};

const Integration = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <div id="how-it-works" className="relative flex flex-col items-center justify-center w-full py-20">
      {/* Heading */}
      <Container className="mb-8">
        <div className="flex flex-col items-center text-center">
          <h2 className="text-3xl md:text-5xl lg:text-6xl font-heading font-semibold !leading-tight">
            All Study Tools in One Place
          </h2>
        </div>
      </Container>

      {/* Desktop: scroll-triggered staggered reveal */}
      <div ref={ref} className="hidden lg:block w-full">
        <motion.div
          className="relative mx-auto max-w-5xl h-[420px] flex items-center justify-center"
          variants={containerVariants}
          initial="hidden"
          animate={isInView ? "visible" : "hidden"}
        >
          {/* Ripple background — fades in with logo */}
          <motion.div
            className="absolute inset-0 flex items-center justify-center pointer-events-none"
            variants={{
              hidden: { opacity: 0 },
              visible: { opacity: 1, transition: { delay: 0.1, duration: 0.8, ease: "easeOut" as const } },
            }}
          >
            <Ripple
              mainCircleSize={150}
              mainCircleOpacity={0.14}
              numCircles={6}
              className="[mask-image:none]"
            />
          </motion.div>

          {/* Content row: icons | hub | icons */}
          <div className="relative z-10 flex items-center gap-6">
            {/* Left icons */}
            <div className="flex items-center gap-5">
              {TOOLS.slice(0, 3).map((t, i) => (
                <motion.div
                  key={`l-${i}`}
                  custom={t.wave}
                  variants={iconVariants}
                  initial="hidden"
                  animate={isInView ? "visible" : "hidden"}
                >
                  <ToolBubble Icon={t.Icon} label={t.label} size={t.size} />
                </motion.div>
              ))}
            </div>

            {/* Center hub */}
            <motion.div
              variants={logoVariants}
              initial="hidden"
              animate={isInView ? "visible" : "hidden"}
            >
              <img
                src="/logo.png"
                alt="Orbit"
                className="w-[88px] h-[88px] object-contain bg-transparent block"
                style={{ filter: "drop-shadow(0 0 40px rgba(59,130,246,0.5))" }}
              />
            </motion.div>

            {/* Right icons */}
            <div className="flex items-center gap-5">
              {TOOLS.slice(3, 6).map((t, i) => (
                <motion.div
                  key={`r-${i}`}
                  custom={t.wave}
                  variants={iconVariants}
                  initial="hidden"
                  animate={isInView ? "visible" : "hidden"}
                >
                  <ToolBubble Icon={t.Icon} label={t.label} size={t.size} />
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Mobile */}
      <Container className="lg:hidden w-full">
        <div className="flex items-center justify-center gap-4 flex-wrap max-w-md mx-auto">
          {TOOLS.map((t, i) => (
            <div key={i} className="flex flex-col items-center gap-2">
              <div className="w-11 h-11 rounded-full bg-[#1a1a24] border border-white/[0.12] flex items-center justify-center">
                <t.Icon className="w-5 h-5 text-white/70" strokeWidth={1.5} />
              </div>
              <span className="text-[10px] font-medium text-white/50 uppercase tracking-wider">
                {t.label}
              </span>
            </div>
          ))}
        </div>
      </Container>

      {/* CTA */}
      <Container delay={0.3} className="mt-12">
        <div className="flex flex-col items-center">
          <Link href="/dashboard">
            <Button
              variant="outline"
              size="lg"
              className="bg-white text-black hover:bg-white/90 border-0 rounded-xl px-8"
            >
              Explore All Tools
              <ArrowRightIcon className="w-4 h-4 ml-2" />
            </Button>
          </Link>
        </div>
      </Container>
    </div>
  );
};

export default Integration;
