"use client";

import { useState } from "react";
import { CheckIcon } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import Container from "@/components/global/container";
import { Button } from "@/components/ui/button";
import { MagicCard } from "@/components/ui/magic-card";
import { cn } from "@/lib/utils";
import Link from "next/link";

type BillPlan = "monthly" | "annually";

const PLANS = [
  {
    id: "basic",
    title: "Basic",
    monthlyPrice: 0,
    annuallyPrice: 0,
    desc: "Get started for free",
    buttonText: "Get Started",
    features: [
      "5 Document Uploads",
      "Basic AI Assistance",
      "Standard Search",
      "Community Support",
    ],
    highlighted: false,
  },
  {
    id: "pro",
    title: "Pro",
    monthlyPrice: 499,
    annuallyPrice: 4490,
    desc: "Perfect for serious students",
    buttonText: "Get Started",
    features: [
      "Unlimited Documents",
      "Advanced AI Assistance",
      "Priority Support",
      "Instant Quiz Generation",
      "Progress Analytics",
    ],
    highlighted: true,
  },
  {
    id: "premium",
    title: "Premium",
    monthlyPrice: 999,
    annuallyPrice: 8990,
    desc: "For advanced exam preparation",
    buttonText: "Get Started",
    features: [
      "Everything in Pro",
      "Custom Study Plans",
      "24/7 Dedicated Support",
      "Personalized Reports",
      "Exam Strategy Sessions",
    ],
    highlighted: false,
  },
];

function formatPrice(value: number) {
  if (value === 0) return "₹0";
  return `₹${value.toLocaleString("en-IN")}`;
}

export function PricingSection() {
  const [billPlan, setBillPlan] = useState<BillPlan>("monthly");

  return (
    <div id="pricing" className="relative flex flex-col items-center justify-center py-24 px-4 md:px-12 w-full">
      <div className="max-w-5xl mx-auto w-full">
        <Container>
          <div className="flex flex-col items-center text-center max-w-2xl mx-auto mb-10">
            <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug">
              Find the right plan that suits{" "}
              <br className="hidden lg:block" />
              <span className="font-subheading italic">your needs</span>
            </h2>
            <p className="text-base md:text-lg text-muted-foreground mt-6 font-heading">
              Select the perfect plan that fits your study needs and budget. All plans include our core features.
            </p>
          </div>
        </Container>

        <Container delay={0.15}>
          <div className="flex items-center justify-center gap-4 mb-10">
            <span className={cn("text-sm font-heading font-medium", billPlan === "monthly" ? "text-foreground" : "text-muted-foreground")}>
              Monthly
            </span>
            <button
              onClick={() => setBillPlan(b => b === "monthly" ? "annually" : "monthly")}
              className="relative rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              aria-label="Toggle billing period"
            >
              <div className="w-12 h-6 rounded-full bg-primary transition-colors" />
              <div
                className={cn(
                  "absolute inline-flex items-center justify-center w-4 h-4 top-1 left-1 rounded-full bg-white transition-transform duration-300",
                  billPlan === "annually" ? "translate-x-6" : "translate-x-0"
                )}
              />
            </button>
            <span className={cn("text-sm font-heading font-medium", billPlan === "annually" ? "text-foreground" : "text-muted-foreground")}>
              Annually
              <span className="ml-2 text-xs text-primary font-heading">Save ~25%</span>
            </span>
          </div>
        </Container>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6">
          {PLANS.map((plan, idx) => (
            <Container key={plan.id} delay={0.1 * idx + 0.2}>
              <MagicCard
                gradientFrom={plan.highlighted ? "#38bdf8" : "#38bdf8"}
                gradientTo={plan.highlighted ? "#3b82f6" : "#3b82f6"}
                gradientColor={plan.highlighted ? "rgba(56,189,248,0.12)" : "rgba(56,189,248,0.06)"}
                className={cn(
                  "rounded-2xl lg:rounded-3xl overflow-hidden h-full",
                  plan.highlighted && "ring-1 ring-primary"
                )}
              >
                <div className="relative flex flex-col h-full p-6 lg:p-8">
                  {plan.highlighted && (
                    <>
                      <div className="absolute inset-x-0 top-1/2 h-16 -rotate-45 bg-primary blur-[6rem] opacity-20 -z-10" />
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-gradient-to-r from-sky-400 to-primary px-4 py-0.5 rounded-full text-[11px] font-heading font-semibold text-white whitespace-nowrap">
                        Most Popular
                      </div>
                    </>
                  )}

                  <div className="pt-4">
                    <h3 className="font-heading font-semibold text-xl text-foreground">{plan.title}</h3>
                    <div className="mt-3 flex items-end gap-1">
                      <AnimatePresence mode="wait">
                        <motion.span
                          key={billPlan + plan.id}
                          initial={{ y: 8, opacity: 0 }}
                          animate={{ y: 0, opacity: 1 }}
                          exit={{ y: -8, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="text-4xl md:text-5xl font-heading font-bold"
                        >
                          {formatPrice(billPlan === "monthly" ? plan.monthlyPrice : plan.annuallyPrice)}
                        </motion.span>
                      </AnimatePresence>
                      <span className="text-sm text-muted-foreground font-heading mb-1">
                        {billPlan === "monthly" ? "/mo" : "/yr"}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground font-heading mt-1">{plan.desc}</p>
                  </div>

                  <div className="mt-6">
                    <Link href="/dashboard" className="block w-full">
                      <Button
                        size="lg"
                        variant={plan.highlighted ? "default" : "outline"}
                        className="w-full font-heading"
                      >
                        {plan.buttonText}
                      </Button>
                    </Link>
                    <div className="h-5 mt-2 overflow-hidden">
                      <AnimatePresence mode="wait">
                        <motion.p
                          key={billPlan}
                          initial={{ y: 12, opacity: 0 }}
                          animate={{ y: 0, opacity: 1 }}
                          exit={{ y: -12, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="text-xs text-center text-muted-foreground font-heading"
                        >
                          {billPlan === "monthly" ? "Billed monthly" : "Billed in one annual payment"}
                        </motion.p>
                      </AnimatePresence>
                    </div>
                  </div>

                  <div className="mt-6 pt-6 border-t border-border/40 flex flex-col gap-3 flex-1">
                    <span className="text-xs font-heading font-semibold text-muted-foreground uppercase tracking-wider">
                      Includes
                    </span>
                    {plan.features.map((feature, i) => (
                      <div key={i} className="flex items-center gap-2.5">
                        <div className="size-4 rounded-full bg-primary/15 flex items-center justify-center flex-shrink-0">
                          <CheckIcon className="size-2.5 text-primary" />
                        </div>
                        <span className="font-heading text-sm">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </MagicCard>
            </Container>
          ))}
        </div>
      </div>
    </div>
  );
}
