"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import NumberFlow from "@number-flow/react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckIcon, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import Container from "@/components/global/container";
import { subscriptionAPI } from "@/lib/api";
import { useAuth } from "@/lib/context/auth-context";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
import { useRazorpay, openRazorpayCheckout, CheckoutOptions } from "@/components/billing/use-razorpay";

type BillingCycle = "monthly" | "annually";

interface LivePlan {
  plan: string;
  monthly_price: number;
  yearly_price: number;
  limits: Record<string, number | string>;
}

const RESOURCE_LABELS: Record<string, string> = {
  mock_test: "mock tests / month",
  flashcard: "flashcards / month",
  ai_material: "AI summaries / month",
  chat_message: "chat messages / month",
  doc_storage: "document storage",
  class_count: "classes / batches",
};

function formatStorage(bytes: number): string {
  if (!Number.isFinite(bytes)) return "Unlimited";
  if (bytes >= 1024 * 1024 * 1024) return `${Math.round(bytes / (1024 * 1024 * 1024))} GB`;
  if (bytes >= 1024 * 1024) return `${Math.round(bytes / (1024 * 1024))} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function formatLimit(key: string, value: number | string): string {
  if (value === "Infinity" || value === Infinity || value === Number.POSITIVE_INFINITY) return "Unlimited";
  const n = typeof value === "number" ? value : parseFloat(value);
  if (Number.isNaN(n)) return String(value);
  if (key === "doc_storage") return formatStorage(n);
  return `${n}`;
}

function planFeatures(plan: LivePlan): string[] {
  return Object.entries(RESOURCE_LABELS).map(([key, label]) => {
    const limit = formatLimit(key, plan.limits[key]);
    return `${limit} ${label}`;
  });
}

function paiseToRupees(paise: number): number {
  return paise / 100;
}

const Pricing = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, isAuthenticated } = useAuth();
  const { toast } = useToast();
  const { isReady: razorpayReady } = useRazorpay();

  const [billPlan, setBillPlan] = useState<BillingCycle>("monthly");
  const [plans, setPlans] = useState<LivePlan[]>([]);
  const [isLoadingPlans, setIsLoadingPlans] = useState(true);
  const [checkoutPlan, setCheckoutPlan] = useState<string | null>(null);

  const initialCycle = searchParams.get("cycle") as BillingCycle | null;

  useEffect(() => {
    if (initialCycle === "monthly" || initialCycle === "annually") {
      setBillPlan(initialCycle);
    }
  }, [initialCycle]);

  useEffect(() => {
    let cancelled = false;
    setIsLoadingPlans(true);
    subscriptionAPI
      .getPlans()
      .then((data) => {
        if (cancelled) return;
        setPlans(data.plans || []);
      })
      .catch(() => {
        if (cancelled) return;
        // Silent fallback: page still renders with empty plans; free plan always available
        setPlans([]);
      })
      .finally(() => setIsLoadingPlans(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const displayedPlans = useMemo(() => {
    if (!plans.length) return [];
    return plans.map((p) => ({
      ...p,
      title: p.plan.charAt(0).toUpperCase() + p.plan.slice(1),
      id: p.plan,
      monthlyPrice: p.monthly_price,
      annuallyPrice: p.yearly_price,
      desc:
        p.plan === "starter"
          ? "Perfect for students who want to try AI-powered learning."
          : p.plan === "pro"
          ? "Ideal for serious students, coaches, and small tuition centers."
          : "For top performers, coaching chains, and schools that need the full experience.",
      badge: p.plan === "pro" ? "Most Popular" : undefined,
    }));
  }, [plans]);

  const handleSwitch = () => {
    setBillPlan((prev) => (prev === "monthly" ? "annually" : "monthly"));
  };

  const handleSelectPlan = async (planId: string) => {
    if (!isAuthenticated) {
      router.push(`/signup?redirect=/pricing&plan=${planId}&cycle=${billPlan}`);
      return;
    }

    if (planId === "starter") {
      router.push("/dashboard");
      return;
    }

    if (!razorpayReady) {
      toast({ title: "Loading checkout...", description: "Please wait a moment and try again." });
      return;
    }

    setCheckoutPlan(planId);
    try {
      const cycle = billPlan === "annually" ? "yearly" : "monthly";
      const session = await subscriptionAPI.checkout(planId as "pro" | "premium", cycle);

      const options: CheckoutOptions = {
        key: session.key_id,
        amount: session.amount,
        currency: session.currency,
        name: "Orbit",
        description: `${planId} (${billPlan})`,
        order_id: session.order_id,
        prefill: {
          name: user?.name || undefined,
          email: user?.email || undefined,
        },
        theme: { color: "#3b82f6" },
        handler: async (response) => {
          try {
            await subscriptionAPI.verify({
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_subscription_id: response.razorpay_order_id,
              razorpay_signature: response.razorpay_signature,
            });
            toast({ title: "Payment successful", description: "Your subscription is now active." });
            router.push("/billing");
          } catch (error) {
            toast({
              title: "Payment verification failed",
              description: getErrorMessage(error),
              variant: "destructive",
            });
          }
        },
        modal: {
          ondismiss: () => setCheckoutPlan(null),
          escape: true,
          animation: true,
        },
      };

      openRazorpayCheckout(options);
    } catch (error) {
      toast({ title: "Checkout failed", description: getErrorMessage(error), variant: "destructive" });
      setCheckoutPlan(null);
    }
  };

  if (isLoadingPlans) {
    return (
      <div className="relative flex flex-col items-center justify-center max-w-5xl py-20 mx-auto">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div id="pricing" className="relative flex flex-col items-center justify-center max-w-5xl py-20 mx-auto">
      <div className="flex flex-col items-center justify-center max-w-2xl mx-auto">
        <Container>
          <div className="flex flex-col items-center text-center max-w-2xl mx-auto">
            <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug mt-6">
              Find the right plan that suits {" "}
              <br className="hidden lg:block" />{" "}
              <span className="font-subheading italic">your needs</span>
            </h2>
            <p className="text-base md:text-lg text-center text-accent-foreground/80 mt-6">
              Start for free and upgrade when you need more power. Orbit grows with your academic journey.
            </p>
          </div>
        </Container>

        <Container delay={0.2}>
          <div className="flex items-center justify-center space-x-4 mt-6">
            <span className="text-base font-medium">Monthly</span>
            <button onClick={handleSwitch} className="relative rounded-full focus:outline-none">
              <div className="w-12 h-6 transition rounded-full shadow-md outline-none bg-blue-500"></div>
              <div
                className={cn(
                  "absolute inline-flex items-center justify-center w-4 h-4 transition-all duration-500 ease-in-out top-1 left-1 rounded-full bg-white",
                  billPlan === "annually" ? "translate-x-6" : "translate-x-0"
                )}
              />
            </button>
            <span className="text-base font-medium">Annually</span>
          </div>
        </Container>
      </div>

      <div className="grid w-full grid-cols-1 lg:grid-cols-3 pt-8 lg:pt-12 gap-4 lg:gap-6 max-w-5xl mx-auto items-stretch">
        {displayedPlans.map((plan, idx) => (
          <Container key={plan.id} delay={0.1 * idx + 0.2} className="h-full">
            <PlanCard
              plan={plan}
              billPlan={billPlan}
              isBusy={checkoutPlan === plan.id}
              onSelect={() => handleSelectPlan(plan.id)}
            />
          </Container>
        ))}
      </div>
    </div>
  );
};

type DisplayPlan = LivePlan & {
  title: string;
  id: string;
  monthlyPrice: number;
  annuallyPrice: number;
  desc: string;
  badge?: string;
};

const PlanCard = ({
  plan,
  billPlan,
  isBusy,
  onSelect,
}: {
  plan: DisplayPlan;
  billPlan: BillingCycle;
  isBusy: boolean;
  onSelect: () => void;
}) => {
  const isPro = plan.id === "pro";
  const price = billPlan === "monthly" ? plan.monthlyPrice : plan.annuallyPrice;

  return (
    <div
      className={cn(
        "flex flex-col relative rounded-3xl lg:rounded-[32px] transition-all bg-background items-start w-full border border-foreground/10 overflow-hidden h-full",
        isPro && "border-blue-500"
      )}
    >
      {isPro && (
        <div className="absolute top-1/2 inset-x-0 mx-auto h-12 -rotate-45 w-full bg-blue-600 rounded-3xl lg:rounded-[32px] blur-[8rem] -z-10"></div>
      )}

      <div className="p-4 md:p-8 flex rounded-t-3xl lg:rounded-t-[32px] flex-col items-start w-full relative">
        <h2 className="font-medium text-xl text-foreground pt-5">{plan.title}</h2>
        <h3 className="mt-3 text-3xl font-medium md:text-5xl">
          <NumberFlow
            value={paiseToRupees(price)}
            suffix={billPlan === "monthly" ? "/mo" : "/yr"}
            format={{
              currency: "INR",
              style: "currency",
              currencySign: "standard",
              minimumFractionDigits: 0,
              maximumFractionDigits: 0,
              currencyDisplay: "narrowSymbol",
            }}
          />
        </h3>
        <p className="text-sm md:text-base text-muted-foreground mt-2">{plan.desc}</p>
      </div>
      <div className="flex flex-col items-start w-full px-4 py-2 md:px-8">
        <Button
          size="lg"
          variant={isPro ? "blue" : "white"}
          className="w-full"
          onClick={onSelect}
          disabled={isBusy}
        >
          {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : plan.id === "starter" ? "Get Started Free" : `Upgrade to ${plan.title}`}
        </Button>
        <div className="h-8 overflow-hidden w-full mx-auto">
          <AnimatePresence mode="wait">
            <motion.span
              key={billPlan}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="text-sm text-center text-muted-foreground mt-3 mx-auto block"
            >
              {billPlan === "monthly" ? "Billed monthly" : "Billed in one annual payment"}
            </motion.span>
          </AnimatePresence>
        </div>
      </div>
      <div className="flex flex-col items-start w-full p-5 mb-4 ml-1 gap-y-2 flex-1">
        <span className="text-base text-left mb-2">Includes:</span>
        {planFeatures(plan).map((feature, index) => (
          <div key={index} className="flex items-center justify-start gap-2">
            <div className="flex items-center justify-center">
              <CheckIcon className="size-5" />
            </div>
            <span>{feature}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Pricing;
