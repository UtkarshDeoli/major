export type PLAN = {
    id: string;
    title: string;
    desc: string;
    monthlyPrice: number; // INR paise (backend currency)
    annuallyPrice: number; // INR paise
    badge?: string;
    buttonText: string;
    features: string[];
    link: string;
};

// Fallback static skeleton. The canonical data is fetched from GET /subscriptions/plans.
export const PLANS: PLAN[] = [
    {
        id: "starter",
        title: "Starter",
        desc: "Perfect for students who want to try AI-powered learning with basic features.",
        monthlyPrice: 0,
        annuallyPrice: 0,
        buttonText: "Get Started Free",
        features: [
            "3 mock tests / month",
            "50 flashcards / month",
            "5 AI summaries / month",
            "100 chat messages / month",
            "50 MB document storage",
            "1 class / batch",
            "Email support"
        ],
        link: "/signup"
    },
    {
        id: "pro",
        title: "Pro",
        desc: "Ideal for serious students and small coaching centers who need unlimited access.",
        monthlyPrice: 29900,
        annuallyPrice: 299000,
        badge: "Most Popular",
        buttonText: "Upgrade to Pro",
        features: [
            "50 mock tests / month",
            "500 flashcards / month",
            "50 AI summaries / month",
            "1,000 chat messages / month",
            "1 GB document storage",
            "10 classes / batches",
            "Priority 24/7 support",
            "PDF analysis & summarization"
        ],
        link: "/pricing"
    },
    {
        id: "premium",
        title: "Premium",
        desc: "For top performers, coaching chains, and schools that need the full Orbit experience.",
        monthlyPrice: 59900,
        annuallyPrice: 599000,
        buttonText: "Go Premium",
        features: [
            "Unlimited mock tests & AI chat",
            "Unlimited flashcards & summaries",
            "10 GB document storage",
            "Unlimited classes / batches",
            "Custom study plans",
            "Exam strategy sessions",
            "Dedicated account manager",
            "Early access to new features"
        ],
        link: "/pricing"
    },
];
