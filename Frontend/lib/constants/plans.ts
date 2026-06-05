export type PLAN = {
    id: string;
    title: string;
    desc: string;
    monthlyPrice: number;
    annuallyPrice: number;
    badge?: string;
    buttonText: string;
    features: string[];
    link: string;
};

export const PLANS: PLAN[] = [
    {
        id: "starter",
        title: "Starter",
        desc: "Perfect for students who want to try AI-powered learning with basic features.",
        monthlyPrice: 0,
        annuallyPrice: 0,
        buttonText: "Get Started Free",
        features: [
            "3 document uploads/month",
            "Basic AI chat (50 msgs/day)",
            "5 quizzes/month",
            "Basic analytics",
            "Email support"
        ],
        link: "/dashboard"
    },
    {
        id: "pro",
        title: "Pro",
        desc: "Ideal for serious students who need unlimited access and advanced features.",
        monthlyPrice: 299,
        annuallyPrice: 2990,
        badge: "Most Popular",
        buttonText: "Upgrade to Pro",
        features: [
            "Unlimited document uploads",
            "Unlimited AI chat",
            "Unlimited quizzes & mock tests",
            "Advanced analytics dashboard",
            "Priority 24/7 support",
            "PDF analysis & summarization"
        ],
        link: "/dashboard"
    },
    {
        id: "premium",
        title: "Premium",
        desc: "For top performers who need the complete Orbit experience.",
        monthlyPrice: 599,
        annuallyPrice: 5990,
        buttonText: "Go Premium",
        features: [
            "Everything in Pro",
            "Custom study plans",
            "Exam strategy sessions",
            "Personalized performance reports",
            "Dedicated account manager",
            "Early access to new features"
        ],
        link: "/dashboard"
    },
];
