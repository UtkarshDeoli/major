import {
    BookOpen,
    Brain,
    FileText,
    GraduationCap,
    Zap
} from "lucide-react";

export const FEATURES = [
    {
        title: "AI-Powered Chat",
        description: "Chat with AI about your study materials. Get instant answers, explanations, and clarifications on any topic.",
        icon: Brain,
        image: "/images/feature-chat.svg",
    },
    {
        title: "Smart Document Upload",
        description: "Upload PDFs, notes, and images. Our AI automatically indexes and organizes all your study materials.",
        icon: FileText,
        image: "/images/feature-upload.svg",
    },
    {
        title: "Quiz Generation",
        description: "Generate custom quizzes from your uploaded content. Test your knowledge and track your progress.",
        icon: GraduationCap,
        image: "/images/feature-quiz.svg",
    },
    {
        title: "Progress Analytics",
        description: "Track your study habits, quiz scores, and improvement over time with detailed analytics.",
        icon: Zap,
        image: "/images/feature-analytics.svg",
    },
    {
        title: "Mock Test Creation",
        description: "Create realistic mock tests that simulate your actual exam environment for better preparation.",
        icon: BookOpen,
        image: "/images/feature-mock.svg",
    }
]
