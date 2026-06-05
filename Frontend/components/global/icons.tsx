import { LucideProps } from "lucide-react";

const Icons = {
    icon: (props: LucideProps) => (
        <svg {...props} width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" stroke="currentColor" strokeWidth="8" fill="none" />
            <path d="M30 50 L45 65 L70 35" stroke="currentColor" strokeWidth="8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    ),
    circle1: (props: LucideProps) => (
        <svg {...props} width="42" height="42" viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M42 21C42 32.598 32.598 42 21 42C9.40202 42 0 32.598 0 21C0 9.40202 9.40202 0 21 0C32.598 0 42 9.40202 42 21ZM1.00838 21C1.00838 32.0411 9.95893 40.9916 21 40.9916C32.0411 40.9916 40.9916 32.0411 40.9916 21C40.9916 9.95893 32.0411 1.00838 21 1.00838C9.95893 1.00838 1.00838 9.95893 1.00838 21Z" fill="currentColor" />
            <circle cx="21" cy="21" r="10" fill="currentColor" />
        </svg>
    ),
    circle2: (props: LucideProps) => (
        <svg {...props} width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="10" cy="10" r="10" fill="currentColor" />
        </svg>
    ),
    logo: (props: LucideProps) => (
        <svg {...props} width="80" height="80" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="80" height="80" rx="16" fill="#3B82F6" />
            <circle cx="40" cy="40" r="25" stroke="white" strokeWidth="4" fill="none" />
            <path d="M28 40 L36 48 L52 32" stroke="white" strokeWidth="4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    ),
    pdf: (props: LucideProps) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
            <polyline points="14 2 14 8 20 8" />
            <path d="M10 13v-1a2 2 0 0 1 4 0v1" />
            <line x1="10" y1="13" x2="10" y2="18" />
            <line x1="14" y1="13" x2="14" y2="18" />
        </svg>
    ),
    chat: (props: LucideProps) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
    ),
    quiz: (props: LucideProps) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
            <path d="M9 12l2 2 4-4" />
        </svg>
    ),
    analytics: (props: LucideProps) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
    ),
    mock: (props: LucideProps) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
        </svg>
    ),
    ai: (props: LucideProps) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" />
            <path d="M9 21h6" />
        </svg>
    ),
    company1: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">IIT Delhi</text>
        </svg>
    ),
    company2: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">IIT Bombay</text>
        </svg>
    ),
    company3: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">IIT Madras</text>
        </svg>
    ),
    company6: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">IIT Kharagpur</text>
        </svg>
    ),
    company7: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">IIT Kanpur</text>
        </svg>
    ),
    company9: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">IIT Roorkee</text>
        </svg>
    ),
    company10: (props: LucideProps) => (
        <svg {...props} width="148" height="48" viewBox="0 0 148 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <text x="10" y="32" fill="currentColor" fontFamily="sans-serif" fontSize="24" fontWeight="bold">BITS Pilani</text>
        </svg>
    ),
};

export default Icons;
