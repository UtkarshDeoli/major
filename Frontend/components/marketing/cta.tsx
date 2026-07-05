"use client";

import Link from "next/link";
import Container from "@/components/global/container";
import { Button } from "@/components/ui/button";
import Particles from "@/components/ui/particles";

const CTA = () => {
    return (
        <div className="relative flex flex-col items-center justify-center w-full py-20">
            <Container className="py-20 max-w-6xl mx-auto">
                <div className="relative flex flex-col items-center justify-center py-12 lg:py-20 px-0 rounded-3xl lg:rounded-[32px] bg-background/20 text-center border border-foreground/20 overflow-hidden">
                    <Particles
                        refresh
                        ease={80}
                        quantity={80}
                        color="#d4d4d4"
                        className="hidden lg:block absolute inset-0 z-0"
                    />
                    <Particles
                        refresh
                        ease={80}
                        quantity={35}
                        color="#d4d4d4"
                        className="block lg:hidden absolute inset-0 z-0"
                    />

                    {/* Clean blue glow beam at top edge */}
                    <div
                        className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none z-10"
                        style={{
                            height: "1px",
                            width: "60%",
                            background: "linear-gradient(90deg, transparent 0%, rgba(85,145,243,0.35) 50%, transparent 100%)",
                        }}
                    />
                    <div
                        className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none z-10"
                        style={{
                            height: "20px",
                            width: "70%",
                            borderRadius: "100%",
                            filter: "blur(6rem)",
                            background: "radial-gradient(ellipse 50% 100% at 50% 50%, rgba(85,145,243,0.75) 0%, transparent 85%)",
                        }}
                    />
                    <div
                        className="absolute top-0 left-1/2 -translate-x-1/2 pointer-events-none z-10"
                        style={{
                            height: "80px",
                            width: "90%",
                            borderRadius: "100%",
                            filter: "blur(8rem)",
                            background: "radial-gradient(ellipse 50% 100% at 50% 50%, rgba(85,145,243,0.35) 0%, transparent 80%)",
                        }}
                    />

                    <h2 className="text-3xl md:text-5xl lg:text-6xl font-heading font-medium !leading-snug">
                        Ready to ace your {" "} <br /> <span className="font-subheading italic">exams</span> ?
                    </h2>
                    <p className="text-sm md:text-lg text-center text-accent-foreground/80 max-w-2xl mx-auto mt-4">
                        Join thousands of students using Orbit to study smarter. Upload your materials, chat with AI, and generate quizzes — all for free.
                    </p>
                    <Link href="/dashboard" className="mt-8">
                        <Button size="lg">
                            Let&apos;s get started
                        </Button>
                    </Link>
                </div>
            </Container>
        </div>
    )
};

export default CTA
