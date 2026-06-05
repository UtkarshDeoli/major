import { FEATURES } from "@/lib/constants/features";
import { cn } from "@/lib/utils";
import Container from "@/components/global/container";
import { MagicCard } from "@/components/ui/magic-card";

const Features = () => {
    return (
        <div id="features" className="relative flex flex-col items-center justify-center w-full py-20">
            <Container>
                <div className="flex flex-col items-center text-center max-w-2xl mx-auto">
                    <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug mt-6">
                        AI-Powered studying <br /> made <span className="font-subheading italic">simple</span>
                    </h2>
                    <p className="text-base md:text-lg text-center text-accent-foreground/80 mt-6">
                        Transform your study routine with AI-powered tools. Upload materials, chat with AI, generate quizzes, and track your progress all in one place.
                    </p>
                </div>
            </Container>

            {/* Bento grid: 3 cols on lg, 2 cols on md */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-8 w-full">
                {FEATURES.map((feature, index) => (
                    <Container
                        key={feature.title}
                        delay={0.1 + index * 0.1}
                        className={cn(
                            "relative flex flex-col",
                            // Row 2 bottom-left wide card
                            index === 3 && "md:col-span-1 lg:col-span-2",
                            // Row 2 bottom-right normal card
                            index === 4 && "md:col-span-1 lg:col-span-1",
                        )}
                    >
                        <MagicCard
                            gradientFrom="#38bdf8"
                            gradientTo="#3b82f6"
                            className="p-5 lg:p-7 rounded-2xl lg:rounded-3xl h-full flex flex-col"
                            gradientColor="rgba(59,130,246,0.1)"
                        >
                            <div className={cn("flex items-center space-x-4 mb-4")}>
                                <div className="flex items-center justify-center size-10 rounded-xl bg-primary/10 shrink-0">
                                    <feature.icon className="size-5 text-primary" />
                                </div>
                                <h3 className="text-lg lg:text-xl font-semibold">
                                    {feature.title}
                                </h3>
                            </div>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                {feature.description}
                            </p>
                        </MagicCard>
                    </Container>
                ))}
            </div>
        </div>
    )
};

export default Features
