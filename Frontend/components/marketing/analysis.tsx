import { BookOpen, Brain, FileText, TrendingUpIcon } from "lucide-react";
import Container from "@/components/global/container";
import { MagicCard } from "@/components/ui/magic-card";

const Analysis = () => {
    return (
        <div className="relative flex flex-col items-center justify-center w-full py-20">
            <Container>
                <div className="flex flex-col items-center text-center max-w-3xl mx-auto mb-16">
                    <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug">
                        Intelligent study  <br /><span className="font-subheading italic">dashboard</span>
                    </h2>
                    <p className="text-base md:text-lg text-accent-foreground/80 mt-4">
                        Gain detailed insights into your study performance and learning progress with our advanced analytics tools.
                    </p>
                </div>
            </Container>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative w-full">

                <Container delay={0.2}>
                    <div className="rounded-2xl bg-background/40 relative">
                        <MagicCard
                            gradientFrom="#38bdf8"
                            gradientTo="#3b82f6"
                            gradientColor="rgba(59,130,246,0.1)"
                            className="p-4 lg:p-8 w-full overflow-hidden"
                        >
                            <div className="absolute bottom-0 right-0 bg-blue-500 w-1/4 h-1/4 blur-[8rem] z-20"></div>
                            <div className="space-y-4">
                                <h3 className="text-xl font-semibold">
                                    Study Insights
                                </h3>
                                <p className="text-sm text-muted-foreground">
                                    Track your learning performance with data-driven insights.
                                </p>

                                <div className="space-y-4">
                                    <div className="flex justify-between items-baseline">
                                        <div>
                                            <div className="text-3xl font-semibold">
                                                87%
                                            </div>
                                            <div className="text-sm text-green-500 flex items-center gap-1 mt-2">
                                                <TrendingUpIcon className="w-4 h-4" />
                                                +12% from last week
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <div className="grid grid-cols-4 text-sm text-muted-foreground py-2">
                                            <div>Subject</div>
                                            <div>Status</div>
                                            <div>Quizzes</div>
                                            <div>Score</div>
                                        </div>
                                        {[
                                            { name: "Physics", status: "Active", quizzes: "24", score: "92%" },
                                            { name: "Chemistry", status: "Active", quizzes: "18", score: "85%" },
                                            { name: "Math", status: "Done", quizzes: "32", score: "88%" },
                                        ].map((subject) => (
                                            <div key={subject.name} className="grid grid-cols-4 text-sm py-2 border-t border-white/5">
                                                <div>{subject.name}</div>
                                                <div>{subject.status}</div>
                                                <div>{subject.quizzes}</div>
                                                <div className="font-semibold">{subject.score}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </MagicCard>
                    </div>
                </Container>

                <Container delay={0.2}>
                    <div className="rounded-2xl bg-background/40 relative">
                        <MagicCard
                            gradientFrom="#38bdf8"
                            gradientTo="#3b82f6"
                            gradientColor="rgba(59,130,246,0.1)"
                            className="p-4 lg:p-8 w-full overflow-hidden"
                        >
                            <div className="absolute bottom-0 right-0 bg-sky-500 w-1/4 h-1/4 blur-[8rem] z-20"></div>
                            <div className="space-y-4">
                                <h3 className="text-xl font-semibold">
                                    Learning Metrics
                                </h3>
                                <p className="text-sm text-muted-foreground">
                                    Understand your study habits and engagement patterns.
                                </p>

                                <div className="space-y-4">
                                    <div className="flex justify-between items-baseline">
                                        <div>
                                            <div className="text-3xl font-semibold">48h</div>
                                            <div className="text-sm text-green-500 flex items-center gap-1 mt-2">
                                                <TrendingUpIcon className="w-4 h-4" />
                                                +8% study time
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <div className="grid grid-cols-4 text-sm text-muted-foreground py-2">
                                            <div>Activity</div>
                                            <div>Hours</div>
                                            <div>Documents</div>
                                            <div>Quizzes</div>
                                        </div>
                                        {[
                                            { activity: "Chat", hours: "18h", docs: "12", quizzes: "8" },
                                            { activity: "Quiz", hours: "15h", docs: "5", quizzes: "24" },
                                            { activity: "Read", hours: "15h", docs: "8", quizzes: "0" },
                                        ].map((metric) => (
                                            <div key={metric.activity} className="grid grid-cols-4 text-sm py-2 border-t border-white/5">
                                                <div>{metric.activity}</div>
                                                <div>{metric.hours}</div>
                                                <div>{metric.docs}</div>
                                                <div className="font-semibold">{metric.quizzes}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </MagicCard>
                    </div>
                </Container>
            </div>
        </div>
    )
};

export default Analysis;
