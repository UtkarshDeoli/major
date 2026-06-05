import { Upload, Brain, GraduationCap, TrendingUpIcon } from "lucide-react";
import Container from "@/components/global/container";
import { MagicCard } from "@/components/ui/magic-card";

export function HowItWorksSection() {
  return (
    <div id="how-it-works" className="relative flex flex-col items-center justify-center w-full py-20 px-4 md:px-12 max-w-screen-xl mx-auto">
      <Container>
        <div className="flex flex-col items-center text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug">
            Intelligent study <br /><span className="font-subheading italic">made easy</span>
          </h2>
          <p className="text-base md:text-lg text-muted-foreground mt-4 font-heading">
            Get started in minutes with our simple 3-step process. Your journey to exam success begins here.
          </p>
        </div>
      </Container>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative w-full">

        <Container delay={0.2}>
          <div className="rounded-2xl bg-background/40 relative border border-border/50">
            <MagicCard
              gradientFrom="#38bdf8"
              gradientTo="#3b82f6"
              gradientColor="rgba(59,130,246,0.1)"
              className="p-4 lg:p-8 w-full overflow-hidden"
            >
              <div className="absolute bottom-0 right-0 bg-primary w-1/4 h-1/4 blur-[8rem] z-20 opacity-40" />
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-sky-400 flex items-center justify-center text-white font-heading font-bold text-xl shadow-lg shadow-primary/30 flex-shrink-0">
                    01
                  </div>
                  <h3 className="text-xl font-heading font-semibold">Upload Your Materials</h3>
                </div>
                <p className="text-sm text-muted-foreground font-heading">
                  Drag and drop your PDF documents, notes, or images. Our system accepts various file formats and organizes everything for you automatically.
                </p>
                <div className="flex flex-wrap gap-2">
                  {["PDF", "DOCX", "Images", "Notes"].map((fmt) => (
                    <span key={fmt} className="px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-heading">
                      {fmt}
                    </span>
                  ))}
                </div>

                <div className="space-y-2 pt-2">
                  <div className="grid grid-cols-3 text-sm text-muted-foreground py-2 font-heading">
                    <div>Type</div>
                    <div>Status</div>
                    <div>Size</div>
                  </div>
                  {[
                    { name: "Physics.pdf", status: "Indexed", size: "2.4MB" },
                    { name: "Chemistry.pdf", status: "Indexed", size: "1.8MB" },
                    { name: "Math Notes", status: "Processing", size: "0.9MB" },
                  ].map((doc) => (
                    <div key={doc.name} className="grid grid-cols-3 text-sm py-2 border-t border-border/50 font-heading">
                      <div className="flex items-center gap-1">
                        <Upload className="size-3 text-primary" />
                        {doc.name}
                      </div>
                      <div className={doc.status === "Indexed" ? "text-green-400" : "text-yellow-400"}>{doc.status}</div>
                      <div>{doc.size}</div>
                    </div>
                  ))}
                </div>
              </div>
            </MagicCard>
          </div>
        </Container>

        <Container delay={0.2}>
          <div className="rounded-2xl bg-background/40 relative border border-border/50">
            <MagicCard
              gradientFrom="#38bdf8"
              gradientTo="#3b82f6"
              gradientColor="rgba(59,130,246,0.1)"
              className="p-4 lg:p-8 w-full overflow-hidden"
            >
              <div className="absolute bottom-0 right-0 bg-sky-500 w-1/4 h-1/4 blur-[8rem] z-20 opacity-40" />
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-sky-400 flex items-center justify-center text-white font-heading font-bold text-xl shadow-lg shadow-primary/30 flex-shrink-0">
                    02
                  </div>
                  <h3 className="text-xl font-heading font-semibold">AI Processing</h3>
                </div>
                <p className="text-sm text-muted-foreground font-heading">
                  Our advanced AI analyzes and indexes your content, making it searchable and ready for intelligent chat interactions and quiz generation.
                </p>

                <div className="space-y-3">
                  {[
                    { label: "Content indexed", done: true },
                    { label: "Keywords extracted", done: true },
                    { label: "Concepts mapped", done: true },
                    { label: "Quiz topics identified", done: true },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0" />
                      <span className="text-sm text-muted-foreground font-heading">{item.label}</span>
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-3 gap-3 text-center pt-2">
                  <div>
                    <div className="text-2xl font-heading font-bold text-primary">24/7</div>
                    <div className="text-xs text-muted-foreground font-heading">AI Support</div>
                  </div>
                  <div>
                    <div className="text-2xl font-heading font-bold text-primary">100+</div>
                    <div className="text-xs text-muted-foreground font-heading">Topics</div>
                  </div>
                  <div>
                    <div className="text-2xl font-heading font-bold text-primary">∞</div>
                    <div className="text-xs text-muted-foreground font-heading">Quizzes</div>
                  </div>
                </div>
              </div>
            </MagicCard>
          </div>
        </Container>

        <Container delay={0.3} className="md:col-span-2">
          <div className="rounded-2xl bg-background/40 relative border border-border/50">
            <MagicCard
              gradientFrom="#38bdf8"
              gradientTo="#3b82f6"
              gradientColor="rgba(59,130,246,0.1)"
              className="p-4 lg:p-8 w-full overflow-hidden"
            >
              <div className="absolute bottom-0 right-0 bg-primary w-1/4 h-1/4 blur-[8rem] z-20 opacity-30" />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-sky-400 flex items-center justify-center text-white font-heading font-bold text-xl shadow-lg shadow-primary/30 flex-shrink-0">
                      03
                    </div>
                    <h3 className="text-xl font-heading font-semibold">Learn & Prepare</h3>
                  </div>
                  <p className="text-sm text-muted-foreground font-heading">
                    Chat with AI, generate quizzes, and master your exam topics. Track your progress and improve continuously with personalized insights.
                  </p>
                </div>
                <div className="space-y-3">
                  {[
                    { label: "Documents Processed", value: "24", trend: "+4 this week" },
                    { label: "Study Hours", value: "48h", trend: "+12h this week" },
                    { label: "Quiz Score Average", value: "87%", trend: "+5% improvement" },
                  ].map((stat) => (
                    <div key={stat.label} className="flex justify-between items-center py-2 border-b border-border/50">
                      <span className="text-sm text-muted-foreground font-heading">{stat.label}</span>
                      <div className="text-right">
                        <div className="text-sm font-heading font-semibold">{stat.value}</div>
                        <div className="text-xs text-green-400 flex items-center gap-1 justify-end">
                          <TrendingUpIcon className="size-3" />
                          {stat.trend}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </MagicCard>
          </div>
        </Container>
      </div>
    </div>
  );
}
