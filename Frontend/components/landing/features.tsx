"use client";

import { FileText, MessageSquare, Search, FileUp, Brain, Zap } from "lucide-react";
import Container from "@/components/global/container";
import { MagicCard } from "@/components/ui/magic-card";

const FEATURES = [
  {
    icon: FileText,
    title: "Document Management",
    description: "Upload and organize your study materials in one centralized location. Support for PDFs, images, and text documents.",
  },
  {
    icon: MessageSquare,
    title: "AI Chat Assistant",
    description: "Get instant answers from our intelligent AI assistant trained on your uploaded materials. Context-aware and always accurate.",
  },
  {
    icon: Search,
    title: "Smart Search",
    description: "Quickly find exactly what you're looking for with powerful semantic search across all your documents.",
  },
  {
    icon: FileUp,
    title: "Easy File Uploads",
    description: "Drag and drop your PDFs, images, and documents. Our system processes everything automatically in seconds.",
  },
  {
    icon: Brain,
    title: "Concept Breakdown",
    description: "Break down complex topics into easily digestible information with AI-powered concept mapping and summaries.",
  },
  {
    icon: Zap,
    title: "Instant Quiz Generation",
    description: "Generate quizzes instantly from your study material to test your knowledge and track progress.",
  },
];

const FORMATS = ["PDF", "DOCX", "JPG", "PNG", "TXT"];

function FeatureCard({ icon: Icon, title, description, delay }: { icon: typeof FileText; title: string; description: string; delay: number }) {
  return (
    <Container delay={delay}>
      <MagicCard
        gradientFrom="#38bdf8"
        gradientTo="#3b82f6"
        gradientColor="rgba(56,189,248,0.08)"
        className="p-6 lg:p-8 rounded-2xl lg:rounded-3xl h-full"
      >
        <div className="flex flex-col gap-4 h-full">
          <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0">
            <Icon className="size-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-heading font-semibold mb-2">{title}</h3>
            <p className="text-sm text-muted-foreground font-heading leading-relaxed">{description}</p>
          </div>
        </div>
      </MagicCard>
    </Container>
  );
}

export function FeaturesSection() {
  const f = FEATURES;

  return (
    <div id="features" className="relative flex flex-col items-center justify-center w-full py-24 px-4 md:px-12 max-w-screen-xl mx-auto">
      <Container>
        <div className="flex flex-col items-center text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug">
            AI-Powered studying <br />made{" "}
            <span className="font-subheading italic">simple</span>
          </h2>
          <p className="text-base md:text-lg text-muted-foreground mt-6 font-heading">
            Transform your exam prep with AI-powered tools. Create quizzes faster, understand content deeper, and make smarter study decisions.
          </p>
        </div>
      </Container>

      {/* Row 1: 3 equal cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 w-full">
        <FeatureCard icon={f[0].icon} title={f[0].title} description={f[0].description} delay={0.1} />
        <FeatureCard icon={f[1].icon} title={f[1].title} description={f[1].description} delay={0.18} />
        <FeatureCard icon={f[2].icon} title={f[2].title} description={f[2].description} delay={0.26} />
      </div>

      {/* Row 2: 2+2+1 columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 w-full mt-4">
        {/* Upload card spans 2 of 5 cols */}
        <Container delay={0.34} className="lg:col-span-2">
          <MagicCard
            gradientFrom="#38bdf8"
            gradientTo="#3b82f6"
            gradientColor="rgba(56,189,248,0.08)"
            className="p-6 lg:p-8 rounded-2xl lg:rounded-3xl h-full"
          >
            <div className="flex flex-col gap-4 h-full">
              <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0">
                <FileUp className="size-5 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-heading font-semibold mb-2">{f[3].title}</h3>
                <p className="text-sm text-muted-foreground font-heading leading-relaxed">{f[3].description}</p>
              </div>
              <div className="mt-auto flex flex-wrap gap-2 pt-4">
                {FORMATS.map((fmt) => (
                  <span key={fmt} className="px-2.5 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-heading">
                    {fmt}
                  </span>
                ))}
              </div>
            </div>
          </MagicCard>
        </Container>

        {/* Concept card spans 2 of 5 cols */}
        <Container delay={0.42} className="lg:col-span-2">
          <MagicCard
            gradientFrom="#38bdf8"
            gradientTo="#3b82f6"
            gradientColor="rgba(56,189,248,0.08)"
            className="p-6 lg:p-8 rounded-2xl lg:rounded-3xl h-full"
          >
            <div className="flex flex-col gap-4 h-full">
              <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0">
                <Brain className="size-5 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-heading font-semibold mb-2">{f[4].title}</h3>
                <p className="text-sm text-muted-foreground font-heading leading-relaxed">{f[4].description}</p>
              </div>
            </div>
          </MagicCard>
        </Container>

        {/* Quiz card spans 1 of 5 cols */}
        <Container delay={0.5} className="lg:col-span-1">
          <MagicCard
            gradientFrom="#38bdf8"
            gradientTo="#3b82f6"
            gradientColor="rgba(56,189,248,0.08)"
            className="p-6 lg:p-8 rounded-2xl lg:rounded-3xl h-full"
          >
            <div className="flex flex-col gap-4 h-full">
              <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0">
                <Zap className="size-5 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-heading font-semibold mb-2">{f[5].title}</h3>
                <p className="text-sm text-muted-foreground font-heading leading-relaxed">{f[5].description}</p>
              </div>
            </div>
          </MagicCard>
        </Container>
      </div>
    </div>
  );
}
