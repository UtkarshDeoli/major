import { Suspense } from "react";
import Navbar from "@/components/marketing/navbar";
import Hero from "@/components/marketing/hero";
import Companies from "@/components/marketing/companies";
import Features from "@/components/marketing/features";
import Analysis from "@/components/marketing/analysis";
import Integration from "@/components/marketing/integration";
import Pricing from "@/components/marketing/pricing";
import LanguageSupport from "@/components/marketing/language-support";
import CTA from "@/components/marketing/cta";
import Footer from "@/components/marketing/footer";
import Wrapper from "@/components/global/wrapper";
import { TestimonialsCarousel } from "@/components/landing/testimonials-carousel";
import { FAQAccordion } from "@/components/landing/faq-accordion";
import Container from "@/components/global/container";

const Divider = () => (
  <div className="w-full max-w-screen-xl mx-auto px-4 md:px-12">
    <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent" />
  </div>
);

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1">
        <Wrapper className="py-20 relative">
          <Hero />
          <Companies />
          <Features />
          <Analysis />
          <Integration />
          <Suspense fallback={<div className="py-20" />}>
            <Pricing />
          </Suspense>
          <LanguageSupport />
          <CTA />
        </Wrapper>

        {
        /* Unique Orbit sections: Testimonials & FAQ */
        }
        
        <Divider />

        <section id="testimonials" className="py-24 px-4 md:px-12 max-w-screen-xl mx-auto w-full">
          <Container>
            <div className="flex flex-col items-center text-center max-w-2xl mx-auto mb-16">
              <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug">
                What <span className="font-subheading italic">students</span> say
              </h2>
              <p className="text-base md:text-lg text-muted-foreground mt-6 font-heading">
                Join thousands of students who have transformed their study habits with Orbit.
              </p>
            </div>
          </Container>
          <TestimonialsCarousel
            testimonials={[
              { name: "Aisha Patel", role: "Medical Student", avatar: "AP", content: "Orbit helped me prepare for my medical entrance exam. The AI chat feature is incredibly helpful for clarifying complex concepts.", rating: 5 },
              { name: "Rahul Sharma", role: "Engineering Student", avatar: "RS", content: "The document upload and search functionality saves me hours of study time. I can quickly find exactly what I need.", rating: 5 },
              { name: "Priya Singh", role: "Commerce Student", avatar: "PS", content: "Quiz generation feature is a game-changer. I can test my knowledge and track my progress effectively.", rating: 5 },
              { name: "Vikram Kumar", role: "Law Student", avatar: "VK", content: "The AI-powered summaries help me review case laws quickly. Best study companion for law exams!", rating: 5 },
              { name: "Ananya Reddy", role: "Science Student", avatar: "AR", content: "Love the smart search feature! It helps me find relevant study material within seconds.", rating: 5 },
              { name: "Sanjay Gupta", role: "CA Student", avatar: "SG", content: "The progress analytics help me identify weak areas. My exam scores improved significantly!", rating: 5 },
            ]}
          />
        </section>

        <Divider />

        <section id="faq" className="py-24 px-4 md:px-12">
          <Container>
            <div className="max-w-4xl mx-auto w-full">
              <div className="flex flex-col items-center text-center max-w-2xl mx-auto mb-16">
                <h2 className="text-2xl md:text-4xl lg:text-5xl font-heading font-medium !leading-snug">
                  Frequently asked <span className="font-subheading italic">questions</span>
                </h2>
                <p className="text-base md:text-lg text-muted-foreground mt-6 font-heading">
                  Have questions? We&apos;ve got answers. If you don&apos;t see yours here, feel free to contact us.
                </p>
              </div>
              <FAQAccordion
                items={[
                  { question: "What types of documents can I upload?", answer: "You can upload PDF files, images (JPG, PNG), and text documents. Our AI will process and index all your study materials." },
                  { question: "How does the AI chat assistance work?", answer: "Our AI assistant answers questions based on your uploaded materials. It understands context and provides accurate, helpful responses." },
                  { question: "Can I use Orbit on mobile devices?", answer: "Yes! Orbit is fully responsive and works great on smartphones, tablets, and desktops." },
                  { question: "What payment methods do you accept?", answer: "We accept all major credit/debit cards, UPI, and net banking. All payments are securely processed." },
                  { question: "Can I cancel my subscription anytime?", answer: "Absolutely! You can cancel your subscription at any time from your account settings. You&apos;ll continue to have access until your billing period ends." },
                  { question: "Is my data secure?", answer: "Yes, we take data security seriously. All your data is encrypted and stored securely. We never share your data with third parties." },
                ]}
                isLoaded={true}
              />
            </div>
          </Container>
        </section>
      </main>

      <Footer />
    </div>
  );
}
