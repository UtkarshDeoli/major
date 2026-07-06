import { Suspense } from "react";
import Pricing from "@/components/marketing/pricing";
import Navbar from "@/components/marketing/navbar";
import Footer from "@/components/marketing/footer";

export const metadata = {
  title: "Pricing — Orbit",
  description: "Choose the right Orbit plan for students, coaching centers, schools, and tuition classes.",
};

export default function PricingPage() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      <main className="flex-1">
        <Suspense fallback={<div className="py-20 flex items-center justify-center"></div>}>
          <Pricing />
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
