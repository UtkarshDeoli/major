"use client";

import { useEffect, useRef, useState } from "react";

declare global {
  interface Window {
    Razorpay?: any;
  }
}

const RAZORPAY_SCRIPT = "https://checkout.razorpay.com/v1/checkout.js";

export function useRazorpay() {
  const [isReady, setIsReady] = useState(false);
  const loadPromiseRef = useRef<Promise<boolean> | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.Razorpay) {
      setIsReady(true);
      return;
    }

    if (!loadPromiseRef.current) {
      loadPromiseRef.current = new Promise((resolve) => {
        const existing = document.querySelector(`script[src="${RAZORPAY_SCRIPT}"]`);
        if (existing) {
          existing.addEventListener("load", () => resolve(true));
          existing.addEventListener("error", () => resolve(false));
          return;
        }

        const script = document.createElement("script");
        script.src = RAZORPAY_SCRIPT;
        script.async = true;
        script.defer = true;
        script.onload = () => resolve(true);
        script.onerror = () => resolve(false);
        document.body.appendChild(script);
      });
    }

    loadPromiseRef.current.then((ready) => setIsReady(ready));
  }, []);

  return { isReady };
}

export interface CheckoutOptions {
  key: string;
  amount: number;
  currency: string;
  name?: string;
  description?: string;
  order_id: string;
  prefill?: {
    name?: string;
    email?: string;
    contact?: string;
  };
  notes?: Record<string, string>;
  theme?: {
    color?: string;
  };
  handler?: (response: {
    razorpay_payment_id: string;
    razorpay_order_id: string;
    razorpay_signature: string;
  }) => void;
  modal?: {
    ondismiss?: () => void;
    escape?: boolean;
    animation?: boolean;
  };
}

export function openRazorpayCheckout(options: CheckoutOptions): void {
  if (typeof window === "undefined" || !window.Razorpay) {
    throw new Error("Razorpay Checkout is not loaded");
  }
  const rzp = new window.Razorpay(options);
  rzp.open();
}
