"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/lib/context/auth-context";
import { getPostAuthRedirect } from "@/lib/auth/redirects";

const ERROR_MESSAGES: Record<string, string> = {
  invalid_state: "Security check failed. Please try again.",
  no_email: "Google account has no email associated. Please use a different account.",
  exchange_failed: "Failed to sign in with Google. Please try again.",
  invalid_request: "Invalid request. Please try again.",
  access_denied: "Google sign-in was denied. Please try again or use email.",
  email_not_verified: "Your Google email is not verified. Please verify your email with Google first.",
};

function CallbackHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { hydrateFromToken } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = searchParams.get("token");
    const errorParam = searchParams.get("error");

    if (errorParam) {
      setError(ERROR_MESSAGES[errorParam] || "An unknown error occurred.");
      return;
    }

    if (!token) {
      setError("No token received from authentication.");
      return;
    }

    let cancelled = false;

    hydrateFromToken(token)
      .then((user) => {
        if (!cancelled) {
          router.replace(getPostAuthRedirect(user));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("We couldn't verify your account. Please sign in again.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [router, searchParams, hydrateFromToken]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0D1520]">
        <div className="text-center max-w-md p-8">
          <div className="text-red-400 text-5xl mb-4">⚠</div>
          <h1 className="text-2xl font-bold text-white mb-3">Sign In Failed</h1>
          <p className="text-gray-400 mb-6">{error}</p>
          <button
            onClick={() => router.replace("/login")}
            className="px-6 py-3 rounded-md bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold hover:from-purple-600 hover:to-blue-600 transition-all"
          >
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0D1520]">
      <div className="text-center">
        <div className="h-12 w-12 mx-auto mb-4 rounded-full border-4 border-purple-500 border-t-transparent animate-spin" />
        <p className="text-gray-400 text-lg">Completing sign in...</p>
      </div>
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center bg-[#0D1520]">
          <div className="text-center">
            <div className="h-12 w-12 mx-auto mb-4 rounded-full border-4 border-purple-500 border-t-transparent animate-spin" />
            <p className="text-gray-400 text-lg">Loading...</p>
          </div>
        </div>
      }
    >
      <CallbackHandler />
    </Suspense>
  );
}
