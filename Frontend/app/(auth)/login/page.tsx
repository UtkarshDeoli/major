import { AuthSplitLayout } from "@/components/auth/auth-split-layout";
import { RedirectIfAuthenticated } from "@/components/auth/redirect-if-authenticated";

export default function LoginPage() {
  return (
    <RedirectIfAuthenticated>
      <AuthSplitLayout
        type="login"
        formTitle="Welcome back"
        formSubtitle="Enter your credentials to access your account"
      />
    </RedirectIfAuthenticated>
  );
}