import { AuthSplitLayout } from "@/components/auth/auth-split-layout";
import { RedirectIfAuthenticated } from "@/components/auth/redirect-if-authenticated";

export default function SignupPage() {
  return (
    <RedirectIfAuthenticated>
      <AuthSplitLayout
        type="signup"
        formTitle="Create your account"
        formSubtitle="Start your journey to exam success"
      />
    </RedirectIfAuthenticated>
  );
}