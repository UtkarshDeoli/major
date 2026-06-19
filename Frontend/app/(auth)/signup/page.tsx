import { AuthSplitLayout } from "@/components/auth/auth-split-layout";

export default function SignupPage() {
  return (
    <AuthSplitLayout
      type="signup"
      formTitle="Create your account"
      formSubtitle="Start your journey to exam success"
    />
  );
}