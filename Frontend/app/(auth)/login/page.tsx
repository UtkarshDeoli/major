import { AuthSplitLayout } from "@/components/auth/auth-split-layout";

export default function LoginPage() {
  return (
    <AuthSplitLayout
      type="login"
      formTitle="Welcome back"
      formSubtitle="Enter your credentials to access your account"
    />
  );
}