export const metadata = { title: "Terms of Service — Orbit" };

export default function TermsPage() {
  return (
    <div className="min-h-screen p-8 lg:p-16 bg-background">
      <article className="max-w-2xl mx-auto prose prose-sm dark:prose-invert">
        <h1>Terms of Service</h1>
        <p>
          By using Orbit, you agree to use the platform for lawful study purposes only. You are
          responsible for the content you upload and for maintaining the security of your account.
        </p>
        <p>
          Orbit provides AI-generated study aids that may be inaccurate. You are responsible for
          verifying any guidance before relying on it for examinations.
        </p>
        <p>
          We may update these terms periodically. Continued use after changes constitutes
          acceptance of the revised terms.
        </p>
      </article>
    </div>
  );
}