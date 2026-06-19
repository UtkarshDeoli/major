export const metadata = { title: "Privacy Policy — Orbit" };

export default function PrivacyPage() {
  return (
    <div className="min-h-screen p-8 lg:p-16 bg-background">
      <article className="max-w-2xl mx-auto prose prose-sm dark:prose-invert">
        <h1>Privacy Policy</h1>
        <p>
          Orbit stores the documents you upload and the tests you generate to provide you with
          study aids. We do not sell your data.
        </p>
        <p>
          You may request deletion of your account and associated data at any time from Settings.
        </p>
        <p>
          Authentication tokens are stored locally in your browser. We use them only to keep you
          signed in.
        </p>
      </article>
    </div>
  );
}