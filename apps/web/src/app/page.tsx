import { Analyzer } from "@/components/Analyzer";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-10 px-6 py-16">
      <header className="flex flex-col gap-3">
        <span className="text-xs font-medium uppercase tracking-[0.2em] text-[color:var(--muted)]">
          Resumora AI
        </span>
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Score your resume against any job description.
        </h1>
        <p className="max-w-2xl text-sm text-[color:var(--muted)] md:text-base">
          Upload a resume, paste a job description, and get a fit score, the top three
          reasons behind it, and three bullet rewrites you can lift into your CV.
        </p>
      </header>

      <Analyzer />
    </main>
  );
}
