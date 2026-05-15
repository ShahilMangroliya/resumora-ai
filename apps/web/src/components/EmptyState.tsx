export function EmptyState() {
  return (
    <section className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-[color:var(--border)] bg-[color:var(--surface)] p-12 text-center">
      <h2 className="text-base font-semibold">No analysis yet</h2>
      <p className="max-w-md text-sm text-[color:var(--muted)]">
        Upload a resume and paste a job description above, then click Analyze. We&rsquo;ll score
        the fit, explain why, and suggest three bullet rewrites.
      </p>
    </section>
  );
}
