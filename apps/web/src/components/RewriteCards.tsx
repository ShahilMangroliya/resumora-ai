import type { BulletRewrite } from "@/lib/types";

export function RewriteCards({ rewrites }: { rewrites: BulletRewrite[] }) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
      <h2 className="text-lg font-semibold">Bullet rewrites</h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {rewrites.map((rewrite, index) => (
          <article
            key={index}
            className="flex flex-col gap-3 rounded-xl border border-[color:var(--border)] p-4"
          >
            {rewrite.original ? (
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wide text-[color:var(--muted)]">Before</span>
                <p className="text-sm line-through decoration-[color:var(--muted)]/70">{rewrite.original}</p>
              </div>
            ) : (
              <span className="text-xs uppercase tracking-wide text-[color:var(--muted)]">Suggested addition</span>
            )}
            <div className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-[color:var(--muted)]">After</span>
              <p className="text-sm font-medium">{rewrite.rewritten}</p>
            </div>
            <p className="text-xs text-[color:var(--muted)]">{rewrite.rationale}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
