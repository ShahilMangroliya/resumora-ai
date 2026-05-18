import type { BulletRewrite } from "@/lib/types";

export function RewriteCards({ rewrites }: { rewrites: BulletRewrite[] }) {
  return (
    <section className="glass p-7 md:p-12">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="italic-display text-lg leading-none text-[color:var(--clay)]">three small edits</span>
          <h2 className="font-display text-3xl text-[color:var(--ink)] md:text-4xl">
            Bullet rewrites
          </h2>
        </div>
        <span className="max-w-[40ch] text-sm text-[color:var(--muted)]">
          Drop these straight into your résumé — tweak as you see fit.
        </span>
      </header>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {rewrites.map((rewrite, index) => (
          <article
            key={index}
            className="glass-quiet relative flex flex-col gap-5 p-6"
          >
            <div className="flex items-baseline justify-between">
              <span className="italic-display text-2xl text-[color:var(--muted-2)] tabular leading-none">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span className="text-xs text-[color:var(--muted)]">
                rewrite {index + 1} / {rewrites.length}
              </span>
            </div>

            {rewrite.original ? (
              <div className="flex flex-col gap-1.5">
                <span className="text-xs uppercase tracking-[0.14em] text-[color:var(--clay-deep)]">
                  Before
                </span>
                <p className="text-sm leading-relaxed text-[color:var(--muted)] line-through decoration-[color:var(--clay)] decoration-[1px] underline-offset-4">
                  {rewrite.original}
                </p>
              </div>
            ) : (
              <span className="text-xs uppercase tracking-[0.14em] text-[color:var(--moss-deep)]">
                Suggested addition
              </span>
            )}

            <div className="flex flex-col gap-1.5">
              <span className="text-xs uppercase tracking-[0.14em] text-[color:var(--teal)]">
                After
              </span>
              <p className="font-display text-base italic leading-snug text-[color:var(--ink)] md:text-lg">
                {rewrite.rewritten}
              </p>
            </div>

            <div className="mt-auto flex items-start gap-2 border-t border-[color:var(--rule)] pt-3">
              <span aria-hidden className="text-xs text-[color:var(--muted)]">¶</span>
              <p className="text-xs leading-relaxed text-[color:var(--ink-soft)]">
                {rewrite.rationale}
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
