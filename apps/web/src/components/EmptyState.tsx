export function EmptyState() {
  return (
    <section className="glass relative overflow-hidden p-10 md:p-14">
      <div className="grid grid-cols-1 items-center gap-10 md:grid-cols-[1fr_auto]">
        <div className="flex flex-col gap-5">
          <span className="eyebrow">— standing by</span>
          <h2 className="font-display text-3xl leading-tight text-[color:var(--ink)] md:text-4xl">
            No analysis yet
          </h2>
          <p className="max-w-prose text-base leading-relaxed text-[color:var(--ink-soft)]">
            Drop a résumé and paste a job description above. The pipeline will return a
            fit score, three reasons for that score, a skill-match report, and three
            bullet rewrites you can drop straight into your CV.
          </p>

          <ul className="grid max-w-md grid-cols-3 gap-3 pt-2">
            <Stat n="~90s" label="warm pipeline" />
            <Stat n="5"    label="stages" />
            <Stat n="0"    label="paid APIs" />
          </ul>
        </div>

        <Ornament />
      </div>
    </section>
  );
}

function Stat({ n, label }: { n: string; label: string }) {
  return (
    <li className="glass-quiet flex flex-col gap-1 p-4">
      <span className="italic-display text-2xl text-[color:var(--ink)]">{n}</span>
      <span className="text-xs text-[color:var(--muted)]">{label}</span>
    </li>
  );
}

function Ornament() {
  return (
    <svg
      width="240"
      height="240"
      viewBox="0 0 240 240"
      aria-hidden
      className="hidden md:block"
    >
      <defs>
        <radialGradient id="orn-a" cx="0.4" cy="0.35" r="0.6">
          <stop offset="0%" stopColor="#ffd4b8" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#ffd4b8" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="orn-b" cx="0.65" cy="0.7" r="0.55">
          <stop offset="0%" stopColor="#d4c8f0" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#d4c8f0" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="orn-c" cx="0.55" cy="0.4" r="0.4">
          <stop offset="0%" stopColor="#c8e6d4" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#c8e6d4" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="120" cy="120" r="110" fill="url(#orn-a)" />
      <circle cx="140" cy="140" r="100" fill="url(#orn-b)" />
      <circle cx="110" cy="100" r="70"  fill="url(#orn-c)" />
      <circle cx="120" cy="120" r="3" fill="rgba(26,24,37,0.5)" />
    </svg>
  );
}
