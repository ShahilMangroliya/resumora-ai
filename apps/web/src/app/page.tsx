import Link from "next/link";

import { Analyzer } from "@/components/Analyzer";

export default function Home() {
  return (
    <main className="relative mx-auto w-full max-w-[1180px] px-6 pb-32 pt-8 md:px-10">
      <TopBar />

      <section className="flex flex-col items-center gap-7 pt-20 text-center md:pt-28">
        <span className="pill rise">
          <span className="dot" />
          <span>Open-source · Self-hosted · No paid APIs</span>
        </span>

        <h1 className="rise delay-1 font-display text-[clamp(2.8rem,7vw,5.8rem)] font-medium leading-[1.02] tracking-[-0.025em] text-[color:var(--ink)]">
          How does your <span className="italic-display text-[color:var(--teal)]">résumé</span>
          <br />
          read the room
          <span className="italic-display text-[color:var(--clay)]">?</span>
        </h1>

        <p className="rise delay-2 max-w-[58ch] text-base leading-relaxed text-[color:var(--ink-soft)] md:text-lg">
          Drop a résumé, paste a job description, and Resumora returns a fit score,
          three reasons behind it, and three bullet rewrites you can paste straight
          into your CV. A quiet little instrument for loud career moves.
        </p>

        <div className="rise delay-3 flex flex-wrap items-center justify-center gap-2 pt-1">
          <Trust label="DistilBERT + LoRA" tone="clay" />
          <Trust label="Sentence transformers" tone="teal" />
          <Trust label="Llama 3.2 · local" tone="moss" />
          <Trust label="FastAPI ↔ Next 16" tone="ochre" />
        </div>
      </section>

      <div className="rise delay-4 mt-20 md:mt-28">
        <Analyzer />
      </div>

      <footer className="mt-32 flex flex-col items-center gap-3 text-center text-sm text-[color:var(--muted)] md:flex-row md:justify-between md:text-left">
        <span>Resumora AI · v0.1 · {new Date().getFullYear()}</span>
        <span className="italic-display text-base text-[color:var(--ink-soft)]">
          made with care, for candidates
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--score-strong)] shadow-[0_0_10px_var(--score-strong)]" />
          pipeline online
        </span>
      </footer>
    </main>
  );
}

function TopBar() {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Mark />
        <span className="font-display text-xl tracking-tight text-[color:var(--ink)]">
          Resumora
        </span>
      </div>
      <nav className="hidden items-center gap-1 md:flex">
        <NavItem href="/" active>Analyze</NavItem>
        <NavItem href="/docs">Docs</NavItem>
        <NavItem href="/docs/learning-guide">Source</NavItem>
      </nav>
      <span className="pill"><span className="dot" /> Shahil</span>
    </div>
  );
}

function NavItem({
  children,
  href,
  active = false,
}: {
  children: React.ReactNode;
  href: string;
  active?: boolean;
}) {
  return (
    <Link
      href={href}
      className={`rounded-full px-4 py-1.5 text-sm transition-colors ${
        active
          ? "bg-white/60 text-[color:var(--ink)] shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_4px_14px_-8px_rgba(20,19,14,0.18)]"
          : "text-[color:var(--ink-soft)] hover:bg-white/40 hover:text-[color:var(--ink)]"
      }`}
    >
      {children}
    </Link>
  );
}

function Mark() {
  return (
    <span
      aria-hidden
      className="inline-grid h-9 w-9 place-items-center rounded-2xl border border-white/15 shadow-[inset_0_1px_0_rgba(255,255,255,0.22),0_8px_18px_-10px_rgba(14,93,82,0.5)]"
      style={{
        background:
          "linear-gradient(135deg, var(--teal) 0%, var(--teal-deep) 100%)",
      }}
    >
      <span className="italic-display text-lg leading-none text-[color:var(--paper-3)]">r</span>
    </span>
  );
}

function Trust({ label, tone }: { label: string; tone: "clay" | "teal" | "moss" | "ochre" }) {
  const bg: Record<typeof tone, string> = {
    clay:  "linear-gradient(135deg, rgba(241,212,196,0.65), rgba(255,253,248,0.55))",
    teal:  "linear-gradient(135deg, rgba(215,229,225,0.7),  rgba(255,253,248,0.55))",
    moss:  "linear-gradient(135deg, rgba(220,227,205,0.7),  rgba(255,253,248,0.55))",
    ochre: "linear-gradient(135deg, rgba(239,224,187,0.7),  rgba(255,253,248,0.55))",
  };
  return (
    <span
      className="rounded-full border border-[color:var(--rule)] px-3 py-1.5 text-xs text-[color:var(--ink-soft)] shadow-[inset_0_1px_0_rgba(255,255,255,0.6)]"
      style={{ background: bg[tone] }}
    >
      {label}
    </span>
  );
}
