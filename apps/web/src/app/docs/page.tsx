import Link from "next/link";

import { listDocs, type DocEntry } from "@/lib/docs";

import { DocsTopBar } from "./_components/DocsTopBar";

export const metadata = {
  title: "Docs — Resumora AI",
  description: "Learning guide, setup, and concept references for Resumora AI.",
};

const GROUP_META: Record<DocEntry["group"], { eyebrow: string; title: string; tone: string }> = {
  guide:    { eyebrow: "start here",      title: "Learning guide", tone: "linear-gradient(135deg, rgba(241,212,196,0.6), rgba(255,253,248,0.65))" },
  setup:    { eyebrow: "prerequisites",   title: "Setup",          tone: "linear-gradient(135deg, rgba(239,224,187,0.6), rgba(255,253,248,0.65))" },
  concepts: { eyebrow: "reference shelf", title: "Concepts",       tone: "linear-gradient(135deg, rgba(215,229,225,0.7), rgba(255,253,248,0.65))" },
};

const GROUP_ORDER: DocEntry["group"][] = ["guide", "setup", "concepts"];

export default async function DocsIndex() {
  const docs = await listDocs();
  const grouped = GROUP_ORDER.map((group) => ({
    group,
    meta: GROUP_META[group],
    items: docs.filter((d) => d.group === group).sort((a, b) => a.order - b.order),
  }));

  return (
    <main className="relative mx-auto w-full max-w-[1180px] px-6 pb-32 pt-8 md:px-10">
      <DocsTopBar />

      <header className="flex flex-col items-center gap-6 pt-16 text-center md:pt-24">
        <span className="pill rise">
          <span className="dot" />
          <span>Help · learning · concept reference</span>
        </span>
        <h1 className="rise delay-1 font-display text-[clamp(2.6rem,6vw,5rem)] font-medium leading-[1.02] tracking-[-0.025em] text-[color:var(--ink)]">
          <span className="italic-display text-[color:var(--teal)]">Docs</span> &amp; help
        </h1>
        <p className="rise delay-2 max-w-[58ch] text-base leading-relaxed text-[color:var(--ink-soft)] md:text-lg">
          A slow on-ramp for software engineers new to ML, a setup checklist for
          running the pipeline locally, and a per-topic reference for every AI
          concept Resumora uses.
        </p>
      </header>

      <div className="rise delay-3 mt-16 flex flex-col gap-10 md:mt-24">
        {grouped.map(({ group, meta, items }) => (
          <section key={group} className="glass p-7 md:p-10">
            <div
              className="mb-7 flex flex-wrap items-end justify-between gap-3 rounded-2xl p-5"
              style={{ background: meta.tone }}
            >
              <div className="flex flex-col gap-1">
                <span className="eyebrow">— {meta.eyebrow}</span>
                <h2 className="font-display text-3xl text-[color:var(--ink)] md:text-4xl">
                  {meta.title}
                </h2>
              </div>
              <span className="text-xs text-[color:var(--muted)] tabular">
                {items.length} {items.length === 1 ? "page" : "pages"}
              </span>
            </div>

            <ul className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {items.map((doc) => (
                <li key={doc.slug}>
                  <Link
                    href={`/docs/${doc.slug.replace(/\/README$/, "")}`}
                    className="glass-quiet group flex h-full flex-col gap-2 p-5 transition-transform hover:-translate-y-0.5"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <h3 className="font-display text-lg italic text-[color:var(--ink)]">
                        {doc.title}
                      </h3>
                      <span aria-hidden className="text-sm text-[color:var(--muted)] transition-transform group-hover:translate-x-1 group-hover:text-[color:var(--ink)]">
                        →
                      </span>
                    </div>
                    {doc.summary && (
                      <p className="text-sm leading-relaxed text-[color:var(--ink-soft)]">
                        {doc.summary}
                      </p>
                    )}
                    <span className="mt-auto pt-2 text-xs text-[color:var(--muted)]">
                      /docs/{doc.slug.replace(/\/README$/, "")}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </main>
  );
}
