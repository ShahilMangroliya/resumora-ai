import Link from "next/link";
import { notFound } from "next/navigation";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { ALL_SLUGS, listDocs, loadDoc, type DocEntry } from "@/lib/docs";

import { DocsTopBar } from "../_components/DocsTopBar";

export const dynamicParams = false;

export async function generateStaticParams() {
  return ALL_SLUGS.map((slug) => ({ slug: slug.split("/") }));
}

const GROUP_LABEL: Record<DocEntry["group"], string> = {
  guide: "Learning guide",
  setup: "Setup",
  concepts: "Concepts",
};

interface PageProps {
  params: Promise<{ slug: string[] }>;
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const joined = slug.join("/");
  const result = (await loadDoc(joined)) ?? (await loadDoc(`${joined}/README`));
  return {
    title: result ? `${result.title} — Resumora Docs` : "Docs",
    description: result ? `Resumora AI documentation: ${result.title}` : undefined,
  };
}

export default async function DocPage({ params }: PageProps) {
  const { slug } = await params;
  const joined = slug.join("/");
  const doc = (await loadDoc(joined)) ?? (await loadDoc(`${joined}/README`));
  if (!doc) notFound();

  const allDocs = await listDocs();
  const current =
    allDocs.find((d) => d.slug === joined || d.slug === `${joined}/README`) ?? null;

  return (
    <main className="relative mx-auto w-full max-w-[1280px] px-6 pb-32 pt-8 md:px-10">
      <DocsTopBar />

      <div className="mt-10 grid grid-cols-1 gap-10 md:grid-cols-[260px_minmax(0,1fr)]">
        <DocsSidebar docs={allDocs} currentSlug={current?.slug ?? joined} />

        <article className="glass relative overflow-hidden p-7 md:p-12">
          <nav className="mb-5 flex flex-wrap items-center gap-2 text-xs text-[color:var(--muted)]">
            <Link href="/docs" className="hover:text-[color:var(--ink)]">Docs</Link>
            <span aria-hidden>/</span>
            {current && (
              <>
                <span>{GROUP_LABEL[current.group]}</span>
                <span aria-hidden>/</span>
              </>
            )}
            <span className="text-[color:var(--ink-soft)]">{doc.title}</span>
          </nav>

          <header className="mb-8 border-b border-[color:var(--rule)] pb-6">
            <span className="italic-display text-lg leading-none text-[color:var(--teal)]">
              {current ? GROUP_LABEL[current.group].toLowerCase() : "reference"}
            </span>
            <h1 className="mt-1 font-display text-4xl leading-[1.05] tracking-[-0.02em] text-[color:var(--ink)] md:text-5xl">
              {doc.title}
            </h1>
          </header>

          <div className="prose-dreamy">
            <Markdown remarkPlugins={[remarkGfm]}>{doc.markdown}</Markdown>
          </div>
        </article>
      </div>
    </main>
  );
}

function DocsSidebar({ docs, currentSlug }: { docs: DocEntry[]; currentSlug: string }) {
  const groups: { group: DocEntry["group"]; items: DocEntry[] }[] = [
    { group: "guide",    items: docs.filter((d) => d.group === "guide").sort((a, b) => a.order - b.order) },
    { group: "setup",    items: docs.filter((d) => d.group === "setup").sort((a, b) => a.order - b.order) },
    { group: "concepts", items: docs.filter((d) => d.group === "concepts").sort((a, b) => a.order - b.order) },
  ];

  return (
    <aside className="md:sticky md:top-8 md:self-start">
      <div className="glass-quiet p-5">
        <Link
          href="/docs"
          className="mb-4 flex items-center gap-2 text-xs text-[color:var(--muted)] hover:text-[color:var(--ink)]"
        >
          <span aria-hidden>←</span>
          <span>All docs</span>
        </Link>

        <nav className="flex flex-col gap-5">
          {groups.map(({ group, items }) => (
            <div key={group} className="flex flex-col gap-1.5">
              <span className="eyebrow">{GROUP_LABEL[group]}</span>
              <ul className="flex flex-col gap-0.5">
                {items.map((doc) => {
                  const active = doc.slug === currentSlug;
                  const href = `/docs/${doc.slug.replace(/\/README$/, "")}`;
                  return (
                    <li key={doc.slug}>
                      <Link
                        href={href}
                        className={`block rounded-lg px-3 py-1.5 text-sm leading-snug transition-colors ${
                          active
                            ? "bg-white/70 text-[color:var(--ink)] shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_4px_14px_-8px_rgba(20,19,14,0.18)]"
                            : "text-[color:var(--ink-soft)] hover:bg-white/40 hover:text-[color:var(--ink)]"
                        }`}
                      >
                        {doc.title}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
      </div>
    </aside>
  );
}
