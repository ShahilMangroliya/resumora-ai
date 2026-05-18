import { promises as fs } from "node:fs";
import path from "node:path";

const DOCS_DIR = process.env.DOCS_DIR
  ? path.resolve(process.env.DOCS_DIR)
  : path.resolve(process.cwd(), "..", "..", "docs");

export interface DocEntry {
  slug: string;
  title: string;
  summary: string;
  group: "guide" | "setup" | "concepts";
  order: number;
}

const SECTIONS: ReadonlyArray<{
  group: DocEntry["group"];
  files: ReadonlyArray<{ slug: string; order: number }>;
}> = [
  { group: "guide", files: [{ slug: "learning-guide", order: 1 }] },
  { group: "setup", files: [{ slug: "ollama-setup", order: 1 }] },
  {
    group: "concepts",
    files: [
      { slug: "concepts/README", order: 0 },
      { slug: "concepts/tokens-and-tokenizers", order: 1 },
      { slug: "concepts/embeddings", order: 2 },
      { slug: "concepts/sentence-transformers", order: 3 },
      { slug: "concepts/transformers", order: 4 },
      { slug: "concepts/classifiers", order: 5 },
      { slug: "concepts/fine-tuning", order: 6 },
      { slug: "concepts/lora-and-peft", order: 7 },
      { slug: "concepts/llms-and-ollama", order: 8 },
      { slug: "concepts/prompting", order: 9 },
      { slug: "concepts/hugging-face-hub", order: 10 },
      { slug: "concepts/whats-next", order: 11 },
    ],
  },
];

function extractTitleAndSummary(raw: string): { title: string; summary: string } {
  const titleMatch = raw.match(/^#\s+(.+?)\s*$/m);
  const title = titleMatch ? titleMatch[1].replace(/[#*_`]/g, "").trim() : "Untitled";
  const afterTitle = titleMatch ? raw.slice(titleMatch.index! + titleMatch[0].length) : raw;
  const paragraph = afterTitle
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .find((block) => block.length > 0 && !block.startsWith(">") && !block.startsWith("```") && !block.startsWith("---"));
  const summary = (paragraph ?? "")
    .replace(/[#*_`]/g, "")
    .replace(/\s+/g, " ")
    .slice(0, 220)
    .trim();
  return { title, summary: summary ? `${summary}${summary.length === 220 ? "…" : ""}` : "" };
}

async function readRaw(slug: string): Promise<string> {
  return fs.readFile(path.join(DOCS_DIR, `${slug}.md`), "utf8");
}

export async function listDocs(): Promise<DocEntry[]> {
  const entries: DocEntry[] = [];
  for (const section of SECTIONS) {
    for (const { slug, order } of section.files) {
      try {
        const raw = await readRaw(slug);
        const { title, summary } = extractTitleAndSummary(raw);
        entries.push({ slug, title, summary, group: section.group, order });
      } catch {
        // skip missing files silently
      }
    }
  }
  return entries;
}

export function docExists(slug: string): boolean {
  return SECTIONS.some((s) => s.files.some((f) => f.slug === slug));
}

/**
 * Rewrite intra-doc links so `[foo](./concepts/embeddings.md#anchor)` becomes
 * `[foo](/docs/concepts/embeddings#anchor)` from any source slug.
 */
export function rewriteLinks(markdown: string, fromSlug: string): string {
  const baseDir = path.posix.dirname(`/docs/${fromSlug}`);
  return markdown.replace(/\]\(([^)]+)\)/g, (match, href: string) => {
    if (/^[a-z]+:\/\//i.test(href) || href.startsWith("#") || href.startsWith("mailto:")) {
      return match;
    }
    let next = href.replace(/\.md(#.*)?$/, "$1");
    if (!next.startsWith("/")) {
      next = path.posix.normalize(path.posix.join(baseDir, next));
    }
    next = next.replace(/\/README$/, "");
    return `](${next})`;
  });
}

export async function loadDoc(slug: string): Promise<{ title: string; markdown: string } | null> {
  if (!docExists(slug)) return null;
  const raw = await readRaw(slug);
  const { title } = extractTitleAndSummary(raw);
  // strip leading H1 — we render the title separately in the page header
  const body = raw.replace(/^#\s+.+\n+/, "");
  return { title, markdown: rewriteLinks(body, slug) };
}

export const ALL_SLUGS: readonly string[] = SECTIONS.flatMap((s) => s.files.map((f) => f.slug));
