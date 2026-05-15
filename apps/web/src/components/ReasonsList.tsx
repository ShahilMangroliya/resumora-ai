import type { Reason } from "@/lib/types";

const CATEGORY_LABEL: Record<Reason["category"], string> = {
  matched_skill: "Match",
  missing_skill: "Gap",
  experience_match: "Experience",
  experience_gap: "Experience gap",
  other: "Note",
};

const CATEGORY_TONE: Record<Reason["category"], string> = {
  matched_skill: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
  missing_skill: "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-200",
  experience_match: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
  experience_gap: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
  other: "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-200",
};

export function ReasonsList({ reasons }: { reasons: Reason[] }) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
      <h2 className="text-lg font-semibold">Why this score</h2>
      <ol className="flex flex-col gap-3">
        {reasons.map((reason, index) => (
          <li
            key={index}
            className="flex flex-col gap-1 rounded-xl border border-[color:var(--border)] p-4"
          >
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-medium">{reason.summary}</p>
              <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs ${CATEGORY_TONE[reason.category]}`}>
                {CATEGORY_LABEL[reason.category]}
              </span>
            </div>
            <p className="text-sm text-[color:var(--muted)]">{reason.evidence}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
