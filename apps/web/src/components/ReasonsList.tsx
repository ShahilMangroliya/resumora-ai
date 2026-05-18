import type { Reason } from "@/lib/types";

const CATEGORY_LABEL: Record<Reason["category"], string> = {
  matched_skill: "Match",
  missing_skill: "Gap",
  experience_match: "Experience",
  experience_gap: "Experience gap",
  other: "Note",
};

const CATEGORY_COLOR: Record<Reason["category"], string> = {
  matched_skill: "#6fb18a",
  missing_skill: "#e26d8a",
  experience_match: "#6fb18a",
  experience_gap: "#e6a955",
  other: "#a17ad4",
};

const CATEGORY_BG: Record<Reason["category"], string> = {
  matched_skill: "linear-gradient(135deg, rgba(200,230,212,0.7), rgba(255,255,255,0.7))",
  missing_skill: "linear-gradient(135deg, rgba(231,168,179,0.65), rgba(255,255,255,0.7))",
  experience_match: "linear-gradient(135deg, rgba(200,230,212,0.7), rgba(255,255,255,0.7))",
  experience_gap: "linear-gradient(135deg, rgba(243,227,168,0.7), rgba(255,255,255,0.7))",
  other: "linear-gradient(135deg, rgba(212,200,240,0.7), rgba(255,255,255,0.7))",
};

export function ReasonsList({ reasons }: { reasons: Reason[] }) {
  return (
    <section className="glass p-7 md:p-12">
      <header className="mb-7 flex flex-wrap items-end justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="italic-display text-lg leading-none text-[color:#a17ad4]">a quiet look at</span>
          <h2 className="font-display text-3xl text-[color:var(--ink)] md:text-4xl">
            Why this score
          </h2>
        </div>
        <span className="text-xs text-[color:var(--muted)]">{reasons.length} reasons</span>
      </header>

      <ol className="flex flex-col gap-4">
        {reasons.map((reason, index) => {
          const color = CATEGORY_COLOR[reason.category];
          const bg = CATEGORY_BG[reason.category];
          return (
            <li
              key={index}
              className="glass-quiet grid grid-cols-[auto_1fr_auto] items-start gap-5 p-5 md:p-6"
            >
              <span className="italic-display text-3xl text-[color:var(--muted-2)] tabular leading-none">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div className="flex flex-col gap-1.5">
                <p className="text-base font-medium leading-snug text-[color:var(--ink)] md:text-lg">
                  {reason.summary}
                </p>
                <p className="max-w-prose text-sm leading-relaxed text-[color:var(--ink-soft)]">
                  {reason.evidence}
                </p>
              </div>
              <span
                className="flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium"
                style={{ color, borderColor: `${color}55`, background: bg }}
              >
                <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: color }} />
                {CATEGORY_LABEL[reason.category]}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
