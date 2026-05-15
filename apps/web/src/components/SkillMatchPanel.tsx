import type { SkillMatch, SkillMatchReport } from "@/lib/types";

interface ChipGroupProps {
  title: string;
  skills: SkillMatch[];
  tone: "match" | "miss";
}

function ChipGroup({ title, skills, tone }: ChipGroupProps) {
  if (skills.length === 0) return null;
  const chipClass =
    tone === "match"
      ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200"
      : "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-200";
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-medium uppercase tracking-wide text-[color:var(--muted)]">{title}</h3>
      <ul className="flex flex-wrap gap-1.5">
        {skills.map((skill) => (
          <li
            key={`${skill.jd_skill}|${skill.resume_skill}`}
            title={
              tone === "match"
                ? `Matched to "${skill.resume_skill}" (sim ${skill.similarity.toFixed(2)})`
                : skill.resume_skill
                  ? `Closest in resume: "${skill.resume_skill}" (sim ${skill.similarity.toFixed(2)})`
                  : "Not found in resume"
            }
            className={`rounded-full px-2.5 py-1 text-xs ${chipClass}`}
          >
            {skill.jd_skill}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SkillMatchPanel({ report }: { report: SkillMatchReport }) {
  const matchPct = Math.round(report.match_rate * 100);
  return (
    <section className="flex flex-col gap-5 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-lg font-semibold">Skill match</h2>
        <span className="text-sm text-[color:var(--muted)]">
          {matchPct}% of required skills matched
        </span>
      </div>
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <ChipGroup title="Required — matched" skills={report.required_matched} tone="match" />
        <ChipGroup title="Required — missing" skills={report.required_missing} tone="miss" />
        <ChipGroup title="Nice to have — matched" skills={report.nice_to_have_matched} tone="match" />
        <ChipGroup title="Nice to have — missing" skills={report.nice_to_have_missing} tone="miss" />
      </div>
    </section>
  );
}
