import type { SkillMatch, SkillMatchReport } from "@/lib/types";

interface ChipGroupProps {
  title: string;
  skills: SkillMatch[];
  tone: "match" | "miss";
}

function ChipGroup({ title, skills, tone }: ChipGroupProps) {
  if (skills.length === 0) return null;
  const cls = tone === "match" ? "chip chip--match" : "chip chip--miss";
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-medium text-[color:var(--ink-soft)]">{title}</h3>
        <span className="text-xs text-[color:var(--muted)] tabular">
          {String(skills.length).padStart(2, "0")}
        </span>
      </div>
      <ul className="flex flex-wrap gap-2">
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
            className={cls}
          >
            <span className="marker" aria-hidden />
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
    <section className="glass p-7 md:p-12">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-5">
        <div className="flex flex-col gap-2">
          <h2 className="font-display text-3xl text-[color:var(--ink)] md:text-4xl">
            Skill match
          </h2>
          <span className="text-sm text-[color:var(--muted)]">
            How many requested skills the résumé covers
          </span>
        </div>
        <div className="flex items-baseline gap-3">
          <span className="italic-display text-5xl text-[color:var(--ink)] tabular md:text-6xl">
            {matchPct}%
          </span>
          <span className="max-w-[18ch] text-sm text-[color:var(--muted)]">
            {matchPct}% of required skills matched
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <ChipGroup title="Required — matched" skills={report.required_matched} tone="match" />
        <ChipGroup title="Required — missing" skills={report.required_missing} tone="miss" />
        <ChipGroup title="Nice to have — matched" skills={report.nice_to_have_matched} tone="match" />
        <ChipGroup title="Nice to have — missing" skills={report.nice_to_have_missing} tone="miss" />
      </div>
    </section>
  );
}
