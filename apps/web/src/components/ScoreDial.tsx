import type { ScoreResult } from "@/lib/types";

const SCORE_MIN = 20;
const SCORE_MAX = 85;

const LABEL_COLOR: Record<ScoreResult["predicted_label"], string> = {
  weak: "var(--color-score-weak)",
  partial: "var(--color-score-partial)",
  strong: "var(--color-score-strong)",
};

const LABEL_COPY: Record<ScoreResult["predicted_label"], string> = {
  weak: "Weak fit",
  partial: "Partial fit",
  strong: "Strong fit",
};

export function ScoreDial({ score }: { score: ScoreResult }) {
  const color = LABEL_COLOR[score.predicted_label];
  const pct = Math.max(0, Math.min(1, (score.score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN))) * 100;
  const confidencePct = Math.round(score.confidence * 100);

  return (
    <section
      aria-label="Fit score"
      className="flex flex-col gap-4 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6"
    >
      <div className="flex items-baseline gap-4">
        <span className="text-6xl font-semibold tabular-nums" style={{ color }}>
          {score.score.toFixed(1)}
        </span>
        <span className="text-sm text-[color:var(--muted)]">/ 100</span>
        <span
          className="ml-auto rounded-full px-3 py-1 text-xs font-medium uppercase tracking-wide text-white"
          style={{ backgroundColor: color }}
        >
          {LABEL_COPY[score.predicted_label]}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        <div className="relative h-2 w-full overflow-hidden rounded-full bg-[color:var(--border)]">
          <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
        </div>
        <div className="flex justify-between text-xs text-[color:var(--muted)]">
          <span>{SCORE_MIN}</span>
          <span>{SCORE_MAX}</span>
        </div>
      </div>

      <p className="text-xs text-[color:var(--muted)]">
        Confidence {confidencePct}% — weak {(score.class_probabilities.weak * 100).toFixed(0)}%, partial{" "}
        {(score.class_probabilities.partial * 100).toFixed(0)}%, strong{" "}
        {(score.class_probabilities.strong * 100).toFixed(0)}%.
      </p>
    </section>
  );
}
