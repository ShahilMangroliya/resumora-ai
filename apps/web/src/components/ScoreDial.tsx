import type { ScoreResult } from "@/lib/types";

const SCORE_MIN = 20;
const SCORE_MAX = 85;

const LABEL_COLOR: Record<ScoreResult["predicted_label"], string> = {
  weak: "var(--score-weak)",
  partial: "var(--score-partial)",
  strong: "var(--score-strong)",
};

const LABEL_COPY: Record<ScoreResult["predicted_label"], string> = {
  weak: "Weak fit",
  partial: "Partial fit",
  strong: "Strong fit",
};

const LABEL_HALO: Record<ScoreResult["predicted_label"], string> = {
  weak: "linear-gradient(135deg, #f7c1cf 0%, #fff 70%)",
  partial: "linear-gradient(135deg, #f9d9a1 0%, #fff 70%)",
  strong: "linear-gradient(135deg, #c6e5d2 0%, #fff 70%)",
};

const GAUGE_START = 135;
const GAUGE_SWEEP = 270;
const RADIUS = 124;
const STROKE = 14;
const CENTER = 140;
const VIEW = 280;

function round(n: number) { return Math.round(n * 1000) / 1000; }

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return { x: round(cx + r * Math.cos(rad)), y: round(cy + r * Math.sin(rad)) };
}

function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number) {
  const start = polar(cx, cy, r, endDeg);
  const end = polar(cx, cy, r, startDeg);
  const large = endDeg - startDeg <= 180 ? 0 : 1;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 0 ${end.x} ${end.y}`;
}

export function ScoreDial({ score }: { score: ScoreResult }) {
  const color = LABEL_COLOR[score.predicted_label];
  const halo = LABEL_HALO[score.predicted_label];
  const ratio = Math.max(0, Math.min(1, (score.score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)));
  const confidencePct = Math.round(score.confidence * 100);

  const trackPath = arcPath(CENTER, CENTER, RADIUS, GAUGE_START, GAUGE_START + GAUGE_SWEEP);
  const progressPath = arcPath(
    CENTER,
    CENTER,
    RADIUS,
    GAUGE_START,
    GAUGE_START + GAUGE_SWEEP * ratio,
  );
  const circ = (2 * Math.PI * RADIUS * GAUGE_SWEEP) / 360;
  const filled = round(circ * ratio);

  return (
    <section
      aria-label="Fit score"
      role="region"
      className="glass relative overflow-hidden p-8 md:p-12"
    >
      {/* soft pastel halo behind the gauge */}
      <div
        aria-hidden
        className="pointer-events-none absolute -left-24 -top-24 h-[28rem] w-[28rem] rounded-full opacity-60 blur-3xl"
        style={{ background: halo }}
      />

      <div className="relative grid grid-cols-1 items-center gap-12 md:grid-cols-[auto_1fr]">
        <div className="relative mx-auto" style={{ width: VIEW, height: VIEW - 40 }}>
          <svg
            viewBox={`0 0 ${VIEW} ${VIEW - 40}`}
            width={VIEW}
            height={VIEW - 40}
            className="overflow-visible"
            aria-hidden
          >
            <defs>
              <linearGradient id="gauge-grad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.95" />
                <stop offset="100%" stopColor={color} stopOpacity="0.55" />
              </linearGradient>
              <filter id="soft-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Background track */}
            <path
              d={trackPath}
              fill="none"
              stroke="rgba(26, 24, 37, 0.07)"
              strokeWidth={STROKE}
              strokeLinecap="round"
            />

            {/* Progress arc */}
            <path
              d={progressPath}
              fill="none"
              stroke="url(#gauge-grad)"
              strokeWidth={STROKE}
              strokeLinecap="round"
              filter="url(#soft-glow)"
              style={{
                strokeDasharray: `${filled} ${round(circ)}`,
                animation: "dash 1300ms cubic-bezier(0.2, 0.7, 0.2, 1) both",
                ["--dash-from" as string]: `${filled}`,
                ["--dash-to" as string]: "0",
              } as React.CSSProperties}
            />

            {/* Subtle tick marks at quartiles */}
            {[0, 0.25, 0.5, 0.75, 1].map((t) => {
              const deg = GAUGE_START + GAUGE_SWEEP * t;
              const a = polar(CENTER, CENTER, RADIUS - 26, deg);
              const b = polar(CENTER, CENTER, RADIUS - 14, deg);
              return (
                <line
                  key={t}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke="rgba(26, 24, 37, 0.18)"
                  strokeWidth={1.2}
                  strokeLinecap="round"
                />
              );
            })}
          </svg>

          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span
              className="italic-display tabular leading-none"
              style={{
                fontSize: "clamp(4.6rem, 7vw, 6.8rem)",
                color: "var(--ink)",
                fontWeight: 400,
              }}
            >
              {score.score.toFixed(1)}
            </span>
            <span className="mt-1 text-xs text-[color:var(--muted)]">
              / 100 · fit score
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="flex flex-wrap items-center gap-3">
            <span
              className="rounded-full px-4 py-1.5 text-sm font-medium"
              style={{
                background: halo,
                border: `1px solid ${color}66`,
                color: "var(--ink)",
                boxShadow: `inset 0 1px 0 rgba(255,255,255,0.7), 0 6px 18px -10px ${color}66`,
              }}
            >
              {LABEL_COPY[score.predicted_label]}
            </span>
            <span className="text-sm text-[color:var(--muted)]">
              Confidence {confidencePct}%
            </span>
          </div>

          <div className="flex flex-col gap-3">
            <ProbBar label="weak"    pct={Math.round(score.class_probabilities.weak * 100)}    color="var(--score-weak)" />
            <ProbBar label="partial" pct={Math.round(score.class_probabilities.partial * 100)} color="var(--score-partial)" />
            <ProbBar label="strong"  pct={Math.round(score.class_probabilities.strong * 100)}  color="var(--score-strong)" />
          </div>

          <p className="border-t border-[color:var(--rule)] pt-4 text-sm leading-relaxed text-[color:var(--ink-soft)]">
            The classifier emits a soft distribution. A high number here doesn&rsquo;t
            guarantee an interview — it means your résumé echoes the language and
            substance of the JD closely.
          </p>
        </div>
      </div>
    </section>
  );
}

function ProbBar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="flex items-center gap-4">
      <span className="w-16 text-xs uppercase tracking-wide text-[color:var(--muted)]">
        {label}
      </span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-[color:var(--rule)]">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}, ${color}cc)`,
            boxShadow: `0 0 14px ${color}55`,
            animation: "fade 900ms ease both",
          }}
        />
      </div>
      <span className="w-20 text-right text-sm tabular text-[color:var(--ink)]">
        {label} {pct}%
      </span>
    </div>
  );
}
