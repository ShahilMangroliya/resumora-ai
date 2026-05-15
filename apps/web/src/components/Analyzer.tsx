"use client";

import { useState } from "react";

import { analyzeResume, ApiError } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

import { AnalyzeForm } from "./AnalyzeForm";
import { EmptyState } from "./EmptyState";
import { ReasonsList } from "./ReasonsList";
import { RewriteCards } from "./RewriteCards";
import { ScoreDial } from "./ScoreDial";
import { SkillMatchPanel } from "./SkillMatchPanel";
import { WarningsBanner } from "./WarningsBanner";

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "success"; result: AnalyzeResponse };

export function Analyzer() {
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  async function handleSubmit(input: { resume: File; jobDescription: string }) {
    setStatus({ kind: "loading" });
    try {
      const result = await analyzeResume(input);
      setStatus({ kind: "success", result });
    } catch (cause) {
      const message =
        cause instanceof ApiError
          ? cause.message
          : "Something went wrong while analyzing. Please try again.";
      setStatus({ kind: "error", message });
    }
  }

  return (
    <div className="flex flex-col gap-10">
      <section className="rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)] p-6">
        <AnalyzeForm pending={status.kind === "loading"} onSubmit={handleSubmit} />
      </section>

      {status.kind === "idle" && <EmptyState />}

      {status.kind === "loading" && (
        <section
          aria-live="polite"
          className="flex items-center justify-center rounded-2xl border border-dashed border-[color:var(--border)] bg-[color:var(--surface)] p-12 text-sm text-[color:var(--muted)]"
        >
          Scoring resume — this may take up to 90 seconds while the model runs.
        </section>
      )}

      {status.kind === "error" && (
        <aside
          role="alert"
          className="rounded-xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-700/60 dark:bg-rose-900/30 dark:text-rose-100"
        >
          {status.message}
        </aside>
      )}

      {status.kind === "success" && (
        <div className="flex flex-col gap-6">
          <WarningsBanner warnings={status.result.warnings} />
          <ScoreDial score={status.result.score} />
          {status.result.reasoning && <ReasonsList reasons={status.result.reasoning.reasons} />}
          {status.result.skill_report && <SkillMatchPanel report={status.result.skill_report} />}
          {status.result.reasoning && <RewriteCards rewrites={status.result.reasoning.rewrites} />}
        </div>
      )}
    </div>
  );
}
