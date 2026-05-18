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
      <section className="glass p-7 md:p-10">
        <div className="mb-7 flex flex-wrap items-end justify-between gap-3">
          <h2 className="font-display text-2xl text-[color:var(--ink)] md:text-3xl">
            <span className="italic-display text-[color:var(--teal)]">Begin</span>{" "}
            an analysis
          </h2>
          <span className="text-xs text-[color:var(--muted)]">
            5-stage pipeline · responds in ~30–90s
          </span>
        </div>
        <AnalyzeForm pending={status.kind === "loading"} onSubmit={handleSubmit} />
      </section>

      {status.kind === "idle" && <EmptyState />}

      {status.kind === "loading" && <LoadingPanel />}

      {status.kind === "error" && (
        <aside role="alert" className="glass flex items-center gap-4 p-6 md:p-7">
          <span
            aria-hidden
            className="grid h-11 w-11 flex-none place-items-center rounded-full text-lg"
            style={{
              background: "linear-gradient(135deg, var(--clay-soft), var(--paper-3))",
              color: "var(--clay-deep)",
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.6), 0 6px 18px -10px rgba(183,82,43,0.45)",
            }}
          >
            !
          </span>
          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium text-[color:var(--clay-deep)]">
              Something didn&rsquo;t go to plan
            </p>
            <p className="text-sm text-[color:var(--ink-soft)]">{status.message}</p>
          </div>
        </aside>
      )}

      {status.kind === "success" && (
        <div className="flex flex-col gap-8">
          <div className="rise"><WarningsBanner warnings={status.result.warnings} /></div>
          <div className="rise delay-1"><ScoreDial score={status.result.score} /></div>
          {status.result.reasoning && (
            <div className="rise delay-2">
              <ReasonsList reasons={status.result.reasoning.reasons} />
            </div>
          )}
          {status.result.skill_report && (
            <div className="rise delay-3">
              <SkillMatchPanel report={status.result.skill_report} />
            </div>
          )}
          {status.result.reasoning && (
            <div className="rise delay-4">
              <RewriteCards rewrites={status.result.reasoning.rewrites} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function LoadingPanel() {
  return (
    <section
      aria-live="polite"
      className="glass flex flex-col items-center justify-center gap-5 p-14 text-center"
    >
      <div className="orbit" aria-hidden />
      <div className="flex flex-col items-center gap-2">
        <h3 className="font-display text-2xl italic text-[color:var(--ink)]">
          reading the room…
        </h3>
        <p className="max-w-md text-sm text-[color:var(--muted)]">
          extract · score · match · reason · rewrite
        </p>
        <p className="max-w-md pt-1 text-sm text-[color:var(--ink-soft)]">
          The first run can take up to 90 seconds while the model warms.
        </p>
      </div>
    </section>
  );
}
