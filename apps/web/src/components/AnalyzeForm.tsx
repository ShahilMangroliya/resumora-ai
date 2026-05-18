"use client";

import { useId, useState } from "react";

interface AnalyzeFormProps {
  pending: boolean;
  onSubmit: (input: { resume: File; jobDescription: string }) => void;
}

export function AnalyzeForm({ pending, onSubmit }: AnalyzeFormProps) {
  const resumeId = useId();
  const jdId = useId();
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");

  const canSubmit = resume !== null && jobDescription.trim().length > 0 && !pending;

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!resume || !jobDescription.trim()) return;
    onSubmit({ resume, jobDescription });
  }

  const wordCount = jobDescription.trim() ? jobDescription.trim().split(/\s+/).length : 0;

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-1 gap-7 md:grid-cols-5">
      <div className="md:col-span-2 flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <label
            htmlFor={resumeId}
            className="text-sm font-medium text-[color:var(--ink)]"
          >
            Resume
          </label>
          {resume && (
            <span className="text-xs text-[color:var(--score-strong)]">✓ loaded</span>
          )}
        </div>

        <div className={`dropzone ${resume ? "dropzone--filled" : ""}`}>
          <div className="dropzone-icon" aria-hidden>
            {resume ? "✓" : "+"}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate font-display text-lg italic text-[color:var(--ink)]">
              {resume ? resume.name : "Drop your résumé here"}
            </p>
            <p className="text-xs text-[color:var(--muted)]">
              {resume ? formatBytes(resume.size) : "PDF · DOCX · TXT — parsed locally"}
            </p>
          </div>
          <input
            id={resumeId}
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={(event) => setResume(event.target.files?.[0] ?? null)}
            required
            aria-label="Resume"
          />
        </div>

        <p className="text-xs text-[color:var(--muted)]">
          We don&rsquo;t store or send your résumé anywhere. The pipeline runs on your
          own machine.
        </p>
      </div>

      <div className="md:col-span-3 flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <label htmlFor={jdId} className="text-sm font-medium text-[color:var(--ink)]">
            Job description
          </label>
          <span className="text-xs text-[color:var(--muted)] tabular">
            {wordCount} {wordCount === 1 ? "word" : "words"}
          </span>
        </div>

        <textarea
          id={jdId}
          value={jobDescription}
          onChange={(event) => setJobDescription(event.target.value)}
          rows={8}
          placeholder="Paste the full job description — title, responsibilities, required and nice-to-have skills…"
          className="field resize-y"
          required
        />

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <p className="max-w-[40ch] text-xs text-[color:var(--muted)]">
            <span className="italic-display text-sm text-[color:var(--ink-soft)]">tip ·</span>{" "}
            include the full requirements section for the sharpest score.
          </p>

          <button type="submit" disabled={!canSubmit} className="btn-soft">
            <span>{pending ? "Analyzing" : "Analyze"}</span>
            <span aria-hidden className="arrow">→</span>
          </button>
        </div>
      </div>
    </form>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
