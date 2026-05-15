"use client";

import { useState } from "react";

interface AnalyzeFormProps {
  pending: boolean;
  onSubmit: (input: { resume: File; jobDescription: string }) => void;
}

export function AnalyzeForm({ pending, onSubmit }: AnalyzeFormProps) {
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");

  const canSubmit = resume !== null && jobDescription.trim().length > 0 && !pending;

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!resume || !jobDescription.trim()) return;
    onSubmit({ resume, jobDescription });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <label className="flex flex-col gap-2">
        <span className="text-sm font-medium">Resume</span>
        <input
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={(event) => setResume(event.target.files?.[0] ?? null)}
          className="block w-full rounded-md border border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-sm file:text-white hover:file:bg-zinc-800 dark:file:bg-zinc-100 dark:file:text-zinc-900"
          required
        />
        <span className="text-xs text-[color:var(--muted)]">PDF, DOCX, or plain text.</span>
      </label>

      <label className="flex flex-col gap-2">
        <span className="text-sm font-medium">Job description</span>
        <textarea
          value={jobDescription}
          onChange={(event) => setJobDescription(event.target.value)}
          rows={10}
          placeholder="Paste the job description here…"
          className="block w-full resize-y rounded-md border border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-sm leading-6 focus:outline-none focus:ring-2 focus:ring-zinc-500"
          required
        />
      </label>

      <button
        type="submit"
        disabled={!canSubmit}
        className="self-start rounded-md bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-400 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
      >
        {pending ? "Analyzing…" : "Analyze"}
      </button>
    </form>
  );
}
