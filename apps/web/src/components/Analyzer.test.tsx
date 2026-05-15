import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AnalyzeResponse } from "@/lib/types";

const { analyzeResumeMock, ApiErrorClass } = vi.hoisted(() => {
  const analyzeResumeMock = vi.fn();
  class ApiErrorClass extends Error {
    readonly status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
      this.name = "ApiError";
    }
  }
  return { analyzeResumeMock, ApiErrorClass };
});

vi.mock("@/lib/api", () => ({
  analyzeResume: analyzeResumeMock,
  ApiError: ApiErrorClass,
}));

import { Analyzer } from "./Analyzer";

const FULL_RESULT: AnalyzeResponse = {
  score: {
    score: 72.5,
    confidence: 0.84,
    class_probabilities: { weak: 0.05, partial: 0.11, strong: 0.84 },
    predicted_label: "strong",
  },
  skill_report: {
    required_matched: [
      { jd_skill: "python", resume_skill: "python", similarity: 0.99, matched: true },
    ],
    required_missing: [],
    nice_to_have_matched: [],
    nice_to_have_missing: [],
    match_rate: 1.0,
  },
  reasoning: {
    reasons: [
      { summary: "Strong Python match.", evidence: "Resume lists Python prominently.", category: "matched_skill" },
      { summary: "Backend depth aligns.", evidence: "5+ years of backend.", category: "experience_match" },
      { summary: "Cloud experience matches.", evidence: "AWS history.", category: "matched_skill" },
    ],
    rewrites: [
      { original: "Built services.", rewritten: "Built FastAPI services serving 10k req/s.", rationale: "Quantified." },
      { original: "Did ETL.", rewritten: "Owned ETL pipelines for 2 TB/day.", rationale: "Quantified scope." },
      { original: "Worked with Docker.", rewritten: "Shipped Docker-based deployments on EKS.", rationale: "Adds the K8s the JD asked for." },
    ],
  },
  warnings: [],
};

const PARTIAL_RESULT: AnalyzeResponse = {
  score: FULL_RESULT.score,
  skill_report: null,
  reasoning: null,
  warnings: ["Profile extraction failed; downstream stages skipped"],
};

async function fillFormAndSubmit() {
  const user = userEvent.setup();
  const file = new File(["resume"], "resume.pdf", { type: "application/pdf" });
  await user.upload(screen.getByLabelText(/^resume$/i), file);
  await user.type(screen.getByLabelText(/job description/i), "a real JD body");
  // userEvent.click on a submit button doesn't reliably fire the form submit
  // event under React 19 + jsdom; fireEvent.submit on the form does.
  const form = screen.getByRole("button", { name: /^analyze$/i }).closest("form");
  if (!form) throw new Error("submit button is not inside a form");
  fireEvent.submit(form);
  return user;
}

describe("Analyzer", () => {
  beforeEach(() => {
    analyzeResumeMock.mockReset();
  });

  afterEach(() => {
    analyzeResumeMock.mockReset();
  });

  it("shows the empty state on first render", () => {
    render(<Analyzer />);
    expect(screen.getByText(/no analysis yet/i)).toBeInTheDocument();
  });

  it("disables the submit button until both resume and JD are provided", async () => {
    const user = userEvent.setup();
    render(<Analyzer />);
    expect(screen.getByRole("button", { name: /^analyze$/i })).toBeDisabled();

    const file = new File(["resume"], "resume.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText(/^resume$/i), file);
    expect(screen.getByRole("button", { name: /^analyze$/i })).toBeDisabled();

    await user.type(screen.getByLabelText(/job description/i), "a JD");
    expect(screen.getByRole("button", { name: /^analyze$/i })).toBeEnabled();
  });

  it("calls analyzeResume with the form values on submit", async () => {
    analyzeResumeMock.mockResolvedValueOnce(FULL_RESULT);
    render(<Analyzer />);
    await fillFormAndSubmit();

    expect(analyzeResumeMock).toHaveBeenCalledTimes(1);
    const args = analyzeResumeMock.mock.calls[0][0];
    expect(args.resume).toBeInstanceOf(File);
    expect(args.resume.name).toBe("resume.pdf");
    expect(args.jobDescription).toBe("a real JD body");
  });

  it("renders the full result panels on a successful analysis", async () => {
    analyzeResumeMock.mockResolvedValueOnce(FULL_RESULT);
    render(<Analyzer />);
    await fillFormAndSubmit();

    expect(await screen.findByText("72.5")).toBeInTheDocument();
    expect(screen.getByText(/why this score/i)).toBeInTheDocument();
    expect(screen.getByText(/skill match/i)).toBeInTheDocument();
    expect(screen.getByText(/bullet rewrites/i)).toBeInTheDocument();
    expect(screen.queryByText(/partial result/i)).not.toBeInTheDocument();
  });

  it("renders only the score and a warnings banner on a partial result", async () => {
    analyzeResumeMock.mockResolvedValueOnce(PARTIAL_RESULT);
    render(<Analyzer />);
    await fillFormAndSubmit();

    expect(await screen.findByText(/partial result/i)).toBeInTheDocument();
    expect(screen.getByText("72.5")).toBeInTheDocument();
    expect(screen.queryByText(/why this score/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/skill match/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/bullet rewrites/i)).not.toBeInTheDocument();
  });

  it("renders an error banner when analyzeResume throws an ApiError", async () => {
    analyzeResumeMock.mockRejectedValueOnce(
      new ApiErrorClass(0, "Could not reach the ResumeFit API. Is the backend running?"),
    );
    render(<Analyzer />);
    await fillFormAndSubmit();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/could not reach the resumefit api/i);
  });
});
