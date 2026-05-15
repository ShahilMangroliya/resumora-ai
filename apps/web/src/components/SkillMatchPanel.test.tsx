import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { SkillMatchReport } from "@/lib/types";

import { SkillMatchPanel } from "./SkillMatchPanel";

function makeReport(overrides: Partial<SkillMatchReport> = {}): SkillMatchReport {
  return {
    required_matched: [
      { jd_skill: "python", resume_skill: "python", similarity: 0.99, matched: true },
    ],
    required_missing: [
      { jd_skill: "kubernetes", resume_skill: "docker", similarity: 0.4, matched: false },
    ],
    nice_to_have_matched: [],
    nice_to_have_missing: [],
    match_rate: 0.5,
    ...overrides,
  };
}

describe("SkillMatchPanel", () => {
  it("renders match-rate as a rounded percentage", () => {
    render(<SkillMatchPanel report={makeReport({ match_rate: 0.6666 })} />);
    expect(screen.getByText(/67% of required skills matched/i)).toBeInTheDocument();
  });

  it("renders only non-empty chip groups", () => {
    render(<SkillMatchPanel report={makeReport()} />);
    expect(screen.getByText(/required — matched/i)).toBeInTheDocument();
    expect(screen.getByText(/required — missing/i)).toBeInTheDocument();
    expect(screen.queryByText(/nice to have — matched/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/nice to have — missing/i)).not.toBeInTheDocument();
  });

  it("shows JD skills as chips", () => {
    render(<SkillMatchPanel report={makeReport()} />);
    expect(screen.getByText("python")).toBeInTheDocument();
    expect(screen.getByText("kubernetes")).toBeInTheDocument();
  });
});
