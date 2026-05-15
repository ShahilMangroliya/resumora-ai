import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ScoreResult } from "@/lib/types";

import { ScoreDial } from "./ScoreDial";

function makeScore(overrides: Partial<ScoreResult> = {}): ScoreResult {
  return {
    score: 72.5,
    confidence: 0.84,
    class_probabilities: { weak: 0.05, partial: 0.11, strong: 0.84 },
    predicted_label: "strong",
    ...overrides,
  };
}

describe("ScoreDial", () => {
  it("renders the score value with one decimal", () => {
    render(<ScoreDial score={makeScore({ score: 72.5 })} />);
    expect(screen.getByText("72.5")).toBeInTheDocument();
  });

  it("renders the label copy that maps to predicted_label", () => {
    const { rerender } = render(<ScoreDial score={makeScore({ predicted_label: "weak" })} />);
    expect(screen.getByText(/weak fit/i)).toBeInTheDocument();

    rerender(<ScoreDial score={makeScore({ predicted_label: "partial" })} />);
    expect(screen.getByText(/partial fit/i)).toBeInTheDocument();

    rerender(<ScoreDial score={makeScore({ predicted_label: "strong" })} />);
    expect(screen.getByText(/strong fit/i)).toBeInTheDocument();
  });

  it("renders rounded confidence and class probabilities", () => {
    render(
      <ScoreDial
        score={makeScore({
          confidence: 0.836,
          class_probabilities: { weak: 0.05, partial: 0.11, strong: 0.84 },
        })}
      />,
    );
    // 0.836 → 84%, weak 5%, partial 11%, strong 84%
    expect(screen.getByText(/confidence 84%/i)).toBeInTheDocument();
    expect(screen.getByText(/weak 5%/i)).toBeInTheDocument();
    expect(screen.getByText(/partial 11%/i)).toBeInTheDocument();
    expect(screen.getByText(/strong 84%/i)).toBeInTheDocument();
  });

  it("exposes the section as a labeled region for assistive tech", () => {
    render(<ScoreDial score={makeScore()} />);
    expect(screen.getByRole("region", { name: /fit score/i })).toBeInTheDocument();
  });
});
