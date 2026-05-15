import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RewriteCards } from "./RewriteCards";

describe("RewriteCards", () => {
  it("shows the 'Before' block when original is non-empty and 'Suggested addition' when empty", () => {
    render(
      <RewriteCards
        rewrites={[
          {
            original: "Built backend services.",
            rewritten: "Built and shipped FastAPI services serving 10k req/s.",
            rationale: "Quantifies impact.",
          },
          {
            original: "",
            rewritten: "Added a new bullet about Kubernetes deployments.",
            rationale: "Fills a gap the JD asked for.",
          },
          {
            original: "Worked on data.",
            rewritten: "Owned ETL pipelines processing 2 TB/day.",
            rationale: "Quantifies scope.",
          },
        ]}
      />,
    );

    // Two "Before" labels (cards 1 and 3) and one "Suggested addition" (card 2).
    expect(screen.getAllByText(/^before$/i)).toHaveLength(2);
    expect(screen.getByText(/suggested addition/i)).toBeInTheDocument();
    expect(screen.getAllByText(/^after$/i)).toHaveLength(3);
  });

  it("renders the rationale alongside each rewrite", () => {
    render(
      <RewriteCards
        rewrites={[
          { original: "a", rewritten: "A!", rationale: "Better verb." },
          { original: "b", rewritten: "B!", rationale: "Quantified." },
          { original: "c", rewritten: "C!", rationale: "JD alignment." },
        ]}
      />,
    );
    expect(screen.getByText(/better verb/i)).toBeInTheDocument();
    expect(screen.getByText(/quantified/i)).toBeInTheDocument();
    expect(screen.getByText(/jd alignment/i)).toBeInTheDocument();
  });
});
