import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { WarningsBanner } from "./WarningsBanner";

describe("WarningsBanner", () => {
  it("renders nothing when warnings is empty", () => {
    const { container } = render(<WarningsBanner warnings={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders each warning as a list item under the 'Partial result' header", () => {
    render(
      <WarningsBanner
        warnings={[
          "Profile extraction failed; downstream stages skipped",
          "Reasoning generation failed",
        ]}
      />,
    );

    expect(screen.getByText(/partial result/i)).toBeInTheDocument();
    expect(screen.getByText(/profile extraction failed/i)).toBeInTheDocument();
    expect(screen.getByText(/reasoning generation failed/i)).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });
});
