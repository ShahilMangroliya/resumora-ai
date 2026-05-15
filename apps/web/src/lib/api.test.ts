import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { analyzeResume, ApiError } from "./api";
import type { AnalyzeResponse } from "./types";

const SAMPLE_RESPONSE: AnalyzeResponse = {
  score: {
    score: 72.5,
    confidence: 0.84,
    class_probabilities: { weak: 0.05, partial: 0.11, strong: 0.84 },
    predicted_label: "strong",
  },
  skill_report: null,
  reasoning: null,
  warnings: [],
};

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  });
}

describe("analyzeResume", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    fetchMock.mockReset();
    vi.unstubAllGlobals();
  });

  it("posts a multipart body to /analyze and returns the parsed JSON on success", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(SAMPLE_RESPONSE));

    const file = new File(["resume bytes"], "resume.pdf", { type: "application/pdf" });
    const result = await analyzeResume({ resume: file, jobDescription: "JD body" });

    expect(result).toEqual(SAMPLE_RESPONSE);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/analyze$/);
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    const body = init.body as FormData;
    expect(body.get("resume")).toBe(file);
    expect(body.get("job_description")).toBe("JD body");
  });

  it("throws ApiError carrying the server's detail message on 4xx with JSON body", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: "resume upload missing a filename" }, { status: 400 }),
    );

    const file = new File(["x"], "resume.pdf");
    await expect(analyzeResume({ resume: file, jobDescription: "jd" })).rejects.toMatchObject({
      name: "ApiError",
      status: 400,
      message: "resume upload missing a filename",
    });
  });

  it("falls back to a generic message when the error body is not JSON", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response("internal server error", { status: 500, headers: { "content-type": "text/plain" } }),
    );

    const file = new File(["x"], "resume.pdf");
    await expect(analyzeResume({ resume: file, jobDescription: "jd" })).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: "Request failed with status 500.",
    });
  });

  it("throws ApiError(0, ...) when fetch itself rejects (network failure)", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    const file = new File(["x"], "resume.pdf");
    await expect(analyzeResume({ resume: file, jobDescription: "jd" })).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
      message: "Could not reach the ResumeFit API. Is the backend running?",
    });
  });

  it("attaches `status` on the thrown ApiError", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "bad" }, { status: 422 }));

    const file = new File(["x"], "resume.pdf");
    try {
      await analyzeResume({ resume: file, jobDescription: "jd" });
      throw new Error("should have thrown");
    } catch (cause) {
      expect(cause).toBeInstanceOf(ApiError);
      expect((cause as ApiError).status).toBe(422);
    }
  });
});
