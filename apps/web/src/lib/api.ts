import { API_URL } from "./env";
import type { AnalyzeResponse } from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export interface AnalyzeInput {
  resume: File;
  jobDescription: string;
}

export async function analyzeResume({ resume, jobDescription }: AnalyzeInput): Promise<AnalyzeResponse> {
  const body = new FormData();
  body.append("resume", resume);
  body.append("job_description", jobDescription);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/analyze`, { method: "POST", body });
  } catch {
    throw new ApiError(0, "Could not reach the Resumora AI API. Is the backend running?");
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      // body wasn't JSON; keep the generic message
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as AnalyzeResponse;
}
