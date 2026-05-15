export type PredictedLabel = "weak" | "partial" | "strong";

export interface ScoreResult {
  score: number;
  confidence: number;
  class_probabilities: Record<PredictedLabel, number>;
  predicted_label: PredictedLabel;
}

export interface SkillMatch {
  jd_skill: string;
  resume_skill: string;
  similarity: number;
  matched: boolean;
}

export interface SkillMatchReport {
  required_matched: SkillMatch[];
  required_missing: SkillMatch[];
  nice_to_have_matched: SkillMatch[];
  nice_to_have_missing: SkillMatch[];
  match_rate: number;
}

export type ReasonCategory =
  | "matched_skill"
  | "missing_skill"
  | "experience_match"
  | "experience_gap"
  | "other";

export interface Reason {
  summary: string;
  evidence: string;
  category: ReasonCategory;
}

export interface BulletRewrite {
  original: string;
  rewritten: string;
  rationale: string;
}

export interface ReasoningResult {
  reasons: Reason[];
  rewrites: BulletRewrite[];
}

export interface AnalyzeResponse {
  score: ScoreResult;
  skill_report: SkillMatchReport | null;
  reasoning: ReasoningResult | null;
  warnings: string[];
}
