# Prompting and Structured Outputs

> **TL;DR** — The "prompt" is the input string you send to an LLM. A
> useful mental model: an LLM call is a **fuzzy function**, and the
> prompt is its source code. Stages 2 and 5 of this pipeline are
> entirely prompt-defined functions.

## Why this matters

You can't write `if/else` inside an LLM. The only knob you have to
control its behavior is the text you send it. Tweaking that text —
"prompt engineering" — is the difference between a flaky toy and a
production-ready stage of your pipeline.

## The chat API shape

Most modern LLMs (Llama, Qwen, Claude, GPT, etc.) accept input as a
list of **messages**, each with a `role`:

| Role | Purpose |
|---|---|
| `system` | Instructions, persona, ground rules. Sets behavior for the whole conversation. |
| `user` | What the user (or your code, acting as the user) sends. |
| `assistant` | What the model said previously, when you're continuing a multi-turn conversation. |

A single stage in Resumora is a *one-shot* function call, so you
typically have one system + one user message:

```python
messages = [
    {"role": "system", "content": "<extraction instructions>"},
    {"role": "user", "content": resume_text},
]
```

## "Function definition" mental model

Treat the system prompt as the docstring + return-type contract:

```
SYSTEM
You are an information-extraction service. Given a resume, return JSON
with this exact shape:
{
  "skills":   [string, ...],
  "roles":    [{"title": string, "years": number}, ...],
  "education": [string, ...]
}

Rules:
- Output ONLY the JSON. No prose, no markdown fences.
- If a field has no data, return an empty list.
- Years must be a number, not a string.
```

The model still might break the contract (more on that below), but
this framing makes the prompt explicit and reviewable.

## Getting structured output

The biggest reliability issue with LLM-as-function is that the model
sometimes returns prose where you wanted JSON, or invalid JSON, or
JSON that doesn't match your schema. Mitigations, in order of strength:

1. **Tell it the schema in the system prompt** — minimum bar.
2. **Few-shot examples** — show 1–3 examples of input → output in the prompt.
3. **`response_format`/JSON mode** — Ollama and most APIs accept a `format: "json"` flag that constrains the decoder to valid JSON tokens.
4. **Validate + retry on failure** — parse with Pydantic; if it
   fails, send the error back to the model and ask for a fix.
5. **Constrained decoding** — libraries like Outlines, Instructor, or
   the upcoming JSON Schema features force *valid* JSON at the
   token-sampling level. Heaviest, but bulletproof.

Resumora uses **(1) + (2) + (4)**: schema in the prompt, examples,
and Pydantic validation. Anything past that is more reliability than
this scope needs.

## Where it lives in Resumora AI

- `packages/pipeline/src/pipeline/extraction/prompts.py` — system + user templates for the *extraction* LLM call.
- `packages/pipeline/src/pipeline/reasoning/prompts.py` — system + user templates for the *reasoning* LLM call. Different shape: this one returns natural-language bullets, not strict JSON.
- `packages/pipeline/src/pipeline/extraction/extract.py` — wraps prompt → call → parse → validate. Where the contract is *enforced*.
- `packages/pipeline/src/pipeline/extraction/models.py` — Pydantic models that define what a "valid" extraction response looks like. Read these alongside the prompt; they're the contract.

## Worth knowing

- **Small wording changes have outsized effects.** Adding *"think step by step"* or *"be concise"* can swing accuracy 10%. Test, don't guess.
- **Examples beat instructions.** One concrete input/output pair usually does more than three paragraphs of rules.
- **Temperature 0 is not deterministic.** It's *low-randomness*, but model implementations still vary. Don't write tests that assert exact LLM outputs.
- **Token budget matters.** Long system prompts cost latency on every call. Trim aggressively.
- **System prompts can be ignored.** Especially in smaller models. If a rule is critical, repeat it in the user message or validate the output.
- **Prompts are code.** Version-control them. Diff them when behavior changes. Resumora's prompts live in `.py` files, not config strings, on purpose.

## Hands-on

Tweak the reasoning prompt and watch the output change:

```bash
# 1. Read packages/pipeline/src/pipeline/reasoning/prompts.py
# 2. Change the system prompt — e.g., add: "Use active voice. Max 12 words per bullet."
# 3. Re-run /analyze (`make dev` should hot-reload)
# 4. Compare the rewrites field before/after
```

For a tougher exercise: write a *bad* prompt deliberately (vague,
contradictory) and watch the model's output degrade. This is the
fastest way to build intuition for what "good prompt" means in
practice.

## Go deeper

- [Anthropic — Prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) — the best concise treatment of the discipline. (Even if you're using Llama, the principles transfer.)
- [OpenAI — Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering).
- [Lilian Weng — Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — academic-leaning survey, very comprehensive.
- [Hugging Face — Open-source prompt engineering guide](https://github.com/huggingface/cookbook/tree/main/notebooks/en) — runnable notebooks.
- [Instructor library](https://python.useinstructor.com/) — if you outgrow ad-hoc Pydantic validation. Wraps any LLM with strict typed outputs.
- [Outlines library](https://github.com/dottxt-ai/outlines) — constrained decoding for guaranteed-valid structured output.

Related concepts: [LLMs and Ollama](./llms-and-ollama.md), [What's next: agents](./whats-next.md).
