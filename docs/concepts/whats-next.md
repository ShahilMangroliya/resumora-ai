# What to Learn Next — RAG, Agents, LLMOps

> Resumora AI is deliberately a launchpad. The skills it exercises —
> fine-tuning, embeddings, local LLMs, structured outputs — are the
> foundations the three most-demanded next topics build on. This
> document points you in the right direction without pretending to
> teach them in full.

## RAG — Retrieval-Augmented Generation

**What it is:** Instead of asking an LLM a question and hoping its
training data covered it, you first *retrieve* relevant context from
your own data (using embeddings), then *augment* the prompt with that
context, then *generate* the answer.

```
question → embed → vector search over your docs → top-K chunks →
    [system prompt + retrieved chunks + question] → LLM → answer
```

**Why it matters:** It's how every "chat with your docs" / "AI
assistant over our knowledge base" / customer-support bot is built in
2026.

**What you already know:**
- Stage 4 of Resumora already uses embeddings + cosine similarity.
  That's the *retrieval* in RAG. Add a chunked document corpus and a
  vector index and you have RAG.
- Stage 5 already does prompted generation. That's the *generation*
  in RAG. Pass it retrieved context instead of structured fields and
  you have RAG.

**Next steps:**
- Add a small vector DB to Resumora (FAISS in-process or pgvector if
  you want SQL) and ingest a corpus of resume *examples*. Retrieve
  similar past resumes when reasoning about the current one.

**Go deeper:**
- [HF Cookbook — RAG recipes](https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain) — concrete notebooks.
- [Lilian Weng — LLM-powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — has a great RAG primer in the retrieval section.
- [Pinecone — RAG learning hub](https://www.pinecone.io/learn/retrieval-augmented-generation/) — clean conceptual writeup.

## Agents

**What it is:** An *agent* is an LLM in a loop that can call tools.
Instead of one-shot input → output, the model alternates between
*"think"* steps and *"act"* steps (where an "act" is calling a Python
function, an API, a database).

```
loop:
    LLM emits either:
       - "answer: ..."   → done
       - "call tool X with args Y"
            ↓
        Python executes tool, returns result
            ↓
        result fed back to LLM as next "user" message
```

**Why it matters:** This is the architectural shift behind everything
labeled "AI agent" — research assistants, coding agents (Claude Code
itself!), autonomous workflows.

**What you already know:**
- Stages 2 and 5 are one-shot LLM calls. That's the *building block*.
  Turn a one-shot call into a loop with tool-use and you have an
  agent.
- The Pydantic-validated structured output from stage 2 is exactly
  the same pattern as tool-call arguments in an agent.

**Next steps:**
- Add a tool to Resumora that lets the LLM *ask follow-up questions*
  about ambiguous resume fields, instead of one-shot extraction.
- Or: build a separate "Resumora coach" that loops through suggesting
  improvements, asking which to apply, and rewriting.

**Go deeper:**
- [Anthropic — Building effective agents](https://www.anthropic.com/news/building-effective-agents) — best framework for thinking about when and how to use agentic patterns.
- [Lilian Weng — LLM-powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — academic survey, exhaustive.
- [HF Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction) — practical, runnable.
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — the most popular framework for building agent loops with explicit state machines.

## LLMOps

**What it is:** "MLOps for LLM systems." The boring-but-essential
operational stack around production LLM features: evaluation
datasets, prompt versioning, drift detection, latency monitoring,
cost monitoring, A/B testing prompts.

**Why it matters:** A demo can be vibes-driven; a product can't.
LLMOps is what separates the two.

**What you already know:**
- `data/gold/` in Resumora is the seed of an **eval set** — small,
  hand-curated, never used for training.
- MLflow logging in `packages/training/` is the start of **experiment
  tracking**.
- The graceful-degradation pattern in `apps/api/src/api/orchestrator.py`
  (return warnings when Ollama is down) is the start of **observability**.

**Next steps:**
- Build a proper eval harness: a CLI that loads `data/gold/`, runs the
  full pipeline, and reports score accuracy, extraction F1, and a
  qualitative comparison of reasoning bullets.
- Wire latency + error metrics into the API via OpenTelemetry.
- Hash + version the prompts; log which prompt-hash served each request.

**Go deeper:**
- [Hamel Husain — Evals are all you need](https://hamel.dev/blog/posts/evals/) — the single best argument for taking evaluation seriously.
- [Eugene Yan — Building LLM Systems](https://eugeneyan.com/writing/llm-patterns/) — comprehensive overview of production patterns.
- [LangSmith / OpenLLMetry / Weights & Biases Traces](https://www.langchain.com/langsmith) — pick one observability tool and learn it.
- [MLflow LLM tracking docs](https://mlflow.org/docs/latest/llms/index.html) — extends what's already wired up in this project.

## A reasonable order to take them in

If you're moving on from Resumora AI:

1. **RAG first.** It's the smallest jump from what's already in the
   codebase and shows up in 80% of "build me an AI feature" requests.
2. **Agents second.** The conceptual jump from "LLM call" to "LLM
   loop" is the single biggest one in modern LLM engineering.
3. **LLMOps in parallel with whichever you build.** You'll need it
   the moment anyone other than you uses your system. Don't bolt it
   on at the end.

## Related concepts in this repo

- [Embeddings](./embeddings.md), [Sentence-transformers](./sentence-transformers.md) — the retrieval half of RAG.
- [Prompting](./prompting.md), [LLMs and Ollama](./llms-and-ollama.md) — the generation half of RAG and the building block of agents.
- [Fine-tuning](./fine-tuning.md), [LoRA and PEFT](./lora-and-peft.md) — sometimes the right answer is fine-tuning a smaller model instead of prompting a bigger one.
