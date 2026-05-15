# LLMs and Ollama

> **TL;DR** — An **LLM** is a *generative* transformer: give it text,
> it produces more text. **Ollama** is a small local server that hosts
> open-weights LLMs (Llama, Qwen, Mistral, etc.) and exposes them via
> HTTP. From your Python code, calling Ollama is just a `requests.post`
> — no API key, no rate limit, no cost.

## Why this exists

You have two ways to use an LLM:

1. **Hosted APIs** (OpenAI, Anthropic, Google) — instant, world-class
   quality, paid per token, requires sending your users' data to a
   third party.
2. **Self-hosted, open-weights models** — slower, smaller, free,
   private. You run them on your own machine or GPU.

Resumora is open-source and meant to be self-contained, so it uses
option 2. The trade-off is that Llama 3.2 3B is much less capable
than GPT-4 — but it's *good enough* for structured extraction and
short reasoning, which is what we need.

## What "LLM" actually means

A few specific things:

- **Decoder-only transformer.** Architecturally the same family as
  GPT-2/3/4 and Claude. See [transformers](./transformers.md).
- **Autoregressive generation.** It predicts the next token, you
  append it to the input, predict the next one, and so on. That's why
  LLM inference is sequential and slower than encoder inference.
- **Trained on lots of internet text plus instruction-following
  fine-tuning.** That's what lets it follow your prompt — the base
  pretrained model just continues text; the *instruct-tuned* version
  responds to commands.
- **Probabilistic.** Same prompt → different outputs. Controlled by
  `temperature` (randomness), `top_p`, `top_k`, etc.

## Open-weights vs open-source

A subtle distinction worth getting right:

- **Open weights** — the trained model parameters are downloadable.
  Llama 3.2, Qwen, Mistral, Gemma all qualify.
- **Open source** — the model *training* code, data, and weights are
  all open. Few real LLMs meet this bar (Pythia, OLMo, the BigScience
  models).

Resumora uses *open-weights* models via Ollama. Good enough for the
self-hosting goal; not perfectly principled in the open-source sense.

## What Ollama actually is

A small Go binary that:

1. Stores models on your disk (`~/.ollama/models/`).
2. Loads them into RAM (or VRAM) on demand, quantized (compressed) so
   they fit on consumer hardware. Llama 3.2 3B ships as ~2GB by
   default.
3. Exposes an HTTP API at `localhost:11434`:

```
POST /api/chat        ← conversational interface (system + user + assistant)
POST /api/generate    ← raw completion interface
POST /api/embeddings  ← embedding interface (alternative to sentence-transformers)
GET  /api/tags        ← list installed models
```

Quantization is the trick that makes this practical on a MacBook —
storing weights as 4-bit ints instead of 16-bit floats, losing a small
amount of quality for 4× less memory.

## Where it lives in Resumora AI

- `packages/pipeline/src/pipeline/extraction/client.py` — wraps the Ollama HTTP API in a typed Python client.
- `packages/pipeline/src/pipeline/extraction/prompts.py` — the system prompt used to make Llama return JSON with extracted skills/roles.
- `packages/pipeline/src/pipeline/extraction/extract.py` — calls the client, parses + validates the JSON response.
- `packages/pipeline/src/pipeline/reasoning/reasoner.py` — second LLM
  call, generates the "three reasons + three bullet rewrites" output.
- `apps/api/src/api/config.py` — `RESUMORA_AI_OLLAMA_URL`, `RESUMORA_AI_OLLAMA_MODEL`, `RESUMORA_AI_OLLAMA_TIMEOUT` settings.
- Setup: [`docs/ollama-setup.md`](../ollama-setup.md).

## Worth knowing

- **First request is slow.** Ollama lazy-loads models on the first
  call. Subsequent calls reuse the loaded model in memory.
- **Models survive after generation.** Ollama keeps the model loaded
  for a few minutes after the last call. You can tune this with
  `OLLAMA_KEEP_ALIVE`.
- **Quality varies a lot model-to-model.** Llama 3.2 3B is fine for
  structured extraction but mediocre at long-form reasoning. Try
  `qwen2.5:3b` or step up to `llama3.1:8b` if your machine can hold
  it.
- **JSON output is not guaranteed.** The model *usually* obeys "return
  JSON" but sometimes returns prose. Always validate with Pydantic and
  have a retry/fallback path — see [prompting](./prompting.md).
- **The Ollama HTTP API is *not* the OpenAI API.** It accepts a
  similar-looking JSON body but the field names differ slightly. If
  you need OpenAI-compatible endpoints, Ollama exposes those at
  `/v1/*`.
- **Streaming is supported.** For long reasoning outputs, set
  `stream: true` and consume newline-delimited JSON. Resumora doesn't
  use streaming because the response is short.

## Hands-on

After `make dev` is running, in a separate terminal:

```bash
# Quick sanity check
curl http://localhost:11434/api/tags

# Direct chat
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
  "stream": false
}'
```

Then change the system prompt in
`pipeline/reasoning/prompts.py` and rerun `/analyze` to see how output
shifts.

## Go deeper

- [Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) — full HTTP API reference.
- [Ollama model library](https://ollama.com/library) — every model you can pull.
- [Meta — Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) — what we actually run.
- [Hugging Face LLM course](https://huggingface.co/learn/llm-course) — covers decoder-only LLMs end-to-end.
- [Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — visual explainer of decoder transformers.
- [Sebastian Raschka — "Understanding Large Language Models"](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — Sebastian's curated LLM reading list.

Related concepts: [Transformers](./transformers.md), [Prompting](./prompting.md), [Embeddings](./embeddings.md) (Ollama can do these too).
