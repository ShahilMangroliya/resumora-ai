# Learning Guide — Resumora AI for Software Engineers New to AI

This guide is for someone who can read Python and ship web apps, but has
not built ML/AI systems before. It explains **what this project is doing
under the AI hood**, in plain software-engineering terms, and points to
the exact files to read next.

If you already know the ML basics, the [README](../README.md) and the
code itself will be enough — this doc is a slower, more pedagogical
on-ramp.

> **Looking for a specific concept?** Every term you'll meet here has
> its own detailed file in [`docs/concepts/`](./concepts/README.md),
> with external links to authoritative sources. This guide is the
> narrative path; that directory is the reference shelf.

---

## 1. The 30-second mental model

Strip the AI vocabulary away and Resumora AI is a normal web app:

```
Browser  →  Next.js  →  POST /analyze  →  FastAPI  →  5-step pipeline  →  JSON
```

The pipeline is just five Python modules called one after the other.
Three of them happen to call neural networks instead of pure functions.
That's the *whole* "AI" surface area:

| Stage | What it really is | Lives in |
|---|---|---|
| 1. Ingestion | A PDF/DOCX/TXT parser → cleaned text | `packages/pipeline/src/pipeline/ingestion/` |
| 2. Extraction | An HTTP call to a local LLM, returns structured JSON | `packages/pipeline/src/pipeline/extraction/` |
| 3. Scoring | A small neural net you trained, returns a number 0–100 | `packages/pipeline/src/pipeline/scoring/` |
| 4. Similarity | Turns words into vectors, measures angle between them | `packages/pipeline/src/pipeline/similarity/` |
| 5. Reasoning | Another LLM call, this time to *write* sentences | `packages/pipeline/src/pipeline/reasoning/` |

If you've ever called the OpenAI API and then post-processed the JSON,
stages 2 and 5 are exactly that — just pointed at a local server
(Ollama) instead of a paid API. Stage 4 is "Levenshtein distance, but
for meaning." Stage 3 is the only stage where we *train our own model*.

That's it. The rest of this guide unpacks each idea once.

---

## 2. AI concepts you will meet, mapped to files

You don't need to read an ML textbook before touching this repo. Here
are the only concepts you need, each tied to a real line of code.
Each section heading links to a deeper standalone reference in
[`docs/concepts/`](./concepts/README.md).

### [Tokens and tokenizers](./concepts/tokens-and-tokenizers.md)

Computers can't feed raw text into a neural net. A **tokenizer** splits
text into integer IDs (e.g. `"resume"` → `[1234, 567]`). Think of it
as `text.encode()`, but the encoding was *learned* from data so that
common pieces of words get short codes.

- Where it shows up: `packages/pipeline/src/pipeline/scoring/loader.py`
  loads the tokenizer that ships with DistilBERT from Hugging Face.

### [Embeddings (vectors that represent meaning)](./concepts/embeddings.md)

An **embedding** is a list of ~384 or ~768 floats that represents the
"meaning" of a piece of text. Two embeddings can be compared with
cosine similarity (basically the angle between them). Closer angle =
more similar meaning.

This is how stage 4 figures out that *"PyTorch"* on a resume and
*"deep learning frameworks"* in the JD are related, without keyword
matching.

- Read: `packages/pipeline/src/pipeline/similarity/_embeddings.py` (turns text into vectors)
- Read: `packages/pipeline/src/pipeline/similarity/matcher.py` (uses cosine similarity)
- Model used: `sentence-transformers/all-MiniLM-L6-v2` — a pretrained model from Hugging Face. We don't train it; we just *use* it.

### [Transformers and "models" you use off the shelf](./concepts/transformers.md)

A **transformer** is a neural network architecture. You don't need to
understand its internals to ship this project — you need to know:

- It takes token IDs in, gives numbers out.
- People pre-train very large ones, publish them on **Hugging Face Hub**, and you can download and run them in two lines of Python.
- Different transformer variants exist for different jobs: **DistilBERT** (small, good for classification), **Llama 3.2** (a "chat" LLM, good for generation), **MiniLM** (small, good for embeddings).

Hugging Face Hub is to ML models what Docker Hub is to containers, or
what GitHub is to source.

### [Classifier (and "classification head")](./concepts/classifiers.md)

A **classifier** is a function `text → category`. In this project the
categories are `weak_fit / partial_fit / strong_fit` and the
"probability" of each is what we turn into the 0–100 score.

A transformer doesn't natively output categories — it outputs vectors.
A **classification head** is a tiny extra layer (literally a linear
layer + softmax) bolted on top that maps the vector to category
probabilities. When you "fine-tune DistilBERT for classification,"
that head is part of what gets trained.

- Read: `packages/pipeline/src/pipeline/scoring/scorer.py`
- Read: `packages/pipeline/src/pipeline/scoring/math.py` — how probabilities become a single 0–100 number.

### [Fine-tuning](./concepts/fine-tuning.md)

**Fine-tuning** = take a model someone else pretrained on the whole
internet, then keep training it a little more on *your* data so it
learns your specific task.

For us: DistilBERT already "knows English." We feed it pairs of
(resume snippet, JD snippet, label) so it learns to score *fit*. That
training does not happen in the API — it happens once, on Google
Colab, and the output gets pushed to Hugging Face Hub. The API just
downloads the trained weights at boot.

- Read: `notebooks/01_train_on_colab.ipynb` — the actual training run.
- Read: `packages/training/src/training/train/` — the importable Python the notebook calls.

### [LoRA (and why we use it)](./concepts/lora-and-peft.md)

If DistilBERT has, say, 67M parameters, fine-tuning *all* of them on a
free Colab GPU is slow and memory-hungry. **LoRA (Low-Rank
Adaptation)** is a trick: you *freeze* the original 67M parameters and
only train a small number of *new* parameters (a few hundred thousand)
that ride alongside them. Cheaper, faster, less likely to overfit on a
small dataset.

**PEFT** ("Parameter-Efficient Fine-Tuning") is the Hugging Face
library that implements LoRA.

- Read: `packages/training/src/training/train/` for how PEFT is configured.
- Read: `packages/pipeline/src/pipeline/scoring/loader.py` for how the LoRA adapter is merged back at inference time.

### [LLMs and Ollama](./concepts/llms-and-ollama.md)

An **LLM** is a generative transformer — give it a string, it writes
the continuation. **Llama 3.2** is one such model (3 billion
parameters, open weights, made by Meta).

**Ollama** is a tiny local server that hosts an LLM and exposes an
HTTP API. From this codebase's point of view, calling Llama is just:

```python
POST http://localhost:11434/api/chat
{ "model": "llama3.2:3b", "messages": [...] }
```

No API key, no cost, no rate limit. The trade-off is you run it
yourself.

- Read: `packages/pipeline/src/pipeline/extraction/client.py` (the HTTP client)
- Read: `packages/pipeline/src/pipeline/extraction/prompts.py` and `.../reasoning/prompts.py` (the actual prompts we send)
- Setup: [`docs/ollama-setup.md`](ollama-setup.md)

### [Prompts as "function definitions"](./concepts/prompting.md)

A useful mental model for new ML engineers: an LLM call is a *fuzzy
function*. The system prompt is the function signature + docstring.
The user prompt is the arguments. You hope the model returns
something parseable.

In stage 2 we ask the LLM "extract skills from this resume, return
JSON." The "function" is defined entirely by the system prompt in
`extraction/prompts.py`. There is no schema enforcement at the model
level — we *parse and validate* the JSON ourselves with Pydantic, and
retry / degrade if it's malformed.

### [Hugging Face Hub](./concepts/hugging-face-hub.md)

A model registry. You push trained weights to it; the API pulls them
at boot. Same idea as a Docker registry or an artifact store. The HF
ecosystem also hosts datasets, demo Spaces, and the libraries
(`transformers`, `datasets`, `peft`) that talk to all of it.

---

## 3. A reading order

If you only have an hour, read in this order:

1. **`README.md`** — production-style overview. The "Architecture"
   diagram is the single most important picture.
2. **`apps/api/src/api/main.py`** — 80 lines. See the HTTP entry point.
3. **`apps/api/src/api/orchestrator.py`** — how the five stages are called in sequence.
4. **`packages/pipeline/src/pipeline/ingestion/`** — start with the
   *non-AI* stage to warm up. It's just parsing.
5. **`packages/pipeline/src/pipeline/similarity/matcher.py`** — your
   first taste of embeddings. Tiny file, big idea.
6. **`packages/pipeline/src/pipeline/extraction/client.py` + `prompts.py`** — your first LLM call.
7. **`packages/pipeline/src/pipeline/scoring/scorer.py`** — your first
   "neural net inference" code. Notice how short it is.
8. **`packages/pipeline/src/pipeline/reasoning/reasoner.py`** — the
   second LLM call, this time for generation rather than extraction.
9. **`notebooks/01_train_on_colab.ipynb`** — leave this for last; it's
   the most ML-heavy file in the repo. Reading it after the rest is
   easier because you'll already understand what it's producing.

Skim, don't deep-read. The goal of the first pass is to know *where
things live*, not to memorize them.

---

## 4. Running it locally for the first time

Follow the [README Quick start](../README.md#quick-start). Three
gotchas worth calling out for a first-timer:

- **Ollama must be running before `make dev`** — see [`docs/ollama-setup.md`](ollama-setup.md). If you skip this, the score still works but extraction and reasoning will fail and the response will include warnings instead. That's by design (graceful degradation), but it's confusing if you don't know it.
- **First request is slow.** The scorer and embedding model load on the first call (or at startup if `RESUMORA_AI_WARMUP_ON_STARTUP=true`). Subsequent calls are fast.
- **You don't need a trained scorer to play.** The default
  `RESUMORA_AI_SCORER_REPO` is plain `distilbert-base-uncased` — an
  *un-fine-tuned* model. Scores will be nonsense, but everything wires
  up. Train your own (or point at a published one) to get real scores.

---

## 5. Small experiments to try after your first run

These will teach you more than reading another article. In rough
order of difficulty:

1. **Inspect the JSON.** Submit a real resume + JD via the web UI, then
   open the Network tab and look at the `/analyze` response. Match
   each field back to the stage that produced it.
2. **Change a prompt.** Tweak the system prompt in
   `pipeline/reasoning/prompts.py` to change the *style* of the rewrite
   bullets (e.g. "use active voice, max 12 words"). Re-run. Notice how
   small wording changes shift the output.
3. **Swap the LLM.** Pull a different Ollama model
   (`ollama pull qwen2.5:3b`) and set
   `RESUMORA_AI_OLLAMA_MODEL=qwen2.5:3b`. Compare quality.
4. **Read one test file end-to-end.** Pick `packages/pipeline/tests/scoring/test_scorer.py`. Tests are often the most legible documentation in a repo.
5. **Train the scorer.** Open the Colab notebook. The first time will
   feel intimidating; by the end you'll have published a model to your
   own HF Hub account and pointed the API at it.

---

## 6. Glossary cheat sheet

| Term | One-line definition |
|---|---|
| Token | An integer ID a chunk of text is encoded to before going into a model. |
| Tokenizer | The encoder that produces tokens (and decodes them back). |
| Embedding | A vector that represents the meaning of text; close vectors = similar meaning. |
| Transformer | The neural-network architecture behind BERT, GPT, Llama, etc. |
| LLM | A large generative transformer. Give it text, it writes more text. |
| Inference | Running an already-trained model. (Opposite of training.) |
| Training | Updating a model's parameters from data. |
| Fine-tuning | Training a *pretrained* model on a small amount of task-specific data. |
| LoRA / PEFT | Cheap fine-tuning: freeze the big model, train a few small extra weights. |
| Classifier | A model that outputs a category (or probabilities over categories). |
| Sentence-transformers | A library + family of models that produce embeddings of sentences. |
| Hugging Face Hub | The "GitHub for models and datasets." |
| Ollama | A local server that hosts an open-weights LLM and exposes an HTTP API. |
| MLflow | An experiment tracker — logs metrics from training runs locally. |
| Colab | Google's free Jupyter-notebooks-with-a-GPU service. We train on its T4 GPU. |
| Prompt | The text you send to an LLM, including the system + user messages. |
| System prompt | The instructions / persona you give an LLM before the user message. |
| Cosine similarity | Cosine of the angle between two vectors; 1.0 = identical direction, 0 = unrelated. |
| Graceful degradation (here) | If Ollama is down, return a score-only response with a `warnings` array instead of failing the whole request. |

---

## 7. Where to go next

After you're comfortable with this repo, the natural next concepts to
learn (and which this project is *deliberately* a launchpad for) are
covered in [`docs/concepts/whats-next.md`](./concepts/whats-next.md):

- **RAG (Retrieval-Augmented Generation)** — same embedding idea as
  stage 4, but used to *retrieve* relevant context before an LLM
  generates. Add a vector DB and you have RAG.
- **Agents** — LLMs that call tools in a loop. Stages 2 and 5 here are
  one-shot LLM calls; an agent is the same call repeated with
  feedback.
- **LLMOps** — the boring operational stuff: latency, eval sets,
  drift, prompt versioning, model registries. The `data/gold/` set and
  MLflow logging in this repo are the start of that.

The skills exercised in this codebase — fine-tuning small models with
LoRA, using HF Hub, running local LLMs, building structured pipelines
on top of LLM calls — are exactly the foundations those next topics
build on.
