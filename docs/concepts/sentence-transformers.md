# Sentence-Transformers

> **TL;DR** — `sentence-transformers` is a library (and a family of
> models) for producing high-quality **sentence-level embeddings** — a
> single vector that represents the meaning of a whole sentence or
> short paragraph. Stage 4 of the pipeline uses it to compare resume
> skills against JD skills.

## Why this exists

BERT was designed for *word-level* tasks. If you feed BERT a sentence
and grab a vector out, it's not a great sentence embedding by
default — for some compare-two-sentences tasks, BERT performed *worse*
than averaging GloVe word vectors.

The 2019 Sentence-BERT paper showed that fine-tuning BERT with a
**siamese network** on sentence-pair data produces dramatically better
sentence embeddings. The `sentence-transformers` library packages
that approach with hundreds of pretrained models, ready to use.

Today, when someone says "I need embeddings for semantic search," they
almost certainly mean a sentence-transformers model or an equivalent
hosted alternative (OpenAI, Voyage, Cohere).

## Cross-encoders vs bi-encoders

A subtle but important distinction:

- **Cross-encoder** — takes *two* inputs jointly, outputs one score.
  Higher quality, but you can't precompute embeddings, so it scales
  badly. Useful for re-ranking a top-50 list.
- **Bi-encoder** — takes *one* input, outputs one vector. Embed each
  side independently, then compare with cosine similarity. Scales to
  millions of items because you can cache and index the vectors.

Sentence-transformers ships both, but the canonical use is the
bi-encoder. That's what Resumora uses.

## How it works (just enough)

1. A pretrained encoder (often MiniLM or MPNet) takes the sentence
   through its transformer stack.
2. Token-level outputs get reduced to a single vector via **mean
   pooling** (sometimes max or `[CLS]`).
3. Optionally L2-normalized so that dot product equals cosine
   similarity.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vec = model.encode("Built React components")
# vec.shape -> (384,)
# Already normalized — dot product = cosine similarity.
```

## The model Resumora uses

`sentence-transformers/all-MiniLM-L6-v2`:

- 22 M parameters (vs. DistilBERT's 67 M).
- 384-dim output (vs. BERT's 768).
- Fast on CPU — ~1ms per sentence on a laptop.
- Trained on 1 B+ sentence pairs from the public web.
- Great default for general semantic similarity. Not best-in-class
  anymore (MTEB leaderboard has newer winners), but a strong
  cost/quality sweet spot.

You could swap in `BAAI/bge-small-en-v1.5` or
`sentence-transformers/all-mpnet-base-v2` for better quality at higher
latency.

## Where it lives in Resumora AI

- `packages/pipeline/src/pipeline/similarity/_embeddings.py` — thin
  wrapper around `SentenceTransformer`, normalizes the call.
- `packages/pipeline/src/pipeline/similarity/matcher.py` — the actual
  business logic: embed both lists of skills, compute pairwise cosine
  similarity, threshold to *matched / missing / extra*.
- `packages/pipeline/src/pipeline/similarity/models.py` — Pydantic
  output shape (`SkillReport.required`, `.nice_to_have`).
- `apps/api/src/api/config.py` — `RESUMORA_AI_MATCHER_DEVICE` setting
  (`cpu` / `cuda` / `mps`).

## Worth knowing

- **`encode()` is batched-friendly.** Pass a list of strings, get a
  matrix. Much faster than calling per-string in a loop.
- **Don't mix model families.** Vectors from MiniLM can't be compared
  to vectors from OpenAI's `text-embedding-3-small`. Pick one
  family and stick to it across the entire pipeline.
- **Short text behaves differently from long text.** Single skill
  tokens ("React") embed less reliably than 1–2 sentence summaries.
  If quality matters, embed *short skill descriptions* rather than
  bare tokens.
- **Asymmetric search needs special models.** Some tasks (find a long
  *document* given a short *query*) work better with models trained
  specifically for asymmetry (e.g. `multi-qa-MiniLM`).
- **The Hub has hundreds of these.** [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks them on standardized benchmarks.

## Hands-on

```python
from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

resume_skills = ["React", "TypeScript", "Node.js", "PostgreSQL"]
jd_required   = ["frontend frameworks", "JavaScript", "REST APIs"]

r = m.encode(resume_skills, normalize_embeddings=True)
j = m.encode(jd_required,   normalize_embeddings=True)

similarity = r @ j.T          # cosine sim, because vectors are normalized
print(similarity.round(2))
# rows = resume skills, cols = JD requirements
```

You'll see "React" lights up against "frontend frameworks", "Node.js"
against "JavaScript", and "PostgreSQL" stays low against everything —
exactly what stage 4 is doing internally.

## Go deeper

- [SBERT.net — official documentation](https://www.sbert.net/) — the library, in depth.
- [Sentence-BERT paper (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084) — the original idea. Short.
- [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) — when picking a model for real work.
- [HF blog — Train your own sentence-transformer](https://huggingface.co/blog/how-to-train-sentence-transformers) — if you ever want to fine-tune one for a domain.
- [MiniLM paper (Wang et al., 2020)](https://arxiv.org/abs/2002.10957) — how the model we use was distilled.

Related concepts: [Embeddings](./embeddings.md), [Transformers](./transformers.md), [Hugging Face Hub](./hugging-face-hub.md).
