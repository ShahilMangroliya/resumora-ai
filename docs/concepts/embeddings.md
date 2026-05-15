# Embeddings

> **TL;DR** — An embedding is a fixed-length vector of floats (~384
> or ~768 dims) that represents the *meaning* of a piece of text.
> Texts with similar meaning have similar vectors. Comparing
> embeddings with cosine similarity is the foundation of "semantic
> search", "RAG", and what stage 4 of this pipeline does.

## Why this exists

Keyword search fails when the words don't literally match. A resume
that says *"Built React components"* and a JD that asks for
*"frontend development experience"* are clearly related — but they
share no words. You can't fix this with regex.

Embeddings solve it by mapping text into a vector space where
*meaning* (not surface form) determines closeness:

```
"React"               ┐
"frontend"            ├──> roughly the same direction
"Vue"                 ┘
"PostgreSQL"          ──> a different direction
"orange juice"        ──> very different direction
```

This is what stage 4 of the pipeline does to compare the *skills the
resume lists* to the *skills the JD asks for*.

## How it works (just enough)

1. A pretrained model — typically a *bi-encoder* (see
   [sentence-transformers](./sentence-transformers.md)) — takes a
   string and outputs a fixed-length vector.
2. To compare two strings, embed both, then compute
   **cosine similarity**: the cosine of the angle between the
   vectors. Range `[-1, 1]`, where `1.0` = identical direction
   (synonymous), `0` = unrelated, `-1` = opposite.
3. For storing and searching many embeddings, you use a *vector
   database* (FAISS, Qdrant, pgvector, etc.). Resumora doesn't need
   one — it only compares two short lists.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
v_resume = model.encode("Built React components")
v_jd     = model.encode("frontend development experience")
sim = np.dot(v_resume, v_jd) / (np.linalg.norm(v_resume) * np.linalg.norm(v_jd))
# ~0.5 — clearly related, not identical
```

## Where it lives in Resumora AI

- `packages/pipeline/src/pipeline/similarity/_embeddings.py` — wraps
  the sentence-transformer model so the rest of the pipeline gets a
  simple `embed(texts) -> np.ndarray` interface.
- `packages/pipeline/src/pipeline/similarity/matcher.py` — computes
  cosine similarity between extracted resume skills and JD skills,
  classifies each as *matched* / *missing* / *nice-to-have*.
- The model used is `sentence-transformers/all-MiniLM-L6-v2`: 22M
  parameters, 384-dim vectors, fast on CPU.

## Worth knowing

- **Cosine similarity, not Euclidean distance.** With normalized
  vectors they're equivalent, but cosine is the convention.
- **Similarity is not a probability.** A score of `0.7` doesn't mean
  "70% match" — it's a number on a scale that depends on the
  embedding model. Always set thresholds empirically.
- **Short text behaves differently from long text.** A single skill
  ("Kubernetes") embeds differently than a paragraph. Mixing
  granularities silently degrades quality.
- **Different models, different vector spaces.** You can't compare a
  MiniLM vector to an OpenAI `text-embedding-3-small` vector. Choose
  one embedding model and use it everywhere.
- **Embeddings are deterministic.** Same input → same vector
  (modulo floating-point on different hardware). You can cache them.

## Hands-on

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
items = [
    "React",
    "frontend",
    "Vue.js",
    "PostgreSQL",
    "orange juice",
]
vecs = m.encode(items)
print(cosine_similarity(vecs).round(2))
```

You'll see a similarity matrix. The web/frontend cluster should score
high amongst themselves, and "orange juice" should be low against
everything.

## Go deeper

- [Sentence-Transformers documentation](https://www.sbert.net/) — practical, code-first.
- [Hugging Face NLP Course — Chapter 5 (semantic search)](https://huggingface.co/learn/nlp-course/chapter5/6).
- [Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — the original embedding intuition, before transformers.
- [Sentence-BERT paper (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084) — why a *separate* embedding model exists, instead of using BERT directly.
- [Massive Text Embedding Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — when you're ready to pick an embedding model for a real workload.

Related concepts: [Sentence-transformers](./sentence-transformers.md) (the library/model family), [Transformers](./transformers.md) (the underlying architecture).
