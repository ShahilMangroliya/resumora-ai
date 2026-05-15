# Transformers (the architecture, not the library)

> **TL;DR** — A "transformer" is a specific neural-network
> architecture introduced in 2017. It underpins BERT, GPT, T5, Llama,
> Claude — basically everything you call "AI" today. You do not need
> to understand its internals to ship this project. You **do** need to
> know that *encoder* transformers (like DistilBERT) are good at
> understanding, and *decoder* transformers (like Llama) are good at
> generating.

## Why this exists

Before transformers, sequence models (RNNs, LSTMs) processed text
token-by-token in order. That made them slow to train (couldn't
parallelize over time), bad at long-range dependencies, and bad at GPU
utilization.

The 2017 paper *"Attention Is All You Need"* swapped recurrence for
**self-attention**: every token sees every other token in parallel,
weighted by learned "attention" scores. That single change unlocked
modern NLP.

## The three flavors

| Family | Examples | What they're good at | In this project |
|---|---|---|---|
| **Encoder-only** | BERT, DistilBERT, RoBERTa | Understanding text → producing a representation (embedding, classification) | Stage 3 (scoring) and stage 4 (similarity) |
| **Decoder-only** | GPT, Llama, Qwen, Claude | Generating text token-by-token | Stages 2 and 5 (extraction and reasoning, via Ollama) |
| **Encoder-decoder** | T5, BART | Tasks where input and output are both sequences (translation, summarization) | Not used here |

Roughly: encoders *read*, decoders *write*, encoder-decoders
*transform*.

## How it works (just enough)

A transformer is a stack of identical *blocks*. Each block does:

1. **Self-attention** — every token computes a weighted sum of every
   other token. The weights are learned. This is how "I" and "she"
   end up connected across a long sentence.
2. **Feed-forward** — a small per-token MLP that transforms the
   attention output.
3. **Residual + layer norm** — boring but essential plumbing that
   makes the whole thing trainable.

DistilBERT has 6 such blocks. Llama 3.2 3B has 28. GPT-4-class models
have 80+. The block structure is the same; the count, width, and
training data change.

You **don't** need this internal model to use these things. But if
you ever want to read a model's source, this is the mental skeleton.

## Where it lives in Resumora AI

- **DistilBERT** (encoder, 67M params): scoring stage. Downloaded
  from HF Hub by `pipeline/scoring/loader.py`.
- **MiniLM** (encoder, 22M params, distilled BERT-family): similarity
  stage. Loaded via `sentence-transformers`.
- **Llama 3.2 3B** (decoder, 3B params): extraction + reasoning.
  Loaded by Ollama, not by our Python directly.

Our code never instantiates a transformer block. We always go through
high-level wrappers: `AutoModelForSequenceClassification`,
`SentenceTransformer`, or an HTTP call to Ollama.

## Worth knowing

- **"Transformer" the architecture vs. `transformers` the library.**
  Hugging Face's library is named after the architecture and supports
  most models built on it. They're different nouns.
- **Context window.** Every transformer has a maximum input length
  (e.g. 512 for DistilBERT, 128K for Llama 3.2). Exceed it and you
  truncate (or worse, error). Cost scales quadratically with context
  length in vanilla attention.
- **Encoders vs decoders matter for inference shape.** An encoder
  produces *one shot* of output (a vector or classification). A
  decoder produces tokens one at a time autoregressively, which is
  why LLM inference is slow.
- **You can't beat scale by tweaking the architecture.** Most
  "new model" announcements are different *training data + scale +
  fine-tuning*, not new architectures.

## Hands-on

Pop the hood on the scorer model:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
print(model)
# DistilBertForSequenceClassification(
#   (distilbert): DistilBertModel(
#     (embeddings): Embeddings(...)
#     (transformer): Transformer(
#       (layer): ModuleList(
#         (0-5): 6 x TransformerBlock(...)   # ← the six "blocks" we mentioned
#       )
#     )
#   )
#   (pre_classifier): Linear(in_features=768, out_features=768)
#   (classifier): Linear(in_features=768, out_features=3)   # ← classification head
#   (dropout): Dropout(p=0.2)
# )
```

Six transformer blocks → a pre-classifier linear → a 3-way classifier
head → softmax (added by the loss function). That's the whole scorer.

## Go deeper

- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the canonical visual explainer. Read this first if you only read one thing.
- [Harvard NLP — The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the paper, line-by-line, with running PyTorch code.
- [Hugging Face NLP Course — Chapter 1 (Transformer models)](https://huggingface.co/learn/nlp-course/chapter1/1).
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original paper. Surprisingly readable.
- [BERT paper (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) — the encoder-only revolution.
- [DistilBERT paper (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108) — the smaller, faster cousin we actually use.
- **3Blue1Brown — "But what is a GPT?"** on YouTube — beautifully animated decoder transformer walkthrough.

Related concepts: [Classifiers](./classifiers.md), [LLMs and Ollama](./llms-and-ollama.md), [Embeddings](./embeddings.md).
