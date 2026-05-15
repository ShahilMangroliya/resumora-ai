# Classifiers

> **TL;DR** — A classifier is a function `input → category`. In this
> project, the input is a `(resume, job description)` pair and the
> output is one of three categories: `weak_fit`, `partial_fit`,
> `strong_fit`. Those category probabilities get turned into a single
> 0–100 score.

## Why this exists

LLMs are flexible but expensive and unreliable for things you do
millions of times. For a constrained question like *"how well does
this resume fit this job?"* a small, dedicated classifier is:

- **Cheaper** — millions of inferences for cents on CPU.
- **Faster** — milliseconds per call.
- **More predictable** — outputs a calibrated distribution, not free-form text.
- **Trainable on your data** — improves with your labels.

The trade-off: it can only answer the exact question you trained it
on. You can't ask the scorer to *explain* the fit (that's what stage
5's LLM is for).

## How it works (just enough)

A modern text classifier is almost always:

```
text  →  tokenizer  →  transformer encoder  →  [CLS] vector  →  linear layer  →  softmax  →  probabilities
```

- The **transformer encoder** (DistilBERT here) turns tokens into vectors.
- The model is engineered so the first token (`[CLS]`) ends up as a *summary* vector representing the whole input.
- A **classification head** (a small `Linear` layer) projects that 768-dim vector down to `num_labels` raw scores ("logits").
- **Softmax** turns those logits into a proper probability distribution.

Training adjusts the encoder *and* the head together with
cross-entropy loss, so the `[CLS]` summary specifically encodes
whatever is useful for *your* labels.

## From probabilities to a score

The model outputs three probabilities `(p_weak, p_partial, p_strong)`.
To get a single number 0–100, this project takes a weighted average:

```
score = (0 * p_weak + 50 * p_partial + 100 * p_strong)
```

That's a deliberate, debuggable transformation. The `label` you see
in the response (`"strong_fit"` etc.) is `argmax` of the same
distribution.

## Where it lives in Resumora AI

- `packages/pipeline/src/pipeline/scoring/loader.py` — loads the
  fine-tuned DistilBERT + LoRA adapter from Hugging Face Hub.
- `packages/pipeline/src/pipeline/scoring/scorer.py` — runs the
  forward pass: tokenize → encode → classify → return logits.
- `packages/pipeline/src/pipeline/scoring/math.py` — softmax,
  argmax-to-label, weighted-mean-to-score. Pure functions — read these
  first, they make the rest obvious.
- `packages/pipeline/src/pipeline/scoring/models.py` — the Pydantic
  output shape (`Score(value, label, probabilities)`).

## Worth knowing

- **The classifier doesn't "understand" — it pattern-matches.** Tiny
  surface changes to the input (e.g. extra whitespace, different
  date formats) can shift the score. Real evaluation matters more than
  vibes.
- **Class imbalance silently hurts.** If your training data is 70%
  `partial_fit`, the model will default toward it. Look at your label
  distribution before trusting accuracy.
- **Accuracy is rarely the right metric.** Use a confusion matrix
  and per-class F1. A 90%-accurate scorer that misses every
  `strong_fit` is useless.
- **The probabilities are not always well-calibrated.** A 90%
  probability doesn't always mean *"correct 90% of the time"* — you
  can calibrate post-hoc with temperature scaling if you care.

## Hands-on

After training (or pointing `RESUMORA_AI_SCORER_REPO` at a published
model):

```python
from pipeline.scoring.loader import load_scorer
from pipeline.scoring.scorer import score_pair

scorer = load_scorer()
result = score_pair(scorer, resume_text="...", jd_text="...")
print(result)
# Score(value=78, label="strong_fit",
#       probabilities={"weak_fit": 0.04, "partial_fit": 0.18, "strong_fit": 0.78})
```

Then perturb the inputs a little — change a date, remove a section —
and watch the probabilities shift. That builds the intuition that the
score is a *learned similarity*, not a rule-based comparison.

## Go deeper

- [Hugging Face NLP Course — Chapter 3 (fine-tuning a classifier)](https://huggingface.co/learn/nlp-course/chapter3/1) — most relevant tutorial.
- [HF Transformers — Text classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification).
- [`AutoModelForSequenceClassification` reference](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification).
- [Sebastian Raschka — Fine-tuning a Text Classifier](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — covers when to fine-tune a classifier vs. use an LLM.

Related concepts: [Transformers](./transformers.md), [Fine-tuning](./fine-tuning.md), [LoRA and PEFT](./lora-and-peft.md).
