# Fine-tuning

> **TL;DR** — Fine-tuning means taking a model someone else already
> trained on a huge corpus, then continuing to train it on *your*
> smaller dataset so it specializes to *your* task. In this project,
> we fine-tune DistilBERT on resume↔JD pairs to turn it into a
> three-class fit classifier.

## Why this exists

Training a transformer from scratch costs millions of dollars and
needs trillions of tokens. You don't have that budget. You don't need
it either — almost all the *language understanding* a useful model
needs has already been baked into pretrained weights. You just need
to teach it the last 1% that's specific to your problem.

Fine-tuning is the entire reason the Hugging Face ecosystem exists:
download pretrained weights, point a Trainer at your labeled CSV,
push the result back to the Hub.

## The three options, in order of cost

| Strategy | What you do | When to use it |
|---|---|---|
| **Zero-shot / prompting** | Don't train. Just ask an LLM the question via prompt. | Quick prototypes, low-volume tasks. No labels needed. |
| **Few-shot / in-context learning** | Put 3–10 examples in the prompt. | Stronger than zero-shot, still no training. Limited by context window. |
| **Fine-tuning** | Update model weights on a labeled dataset. | High volume, latency matters, accuracy matters, or you have proprietary signal. |

Within "fine-tuning" there are sub-options:

- **Full fine-tuning** — update every parameter. Best quality, most expensive, easiest to overfit. Doesn't fit on a free GPU for most useful models.
- **PEFT (parameter-efficient fine-tuning)** — freeze the big model, train tiny add-on weights. Almost as good, fits anywhere. [LoRA](./lora-and-peft.md) is the most popular kind.
- **Feature extraction** — freeze the model entirely, train only a classification head. Worst quality, cheapest. Mostly historical now.

Resumora AI uses **LoRA fine-tuning** because we want quality but the
budget is a free Colab T4 GPU.

## The fine-tuning recipe (pseudocode)

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

model_id = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)

# Add LoRA adapters — see lora-and-peft.md
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"]))

ds = load_dataset("json", data_files={"train": "train.jsonl", "eval": "eval.jsonl"})
ds = ds.map(lambda b: tok(b["resume"], b["jd"], truncation=True, padding="max_length"))

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./out", num_train_epochs=3, per_device_train_batch_size=16, eval_strategy="epoch"),
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    tokenizer=tok,
)
trainer.train()
trainer.push_to_hub("resumora-ai-distilbert-lora")  # → Hugging Face Hub
```

That's the entire structure. The real `packages/training` code is a
bit longer because it adds config, logging, eval, and a CLI — but the
core loop is exactly this.

## Where it lives in Resumora AI

- `notebooks/01_train_on_colab.ipynb` — the runnable training notebook (designed for Colab T4 GPU).
- `packages/training/src/training/train/` — the Python that the notebook imports. Configurable, testable, callable from a script.
- `packages/training/src/training/dataset/` — synthetic data generation + JSONL loaders.
- `packages/training/src/training/publish/` — pushes the trained adapter to Hugging Face Hub.
- `data/gold/` — a small, hand-curated evaluation set used to measure quality. **Never train on it.**

## Worth knowing

- **More data > more epochs.** Three epochs over 10,000 examples beats
  thirty epochs over 1,000 — overfitting kicks in fast.
- **Validation set discipline.** Keep an *untouched* eval set. If you
  ever look at it during training-loop tuning, you've contaminated it.
- **Synthetic data is a real tool, not a hack.** Resumora generates
  synthetic resume↔JD pairs with an LLM because labeling 10K real ones
  is impractical. You then *validate* synthetic-trained models on a
  small *real* gold set.
- **Catastrophic forgetting is real.** Fine-tuning *can* erase
  general-purpose abilities, especially with full fine-tuning. LoRA
  largely sidesteps this.
- **Save the eval metrics with the weights.** Pushing a model without
  the F1/accuracy it achieved is a mistake — it's metadata you'll want
  six months later.

## Hands-on

You don't have to spin up Colab to play with the *concept*. On any
laptop, fine-tune a tiny model on a tiny dataset:

```python
# Around 30 lines based on the recipe above. Use a tiny model like
# "distilbert-base-uncased" and ~1000 examples of any binary
# classification dataset (e.g. the imdb dataset from HF Hub).
# It'll train in a few minutes on CPU.
```

Once that clicks, open `notebooks/01_train_on_colab.ipynb` and you'll
recognize most of it.

## Go deeper

- [Hugging Face NLP Course — Chapter 3](https://huggingface.co/learn/nlp-course/chapter3/1) — best end-to-end tutorial.
- [HF Transformers — Training](https://huggingface.co/docs/transformers/training) — the official `Trainer` reference.
- [HF Datasets documentation](https://huggingface.co/docs/datasets/index) — how to load and preprocess data.
- [Sebastian Raschka — "Practical Tips for Fine-tuning"](https://magazine.sebastianraschka.com/) — Sebastian's substack is the single best living resource on practical fine-tuning.
- [MLflow Tracking docs](https://mlflow.org/docs/latest/tracking.html) — the experiment tracker this project uses.

Related concepts: [LoRA and PEFT](./lora-and-peft.md), [Classifiers](./classifiers.md), [Hugging Face Hub](./hugging-face-hub.md).
