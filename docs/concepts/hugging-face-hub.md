# Hugging Face Hub

> **TL;DR** — Hugging Face Hub is the central registry for the ML
> ecosystem. Models, datasets, and demos all live there. From this
> project's perspective it's "Docker Hub for transformers": you push
> trained models to it, the API pulls them at boot.

## What's actually on it

| Asset | What it is | Example |
|---|---|---|
| **Models** | Trained weights + tokenizer + config. Any size, any framework. | `distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`, your own `<user>/resumora-ai-distilbert-lora` |
| **Datasets** | Versioned datasets, often with a streaming-friendly format. | `imdb`, `squad`, your own synthetic resume↔JD set |
| **Spaces** | Free hosted demos (Gradio/Streamlit apps) — great for portfolio. | A working Resumora demo could live here. |
| **Papers** | Linked discussions tying papers to their model implementations. | |
| **Inference Endpoints** | Pay-per-use hosted inference (not used in this project). | |

The "Hub" is a single git-LFS-backed platform. Every model is just a
git repo with weights as large files.

## Why it matters

Before HF Hub, every research lab had its own download script with
its own quirky tar format. Now:

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("any/repo-id")
```

`from_pretrained` does the git clone, caches under
`~/.cache/huggingface/`, and gives you a working model. That uniform
contract is the entire reason the modern ML ecosystem moves so fast.

## Repo anatomy

A typical model repo looks like:

```
your-user/resumora-ai-distilbert-lora/
├── README.md             ← the "model card" — required by the Hub
├── adapter_config.json   ← LoRA config
├── adapter_model.bin     ← LoRA weights (tiny — ~1 MB)
├── tokenizer.json        ← tokenizer files
├── tokenizer_config.json
├── special_tokens_map.json
└── config.json           ← base model config
```

For full fine-tunes there'd also be a `pytorch_model.bin` or
`model.safetensors` (the big weights file, ~250 MB for DistilBERT).

The **model card** (`README.md`) is non-negotiable. It documents:

- What the model does
- What it was trained on (datasets, sources)
- How to use it (code snippet)
- License
- Evaluation metrics on a known benchmark
- Known limitations and biases

Resumora's training code generates a model card from a template — see
`packages/training/src/training/publish/`.

## Auth

The Hub is free to *read* (for public models). Pushing requires a
token:

```bash
huggingface-cli login          # one-time, stores token in ~/.cache/huggingface/token
# OR
export HF_TOKEN=hf_xxx          # for CI / Colab
```

In the Colab notebook this is the `HF_TOKEN` cell.

## Where it lives in Resumora AI

- **Reading from the Hub**:
  - `packages/pipeline/src/pipeline/scoring/loader.py` — pulls the base DistilBERT + your fine-tuned LoRA adapter at API startup.
  - `packages/pipeline/src/pipeline/similarity/_embeddings.py` — pulls the MiniLM sentence-transformer.
- **Writing to the Hub**:
  - `packages/training/src/training/publish/` — the post-training step that pushes weights + tokenizer + model card.
  - `notebooks/01_train_on_colab.ipynb` — invokes the publisher at the end of a successful run.
- **Configuration**: `RESUMORA_AI_SCORER_REPO` env var picks which Hub repo the API loads at boot.

## Worth knowing

- **The cache is persistent.** First `from_pretrained` downloads;
  subsequent calls are instant. On Colab the cache is *not*
  persistent — every notebook session re-downloads. (Use a Colab
  drive mount if this gets annoying.)
- **Public vs private repos.** Default is public. Set `private=True`
  in `push_to_hub` if you need to gate access. Private repos require
  authenticated reads.
- **Repo IDs are `user/name` or `org/name`.** No slashes inside the
  name itself.
- **Version pinning is via revisions.** `from_pretrained("user/name", revision="v2")` pulls a specific git tag or commit SHA. Worth doing in production so a Hub update doesn't change your behavior.
- **Don't push your `.env`, your training data, your secrets.** Hub
  repos are public by default and indexed by Google.
- **License the model.** Apache 2.0 inherits from base DistilBERT and
  is a safe default for derived works.

## Hands-on

```bash
# Browse
open https://huggingface.co/distilbert-base-uncased
open https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Pull from CLI
pip install huggingface_hub
huggingface-cli download distilbert-base-uncased

# Inspect cache
ls ~/.cache/huggingface/hub/
```

After training, your own repo will appear at
`https://huggingface.co/<your-user>/resumora-ai-distilbert-lora` —
a useful portfolio link to share alongside the GitHub repo.

## Go deeper

- [Hugging Face Hub documentation](https://huggingface.co/docs/hub/index) — full reference.
- [`huggingface_hub` Python library docs](https://huggingface.co/docs/huggingface_hub/index) — programmatic Hub access.
- [Model card guide](https://huggingface.co/docs/hub/model-cards) — what a good model card includes.
- [HF NLP Course — Chapter 4 (sharing models)](https://huggingface.co/learn/nlp-course/chapter4/1).
- [Hugging Face Spaces docs](https://huggingface.co/docs/hub/spaces) — free Gradio/Streamlit hosting; great place to put a live demo of your trained scorer.

Related concepts: [Fine-tuning](./fine-tuning.md), [LoRA and PEFT](./lora-and-peft.md), [Transformers](./transformers.md).
