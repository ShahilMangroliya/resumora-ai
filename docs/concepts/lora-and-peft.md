# LoRA and PEFT

> **TL;DR** — LoRA ("Low-Rank Adaptation") is a fine-tuning trick:
> freeze the big pretrained model and only train a small number of new
> parameters that ride alongside it. Result: ~99% of the quality of
> full fine-tuning, ~1% of the GPU memory. PEFT is the Hugging Face
> library that implements LoRA (and a few siblings).

## Why this exists

Full fine-tuning updates every parameter in the model. For
DistilBERT (67M params) that's tractable. For Llama 3.2 3B
(~3 billion params) you need ~24GB of GPU memory just to hold
optimizer state — gone is your free Colab T4.

LoRA's insight: when you fine-tune, the *change* to the weight matrix
is *low rank*. You don't need to store the whole change; you can
approximate it with the product of two skinny matrices.

```
Original frozen weight matrix:   W   (d × d, big)
LoRA addition (trainable):       B · A    where A is (d × r), B is (r × d), and r is small (4, 8, 16, 32)

Effective weight at inference:   W + B·A
```

If `d = 768` and `r = 8`, the LoRA matrices have `~12,000` parameters
instead of `~590,000` for the same weight matrix — a 50× reduction.
Multiply across all layers and you can fine-tune a 3B model with
only a few million trainable parameters.

## How it shows up in practice

You don't write the math. The `peft` library wraps any HF model:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

config = LoraConfig(
    r=8,                            # rank — bigger = more capacity, more memory
    lora_alpha=16,                  # scaling factor (often 2 * r)
    target_modules=["q_lin", "v_lin"],  # which weights to wrap (attention projections, typically)
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 296K  ||  all params: 67M  ||  trainable%: 0.44
```

Then you call `Trainer` like normal. Only the LoRA parameters update.

## At inference time: merge or stay split?

Two options:

1. **Merge** — at load time, compute `W' = W + B·A` once and discard
   the LoRA matrices. Same speed as the original model. **What
   Resumora does.**
2. **Keep separate** — useful when you want to swap *multiple* LoRA
   adapters into the same base model (e.g. one adapter per customer).
   Slightly slower per forward pass.

## Where it lives in Resumora AI

- `packages/training/src/training/train/` — applies `LoraConfig` during fine-tuning.
- `packages/pipeline/src/pipeline/scoring/loader.py` — uses `PeftModel.from_pretrained` to load the base DistilBERT + LoRA adapter, then merges the adapter for inference.
- The published model on Hugging Face Hub is the small LoRA delta —
  not a full 67M-param checkpoint. That makes downloads fast and Hub
  storage cheap.

## Worth knowing

- **`r` is your main knob.** Start with `r=8`. If quality is low, try
  16 or 32. Going higher rarely helps and costs memory.
- **`target_modules` matters.** Wrapping just the attention `q_proj`
  and `v_proj` is the LoRA paper's recommendation and works for most
  tasks. Wrap more if your task is harder.
- **Quality ceiling is ~ full fine-tune.** Don't expect LoRA to *beat*
  full fine-tuning — it just gets close at a fraction of the cost.
- **LoRA adapters are tiny.** A typical Resumora-scale adapter is
  ~1–3 MB. Easy to version, easy to ship.
- **Alternatives exist.** QLoRA (quantized LoRA) for even tighter
  budgets, IA3, prefix tuning. PEFT supports them; LoRA is the
  default for good reason.

## Hands-on

After running the training notebook, inspect the saved adapter:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
model = PeftModel.from_pretrained(base, "<your-user>/resumora-ai-distilbert-lora")
model.print_trainable_parameters()

# Merge if you want a single self-contained model
merged = model.merge_and_unload()
```

The merged model is what the API actually uses.

## Go deeper

- [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — short, accessible, worth reading the first half.
- [PEFT documentation](https://huggingface.co/docs/peft/index) — full library reference.
- [HF blog — Make LLMs lighter with PEFT](https://huggingface.co/blog/peft).
- [HF blog — 4-bit fine-tuning with QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — the next step when memory is tighter still.
- [Sebastian Raschka — Practical tips on LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — empirical guidance on rank, alpha, target modules.

Related concepts: [Fine-tuning](./fine-tuning.md), [Classifiers](./classifiers.md), [Hugging Face Hub](./hugging-face-hub.md).
