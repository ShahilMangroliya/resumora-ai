# Tokens and Tokenizers

> **TL;DR** — A neural net can't ingest raw strings. A *tokenizer* is a
> learned encoder that splits text into chunks ("tokens") and maps each
> chunk to an integer ID. `"resume fit"` might become `[7984, 4682]`.

## Why this exists

Naively splitting on whitespace breaks down fast:

- *"resume"* and *"resumes"* would be unrelated tokens, exploding the vocabulary.
- *"Müller"*, emojis, and Chinese characters wouldn't fit at all.
- Rare technical words (*"distilbert"*) would never be seen during training.

The fix is **subword tokenization**. Common words stay whole; rare
words get split into known sub-pieces. *"resumora"* might tokenize as
*"res" + "##umora"* — both pieces are known, so the model can handle
words it has never seen as long as their *parts* are familiar.

Three flavors dominate:

| Algorithm | Used by | Notes |
|---|---|---|
| **WordPiece** | BERT, DistilBERT | Subwords prefixed with `##`. |
| **BPE / Byte-Pair Encoding** | GPT, Llama | Operates on bytes — never out-of-vocabulary. |
| **SentencePiece / Unigram** | T5, Llama (variant) | Treats spaces as a normal character; good for non-English. |

You don't need to know the internals — but you should know **the same
text produces different token counts under different tokenizers**.
That affects context-window budgets and inference cost.

## How it works (just enough)

1. The tokenizer ships *with* the model and was trained on the same
   corpus. You should never mix a tokenizer from model A with weights
   from model B.
2. Encoding: `text → list of integer IDs`, plus an *attention mask*
   that tells the model which positions are real vs. padding.
3. Decoding: `list of integer IDs → text`. Useful when sampling from
   an LLM, since the model generates token IDs and you need to display
   the result.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tok("a Python developer with 5 years of experience")
# -> {'input_ids': [101, 1037, 18750, 9722, 2007, 1019, 2086, 1997, 3325, 102],
#     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

The `101` and `102` are *special tokens* (`[CLS]` and `[SEP]` for
BERT-family models). The classifier head reads off the `[CLS]`
position to make its prediction.

## Where it lives in Resumora AI

- `packages/pipeline/src/pipeline/scoring/loader.py` — loads the
  DistilBERT tokenizer at API startup.
- `packages/pipeline/src/pipeline/scoring/scorer.py` — uses the
  tokenizer to encode resume + JD before feeding them to the classifier.
- The sentence-transformer model in stage 4 has its **own**
  tokenizer baked in; you don't see it because the
  `sentence-transformers` library hides it behind `.encode()`.

## Worth knowing

- **Token count ≠ word count.** A 1000-word resume might be 1300–1500
  tokens. Some context limits (e.g. DistilBERT's 512) bite earlier
  than you'd think.
- **Special tokens matter.** `[CLS]`, `[SEP]`, `[PAD]`, `<s>`, `</s>`,
  `<|im_start|>` — different model families use different conventions.
  Always use `AutoTokenizer` to get the right ones.
- **The tokenizer is part of the model contract.** When you publish a
  fine-tuned model to Hugging Face Hub, push the tokenizer files too,
  or downstream users will be stuck.
- **Long inputs get truncated, not errored.** Default `truncation=True`
  silently drops content past the limit — easy bug to miss.

## Hands-on

Open a Python shell after `make install` and play:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# How many tokens is your resume?
text = open("/path/to/your/resume.txt").read()
print(len(tok(text)["input_ids"]))

# What does subword splitting actually look like?
print(tok.tokenize("Resumora is a resume analyzer using DistilBERT"))
# -> ['res', '##um', '##ora', 'is', 'a', 'resume', 'an', '##aly', '##zer', 'using', 'di', '##sti', '##lbert']
```

Notice how *"Resumora"* and *"DistilBERT"* — words the tokenizer never
saw in training — get reconstructed from known sub-pieces.

## Go deeper

- [Hugging Face NLP Course — Chapter 6: The 🤗 Tokenizers library](https://huggingface.co/learn/nlp-course/chapter6/1) — best free deep dive.
- [Hugging Face Tokenizers docs](https://huggingface.co/docs/transformers/tokenizer_summary) — the official summary of WordPiece / BPE / Unigram.
- **Andrej Karpathy — "Let's build the GPT tokenizer"** on YouTube — 2-hour video implementing BPE from scratch. The single best teaching resource on this topic.
- [Original BPE paper (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909) — short, readable.
