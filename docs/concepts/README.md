# Concepts — A Reference for AI Newcomers

This directory is a **per-concept reference** for the AI/ML ideas
Resumora AI uses. Each file is self-contained: it explains one
concept, shows where it lives in this codebase, and links to
authoritative external resources if you want to go deeper.

If you're brand new, start with the narrative
[`../learning-guide.md`](../learning-guide.md) first — it has the
recommended *reading order* through the code. Once you've done a first
pass, come back here and read these in whatever order you need.

## Index

### Working with text
- [Tokens and tokenizers](./tokens-and-tokenizers.md) — how raw text becomes numbers a model can read.
- [Embeddings](./embeddings.md) — vectors that represent meaning, the foundation of semantic search.
- [Sentence-transformers](./sentence-transformers.md) — turning whole sentences (not just words) into embeddings.

### Neural-network models
- [Transformers](./transformers.md) — the architecture under BERT, GPT, Llama, and everything else here.
- [Classifiers](./classifiers.md) — turning a transformer into a category-predicting function.
- [LLMs and Ollama](./llms-and-ollama.md) — generative models, run locally with no API key.

### Training your own model
- [Fine-tuning](./fine-tuning.md) — taking a pretrained model and teaching it your task.
- [LoRA and PEFT](./lora-and-peft.md) — cheap fine-tuning that fits on a free Colab GPU.
- [Hugging Face Hub](./hugging-face-hub.md) — the registry where models and datasets live.

### Working with LLMs
- [Prompting and structured outputs](./prompting.md) — getting an LLM to behave like a function.

### Beyond this project
- [What to learn next: RAG, agents, LLMOps](./whats-next.md) — natural extensions of what you've built here.

## How to use this

- Each file ends with a **Go deeper** section. Those external links are
  the ones I'd trust if I had to pick five resources per topic — HF
  official docs, original papers, and a couple of canonical blog
  posts.
- Almost every section has a **Where it lives** block that points to
  the exact file in this repo where the concept shows up. Read the
  concept, then read the file. That round trip is the fastest way to
  build real understanding.
- Code snippets in these docs are illustrative, not literal copies of
  the project. When in doubt, the source is authoritative.
