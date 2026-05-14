# Ollama Setup

Ollama is the local, open-source LLM runtime used for the extraction and
reasoning stages of the pipeline. It runs on Apple Silicon via Metal — no
GPU cloud, no API key, no cost.

## Install (macOS)

```bash
brew install ollama
```

## Start the server

```bash
ollama serve
```

Leave this running in its own terminal (or it runs as a background service
after `brew services start ollama`).

If `ollama serve` fails with "address already in use", the server is already running as a background service — skip to the next step.

## Pull the dev model

```bash
ollama pull llama3.2:3b
```

`llama3.2:3b` is small enough to run comfortably on a laptop and is the
default model for local development.

## Verify

```bash
ollama run llama3.2:3b "Reply with exactly: ok"
```

Expected: the model replies with `ok` (or a short response containing it).
