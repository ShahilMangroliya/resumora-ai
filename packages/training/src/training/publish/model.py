from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from textwrap import dedent

from huggingface_hub import HfApi, ModelCard, ModelCardData


def build_model_card(
    *,
    base_model: str,
    dataset_repo: str,
    train_config: dict,
    eval_metrics: dict,
) -> ModelCard:
    """Assemble the public-facing model card.

    All the disclosures from supplement §7.1 live in this string. Editing the
    card means editing them in one place.
    """
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",
        library_name="peft",
        base_model=base_model,
        tags=["resume", "job-matching", "lora", "classification"],
        pipeline_tag="text-classification",
    )

    per_class = eval_metrics.get("per_class_f1", {})
    eval_n = eval_metrics.get("n", "?")

    body = dedent(
        f"""\
        # Resumora AI — DistilBERT + LoRA fit classifier

        Fine-tuned classifier that scores a (resume, job description) pair as one of
        `weak` / `partial` / `strong` fit, and produces a continuous score in `[20, 85]`
        via the expected value `softmax(logits) · [20, 55, 85]`.

        Built as the model artifact for the [Resumora AI](https://github.com/) portfolio
        project. **Not for hiring decisions.**

        ## Score range

        The model output is bounded to `[20, 85]`, not `[0, 100]`. This is by design —
        the score is the softmax-weighted average of three bucket midpoints
        (`weak=20`, `partial=55`, `strong=85`). Numbers outside that range are not
        produced. The 0-100 product surface in the README is honored by honest
        disclosure, not by rescaling.

        ## Intended use

        - **Portfolio demonstration** of LoRA fine-tuning on top of DistilBERT.
        - **NOT for hiring decisions**, screening, or any consequential evaluation
          of a real person's application.

        ## Training data

        - **Synthetic** pairs generated with Ollama + `llama3.2:3b` covering ~15 roles
          x seniorities x 3 fit-levels. Each label was requested in the generator
          prompt — the label comes for free.
        - Dataset: [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo}).
        - Synthetic-data risk: the classifier may have learned the generator's
          stylistic tics. The gold-set evaluation below is the only trusted signal.

        ## Evaluation

        Evaluated against a hand-curated gold set (n = {eval_n}, never trained on):

        - accuracy: **{eval_metrics.get("accuracy", "?"):.3f}**
        - macro F1: **{eval_metrics.get("macro_f1", "?"):.3f}**
        - per-class F1: weak {per_class.get("weak", "?"):.3f}, partial {per_class.get("partial", "?"):.3f}, strong {per_class.get("strong", "?"):.3f}
        - MAE (expected-value score vs bucket midpoint): **{eval_metrics.get("mae", "?"):.2f}**

        ## Limitations

        - English-only.
        - Gold set is small (low statistical power on small differences).
        - No demographic-bias evaluation has been performed.
        - The model has no notion of seniority sub-genres beyond what fit into the
          synthetic prompts.

        ## Reproducibility

        - Base model: `{base_model}`.
        - Training config (subset):

        ```json
        {json.dumps(train_config, indent=2)}
        ```

        License: apache-2.0 (matches the base DistilBERT license).
        """
    )

    card = ModelCard.from_template(card_data, template_str=body)
    card.content = "---\n" + card_data.to_yaml() + "\n---\n\n" + body
    return card


def push_model(
    *,
    repo_id: str,
    model_dir: Path,
    model_card: ModelCard,
    hf_token: str,
    commit_message: str = "update model",
) -> None:
    """Create the model repo if missing and upload the adapter + card.

    Never run from CI. Run locally with the user's HF_TOKEN env var set.
    """
    if not hf_token:
        raise ValueError("HF token is required (pass --token or set HF_TOKEN env var)")
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=commit_message,
    )
    model_card.push_to_hub(repo_id, token=hf_token)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="training.publish.model")
    parser.add_argument("--repo", required=True, help="HF model repo id, e.g. user/resumora-ai-distilbert-lora")
    parser.add_argument("--model-dir", type=Path, required=True, help="local model dir produced by training")
    parser.add_argument("--metrics-json", type=Path, required=True, help="final_metrics.json or eval report json")
    parser.add_argument("--dataset-repo", required=True, help="HF dataset repo id used during training")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--train-config-json", default="{}", help="JSON string of the training config to embed")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument("--message", default="update model")
    args = parser.parse_args(argv)

    metrics_loaded = json.loads(args.metrics_json.read_text())
    train_config = json.loads(args.train_config_json)

    card = build_model_card(
        base_model=args.base_model,
        dataset_repo=args.dataset_repo,
        train_config=train_config,
        eval_metrics=metrics_loaded,
    )

    push_model(
        repo_id=args.repo,
        model_dir=args.model_dir,
        model_card=card,
        hf_token=args.token,
        commit_message=args.message,
    )
    print(f"uploaded {args.model_dir} -> {args.repo}")


if __name__ == "__main__":
    main()
