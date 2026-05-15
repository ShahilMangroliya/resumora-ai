from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def push_dataset(
    *,
    repo_id: str,
    folder: Path,
    hf_token: str,
    commit_message: str = "update dataset",
) -> None:
    """Create the dataset repo if missing and upload everything in `folder`.

    Never run from CI. Run locally with the user's HF_TOKEN env var set.
    """
    if not hf_token:
        raise ValueError("HF token is required (pass --token or set HF_TOKEN env var)")
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(folder)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        commit_message=commit_message,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="training.publish.to_hf")
    parser.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/resumora-ai")
    parser.add_argument("--folder", type=Path, required=True, help="local folder to upload")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument("--message", default="update dataset")
    args = parser.parse_args(argv)

    push_dataset(
        repo_id=args.repo,
        folder=args.folder,
        hf_token=args.token,
        commit_message=args.message,
    )
    print(f"uploaded {args.folder} → {args.repo}")


if __name__ == "__main__":
    main()
