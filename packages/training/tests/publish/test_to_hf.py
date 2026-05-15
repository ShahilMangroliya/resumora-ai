from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training.publish import to_hf


def test_push_dataset_calls_create_repo_and_upload_folder(tmp_path: Path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "pairs.jsonl").write_text("{}\n")

    api = MagicMock()
    monkeypatch.setattr(to_hf, "HfApi", lambda token=None: api)

    to_hf.push_dataset(
        repo_id="user/dataset",
        folder=folder,
        hf_token="hf_test_token",
        commit_message="release",
    )

    api.create_repo.assert_called_once()
    create_kwargs = api.create_repo.call_args.kwargs
    assert create_kwargs["repo_id"] == "user/dataset"
    assert create_kwargs["repo_type"] == "dataset"
    assert create_kwargs["exist_ok"] is True

    api.upload_folder.assert_called_once()
    upload_kwargs = api.upload_folder.call_args.kwargs
    assert upload_kwargs["repo_id"] == "user/dataset"
    assert upload_kwargs["repo_type"] == "dataset"
    assert upload_kwargs["folder_path"] == str(folder)
    assert upload_kwargs["commit_message"] == "release"


def test_push_dataset_refuses_when_folder_is_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(to_hf, "HfApi", lambda token=None: MagicMock())
    with pytest.raises(FileNotFoundError):
        to_hf.push_dataset(
            repo_id="user/dataset",
            folder=tmp_path / "nope",
            hf_token="hf_test_token",
        )


def test_push_dataset_refuses_when_token_is_blank(tmp_path: Path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "x.jsonl").write_text("{}\n")
    monkeypatch.setattr(to_hf, "HfApi", lambda token=None: MagicMock())
    with pytest.raises(ValueError, match="HF token"):
        to_hf.push_dataset(repo_id="user/dataset", folder=folder, hf_token="")
