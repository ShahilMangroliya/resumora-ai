import torch

from training.dataset.schema import Pair
from training.train import evaluate as eval_mod


def _pair(pid: str, label: str, score: int) -> Pair:
    return Pair(
        pair_id=pid,
        resume_text="resume " + pid,
        jd_text="jd " + pid,
        label=label,
        score=score,
        role="backend_dev",
        seniority="senior",
        source="gold",
        generator_model="manual",
        generated_at="2026-05-15T00:00:00Z",
        prompt_seed=0,
    )


class _StubModel(torch.nn.Module):
    """Returns fixed logits per pair_id so the test controls every prediction.

    Maps the first token id (which the stub tokenizer encodes from the pair_id)
    to a 3-vector of logits.
    """

    def __init__(self, logits_by_first_token: dict[int, list[float]]):
        super().__init__()
        self._logits = logits_by_first_token

    def forward(self, input_ids, attention_mask):
        rows = []
        for row in input_ids:
            first = int(row[0])
            rows.append(self._logits[first])
        return type("Out", (), {"logits": torch.tensor(rows)})


class _StubTokenizer:
    """Encodes each pair as `[token_for_pid]` so the stub model can identify it."""

    def __init__(self, pid_to_token: dict[str, int]):
        self._map = pid_to_token

    def __call__(self, text_a, text_b, truncation, max_length, return_tensors=None, padding=None):
        # Use the resume text "resume <pid>" to recover the pid.
        pid = text_a.split()[-1]
        token = self._map[pid]
        ids = torch.tensor([[token]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_evaluate_against_gold_perfect_predictions():
    pairs = [_pair("p1", "weak", 20), _pair("p2", "partial", 55), _pair("p3", "strong", 85)]
    tokenizer = _StubTokenizer({"p1": 10, "p2": 11, "p3": 12})
    model = _StubModel({
        10: [10.0, 0.0, 0.0],   # weak
        11: [0.0, 10.0, 0.0],   # partial
        12: [0.0, 0.0, 10.0],   # strong
    })

    report = eval_mod.evaluate_against_gold(
        model=model, tokenizer=tokenizer, gold_pairs=pairs, max_length=16, device="cpu",
    )
    assert report.n == 3
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.mae < 5.0
    assert report.confusion_matrix == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_evaluate_against_gold_one_misclassification():
    pairs = [_pair("p1", "weak", 20), _pair("p2", "weak", 20)]
    tokenizer = _StubTokenizer({"p1": 10, "p2": 11})
    model = _StubModel({
        10: [10.0, 0.0, 0.0],     # predicts weak (correct)
        11: [0.0, 10.0, 0.0],     # predicts partial (wrong — labelled weak)
    })

    report = eval_mod.evaluate_against_gold(
        model=model, tokenizer=tokenizer, gold_pairs=pairs, max_length=16, device="cpu",
    )
    assert report.n == 2
    assert report.accuracy == 0.5
    # Confusion: 1 weak->weak, 1 weak->partial. Row index = true label, col = pred.
    assert report.confusion_matrix[0][0] == 1
    assert report.confusion_matrix[0][1] == 1


def test_render_report_includes_key_lines():
    report = eval_mod.EvalReport(
        accuracy=0.5,
        macro_f1=0.42,
        per_class_f1={"weak": 0.5, "partial": 0.0, "strong": 0.75},
        mae=12.3,
        confusion_matrix=[[1, 1, 0], [0, 0, 0], [0, 0, 1]],
        n=3,
    )
    rendered = eval_mod.render_report(report)
    assert "accuracy" in rendered.lower()
    assert "macro_f1" in rendered.lower()
    assert "confusion" in rendered.lower()
