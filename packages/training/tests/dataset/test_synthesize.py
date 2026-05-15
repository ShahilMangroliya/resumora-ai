import pytest

from training.dataset.roles import get_role
from training.dataset.schema import Pair
from training.dataset.synthesize import generate_pair


def test_generate_pair_returns_validated_pair(fake_ollama):
    client = fake_ollama(
        [
            {
                "resume_text": "Resume body...",
                "jd_text": "JD body...",
            }
        ]
    )
    role = get_role("backend-dev")
    pair = generate_pair(role, "mid", "strong", client=client, seed=1)
    assert isinstance(pair, Pair)
    assert pair.label == "strong"
    assert pair.score == 85
    assert pair.role == "backend-dev"
    assert pair.seniority == "mid"
    assert pair.source == "synthetic"
    assert pair.resume_text == "Resume body..."
    assert pair.jd_text == "JD body..."
    assert pair.prompt_seed == 1
    assert pair.generated_at.endswith("Z")


def test_generate_pair_passes_role_seniority_and_label_to_prompt(fake_ollama):
    client = fake_ollama(
        [{"resume_text": "r", "jd_text": "j"}]
    )
    role = get_role("data-analyst")
    generate_pair(role, "senior", "weak", client=client, seed=7)
    prompt = client.seen_prompts[0]
    assert "Data Analyst" in prompt
    assert "senior" in prompt
    assert "weak" in prompt


def test_pair_id_is_deterministic_for_same_inputs(fake_ollama):
    role = get_role("backend-dev")
    client_a = fake_ollama([{"resume_text": "r", "jd_text": "j"}])
    client_b = fake_ollama([{"resume_text": "r", "jd_text": "j"}])
    a = generate_pair(role, "mid", "strong", client=client_a, seed=42)
    b = generate_pair(role, "mid", "strong", client=client_b, seed=42)
    assert a.pair_id == b.pair_id


def test_pair_id_changes_when_seed_changes(fake_ollama):
    role = get_role("backend-dev")
    client_a = fake_ollama([{"resume_text": "r", "jd_text": "j"}])
    client_b = fake_ollama([{"resume_text": "r", "jd_text": "j"}])
    a = generate_pair(role, "mid", "strong", client=client_a, seed=1)
    b = generate_pair(role, "mid", "strong", client=client_b, seed=2)
    assert a.pair_id != b.pair_id


def test_generate_pair_raises_on_missing_field(fake_ollama):
    client = fake_ollama([{"resume_text": "r"}])
    role = get_role("backend-dev")
    with pytest.raises(Exception):
        generate_pair(role, "mid", "strong", client=client, seed=1)
