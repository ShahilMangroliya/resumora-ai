from dataclasses import dataclass, field

import pytest


@dataclass
class FakeOllama:
    payloads: list[dict]
    seen_prompts: list[str] = field(default_factory=list)

    def generate_json(self, prompt: str) -> dict:
        self.seen_prompts.append(prompt)
        return self.payloads.pop(0)


@pytest.fixture
def fake_ollama():
    def _factory(payloads: list[dict]) -> FakeOllama:
        return FakeOllama(payloads=payloads)

    return _factory
