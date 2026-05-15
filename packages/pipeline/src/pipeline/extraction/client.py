import json
from typing import Any

import httpx

from pipeline.extraction.errors import ExtractionError


class OllamaClient:
    """Thin synchronous client for the local Ollama HTTP API.

    Exposes a single method, ``generate_json``, that returns a parsed
    JSON object. Network errors, 5xx responses, timeouts, and
    unparseable bodies (after one retry) all raise ExtractionError.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            transport=transport,
        )

    def generate_json(self, prompt: str) -> dict[str, Any]:
        last_body = ""
        for attempt in range(2):
            try:
                response = self._client.post(
                    "/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                    },
                )
            except httpx.HTTPError as exc:
                raise ExtractionError(f"Ollama unreachable: {exc}") from exc

            if response.status_code >= 500:
                raise ExtractionError(
                    f"Ollama returned {response.status_code}: {response.text[:200]}"
                )
            response.raise_for_status()

            last_body = response.json().get("response", "")
            try:
                return json.loads(last_body)
            except json.JSONDecodeError:
                if attempt == 1:
                    break
                continue

        raise ExtractionError(
            f"Could not parse Ollama JSON response after retry: {last_body[:200]!r}"
        )
