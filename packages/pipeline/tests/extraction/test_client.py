import httpx
import pytest

from pipeline.extraction.client import OllamaClient
from pipeline.extraction.errors import ExtractionError


def _transport(handler):
    return httpx.MockTransport(handler)


def test_returns_parsed_json_object_on_success():
    def handler(request):
        return httpx.Response(200, json={"response": '{"skills": ["python"]}'})

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        transport=_transport(handler),
    )
    assert client.generate_json("prompt") == {"skills": ["python"]}


def test_sends_format_json_and_model():
    seen = {}

    def handler(request):
        seen["json"] = request.read()
        return httpx.Response(200, json={"response": "{}"})

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        transport=_transport(handler),
    )
    client.generate_json("prompt-text")

    body = seen["json"].decode("utf-8")
    assert '"model":"llama3.2:3b"' in body
    assert '"prompt":"prompt-text"' in body
    assert '"format":"json"' in body
    assert '"stream":false' in body


def test_retries_once_on_unparseable_response_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        body = "garbage" if calls["n"] == 1 else '{"ok": true}'
        return httpx.Response(200, json={"response": body})

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        transport=_transport(handler),
    )
    assert client.generate_json("p") == {"ok": True}
    assert calls["n"] == 2


def test_raises_extraction_error_after_two_unparseable_responses():
    def handler(request):
        return httpx.Response(200, json={"response": "still garbage"})

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        transport=_transport(handler),
    )
    with pytest.raises(ExtractionError, match="parse"):
        client.generate_json("p")


def test_raises_extraction_error_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        transport=_transport(handler),
    )
    with pytest.raises(ExtractionError, match="unreachable|connect"):
        client.generate_json("p")


def test_raises_extraction_error_on_http_5xx():
    def handler(request):
        return httpx.Response(500, text="ouch")

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        transport=_transport(handler),
    )
    with pytest.raises(ExtractionError, match="5"):
        client.generate_json("p")
