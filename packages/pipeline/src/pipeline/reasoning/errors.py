class ReasoningError(Exception):
    """Raised when the reasoning stage cannot produce a result.

    Covers Ollama transport failures, timeouts, unparseable model output,
    and JSON payloads that do not satisfy the `ReasoningResult` shape.
    The API layer maps this to a partial-result response (score-only).
    """
