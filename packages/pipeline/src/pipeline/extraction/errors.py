class ExtractionError(Exception):
    """Raised when the extraction stage cannot produce a profile.

    Covers Ollama transport failures, timeout, and unparseable model
    output. The API layer maps this to an HTTP 503 response.
    """
