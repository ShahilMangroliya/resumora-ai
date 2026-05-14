class IngestionError(Exception):
    """Raised when a resume or job description cannot be ingested.

    The API layer maps this to an HTTP 400 response.
    """
