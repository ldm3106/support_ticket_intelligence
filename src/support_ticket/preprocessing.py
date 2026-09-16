import re
import unicodedata


def prepare_text(text: str) -> str:
    """Normalize whitespace, retaining negation and punctuation for the model."""
    if not isinstance(text, str):
        raise TypeError("Ticket text must be a string.")
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    if not text:
        raise ValueError("Ticket text must not be empty.")
    return text


def text_key(text: str) -> str:
    return prepare_text(text).casefold()
