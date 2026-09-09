"""Bayan text preprocessing utilities."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Normalize Arabic/English feedback while preserving useful signals."""
    if not isinstance(text, str):
        return text

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Remove Arabic tatweel
    text = text.replace("ـ", "")

    # Replace HTML line breaks with spaces
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Reduce any character repeated 3+ times to 2 occurrences
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def mask_pii(text: str) -> str:
    """Mask supported Saudi phone numbers and national-ID-shaped values."""
    if not isinstance(text, str):
        return text

    # Saudi mobile numbers:
    # 05xxxxxxxx
    # +9665xxxxxxxx
    # 9665xxxxxxxx
    phone_pattern = r"(?<!\d)(?:\+?966|0)5\d{8}(?!\d)"
    text = re.sub(phone_pattern, "<PHONE>", text)

    # Saudi national-ID-shaped values starting with 1 or 2
    national_id_pattern = r"(?<!\d)[12]\d{9}(?!\d)"
    text = re.sub(national_id_pattern, "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared Bayan preprocessing pipeline."""
    text = normalize(text)
    text = mask_pii(text)
    return text