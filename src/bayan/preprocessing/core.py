"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Remove Arabic tatweel
    text = text.replace("ـ", "")

    # Reduce repeated characters to a maximum of two
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Clean repeated whitespace, newlines, and tabs
    text = re.sub(r"\s+", " ", text).strip()

    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""

    # Saudi phone numbers:
    # 0551234567
    # +966551234567
    # 966551234567
    phone_pattern = r"(?<!\d)(?:\+966|966|0)5\d{8}(?!\d)"
    text = re.sub(phone_pattern, "<PHONE>", text)

    # Saudi National ID / Iqama-shaped numbers
    # Must be 10 digits and start with 1 or 2
    national_id_pattern = r"(?<!\d)[12]\d{9}(?!\d)"
    text = re.sub(national_id_pattern, "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""

    text = mask_pii(text)
    text = normalize(text)

    return text