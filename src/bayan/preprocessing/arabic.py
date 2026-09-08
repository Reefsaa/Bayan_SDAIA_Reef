"""Lab 4 starter: per-model Arabic normalisation profiles."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    # TODO(Lab 4): implement the two course profiles and preserve a separate display copy.
    raise NotImplementedError


def segment(text: str) -> list[str]:
    # TODO(Lab 4): wire the chosen CAMeL Tools clitic segmentation scheme.
    raise NotImplementedError
"""Lab 4: per-model Arabic normalisation profiles."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    text = unicodedata.normalize("NFC", text)

    # Remove tatweel
    text = text.replace("ـ", "")

    # Remove Arabic diacritics when requested
    if profile.dediacritize:
        text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)

    if profile.name == "bayan_ar_v1":
        text = (
            text.replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ؤ", "و")
            .replace("ئ", "ي")
            .replace("ى", "ي")
            .replace("ة", "ه")
        )

    return text


def segment(text: str) -> list[str]:
    # Lab 4 Step 3 will implement CAMeL Tools clitic segmentation.
    return text.split()