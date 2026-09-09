"""Lab 4 starter: per-model Arabic normalisation profiles."""

from dataclasses import dataclass
import re
import unicodedata

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer


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
        text = re.sub(
            r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
            "",
            text,
        )

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


# Load CAMeL models only once
_MLE = MLEDisambiguator.pretrained()

_TOKENIZER = MorphologicalTokenizer(
    disambiguator=_MLE,
    scheme="d3tok",
    split=True,
)


def segment(text: str) -> list[str]:
    words = text.split()
    return _TOKENIZER.tokenize(words)