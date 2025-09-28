from __future__ import annotations
import re, unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

# Macedonian-specific letters used for a light-weight filter
MK_LETTERS = set("ЃѓЌќЉљЊњЏџЅѕ")
CYRILLIC_RE = re.compile(r"[А-Яа-яЀ-ӿ]")   # broad Cyrillic range
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b")
SPACE_RE = re.compile(r"\s+")

KEEP_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЀ-ӿ\s]+")

def normalize(text: str, lowercase: bool = True) -> str:
    """NFKC, strip urls/emails, drop punctuation-like chars, collapse spaces."""
    s = unicodedata.normalize("NFKC", text)
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = KEEP_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    if lowercase:
        s = s.lower()
    return s

def is_macedonian(text: str, min_cyr_ratio: float = 0.6) -> bool:
    """Very light heuristic: mostly Cyrillic including at least one mk-specific letter."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyr = sum(bool(CYRILLIC_RE.match(c)) for c in letters)
    if cyr / len(letters) < min_cyr_ratio:
        return False
    return any(c in MK_LETTERS for c in text)

def iter_tokens(lines: Iterable[str]) -> Iterator[str]:
    """Split lines on whitespace."""
    for line in lines:
        for tok in line.split():
            yield tok

def build_top_vocab(clean_path: Path, out_path: Path, top_n: int = 200_000) -> None:
    """Counts tokens in a cleaned corpus and write the top-N list (one token per line)."""
    cnt = Counter()
    with clean_path.open("r", encoding="utf-8") as f:
        for line in f:
            cnt.update(line.strip().split())
    vocab = [w for w, _ in cnt.most_common(top_n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(vocab), encoding="utf-8")
