"""BM25 — `STOP`, `bm25_tokens` e `index_text` copiados do notebook (seção 3)."""

import re
import unicodedata

from rank_bm25 import BM25Okapi

STOP = set(
    (
        "a o e de da do das dos em no na nos nas um uma para por com que se ao os as sao "
        "the of and to in is for on with by as are was were be this that from at an or"
    ).split()
)


def bm25_tokens(s):
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
    return [t for t in re.findall(r"[a-z0-9]+", s) if t not in STOP and len(t) > 1]


def index_text(c):
    return f"{c['title']} — {c['section']}\n{c['text']}"


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([bm25_tokens(index_text(c)) for c in chunks])
