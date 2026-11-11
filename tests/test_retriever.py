"""F3 — RRF, select_diverse, bm25_tokens e fichas do catálogo (iguais ao notebook)."""

import re
from collections import defaultdict

import nbformat
import pytest

from refauna_slm.config import ROOT
from refauna_slm.index import lexical
from refauna_slm.rag import retriever
from refauna_slm.rag.pipeline import extract_citations, pages

NB = ROOT / "notebooks" / "refauna_gemma4_finetune_rag.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    """Executa a célula de BM25/RRF/select_diverse do notebook."""
    cells = nbformat.read(NB, as_version=4).cells
    src = next(c.source for c in cells if c.cell_type == "code" and "def rrf_fuse" in c.source)
    idx = next(c.source for c in cells if c.cell_type == "code" and "def index_text" in c.source)
    ns: dict = {"re": re, "unicodedata": __import__("unicodedata"), "defaultdict": defaultdict}
    exec(compile(src, str(NB), "exec"), ns)  # noqa: S102 — código do próprio projeto
    exec(re.search(r"def index_text\(c\):\n(?:    .*\n)+", idx + "\n").group(0), ns)  # noqa: S102
    return ns


def c(doc, idx, **kw):
    return {"chunk_id": f"{doc}_c{idx:04d}", "doc_id": doc, "idx": idx, "title": "T", "section": "S", "text": "x"} | kw


def test_bm25_tokens_equals_notebook(nb):
    for s in ["Dasyprocta leporina no PNT", "Reintrodução de ANTAS (Tapirus terrestris) em 2017!", "the of and a o"]:
        assert lexical.bm25_tokens(s) == nb["bm25_tokens"](s)
    assert lexical.STOP == nb["STOP"]
    assert lexical.bm25_tokens("Reintrodução da Cutia") == ["reintroducao", "cutia"]


def test_index_text_equals_notebook(nb):
    ch = c("D", 0, title="Título", section="Results", text="Texto")
    assert lexical.index_text(ch) == nb["index_text"](ch) == "Título — Results\nTexto"


def test_rrf_equals_notebook(nb):
    rankings = [["a", "b", "c", "d"], ["c", "a", "e"]]
    assert retriever.rrf_fuse(rankings) == nb["rrf_fuse"](rankings)
    assert retriever.rrf_fuse(rankings)[:2] == ["a", "c"]


def test_select_diverse_drops_neighbors(nb):
    chunks = [c("D1", 3), c("D1", 4), c("D1", 6), c("D2", 4), c("D1", -100, chunk_id="D1_ficha")]
    by_id = {x["chunk_id"]: x for x in chunks}
    ranked = [x["chunk_id"] for x in chunks]
    out = retriever.select_diverse(ranked, by_id, 4)
    assert [x["chunk_id"] for x in out] == ["D1_c0003", "D1_c0006", "D2_c0004", "D1_ficha"]
    assert out == nb["select_diverse"](ranked, by_id, 4)
    assert retriever.select_diverse(ranked, by_id, 2, exclude_ids={"D1_c0003"})[0]["chunk_id"] == "D1_c0004"


def test_pages_and_catalog():
    assert pages({"section_canon": "catalog", "page_start": 0, "page_end": 0}) == "Ficha do catálogo"
    assert pages({"section_canon": "results", "page_start": 4, "page_end": 5}) == "p. 4–5"
    assert pages({"section_canon": "results", "page_start": 4, "page_end": 4}) == "p. 4"


def test_extract_citations():
    ctx = [c("D1", 0, year=2014, doi="10.1/x"), c("D2", 1, section_canon="catalog", page_start=0, page_end=0)]
    cites, invalid = extract_citations("Fato [2]. Outro [1] e [2]. Erro [5].", ctx)
    assert [x.n for x in cites] == [2, 1] and invalid == [5]
    assert cites[0].pages == "Ficha do catálogo" and cites[1].doi == "10.1/x"
