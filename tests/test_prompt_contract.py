"""Contrato de prompt (PRD §4.1): prompts.yaml e contract.py == célula "Contrato de prompt" dos notebooks."""

import json
import re
from pathlib import Path

import nbformat
import pytest

from refauna_slm import contract
from refauna_slm.config import ROOT, load_yaml

NOTEBOOKS = {
    "gemma": ROOT / "notebooks" / "refauna_gemma4_finetune_rag.ipynb",
    "qwen": ROOT / "notebooks" / "refauna_qwen35_2b_finetune_rag.ipynb",
}
CONTRACT_FUNCS = ["short_ref", "format_context", "build_user_message", "build_messages", "is_abstention", "is_ptbr"]

SAMPLE_CHUNKS = [
    {
        "doc_id": "REF-L1-004",
        "authors": "Bruno Cid; L. Figueira; F. A. S. Fernandez",
        "year": 2014,
        "title": "Short-Term Success in the Reintroduction of the Red-Humped Agouti Dasyprocta leporina",
        "text": "  Texto A.  ",
    },
    {"doc_id": "X1", "authors": "Fernandez, F. & Silva, J.", "year": None, "title": "Curto", "text": "Texto B"},
    {"doc_id": "X2", "authors": "Genes et al.", "year": 2017, "title": "", "text": "Texto C"},
    {"doc_id": "X3", "authors": "Alexandre Silva Júnior", "year": 2020, "title": "Tese", "text": "Texto D"},
    {"doc_id": "X4", "authors": "", "year": 2019, "title": "Sem autor", "text": "Texto E"},
]
SAMPLE_TEXTS = [
    "Não encontrei essa informação nos documentos do Refauna.",
    "Os trechos recuperados não contêm a resposta.",
    "As cutias dispersam sementes de palmeiras [1].",
    "The agoutis were reintroduced in Tijuca National Park and their population grew.",
    "",
    None,
]


def contract_cell(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    cells = [c.source for c in nb.cells if c.cell_type == "code" and c.source.lstrip().startswith("SYSTEM_PROMPT = (")]
    assert len(cells) == 1, f"célula 'Contrato de prompt' não encontrada (ou duplicada) em {path.name}"
    return cells[0]


def notebook_namespace(path: Path) -> dict:
    ns: dict = {"re": re}
    exec(compile(contract_cell(path), str(path), "exec"), ns)  # noqa: S102 — código do próprio projeto
    return ns


@pytest.fixture(scope="module", params=list(NOTEBOOKS), ids=list(NOTEBOOKS))
def nb(request) -> dict:
    return notebook_namespace(NOTEBOOKS[request.param])


def real_chunks(n: int = 40) -> list[dict]:
    """Amostra de chunks reais (se `refauna sync` já rodou), incluindo fichas do catálogo."""
    path = ROOT / "data" / "chunks.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:: max(1, len(rows) // n)]


def test_prompts_yaml_equals_notebook(nb):
    assert load_yaml("prompts.yaml")["system_prompt"] == nb["SYSTEM_PROMPT"]


def test_contract_system_prompt_equals_notebook(nb):
    assert contract.SYSTEM_PROMPT == nb["SYSTEM_PROMPT"]


def test_system_prompt_is_single_paragraph():
    sp = contract.SYSTEM_PROMPT
    assert "\n" not in sp and "  " not in sp and sp == sp.strip()


def test_abstention_regex_equals_notebook(nb):
    assert contract.ABSTAIN_RE.pattern == nb["ABSTAIN_RE"].pattern
    assert contract.ABSTAIN_RE.flags == nb["ABSTAIN_RE"].flags


def test_notebook_abstain_answers_are_detected(nb):
    assert all(contract.is_abstention(a) for a in nb["ABSTAIN_ANSWERS"])


def test_stopword_sets_equal_notebook(nb):
    assert contract.PT_SW == nb["PT_SW"] and contract.EN_SW == nb["EN_SW"] and contract.NAME_SUFFIX == nb["NAME_SUFFIX"]


@pytest.mark.parametrize("name", CONTRACT_FUNCS)
def test_function_source_equals_notebook(nb, name):
    """O corpo de cada função é idêntico ao do notebook (ignorando só formatação)."""
    import inspect

    def norm(src: str) -> str:  # sem comentários, espaços nem vírgulas finais (estilo do ruff format)
        return re.sub(r",(?=[\]\)\}])", "", re.sub(r"\s+", "", re.sub(r"#[^\n]*", "", src)))

    local_src = inspect.getsource(getattr(contract, name))
    nb_src = re.search(rf"^def {name}\(.*?(?=^\S|\Z)", contract_cell(NOTEBOOKS["gemma"]), re.S | re.M).group(0)
    assert norm(local_src) == norm(nb_src)


def test_functions_behave_like_notebook(nb):
    chunks = SAMPLE_CHUNKS + real_chunks()
    for c in chunks:
        assert contract.short_ref(c) == nb["short_ref"](c)
    for i in range(0, len(chunks), 4):
        group = chunks[i : i + 4]
        assert contract.format_context(group) == nb["format_context"](group)
        for q in ["  Qual o papel das cutias?  ", "O que é refaunação?"]:
            assert contract.build_user_message(q, group) == nb["build_user_message"](q, group)
            assert contract.build_messages(q, group) == nb["build_messages"](q, group)
    assert contract.build_messages(" Pergunta sem contexto ") == nb["build_messages"](" Pergunta sem contexto ")
    assert contract.build_user_message("P", []) == "P"
    for t in SAMPLE_TEXTS:
        assert contract.is_abstention(t) == nb["is_abstention"](t)
        assert contract.is_ptbr(t) == nb["is_ptbr"](t)


def test_short_ref_format():
    assert (
        contract.short_ref(SAMPLE_CHUNKS[0])
        == "(Cid et al., 2014 — Short-Term Success in the Reintroduction of the Red-Humpe...)"
    )
    assert contract.short_ref(SAMPLE_CHUNKS[3]) == "(Silva, 2020 — Tese)"
    assert contract.short_ref(SAMPLE_CHUNKS[4]) == "(2019 — Sem autor)"


def test_user_message_layout():
    msg = contract.build_user_message("Pergunta?", SAMPLE_CHUNKS[:2])
    assert msg.startswith("CONTEXTO:\n[1] (Cid et al., 2014 — ")
    assert "\n[2] (Fernandez et al., s.d. — Curto) Texto B\n\nPERGUNTA: Pergunta?" in msg


def test_sft_test_messages_match_contract():
    """As mensagens de sft/test.jsonl (geradas no Colab) são reproduzidas por build_messages()."""
    test_path, chunks_path = ROOT / "data" / "sft" / "test.jsonl", ROOT / "data" / "chunks.jsonl"
    if not (test_path.exists() and chunks_path.exists()):
        pytest.skip("rode `refauna sync` primeiro")
    by_id = {c["chunk_id"]: c for c in map(json.loads, chunks_path.read_text(encoding="utf-8").splitlines()) if c}
    for e in map(json.loads, test_path.read_text(encoding="utf-8").splitlines()):
        ctx = [by_id[cid] for cid in e["context_chunk_ids"]] or None
        assert contract.build_messages(e["question"], ctx) == e["messages"][:2], e["id"]
