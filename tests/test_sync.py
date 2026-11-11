"""F1 — refauna sync: cópia idempotente e validação dos artefatos."""

import json

import numpy as np
import pytest

from refauna_slm import sync


def chunk(doc_id="D1", idx=0, **kw):
    c = {
        "chunk_id": f"{doc_id}_c{idx:04d}",
        "idx": idx,
        "doc_id": doc_id,
        "title": "T",
        "authors": "A; B",
        "year": 2014,
        "doi": "",
        "license": "a verificar",
        "redistributable": False,
        "section": "Results",
        "section_canon": "results",
        "page_start": 1,
        "page_end": 1,
        "text": "texto",
        "n_tokens": 10,
    }
    return c | kw


def ficha(doc_id="D9"):
    return chunk(
        doc_id,
        chunk_id=f"{doc_id}_ficha",
        idx=-100,
        section="Ficha do catálogo",
        section_canon="catalog",
        page_start=0,
        page_end=0,
        year=None,
        redistributable=True,
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sft(kind, i):
    return {
        "id": f"sft_{i:05d}",
        "doc_id": "D1",
        "kind": kind,
        "question_type": "factual",
        "question": "q?",
        "reference": "r",
        "messages": [],
        "oracle_index": None,
        "source_chunk_ids": ["D1_c0000"],
        "context_chunk_ids": [],
    }


@pytest.fixture
def drive(tmp_path):
    d = tmp_path / "drive"
    rows = [chunk("D1", 0), chunk("D1", 1), chunk("D2", 0), ficha("D1"), ficha("D9")]
    write_jsonl(d / "data" / "chunks.jsonl", rows)
    np.save(d / "data" / "chunk_embeddings.npy", np.zeros((len(rows), 8), dtype=np.float32))
    write_jsonl(d / "data" / "docs.jsonl", [{"doc_id": "D1"}, {"doc_id": "D2"}])
    write_jsonl(d / "data" / "sft" / "test.jsonl", [sft("rag", 0), sft("rag", 1), sft("unanswerable", 2)])
    (d / "outputs" / "run-a").mkdir(parents=True)
    (d / "outputs" / "run-a" / "train_report.json").write_text("{}")
    return d


def test_copy_is_idempotent(drive, tmp_path):
    data = tmp_path / "data"
    first = sync.copy_artifacts(drive, data)
    assert "chunks.jsonl" in first.copied and "outputs/run-a/train_report.json" in first.copied
    assert "gold/gold_questions.jsonl" in first.missing
    second = sync.copy_artifacts(drive, data)
    assert second.copied == [] and set(second.unchanged) == set(first.copied)


def test_copy_updates_changed_file(drive, tmp_path):
    data = tmp_path / "data"
    sync.copy_artifacts(drive, data)
    write_jsonl(drive / "data" / "docs.jsonl", [{"doc_id": "D1"}])
    assert sync.copy_artifacts(drive, data).copied == ["docs.jsonl"]


def test_validate_report(drive, tmp_path):
    data = tmp_path / "data"
    sync.copy_artifacts(drive, data)
    rep = sync.validate(data)
    assert (rep.n_docs, rep.n_text_chunks, rep.n_fichas, rep.n_docs_jsonl) == (2, 3, 2, 2)
    assert rep.embeddings_ok and rep.embeddings_shape == (5, 8)
    assert rep.test_by_kind == {"rag": 2, "unanswerable": 1}
    assert rep.runs == ["run-a"]


def test_missing_chunks_fails_clearly(tmp_path):
    with pytest.raises(sync.SyncError, match="chunks.jsonl"):
        sync.validate(tmp_path)


def test_schema_errors(tmp_path):
    bad = chunk(year="2014")
    del bad["text"]
    write_jsonl(tmp_path / "chunks.jsonl", [bad, chunk(), ficha() | {"idx": 3}])
    with pytest.raises(sync.SyncError) as e:
        sync.validate(tmp_path)
    msg = str(e.value)
    assert "falta o campo 'text'" in msg and "'year' com tipo str" in msg and "duplicado" in msg and "ficha" in msg


def test_embeddings_mismatch_is_warning(drive, tmp_path):
    data = tmp_path / "data"
    sync.copy_artifacts(drive, data)
    np.save(data / "chunk_embeddings.npy", np.zeros((2, 8), dtype=np.float32))
    rep = sync.validate(data)
    assert not rep.embeddings_ok and any("não será reaproveitado" in w for w in rep.warnings)


def test_manual_list_mentions_required_file():
    assert any("chunks.jsonl" in line and "obrigatório" in line for line in sync.manual_download_list())
