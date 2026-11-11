"""F1 — copia os artefatos do Colab (Google Drive) para data/ e valida (PRD §4.2)."""

from __future__ import annotations

import filecmp
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# (caminho no Drive relativo a refauna-slm/, caminho local relativo a data/, obrigatório)
FIXED_ARTIFACTS: list[tuple[str, str, bool]] = [
    ("data/chunks.jsonl", "chunks.jsonl", True),
    ("data/chunk_embeddings.npy", "chunk_embeddings.npy", False),
    ("data/docs.jsonl", "docs.jsonl", False),
    ("data/sft/test.jsonl", "sft/test.jsonl", False),
    ("outputs/comparacao_modelos.csv", "outputs/comparacao_modelos.csv", False),
    ("outputs/comparacao_modelos.png", "outputs/comparacao_modelos.png", False),
    ("gold_questions.jsonl", "gold/gold_questions.jsonl", False),
]
RUN_FILES = [
    "train_report.json",
    "eval_resumo.csv",
    "eval_resumo.json",
    "eval_detalhado.csv",
    "loss.png",
    "eval_notas.png",
]

# Schema de chunks.jsonl (PRD §4.1): campo -> tipos aceitos
CHUNK_SCHEMA: dict[str, tuple[type, ...]] = {
    "chunk_id": (str,),
    "idx": (int,),
    "doc_id": (str,),
    "title": (str,),
    "authors": (str,),
    "year": (int, type(None)),
    "doi": (str,),
    "license": (str,),
    "redistributable": (bool,),
    "section": (str,),
    "section_canon": (str,),
    "page_start": (int,),
    "page_end": (int,),
    "text": (str,),
    "n_tokens": (int,),
}
SFT_FIELDS = [
    "id",
    "doc_id",
    "kind",
    "question_type",
    "question",
    "reference",
    "messages",
    "oracle_index",
    "source_chunk_ids",
    "context_chunk_ids",
]
SFT_KINDS = {"rag", "unanswerable", "closed_book"}


class SyncError(Exception):
    """Falha que impede o uso dos artefatos (ex.: falta chunks.jsonl)."""


@dataclass
class CopyResult:
    copied: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    n_docs: int = 0
    n_docs_jsonl: int | None = None
    n_text_chunks: int = 0
    n_fichas: int = 0
    embeddings_shape: tuple[int, ...] | None = None
    embeddings_ok: bool = False
    test_by_kind: dict[str, int] = field(default_factory=dict)
    runs: list[str] = field(default_factory=list)
    has_gold: bool = False
    warnings: list[str] = field(default_factory=list)


def manual_download_list() -> list[str]:
    """Lista de arquivos para baixar à mão do Drive (quando REFAUNA_DRIVE_DIR não existe)."""
    lines = [
        f"refauna-slm/{src}  ->  data/{dst}{'  (obrigatório)' if req else ''}" for src, dst, req in FIXED_ARTIFACTS
    ]
    lines += [f"refauna-slm/outputs/<run>/{f}  ->  data/outputs/<run>/{f}" for f in RUN_FILES]
    return lines


def _plan(drive_dir: Path) -> list[tuple[Path, str, bool]]:
    plan = [(drive_dir / src, dst, req) for src, dst, req in FIXED_ARTIFACTS]
    for run_dir in sorted(p for p in (drive_dir / "outputs").glob("*") if p.is_dir()):
        plan += [(run_dir / f, f"outputs/{run_dir.name}/{f}", False) for f in RUN_FILES]
    return plan


def copy_artifacts(drive_dir: Path, data_dir: Path) -> CopyResult:
    """Copia só o que mudou (idempotente). Arquivos ausentes no Drive são listados, não apagados localmente."""
    res = CopyResult()
    for src, dst, _req in _plan(drive_dir):
        if not src.exists():
            res.missing.append(dst)
            continue
        target = data_dir / dst
        if target.exists() and filecmp.cmp(src, target, shallow=False):
            res.unchanged.append(dst)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        res.copied.append(dst)
    return res


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_chunks(chunks: list[dict]) -> list[str]:
    """Erros de schema em chunks.jsonl (lista vazia = ok)."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, c in enumerate(chunks):
        where = c.get("chunk_id", f"linha {i + 1}")
        for key, types in CHUNK_SCHEMA.items():
            if key not in c:
                errors.append(f"{where}: falta o campo '{key}'")
            elif not isinstance(c[key], types) or (bool not in types and isinstance(c[key], bool)):
                errors.append(f"{where}: '{key}' com tipo {type(c[key]).__name__}")
        cid = c.get("chunk_id")
        if cid in seen:
            errors.append(f"{where}: chunk_id duplicado")
        seen.add(cid)
        if c.get("section_canon") == "catalog":
            if cid != f"{c.get('doc_id')}_ficha" or c.get("idx") != -100 or c.get("page_start") != 0:
                errors.append(f"{where}: ficha do catálogo fora do padrão (<doc_id>_ficha, idx=-100, páginas 0)")
        if len(errors) >= 20:
            errors.append("... (mais erros omitidos)")
            break
    return errors


def validate(data_dir: Path) -> ValidationReport:
    """Valida os artefatos locais (PRD F1.2). Levanta SyncError se chunks.jsonl faltar ou for inválido."""
    rep = ValidationReport()
    chunks_path = data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SyncError(
            f"Falta {chunks_path}. Sem ele não há índice do RAG. Copie refauna-slm/data/chunks.jsonl do Drive."
        )
    try:
        chunks = read_jsonl(chunks_path)
    except json.JSONDecodeError as e:
        raise SyncError(f"{chunks_path} não é um JSONL válido: {e}") from e
    if not chunks:
        raise SyncError(f"{chunks_path} está vazio.")
    errors = validate_chunks(chunks)
    if errors:
        raise SyncError("chunks.jsonl fora do schema (PRD §4.1):\n  " + "\n  ".join(errors))

    text = [c for c in chunks if c["section_canon"] != "catalog"]
    rep.n_text_chunks, rep.n_fichas = len(text), len(chunks) - len(text)
    rep.n_docs = len({c["doc_id"] for c in text})

    docs_path = data_dir / "docs.jsonl"
    if docs_path.exists():
        rep.n_docs_jsonl = len(read_jsonl(docs_path))
    else:
        rep.warnings.append("docs.jsonl ausente (metadados da UI/atribuição).")

    emb_path = data_dir / "chunk_embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path, mmap_mode="r")
        rep.embeddings_shape = tuple(emb.shape)
        rep.embeddings_ok = emb.ndim == 2 and emb.shape[0] == len(chunks)
        if not rep.embeddings_ok:
            rep.warnings.append(
                f"chunk_embeddings.npy tem {emb.shape[0]} linhas, mas há {len(chunks)} chunks: "
                "não será reaproveitado (o índice recalcula os embeddings)."
            )
    else:
        rep.warnings.append("chunk_embeddings.npy ausente: `refauna index` vai calcular os embeddings no M1.")

    test_path = data_dir / "sft" / "test.jsonl"
    if test_path.exists():
        try:
            test = read_jsonl(test_path)
        except json.JSONDecodeError as e:
            raise SyncError(f"{test_path} não é um JSONL válido: {e}") from e
        bad = [e.get("id", "?") for e in test if any(k not in e for k in SFT_FIELDS) or e.get("kind") not in SFT_KINDS]
        if bad:
            rep.warnings.append(f"sft/test.jsonl: {len(bad)} exemplos fora do schema (ex.: {bad[:3]}).")
        rep.test_by_kind = dict(Counter(e.get("kind") for e in test))
        ids = {c["chunk_id"] for c in chunks}
        orphan = sum(1 for e in test for cid in e.get("source_chunk_ids", []) if cid not in ids)
        if orphan:
            rep.warnings.append(f"sft/test.jsonl referencia {orphan} chunk_ids que não estão em chunks.jsonl.")
    else:
        rep.warnings.append("sft/test.jsonl ausente: avaliação local e paridade indisponíveis.")

    rep.runs = sorted(p.parent.name for p in (data_dir / "outputs").glob("*/train_report.json"))
    if not (data_dir / "outputs" / "comparacao_modelos.csv").exists():
        rep.warnings.append("outputs/comparacao_modelos.csv ausente (slide Gemma × Qwen).")
    rep.has_gold = (data_dir / "gold" / "gold_questions.jsonl").exists()
    return rep
