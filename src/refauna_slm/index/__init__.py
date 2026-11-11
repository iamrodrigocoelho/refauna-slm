"""Índice local (F3.1): ChromaDB persistente, reaproveitando chunk_embeddings.npy quando válido."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from refauna_slm.config import Settings
from refauna_slm.index.embedder import Embedder
from refauna_slm.index.lexical import index_text
from refauna_slm.index.vector_store import VectorStore, fingerprint
from refauna_slm.sync import SyncError, read_jsonl


@dataclass
class IndexResult:
    n: int
    embeddings_source: str  # "chunk_embeddings.npy" | "calculado no Mac" | "existente"
    seconds: float
    path: Path


def load_chunks(settings: Settings) -> list[dict]:
    path = settings.data_dir / "chunks.jsonl"
    if not path.exists():
        raise SyncError(f"Falta {path}. Rode `refauna sync` primeiro.")
    return read_jsonl(path)


def build_index(
    settings: Settings,
    rebuild: bool = False,
    embedder: Embedder | None = None,
    on_status: Callable[[str], None] | None = None,
) -> IndexResult:
    import time

    t0 = time.time()
    chunks_path = settings.data_dir / "chunks.jsonl"
    chunks = load_chunks(settings)
    fp = fingerprint(chunks_path)
    model = settings.retrieval.embed_model
    store = VectorStore(settings.chroma_dir)
    if not rebuild and store.is_current(fp, model):
        return IndexResult(store.manifest["n"], "existente", time.time() - t0, store.path)

    emb_path = settings.data_dir / "chunk_embeddings.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    if emb is not None and emb.ndim == 2 and len(emb) == len(chunks):
        source = "chunk_embeddings.npy"
    else:
        if on_status:
            on_status(f"calculando {len(chunks)} embeddings com {model} (batch 16)...")
        embedder = embedder or Embedder(model)
        emb = embedder.encode_document([index_text(c) for c in chunks])
        source = "calculado no Mac"
    if on_status:
        on_status("gravando ChromaDB...")
    n = store.build(chunks, np.asarray(emb, dtype=np.float32), fp, model, source)
    return IndexResult(n, source, time.time() - t0, store.path)
