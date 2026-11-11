"""ChromaDB persistente em data/chroma (cosseno), montado a partir de chunks.jsonl."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import chromadb
import numpy as np

COLLECTION = "refauna"
MANIFEST = "refauna_manifest.json"


def chroma_meta(c):
    """Copiado do notebook (seção 3)."""
    m = {k: (c.get(k) or "") for k in ("doc_id", "title", "authors", "doi", "section", "section_canon", "license")}
    m.update(year=int(c.get("year") or 0), page_start=int(c["page_start"]), page_end=int(c["page_end"]))
    return m


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class VectorStore:
    def __init__(self, path: Path):
        self.path = path

    @property
    def manifest(self) -> dict | None:
        p = self.path / MANIFEST
        return json.loads(p.read_text()) if p.exists() else None

    def is_current(self, chunks_fp: str, embed_model: str) -> bool:
        m = self.manifest
        return bool(m) and m.get("chunks_sha") == chunks_fp and m.get("embed_model") == embed_model

    def build(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        chunks_fp: str,
        embed_model: str,
        embeddings_source: str,
        on_batch: Callable[[int], None] | None = None,
    ) -> int:
        """Recria a coleção do zero (apaga só data/chroma/)."""
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True, exist_ok=True)
        col = self.client().create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 500):
            b = chunks[i : i + 500]
            col.add(
                ids=[c["chunk_id"] for c in b],
                embeddings=embeddings[i : i + 500].tolist(),
                documents=[c["text"] for c in b],
                metadatas=[chroma_meta(c) for c in b],
            )
            if on_batch:
                on_batch(len(b))
        (self.path / MANIFEST).write_text(
            json.dumps(
                {
                    "chunks_sha": chunks_fp,
                    "embed_model": embed_model,
                    "embeddings_source": embeddings_source,
                    "n": col.count(),
                    "chromadb": chromadb.__version__,
                },
                indent=2,
            )
        )
        return col.count()

    def client(self):
        return chromadb.PersistentClient(path=str(self.path), settings=chromadb.Settings(anonymized_telemetry=False))

    def collection(self):
        return self.client().get_collection(COLLECTION)

    def query(self, embedding: np.ndarray, n: int) -> list[str]:
        col = self.collection()
        return col.query(query_embeddings=[embedding.tolist()], n_results=min(n, col.count()))["ids"][0]
