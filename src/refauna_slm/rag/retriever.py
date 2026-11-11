"""Busca híbrida (F3.3): denso top-20 + BM25 top-20 → RRF → select_diverse → top-4.

`rrf_fuse`, `select_diverse`, `hybrid_search` e `retrieval_metrics` são cópias do notebook (seção 3).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from refauna_slm.config import Settings
from refauna_slm.index import load_chunks
from refauna_slm.index.embedder import Embedder
from refauna_slm.index.lexical import bm25_tokens, build_bm25
from refauna_slm.index.vector_store import VectorStore, fingerprint
from refauna_slm.sync import SyncError


def rrf_fuse(rankings, rrf_k=60):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] += 1.0 / (rrf_k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def select_diverse(ranked_ids, by_id, k, exclude_ids=()):
    """Remove excluídos e chunks vizinhos (sobrepostos) do mesmo documento."""
    out = []
    for cid in ranked_ids:
        if cid in exclude_ids:
            continue
        c = by_id[cid]
        if any(o["doc_id"] == c["doc_id"] and abs(o["idx"] - c["idx"]) <= 1 for o in out):
            continue
        out.append(c)
        if len(out) == k:
            break
    return out


class HybridRetriever:
    """Carrega chunks, BM25 (em memória) e o Chroma persistido por `refauna index`."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.cfg = settings.retrieval
        self.chunks = load_chunks(settings)
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.store = VectorStore(settings.chroma_dir)
        if not self.store.is_current(fingerprint(settings.data_dir / "chunks.jsonl"), self.cfg.embed_model):
            raise SyncError("Índice ausente ou desatualizado em relação a chunks.jsonl. Rode `refauna index`.")
        self.bm25 = build_bm25(self.chunks)
        self.embedder = embedder or Embedder(self.cfg.embed_model)

    def warmup(self) -> None:
        self.embedder.encode_query("aquecimento")

    def hybrid_search(self, query, k=None, n_candidates=None, exclude_ids=()):
        k = k or self.cfg.top_k
        n_candidates = n_candidates or self.cfg.n_candidates
        q = self.embedder.encode_query(query)
        dense = self.store.query(q, n_candidates)
        scores = self.bm25.get_scores(bm25_tokens(query))
        lexical = [self.chunk_ids[i] for i in np.argsort(scores)[::-1][:n_candidates]]
        return select_diverse(rrf_fuse([dense, lexical], self.cfg.rrf_k), self.chunk_by_id, k, exclude_ids)

    def retrieval_metrics(self, items, k=None):
        """Recall@k, MRR@20 e doc_recall@k — mesma definição do notebook (aceita o chunk vizinho)."""
        k = k or self.cfg.top_k
        hit = doc_hit = 0
        rr = 0.0
        for e in items:
            src = self.chunk_by_id[e["source_chunk_ids"][0]]
            res = self.hybrid_search(e["question"], k=20)
            pos = next(
                (i for i, c in enumerate(res) if c["doc_id"] == src["doc_id"] and abs(c["idx"] - src["idx"]) <= 1),
                None,
            )
            hit += pos is not None and pos < k
            rr += 1 / (pos + 1) if pos is not None else 0
            doc_hit += any(c["doc_id"] == src["doc_id"] for c in res[:k])
        n = len(items)
        return {"n": n, f"recall@{k}": hit / n, "mrr@20": rr / n, f"doc_recall@{k}": doc_hit / n}
