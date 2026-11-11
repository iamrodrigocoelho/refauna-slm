"""EmbeddingGemma (o MESMO modelo do Colab): float32 (não suporta float16), MPS no M1, batch ≤ 16."""

from __future__ import annotations

import logging
import os
import sys
from functools import cached_property
from pathlib import Path

import numpy as np


def quiet_and_offline(model_name: str) -> None:
    """Silencia logs de carga e, se o modelo já está no cache do HF, não toca a rede (demo offline)."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub")
    cached = any((hub.expanduser() / f"models--{model_name.replace('/', '--')}" / "snapshots").glob("*/config.json"))
    if cached and os.environ.setdefault("HF_HUB_OFFLINE", "1") == "1":
        hf = sys.modules.get("huggingface_hub.constants")  # já importado: a constante foi lida antes do env
        if hf is not None:
            hf.HF_HUB_OFFLINE = True


def best_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class Embedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16):
        self.model_name = model_name
        self.device = device or best_device()
        self.batch_size = batch_size

    @cached_property
    def model(self):
        quiet_and_offline(self.model_name)
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        return SentenceTransformer(self.model_name, device=self.device, model_kwargs={"torch_dtype": torch.float32})

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode_query(query, normalize_embeddings=True)

    def encode_document(self, texts: list[str], show_progress_bar: bool = True) -> np.ndarray:
        return self.model.encode_document(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=show_progress_bar
        )
