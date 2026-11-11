"""F6.3 — benchmark no M1: carga a frio, latência p50/p95, tokens/s, RAM pico do Ollama e tamanho do GGUF.

Mede só o modelo: os contextos de RAG são calculados antes (a busca tem medição própria). A carga a
frio usa uma pergunta fora do conjunto medido — o Ollama reaproveita o prefixo do prompt anterior
(cache KV), o que inflaria os tokens/s de prompt e baixaria a latência da 1ª resposta a quente.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections.abc import Callable

from refauna_slm.config import Settings
from refauna_slm.contract import build_messages
from refauna_slm.eval.memory import footprint_gb, ollama_procs
from refauna_slm.rag.backends import OllamaBackend
from refauna_slm.rag.retriever import HybridRetriever

COLD_QUESTION = "O que é defaunação?"  # fora de demo.examples


def percentile(values: list[float], p: float) -> float:
    """Percentil com interpolação linear (p em 0–100)."""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class OllamaPeak(threading.Thread):
    """Pico de phys_footprint dos processos do Ollama enquanto ativo."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.peak, self._halt = interval, 0.0, threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.peak = max(self.peak, footprint_gb([p.pid for p in ollama_procs()]))
            self._halt.wait(self.interval)

    def stop(self) -> float:
        self._halt.set()
        self.join(timeout=5)
        return self.peak


def run_bench(
    settings: Settings,
    models: list[str],
    retriever: HybridRetriever,
    backend: OllamaBackend,
    warm_runs: int = 6,
    on_step: Callable[[str], None] | None = None,
) -> dict:
    questions = settings.demo.examples[:warm_runs]
    t0 = time.time()
    contexts = [retriever.hybrid_search(q) for q in questions]
    search_ms = (time.time() - t0) / len(questions) * 1000
    cold_ctx = retriever.hybrid_search(COLD_QUESTION)
    sizes = {m.model: m.size for m in backend.client.list().models}
    results = []
    for key in models:
        spec = settings.model(key)
        for tag in [m.model for m in backend.client.ps().models]:
            backend.unload(tag)  # carga a frio de verdade
        backend.loaded = None
        time.sleep(2)
        peak = OllamaPeak()
        peak.start()
        if on_step:
            on_step(f"{key}: carga a frio")
        cold = backend.chat(spec, build_messages(COLD_QUESTION, cold_ctx))
        warm = []
        for i, (q, ctx) in enumerate(zip(questions, contexts, strict=True), 1):
            if on_step:
                on_step(f"{key}: {i}/{len(questions)}")
            warm.append(backend.chat(spec, build_messages(q, ctx)))
        ram = peak.stop()
        lat = [g.latency_s for g in warm]
        prompt_tps = [g.prompt_eval_count / g.prompt_eval_duration_s for g in warm if g.prompt_eval_duration_s]
        results.append(
            {
                "model": key,
                "label": spec.label,
                "tag": spec.tag,
                "gguf_gb": round(sizes.get(spec.tag, 0) / 1e9, 2),  # GB decimal, como no `ollama list`
                "carga_fria_s": round(cold.load_duration_s, 1),
                "primeira_resposta_s": round(cold.latency_s, 1),
                "latencia_p50_s": round(percentile(lat, 50), 2),
                "latencia_p95_s": round(percentile(lat, 95), 2),
                "tokens_s": round(statistics.mean(g.tokens_per_s for g in warm if g.tokens_per_s), 1),
                "prompt_tokens_s": round(statistics.median(prompt_tps), 1) if prompt_tps else None,
                "tokens_resposta": round(statistics.mean(g.eval_count for g in warm), 1),
                "tokens_prompt": round(statistics.mean(g.prompt_eval_count for g in warm), 1),
                "ram_ollama_pico_gb": round(ram, 2),
                "n": len(warm),
            }
        )
    backend.unload(backend.loaded or "")
    return {
        "warm_runs": len(questions),
        "temperature": settings.ollama.options.get("temperature"),
        "busca_ms": round(search_ms, 1),
        "app_embedder_gb": round(footprint_gb([os.getpid()]), 2),
        "embedder_device": retriever.embedder.device,
        "models": results,
    }


def save_bench(result: dict, settings: Settings) -> None:
    settings.local_eval_dir.mkdir(parents=True, exist_ok=True)
    (settings.local_eval_dir / "bench.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
