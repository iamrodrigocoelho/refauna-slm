"""OllamaBackend: /api/chat com streaming, think=false no Qwen e um modelo carregado por vez."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import ollama

from refauna_slm.config import ModelSpec, Settings

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_think(text: str) -> str:
    """Remove <think>…</think> residual (e um bloco aberto sem fechamento)."""
    return THINK_RE.sub("", text or "").strip()


@dataclass
class Generation:
    text: str = ""
    latency_s: float = 0.0
    eval_count: int = 0
    eval_duration_s: float = 0.0
    load_duration_s: float = 0.0
    prompt_eval_count: int = 0
    prompt_eval_duration_s: float = 0.0
    done_reason: str | None = None
    raw_thinking: str = ""

    @property
    def tokens_per_s(self) -> float | None:
        return self.eval_count / self.eval_duration_s if self.eval_duration_s else None


@dataclass
class OllamaBackend:
    settings: Settings
    loaded: str | None = None
    client: ollama.Client = field(init=False)

    def __post_init__(self):
        self.client = ollama.Client(host=self.settings.ollama.host)

    def payload(self, spec: ModelSpec, messages: list[dict], stream: bool, **options) -> dict:
        """Argumentos de /api/chat (testável sem Ollama). Qwen ⇒ think=false."""
        kw = {
            "model": spec.tag,
            "messages": messages,
            "stream": stream,
            "options": {**self.settings.ollama.options, **options},
        }
        if spec.think is not None:
            kw["think"] = spec.think
        return kw

    def unload(self, tag: str) -> None:
        try:
            self.client.generate(model=tag, keep_alive=0)
        except ollama.ResponseError:
            pass

    def ensure_only(self, spec: ModelSpec, timeout_s: float = 15.0) -> None:
        """Um modelo por vez: descarrega os outros (keep_alive=0) e ESPERA saírem da memória.

        O keep_alive=0 retorna antes de o processo liberar a RAM; sem esperar, os dois modelos
        coexistem por alguns segundos e o M1 de 8 GB entra em swap.
        """
        running = {m.model for m in self.client.ps().models}
        others = {t for t in running | ({self.loaded} if self.loaded else set()) if t and t != spec.tag}
        for tag in others:
            self.unload(tag)
        deadline = time.time() + timeout_s
        while others & {m.model for m in self.client.ps().models} and time.time() < deadline:
            time.sleep(0.2)
        self.loaded = spec.tag

    def stream_chat(self, spec: ModelSpec, messages: list[dict], **options) -> Iterator[tuple[str, Generation]]:
        """Gera (delta, estado acumulado). O texto do estado já vem sem <think>."""
        self.ensure_only(spec)
        gen = Generation()
        raw = ""
        t0 = time.time()
        for part in self.client.chat(**self.payload(spec, messages, stream=True, **options)):
            msg = part.message
            if msg.thinking:
                gen.raw_thinking += msg.thinking
            delta = msg.content or ""
            raw += delta
            gen.text = strip_think(raw)
            if part.done:
                gen.eval_count = part.eval_count or 0
                gen.eval_duration_s = (part.eval_duration or 0) / 1e9
                gen.load_duration_s = (part.load_duration or 0) / 1e9
                gen.prompt_eval_count = part.prompt_eval_count or 0
                gen.prompt_eval_duration_s = (part.prompt_eval_duration or 0) / 1e9
                gen.done_reason = part.done_reason
            gen.latency_s = time.time() - t0
            yield delta, gen

    def chat(self, spec: ModelSpec, messages: list[dict], **options) -> Generation:
        gen = Generation()
        for _, partial in self.stream_chat(spec, messages, **options):
            gen = partial
        return gen
