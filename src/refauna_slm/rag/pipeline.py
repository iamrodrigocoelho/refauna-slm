"""F4 — ask(question, model, use_rag): busca → contrato de prompt → Ollama → citações [n]."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from refauna_slm.config import Settings
from refauna_slm.contract import build_messages, is_abstention, is_ptbr, short_ref
from refauna_slm.rag.backends import Generation, OllamaBackend
from refauna_slm.rag.retriever import HybridRetriever

CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    n: int
    chunk_id: str
    ref: str
    title: str
    year: int | None
    section: str
    pages: str
    doi: str
    text: str


@dataclass
class Answer:
    question: str
    model: str
    use_rag: bool
    answer: str = ""
    contexts: list[dict] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    search_s: float = 0.0
    generation: Generation = field(default_factory=Generation)

    @property
    def latency_s(self) -> float:
        return self.search_s + self.generation.latency_s

    @property
    def tokens_per_s(self) -> float | None:
        return self.generation.tokens_per_s

    @property
    def abstained(self) -> bool:
        return is_abstention(self.answer)

    @property
    def ptbr(self) -> bool:
        return is_ptbr(self.answer)


def pages(c: dict) -> str:
    """'Ficha do catálogo' para fichas; 'p. 4' ou 'p. 4–5' para trechos."""
    if c.get("section_canon") == "catalog":
        return "Ficha do catálogo"
    a, b = c.get("page_start"), c.get("page_end")
    return f"p. {a}" if a == b else f"p. {a}–{b}"


def extract_citations(answer: str, contexts: list[dict]) -> tuple[list[Citation], list[int]]:
    """Números [n] citados → trechos do contexto (na ordem da 1ª citação). Fora de 1..k = inválidos."""
    seen, cites, invalid = set(), [], []
    for m in CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        if not 1 <= n <= len(contexts):
            invalid.append(n)
            continue
        c = contexts[n - 1]
        cites.append(
            Citation(
                n=n,
                chunk_id=c["chunk_id"],
                ref=short_ref(c),
                title=c.get("title", ""),
                year=c.get("year"),
                section=c.get("section", ""),
                pages=pages(c),
                doi=c.get("doi", ""),
                text=c.get("text", ""),
            )
        )
    return cites, invalid


class RagPipeline:
    def __init__(
        self, settings: Settings, retriever: HybridRetriever | None = None, backend: OllamaBackend | None = None
    ):
        self.settings = settings
        self._retriever = retriever
        self.backend = backend or OllamaBackend(settings)

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.settings)
        return self._retriever

    def stream(
        self,
        question: str,
        model: str = "gemma_ft",
        use_rag: bool = True,
        contexts: list[dict] | None = None,
        **options,
    ) -> Iterator[tuple[str, Answer]]:
        """Gera (delta, Answer parcial). `contexts` fixos ignoram a busca (avaliação de abstenção)."""
        spec = self.settings.model(model)
        ans = Answer(question=question, model=model, use_rag=use_rag)
        if use_rag:
            t0 = time.time()
            ans.contexts = contexts if contexts is not None else self.retriever.hybrid_search(question)
            ans.search_s = time.time() - t0
        msgs = build_messages(question, ans.contexts or None)
        for delta, gen in self.backend.stream_chat(spec, msgs, **options):
            ans.generation, ans.answer = gen, gen.text
            yield delta, ans
        ans.citations, ans.invalid_citations = extract_citations(ans.answer, ans.contexts)
        yield "", ans

    def ask(self, question: str, model: str = "gemma_ft", use_rag: bool = True, **kw) -> Answer:
        ans = None
        for _, partial in self.stream(question, model, use_rag, **kw):
            ans = partial
        return ans
