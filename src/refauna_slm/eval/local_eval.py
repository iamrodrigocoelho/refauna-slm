"""F6.1 — paridade Colab × Ollama: mesmas perguntas, FT + RAG, temperature=0.

Compara por item com o `eval_detalhado.csv` do Colab (config "D: FT+RAG"): idioma, presença de
citação, citação correta e abstenção. Perguntas `unanswerable` usam os contextos fixos do teste
(como a trilha de abstenção do notebook) e devem gerar abstenção.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

import pandas as pd

from refauna_slm.config import Settings
from refauna_slm.contract import is_abstention, is_ptbr
from refauna_slm.rag.pipeline import CITE_RE, RagPipeline
from refauna_slm.sync import read_jsonl

COLAB_CONFIG = "D: FT+RAG"
# 10 perguntas fixas: 5 rag + 2 closed_book (avaliadas no Colab) + 3 unanswerable (contexto fixo)
N_RAG, N_CB, N_UN = 5, 2, 3


@dataclass
class ParityRow:
    model: str
    id: str
    kind: str
    question: str
    local_answer: str
    colab_answer: str | None
    local_ptbr: bool
    colab_ptbr: bool | None
    local_has_cite: bool
    colab_has_cite: bool | None
    local_cite_ok: bool | None
    colab_cite_ok: bool | None
    local_abstained: bool
    colab_abstained: bool | None
    expected_abstention: bool
    invalid_cites: str
    think_leak: bool
    truncated: bool
    local_chars: int
    colab_chars: int | None
    latency_s: float
    tokens_per_s: float | None
    divergences: str


def cite_ok(answer: str, contexts: list[dict], doc_id: str) -> bool:
    """Mesma definição do notebook: cita [n] de um trecho do documento de origem."""
    cites = [int(x) for x in re.findall(r"\[(\d)\]", answer)]
    return bool(cites) and any(1 <= n <= len(contexts) and contexts[n - 1]["doc_id"] == doc_id for n in cites)


def _bool(v) -> bool | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v).strip().lower() in {"true", "1", "1.0"}


def select_items(settings: Settings, colab: pd.DataFrame) -> list[dict]:
    """Seleção determinística (seed do projeto) e igual para todos os modelos."""
    test = read_jsonl(settings.data_dir / "sft" / "test.jsonl")
    by_q = {e["question"]: e for e in test}
    d = colab[colab.config == COLAB_CONFIG]
    rng = random.Random(settings.seed)
    items = []
    for kind, n in (("rag", N_RAG), ("closed_book", N_CB)):
        qs = [q for q in d[d.kind == kind].question.tolist() if q in by_q]
        items += [by_q[q] for q in rng.sample(qs, min(n, len(qs)))]
    un = [e for e in test if e["kind"] == "unanswerable"]
    items += rng.sample(un, min(N_UN, len(un)))
    return items


def run_parity(
    settings: Settings,
    model: str,
    pipeline: RagPipeline,
    on_item: Callable[[int, int, dict], None] | None = None,
) -> pd.DataFrame:
    spec = settings.model(model)
    if not spec.colab_run:
        raise ValueError(f"{model} não tem colab_run no settings.yaml (paridade só para modelos FT).")
    colab = pd.read_csv(settings.data_dir / "outputs" / spec.colab_run / "eval_detalhado.csv")
    colab_d = colab[colab.config == COLAB_CONFIG].set_index("question")
    by_id = pipeline.retriever.chunk_by_id
    rows = []
    items = select_items(settings, colab)
    for i, e in enumerate(items, 1):
        if on_item:
            on_item(i, len(items), e)
        fixed = [by_id[c] for c in e["context_chunk_ids"]] if e["kind"] == "unanswerable" else None
        ans = pipeline.ask(e["question"], model, use_rag=True, contexts=fixed, temperature=0)
        local = ans.answer
        c = colab_d.loc[e["question"]] if e["question"] in colab_d.index else None
        colab_ans = None if c is None else str(c["answer"])
        expected_abst = e["kind"] == "unanswerable"
        row = ParityRow(
            model=model,
            id=e["id"],
            kind=e["kind"],
            question=e["question"],
            local_answer=local,
            colab_answer=colab_ans,
            local_ptbr=is_ptbr(local),
            colab_ptbr=None if c is None else _bool(c["ptbr"]),
            local_has_cite=bool(CITE_RE.search(local)),
            colab_has_cite=None if colab_ans is None else bool(CITE_RE.search(colab_ans)),
            local_cite_ok=cite_ok(local, ans.contexts, e["doc_id"]) if e["kind"] == "rag" else None,
            colab_cite_ok=None if c is None or e["kind"] != "rag" else _bool(c["cite_ok"]),
            local_abstained=is_abstention(local),
            colab_abstained=None if colab_ans is None else is_abstention(colab_ans),
            expected_abstention=expected_abst,
            invalid_cites=",".join(map(str, ans.invalid_citations)),
            think_leak=bool(ans.generation.raw_thinking) or "<think>" in local,
            truncated=ans.generation.done_reason == "length",
            local_chars=len(local),
            colab_chars=None if colab_ans is None else len(colab_ans),
            latency_s=round(ans.latency_s, 2),
            tokens_per_s=round(ans.tokens_per_s, 1) if ans.tokens_per_s else None,
            divergences="",
        )
        row.divergences = "; ".join(divergences(row))
        rows.append(asdict(row))
    return pd.DataFrame(rows)


def divergences(r: ParityRow) -> list[str]:
    out = []
    if r.colab_ptbr is not None and r.local_ptbr != r.colab_ptbr:
        out.append(f"idioma (local={'pt' if r.local_ptbr else 'en'}, colab={'pt' if r.colab_ptbr else 'en'})")
    if not r.local_ptbr and r.colab_ptbr is None:
        out.append("resposta não está em PT-BR")
    if r.colab_has_cite is not None and r.local_has_cite != r.colab_has_cite:
        out.append(
            f"citação (local={'sim' if r.local_has_cite else 'não'}, colab={'sim' if r.colab_has_cite else 'não'})"
        )
    if r.colab_cite_ok is not None and r.local_cite_ok != r.colab_cite_ok:
        out.append(f"citação correta (local={r.local_cite_ok}, colab={r.colab_cite_ok})")
    if r.expected_abstention and not r.local_abstained:
        out.append("não se absteve (esperado)")
    if not r.expected_abstention and r.colab_abstained is not None and r.local_abstained != r.colab_abstained:
        out.append(f"abstenção (local={r.local_abstained}, colab={r.colab_abstained})")
    if r.invalid_cites:
        out.append(f"cita trecho inexistente [{r.invalid_cites}]")
    if r.think_leak:
        out.append("raciocínio <think> vazou")
    if r.truncated:
        out.append("cortada em num_predict")
    return out


def summarize(df: pd.DataFrame, settings: Settings, model: str) -> dict:
    """Taxas locais × Colab (mesmos itens) + taxas agregadas do Colab para abstenção."""
    spec = settings.model(model)
    resumo = json.loads((settings.data_dir / "outputs" / spec.colab_run / "eval_resumo.json").read_text())
    colab_d = next((s for s in resumo["summary"] if s["config"] == COLAB_CONFIG), {})
    ans = df[df.kind != "unanswerable"]
    rag = df[df.kind == "rag"]
    un = df[df.kind == "unanswerable"]

    def rate(s):
        s = s.dropna()
        return round(float(s.astype(float).mean()), 3) if len(s) else None

    return {
        "model": model,
        "n": len(df),
        "itens_com_divergencia": int((df.divergences != "").sum()),
        "ptbr": {"local": rate(ans.local_ptbr), "colab": rate(ans.colab_ptbr)},
        "com_citacao": {"local": rate(ans.local_has_cite), "colab": rate(ans.colab_has_cite)},
        "citacao_correta_rag": {"local": rate(rag.local_cite_ok), "colab": rate(rag.colab_cite_ok)},
        "abstencao_correta": {"local": rate(un.local_abstained), "colab_agregado": colab_d.get("abstencao_correta")},
        "falsa_abstencao": {"local": rate(ans.local_abstained), "colab": rate(ans.colab_abstained)},
        "chars_medio": {"local": round(float(ans.local_chars.mean())), "colab": round(float(ans.colab_chars.mean()))},
        "latencia_s_media": round(float(df.latency_s.mean()), 1),
        "tokens_por_s_medio": rate(df.tokens_per_s),
        "think_leak": int(df.think_leak.sum()),
        "truncadas": int(df.truncated.sum()),
    }


def save(df: pd.DataFrame, summary: dict, settings: Settings, model: str) -> None:
    out = settings.local_eval_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"parity_{model}.csv", index=False)
    (out / f"parity_{model}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- F6.2 — A/B/C/D local (métricas automáticas)

CONFIG_NAMES = {
    ("base", False): "A: base",
    ("base", True): "B: base+RAG",
    ("ft", False): "C: FT",
    ("ft", True): "D: FT+RAG",
}


def families(settings: Settings) -> dict[str, dict[str, str]]:
    """{'gemma': {'base': 'gemma_base', 'ft': 'gemma_ft'}, ...} só com modelos ativos."""
    out: dict[str, dict[str, str]] = {}
    for key, m in settings.models.items():
        if m.enabled:
            out.setdefault(m.family, {})["ft" if key.endswith("_ft") else "base"] = key
    return out


def sample_items(settings: Settings, n: int) -> list[dict]:
    """Amostra estratificada e determinística do sft/test.jsonl (mesma para todos os modelos)."""
    test = read_jsonl(settings.data_dir / "sft" / "test.jsonl")
    rng = random.Random(settings.seed)
    items = []
    for kind, frac in settings.eval.mix.items():
        pool = [e for e in test if e["kind"] == kind]
        items += rng.sample(pool, min(len(pool), round(n * frac)))
    return items


def oracle_in(contexts: list[dict], e: dict, by_id: dict) -> bool:
    """O trecho de origem (ou o vizinho, como no Recall@4 do notebook) está no contexto?"""
    src = by_id.get((e.get("source_chunk_ids") or [None])[0])
    return bool(src) and any(c["doc_id"] == src["doc_id"] and abs(c["idx"] - src["idx"]) <= 1 for c in contexts)


def run_abcd(
    settings: Settings,
    pipeline: RagPipeline,
    n: int | None = None,
    judge=None,
    on_item: Callable[[str, int, int], None] | None = None,
    configs: set[str] | None = None,
) -> pd.DataFrame:
    """Roda A/B/C/D por família, um modelo por vez (todas as perguntas antes de trocar de modelo).

    `configs`: subconjunto de {"A", "B", "C", "D"} (padrão: todos os disponíveis).
    """
    items = sample_items(settings, n or settings.eval.n_items)
    by_id = pipeline.retriever.chunk_by_id
    # Mesma busca para todos os modelos; `unanswerable` usa os contextos fixos do teste (trilha de abstenção)
    ctx_cache = {
        e["id"]: [by_id[c] for c in e["context_chunk_ids"]]
        if e["kind"] == "unanswerable"
        else pipeline.retriever.hybrid_search(e["question"])
        for e in items
    }
    rows = []
    for family, keys in families(settings).items():
        for role, model in sorted(keys.items()):
            for use_rag in (False, True):
                config = CONFIG_NAMES[(role, use_rag)]
                if configs and config[0] not in configs:
                    continue
                todo = [e for e in items if use_rag or e["kind"] != "unanswerable"]
                for i, e in enumerate(todo, 1):
                    if on_item:
                        on_item(f"{family} {config}", i, len(todo))
                    ctx = ctx_cache[e["id"]] if use_rag else None
                    ans = pipeline.ask(e["question"], model, use_rag=use_rag, contexts=ctx, temperature=0)
                    text = ans.answer
                    has_oracle = oracle_in(ctx, e, by_id) if ctx else False
                    row = {
                        "family": family,
                        "model": model,
                        "config": config,
                        "id": e["id"],
                        "kind": e["kind"],
                        "question": e["question"],
                        "answer": text,
                        "ptbr": is_ptbr(text),
                        "has_cite": bool(CITE_RE.search(text)) if use_rag else None,
                        "cite_ok": cite_ok(text, ctx, e["doc_id"]) if use_rag and e["kind"] == "rag" else None,
                        "abstained": is_abstention(text),
                        "oracle_in_ctx": has_oracle if use_rag and e["kind"] == "rag" else None,
                        "invalid_cites": ",".join(map(str, ans.invalid_citations)),
                        "latency_s": round(ans.generation.latency_s, 2),
                        "tokens_per_s": round(ans.tokens_per_s, 1) if ans.tokens_per_s else None,
                        "eval_count": ans.generation.eval_count,
                    }
                    if judge is not None and e["kind"] != "unanswerable":
                        ref = re.sub(r"\s?\[\d\]", "", e["reference"])
                        j = judge(e["question"], ref, text, ctx)
                        row.update(correcao=j.get("correcao"), fidelidade=j.get("fidelidade"), juiz_erro=j.get("erro"))
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_abcd(df: pd.DataFrame) -> list[dict]:
    """Uma linha por (família, config). Falsa abstenção: só quando o trecho certo FOI recuperado."""

    def rate(s):
        s = s.dropna()
        return round(float(s.astype(float).mean()), 3) if len(s) else None

    out = []
    for (family, config), g in df.groupby(["family", "config"], sort=True):
        answerable = g[g.kind != "unanswerable"]
        rag = g[g.kind == "rag"]
        row = {
            "family": family,
            "config": config,
            "model": g.model.iloc[0],
            "n": len(g),
            "ptbr": rate(answerable.ptbr),
            "com_citacao": rate(answerable.has_cite),
            "citacao_correta": rate(rag.cite_ok),
            "n_sem_resposta": int((g.kind == "unanswerable").sum()) if config.endswith("RAG") else None,
            "n_trecho_certo": int((rag.oracle_in_ctx == True).sum()) if config.endswith("RAG") else None,  # noqa: E712
            "abstencao_correta": rate(g[g.kind == "unanswerable"].abstained) if config.endswith("RAG") else None,
            "falsa_abstencao": rate(rag[rag.oracle_in_ctx == True].abstained) if config.endswith("RAG") else None,  # noqa: E712
            "latencia_s": round(float(g.latency_s.mean()), 2),
            "tokens_s": rate(g.tokens_per_s),
            "tokens_resposta": round(float(answerable.eval_count.mean()), 1),
        }
        if "correcao" in g:
            row.update(correcao=rate(answerable.correcao), fidelidade=rate(answerable.fidelidade))
        out.append(row)
    return out


def save_abcd(df: pd.DataFrame, summary: list[dict], settings: Settings, meta: dict) -> None:
    out = settings.local_eval_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "abcd_detalhado.csv", index=False)
    (out / "abcd_resumo.json").write_text(json.dumps({**meta, "summary": summary}, indent=2, ensure_ascii=False))
