"""F6.2–F6.4 — resumo A/B/C/D, percentis do benchmark e leitura calculada do relatório (sem Ollama)."""

import pandas as pd
import pytest

from refauna_slm.eval import local_eval
from refauna_slm.eval.bench import percentile
from refauna_slm.eval.report import br, hypothesis_md, md_table, reading_md

CONFIGS = ["A: base", "B: base+RAG", "C: FT", "D: FT+RAG"]


def colab(correcao, **extra):
    df = pd.DataFrame({"config": CONFIGS, "correcao": correcao, **extra}).set_index("config")
    return {"summary": df}


def test_percentile():
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4, 5, 6], 95) == pytest.approx(5.75)


def test_hypothesis_negative_result_is_reported():
    md = hypothesis_md({"Qwen3.5-2B": colab([1.72, 2.52, 2.20, 2.32])})
    assert "**B > D > C > A**" in md and "Não confirma" in md


def test_hypothesis_confirmed():
    assert "Confirma a hipótese" in hypothesis_md({"X": colab([1.0, 2.0, 1.5, 3.0])})


def test_reading_is_computed_per_model():
    gemma = colab([1.12, 2.47, 2.17, 2.29], fidelidade=[None, 4.35, None, 4.17], latencia_s=[4.3, 13.0, 15.6, 11.3])
    qwen = colab(
        [1.72, 2.52, 2.20, 2.32], abstencao_correta=[None, 0.125, None, 1.0], latencia_s=[17.1, 14.3, 6.3, 4.5]
    )
    md = reading_md({"Gemma 4 E2B": gemma, "Qwen3.5-2B": qwen}, {"local": {"recall@4": 0.419}})
    assert "RAG sozinho (B) supera" in md and "**não** levou D" in md and "Recall@4 = 0.42" in md
    assert "No **Qwen3.5-2B**, de B para D: melhorou abstenção correta 0.12 → 1.00, latência 14.30 → 4.50" in md
    # Gemma: quedas de < 0,2 na nota ficam no ruído (coerente com o aviso do próprio texto)
    assert "No **Gemma 4 E2B**, de B para D: melhorou latência 13.00 → 11.30." in md


def test_reading_reports_real_drops():
    x = colab([1.0, 3.0, 2.0, 2.5], fidelidade=[None, 4.5, None, 3.9])
    assert "piorou correção 3.00 → 2.50, fidelidade 4.50 → 3.90" in reading_md({"X": x}, None)


def test_md_table_and_br():
    assert md_table([["a", 0.5, None]], ["x", "y", "z"]).splitlines()[-1] == "| a | 0.50 | — |"
    assert br("4.35") == "4,35"


def test_summarize_abcd_false_abstention_only_with_oracle():
    rows = [
        # B: rag com oráculo no contexto e abstenção → falsa abstenção; sem oráculo não conta
        dict(kind="rag", config="B: base+RAG", abstained=True, oracle_in_ctx=True, cite_ok=False, has_cite=False),
        dict(kind="rag", config="B: base+RAG", abstained=True, oracle_in_ctx=False, cite_ok=False, has_cite=False),
        dict(kind="rag", config="B: base+RAG", abstained=False, oracle_in_ctx=True, cite_ok=True, has_cite=True),
        dict(
            kind="unanswerable", config="B: base+RAG", abstained=True, oracle_in_ctx=None, cite_ok=None, has_cite=False
        ),
        dict(kind="rag", config="A: base", abstained=False, oracle_in_ctx=None, cite_ok=None, has_cite=None),
    ]
    df = pd.DataFrame([{"family": "qwen", "model": "qwen_base", "id": str(i), "ptbr": True, "latency_s": 1.0,
                        "tokens_per_s": 30.0, "eval_count": 50, **r} for i, r in enumerate(rows)])  # fmt: skip
    s = {r["config"]: r for r in local_eval.summarize_abcd(df)}
    assert s["B: base+RAG"]["falsa_abstencao"] == 0.5
    assert s["B: base+RAG"]["abstencao_correta"] == 1.0
    assert s["B: base+RAG"]["citacao_correta"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["A: base"]["abstencao_correta"] is None and s["A: base"]["falsa_abstencao"] is None


def test_families_skip_disabled():
    from refauna_slm.config import get_settings

    f = local_eval.families(get_settings())
    assert f["gemma"] == {"ft": "gemma_ft"} and set(f["qwen"]) == {"base", "ft"}
