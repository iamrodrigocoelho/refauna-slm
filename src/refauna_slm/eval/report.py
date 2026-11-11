"""F6.4 — docs/avaliacao.md + PNGs para slide (fundo claro, fonte grande) a partir de data/outputs e data/local_eval."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from refauna_slm.config import ROOT, Settings

# Paleta categórica de referência (validada: CVD ΔE ≥ 9,2; verde-água < 3:1 → sempre com rótulo de valor)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = {"Gemma 4 E2B": "#2a78d6", "Qwen3.5-2B": "#eb6834", "Qwen3.5-2B (original)": "#1baf7a"}
CONFIGS = ["A: base", "B: base+RAG", "C: FT", "D: FT+RAG"]
SHORT = {"A: base": "A\nbase", "B: base+RAG": "B\nbase+RAG", "C: FT": "C\nFT", "D: FT+RAG": "D\nFT+RAG"}
FAMILY_LABEL = {"gemma": "Gemma 4 E2B", "qwen": "Qwen3.5-2B"}


def family_of_run(run: str) -> str:
    return "Qwen3.5-2B" if run.startswith("qwen") else "Gemma 4 E2B"


# ---------------------------------------------------------------- carga dos resultados


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def colab_runs(settings: Settings) -> dict[str, dict]:
    """{familia: {'run', 'summary': DataFrame por config, 'retrieval', 'train'}}"""
    out = {}
    for rep in sorted((settings.data_dir / "outputs").glob("*/eval_resumo.json")):
        j = json.loads(rep.read_text())
        out[family_of_run(rep.parent.name)] = {
            "run": rep.parent.name,
            "summary": pd.DataFrame(j["summary"]).set_index("config"),
            "retrieval": j["retrieval"],
            "train": load_json(rep.parent / "train_report.json") or {},
        }
    return out


# ---------------------------------------------------------------- gráficos


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 1,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 15,
        }
    )
    return plt


def br(text: str) -> str:
    """Separador decimal brasileiro nos rótulos dos slides."""
    return text.replace(".", ",")


def headroom(ax, top: float, pct: bool = False) -> None:
    """Folga acima da barra mais alta para o rótulo de valor não encostar no título."""
    if pct:
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    else:
        ax.set_ylim(0, top * 1.15 if top else 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%" if pct else br(f"{v:g}"))


def grouped_bars(ax, categories, series: dict[str, list], fmt="{:.2f}", ylim=None, pct=False):
    """Barras agrupadas finas, 2px de respiro entre barras, rótulo de valor em tinta de texto."""
    import numpy as np

    n = len(series)
    width = 0.78 / max(n, 1)
    x = np.arange(len(categories))
    for i, (name, values) in enumerate(series.items()):
        vals = [np.nan if v is None else (v * 100 if pct else v) for v in values]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=name, color=COLORS.get(name, INK_2),
                      edgecolor=SURFACE, linewidth=2)  # fmt: skip
        labels = ["" if np.isnan(v) else (f"{v:.0f}%" if pct else br(fmt.format(v))) for v in vals]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=13, color=INK)
    ax.set_xticks(x, categories)
    ax.tick_params(axis="x", length=0)
    top = max((v * (100 if pct else 1) for vs in series.values() for v in vs if v is not None), default=0)
    headroom(ax, max(top, ylim[1] if ylim and not pct else 0), pct)


def plot_colab(colab: dict, out: Path) -> list[Path]:
    plt = _style()
    paths = []
    fams = [f for f in ("Gemma 4 E2B", "Qwen3.5-2B") if f in colab]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    grouped_bars(ax, [SHORT[c] for c in CONFIGS],
                 {f: [colab[f]["summary"].correcao.get(c) for c in CONFIGS] for f in fams}, ylim=(0, 5))  # fmt: skip
    ax.set_ylim(0, 5.4)
    ax.set_title("Correção das respostas (juiz LLM, 1–5) — avaliação no Colab", loc="left")
    ax.set_ylabel("nota média")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    paths.append(_save(fig, out / "colab_correcao.png"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    rag = ["B: base+RAG", "D: FT+RAG"]
    for ax, (metric, title, pct, ylim) in zip(
        axes,
        [("fidelidade", "Fidelidade ao contexto (1–5)", False, (0, 5)),
         ("citacao_correta", "Citação correta", True, (0, 100)),
         ("abstencao_correta", "Abstenção correta", True, (0, 100))],
        strict=True,
    ):  # fmt: skip
        grouped_bars(ax, ["B\nbase+RAG", "D\nFT+RAG"],
                     {f: [colab[f]["summary"][metric].get(c) for c in rag] for f in fams}, pct=pct, ylim=ylim)  # fmt: skip
        ax.set_title(title, loc="left", fontsize=17)
    _bottom_legend(fig, axes[0])
    fig.suptitle("O que o fine-tuning mudou com RAG — Colab", x=0.01, ha="left", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    paths.append(_save(fig, out / "colab_comportamento.png"))
    plt.close("all")
    return paths


def plot_local_abcd(abcd: dict, out: Path) -> Path | None:
    if not abcd:
        return None
    plt = _style()
    df = pd.DataFrame(abcd["summary"])
    fams = [f for f in ("gemma", "qwen") if f in set(df.family)]

    def values(metric, cfgs):
        return {FAMILY_LABEL[f]: [_get(df, f, c, metric) for c in cfgs] for f in fams}

    fig, axes = plt.subplots(1, 4, figsize=(19, 6.2))
    panels = [
        ("citacao_correta", "Citação correta", ["B: base+RAG", "D: FT+RAG"], True, (0, 100)),
        ("abstencao_correta", "Abstenção correta", ["B: base+RAG", "D: FT+RAG"], True, (0, 100)),
        ("falsa_abstencao", "Falsa abstenção", ["B: base+RAG", "D: FT+RAG"], True, (0, 100)),
        ("latencia_s", "Latência média (s)", CONFIGS, False, None),
    ]
    for ax, (metric, title, cfgs, pct, ylim) in zip(axes, panels, strict=True):
        grouped_bars(ax, [SHORT[c] for c in cfgs], values(metric, cfgs), pct=pct, ylim=ylim, fmt="{:.1f}")
        ax.set_title(title, loc="left", fontsize=17)
    _bottom_legend(fig, axes[0])
    fig.suptitle(f"A/B/C/D no MacBook Air M1 · Ollama · {abcd.get('n_items', '?')} perguntas, métricas automáticas",
                 x=0.01, ha="left", fontsize=20, fontweight="bold")  # fmt: skip
    if "gemma" in fams and _get(df, "gemma", "B: base+RAG", "latencia_s") is None:
        fig.text(0.01, 0.885, "Gemma: só C e D — o Gemma 4 E2B original não cabe no M1 de 8 GB (A/B do Gemma: ver Colab).",
                 ha="left", fontsize=14, color=INK_2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    path = _save(fig, out / "local_abcd.png")
    plt.close("all")
    return path


def plot_bench(bench: dict, out: Path) -> Path | None:
    if not bench:
        return None
    plt = _style()
    names = {"gemma_ft": "Gemma 4 E2B", "qwen_ft": "Qwen3.5-2B", "qwen_base": "Qwen3.5-2B (original)"}
    ms = [m for m in bench["models"] if m["model"] in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    for ax, (metric, title, fmt) in zip(
        axes,
        [("tokens_s", "Velocidade (tokens/s)", "{:.0f}"), ("latencia_p50_s", "Latência p50 (s)", "{:.1f}"),
         ("gguf_gb", "Tamanho do modelo (GB)", "{:.1f}")],
        strict=True,
    ):  # fmt: skip
        for i, m in enumerate(ms):
            name = names[m["model"]]
            bar = ax.bar([i], [m[metric]], 0.6, color=COLORS[name], edgecolor=SURFACE, linewidth=2,
                         label=name + ("" if m["model"].endswith("base") else " · Refauna"))  # fmt: skip
            ax.bar_label(bar, labels=[br(fmt.format(m[metric]))], padding=3, fontsize=14, color=INK)
        ax.set_xticks([])
        headroom(ax, max(m[metric] for m in ms))
        ax.set_title(title, loc="left", fontsize=17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=15)
    fig.suptitle("Desempenho no MacBook Air M1 (8 GB) · GGUF via Ollama, RAG top-4", x=0.01, ha="left",
                 fontsize=20, fontweight="bold")  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = _save(fig, out / "bench_m1.png")
    plt.close("all")
    return path


def _bottom_legend(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(labels), fontsize=15)


def _get(df, family, config, metric):
    r = df[(df.family == family) & (df.config == config)]
    if r.empty or metric not in r:
        return None
    v = r[metric].iloc[0]
    return None if pd.isna(v) else float(v)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


# ---------------------------------------------------------------- markdown


def md_table(rows: list[list], header: list[str]) -> str:
    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v * 100:.0f}%"


def hypothesis_md(colab: dict) -> str:
    """Hipótese D > B > C > A, discutida com os números (resultado negativo também é conteúdo)."""
    lines = []
    for fam, r in colab.items():
        s = r["summary"]
        cor = {c: s.correcao.get(c) for c in CONFIGS}
        order = " > ".join(c[0] for c in sorted(CONFIGS, key=lambda c: -cor[c]))
        lines.append(f"- **{fam}** — ordem observada em correção: **{order}** "
                     f"(A {cor['A: base']:.2f} · B {cor['B: base+RAG']:.2f} · C {cor['C: FT']:.2f} · D {cor['D: FT+RAG']:.2f}). "
                     f"{'Confirma' if order == 'D > B > C > A' else 'Não confirma'} a hipótese D > B > C > A.")  # fmt: skip
        d, b = s.loc["D: FT+RAG"], s.loc["B: base+RAG"]
        diffs = []
        for metric, label in [("fidelidade", "fidelidade"), ("citacao_correta", "citação correta"),
                              ("abstencao_correta", "abstenção correta"), ("latencia_s", "latência (s)")]:  # fmt: skip
            if pd.notna(d.get(metric)) and pd.notna(b.get(metric)):
                diffs.append(f"{label} {b[metric]:.2f} → {d[metric]:.2f}")
        lines.append(f"  - B → D (efeito do fine-tuning com RAG): {'; '.join(diffs)}.")
    return "\n".join(lines)


def reading_md(colab: dict, retr: dict | None) -> str:
    """Leitura dos números: o que o fine-tuning mudou de B para D, por modelo (calculado, não fixo)."""
    better_high = {"correcao": "correção", "fidelidade": "fidelidade", "citacao_correta": "citação correta",
                   "abstencao_correta": "abstenção correta"}  # fmt: skip
    better_low = {"falsa_abstencao": "falsa abstenção", "latencia_s": "latência"}
    d_beats_b = [
        f for f, r in colab.items() if r["summary"].correcao["D: FT+RAG"] > r["summary"].correcao["B: base+RAG"] + 0.2
    ]
    b_beats_c = all(r["summary"].correcao["B: base+RAG"] > r["summary"].correcao["C: FT"] for r in colab.values())
    out = ["**Leitura dos números:**"]
    if b_beats_c:
        out.append(
            "em correção, o RAG sozinho (B) supera o fine-tuning sozinho (C) em todos os modelos — os fatos vêm da busca."
        )
    if not d_beats_b:
        rec = f" (Recall@4 = {retr['local']['recall@4']:.2f})" if retr else ""
        out.append(f"O fine-tuning **não** levou D claramente acima de B em correção (diferença < 0,2 ou negativa): "
                   f"a correção com RAG fica limitada pela recuperação{rec}.")  # fmt: skip
    for fam, r in colab.items():
        b, d = r["summary"].loc["B: base+RAG"], r["summary"].loc["D: FT+RAG"]
        up, down = [], []
        for metrics, higher_is_better in ((better_high, True), (better_low, False)):
            for m, label in metrics.items():
                if m not in b or pd.isna(b[m]) or pd.isna(d[m]):
                    continue
                delta = d[m] - b[m]
                tol = 0.2 if m in ("correcao", "fidelidade") else (1.0 if m == "latencia_s" else 0.05)
                if abs(delta) > tol:
                    (up if (delta > 0) == higher_is_better else down).append(f"{label} {b[m]:.2f} → {d[m]:.2f}")
        out.append(f"No **{fam}**, de B para D: melhorou {', '.join(up) or 'nada além do ruído'}"
                   f"{'; piorou ' + ', '.join(down) if down else ''}.")  # fmt: skip
    out.append(
        "Diferenças pequenas (< 0,2 nas notas, < 5 pontos nas taxas) estão dentro do ruído de amostras de ~60 perguntas."
    )
    return " ".join(out)


def build_report(settings: Settings, out_md: Path | None = None) -> tuple[Path, list[Path]]:
    out_md = out_md or ROOT / "docs" / "avaliacao.md"
    img = out_md.parent / "img"
    d = settings.local_eval_dir
    colab = colab_runs(settings)
    abcd, bench = load_json(d / "abcd_resumo.json"), load_json(d / "bench.json")
    retr, ram = load_json(d / "retrieval.json"), load_json(d / "ram_demo.json")
    parity = [json.loads(p.read_text()) for p in sorted(d.glob("parity_*.json"))]

    pngs = plot_colab(colab, img) if colab else []
    pngs += [p for p in (plot_local_abcd(abcd, img), plot_bench(bench, img)) if p]
    rel = {p.name: f"img/{p.name}" for p in pngs}

    md = [f"# Avaliação — Refauna SLM\n\n_Gerado por `refauna report` em {date.today():%d/%m/%Y}. "
          "Não edite à mão: rode `refauna eval`, `refauna bench` e `refauna report` de novo._\n"]  # fmt: skip

    # Resumo
    md.append("## Resumo\n")
    bullets = []
    for fam, r in colab.items():
        s = r["summary"]
        bullets.append(f"**{fam}** (Colab, juiz LLM): correção A {s.correcao['A: base']:.2f} → D {s.correcao['D: FT+RAG']:.2f}; "
                       f"o RAG é o maior salto (A → B {s.correcao['A: base']:.2f} → {s.correcao['B: base+RAG']:.2f}); "
                       f"abstenção correta B {pct(s.abstencao_correta.get('B: base+RAG'))} → D {pct(s.abstencao_correta.get('D: FT+RAG'))}.")  # fmt: skip
    if retr:
        bullets.append(f"Recuperação híbrida idêntica no Mac e no Colab: Recall@4 = {retr['local']['recall@4']:.3f} "
                       f"(gargalo do sistema: em {1 - retr['local']['recall@4']:.0%} das perguntas o trecho certo não chega ao top-4).")  # fmt: skip
    if bench:
        fast = max(bench["models"], key=lambda m: m["tokens_s"])
        bullets.append(f"No M1, o mais rápido é **{fast['label']}**: {fast['tokens_s']:.0f} tokens/s, latência p50 {fast['latencia_p50_s']:.1f} s.")  # fmt: skip
    if ram:
        bullets.append(f"Demo com 10 perguntas seguidas ({' ↔ '.join(ram['models'])}): pico app + Ollama = "
                       f"{ram['peak']['total_gb']:.2f} GB ({'dentro' if ram['peak']['total_gb'] <= 6 else 'acima'} do limite de 6 GB).")  # fmt: skip
    md.append("\n".join(f"- {b}" for b in bullets) + "\n")

    # Recuperação
    if colab or retr:
        md.append("## Recuperação (EmbeddingGemma + BM25 + RRF, top-4)\n")
        rr = next(iter(colab.values()))["retrieval"] if colab else {}
        rows = [[m, rr.get(m), (retr or {}).get("local", {}).get(m)] for m in ("recall@4", "mrr@20", "doc_recall@4")]
        md.append(md_table(rows, ["métrica", "Colab", "Mac (local)"]))
        if retr:
            md.append(f"\nBusca média no M1: **{retr['search_ms']:.0f} ms** por pergunta (aceite: < 300 ms). "
                      f"n = {rr.get('n', '?')} perguntas `rag` + `unanswerable` do teste.\n")  # fmt: skip

    # Colab
    if colab:
        md.append("## A/B/C/D no Colab (juiz LLM)\n")
        md.append("A = base · B = base + RAG · C = fine-tunado · D = fine-tunado + RAG. Correção e fidelidade de 1 a 5 "
                  "(juiz `claude-sonnet-5-5`); citação, abstenção e PT-BR de 0 a 1; geração com temperatura 0.\n")  # fmt: skip
        if "colab_correcao.png" in rel:
            md.append(f"![Correção A/B/C/D]({rel['colab_correcao.png']})\n")
        cols = [
            "correcao",
            "fidelidade",
            "citacao_correta",
            "abstencao_correta",
            "falsa_abstencao",
            "ptbr",
            "latencia_s",
        ]
        for fam, r in colab.items():
            t = r["train"]
            md.append(f"### {fam} · `{r['run']}`\n")
            if t:
                md.append(f"{'QLoRA 4-bit' if t.get('load_in_4bit') else 'LoRA 16-bit'}, r={t.get('lora_r')}, "
                          f"{t.get('epochs')} épocas · {t.get('train_examples')} exemplos · {t.get('minutes')} min numa "
                          f"{t.get('gpu')} · pico {t.get('peak_vram_gb')} GB de VRAM.\n")  # fmt: skip
            s = r["summary"]
            md.append(md_table([[c, *(s[m].get(c) if m in s else None for m in cols)] for c in CONFIGS],
                               ["config", *(c.replace("_", " ") for c in cols)]) + "\n")  # fmt: skip
        if "colab_comportamento.png" in rel:
            md.append(f"![Fidelidade, citação e abstenção]({rel['colab_comportamento.png']})\n")

        md.append("## A hipótese D > B > C > A\n")
        md.append(hypothesis_md(colab) + "\n")
        md.append(reading_md(colab, retr) + "\n")

        md.append("## Gemma 4 E2B × Qwen3.5-2B (mesmos dados, prompt e juiz)\n")
        rows = []
        for fam, r in colab.items():
            s, t = r["summary"].loc["D: FT+RAG"], r["train"]
            rows.append([fam, "QLoRA 4-bit" if t.get("load_in_4bit") else "LoRA 16-bit", t.get("minutes"),
                         t.get("peak_vram_gb"), s.get("correcao"), s.get("fidelidade"), s.get("citacao_correta"),
                         s.get("abstencao_correta"), s.get("latencia_s")])  # fmt: skip
        md.append(md_table(rows, ["modelo", "método", "min de treino", "VRAM pico (GB)", "correção D", "fidelidade D",
                                  "citação D", "abstenção D", "latência D (s)"]))  # fmt: skip
        md.append("\nA `eval_loss` **não** é comparável entre os dois (tokenizers diferentes).\n")

    # Local A/B/C/D
    if abcd:
        md.append("## A/B/C/D no MacBook Air M1 (Ollama, métricas automáticas)\n")
        md.append(f"{abcd['n_items']} perguntas do `sft/test.jsonl` (60% `rag`, 20% `closed_book`, 20% `unanswerable`), "
                  "temperatura 0, GGUF Q4_K_M (Qwen original: mxfp8). Sem juiz: citação correta = cita um trecho do "
                  "documento de origem; abstenção correta = se abstém nas `unanswerable` (contexto fixo sem a resposta); "
                  "falsa abstenção = se abstém numa pergunta respondível **com o trecho certo no contexto**. "
                  "**Gemma original indisponível no M1** (a tag `gemma4:e2b` é nvfp4 de 7,5 GB): só C e D para o Gemma.\n")  # fmt: skip
        if "local_abcd.png" in rel:
            md.append(f"![A/B/C/D local]({rel['local_abcd.png']})\n")
        cols = ["n", "ptbr", "com_citacao", "citacao_correta", "abstencao_correta", "falsa_abstencao", "latencia_s",
                "tokens_s", "tokens_resposta"] + (["correcao", "fidelidade"] if abcd.get("judge") else [])  # fmt: skip
        rows = [
            [FAMILY_LABEL.get(r["family"], r["family"]), r["config"], *(r.get(c) for c in cols)]
            for r in abcd["summary"]
        ]
        md.append(md_table(rows, ["modelo", "config", *(c.replace("_", " ") for c in cols)]) + "\n")
        rag_rows = [r for r in abcd["summary"] if r.get("n_sem_resposta") is not None]
        if rag_rows:
            md.append(f"Bases das taxas com RAG: abstenção correta sobre {rag_rows[0]['n_sem_resposta']} perguntas sem resposta; "
                      f"falsa abstenção sobre {rag_rows[0]['n_trecho_certo']} perguntas `rag` em que o trecho certo foi recuperado "
                      "(as mesmas para todos os modelos). Uma resposta muda a taxa em 7–13 pontos.\n")  # fmt: skip

    # Paridade
    if parity:
        md.append("## Paridade Colab × Ollama (F6.1)\n")
        md.append("10 perguntas fixas, FT + RAG, temperatura 0, mesmas métricas automáticas nos dois lados. "
                  "Divergência grande de formato indicaria template de chat errado no GGUF.\n")  # fmt: skip
        rows = []
        for p in parity:
            rows.append([p["model"], f"{pct(p['ptbr']['local'])} / {pct(p['ptbr']['colab'])}",
                         f"{pct(p['com_citacao']['local'])} / {pct(p['com_citacao']['colab'])}",
                         f"{pct(p['citacao_correta_rag']['local'])} / {pct(p['citacao_correta_rag']['colab'])}",
                         f"{pct(p['falsa_abstencao']['local'])} / {pct(p['falsa_abstencao']['colab'])}",
                         f"{p['itens_com_divergencia']}/{p['n']}"])  # fmt: skip
        md.append(md_table(rows, ["modelo", "PT-BR (Mac / Colab)", "com citação", "citação correta",
                                  "abstenção em respondíveis", "itens divergentes"]))  # fmt: skip
        fmt_ok = all(abs((p["ptbr"]["local"] or 0) - (p["ptbr"]["colab"] or 0)) <= 0.1
                     and abs((p["com_citacao"]["local"] or 0) - (p["com_citacao"]["colab"] or 0)) <= 0.2 for p in parity)  # fmt: skip
        md.append("\nIdioma e presença de citações batem com o Colab: os templates de chat dos GGUF estão corretos.\n"
                  if fmt_ok else "\n⚠️ Divergência de formato (idioma ou citações): verifique o template de chat do GGUF (PRD F8.4).\n")  # fmt: skip

    # Benchmark
    if bench:
        md.append("## Benchmark no MacBook Air M1 (8 GB)\n")
        md.append(f"{bench['warm_runs']} respostas a quente por modelo (perguntas da demo, RAG top-4, temperatura "
                  f"{bench['temperature']}), depois de uma carga a frio com pergunta separada. "
                  "**RAM Ollama** = pico de `phys_footprint` (macOS): quando o Ollama mapeia os pesos do disco (mmap), "
                  "eles não entram nessa medida e o valor fica abaixo do tamanho do GGUF — por isso o gráfico mostra o "
                  "tamanho do modelo, e o critério de RAM é verificado de ponta a ponta na seção seguinte.\n")  # fmt: skip
        if "bench_m1.png" in rel:
            md.append(f"![Benchmark M1]({rel['bench_m1.png']})\n")
        cols = ["gguf_gb", "carga_fria_s", "latencia_p50_s", "latencia_p95_s", "tokens_s", "prompt_tokens_s",
                "tokens_resposta", "ram_ollama_pico_gb"]  # fmt: skip
        md.append(md_table([[m["label"], *(m.get(c) for c in cols)] for m in bench["models"]],
                           ["modelo", *(c.replace("_", " ") for c in cols)]))  # fmt: skip
        md.append(f"\nBusca híbrida: {bench['busca_ms']:.0f} ms por pergunta · app com EmbeddingGemma "
                  f"({bench['embedder_device']}): {bench['app_embedder_gb']:.2f} GB.\n")  # fmt: skip

    if ram:
        p = ram["peak"]
        md.append("## RAM da demo (10 perguntas seguidas no app)\n")
        md.append(md_table([[" ↔ ".join(ram["models"]), p["app_gb"], p["ollama_gb"], p["total_gb"],
                             "✅" if p["total_gb"] <= 6 else "❌", "✅" if ram.get("single_model_loaded") else "❌"]],
                           ["modelos alternados", "app (GB)", "Ollama (GB)", "pico juntos (GB)", "≤ 6 GB",
                            "um modelo por vez"]))  # fmt: skip
        if "gemma_ft" in ram["models"]:
            md.append("\nO Gemma FT roda como variante **só-texto** (sem o projetor de visão/áudio de 0,99 GB do GGUF do HF); "
                      "com o projetor, o pico era 6,55 GB.\n")  # fmt: skip

    md.append("## Limitações\n")
    md.append("- Amostras pequenas (Colab n ≈ 60 + 60 de abstenção; local n = 40): diferenças de poucos pontos são ruído.\n"
              "- Correção e fidelidade vêm de um juiz LLM; as métricas locais são automáticas (regex de citação e abstenção).\n"
              "- O Gemma original não roda no M1 de 8 GB; o Qwen original local está em mxfp8, não Q4_K_M como os fine-tunados.\n"
              "- Perguntas sintéticas (geradas a partir dos trechos), com revisão humana amostral; sem gold set ainda.\n")  # fmt: skip

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    return out_md, pngs
