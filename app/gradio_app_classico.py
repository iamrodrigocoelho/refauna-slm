"""App Gradio CLÁSSICO (antes do redesenho): plano B da demo. Uso: `uv run python app/gradio_app_classico.py`."""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # nenhuma chamada externa durante a demo

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from refauna_slm.config import Settings, get_settings  # noqa: E402
from refauna_slm.rag.pipeline import Answer, RagPipeline  # noqa: E402
from refauna_slm.sync import read_jsonl  # noqa: E402

MAX_COMPARE = 4
EXCERPT_CHARS = 320
CSS = """
.fonte {font-size: 0.92em}
.card {min-height: 220px}
footer {display: none !important}
"""


# ---------------------------------------------------------------- formatação


def model_choices(s: Settings) -> list[tuple[str, str]]:
    return [(m.label, k) for k, m in s.models.items() if m.enabled]


def preset_choices(s: Settings) -> list[tuple[str, str]]:
    out = []
    for k, m in s.models.items():
        if m.enabled:
            out += [(f"{m.label} · sem RAG", f"{k}|norag"), (f"{m.label} + RAG", f"{k}|rag")]
    return out


def doi_link(doi: str) -> str:
    return f" · [doi:{doi}](https://doi.org/{doi})" if doi else ""


def sources_md(ans: Answer) -> str:
    """Painel de fontes: trechos citados [n] (e, à parte, os recuperados e não citados)."""
    if not ans.use_rag:
        return "_Sem RAG: o modelo respondeu só com o que aprendeu no treino, sem fontes._"
    if not ans.contexts:
        return "_Nenhum trecho recuperado._"
    cited = {c.n for c in ans.citations}
    lines = []
    for c in ans.citations:
        year = c.year or "s.d."
        where = c.pages if c.pages == "Ficha do catálogo" else f"{c.section[:60]} · {c.pages}"
        excerpt = c.text.strip().replace("\n", " ")
        excerpt = excerpt if len(excerpt) <= EXCERPT_CHARS else excerpt[:EXCERPT_CHARS].rstrip() + "…"
        lines.append(f"**[{c.n}] {c.title}** ({year}) · {where}{doi_link(c.doi)}\n\n> {excerpt}")
    if not lines:
        lines.append("_A resposta não citou nenhum trecho._")
    others = [(i, ctx) for i, ctx in enumerate(ans.contexts, 1) if i not in cited]
    if others:
        lines.append("<details><summary>Trechos recuperados e não citados</summary>\n")
        for i, ctx in others:
            from refauna_slm.contract import short_ref
            from refauna_slm.rag.pipeline import pages

            lines.append(f"- [{i}] {short_ref(ctx)} · {pages(ctx)}")
        lines.append("</details>")
    if ans.invalid_citations:
        lines.append(f"⚠️ A resposta cita trechos inexistentes: {ans.invalid_citations}")
    return "\n\n".join(lines)


def stats_md(ans: Answer) -> str:
    g = ans.generation
    tps = f"{ans.tokens_per_s:.1f} tok/s" if ans.tokens_per_s else "— tok/s"
    rag = f"busca {ans.search_s * 1000:.0f} ms · " if ans.use_rag and ans.search_s > 0.001 else ""
    load = f" · carga do modelo {g.load_duration_s:.1f}s" if g.load_duration_s >= 0.5 else ""
    return f"⏱ {ans.latency_s:.1f}s · {rag}{g.eval_count} tokens · {tps}{load}"


# ---------------------------------------------------------------- resultados


def md_table(df: pd.DataFrame) -> str:
    """DataFrame -> tabela Markdown (índice na 1ª coluna), números com 2 casas."""

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    cols = [df.index.name or "", *(str(c).replace("_", " ") for c in df.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += [
        "| " + " | ".join([str(i), *(cell(v) for v in row)]) + " |"
        for i, row in zip(df.index, df.itertuples(index=False), strict=True)
    ]
    return "\n".join(lines)


def results_md(s: Settings) -> str:
    out = s.data_dir / "outputs"
    parts = ["## Avaliação no Colab (A/B/C/D, juiz LLM)",
             "A = base · B = base + RAG · C = fine-tunado · D = fine-tunado + RAG. "
             "Notas de 1 a 5; citação/abstenção/PT-BR de 0 a 1."]  # fmt: skip
    for rep in sorted(out.glob("*/eval_resumo.json")):
        r = json.loads(rep.read_text())
        tr_path = rep.parent / "train_report.json"
        tr = json.loads(tr_path.read_text()) if tr_path.exists() else {}
        df = pd.DataFrame(r["summary"]).set_index("config")
        cols = [c for c in ["correcao", "fidelidade", "citacao_correta", "abstencao_correta", "falsa_abstencao", "ptbr", "latencia_s"] if c in df]  # fmt: skip
        head = f"### {rep.parent.name}"
        if tr:
            head += f"\n{tr.get('gpu', '')} · {tr.get('minutes', '?')} min de treino · {tr.get('train_examples', '?')} exemplos · pico {tr.get('peak_vram_gb', '?')} GB de VRAM"  # fmt: skip
        parts += [head, md_table(df[cols])]
    rr = next(iter(json.loads(p.read_text())["retrieval"] for p in out.glob("*/eval_resumo.json")), None)
    if rr:
        parts.append(f"**Recuperação híbrida** (n={rr['n']}): Recall@4 = {rr['recall@4']:.2f} · "
                     f"MRR@20 = {rr['mrr@20']:.2f} · Recall@4 por documento = {rr['doc_recall@4']:.2f}")  # fmt: skip
    return "\n\n".join(parts)


def local_md(s: Settings) -> str:
    d = s.local_eval_dir
    parts = ["## Avaliação local no MacBook Air M1 (Ollama, GGUF Q4_K_M)"]
    rp = d / "retrieval.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        loc, col = r["local"], r.get("colab", {})
        parts.append(f"**Recuperação:** Recall@4 local = {loc['recall@4']:.3f} (Colab {col.get('recall@4', float('nan')):.3f}) · "
                     f"busca média {r['search_ms']:.0f} ms")  # fmt: skip
    rows = []
    for p in sorted(d.glob("parity_*.json")):
        j = json.loads(p.read_text())
        rows.append({"modelo": j["model"], "PT-BR": j["ptbr"]["local"], "com citação": j["com_citacao"]["local"],
                     "citação correta": j["citacao_correta_rag"]["local"], "abstenção correta": j["abstencao_correta"]["local"],
                     "falsa abstenção": j["falsa_abstencao"]["local"], "latência (s)": j["latencia_s_media"],
                     "tok/s": j["tokens_por_s_medio"], "divergências": f"{j['itens_com_divergencia']}/{j['n']}"})  # fmt: skip
    if rows:
        parts += ["**Paridade Colab × Ollama** (10 perguntas, FT + RAG, temperatura 0):",
                  md_table(pd.DataFrame(rows).set_index("modelo"))]  # fmt: skip
    ram = d / "ram_demo.json"
    if ram.exists():
        r = json.loads(ram.read_text())
        parts.append(f"**RAM (10 perguntas seguidas):** pico app + Ollama = {r['peak']['total_gb']:.2f} GB · "
                     f"swap +{r['peak']['swap_used_gb'] - r['peak']['swap_start_gb']:.2f} GB")  # fmt: skip
    if len(parts) == 1:
        parts.append("_Ainda sem resultados locais. Rode `refauna eval --retrieval --parity`._")
    return "\n\n".join(parts)


def sources_list_md(s: Settings) -> str:
    """Atribuição (PRD §8.4): os artigos usados, com DOI."""
    path = s.data_dir / "docs.jsonl"
    if not path.exists():
        return ""
    docs = sorted(read_jsonl(path), key=lambda d: (d.get("year") or 0, d.get("title", "")))
    items = [
        f"- {d.get('authors') or ''} ({d.get('year') or 's.d.'}). **{d['title']}**{doi_link(d.get('doi', ''))}"
        for d in docs
    ]
    return "## Fontes (atribuição)\n\nDocumentos com texto no índice do RAG, produção científica do [Refauna](https://refauna.org.br):\n\n" + "\n".join(items)  # fmt: skip


# ---------------------------------------------------------------- app


def example_buttons(examples: list[str], box: gr.Textbox, io: dict) -> None:
    """Um clique preenche a pergunta e já envia (mais rápido na demo do que preencher + Enter)."""
    gr.Markdown("**Perguntas de exemplo** · um clique pergunta", elem_classes="fonte")
    with gr.Row():
        for q in examples:
            btn = gr.Button(q, size="sm", variant="secondary", min_width=200)
            btn.click(lambda q=q: q, None, box, api_name=False, queue=False).then(**io, api_name=False)


def build_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> gr.Blocks:
    s = settings or get_settings()
    pipe = pipeline or RagPipeline(s)
    pipe.retriever.warmup()  # carrega EmbeddingGemma + índice antes da primeira pergunta
    choices = model_choices(s)
    keys = [k for _, k in choices]
    default_model = (
        s.demo.default_model if s.demo.default_model in keys else next((k for k in keys if k.endswith("_ft")), keys[0])
    )
    presets = dict((k, label) for label, k in preset_choices(s))

    def chat(question, history, model_key, use_rag):
        question = (question or "").strip()
        if not question:
            gr.Warning("Escreva uma pergunta.")
            yield history, "", gr.skip(), gr.skip()
            return
        history = [*history, {"role": "user", "content": question}, {"role": "assistant", "content": "…"}]
        yield history, "", "_buscando trechos e carregando o modelo…_", ""
        ans = None
        for _, ans in pipe.stream(question, model_key, use_rag=use_rag):  # noqa: B007
            history[-1]["content"] = ans.answer or "…"
            yield history, "", "_gerando…_", stats_md(ans)
        yield history, "", sources_md(ans), stats_md(ans)

    def compare(question, selected):
        question = (question or "").strip()
        if not question or not selected:
            gr.Warning("Escreva uma pergunta." if not question else "Escolha ao menos uma configuração.")
            yield tuple(gr.skip() for _ in range(MAX_COMPARE + 1))
            return
        if len(selected) > MAX_COMPARE:
            gr.Warning(f"Máximo de {MAX_COMPARE}: usando as {MAX_COMPARE} primeiras.")
        selected = [k for k in presets if k in selected][:MAX_COMPARE]
        cards = [""] * MAX_COMPARE
        for i, key in enumerate(selected):
            cards[i] = f"### {presets[key]}\n\n_aguardando…_"
        # A mesma busca para todas as configurações com RAG: comparação justa e um modelo por vez
        t0 = time.time()
        ctx = pipe.retriever.hybrid_search(question) if any(k.endswith("|rag") for k in selected) else []
        search_ms = (time.time() - t0) * 1000
        ctx_md = ""
        yield (*cards, ctx_md)
        for i, key in enumerate(selected):
            model_key, mode = key.split("|")
            use_rag = mode == "rag"
            ans = None
            for _, ans in pipe.stream(question, model_key, use_rag=use_rag, contexts=ctx if use_rag else None):  # noqa: B007
                cards[i] = f"### {presets[key]}\n\n{ans.answer or '…'}\n\n<sub>{stats_md(ans)}</sub>"
                yield (*cards, ctx_md)
            cites = ", ".join(f"[{c.n}] {c.ref}" for c in ans.citations) if ans.citations else ("sem citações" if use_rag else "sem RAG")  # fmt: skip
            cards[i] += f"\n\n<sub>📚 {cites}</sub>"
            if ctx and not ctx_md:
                from refauna_slm.contract import format_context

                ctx_md = f"Trechos enviados às configurações com RAG (os mesmos para todas · busca {search_ms:.0f} ms):\n\n```text\n" + format_context(ctx) + "\n```"  # fmt: skip
            yield (*cards, ctx_md)

    with gr.Blocks(title="Refauna SLM", fill_width=True) as demo:
        gr.Markdown(
            "# 🌳 Refauna SLM\n"
            "Pequenos modelos (Gemma 4 E2B · Qwen3.5-2B) com **fine-tuning + RAG** sobre a produção científica do "
            "[Refauna](https://refauna.org.br). Rodando **offline** num MacBook Air M1 de 8 GB via Ollama."
        )
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=380, label="Assistente Refauna", buttons=["copy"])
                    question = gr.Textbox(placeholder="Pergunte sobre refaunação, dispersão de sementes, espécies…",
                                          show_label=False, submit_btn="Perguntar")  # fmt: skip
                    stats = gr.Markdown()
                with gr.Column(scale=2):
                    model = gr.Dropdown(choices, value=default_model, label="Modelo")
                    use_rag = gr.Checkbox(value=True, label="Usar RAG (trechos dos artigos do Refauna)")
                    gr.Markdown("Cada pergunta é independente: o modelo não vê as anteriores.", elem_classes="fonte")
                    sources = gr.Markdown("_As fontes citadas aparecem aqui._", elem_classes="fonte")
                    clear = gr.Button("Limpar conversa", size="sm")
            chat_io = dict(fn=chat, inputs=[question, chatbot, model, use_rag], outputs=[chatbot, question, sources, stats],
                           concurrency_id="ollama", concurrency_limit=1)  # fmt: skip
            question.submit(**chat_io, api_name="chat")
            example_buttons(s.demo.examples, question, chat_io)
            clear.click(lambda: ([], "_As fontes citadas aparecem aqui._", ""), None, [chatbot, sources, stats],
                        api_name=False)  # fmt: skip

        with gr.Tab("⚖️ Comparar"):
            gr.Markdown(f"A mesma pergunta respondida **em sequência** por até {MAX_COMPARE} configurações "
                        "(um modelo carregado por vez).")  # fmt: skip
            with gr.Row():
                cq = gr.Textbox(placeholder="Pergunta para comparar…", show_label=False, scale=4)
                run = gr.Button("Comparar", variant="primary", scale=1)
            sel = gr.CheckboxGroup([(v, k) for k, v in presets.items()],
                                   value=[k for k in s.demo.compare_defaults if k in presets][:MAX_COMPARE],
                                   label=f"Configurações (até {MAX_COMPARE})")  # fmt: skip
            with gr.Row(equal_height=True):
                cards = [gr.Markdown(elem_classes="card") for _ in range(MAX_COMPARE)]
            with gr.Accordion("Trechos recuperados", open=False):
                ctx_box = gr.Markdown()
            compare_io = dict(fn=compare, inputs=[cq, sel], outputs=[*cards, ctx_box], concurrency_id="ollama",
                              concurrency_limit=1)  # fmt: skip
            gr.on([run.click, cq.submit], **compare_io, api_name="compare")
            example_buttons(s.demo.examples, cq, compare_io)

        with gr.Tab("📊 Resultados"):
            img = s.data_dir / "outputs" / "comparacao_modelos.png"
            if img.exists():
                gr.Image(str(img), label="Gemma 4 E2B × Qwen3.5-2B (Colab)", show_label=True, interactive=False,
                         buttons=[])  # fmt: skip
            gr.Markdown(results_md(s))
            local = gr.Markdown(local_md(s))
            gr.Button("Atualizar resultados locais", size="sm").click(lambda: local_md(s), None, local, api_name=False)
            gr.Markdown(sources_list_md(s), elem_classes="fonte")
    return demo


def main(host: str = "127.0.0.1", port: int | None = None, inbrowser: bool = True) -> None:
    s = get_settings()
    demo = build_app(s)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port or s.demo.server_port,
        inbrowser=inbrowser,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
        css=CSS,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
