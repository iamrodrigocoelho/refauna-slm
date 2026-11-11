"""F5 — app Gradio do Refauna SLM (Assistente Refauna): Chat, Comparar e Resultados. Roda 100% offline.

Uso: `uv run refauna serve` (ou `uv run python app/gradio_app.py`).
Só interface: busca, contrato de prompt e Ollama vêm de `refauna_slm.rag.pipeline` sem alteração.
CSS/JS próprios em app/assets/ (sem CDN, sem fontes da internet: fontes do sistema).
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # nenhuma chamada externa durante a demo

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from refauna_slm.config import Settings, get_settings  # noqa: E402
from refauna_slm.contract import format_context, short_ref  # noqa: E402
from refauna_slm.rag.pipeline import Answer, RagPipeline, pages  # noqa: E402
from refauna_slm.sync import read_jsonl  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
MAX_COMPARE = 4
EXCERPT_CHARS = 240
TITLE_CHARS = 60
CITE_RE = re.compile(r"\[(\d+)\]")
SYSTEM_FONTS = ["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]  # fmt: skip
MONO_FONTS = ["ui-monospace", "SF Mono", "Menlo", "Consolas", "monospace"]

# Uma linha por modelo no seletor (texto de interface; repositórios/tags continuam só no settings.yaml).
MODEL_BLURBS = {
    "qwen_ft": "rápido, ajustado para o Refauna",
    "gemma_ft": "ajustado para o Refauna, mais lento",
    "qwen_base": "original, sem ajuste — para comparar",
    "gemma_base": "original, sem ajuste",
}
GREETING = "Por onde começamos?"
PLACEHOLDER = "Pergunte ao Assistente Refauna"
DISCLAIMER = "O Assistente Refauna pode errar. Confira as fontes citadas."


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


def num(x: float, nd: int = 1) -> str:
    """Número no formato PT-BR (vírgula decimal)."""
    return f"{x:.{nd}f}".replace(".", ",")


def esc(x) -> str:
    return html.escape(str(x or ""), quote=True)


def link_citations(text: str, n_ctx: int) -> str:
    """[n] válidos viram pílulas clicáveis (o número é o texto) que abrem a fonte n; os inválidos ficam como texto."""

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_ctx:
            return f'<a class="rf-cite" href="#" title="Ver fonte {n}">{n}</a>'
        return m.group(0)

    return CITE_RE.sub(repl, text) if n_ctx else text


def sources_html(ans: Answer) -> str:
    """Bloco recolhível 'Fontes': os 4 trechos recuperados, numerados como no prompt; os citados em destaque.

    Fica numa linha só (sem linhas em branco) para o Markdown do Chatbot tratá-lo como um bloco HTML.
    """
    if not ans.use_rag:
        # "treino" sozinho confunde com o fine-tuning: deixar claro o que cada modelo sabe sem os artigos
        if ans.model.endswith("_ft"):
            note = ("Sem consulta aos documentos: modelo ajustado para o Refauna, mas sem os trechos dos artigos. "
                    "Pode inventar — confira.")  # fmt: skip
        else:
            note = ("Sem consulta aos documentos e sem o ajuste do Refauna: conhecimento geral do modelo original "
                    "(pré-treino). Pode inventar — confira.")  # fmt: skip
        return f'<div class="rf-norag">{note}</div>'
    if not ans.contexts:
        return '<div class="rf-norag">Nenhum trecho recuperado.</div>'
    cited = {c.n for c in ans.citations}
    items = []
    for i, ctx in enumerate(ans.contexts, 1):
        where = pages(ctx)
        section = (ctx.get("section") or "").strip()
        meta = [str(ctx.get("year") or "s.d.")]
        if where != "Ficha do catálogo" and section:
            meta.append(esc(section[:70]))
        meta.append(esc(where))
        if doi := ctx.get("doi"):
            meta.append(f'<a href="https://doi.org/{esc(doi)}" target="_blank" rel="noopener">doi:{esc(doi)}</a>')
        excerpt = " ".join((ctx.get("text") or "").split())
        excerpt = excerpt if len(excerpt) <= EXCERPT_CHARS else excerpt[:EXCERPT_CHARS].rstrip() + "…"
        cls = f"rf-src rf-s{i}" + ("" if i in cited else " rf-src-uncited")  # classe, não data-*: o sanitizador remove
        tag = "" if i in cited else '<span class="rf-tag">não citado</span>'
        items.append(
            f'<li class="{cls}"><span class="rf-n">{i}</span><div class="rf-src-body">'
            f'<div class="rf-src-title">{esc(ctx.get("title") or short_ref(ctx))}{tag}</div>'
            f'<div class="rf-src-meta">{" · ".join(meta)}</div>'
            f'<div class="rf-src-quote">{esc(excerpt)}</div></div></li>'
        )
    summary = f"Fontes · {len(cited)} citada{'s' if len(cited) != 1 else ''} de {len(ans.contexts)}"
    warn = ""
    if ans.invalid_citations:
        warn = f'<div class="rf-warn">A resposta cita trechos inexistentes: {esc(ans.invalid_citations)}</div>'
    return f'<details class="rf-sources"><summary>{summary}</summary><ol>{"".join(items)}</ol>{warn}</details>'


def meta_html(ans: Answer, label: str) -> str:
    """Linha discreta sob a resposta: modelo · RAG · tempo · tokens/s."""
    g = ans.generation
    parts = [esc(label), "com documentos" if ans.use_rag else "sem documentos", f"{num(ans.latency_s)} s"]
    if ans.tokens_per_s:
        parts.append(f"{num(ans.tokens_per_s)} tokens/s")
    if g.eval_count:  # o Ollama só informa a contagem no fim do streaming
        parts.append(f"{g.eval_count} tokens")
    if g.load_duration_s >= 0.5:
        parts.append(f"carga do modelo {num(g.load_duration_s)} s")
    return f'<div class="rf-meta">{" · ".join(parts)}</div>'


def answer_html(ans: Answer | None, label: str, use_rag: bool, final: bool) -> str:
    """Conteúdo da mensagem do assistente: resposta (com [n] clicáveis) + Fontes + métricas."""
    if ans is None or not ans.answer:
        status = "Consultando documentos do Refauna…" if use_rag else "Gerando resposta…"
        if ans is not None and use_rag:
            status = "Gerando resposta…"
        return f'<div class="rf-status">{status}</div>'
    body = link_citations(ans.answer, len(ans.contexts))
    if not final:
        return body
    return f"{body}\n\n{sources_html(ans)}\n\n{meta_html(ans, label)}"


def stats_md(ans: Answer) -> str:
    g = ans.generation
    tps = f"{ans.tokens_per_s:.1f} tok/s" if ans.tokens_per_s else "— tok/s"
    rag = f"busca {ans.search_s * 1000:.0f} ms · " if ans.use_rag and ans.search_s > 0.001 else ""
    load = f" · carga do modelo {g.load_duration_s:.1f}s" if g.load_duration_s >= 0.5 else ""
    return f"⏱ {ans.latency_s:.1f}s · {rag}{g.eval_count} tokens · {tps}{load}"


def message_text(content) -> str:
    """Texto de uma mensagem do Chatbot: o Gradio 6 devolve o histórico como lista de partes {'type','text'}."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, list | tuple):
        return " ".join(message_text(c) for c in content).strip()
    return ""


def title_html(history: list[dict]) -> str:
    first = next((message_text(m.get("content")) for m in history if m.get("role") == "user"), "")
    first = first if len(first) <= TITLE_CHARS else first[:TITLE_CHARS].rstrip() + "…"
    return f'<div class="rf-title-text">{esc(first)}</div>' if first else ""


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


# ---------------------------------------------------------------- peças de HTML estático


def picker_html(s: Settings, current: str) -> str:
    """Seletor de modelo (topo à esquerda): botão com o modelo atual + menu com os 4 modelos."""
    items = []
    # disponíveis primeiro; dentro deles, os ajustados para o Refauna antes dos originais
    order = sorted(s.models.items(), key=lambda km: (not km[1].enabled, not km[0].endswith("_ft")))
    for k, m in order:
        blurb = MODEL_BLURBS.get(k, "") if m.enabled else (m.note or "indisponível")
        cls = "rf-model-item" + (" is-current" if k == current else "") + ("" if m.enabled else " is-disabled")
        attrs = f'data-rf-model="{esc(k)}" role="menuitemradio" aria-checked="{str(k == current).lower()}"'
        attrs += "" if m.enabled else ' aria-disabled="true"'
        items.append(
            f'<button type="button" class="{cls}" {attrs} data-label="{esc(m.label)}">'
            f'<span class="rf-model-text"><span class="rf-model-name">{esc(m.label)}</span>'
            f'<span class="rf-model-desc">{esc(blurb)}</span></span><span class="rf-check" aria-hidden="true"></span></button>'
        )
    return (
        '<div class="rf-left">'
        '<button type="button" class="rf-icon-btn rf-sb-open" data-rf-act="sidebar" title="Abrir barra lateral" aria-label="Abrir barra lateral"><span class="rf-i rf-i-sidebar"></span></button>'
        '<div class="rf-picker" id="rf-picker">'
        f'<button type="button" class="rf-picker-btn" data-rf-act="model-menu" aria-haspopup="menu" aria-expanded="false">'
        f'<span class="rf-picker-label">{esc(s.models[current].label)}</span><span class="rf-i rf-i-chevron"></span></button>'
        f'<div class="rf-menu" role="menu"><div class="rf-menu-head">Modelo</div>{"".join(items)}</div></div>'
        '<button type="button" class="rf-icon-btn rf-sb-close" data-rf-act="sidebar" title="Fechar barra lateral" aria-label="Fechar barra lateral"><span class="rf-i rf-i-sidebar"></span></button>'
        "</div>"
    )  # fmt: skip


RIGHT_HTML = (
    '<div class="rf-seg" role="tablist">'
    '<button type="button" data-rf-nav="chat" role="tab">Chat</button>'
    '<button type="button" data-rf-nav="compare" role="tab">Comparar</button>'
    '<button type="button" data-rf-nav="results" role="tab">Resultados</button></div>'
    '<button type="button" class="rf-icon-btn rf-theme" data-rf-act="theme" title="Alternar tema claro/escuro" aria-label="Alternar tema claro/escuro">'
    '<span class="rf-i rf-i-sun"></span><span class="rf-i rf-i-moon"></span></button>'
)  # fmt: skip

NAV_HTML = (
    '<nav class="rf-nav">'
    '<button type="button" class="rf-item" data-rf-nav="compare"><span class="rf-i rf-i-compare"></span>Comparar modelos</button>'
    '<button type="button" class="rf-item" data-rf-nav="results"><span class="rf-i rf-i-chart"></span>Resultados</button>'
    "</nav>"
    '<button type="button" class="rf-section" data-rf-act="examples" aria-expanded="true">Perguntas de exemplo<span class="rf-i rf-i-chevron-r"></span></button>'
)  # fmt: skip

PROFILE_HTML = (
    '<div class="rf-profile"><span class="rf-avatar">RF</span><span class="rf-profile-text">'
    '<span class="rf-profile-name">Assistente Refauna</span><span class="rf-profile-sub">Offline · MacBook Air M1</span>'
    "</span></div>"
)  # fmt: skip


# ---------------------------------------------------------------- app


def build_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> gr.Blocks:
    s = settings or get_settings()
    pipe = pipeline or RagPipeline(s)
    pipe.retriever.warmup()  # carrega EmbeddingGemma + índice antes da primeira pergunta
    keys = [k for _, k in model_choices(s)]
    default_model = (
        s.demo.default_model if s.demo.default_model in keys else next((k for k in keys if k.endswith("_ft")), keys[0])
    )
    presets = dict((k, label) for label, k in preset_choices(s))

    def chat(question, history, model_key, use_rag):
        """Uma pergunta independente (o modelo não vê as anteriores). Saídas: chatbot, caixa, título, métricas."""
        question = (question or "").strip()
        if not question:
            gr.Warning("Escreva uma pergunta.")
            yield history, "", gr.skip(), gr.skip()
            return
        label = s.models[model_key].label
        history = [*history, {"role": "user", "content": question},
                   {"role": "assistant", "content": answer_html(None, label, use_rag, False)}]  # fmt: skip
        title = title_html(history)
        yield history, "", title, ""
        ans = None
        for _, ans in pipe.stream(question, model_key, use_rag=use_rag):  # noqa: B007
            history[-1]["content"] = answer_html(ans, label, use_rag, False)
            yield history, "", title, stats_md(ans)
        history[-1]["content"] = answer_html(ans, label, use_rag, True)
        yield history, "", title, stats_md(ans)

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
            cards[i] = f'### {presets[key]}\n\n<div class="rf-status">aguardando…</div>'
        # A mesma busca para todas as configurações com RAG: comparação justa e um modelo por vez
        t0 = time.time()
        ctx = pipe.retriever.hybrid_search(question) if any(k.endswith("|rag") for k in selected) else []
        search_ms = (time.time() - t0) * 1000
        ctx_md = ""
        yield (*cards, ctx_md)
        for i, key in enumerate(selected):
            model_key, mode = key.split("|")
            use_rag = mode == "rag"
            label = s.models[model_key].label
            cards[i] = f'### {presets[key]}\n\n<div class="rf-status">carregando o modelo…</div>'
            yield (*cards, ctx_md)
            ans = None
            for _, ans in pipe.stream(question, model_key, use_rag=use_rag, contexts=ctx if use_rag else None):  # noqa: B007
                body = link_citations(ans.answer, len(ans.contexts)) or '<div class="rf-status">gerando…</div>'
                cards[i] = f"### {presets[key]}\n\n{body}\n\n{meta_html(ans, label)}"
                yield (*cards, ctx_md)
            cites = ", ".join(f"[{c.n}] {c.ref}" for c in ans.citations) if ans.citations else ("sem citações" if use_rag else "sem documentos")  # fmt: skip
            cards[i] += f'\n\n<div class="rf-meta">Fontes: {esc(cites)}</div>'
            if ctx and not ctx_md:
                ctx_md = f"Trechos enviados às configurações com RAG (os mesmos para todas · busca {search_ms:.0f} ms):\n\n```text\n" + format_context(ctx) + "\n```"  # fmt: skip
            yield (*cards, ctx_md)

    def show_stop():
        return gr.update(visible=False), gr.update(visible=True)

    def show_send():
        return gr.update(visible=True), gr.update(visible=False)

    with gr.Blocks(title="Assistente Refauna", fill_width=True, analytics_enabled=False) as demo:
        # ---- topo (fixo): seletor de modelo à esquerda, título/abas no centro, tema à direita
        with gr.Row(elem_id="rf-topbar"):
            gr.HTML(picker_html(s, default_model), elem_id="rf-topleft", js_on_load=None)
            title = gr.HTML("", elem_id="rf-title", js_on_load=None)
            gr.HTML(RIGHT_HTML, elem_id="rf-topright", js_on_load=None)
        # ---- barra lateral recolhível
        with gr.Column(elem_id="rf-sidebar"):
            new_chat = gr.Button("Nova conversa", elem_classes=["rf-item", "rf-item-new"])
            gr.HTML(NAV_HTML, js_on_load=None)
            with gr.Column(elem_id="rf-examples"):
                ex_buttons = [gr.Button(q, elem_classes=["rf-item", "rf-example"]) for q in s.demo.examples]
            gr.HTML(PROFILE_HTML, elem_id="rf-profile-wrap", js_on_load=None)

        with gr.Column(elem_id="rf-main"):
            # ---- Chat
            with gr.Column(elem_id="rf-view-chat", elem_classes="rf-view"):
                chatbot = gr.Chatbot(
                    elem_id="rf-chatbot", show_label=False, container=False, height=None,
                    buttons=["copy"], feedback_options=None, layout="bubble", sanitize_html=True,
                    placeholder=f'<div class="rf-greeting">{GREETING}</div>',
                )  # fmt: skip
                with gr.Column(elem_id="rf-dock"):
                    gr.HTML(f'<div class="rf-disclaimer">{DISCLAIMER}</div>', elem_id="rf-disclaimer-wrap",
                            js_on_load=None)  # fmt: skip
                    with gr.Row(elem_id="rf-composer", elem_classes="rf-composer"):
                        use_rag = gr.Checkbox(value=True, label="Consultar documentos do Refauna", elem_id="rf-rag",
                                              container=False)  # fmt: skip
                        question = gr.Textbox(placeholder=PLACEHOLDER, show_label=False, container=False, lines=1,
                                              max_lines=6, elem_id="rf-input", autofocus=True)  # fmt: skip
                        send = gr.Button("", elem_id="rf-send", elem_classes="rf-round-btn")
                        stop = gr.Button("", elem_id="rf-stop", elem_classes="rf-round-btn", visible=False)
                    with gr.Row(elem_id="rf-suggestions"):
                        sugg_buttons = [gr.Button(q, elem_classes="rf-sugg") for q in s.demo.examples[:3]]
                model = gr.Radio(keys, value=default_model, elem_id="rf-model", visible="hidden", show_label=False)
                stats = gr.Textbox(visible="hidden", elem_id="rf-stats", show_label=False)

            # ---- Comparar
            with gr.Column(elem_id="rf-view-compare", elem_classes=["rf-view", "rf-page"]):
                gr.HTML('<h1 class="rf-h1">Comparar modelos</h1><p class="rf-sub">A mesma pergunta respondida '
                        f"<b>em sequência</b> por até {MAX_COMPARE} configurações — um modelo carregado por vez.</p>",
                        js_on_load=None)  # fmt: skip
                with gr.Row(elem_classes="rf-composer", elem_id="rf-compare-composer"):
                    cq = gr.Textbox(placeholder="Pergunta para comparar", show_label=False, container=False, lines=1,
                                    max_lines=4, elem_id="rf-cq")  # fmt: skip
                    run = gr.Button("", elem_id="rf-run", elem_classes="rf-round-btn")
                sel = gr.CheckboxGroup([(v, k) for k, v in presets.items()],
                                       value=[k for k in s.demo.compare_defaults if k in presets][:MAX_COMPARE],
                                       show_label=False, elem_id="rf-presets", container=False)  # fmt: skip
                with gr.Row(elem_classes="rf-chips"):
                    cmp_examples = [gr.Button(q, elem_classes="rf-sugg") for q in s.demo.examples]
                with gr.Row(elem_id="rf-cards"):
                    cards = [gr.Markdown(elem_classes="rf-card", container=False) for _ in range(MAX_COMPARE)]
                with gr.Accordion("Trechos recuperados", open=False, elem_id="rf-ctx"):
                    ctx_box = gr.Markdown()

            # ---- Resultados
            with gr.Column(elem_id="rf-view-results", elem_classes=["rf-view", "rf-page"]):
                gr.HTML('<h1 class="rf-h1">Resultados</h1>', js_on_load=None)
                img = s.data_dir / "outputs" / "comparacao_modelos.png"
                if img.exists():
                    gr.Image(str(img), label="Gemma 4 E2B × Qwen3.5-2B (Colab)", show_label=False,
                             interactive=False, buttons=[], elem_id="rf-results-img")  # fmt: skip
                gr.Markdown(results_md(s), elem_classes="rf-md")
                local = gr.Markdown(local_md(s), elem_classes="rf-md")
                refresh = gr.Button("Atualizar resultados locais", elem_classes="rf-pill-btn")
                gr.Markdown(sources_list_md(s), elem_classes=["rf-md", "rf-small"])

        # ---- eventos do Chat (um modelo por vez: fila única "ollama")
        chat_io = dict(fn=chat, inputs=[question, chatbot, model, use_rag], outputs=[chatbot, question, title, stats],
                       concurrency_id="ollama", concurrency_limit=1, show_progress="hidden")  # fmt: skip
        to_chat = "() => { window.rfView && window.rfView('chat'); }"
        gens = []

        def wire(event, fill: str | None = None, api: str | bool = False):
            """Dispara a pergunta: (preenche) → mostra Parar → gera → mostra Enviar."""
            if fill is not None:
                event = event(lambda q=fill: q, None, question, js=to_chat, queue=False, api_name=False)
                event = event.then(show_stop, None, [send, stop], queue=False, api_name=False)
            else:
                event = event(show_stop, None, [send, stop], queue=False, api_name=False)
            gen = event.then(**chat_io, api_name=api)
            gen.then(show_send, None, [send, stop], queue=False, api_name=False)
            gens.append(gen)

        wire(question.submit, api="chat")
        wire(send.click)
        for b, q in zip([*ex_buttons, *sugg_buttons], [*s.demo.examples, *s.demo.examples[:3]], strict=True):
            wire(b.click, fill=q)
        stop.click(show_send, None, [send, stop], cancels=gens, queue=False, api_name=False)
        new_chat.click(lambda: ([], "", "", ""), None, [chatbot, question, title, stats], js=to_chat,
                       cancels=gens, queue=False, api_name=False).then(
            show_send, None, [send, stop], queue=False, api_name=False)  # fmt: skip

        # ---- eventos do Comparar
        compare_io = dict(fn=compare, inputs=[cq, sel], outputs=[*cards, ctx_box], concurrency_id="ollama",
                          concurrency_limit=1, show_progress="hidden")  # fmt: skip
        gr.on([run.click, cq.submit], **compare_io, api_name="compare")
        for b, q in zip(cmp_examples, s.demo.examples, strict=True):
            b.click(lambda q=q: q, None, cq, queue=False, api_name=False).then(**compare_io, api_name=False)
        refresh.click(lambda: local_md(s), None, local, api_name=False)
    return demo


def theme() -> gr.themes.Base:
    """Tema base com fontes do sistema (nada de Google Fonts); as cores finas ficam no CSS."""
    return gr.themes.Base(primary_hue="blue", neutral_hue="neutral", font=SYSTEM_FONTS, font_mono=MONO_FONTS).set(
        body_background_fill="#fcfcfc", body_background_fill_dark="#000000",
        body_text_size="16px", body_text_color="#0d0d0d", body_text_color_dark="#ededed",
    )  # fmt: skip


def main(host: str = "127.0.0.1", port: int | None = None, inbrowser: bool = True) -> None:
    s = get_settings()
    demo = build_app(s)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port or s.demo.server_port,
        inbrowser=inbrowser,
        theme=theme(),
        css_paths=[ASSETS / "refauna.css"],
        js=(ASSETS / "refauna.js").read_text(encoding="utf-8"),
        head='<meta name="color-scheme" content="light dark">',
        footer_links=[],
    )


if __name__ == "__main__":
    main()
