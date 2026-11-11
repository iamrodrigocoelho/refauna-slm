"""CLI `refauna` (Typer)."""

from __future__ import annotations

import itertools
from typing import Annotated

import httpx
import ollama
import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.table import Table

from refauna_slm import models as model_registry
from refauna_slm import sync as sync_mod
from refauna_slm.config import ROOT, get_settings

app = typer.Typer(help="Refauna SLM — RAG local offline (Gemma 4 E2B / Qwen3.5-2B via Ollama).", no_args_is_help=True)
models_app = typer.Typer(help="Registro de modelos do config/settings.yaml no Ollama.", no_args_is_help=True)
app.add_typer(models_app, name="models")
console = Console()


def _rel(p) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


@app.command()
def sync() -> None:
    """Copia os artefatos do Colab (Drive) para data/ e valida (F1)."""
    s = get_settings()
    data_dir = s.data_dir
    drive = s.drive_dir

    if drive is None or not drive.exists():
        why = "REFAUNA_DRIVE_DIR não definida" if drive is None else f"pasta não encontrada: {drive}"
        console.print(f"[yellow]⚠️  {why}.[/] Nada copiado. Baixe manualmente do Google Drive:")
        for line in sync_mod.manual_download_list():
            console.print(f"   • {line}")
        console.print(f"   (o projeto local fica em {_rel(data_dir)}/; validando o que já existe lá)\n")
    else:
        console.print(f"📂 Drive: {drive}")
        res = sync_mod.copy_artifacts(drive, data_dir)
        console.print(
            f"   copiados: {len(res.copied)} · sem mudança: {len(res.unchanged)} · ausentes no Drive: {len(res.missing)}"
        )
        for f in res.copied:
            console.print(f"   [green]+[/] data/{f}")
        for f in res.missing:
            console.print(f"   [dim]- {f} (não está no Drive)[/]")
        console.print()

    try:
        rep = sync_mod.validate(data_dir)
    except sync_mod.SyncError as e:
        console.print(f"[red]❌ {e}[/]")
        raise typer.Exit(1) from e

    t = Table(title="Artefatos em data/", show_header=False, title_justify="left")
    t.add_column(style="bold")
    t.add_column()
    docs = f"{rep.n_docs} com texto" + (f" · docs.jsonl: {rep.n_docs_jsonl}" if rep.n_docs_jsonl is not None else "")
    t.add_row("Documentos", docs)
    t.add_row("Trechos de texto", str(rep.n_text_chunks))
    t.add_row("Fichas do catálogo", str(rep.n_fichas))
    emb = (
        "ausente"
        if rep.embeddings_shape is None
        else f"{rep.embeddings_shape} " + ("✅ reaproveitável" if rep.embeddings_ok else "❌ não bate")
    )
    t.add_row("chunk_embeddings.npy", emb)
    if rep.test_by_kind:
        kinds = " · ".join(f"{k}: {v}" for k, v in sorted(rep.test_by_kind.items()))
        t.add_row("sft/test.jsonl", f"{sum(rep.test_by_kind.values())} exemplos ({kinds})")
    t.add_row("Runs do Colab", ", ".join(rep.runs) or "nenhum")
    t.add_row("Gold set", "sim" if rep.has_gold else "não")
    console.print(t)
    for w in rep.warnings:
        console.print(f"[yellow]⚠️  {w}[/]")
    console.print("[green]✅ chunks.jsonl válido (schema §4.1).[/]")


@models_app.command("pull")
def models_pull() -> None:
    """Baixa os modelos ativos do registro no Ollama (precisa de internet)."""
    s = get_settings()
    cli = model_registry.client(s)
    failed = []
    selected = model_registry.select(s, None)
    _print_disabled(s, selected)
    for key, spec in selected.items():
        console.print(f"⬇️  [bold]{key}[/] · {spec.label} · {spec.tag}")
        try:
            with Progress(
                TextColumn("   {task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
                transient=True,
            ) as prog:
                tasks: dict[str, int] = {}
                for ev in model_registry.pull(cli, spec):
                    if ev.digest and ev.total:
                        if ev.digest not in tasks:
                            tasks[ev.digest] = prog.add_task(ev.digest[7:19], total=ev.total)
                        prog.update(tasks[ev.digest], completed=ev.completed or 0)
            console.print("   [green]✅ ok[/]")
            if spec.text_only:
                path = model_registry.build_text_only(s, key)
                console.print(f"   [green]✅ variante só-texto {spec.tag}[/] ({_rel(path)})")
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError, RuntimeError) as e:
            failed.append(key)
            console.print(f"   [red]❌ {e}[/]")
            if spec.tag.startswith("hf.co/"):
                console.print(f"   [dim]{model_registry.PRIVATE_HINT}[/]")
    if failed:
        console.print(f"[red]Falharam: {', '.join(failed)}[/]")
        raise typer.Exit(1)


@models_app.command("build")
def models_build() -> None:
    """Recria as variantes só-texto (text_only no settings.yaml) a partir das tags de origem já baixadas."""
    s = get_settings()
    keys = [k for k, m in s.models.items() if m.text_only and m.enabled]
    if not keys:
        console.print("Nenhum modelo com text_only no settings.yaml.")
        return
    for key in keys:
        spec = s.model(key)
        try:
            path = model_registry.build_text_only(s, key)
        except (ollama.ResponseError, RuntimeError, ValueError) as e:
            console.print(f"[red]❌ {key}: {e}[/]")
            if "not found" in str(e):
                console.print(f"[dim]Baixe a origem antes: ollama pull {spec.source}[/]")
            raise typer.Exit(1) from e
        console.print(f"[green]✅ {key}[/]: {spec.source} → {spec.tag} (sem projetor de visão/áudio) · {_rel(path)}")


@models_app.command("check")
def models_check(
    smoke: bool = typer.Option(False, "--smoke", help="Gera algumas palavras com cada modelo (um por vez)."),
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Chave (qwen_ft) ou família (qwen); repetível.")
    ] = None,
) -> None:
    """Confere se os modelos estão no Ollama local, ou seja, disponíveis offline."""
    s = get_settings()
    try:
        model_registry.select(s, model)
    except KeyError as e:
        console.print(f"[red]❌ {e.args[0]}[/]")
        raise typer.Exit(2) from e
    try:
        version = model_registry.server_version(s.ollama.host)
    except (httpx.HTTPError, ConnectionError) as e:
        console.print(
            f"[red]❌ Ollama não responde em {s.ollama.host} ({e}). Abra o app Ollama ou rode `ollama serve`.[/]"
        )
        raise typer.Exit(1) from e
    console.print(f"🦙 Ollama {version} em {s.ollama.host}")

    with console.status("verificando...") as status:
        results = model_registry.check(
            s, smoke=smoke, on_model=lambda k: status.update(f"verificando {k}..."), keys=model
        )

    t = Table(title="Modelos (config/settings.yaml)", title_justify="left")
    for col in ("chave", "rótulo", "tag", "família", "think", "status", "tamanho", "quant.", "params"):
        t.add_column(col)
    if smoke:
        t.add_column("smoke")
    for r in results:
        think = "false" if r.spec.think is False else "—"
        status_txt = "[green]✅ local[/]" if r.present else "[red]❌ ausente[/]"
        row = [
            r.key,
            r.spec.label,
            r.spec.tag,
            r.spec.family,
            think,
            status_txt,
            f"{r.size_gb} GB" if r.size_gb else "",
            r.quantization or "",
            r.parameter_size or "",
        ]
        if smoke:
            row.append(("[green]" if r.smoke_ok else "[red]") + (r.smoke or "") + "[/]")
        t.add_row(*row)
    console.print(t)
    _print_disabled(s, model_registry.select(s, model))

    missing = [r for r in results if not r.present]
    bad_smoke = [r for r in results if smoke and r.present and not r.smoke_ok]
    for r in results:
        if r.error:
            console.print(f"[yellow]⚠️  {r.key}: {r.error}[/]")
        if r.present and r.spec.text_only and r.multimodal:
            console.print(f"[yellow]⚠️  {r.key}: ainda tem projetor de visão/áudio — rode `refauna models build`.[/]")
        if r.present and r.has_template is False:
            console.print(f"[yellow]⚠️  {r.key}: GGUF sem TEMPLATE de chat — ver PRD F8.4 (Modelfile).[/]")
    if any(r.spec.text_only for r in missing):
        console.print(
            "[dim]Variante só-texto ausente: `refauna models build` (offline, se a origem já foi baixada).[/]"
        )
    if missing:
        console.print(
            f"[red]❌ Faltam {len(missing)} modelo(s): {', '.join(r.key for r in missing)}. Rode `refauna models pull` com internet.[/]"
        )
        if any(r.spec.tag.startswith("hf.co/") for r in missing):
            console.print(f"[dim]{model_registry.PRIVATE_HINT}[/]")
        raise typer.Exit(1)
    if bad_smoke:
        console.print(f"[red]❌ Smoke test falhou: {', '.join(r.key for r in bad_smoke)}[/]")
        raise typer.Exit(1)
    console.print(f"[green]✅ {len(results)}/{len(results)} modelos disponíveis offline.[/]")


def _print_disabled(s, selected) -> None:
    for key, spec in model_registry.disabled(s, selected).items():
        note = f" — {spec.note}" if spec.note else ""
        console.print(f"[dim]⏸  {key} ({spec.tag}) desativado no settings.yaml{note}[/]")


@app.command()
def index(
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Recria o índice mesmo se atualizado.")] = False,
) -> None:
    """Constrói o ChromaDB em data/chroma (+ BM25 em memória), reaproveitando chunk_embeddings.npy (F3)."""
    import time

    from refauna_slm.index import build_index
    from refauna_slm.rag.retriever import HybridRetriever

    s = get_settings()
    try:
        with console.status("indexando...") as status:
            res = build_index(s, rebuild=rebuild, on_status=status.update)
            status.update("carregando EmbeddingGemma + BM25...")
            r = HybridRetriever(s)
            r.warmup()
            t0 = time.time()
            for q in SAMPLE_QUERIES:
                r.hybrid_search(q)
            search_ms = (time.time() - t0) / len(SAMPLE_QUERIES) * 1000
    except sync_mod.SyncError as e:
        console.print(f"[red]❌ {e}[/]")
        raise typer.Exit(1) from e
    verb = "já estava atualizado" if res.embeddings_source == "existente" else f"construído em {res.seconds:.1f}s"
    console.print(f"✅ Índice {verb}: {res.n} vetores em {_rel(res.path)} (embeddings: {res.embeddings_source})")
    console.print(
        f"   BM25: {len(r.chunks)} documentos · EmbeddingGemma em {r.embedder.device} · busca média {search_ms:.0f} ms"
    )
    q = SAMPLE_QUERIES[0]
    console.print(f"\n❓ {q}")
    for i, c in enumerate(r.hybrid_search(q), 1):
        from refauna_slm.contract import short_ref
        from refauna_slm.rag.pipeline import pages

        where = pages(c) if c["section_canon"] == "catalog" else f"{c['section'][:40]} · {pages(c)}"
        console.print(f"  [{i}] {short_ref(c)} · {where}", highlight=False, markup=False)


SAMPLE_QUERIES = [
    "O que é refaunação e por que ela importa para a Mata Atlântica?",
    "Quais animais foram reintroduzidos no Parque Nacional da Tijuca?",
    "Qual o papel das cutias na dispersão de sementes?",
    "Que estudos o Refauna tem sobre jabutis?",
]


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Pergunta em português.")],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="qwen_ft | gemma_ft | qwen_base (padrão: demo.default_model)")
    ] = None,
    no_rag: Annotated[bool, typer.Option("--no-rag", help="Pergunta sem contexto.")] = False,
    show_context: Annotated[bool, typer.Option("--show-context", help="Mostra os trechos enviados ao modelo.")] = False,
    temperature: Annotated[float | None, typer.Option(help="Sobrescreve a temperatura do settings.yaml.")] = None,
) -> None:
    """Pergunta → busca híbrida → Ollama (streaming) → resposta com citações [n] (F4)."""
    from refauna_slm.rag.pipeline import RagPipeline

    s = get_settings()
    model = model or s.demo.default_model
    try:
        spec = s.model(model)
    except KeyError as e:
        console.print(f"[red]❌ {e.args[0]}[/]")
        raise typer.Exit(2) from e
    if not spec.enabled:
        console.print(f"[yellow]⚠️  {model} está desativado no settings.yaml ({spec.note}). Tentando mesmo assim.[/]")
    opts = {} if temperature is None else {"temperature": temperature}
    pipe = RagPipeline(s)
    console.print(f"[dim]{spec.label} · {'RAG top-' + str(s.retrieval.top_k) if not no_rag else 'sem RAG'}[/]")
    printed = ""
    try:
        stream = pipe.stream(question, model, use_rag=not no_rag, **opts)
        with console.status("carregando EmbeddingGemma, índice e o modelo...") as status:
            if not no_rag:
                pipe.retriever.warmup()  # a carga do EmbeddingGemma não entra no tempo de busca
            status.update(f"gerando com {spec.tag}...")
            first = next(stream)
        ans = first[1]
        if show_context and ans.contexts:
            from refauna_slm.contract import format_context

            console.print(f"[dim]{format_context(ans.contexts)}[/]\n", highlight=False, markup=False)
        for _delta, partial in itertools.chain([first], stream):
            ans = partial
            if ans.answer.startswith(printed) and len(ans.answer) > len(printed):
                console.print(ans.answer[len(printed) :], end="", highlight=False, markup=False)
                printed = ans.answer
    except sync_mod.SyncError as e:
        console.print(f"[red]❌ {e}[/]")
        raise typer.Exit(1) from e
    except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as e:
        console.print(f"\n[red]❌ Ollama: {e}[/]")
        raise typer.Exit(1) from e
    console.print("\n")
    for c in ans.citations:
        doi = f" · doi:{c.doi}" if c.doi else ""
        where = c.pages if c.pages == "Ficha do catálogo" else f"{c.section[:40]} · {c.pages}"
        console.print(f"  [{c.n}] {c.ref} · {where}{doi}", highlight=False, markup=False)
    if ans.invalid_citations:
        console.print(f"[yellow]⚠️  cita trechos inexistentes: {ans.invalid_citations}[/]")
    if ans.contexts and not ans.citations:
        console.print("[dim](nenhuma citação \\[n] na resposta)[/]")
    g = ans.generation
    tps = f"{ans.tokens_per_s:.1f} tok/s" if ans.tokens_per_s else "—"
    console.print(
        f"[dim]⏱ {ans.latency_s:.1f}s (busca {ans.search_s * 1000:.0f} ms · carga {g.load_duration_s:.1f}s) · "
        f"{g.eval_count} tokens · {tps} · PT-BR={'sim' if ans.ptbr else 'não'} · abstenção={'sim' if ans.abstained else 'não'}[/]"
    )


@app.command("eval")
def eval_cmd(
    parity: Annotated[
        bool, typer.Option("--parity", help="F6.1: paridade Colab × Ollama (10 perguntas, temp=0).")
    ] = False,
    retrieval: Annotated[bool, typer.Option("--retrieval", help="Recall@4/MRR no sft/test.jsonl vs Colab.")] = False,
    configs: Annotated[str, typer.Option(help="Configurações do A/B/C/D, ex.: B,D.")] = "A,B,C,D",
    n: Annotated[int | None, typer.Option(help="Perguntas da amostra (padrão: eval.n_items).")] = None,
    judge: Annotated[bool, typer.Option("--judge", help="Juiz LLM (mesmo do Colab; requer internet).")] = False,
    model: Annotated[list[str] | None, typer.Option("--model", "-m", help="Modelos FT para a paridade.")] = None,
) -> None:
    """Avaliação local: A/B/C/D com métricas automáticas (F6.2); --retrieval e --parity à parte."""
    s = get_settings()
    try:
        if retrieval:
            _eval_retrieval(s)
        if parity:
            _eval_parity(s, model or [k for k, m in s.models.items() if m.colab_run and m.enabled])
        if not (parity or retrieval):
            _eval_abcd(s, {c.strip().upper() for c in configs.split(",") if c.strip()}, n, judge)
    except sync_mod.SyncError as e:
        console.print(f"[red]❌ {e}[/]")
        raise typer.Exit(1) from e


def _eval_abcd(s, configs: set[str], n: int | None, use_judge: bool) -> None:
    import time

    from refauna_slm.eval import local_eval
    from refauna_slm.rag.pipeline import RagPipeline

    judge = None
    if use_judge:
        try:
            from refauna_slm.eval.judge import Judge

            judge = Judge(s.eval.judge_model)
        except ImportError as e:
            console.print("[red]❌ Juiz requer o pacote anthropic: uv sync --extra judge[/]")
            raise typer.Exit(1) from e
    pipe = RagPipeline(s)
    t0 = time.time()
    with console.status("A/B/C/D local...") as status:
        pipe.retriever.warmup()
        df = local_eval.run_abcd(
            s, pipe, n=n, judge=judge, configs=configs,
            on_item=lambda cfg, i, total: status.update(f"A/B/C/D · {cfg}: {i}/{total}"),
        )  # fmt: skip
    summary = local_eval.summarize_abcd(df)
    meta = {"n_items": int(df.id.nunique()), "configs": sorted(configs), "judge": s.eval.judge_model if judge else None,
            "minutos": round((time.time() - t0) / 60, 1), "temperature": 0}  # fmt: skip
    local_eval.save_abcd(df, summary, s, meta)
    cols = ["config", "n", "ptbr", "com_citacao", "citacao_correta", "abstencao_correta", "falsa_abstencao",
            "latencia_s", "tokens_s", "tokens_resposta"] + (["correcao", "fidelidade"] if judge else [])  # fmt: skip
    for family in sorted({r["family"] for r in summary}):
        t = Table(title=f"A/B/C/D local · {family} ({meta['n_items']} perguntas, temp=0)", title_justify="left")
        for c in cols:
            t.add_column(c.replace("_", " "), justify="right")
        for r in (r for r in summary if r["family"] == family):
            t.add_row(*(_fmt(r.get(c)) for c in cols))
        console.print(t)
    console.print("[dim]falsa abstenção = se absteve numa pergunta respondível com o trecho certo no contexto.[/]")
    console.print(f"[dim]{meta['minutos']} min · detalhes: {_rel(s.local_eval_dir / 'abcd_detalhado.csv')}[/]")


@app.command()
def bench(
    model: Annotated[list[str] | None, typer.Option("--model", "-m", help="Padrão: todos os modelos ativos.")] = None,
    runs: Annotated[int, typer.Option(help="Respostas a quente por modelo.")] = 6,
) -> None:
    """Benchmark no M1 (F6.3): carga a frio, latência p50/p95, tokens/s, RAM pico e tamanho do GGUF."""
    from refauna_slm.eval.bench import run_bench, save_bench
    from refauna_slm.rag.backends import OllamaBackend
    from refauna_slm.rag.retriever import HybridRetriever

    s = get_settings()
    keys = model or [k for k, m in s.models.items() if m.enabled]
    with console.status("benchmark...") as status:
        r = HybridRetriever(s)
        r.warmup()
        res = run_bench(s, keys, r, OllamaBackend(s), warm_runs=runs, on_step=status.update)
    save_bench(res, s)
    t = Table(title=f"Benchmark no M1 ({res['warm_runs']} respostas a quente, RAG top-4)", title_justify="left")
    cols = ["model", "gguf_gb", "carga_fria_s", "latencia_p50_s", "latencia_p95_s", "tokens_s", "prompt_tokens_s",
            "tokens_resposta", "ram_ollama_pico_gb"]  # fmt: skip
    for c in cols:
        t.add_column(c.replace("_", " "), justify="right")
    for m in res["models"]:
        t.add_row(*(_fmt(m.get(c)) for c in cols))
    console.print(t)
    console.print(f"Busca híbrida: {res['busca_ms']} ms/pergunta · app (EmbeddingGemma em {res['embedder_device']}): "
                  f"{res['app_embedder_gb']} GB")  # fmt: skip
    console.print(f"[dim]salvo em {_rel(s.local_eval_dir / 'bench.json')}[/]")


def _eval_retrieval(s) -> None:
    import json
    import time

    from refauna_slm.rag.retriever import HybridRetriever

    with console.status("Recall@4 no sft/test.jsonl..."):
        r = HybridRetriever(s)
        r.warmup()
        test = sync_mod.read_jsonl(s.data_dir / "sft" / "test.jsonl")
        items = [e for e in test if e["kind"] in ("rag", "unanswerable")]
        t0 = time.time()
        local = r.retrieval_metrics(items)
        ms = (time.time() - t0) / len(items) * 1000
    runs = sorted((s.data_dir / "outputs").glob("*/eval_resumo.json"))
    colab = json.loads(runs[0].read_text())["retrieval"] if runs else {}
    k = s.retrieval.top_k
    t = Table(title=f"Recuperação híbrida — sft/test.jsonl (rag + unanswerable, n={local['n']})", title_justify="left")
    for col in ("métrica", "local (M1)", "Colab", "diferença"):
        t.add_column(col, justify="right")
    for m in (f"recall@{k}", "mrr@20", f"doc_recall@{k}"):
        c = colab.get(m)
        diff = "" if c is None else f"{local[m] - c:+.3f}"
        t.add_row(m, f"{local[m]:.3f}", "" if c is None else f"{c:.3f}", diff)
    console.print(t)
    ok = colab and abs(local[f"recall@{k}"] - colab[f"recall@{k}"]) <= 0.03
    console.print(f"Busca média: {ms:.0f} ms/pergunta (aceite: < 300 ms) {'✅' if ms < 300 else '❌'}")
    console.print(f"Recall@{k}: |local − Colab| ≤ 0,03 {'✅' if ok else '❌'}")
    s.local_eval_dir.mkdir(parents=True, exist_ok=True)
    (s.local_eval_dir / "retrieval.json").write_text(
        json.dumps({"local": local, "colab": colab, "search_ms": round(ms, 1)}, indent=2)
    )


def _eval_parity(s, models: list[str]) -> None:
    from refauna_slm.eval import local_eval
    from refauna_slm.rag.pipeline import RagPipeline

    pipe = RagPipeline(s)
    for model in models:
        with console.status(f"paridade {model}...") as status:
            df = local_eval.run_parity(
                s, model, pipe, on_item=lambda i, n, e, m=model: status.update(f"paridade {m}: {i}/{n} ({e['kind']})")
            )
        summary = local_eval.summarize(df, s, model)
        local_eval.save(df, summary, s, model)
        t = Table(title=f"Paridade {model} × Colab ({summary['n']} perguntas, FT+RAG, temp=0)", title_justify="left")
        for col in ("métrica", "local (Ollama)", "Colab"):
            t.add_column(col, justify="right")
        for key in (
            "ptbr",
            "com_citacao",
            "citacao_correta_rag",
            "abstencao_correta",
            "falsa_abstencao",
            "chars_medio",
        ):
            v = summary[key]
            colab = v.get("colab", v.get("colab_agregado"))
            t.add_row(key + (" (agregado Colab)" if "colab_agregado" in v else ""), _fmt(v["local"]), _fmt(colab))
        console.print(t)
        console.print(
            f"latência média {summary['latencia_s_media']}s · {summary['tokens_por_s_medio']} tok/s · "
            f"think vazado: {summary['think_leak']} · cortadas: {summary['truncadas']}"
        )
        div = df[df.divergences != ""]
        console.print(f"[bold]Divergências: {len(div)}/{len(df)} itens[/]")
        for r in div.itertuples():
            console.print(f"  • [{r.kind}] {r.question[:80]}", highlight=False, markup=False)
            console.print(f"    → {r.divergences}", style="yellow", highlight=False, markup=False)
        console.print(f"[dim]detalhes: {_rel(s.local_eval_dir / f'parity_{model}.csv')}[/]\n")


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


@app.command()
def serve(
    port: Annotated[int | None, typer.Option(help="Porta (padrão: demo.server_port do settings.yaml).")] = None,
    host: Annotated[str, typer.Option(help="Use 0.0.0.0 para abrir na rede local.")] = "127.0.0.1",
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Não abre o navegador.")] = False,
) -> None:
    """App Gradio: Chat, Comparar e Resultados (F5)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gradio_app", ROOT / "app" / "gradio_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.main(host=host, port=port, inbrowser=not no_browser)
    except sync_mod.SyncError as e:
        console.print(f"[red]❌ {e}[/]")
        raise typer.Exit(1) from e


@app.command()
def report() -> None:
    """Gera docs/avaliacao.md + PNGs para slide a partir de data/outputs e data/local_eval (F6.4)."""
    from refauna_slm.eval.report import build_report

    md, pngs = build_report(get_settings())
    console.print(f"✅ {_rel(md)}")
    for p in pngs:
        console.print(f"   🖼  {_rel(p)}")


@app.command()
def cards(
    push: Annotated[bool, typer.Option("--push", help="Envia os README.md ao Hugging Face (só após revisão).")] = False,
) -> None:
    """Gera os model cards do HF em model_cards/ a partir dos resultados (PRD F8.3)."""
    from refauna_slm.model_cards import build_cards, push_cards

    s = get_settings()
    for p in build_cards(s):
        console.print(f"📝 {_rel(p)}")
    if push:
        for url in push_cards(s):
            console.print(f"[green]✅ {url}[/]")


if __name__ == "__main__":
    app()
