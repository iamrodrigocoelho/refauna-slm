"""Model cards do Hugging Face (PRD F8.3) gerados a partir dos dados do projeto.

`refauna cards` escreve model_cards/<repo>/README.md para revisão; `--push` envia (só com aprovação).
Números vêm de data/outputs (Colab) e data/local_eval (Mac) — os mesmos do docs/avaliacao.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from refauna_slm.config import ROOT, CardFamily, Settings
from refauna_slm.sync import read_jsonl

OUT_DIR = ROOT / "model_cards"
CONFIGS = ["A: base", "B: base+RAG", "C: FT", "D: FT+RAG"]
KIND = {"lora": "adapter LoRA", "merged": "modelo mesclado (16-bit)", "gguf": "GGUF (Q4_K_M e Q8_0)"}
LIBRARY = {"lora": "peft", "merged": "transformers", "gguf": "gguf"}
RELATION = {"lora": "adapter", "merged": "finetune", "gguf": "quantized"}
TAGS = {"gemma": "gemma4", "qwen": "qwen3.5"}
PLURAL = {"Artigo": "artigos", "Resumo em anais": "resumos em anais", "Capítulo": "capítulos de livro",
          "Iniciação científica / Projeto": "trabalhos de iniciação científica", "Dissertação/Tese/TCC": "dissertações"}  # fmt: skip
SINGULAR = {"Artigo/Boletim": "artigo em boletim", "Dissertação/Tese/TCC": "dissertação", "Capítulo": "capítulo de livro",
            "Artigo": "artigo", "Resumo em anais": "resumo em anais"}  # fmt: skip


def br(v: float, nd: int = 1) -> str:
    """Número no texto corrido em PT-BR (vírgula decimal)."""
    return f"{v:.{nd}f}".replace(".", ",")


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _f(v, nd=2, pct=False) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}"


def attribution(settings: Settings) -> str:
    docs = sorted(read_jsonl(settings.data_dir / "docs.jsonl"), key=lambda d: (d.get("year") or 0, d["title"]))
    lines = []
    for d in docs:
        authors = (d.get("authors") or "").replace(";", ",").strip()
        doi = f" DOI: [{d['doi']}](https://doi.org/{d['doi']})" if d.get("doi") else ""
        tipo = f" _{d['tipo_documento']}_." if d.get("tipo_documento") else ""
        who = authors.rstrip(".") + ". " if authors else ""
        lines.append(f"- {who}({d.get('year') or 's.d.'}). **{d['title'].rstrip('.')}**.{tipo}{doi}")
    return "\n".join(lines)


def corpus_summary(settings: Settings) -> str:
    docs = read_jsonl(settings.data_dir / "docs.jsonl")
    kinds = Counter(d.get("tipo_documento") or "outro" for d in docs)
    parts = ", ".join(
        f"{n} {PLURAL.get(k, k.lower()) if n > 1 else SINGULAR.get(k, k.lower())}" for k, n in kinds.most_common()
    )
    return f"{len(docs)} trabalhos do Refauna com texto ({parts})"


def colab_table(settings: Settings, fam: CardFamily) -> tuple[str, dict]:
    r = _load(settings.data_dir / "outputs" / fam.colab_run / "eval_resumo.json") or {}
    rows = {s["config"]: s for s in r.get("summary", [])}
    head = "| config | correção (1–5) | fidelidade (1–5) | citação correta | abstenção correta | falsa abstenção | PT-BR | latência (s, L4) |"
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        s = rows.get(c, {})
        lines.append(f"| {c} | {_f(s.get('correcao'))} | {_f(s.get('fidelidade'))} | {_f(s.get('citacao_correta'), pct=True)} | "
                     f"{_f(s.get('abstencao_correta'), pct=True)} | {_f(s.get('falsa_abstencao'), pct=True)} | "
                     f"{_f(s.get('ptbr'), pct=True)} | {_f(s.get('latencia_s'), 1)} |")  # fmt: skip
    return "\n".join(lines), {"rows": rows, "retrieval": r.get("retrieval", {})}


def local_section(settings: Settings, family: str) -> str:
    d = settings.local_eval_dir
    abcd, bench = _load(d / "abcd_resumo.json"), _load(d / "bench.json")
    ft_key = f"{family}_ft"
    out = []
    if abcd:
        rows = [r for r in abcd["summary"] if r["family"] == family and r["config"] == "D: FT+RAG"]
        if rows:
            r = rows[0]
            out.append(f"- **D (FT + RAG), {abcd['n_items']} perguntas de teste, temperatura 0:** citação correta "
                       f"{_f(r['citacao_correta'], pct=True)}, abstenção correta {_f(r['abstencao_correta'], pct=True)} "
                       f"(sobre {r.get('n_sem_resposta', '?')} perguntas sem resposta), falsa abstenção "
                       f"{_f(r['falsa_abstencao'], pct=True)} (sobre {r.get('n_trecho_certo', '?')} com o trecho certo no contexto), "
                       f"PT-BR {_f(r['ptbr'], pct=True)}.")  # fmt: skip
    if bench:
        m = next((m for m in bench["models"] if m["model"] == ft_key), None)
        if m:
            out.append(f"- **Desempenho (GGUF Q4_K_M via Ollama, RAG top-4):** {m['tokens_s']:.0f} tokens/s, latência "
                       f"p50 {br(m['latencia_p50_s'])} s (p95 {br(m['latencia_p95_s'])} s), carga a frio {br(m['carga_fria_s'])} s, "
                       f"~{m['tokens_resposta']:.0f} tokens por resposta.")  # fmt: skip
    if not out:
        return ""
    return "### No MacBook Air M1 (8 GB), offline\n" + "\n".join(out) + "\n"


def reading(rows: dict) -> str:
    b, c, d = rows.get("B: base+RAG", {}), rows.get("C: FT", {}), rows.get("D: FT+RAG", {})
    if not (b and d):
        return ""
    msg = ["**Leitura:** o maior ganho de correção vem do RAG (A → B)."]
    if (d.get("correcao") or 0) <= (b.get("correcao") or 0) + 0.2:
        msg.append("Com RAG, o fine-tuning não aumentou a correção (a busca é o gargalo: Recall@4 ≈ 0,42);")
    changes = []
    for m, label in [
        ("fidelidade", "fidelidade"),
        ("citacao_correta", "citação correta"),
        ("abstencao_correta", "abstenção correta"),
    ]:
        if b.get(m) is not None and d.get(m) is not None and abs(d[m] - b[m]) > (0.2 if m == "fidelidade" else 0.05):
            fmt = (lambda v: br(v, 2)) if m == "fidelidade" else (lambda v: f"{v * 100:.0f}%")
            changes.append(f"{label} {fmt(b[m])} → {fmt(d[m])}")
    if b.get("latencia_s") and d.get("latencia_s") and d["latencia_s"] < b["latencia_s"] - 1:
        changes.append(f"latência {br(b['latencia_s'])} → {br(d['latencia_s'])} s")
    msg.append(f"o efeito dele foi de comportamento: {', '.join(changes)}." if changes
               else "o efeito dele ficou dentro do ruído da amostra.")  # fmt: skip
    ab, ad = b.get("abstencao_correta"), d.get("abstencao_correta")
    if ab is not None and ad is not None and ab >= 0.8 and abs(ad - ab) <= 0.05:
        msg.append(
            f"O modelo-base já se abstinha bem quando o contexto não tinha a resposta ({ab * 100:.0f}%), e o fine-tuning manteve isso."
        )
    if c.get("correcao") is not None and b.get("correcao") is not None and b["correcao"] > c["correcao"]:
        msg.append("Sem RAG (C), o modelo erra fatos: use-o com RAG.")
    return " ".join(msg)


def card(settings: Settings, family: str, kind: str, system_prompt: str) -> str:
    fam = settings.hf_cards[family]
    other = next((f for k, f in settings.hf_cards.items() if k != family), None)
    train = _load(settings.data_dir / "outputs" / fam.colab_run / "train_report.json") or {}
    table, colab = colab_table(settings, fam)
    rr = colab["retrieval"]
    gguf_repo, merged_repo, lora_repo = fam.repos["gguf"], fam.repos["merged"], fam.repos["lora"]
    base_model = merged_repo if kind == "gguf" else fam.base
    method = "QLoRA 4-bit" if train.get("load_in_4bit") else "LoRA 16-bit (bf16)"
    is_qwen = family == "qwen"

    front = (f"---\nlicense: apache-2.0\nbase_model: {base_model}\nbase_model_relation: {RELATION[kind]}\n"
             f"language: [pt, en]\nlibrary_name: {LIBRARY[kind]}\npipeline_tag: text-generation\n"
             f"tags: [{TAGS[family]}, unsloth, {'qlora' if train.get('load_in_4bit') else 'lora'}, rag, raft, refauna, "
             f"biodiversidade, conservacao, mata-atlantica, refaunacao, portuguese]\n---\n")  # fmt: skip

    variants = "\n".join(f"| {KIND[k]} | [`{repo}`](https://huggingface.co/{repo}){' **← este repositório**' if k == kind else ''} |"
                         for k, repo in fam.repos.items())  # fmt: skip
    sibling = (
        f"\nModelo irmão, com os mesmos dados, prompt e avaliação: [{other.name}](https://huggingface.co/{other.repos['gguf']}).\n"
        if other
        else ""
    )

    think_note = ("O Qwen3.5 liga o raciocínio (`<think>`) por padrão: **desligue** — `enable_thinking=False` no "
                  "`apply_chat_template` e `\"think\": false` na API do Ollama. O modelo foi treinado sem raciocínio."
                  if is_qwen else
                  "Sem raciocínio (template `gemma-4` sem thinking). No Ollama, **não** envie o parâmetro `think`: "
                  "nas versões 0.40+, `think: false` altera o prompt do Gemma 4 e as respostas se afastam do treino.")  # fmt: skip

    ollama_tag = f"hf.co/{gguf_repo}:Q4_K_M"
    if is_qwen:
        ollama_note = 'Na API (`/api/chat`), envie `"think": false` em toda chamada.'
    else:
        ollama_note = "Envie as mensagens **sem** o parâmetro `think`."
    text_only = "" if is_qwen else f"""
**Pouca memória (ex.: Mac de 8 GB)?** O `ollama run` acima carrega também o projetor de visão (`*-mmproj.gguf`, ~1 GB), que este projeto, só de texto, não usa. O `Modelfile` do repositório GGUF cria uma variante **só-texto**, com os mesmos pesos, o mesmo template e os mesmos stops:
```bash
hf download {gguf_repo} gemma-4-E2B-it.Q4_K_M.gguf Modelfile --local-dir {family}-refauna
cd {family}-refauna && ollama create {family}-refauna-texto -f Modelfile
```
Validada com temperatura 0: respostas idênticas às do `ollama run` original. No MacBook Air M1 de 8 GB, reduziu o pico de memória da demo de 6,55 GB para 5,70 GB.
"""  # fmt: skip
    ollama = f"""### Ollama
```bash
ollama run {ollama_tag}
```
{ollama_note}
{text_only}
"""
    tf = f"""### Transformers
```python
from transformers import AutoTokenizer, pipeline
tok = AutoTokenizer.from_pretrained("{merged_repo}")
pipe = pipeline("text-generation", model="{merged_repo}", tokenizer=tok, torch_dtype="auto", device_map="auto")
messages = [{{"role": "system", "content": SYSTEM_PROMPT}}, {{"role": "user", "content": user_message}}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True{', enable_thinking=False' if is_qwen else ''})
print(pipe(prompt, max_new_tokens=512, do_sample=False, return_full_text=False)[0]["generated_text"])
```
"""  # fmt: skip
    peft = f"""### Adapter LoRA (Unsloth, GPU NVIDIA)
O adapter foi treinado com a Unsloth sobre `{fam.unsloth_base}`; a forma mais simples de carregá-lo é a mesma do treino:
```python
from unsloth import {'FastLanguageModel as Loader' if is_qwen else 'FastModel as Loader'}
model, tokenizer = Loader.from_pretrained("{lora_repo}", max_seq_length=4096, load_in_4bit={'False' if is_qwen else 'True'})
Loader.for_inference(model)
```
Para usar sem a Unsloth, prefira o modelo mesclado [`{merged_repo}`](https://huggingface.co/{merged_repo}).
""" if kind == "lora" else ""  # fmt: skip

    usage = {"gguf": ollama + tf, "merged": tf + ollama, "lora": peft + tf}[kind]

    return f"""{front}
# {fam.name} (PT-BR)

Small Language Model especializado em **defaunação, refaunação, reintrodução de fauna, dispersão de sementes e conservação da Mata Atlântica**, a partir da produção científica do [Refauna](https://refauna.org.br) — iniciativa criada em 2010 que devolve vertebrados localmente extintos a remanescentes de Mata Atlântica no Rio de Janeiro. Responde em português do Brasil, **citando os trechos** recebidos (`[n]`) e **dizendo quando não encontrou** a resposta nos documentos.

Desenvolvido para a palestra *"Pequenos modelos, conhecimento especializado: fine-tuning e RAG com os dados do Refauna"* (TDC Stage @ Rec'n'Play 2026).

**English:** {fam.name.split(" ·")[0]} fine-tuned ({method}, Unsloth) in Brazilian Portuguese to answer questions about Atlantic Forest rewilding from Refauna research papers. Trained RAFT-style to use retrieved passages, cite them as `[n]` and abstain when the answer is not in the context. Use it with RAG.

| variante | repositório |
|---|---|
{variants}
{sibling}
## Como foi treinado
- **Base:** `{fam.base}` (Apache 2.0) via `{fam.unsloth_base}`.
- **Método:** {method}, r={train.get("lora_r", 16)}, alpha={train.get("lora_r", 16)}, {train.get("epochs", 2)} épocas, lr={train.get("lr", 2e-4)}, loss só na resposta (`train_on_responses_only`), {"template ChatML sem raciocínio" if is_qwen else "só camadas de linguagem, template `gemma-4` sem thinking"}.
- **Dados:** {train.get("train_examples", "?")} exemplos sintéticos em PT-BR gerados a partir de {corpus_summary(settings)}, no formato do RAG (estilo RAFT): ~60% com o trecho correto entre 3 distratores e resposta citando `[n]`, ~15% com contexto sem a resposta (abstenção), ~25% sem contexto. Deduplicados, com revisão humana amostral e **split por documento** (os documentos de teste ficam fora do treino).
- **Hardware:** {train.get("gpu", "?")}, {train.get("minutes", "?")} min, pico de {train.get("peak_vram_gb", "?")} GB de VRAM.

## Uso recomendado: com RAG
O modelo foi treinado para receber até 4 trechos numerados e citar `[n]`. Use **exatamente** este formato (diferenças no prompt pioram o modelo):

```text
system: {system_prompt}
user:   CONTEXTO:
        [1] (Sobrenome et al., ano — Título com até 60 caracteres) texto do trecho
        [2] (...) texto
        [3] (...) texto
        [4] (...) texto

        PERGUNTA: <pergunta>
```
Sem contexto, a mensagem do usuário é só a pergunta. {think_note} Busca usada no treino e na avaliação: `google/embeddinggemma-300m` + BM25, fusão RRF (k=60), top-4.

{usage}
## Avaliação
### No Colab (juiz LLM `claude-sonnet-5-5`, temperatura 0)
A = base · B = base + RAG · C = fine-tunado · D = fine-tunado + RAG. Correção e fidelidade de 1 a 5; taxas de 0 a 100%. ~60 perguntas dos documentos de teste (+ 60 na trilha de abstenção).

{table}

Recuperação híbrida no teste (n={rr.get("n", "?")}): Recall@4 = {_f(rr.get("recall@4"))}, MRR@20 = {_f(rr.get("mrr@20"))}.

{reading(colab["rows"])}

{local_section(settings, family)}
## Limitações
- Uso educacional e de divulgação científica; **não substitui a literatura original** nem orienta decisões de manejo — consulte as fontes citadas.
- Sem RAG, o modelo erra números, datas e nomes de espécies. Com RAG, a qualidade depende da busca: em ~58% das perguntas de teste o trecho certo não ficou entre os 4 recuperados.
- Conhecimento restrito a {len(read_jsonl(settings.data_dir / "docs.jsonl"))} trabalhos; não cobre todo o trabalho do Refauna.
- Avaliação com amostras pequenas e perguntas sintéticas; diferenças pequenas entre configurações são ruído.

## Dados e direitos
Os textos usados para gerar os dados de treino pertencem aos seus autores e editoras; as licenças estão em verificação. **Este repositório não contém os textos dos artigos** nem o índice de busca — só os pesos do modelo. Para usar com RAG, monte o índice com os seus próprios documentos.

## Fontes (atribuição)
{attribution(settings)}
"""


MODELFILE_HEADER = """# {name} — variante só-texto para o Ollama (sem o projetor de visão/áudio).
# Mesmo template e stops do GGUF; validado: respostas idênticas ao `ollama run hf.co/...` com temperatura 0.
# Uso: coloque este arquivo e {gguf} na mesma pasta e rode
#   ollama create {family}-refauna-texto -f Modelfile
# Não envie o parâmetro `think` na API (ele altera o prompt do Gemma 4).
"""
HF_MODELFILES = {"gemma": "gemma-4-E2B-it.Q4_K_M.gguf"}  # só onde a variante só-texto foi validada


def hf_modelfile(settings: Settings, family: str) -> str | None:
    """Modelfile publicável: o ollama/Modelfile.<família>_ft local com FROM relativo ao GGUF do repositório."""
    local = ROOT / "ollama" / f"Modelfile.{family}_ft"
    gguf = HF_MODELFILES.get(family)
    if not (gguf and local.exists()):
        return None
    body = re.sub(r"^#.*\n", "", local.read_text(), flags=re.M)
    body = re.sub(r"^FROM .+$", f"FROM ./{gguf}", body, count=1, flags=re.M)
    return MODELFILE_HEADER.format(name=settings.hf_cards[family].name, gguf=gguf, family=family) + body


def build_cards(settings: Settings, out_dir: Path = OUT_DIR) -> list[Path]:
    from refauna_slm.config import load_yaml

    system_prompt = load_yaml("prompts.yaml")["system_prompt"]
    paths = []
    for family, fam in settings.hf_cards.items():
        for kind, repo in fam.repos.items():
            path = out_dir / repo.replace("/", "__") / "README.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            text = card(settings, family, kind, system_prompt)
            path.write_text("\n".join(line.rstrip() for line in text.splitlines()).strip() + "\n")
            paths.append(path)
            mf = hf_modelfile(settings, family) if kind == "gguf" else None
            if mf:
                (path.parent / "Modelfile").write_text(mf)
                paths.append(path.parent / "Modelfile")
    return paths


def push_cards(settings: Settings, out_dir: Path = OUT_DIR, message: str = "Revisão do model card") -> list[str]:
    """Envia os README.md gerados (só depois de revisados e aprovados)."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api, urls = HfApi(), []
    for fam in settings.hf_cards.values():
        for repo in fam.repos.values():
            folder = out_dir / repo.replace("/", "__")
            ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(folder / n))
                   for n in ("README.md", "Modelfile") if (folder / n).exists()]  # fmt: skip
            try:
                info = api.create_commit(repo, ops, commit_message=message)
            except HfHubHTTPError as e:
                if e.response is None or e.response.status_code != 403:
                    raise
                # Token sem escrita no repositório (ex.: organização): abre um pull request para aprovação no site
                info = api.create_commit(repo, ops, commit_message=message, create_pr=True,
                                         commit_description="Revisão do model card gerada por `refauna cards`.")  # fmt: skip
            urls.append(str(info.pr_url or info.commit_url))
    return urls
