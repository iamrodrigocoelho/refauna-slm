# CLAUDE.md — Refauna SLM (aplicação local)

## Contexto
- Projeto da palestra do TDC Stage @ Rec'n'Play 2026 (11/11/2026). Requisitos completos em `docs/PRD.md` (versão pública) — leia antes de mudar algo relevante.
- Dados, dataset, treino, avaliação A/B/C/D e publicação **já foram feitos no Google Colab** (`notebooks/`). Este repositório é **a aplicação local**: RAG offline, CLI, app Gradio, avaliação local e relatórios.
- Hardware: **MacBook Air M1, 8 GB de RAM**. **Nunca** propor treino local nem carregar LLMs com `transformers`. LLMs só via **Ollama**; embeddings via `sentence-transformers` (EmbeddingGemma, float32, MPS, batch ≤ 16).

## Regras
1. **Não reimplementar** ingestão de PDFs, chunking, geração de dataset ou treino. Consumir os artefatos do Colab listados no PRD §4.2 (copiados para `data/` por `refauna sync`).
2. **O contrato de prompt (PRD §4.1) é lei.** `SYSTEM_PROMPT`, `short_ref`, `format_context`, `build_user_message`, `build_messages`, `is_abstention`, `is_ptbr`, `bm25_tokens`, `rrf_fuse`, `select_diverse` são **copiados do notebook**, nunca reescritos. O texto do system prompt vive em `config/prompts.yaml`; `tests/test_prompt_contract.py` extrai a célula direto do `.ipynb` e compara. TOP_K = 4.
3. **Qwen:** sempre `"think": false` na API do Ollama e remover `<think>…</think>` residual.
4. **Um modelo por vez** no Ollama: ao trocar de modelo, descarregar o anterior (`keep_alive: 0`).
5. **Repositórios do HF / tags do Ollama só em `config/settings.yaml`** (um por modelo). Nenhum nome de repositório fixo no código. **Não usar `Refauna/tapirus-open-4b`.**
6. **Fases** na ordem do cronograma (PRD §11). Ao fim de cada fase: `make lint`, `make test`, mostrar o critério de aceite e fazer commit convencional (`feat:`, `fix:`, `docs:`).
7. **Nunca versionar** `data/`, PDFs, `*.gguf`, `.env`, `data/chroma/`.
8. **Não alterar** os notebooks em `notebooks/` sem pedir.
9. **Perguntar antes de:** publicar ou alterar algo no Hugging Face, apagar dados, mudar modelos ou o modelo de embeddings (`google/embeddinggemma-300m`, o mesmo do Colab).

## Comandos
```bash
make setup                    # uv sync --all-extras (Python 3.11)
uv run refauna sync           # Drive -> data/ + validação (F1)
uv run refauna models pull    # baixa os modelos e gera a variante só-texto do Gemma (F2, com internet)
uv run refauna models build   # recria a variante só-texto offline (ollama/Modelfile.gemma_ft)
uv run refauna models check [--smoke] [-m qwen]   # confirma disponibilidade offline (F2)
uv run refauna index [--rebuild]                  # Chroma em data/chroma, reaproveita chunk_embeddings.npy (F3)
uv run refauna ask "pergunta" -m qwen_ft [--no-rag] [--show-context]   # F4
uv run refauna eval --retrieval                   # Recall@4 local × Colab
uv run refauna eval --parity [-m gemma_ft]        # F6.1, resultados em data/local_eval/
uv run refauna eval [--configs B,D] [--n 40] [--judge]   # F6.2 A/B/C/D local (~40 min com os 3 modelos)
uv run refauna bench                              # F6.3 latência, tokens/s, tamanho no M1
uv run refauna report                             # F6.4 docs/avaliacao.md + docs/img/*.png (não editar à mão)
uv run refauna serve                              # app Gradio (F5): Chat · Comparar · Resultados
uv run python scripts/ram_demo.py                 # RAM em 10 perguntas seguidas no app rodando
make lint && make test        # ruff + pytest
```

## Estilo
- Python 3.11, `uv`, `ruff`, `pytest`, type hints, docstrings curtas em PT-BR. Mensagens da CLI em PT-BR.
- Configuração: `src/refauna_slm/config.py` (pydantic-settings: `.env` + `config/settings.yaml` + `config/prompts.yaml`).
- Respostas do assistente sempre em português do Brasil. Seed fixa 3407.
- Memória no M1: medir com `footprint` (macOS), não só RSS — pesos via mmap e buffers Metal escapam do RSS.
