# Decisões de arquitetura — Refauna SLM

ADRs curtos, base dos slides. Cada um: **contexto → decisão → consequência**, com os números que a sustentam.
Números de avaliação: [`avaliacao.md`](avaliacao.md) (gerado por `refauna report`).

---

## ADR-01 · Fine-tuning no formato do RAG (estilo RAFT)
- **Contexto:** modelos de 2B se perdem com contexto ruidoso; fine-tuning não memoriza fatos de forma confiável.
- **Decisão:** treinar no mesmo formato da inferência: trecho correto misturado a 3 distratores vindos da própria busca, resposta citando `[n]` (~60%); contextos sem a resposta com abstenção (~15%); perguntas sem contexto (~25%).
- **Consequência:** o fine-tuning ensina a *ler, filtrar, citar e recusar*; os fatos vêm do RAG. A avaliação confirma a divisão de trabalho (ADR-12).

## ADR-02 · Dados: manifesto, auditoria e split por documento
- **Contexto:** dos 36 PDFs em `training-files/`, só 22 tinham o conteúdo esperado (9 cópias de uma dissertação não relacionada, páginas de política de privacidade, PDFs só com título, compilados repetidos).
- **Decisão:** `sources.csv` como fonte da verdade (`incluir_texto`, `paginas`); 93 trabalhos viram **fichas do catálogo** no índice (inclusive os ~70 sem PDF); ~1.000 exemplos sintéticos (Claude Sonnet 5.5), deduplicados, com revisão humana amostral; **split por documento** (~15% dos artigos só no teste).
- **Consequência:** o teste mede generalização para fatos nunca vistos no treino — é o RAG que precisa trazê-los. As fichas permitem responder "que estudos existem sobre X?".

## ADR-03 · Extração de PDF: lições que viraram regra
- **Contexto:** o modo *layout* do pymupdf4llm 2.x transformava texto em duas colunas em tabelas quebradas; artigos da *Oryx* codificam algarismos na área de uso privado do Unicode (U+F730–F739); listas de referências sem título poluíam a busca.
- **Decisão:** modo clássico; mapa PUA → 0–9; remoção de referências por título e por densidade de citações; marcadores de página preservados.
- **Consequência:** "14 antas" não vira " antas"; cada trecho sabe sua página, o que torna a citação verificável.

## ADR-04 · Busca híbrida: EmbeddingGemma + BM25 + RRF, top-4
- **Contexto:** a busca densa entende paráfrases PT↔EN; a léxica acerta nomes científicos e siglas (*Dasyprocta leporina*, PNT).
- **Decisão:** denso top-20 + BM25 top-20 → Reciprocal Rank Fusion (k=60) → remove vizinhos ±1 do mesmo documento → top-4 (igual ao treino). Embeddings do Colab reaproveitados no Mac; Chroma reconstruído localmente.
- **Consequência:** Recall@4 = **0,419** no Mac e no Colab (idênticos), busca em ~50–75 ms no M1. É o **gargalo** do sistema: em 58% das perguntas o trecho certo não chega ao top-4 (ADR-12).

## ADR-05 · Dois modelos-base, mesmo protocolo
- **Contexto:** a palestra compara modelos abertos pequenos com os mesmos dados, prompt e juiz.
- **Decisão:** **Gemma 4 E2B com QLoRA 4-bit** (27,8 min numa L4, 10,6 GB de VRAM) e **Qwen3.5-2B com LoRA 16-bit** (14,9 min, 7,9 GB — a Unsloth não recomenda QLoRA no Qwen3.5); r=16, 2 épocas, lr 2e-4, `train_on_responses_only`.
- **Consequência:** comparação justa nas métricas de comportamento; a `eval_loss` **não** é comparável (tokenizers diferentes).

## ADR-06 · O contrato de prompt é lei (e é testado)
- **Contexto:** divergência entre o prompt de treino e o de inferência é a causa nº 1 de modelo fine-tunado "piorar" depois de exportado.
- **Decisão:** system prompt, formato do contexto e funções (`short_ref`, `format_context`, `build_messages`, `is_abstention`, `bm25_tokens`, `rrf_fuse`, `select_diverse`) **copiados** do notebook; `tests/test_prompt_contract.py` extrai a célula direto do `.ipynb` e compara texto, código e comportamento — e reproduz exatamente as mensagens dos 97 exemplos de teste.
- **Consequência:** paridade Colab × Ollama sem divergência de formato nos dois modelos: PT-BR 100% nos dois lados; presença de citações igual (Gemma 71%/71%) ou maior no Mac (Qwen 86%/71%).

## ADR-07 · Sem raciocínio — e cada família do seu jeito
- **Contexto:** o template do Qwen3.5 liga o raciocínio (`<think>`) por padrão; treinamos os dois sem raciocínio.
- **Decisão:** Qwen: `"think": false` em toda chamada + remoção de `<think>` residual. Gemma: **nenhum** parâmetro `think` — no Ollama 0.40, enviar `think: false` ao Gemma muda 2 tokens do prompt e as respostas divergem do treino.
- **Consequência:** zero vazamento de raciocínio na paridade; prompt do Gemma idêntico ao do GGUF original (5/5 respostas iguais com temperatura 0).

## ADR-08 · Inferência local: Ollama + GGUF Q4_K_M, um modelo por vez
- **Contexto:** MacBook Air M1 com 8 GB; o macOS deixa ~4,8 GiB para a GPU.
- **Decisão:** LLMs só via Ollama (nunca `transformers` no Mac); trocar de modelo descarrega o anterior (`keep_alive: 0`) **e espera ele sair da memória** antes de carregar o próximo; EmbeddingGemma em float32 na MPS.
- **Consequência:** nunca dois modelos na RAM; a aba Comparar roda as configurações em sequência.

## ADR-09 · Gemma fine-tunado só-texto (sem projetor de visão/áudio)
- **Contexto:** o GGUF do HF carrega um projetor clip de **0,99 GB** (475M parâmetros) que um projeto só de texto nunca usa; com ele, app + Gemma chegavam a **6,55 GB** em 10 perguntas seguidas (aceite: ≤ 6 GB) e o Mac entrava em swap (uma resposta levou 72 s).
- **Decisão:** variante local `refauna/gemma-4-e2b-refauna-text` gerada por `refauna models build` (PRD F8.4): mesmos pesos, template e stops, sem o projetor.
- **Consequência:** pico de **5,70 GB** ✅; 5/5 respostas idênticas ao original com temperatura 0; paridade com as mesmas métricas. O repositório do HF não muda.

## ADR-10 · Gemma original fora da demo no M1
- **Contexto:** a tag `gemma4:e2b` do Ollama é nvfp4 de 7,5 GB e exige 6,9 GiB — não cabe nos ~4,8 GiB disponíveis.
- **Decisão:** `gemma_base` desativado no `settings.yaml` (`enabled: false`); a comparação "original × fine-tunado" ao vivo usa o Qwen; os números A/B do Gemma vêm do Colab.
- **Consequência:** a avaliação local do Gemma tem só C e D. Reativar exige uma tag Q4_K_M do Gemma 4 E2B original.

## ADR-11 · Avaliar com juiz no Colab e com métricas automáticas no Mac
- **Contexto:** correção exige juiz LLM (internet, custo); a demo é offline.
- **Decisão:** Colab: juiz `claude-sonnet-5-5` (correção, fidelidade) + citação, abstenção, PT-BR. Mac: métricas automáticas (citação correta, abstenção, PT-BR, latência), juiz opcional com `--judge`. **Falsa abstenção** só conta quando o trecho certo estava no contexto — com Recall@4 de 0,42, abster-se muitas vezes é o comportamento certo.
- **Consequência:** números locais comparáveis entre si e com a paridade; notas do juiz só no Colab (n ≈ 60).

## ADR-12 · Resultado: o RAG traz os fatos, o fine-tuning traz o comportamento
- **Contexto:** hipótese da palestra: D (FT+RAG) > B (base+RAG) > C (FT) > A (base).
- **Resultado (Colab, correção 1–5):** Gemma A 1,12 · B 2,47 · C 2,17 · D 2,29; Qwen A 1,72 · B 2,52 · C 2,20 · D 2,32. Ordem observada **B > D > C > A** nos dois — a hipótese **não** se confirma em correção.
- **Leitura:** o maior salto é o RAG (A → B). O fine-tuning não aumentou a correção com RAG porque ela fica limitada pela recuperação (Recall@4 ≈ 0,42). Ele mudou o **comportamento**: no Qwen, fidelidade 3,17 → 4,41, abstenção correta 12% → 100%, citação correta 33% → 45% e latência 14,3 → 4,5 s; no Gemma, que já se abstinha bem como base (88%), o efeito foi pequeno (fidelidade 4,35 → 4,17, latência 13,0 → 11,3 s).
- **Confirmado no Mac** (Ollama, 40 perguntas, métricas automáticas): Qwen de B para D — citação correta 17% → 54%, abstenção correta 12% → 100% (8 perguntas), respostas de 270 → 70 tokens e latência 17,2 → 5,9 s; custo: falsa abstenção 0% → 14% (2 de 14 perguntas com o trecho certo no contexto). Gemma D: citação correta 58%, abstenção correta 75%, falsa abstenção 0%.
- **Consequência:** o próximo ganho de correção vem da **busca** (reranker, chunking, mais documentos), não de mais treino. Resultado negativo também é conteúdo — é o slide mais honesto da palestra.

## ADR-13 · Medir memória com `footprint`, não com RSS
- **Contexto:** no Apple Silicon, pesos mapeados do disco (mmap) e buffers Metal escapam do RSS; o `ollama ps` reportava 0,24 GB para um Gemma de 3,4 GB.
- **Decisão:** `phys_footprint` (`/usr/bin/footprint`) para app e Ollama; medir com prompts de RAG reais (um prompt curto não carrega o modelo inteiro).
- **Consequência:** os números de RAM do relatório refletem o que derruba a demo; o critério de 6 GB é verificável (`scripts/ram_demo.py`).

## ADR-14 · Governança dos dados
- **Contexto:** licenças dos artigos "a verificar"; todos `redistributable = false`.
- **Decisão:** dataset `refauna-sft-ptbr` privado; nenhum trecho de artigo em repositório público (`data/` fora do git); atribuição com DOI no app (aba Resultados) e nos model cards; e-mails de autores removidos na limpeza.
- **Consequência:** modelos podem ser públicos; índice e dataset só depois da verificação de licenças e do "ok" escrito do Refauna.

## ADR-15 · Modelo principal da demo: Qwen3.5-2B · Refauna — **aceita em 02/11/2026** (PRD §14.5)
- **Contexto:** os dois fine-tunados rodam no M1; a demo precisa de respostas rápidas e de uma abstenção confiável ao vivo.
- **Números:** Qwen FT — 30 tokens/s, latência p50 5,7 s, 1,99 GB; Colab D: fidelidade 4,41, abstenção 100%; Mac D: abstenção 100%, citação correta 54%. Gemma FT — 21 tokens/s, p50 8,6 s, 3,43 GB; Colab D: fidelidade 4,17, abstenção 88%; Mac D: abstenção 75%, citação correta 58%, sem falsa abstenção. Correção (juiz) empatada: 2,32 × 2,29.
- **Decisão:** **Qwen3.5-2B · Refauna como modelo principal** (Chat), por velocidade, memória e abstenção; Gemma FT na aba Comparar, onde a citação um pouco melhor dele aparece lado a lado.
- **Consequência:** o app abre no Qwen FT (`demo.default_model` no `settings.yaml`) e `refauna ask` usa ele por padrão; a pergunta de abstenção da demo ("quantas onças-pintadas…") fica com o modelo que se abstém em 100% dos casos medidos; sobra memória para o navegador durante a palestra.

---

## Em aberto
- **Qwen na organização `Refauna`** e Collection da organização (PRD F8.1–F8.2).
- **Qwen original em Q4_K_M** para igualar a quantização com o fine-tunado (hoje mxfp8, 3,1 GB).
