# Primeira versão (outubro de 2026)

Esta pasta guarda a primeira versão do projeto, como estava antes do app local. Ela fica aqui como registro; o código atual está na raiz do repositório.

## O que era

- **Modelo-base:** Gemma 4 E4B (desafiante: Qwen3.5-4B), escolhido por um teste de desempate sem treino.
- **Ingestão:** PDFs → trechos com metadados usando Docling.
- **Embeddings:** BGE-M3.
- **Banco vetorial:** Qdrant em modo local.
- **Notebooks:** `01_desempate_modelos`, `02_ingestao` e `03_rag` (índice híbrido, recall@5, respostas com citação), com as funções em `src/refauna_rag.py`.

## Por que foi substituída

1. **Folga em 8 GB.** O E4B com o BGE-M3 e o Qdrant deixava pouca memória livre no MacBook Air M1 de 8 GB durante a demo. A versão final usa modelos de ~2B (Gemma 4 E2B e Qwen3.5-2B) com fine-tuning.
2. **O mesmo embedding no Colab e no Mac.** A versão final usa o EmbeddingGemma-300m nos dois lados, para que o índice gerado no Colab seja o mesmo consultado no Mac.

As decisões estão registradas em [`docs/diario.md`](../docs/diario.md) e em [`docs/decisoes.md`](../docs/decisoes.md).

## Como rodar

Testes (sem GPU e sem rede):

```bash
cd primeira-versao
pip install -r requirements-dev.txt
pytest tests/
```

Notebooks: abra no Colab ou no Kaggle com GPU T4 e execute as células em ordem. Eles clonam o repositório e leem `refauna-slm/src` e `refauna-slm/data/avaliacao/`, os caminhos de antes da mudança para esta pasta. Para rodá-los sem ajustar nada, use o commit `fd57b89` (último da primeira versão):

```bash
git clone https://github.com/iamrodrigocoelho/refauna-slm.git
cd refauna-slm && git checkout fd57b89
```

Ou troque nos notebooks `refauna-slm/src` por `refauna-slm/primeira-versao/src` e `refauna-slm/data/avaliacao` por `refauna-slm/primeira-versao/data/avaliacao`.
