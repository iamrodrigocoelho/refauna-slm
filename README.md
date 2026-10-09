# refauna-slm

Modelo de linguagem pequeno e aberto, especializado em refaunação e conservação da Mata Atlântica com os dados de pesquisa do [Refauna](https://refauna.org.br).

O projeto faz o fine-tuning do **Gemma 4 E4B** com **QLoRA** e o combina com um **RAG** sobre os documentos do Refauna:

- o **RAG** fornece os fatos e permite citar a fonte;
- o **fine-tuning** ensina o comportamento: responder em português a partir de fontes em inglês, usar o vocabulário da área, citar o trecho usado e dizer que não sabe quando o acervo não cobre a pergunta.

Tudo é treinado em GPU gratuita (T4 de 16 GB, no Colab ou no Kaggle) e roda localmente, sem internet.

> Projeto em desenvolvimento, apresentado no TDC Stage at Rec'n'Play 2026 (11 de novembro), na palestra "Pequenos modelos, conhecimento especializado: fine-tuning e RAG com os dados do Refauna".

## Stack

| Camada | Escolha |
| --- | --- |
| Modelo-base | Gemma 4 E4B (desafiante: Qwen3.5-4B) |
| Fine-tuning | QLoRA em 4 bits com Unsloth (sobre PEFT e TRL) |
| Ingestão de PDFs | Docling |
| Embeddings | BGE-M3 |
| Banco vetorial | Qdrant em modo local |
| Inferência local | GGUF com llama.cpp ou Ollama |
| Publicação | Hugging Face |

## Estrutura

```
refauna-slm/
├── data/
│   ├── README.md              regras de governança dos dados
│   └── avaliacao/             perguntas de avaliação (só material público)
├── notebooks/
│   ├── 01_desempate_modelos.ipynb   Gemma 4 E4B contra Qwen3.5-4B, sem treino
│   ├── 02_ingestao.ipynb            PDFs → trechos com metadados (Docling)
│   └── 03_rag.ipynb                 índice híbrido, recall@5 e respostas com citação
├── src/
│   └── refauna_rag.py         funções do RAG usadas pelos notebooks
├── tests/                     testes do RAG, sem GPU e sem rede
├── docs/
│   ├── diario.md              decisões, tropeços e lições
│   ├── artigos_refauna.csv    fontes encontradas, com acesso e licença
│   └── documentos_modelo.csv  modelo do documentos.csv que fica no Drive
└── README.md
```

Próximos notebooks: `04_dataset_treino`, `05_finetune_qlora`, `06_avaliacao`.

## Testes

```
pip install -r requirements-dev.txt
pytest tests/
```

## Roteiro

- [ ] Semana 1 (até 14/10): dados levantados, teste de desempate, modelo-base escolhido
- [ ] Semana 2 (até 21/10): ingestão, RAG com o modelo-base, conjunto de avaliação completo
- [ ] Semana 3 (até 28/10): dataset de treino e primeiro fine-tune
- [ ] Semana 4 (até 4/11): avaliação das quatro configurações e publicação no Hugging Face
- [ ] Semana 5 (até 10/11): demo e ensaio

## Dados

Os documentos e dados de campo do Refauna **não** ficam neste repositório. Veja [`data/README.md`](data/README.md).

## Como rodar

Cada notebook abre direto no Colab ou no Kaggle com GPU T4. Ative a GPU em *Ambiente de execução › Alterar o tipo de ambiente de execução* (Colab) ou em *Settings › Accelerator* (Kaggle) e execute as células em ordem.
