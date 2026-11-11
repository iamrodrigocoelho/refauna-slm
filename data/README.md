# Dados

Os documentos e dados de campo do Refauna **não são versionados** neste repositório. O `.gitignore` bloqueia tudo em `data/`, exceto este arquivo. As perguntas de avaliação da primeira versão estão em `primeira-versao/data/avaliacao/`.

## Onde ficam

| Tipo | Onde guardar |
| --- | --- |
| PDFs, relatórios, teses | Google Drive ou dataset privado no Kaggle |
| Dados de campo | Google Drive, já agregados |
| PDFs que alimentam o RAG e o treino | `training-files/` (local) ou `MyDrive/refauna-slm/training-files/` (Colab), não versionados |
| Ficha de cada documento | `training-files/sources.csv` (modelo em [`training-files/sources.example.csv`](../training-files/sources.example.csv)) |

## Pasta no Google Drive

Os notebooks do Colab esperam esta estrutura em `MyDrive/refauna-slm/`:

```
refauna-slm/
├── training-files/
│   ├── sources.csv     uma linha por documento (modelo em training-files/sources.example.csv)
│   └── *.pdf           os documentos
├── data/               criada pelos notebooks (trechos, índice, conjunto de treino)
└── outputs/            criada pelos notebooks (adaptadores, avaliações, gráficos)
```

As colunas do `sources.csv` estão descritas em [`training-files/README.md`](../training-files/README.md). As que mais importam: `doc_id`, `filename`, `incluir_texto` (`sim` extrai o texto do PDF; `nao` usa só a ficha de catálogo), `paginas`, `license` e `redistributable`.

No Mac, o app local lê a mesma pasta pelo Google Drive para desktop (variável `REFAUNA_DRIVE_DIR` no `.env`) ou uma cópia local em `data/`.

Não é preciso compartilhar a pasta por link: o Colab lê direto do seu Drive. Documentos internos nunca devem ficar em pasta pública.

## Rótulos de acesso

Cada documento recebe um rótulo antes de entrar no projeto:

- **público**: pode ir para o índice, para o treino e para o Hugging Face;
- **interno**: pode ir para o índice local, mas não para o treino do modelo publicado;
- **restrito**: fica fora do projeto.

## Regras

1. Coordenadas, trilhas e pontos de soltura nunca entram no índice nem no treino, pelo risco de caça.
2. Dados de teses em andamento só entram com autorização do autor e do orientador.
3. A versão publicada do modelo é treinada apenas com material público.
4. Nada é publicado sem revisão do time do Refauna.
