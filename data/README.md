# Dados

Os documentos e dados de campo do Refauna **não são versionados** neste repositório. O `.gitignore` bloqueia tudo em `data/`, exceto este arquivo e os CSVs de `data/avaliacao/`.

## Onde ficam

| Tipo | Onde guardar |
| --- | --- |
| PDFs, relatórios, teses | Google Drive ou dataset privado no Kaggle |
| Dados de campo | Google Drive, já agregados |
| Perguntas de avaliação feitas com material público | `data/avaliacao/` (versionado) |

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
