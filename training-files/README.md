# training-files

Coloque aqui (ou em `MyDrive/refauna-slm/training-files/`, no Colab) os PDFs que vão alimentar o RAG e o treino, junto com um `sources.csv` que descreve cada um. Os PDFs **não são versionados**.

`sources.example.csv` mostra o formato com dois artigos do Refauna. As colunas que mais importam:

| Coluna | Para que serve |
|---|---|
| `doc_id` | Identificador único do documento (ex.: `REF-L1-004`) |
| `filename` | Nome do PDF nesta pasta (vazio se você só tem os metadados) |
| `incluir_texto` | `sim` para extrair o texto do PDF; `nao` para usar só a ficha de catálogo |
| `paginas` | Intervalo de páginas a usar (ex.: `43-46`), útil para capítulos de livros e boletins; vazio = todas |
| `title`, `authors`, `year`, `doi` | Metadados usados nas citações e no catálogo |
| `license`, `redistributable` | Licença do documento e se o texto pode ser redistribuído (`true`/`false`) |

As demais colunas (espécies, projeto, local etc.) entram nas fichas do catálogo e ajudam a responder perguntas como "que estudos existem sobre a cutia?".
