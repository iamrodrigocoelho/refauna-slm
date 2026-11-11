"""RAG do refauna-slm: catálogo, ingestão, índice, busca e prompt.

Os notebooks 02_ingestao e 03_rag importam este módulo. As dependências pesadas
(docling, transformers, torch, qdrant-client) são importadas dentro das funções
que as usam, para o módulo carregar mesmo onde elas não estão instaladas.
"""
from __future__ import annotations

import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

ACESSOS = ("publico", "interno", "restrito")
COLUNAS_CATALOGO = ["arquivo", "titulo", "autores", "ano", "especie", "tipo", "idioma", "acesso", "fonte"]
EXTENSOES = {".pdf", ".html", ".htm", ".md", ".docx"}
PASTAS_IGNORADAS = {"processado"}

# Seções de artigo que só atrapalham a busca.
SECOES_IGNORADAS = re.compile(
    r"^\s*(\d+[.)]?\s*)?(references|referências|referencias|literature cited|bibliography|bibliografia|"
    r"acknowledg|agradecimentos|supplementary|supporting information|material suplementar|"
    r"author contributions|conflict of interest|data availability|funding)",
    re.IGNORECASE,
)

COLECAO = "refauna"
RESPOSTA_SEM_BASE = (
    "Não encontrei no acervo do Refauna informação suficiente para responder a essa pergunta."
)


# --------------------------------------------------------------------------
# Catálogo (documentos.csv)
# --------------------------------------------------------------------------

def carregar_catalogo(pasta_dados: str | Path) -> pd.DataFrame:
    """Lê dados/documentos.csv e confere colunas, rótulos de acesso e duplicatas."""
    caminho = Path(pasta_dados) / "documentos.csv"
    catalogo = pd.read_csv(caminho, dtype=str).fillna("")
    catalogo.columns = [c.strip() for c in catalogo.columns]
    for coluna in catalogo.columns:
        catalogo[coluna] = catalogo[coluna].str.strip()

    faltam = [c for c in COLUNAS_CATALOGO if c not in catalogo.columns]
    if faltam:
        raise ValueError(f"documentos.csv sem as colunas: {', '.join(faltam)}")

    invalidos = catalogo.loc[~catalogo["acesso"].isin(ACESSOS), "arquivo"].tolist()
    if invalidos:
        raise ValueError(f"Rótulo de acesso inválido (use {', '.join(ACESSOS)}): {invalidos}")

    duplicados = catalogo.loc[catalogo["arquivo"].duplicated(), "arquivo"].tolist()
    if duplicados:
        raise ValueError(f"Arquivos repetidos no documentos.csv: {duplicados}")
    return catalogo


def conferir_catalogo(catalogo: pd.DataFrame, pasta_dados: str | Path) -> dict[str, list[str]]:
    """Compara o catálogo com os arquivos da pasta. Devolve listas de problemas."""
    pasta = Path(pasta_dados)
    no_disco = {
        p.relative_to(pasta).as_posix()
        for p in pasta.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSOES
        and not PASTAS_IGNORADAS.intersection(p.relative_to(pasta).parts)
    }
    no_csv = set(catalogo["arquivo"])
    pasta_diferente = [
        linha.arquivo
        for linha in catalogo.itertuples()
        if "/" in linha.arquivo and linha.arquivo.split("/", 1)[0] != linha.acesso
    ]
    return {
        "faltando_na_pasta": sorted(no_csv - no_disco),
        "sem_linha_no_csv": sorted(no_disco - no_csv),
        "pasta_diferente_do_acesso": sorted(pasta_diferente),
    }


# --------------------------------------------------------------------------
# Ingestão: Docling converte, HybridChunker fatia
# --------------------------------------------------------------------------

def criar_conversor():
    """Conversor do Docling para PDF, HTML, Markdown e DOCX."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def criar_fatiador(max_tokens: int = 450, modelo_tokenizador: str = "BAAI/bge-m3"):
    """Fatiador do Docling que respeita seções e conta tokens com o tokenizador do BGE-M3."""
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tokenizador = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(modelo_tokenizador),
        max_tokens=max_tokens,
    )
    return HybridChunker(tokenizer=tokenizador, merge_peers=True)


def _secao_ignorada(titulos: Iterable[str]) -> bool:
    return any(SECOES_IGNORADAS.match(t or "") for t in titulos)


def _slug(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def fatiar_documento(documento, meta: dict, fatiador, min_caracteres: int = 80) -> list[dict]:
    """Transforma um DoclingDocument em trechos com os metadados do catálogo."""
    trechos = []
    base = _slug(meta["arquivo"])
    for ordem, pedaco in enumerate(fatiador.chunk(dl_doc=documento)):
        titulos = list(getattr(pedaco.meta, "headings", None) or [])
        if _secao_ignorada(titulos):
            continue
        texto = pedaco.text.strip()
        if len(texto) < min_caracteres:
            continue
        contexto = fatiador.contextualize(chunk=pedaco)
        trechos.append({
            "id": f"{base}-{ordem:04d}",
            "texto": texto,
            # O título do documento entra no texto indexado para ajudar a busca.
            "texto_indexado": f"{meta['titulo']}\n{contexto}",
            "secao": " > ".join(titulos),
            **{c: meta.get(c, "") for c in COLUNAS_CATALOGO},
        })
    return trechos


def ingerir(
    catalogo: pd.DataFrame,
    pasta_dados: str | Path,
    conversor,
    fatiador,
    acessos: Iterable[str] = ("publico", "interno"),
) -> tuple[list[dict], pd.DataFrame]:
    """Converte e fatia cada documento do catálogo. Documentos 'restrito' nunca entram."""
    acessos = set(acessos) - {"restrito"}
    pasta = Path(pasta_dados)
    todos, relatorio = [], []
    for meta in catalogo.to_dict("records"):
        if meta["acesso"] not in acessos:
            relatorio.append({"arquivo": meta["arquivo"], "trechos": 0, "situacao": f"ignorado ({meta['acesso']})"})
            continue
        caminho = pasta / meta["arquivo"]
        if not caminho.exists():
            relatorio.append({"arquivo": meta["arquivo"], "trechos": 0, "situacao": "arquivo não encontrado"})
            continue
        try:
            documento = conversor.convert(str(caminho)).document
            trechos = fatiar_documento(documento, meta, fatiador)
        except Exception as erro:  # um PDF ruim não deve parar a ingestão inteira
            relatorio.append({"arquivo": meta["arquivo"], "trechos": 0, "situacao": f"erro: {erro}"})
            continue
        todos.extend(trechos)
        relatorio.append({"arquivo": meta["arquivo"], "trechos": len(trechos), "situacao": "ok"})
    return todos, pd.DataFrame(relatorio)


def salvar_trechos(trechos: list[dict], caminho: str | Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for trecho in trechos:
            arquivo.write(json.dumps(trecho, ensure_ascii=False) + "\n")


def carregar_trechos(caminho: str | Path) -> list[dict]:
    with Path(caminho).open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


# --------------------------------------------------------------------------
# Embeddings: BGE-M3, vetores densos e esparsos
# --------------------------------------------------------------------------

class EmbedderBGEM3:
    """BGE-M3 com transformers puro: vetor denso (CLS normalizado) e esparso (pesos por token).

    Reproduz a parte densa e esparsa do BGEM3FlagModel sem depender do pacote FlagEmbedding.
    """

    def __init__(self, modelo: str = "BAAI/bge-m3", dispositivo: str | None = None,
                 fp16: bool = True, max_length: int = 1024):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizador = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModel.from_pretrained(modelo).to(self.dispositivo).eval()
        self.camada_esparsa = torch.nn.Linear(self.modelo.config.hidden_size, 1).to(self.dispositivo)
        pesos = torch.load(hf_hub_download(modelo, "sparse_linear.pt"), map_location=self.dispositivo)
        self.camada_esparsa.load_state_dict(pesos)
        if fp16 and self.dispositivo == "cuda":
            self.modelo.half()
            self.camada_esparsa.half()
        self.especiais = set(self.tokenizador.all_special_ids)

    def __call__(self, textos: list[str], lote: int = 16):
        import numpy as np

        torch = self.torch
        densos, esparsos = [], []
        for inicio in range(0, len(textos), lote):
            entrada = self.tokenizador(
                textos[inicio:inicio + lote], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.dispositivo)
            with torch.no_grad():
                oculto = self.modelo(**entrada).last_hidden_state
                denso = torch.nn.functional.normalize(oculto[:, 0].float(), dim=-1)
                peso = torch.relu(self.camada_esparsa(oculto)).squeeze(-1).float()
            for ids, pesos, mascara in zip(entrada["input_ids"].tolist(), peso.tolist(),
                                           entrada["attention_mask"].tolist()):
                vetor: dict[int, float] = {}
                for token, valor, ativo in zip(ids, pesos, mascara):
                    if ativo and valor > 0 and token not in self.especiais:
                        vetor[token] = max(vetor.get(token, 0.0), valor)
                esparsos.append(vetor)
            densos.extend(denso.cpu().numpy())
        return np.asarray(densos, dtype="float32"), esparsos


# --------------------------------------------------------------------------
# Índice e busca no Qdrant (modo local, sem servidor)
# --------------------------------------------------------------------------

def _id_ponto(id_trecho: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, id_trecho))


def _vetor_esparso(pesos: dict[int, float]):
    from qdrant_client import models

    indices = list(pesos.keys())
    return models.SparseVector(indices=indices, values=[float(pesos[i]) for i in indices])


def criar_indice(trechos: list[dict], embedder: Callable, caminho: str | Path | None = None,
                 colecao: str = COLECAO, lote: int = 32):
    """Cria (ou recria) a coleção com vetores densos e esparsos de cada trecho.

    caminho=None cria o índice só na memória, útil para testes.
    """
    from qdrant_client import QdrantClient, models

    cliente = QdrantClient(path=str(caminho)) if caminho else QdrantClient(location=":memory:")
    if cliente.collection_exists(colecao):
        cliente.delete_collection(colecao)

    colecao_criada = False
    for inicio in range(0, len(trechos), lote):
        parte = trechos[inicio:inicio + lote]
        densos, esparsos = embedder([t["texto_indexado"] for t in parte])
        if not colecao_criada:
            cliente.create_collection(
                colecao,
                vectors_config={"denso": models.VectorParams(size=len(densos[0]), distance=models.Distance.COSINE)},
                sparse_vectors_config={"esparso": models.SparseVectorParams()},
            )
            colecao_criada = True
        cliente.upsert(colecao, points=[
            models.PointStruct(
                id=_id_ponto(t["id"]),
                vector={"denso": [float(x) for x in d], "esparso": _vetor_esparso(e)},
                payload={k: v for k, v in t.items() if k != "texto_indexado"},
            )
            for t, d, e in zip(parte, densos, esparsos)
        ])
    return cliente


def _filtro(acessos: Iterable[str], especie: str | None):
    from qdrant_client import models

    condicoes = [models.FieldCondition(key="acesso", match=models.MatchAny(any=list(acessos)))]
    if especie:
        condicoes.append(models.FieldCondition(key="especie", match=models.MatchAny(any=[especie, "varias"])))
    return models.Filter(must=condicoes)


def buscar(cliente, embedder: Callable, pergunta: str, k: int = 5, candidatos: int = 20,
           acessos: Iterable[str] = ("publico", "interno"), especie: str | None = None,
           colecao: str = COLECAO) -> tuple[list[dict], float]:
    """Busca híbrida (densa + esparsa, fundidas por RRF).

    Devolve os k trechos e a maior similaridade densa, usada para decidir a recusa.
    """
    from qdrant_client import models

    densos, esparsos = embedder([pergunta])
    denso = [float(x) for x in densos[0]]
    filtro = _filtro(acessos, especie)
    resposta = cliente.query_points(
        colecao,
        prefetch=[
            models.Prefetch(query=denso, using="denso", limit=candidatos, filter=filtro),
            models.Prefetch(query=_vetor_esparso(esparsos[0]), using="esparso", limit=candidatos, filter=filtro),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    melhor = cliente.query_points(colecao, query=denso, using="denso", limit=1,
                                  query_filter=filtro, with_payload=False).points
    similaridade = float(melhor[0].score) if melhor else 0.0
    trechos = [{**ponto.payload, "pontuacao": float(ponto.score)} for ponto in resposta.points]
    return trechos, similaridade


# --------------------------------------------------------------------------
# Avaliação da busca
# --------------------------------------------------------------------------

def _palavras(texto: str) -> set[str]:
    texto = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return {p for p in re.findall(r"[a-z0-9]+", texto) if len(p) > 2}


def contem_trecho(esperado: str, texto: str, limiar: float = 0.6) -> bool:
    """O trecho recuperado contém o trecho esperado? Mede a fração de palavras em comum."""
    alvo = _palavras(esperado)
    if not alvo:
        return False
    return len(alvo & _palavras(texto)) / len(alvo) >= limiar


def avaliar_busca(perguntas: pd.DataFrame, cliente, embedder: Callable, fontes_indexadas: Iterable[str],
                  k: int = 5, **kwargs) -> pd.DataFrame:
    """Recall@k: para cada pergunta cuja fonte está no índice, o trecho esperado apareceu entre os k primeiros?

    Também registra a similaridade densa de todas as perguntas, inclusive as fora do escopo,
    para calibrar o limiar de recusa.
    """
    fontes = set(fontes_indexadas)
    linhas = []
    for p in perguntas.to_dict("records"):
        trechos, similaridade = buscar(cliente, embedder, p["pergunta"], k=k, **kwargs)
        avaliavel = p["tipo"] != "fora_escopo" and p.get("fonte", "") in fontes
        posicao = next((i for i, t in enumerate(trechos, 1) if contem_trecho(p["trecho"], t["texto"])), None)
        linhas.append({
            "id": p["id"], "tipo": p["tipo"], "avaliavel": avaliavel,
            "encontrado": bool(posicao) if avaliavel else None,
            "posicao": posicao if avaliavel else None,
            "similaridade": round(similaridade, 3),
            "primeira_fonte": trechos[0]["arquivo"] if trechos else "",
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Prompt, citações e resposta
# --------------------------------------------------------------------------

SISTEMA = {
    "publico": (
        "Você é um assistente do Refauna, iniciativa que reintroduz animais nativos na Mata Atlântica. "
        "Explique em linguagem simples, para quem não é da área, em poucas frases."
    ),
    "pesquisador": (
        "Você é um assistente técnico do Refauna para pesquisadores em ecologia e conservação. "
        "Use a terminologia da área e seja preciso com números, espécies e locais."
    ),
}

REGRAS = (
    " Responda sempre em português do Brasil, mesmo quando os trechos estiverem em inglês. "
    "Use apenas as informações dos trechos numerados. Indique a fonte de cada afirmação com o número "
    "do trecho entre colchetes, como [1]. Se os trechos não trouxerem a resposta, diga que o acervo "
    "consultado não cobre a pergunta e não invente."
)


def citacao_curta(trecho: dict) -> str:
    autores = [a.strip() for a in re.split(r";", trecho.get("autores", "")) if a.strip()]
    if not autores:
        primeiro = "Refauna"
    else:
        primeiro = autores[0].split()[0].rstrip(",")
    sufixo = " et al." if len(autores) > 2 or "et al" in trecho.get("autores", "") else (
        f" e {autores[1].split()[0]}" if len(autores) == 2 else "")
    ano = trecho.get("ano") or "s.d."
    return f"{primeiro}{sufixo}, {ano}"


def montar_mensagens(pergunta: str, trechos: list[dict], modo: str = "publico") -> list[dict]:
    blocos = [
        f"[{i}] {citacao_curta(t)}. {t.get('titulo', '')}\n{t['texto']}"
        for i, t in enumerate(trechos, 1)
    ]
    usuario = "Trechos:\n\n" + "\n\n".join(blocos) + f"\n\nPergunta: {pergunta}"
    return [
        {"role": "system", "content": [{"type": "text", "text": SISTEMA[modo] + REGRAS}]},
        {"role": "user", "content": [{"type": "text", "text": usuario}]},
    ]


def fontes_citadas(resposta: str, trechos: list[dict]) -> list[str]:
    """Lista, sem repetir, as fontes que a resposta citou como [n]."""
    numeros = sorted({int(n) for n in re.findall(r"\[(\d+)\]", resposta)})
    fontes, vistas = [], set()
    for n in numeros:
        if 1 <= n <= len(trechos):
            t = trechos[n - 1]
            if t["arquivo"] in vistas:
                continue
            vistas.add(t["arquivo"])
            fontes.append(f"[{n}] {citacao_curta(t)}. {t.get('titulo', '')}. {t.get('fonte', '')}".strip())
    return fontes


def responder(pergunta: str, cliente, embedder: Callable, gerar: Callable[[list[dict]], str],
              modo: str = "publico", limiar: float = 0.45, k: int = 5, **kwargs) -> dict:
    """Busca, decide se há base para responder e chama o modelo.

    `gerar` recebe as mensagens e devolve o texto: é onde entra o Gemma ou o Qwen.
    """
    trechos, similaridade = buscar(cliente, embedder, pergunta, k=k, **kwargs)
    resultado = {"pergunta": pergunta, "modo": modo, "similaridade": round(similaridade, 3),
                 "trechos": [t["id"] for t in trechos]}
    if not trechos or similaridade < limiar:
        return {**resultado, "resposta": RESPOSTA_SEM_BASE, "fontes": [], "recusa_por_limiar": True}
    texto = gerar(montar_mensagens(pergunta, trechos, modo))
    return {**resultado, "resposta": texto, "fontes": fontes_citadas(texto, trechos), "recusa_por_limiar": False}
