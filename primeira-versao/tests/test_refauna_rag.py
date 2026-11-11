"""Testes do módulo refauna_rag sem GPU, sem rede e sem modelos.

Usam um documento montado em código e um embedder falso (saco de palavras com hash),
para conferir catálogo, fatiamento, índice, busca, avaliação e citações.

Rodar: pip install "docling-core[chunking]" qdrant-client pandas pytest && pytest tests/
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import refauna_rag as rr  # noqa: E402

DIM = 256


def _hash(palavra: str) -> int:
    return int(hashlib.md5(palavra.encode()).hexdigest(), 16) % DIM


def embedder_falso(textos):
    densos, esparsos = [], []
    for texto in textos:
        vetor = np.zeros(DIM, dtype="float32")
        pesos = {}
        for palavra in rr._palavras(texto):
            h = _hash(palavra)
            vetor[h] += 1
            pesos[h] = pesos.get(h, 0.0) + 1.0
        norma = np.linalg.norm(vetor) or 1.0
        densos.append(vetor / norma)
        esparsos.append(pesos)
    return np.asarray(densos), esparsos


CATALOGO_CSV = """arquivo,titulo,autores,ano,especie,tipo,idioma,acesso,fonte
publico/cutias.pdf,Reintrodução de cutias,Cid B; Pires AS; Fernandez FAS,2014,cutia,artigo,pt,publico,https://exemplo/cutias
interno/bugios.pdf,Bugios na Tijuca,Genes L; Fernandez FAS,2019,bugio,artigo,pt,interno,https://exemplo/bugios
restrito/campo.pdf,Dados de campo,Refauna,2025,varias,relatorio,pt,restrito,
"""


@pytest.fixture
def pasta(tmp_path):
    (tmp_path / "documentos.csv").write_text(CATALOGO_CSV, encoding="utf-8")
    for nome in ["publico/cutias.pdf", "interno/bugios.pdf", "publico/sobrando.pdf"]:
        (tmp_path / nome).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / nome).write_bytes(b"%PDF-1.4")
    return tmp_path


def test_catalogo_e_conferencia(pasta):
    catalogo = rr.carregar_catalogo(pasta)
    assert len(catalogo) == 3
    problemas = rr.conferir_catalogo(catalogo, pasta)
    assert problemas["faltando_na_pasta"] == ["restrito/campo.pdf"]
    assert problemas["sem_linha_no_csv"] == ["publico/sobrando.pdf"]
    assert problemas["pasta_diferente_do_acesso"] == []


def test_catalogo_rejeita_acesso_invalido(tmp_path):
    (tmp_path / "documentos.csv").write_text(
        CATALOGO_CSV.replace(",publico,https://exemplo/cutias", ",aberto,https://exemplo/cutias"), encoding="utf-8")
    with pytest.raises(ValueError, match="acesso"):
        rr.carregar_catalogo(tmp_path)


# ---------- fatiamento com documento sintético ----------

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  # noqa: E402
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer  # noqa: E402
from docling_core.types.doc import DocItemLabel, DoclingDocument  # noqa: E402


class TokenizadorPalavras(BaseTokenizer):
    max_tokens: int = 60

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return None


def documento_sintetico() -> DoclingDocument:
    doc = DoclingDocument(name="cutias")
    doc.add_heading("Resultados", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text=(
        "Foram soltas onze cutias no Parque Nacional da Tijuca depois da aclimatação. "
        "Dez filhotes foram observados com quatro fêmeas reintroduzidas em cinco eventos reprodutivos."))
    doc.add_heading("Dispersão de sementes", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text=(
        "As cutias enterraram sementes de palmito-juçara e de outras espécies de sementes grandes, "
        "o que favorece o recrutamento de plântulas longe da planta-mãe."))
    doc.add_heading("References", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text=(
        "Fernandez FAS, Pires AS. Rewilding the Atlantic Forest. Perspectives in Ecology and Conservation 2017."))
    return doc


def _trechos():
    meta = rr.carregar_catalogo  # só para lembrar a origem dos campos
    meta = {"arquivo": "publico/cutias.pdf", "titulo": "Reintrodução de cutias", "autores": "Cid B; Pires AS; Fernandez FAS",
            "ano": "2014", "especie": "cutia", "tipo": "artigo", "idioma": "pt", "acesso": "publico",
            "fonte": "https://exemplo/cutias"}
    fatiador = HybridChunker(tokenizer=TokenizadorPalavras(), merge_peers=True)
    return rr.fatiar_documento(documento_sintetico(), meta, fatiador, min_caracteres=40)


def test_fatiamento_ignora_referencias_e_leva_metadados():
    trechos = _trechos()
    assert trechos, "nenhum trecho gerado"
    textos = " ".join(t["texto"] for t in trechos)
    assert "onze cutias" in textos and "palmito" in textos
    assert "Perspectives in Ecology" not in textos
    assert all(t["acesso"] == "publico" and t["ano"] == "2014" for t in trechos)
    assert all(t["texto_indexado"].startswith("Reintrodução de cutias") for t in trechos)
    assert len({t["id"] for t in trechos}) == len(trechos)


def test_salvar_e_carregar(tmp_path):
    trechos = _trechos()
    rr.salvar_trechos(trechos, tmp_path / "processado" / "trechos.jsonl")
    assert rr.carregar_trechos(tmp_path / "processado" / "trechos.jsonl") == trechos


# ---------- índice, busca, filtro de acesso e avaliação ----------

def _indice():
    trechos = _trechos() + [{
        "id": "interno-bugios-0000", "texto": "Os bugios reintroduzidos interagiram com mais de sessenta espécies vegetais.",
        "texto_indexado": "Bugios na Tijuca\nOs bugios reintroduzidos interagiram com mais de sessenta espécies vegetais.",
        "secao": "Resultados", "arquivo": "interno/bugios.pdf", "titulo": "Bugios na Tijuca",
        "autores": "Genes L; Fernandez FAS", "ano": "2019", "especie": "bugio", "tipo": "artigo",
        "idioma": "pt", "acesso": "interno", "fonte": "https://exemplo/bugios",
    }]
    return trechos, rr.criar_indice(trechos, embedder_falso, caminho=None)


def test_busca_e_filtro_de_acesso():
    _, cliente = _indice()
    trechos, similaridade = rr.buscar(cliente, embedder_falso, "Quantas espécies vegetais os bugios usaram?", k=3)
    assert trechos[0]["arquivo"] == "interno/bugios.pdf"
    assert 0 < similaridade <= 1.0001
    so_publico, _ = rr.buscar(cliente, embedder_falso, "bugios espécies vegetais", k=3, acessos=("publico",))
    assert all(t["acesso"] == "publico" for t in so_publico)


def test_avaliar_busca():
    _, cliente = _indice()
    perguntas = pd.DataFrame([
        {"id": "a", "tipo": "tecnica", "pergunta": "Quantos filhotes de cutia foram observados?",
         "trecho": "Dez filhotes foram observados com quatro fêmeas reintroduzidas", "fonte": "https://exemplo/cutias"},
        {"id": "b", "tipo": "fora_escopo", "pergunta": "Quantas onças existem?", "trecho": "", "fonte": ""},
        {"id": "c", "tipo": "tecnica", "pergunta": "Qual a dieta da anta?", "trecho": "x", "fonte": "https://nao-indexada"},
    ])
    resultado = rr.avaliar_busca(perguntas, cliente, embedder_falso, ["https://exemplo/cutias", "https://exemplo/bugios"], k=3)
    linha = resultado.set_index("id")
    assert linha.loc["a", "avaliavel"] and linha.loc["a", "encontrado"]
    assert not linha.loc["b", "avaliavel"] and not linha.loc["c", "avaliavel"]


def test_prompt_citacoes_e_recusa():
    trechos, cliente = _indice()
    msgs = rr.montar_mensagens("Pergunta?", trechos[:2], modo="pesquisador")
    assert "[1] Cid et al., 2014." in msgs[1]["content"][0]["text"]
    assert "português do Brasil" in msgs[0]["content"][0]["text"]
    assert rr.citacao_curta({"autores": "Genes L; Fernandez FAS", "ano": "2019"}) == "Genes e Fernandez, 2019"

    fontes = rr.fontes_citadas("Foram onze [1], com filhotes [1][2] e [9].", trechos[:2])
    assert len(fontes) == 1 and fontes[0].startswith("[1] Cid et al., 2014")

    chamado = []
    resultado = rr.responder("Quantas onças-pintadas existem na lua?", cliente, embedder_falso,
                             gerar=lambda m: chamado.append(m) or "x", limiar=0.99)
    assert resultado["recusa_por_limiar"] and not chamado
    assert resultado["resposta"] == rr.RESPOSTA_SEM_BASE

    resultado = rr.responder("Quantas cutias foram soltas?", cliente, embedder_falso,
                             gerar=lambda m: "Onze cutias [1].", limiar=0.0)
    assert not resultado["recusa_por_limiar"] and resultado["fontes"]
