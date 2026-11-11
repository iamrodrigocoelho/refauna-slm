"""Juiz LLM opcional (`refauna eval --judge`): o MESMO prompt e o mesmo modelo do Colab.

Requer internet e credenciais da Anthropic (ANTHROPIC_API_KEY ou `ant auth login`).
"""

from __future__ import annotations

import json
import re

from refauna_slm.contract import format_context

JUDGE_SYSTEM = "Você é um avaliador rigoroso de respostas sobre ecologia e conservação. Responda apenas com JSON."


def parse_json(txt):
    """Copiado do notebook."""
    if not txt:
        return None
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(txt[a : b + 1])
    except json.JSONDecodeError:
        return None


def judge_prompt(question, reference, answer, contexts=None):
    """Copiado do notebook (seção 7)."""
    ctx_block = f"\nCONTEXTO DISPONÍVEL AO MODELO:\n{format_context(contexts)}\n" if contexts else ""
    return f"""PERGUNTA: {question}

RESPOSTA DE REFERÊNCIA: {reference}
{ctx_block}
RESPOSTA DO MODELO: {answer}

Avalie:
- "correcao" (1-5): a resposta do modelo transmite o mesmo conteúdo factual da referência? 5 = equivalente e completa; 3 = parcialmente correta; 1 = errada, irrelevante ou em outro idioma.
- "fidelidade" (1-5 ou null): {"tudo o que o modelo afirma está apoiado no CONTEXTO? 5 = nada inventado; 1 = muitas informações sem apoio." if contexts else "use null (não havia contexto)."}
Responda só com JSON: {{"correcao": int, "fidelidade": int|null, "comentario": "uma frase"}}"""


class Judge:
    def __init__(self, model: str):
        import anthropic  # dependência opcional: uv sync --extra judge

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model

    def __call__(self, question, reference, answer, contexts=None) -> dict:
        try:
            r = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2000,  # pensamento adaptativo conta aqui; a resposta em si é um JSON curto
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": judge_prompt(question, reference, answer, contexts)}],
            )
        except self.anthropic.APIStatusError as e:
            return {"erro": f"{e.status_code}: {e.message}"[:200]}
        except self.anthropic.APIConnectionError:
            return {"erro": "sem conexão com a API da Anthropic"}
        if r.stop_reason == "refusal":
            return {"erro": "recusa do juiz"}
        text = "".join(b.text for b in r.content if b.type == "text")
        return parse_json(text) or {"erro": "JSON inválido do juiz"}
