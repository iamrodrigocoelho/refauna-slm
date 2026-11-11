"""Contrato de prompt (PRD §4.1) — cópia da célula "Contrato de prompt" do notebook do Gemma.

Não edite as funções: qualquer divergência do treino piora o modelo fine-tunado.
`tests/test_prompt_contract.py` compara este módulo com o .ipynb.
"""

import re

from refauna_slm.config import load_yaml

SYSTEM_PROMPT = load_yaml("prompts.yaml")["system_prompt"]

ABSTAIN_RE = re.compile(
    r"não (encontrei|consigo responder|é possível responder|há informaç|aparece)|não contêm a resposta", re.I
)


def is_abstention(text):
    return bool(ABSTAIN_RE.search(text or ""))


NAME_SUFFIX = {"jr", "júnior", "junior", "filho", "neto", "sobrinho"}


def short_ref(c):
    """'(Cid et al., 2014 — Título curto)'. Aceita autores separados por ';' (planilha) ou ','."""
    raw = (c.get("authors") or "").strip()
    has_etal = bool(re.search(r"\bet al\b", raw))
    raw = re.sub(r"\bet al\.?", "", raw).strip(" ;,")
    parts = [a.strip() for a in re.split(r";" if ";" in raw else r",| & | and ", raw) if a.strip()]
    first = parts[0].split(",")[0] if parts else ""  # "Fernandez, F." -> "Fernandez"
    words = [w for w in first.split() if w.rstrip(".").lower() not in NAME_SUFFIX]
    surname = words[-1] if words else ""
    year = c.get("year") or "s.d."
    title = (c.get("title") or c.get("doc_id", "")).strip()
    title = title if len(title) <= 60 else title[:57].rstrip() + "..."
    who = (f"{surname} et al., " if (len(parts) > 1 or has_etal) else f"{surname}, ") if surname else ""
    return f"({who}{year} — {title})"


def format_context(chunks):
    return "\n".join(f"[{i}] {short_ref(c)} {c['text'].strip()}" for i, c in enumerate(chunks, 1))


def build_user_message(question, chunks=None):
    if chunks:
        return f"CONTEXTO:\n{format_context(chunks)}\n\nPERGUNTA: {question.strip()}"
    return question.strip()


def build_messages(question, chunks=None):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]


PT_SW = set("de que não para com uma os as dos das em é são também pela pelo mais como sua seu ao".split())
EN_SW = set("the and of to is are with for that this which were from by their".split())


def is_ptbr(text):
    toks = re.findall(r"\w+", (text or "").lower())
    return sum(t in PT_SW for t in toks) >= sum(t in EN_SW for t in toks)
