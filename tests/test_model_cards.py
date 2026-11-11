"""PRD F8.3 — model cards gerados a partir dos dados (sem rede)."""

import re

import pytest

from refauna_slm.config import ROOT, get_settings, load_yaml
from refauna_slm.model_cards import card

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "docs.jsonl").exists(), reason="rode `refauna sync` primeiro")


@pytest.fixture(scope="module")
def cards() -> dict[tuple[str, str], str]:
    s = get_settings()
    sp = load_yaml("prompts.yaml")["system_prompt"]
    return {(f, k): card(s, f, k, sp) for f, fam in s.hf_cards.items() for k in fam.repos}


def test_repos_only_from_settings(cards):
    s = get_settings()
    known = {r for fam in s.hf_cards.values() for r in fam.repos.values()} | {fam.base for fam in s.hf_cards.values()}
    for text in cards.values():
        for repo in re.findall(r"huggingface\.co/([\w.-]+/[\w.-]+)", text):
            assert repo in known, repo
        assert "iamrodrigocoelho/gemma" not in text and "tapirus-open" not in text


def test_front_matter_relation(cards):
    assert "base_model_relation: adapter" in cards[("gemma", "lora")]
    assert "base_model: Refauna/gemma-4-e2b-refauna\nbase_model_relation: quantized" in cards[("gemma", "gguf")]
    assert "library_name: transformers" in cards[("qwen", "merged")]


def test_contract_and_thinking(cards):
    sp = load_yaml("prompts.yaml")["system_prompt"]
    for (family, _), text in cards.items():
        assert sp in text and "PERGUNTA: <pergunta>" in text
        if family == "qwen":
            assert "enable_thinking=False" in text and '"think": false' in text
            assert "Pouca memória" not in text  # variante só-texto só foi validada no Gemma
        else:
            assert "**não** envie o parâmetro `think`" in text


def test_attribution_punctuation(cards):
    text = cards[("gemma", "gguf")]
    assert "et al.." not in text and "Jr.." not in text and "doi.org/10." in text


def test_honest_reading(cards):
    assert "não aumentou a correção" in cards[("qwen", "gguf")]
    assert "abstenção correta 12% → 100%" in cards[("qwen", "gguf")]
