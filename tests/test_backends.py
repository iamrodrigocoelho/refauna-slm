"""F4 — payload do Ollama (think:false p/ Qwen), limpeza de <think> e um modelo por vez."""

from types import SimpleNamespace

from refauna_slm.config import ModelSpec, Settings
from refauna_slm.rag.backends import OllamaBackend, strip_think

QWEN = ModelSpec(label="q", tag="hf.co/x/qwen:Q4_K_M", family="qwen")
GEMMA = ModelSpec(label="g", tag="hf.co/x/gemma:Q4_K_M", family="gemma")
MSGS = [{"role": "user", "content": "oi"}]


def backend() -> OllamaBackend:
    return OllamaBackend(Settings(ollama={"options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": 512}}))


def test_qwen_payload_sends_think_false():
    p = backend().payload(QWEN, MSGS, stream=True)
    assert p["think"] is False and p["model"] == QWEN.tag and p["stream"] is True
    assert p["options"] == {"temperature": 0.3, "num_ctx": 4096, "num_predict": 512}


def test_gemma_payload_has_no_think():
    assert "think" not in backend().payload(GEMMA, MSGS, stream=False)


def test_options_override():
    assert backend().payload(QWEN, MSGS, stream=False, temperature=0)["options"]["temperature"] == 0


def test_strip_think():
    assert strip_think("<think>raciocínio\nlongo</think>\n\nResposta [1].") == "Resposta [1]."
    assert strip_think("Resposta.") == "Resposta."
    assert strip_think("<think>sem fim") == ""
    assert strip_think(None) == ""


class FakeClient:
    def __init__(self, running):
        self.running, self.unloaded, self.chats = running, [], []

    def ps(self):
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.running])

    def generate(self, model, keep_alive):
        assert keep_alive == 0
        self.unloaded.append(model)
        self.running.remove(model)  # o Ollama tira o modelo da memória

    def chat(self, **kw):
        self.chats.append(kw)
        msg = SimpleNamespace(content="<think>x</think>Olá [1]", thinking=None)
        yield SimpleNamespace(
            message=msg, done=True, eval_count=10, eval_duration=2e9, load_duration=0, prompt_eval_count=5, prompt_eval_duration=1e9,
            done_reason="stop",
        )  # fmt: skip


def test_one_model_at_a_time_and_stats():
    b = backend()
    b.client = FakeClient(running=[GEMMA.tag, QWEN.tag])
    gen = b.chat(QWEN, MSGS)
    assert b.client.unloaded == [GEMMA.tag]  # descarrega o outro, mantém o atual
    assert b.client.chats[0]["think"] is False
    assert gen.text == "Olá [1]" and gen.tokens_per_s == 5.0


def test_waits_until_previous_model_leaves_memory():
    """keep_alive=0 é assíncrono: só carrega o próximo depois que o anterior sai do `ollama ps`."""
    b = backend()
    fake = FakeClient(running=[GEMMA.tag])
    polls = []

    def slow_unload(model, keep_alive):
        fake.unloaded.append(model)  # ainda aparece no ps nas 2 primeiras consultas

    def ps():
        polls.append(1)
        if len(polls) > 3:
            fake.running.clear()
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in fake.running])

    fake.generate, fake.ps = slow_unload, ps
    b.client = fake
    b.ensure_only(QWEN)
    assert fake.unloaded == [GEMMA.tag] and fake.running == [] and len(polls) >= 4
