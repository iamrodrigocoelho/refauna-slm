"""F2 — registro de modelos: baixa (pull) e verifica (check) os modelos do settings.yaml no Ollama."""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama

from refauna_slm.config import ROOT, ModelSpec, Settings

PRIVATE_HINT = (
    "Repositório privado no HF? Cadastre a chave pública do Ollama (cat ~/.ollama/id_ed25519.pub) "
    "em huggingface.co → Settings → SSH keys, ou torne o repositório público."
)


@dataclass
class ModelStatus:
    key: str
    spec: ModelSpec
    present: bool
    size_gb: float | None = None
    quantization: str | None = None
    parameter_size: str | None = None
    arch: str | None = None
    has_template: bool | None = None
    can_think: bool | None = None
    multimodal: bool | None = None
    smoke: str | None = None
    smoke_ok: bool | None = None
    error: str | None = None


def client(settings: Settings) -> ollama.Client:
    return ollama.Client(host=settings.ollama.host)


def normalize_tag(tag: str) -> str:
    """'gemma4' e 'gemma4:latest' são o mesmo modelo para o Ollama."""
    return tag if ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"


def server_version(host: str) -> str:
    """Versão do servidor Ollama (levanta httpx.HTTPError se não estiver rodando)."""
    return httpx.get(f"{host.rstrip('/')}/api/version", timeout=5).json().get("version", "?")


def pull(cli: ollama.Client, spec: ModelSpec) -> Iterator[ollama.ProgressResponse]:
    """Baixa a tag (ou a `source` do HF, quando `tag` é uma variante local)."""
    yield from cli.pull(spec.source or spec.tag, stream=True)


def text_only_modelfile(modelfile: str) -> str:
    """Modelfile só-texto: mantém o 1º FROM (pesos), TEMPLATE e PARAMETERs; descarta o FROM do projetor.

    Não acrescenta RENDERER/PARSER nem think: com isso o prompt fica idêntico ao do GGUF original
    (validado: 5/5 respostas iguais com temperatura 0; com think=false o prompt muda 2 tokens).
    """
    froms = re.findall(r"^FROM (.+)$", modelfile, re.M)
    if not froms:
        raise ValueError("Modelfile sem FROM")
    tmpl = re.search(r'^TEMPLATE ".*?"\n(?=PARAMETER|LICENSE|SYSTEM|MESSAGE|\Z)', modelfile, re.M | re.S)
    params = re.findall(r"^PARAMETER .+$", modelfile, re.M)
    parts = [f"FROM {froms[0]}"] + ([tmpl.group(0).rstrip("\n")] if tmpl else []) + params
    return "# Gerado por `refauna models build` — não editar à mão (PRD F8.4).\n" + "\n".join(parts) + "\n"


def build_text_only(settings: Settings, key: str) -> Path:
    """Gera ollama/Modelfile.<key> a partir de `source` e cria `tag` no Ollama (sem internet)."""
    spec = settings.model(key)
    if not (spec.text_only and spec.source):
        raise ValueError(f"{key} não é uma variante só-texto (text_only + source no settings.yaml).")
    cli = client(settings)
    path = ROOT / "ollama" / f"Modelfile.{key}"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text_only_modelfile(cli.show(spec.source).modelfile))
    env = {**os.environ, "OLLAMA_HOST": settings.ollama.host}
    res = subprocess.run(["ollama", "create", spec.tag, "-f", str(path)], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"ollama create falhou: {res.stderr.strip()[-300:]}")
    caps = cli.show(spec.tag).capabilities or []
    if "vision" in caps or "audio" in caps:
        raise RuntimeError(f"{spec.tag} ainda tem capacidades multimodais: {caps}")
    return path


def select(settings: Settings, keys: list[str] | None) -> dict[str, ModelSpec]:
    """Filtra o registro por chave (gemma_ft) ou família (qwen). Vazio = todos os ativos.

    Modelos desativados só entram quando pedidos pela chave.
    """
    if not keys:
        return {k: s for k, s in settings.models.items() if s.enabled}
    unknown = [k for k in keys if k not in settings.models and k not in {s.family for s in settings.models.values()}]
    if unknown:
        raise KeyError(f"modelo/família desconhecido(a): {', '.join(unknown)} (opções: {', '.join(settings.models)})")
    return {k: s for k, s in settings.models.items() if k in keys or (s.enabled and s.family in keys)}


def disabled(settings: Settings, selected: dict[str, ModelSpec]) -> dict[str, ModelSpec]:
    """Modelos desativados que ficaram de fora da seleção."""
    return {k: s for k, s in settings.models.items() if not s.enabled and k not in selected}


def check(
    settings: Settings,
    smoke: bool = False,
    on_model: Callable[[str], None] | None = None,
    keys: list[str] | None = None,
) -> list[ModelStatus]:
    """Confere se cada modelo do registro está no Ollama local (= disponível offline)."""
    cli = client(settings)
    local = {normalize_tag(m.model): m for m in cli.list().models}
    out = []
    for key, spec in select(settings, keys).items():
        if on_model:
            on_model(key)
        m = local.get(normalize_tag(spec.tag))
        st = ModelStatus(key=key, spec=spec, present=m is not None)
        if m is not None:
            st.size_gb = round(m.size / 1e9, 2)
            st.quantization = m.details.quantization_level if m.details else None
            st.parameter_size = m.details.parameter_size if m.details else None
            st.arch = m.details.family if m.details else None
            try:
                info = cli.show(spec.tag)
                st.has_template = bool(info.template)
                st.can_think = "thinking" in (info.capabilities or [])
                st.multimodal = bool({"vision", "audio"} & set(info.capabilities or []))
            except ollama.ResponseError as e:
                st.error = str(e)
            if smoke:
                _smoke(cli, settings, st)
        out.append(st)
    return out


def _smoke(cli: ollama.Client, settings: Settings, st: ModelStatus) -> None:
    """Geração curtíssima, já com think=false (Qwen) e keep_alive=0 (um modelo por vez)."""
    t0 = time.time()
    try:
        r = cli.chat(
            model=st.spec.tag,
            messages=[{"role": "user", "content": "Responda só: ok"}],
            think=st.spec.think,
            keep_alive=0,
            options={**settings.ollama.options, "num_predict": 8, "temperature": 0},
        )
        text = r.message.content or ""
        leaked_think = bool(r.message.thinking) or "<think>" in text
        preview = " ".join(text.split())[:30]
        st.smoke = f"{time.time() - t0:.1f}s · {preview!r}"
        st.smoke_ok = bool(text.strip()) and not leaked_think
        if leaked_think:
            st.smoke += " · ⚠️ raciocínio ligado"
    except Exception as e:  # noqa: BLE001 — reportar qualquer falha do Ollama na tabela
        st.smoke, st.smoke_ok = f"{type(e).__name__}: {e}"[:80], False
