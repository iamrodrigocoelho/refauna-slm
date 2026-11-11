"""Configuração: .env + config/settings.yaml + config/prompts.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    local_eval_dir: Path = Path("data/local_eval")


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    options: dict[str, float | int] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    label: str
    tag: str
    family: Literal["gemma", "qwen"]
    enabled: bool = True
    note: str = ""
    colab_run: str | None = None  # pasta em data/outputs/ com a avaliação do Colab (modelos FT)
    source: str | None = None  # tag de origem (HF) quando `tag` é uma variante local gerada a partir dela
    text_only: bool = False  # gerar `tag` a partir de `source` sem o projetor de visão/áudio

    @property
    def think(self) -> bool | None:
        """Qwen3.5 liga o raciocínio por padrão: sempre enviar think=false."""
        return False if self.family == "qwen" else None


class RetrievalConfig(BaseModel):
    top_k: int = 4
    n_candidates: int = 20
    rrf_k: int = 60
    embed_model: str = "google/embeddinggemma-300m"


class EvalConfig(BaseModel):
    """Avaliação local (F6.2): tamanho da amostra e juiz opcional (o mesmo modelo do Colab)."""

    n_items: int = 40
    mix: dict[str, float] = Field(default_factory=lambda: {"rag": 0.6, "closed_book": 0.2, "unanswerable": 0.2})
    judge_model: str = "claude-sonnet-5-5"


class DemoConfig(BaseModel):
    """App Gradio (F5): perguntas de exemplo e configurações pré-selecionadas na aba Comparar."""

    default_model: str = "qwen_ft"  # modelo principal da demo (ADR-15)
    examples: list[str] = Field(default_factory=list)
    compare_defaults: list[str] = Field(default_factory=list)  # "<modelo>|rag" ou "<modelo>|norag"
    server_port: int = 7860


class CardFamily(BaseModel):
    """Repositórios de uma família no HF (LoRA, mesclado, GGUF) para gerar os model cards."""

    name: str
    base: str
    unsloth_base: str
    colab_run: str
    repos: dict[str, str]


class Settings(BaseSettings):
    """Segredos e caminhos vindos do ambiente/.env; o resto vem dos YAML."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str | None = None
    refauna_drive_dir: str | None = None
    anthropic_api_key: str | None = None

    paths: PathsConfig = Field(default_factory=PathsConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    hf_cards: dict[str, CardFamily] = Field(default_factory=dict)
    seed: int = 3407
    system_prompt: str = ""

    @property
    def data_dir(self) -> Path:
        return _abs(self.paths.data_dir)

    @property
    def chroma_dir(self) -> Path:
        return _abs(self.paths.chroma_dir)

    @property
    def local_eval_dir(self) -> Path:
        return _abs(self.paths.local_eval_dir)

    def model(self, key: str) -> ModelSpec:
        if key not in self.models:
            raise KeyError(f"modelo desconhecido: {key} (opções: {', '.join(self.models)})")
        return self.models[key]

    @property
    def drive_dir(self) -> Path | None:
        if not self.refauna_drive_dir or not self.refauna_drive_dir.strip():
            return None
        return Path(self.refauna_drive_dir.strip().strip('"')).expanduser()


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    data = load_yaml("settings.yaml")
    data["system_prompt"] = load_yaml("prompts.yaml")["system_prompt"]
    return Settings(**data)
