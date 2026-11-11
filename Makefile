.PHONY: setup sync models-pull models-check index ask eval bench report app space lint test check

Q ?= O que é refaunação e por que ela importa para a Mata Atlântica?
MODEL ?= qwen_ft

setup:          ## instala dependências (Python 3.11 via uv)
	uv sync --all-extras

sync:           ## artefatos do Drive -> data/ (+ validação)
	uv run refauna sync

models-pull:    ## baixa os 4 modelos no Ollama (precisa de internet)
	uv run refauna models pull

models-check:   ## confirma que os 4 modelos estão disponíveis offline
	uv run refauna models check

index:          ## Chroma + BM25 (fase 2)
	uv run refauna index

ask:            ## make ask Q="pergunta" MODEL=gemma_ft (fase 2)
	uv run refauna ask "$(Q)" --model $(MODEL)

eval:           ## avaliação local A/B/C/D (fase 4)
	uv run refauna eval

bench:          ## latência, tokens/s, RAM (fase 4)
	uv run refauna bench

report:         ## docs/avaliacao.md + PNGs (fase 4)
	uv run refauna report

app:            ## app Gradio (fase 3)
	uv run refauna serve

space:          ## HF Space (opcional)
	@echo "F7 (opcional) ainda não implementado"

lint:
	uv run ruff check src tests app
	uv run ruff format --check src tests app

test:
	uv run pytest -q

check: lint test
