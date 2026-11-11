"""F5 — RAM durante 10 perguntas seguidas no app Gradio em execução (aceite: ≤ 6 GB).

Uso (com o app rodando em outro terminal, `uv run refauna serve`):
    uv run python scripts/ram_demo.py [--url http://127.0.0.1:7860] [--models qwen_ft,gemma_ft] [-n 10]

Alterna os modelos a cada pergunta (pior caso: troca de modelo sempre). Salva data/local_eval/ram_demo.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from urllib.parse import urlparse

import ollama
import psutil
from gradio_client import Client

from refauna_slm.config import get_settings
from refauna_slm.eval.memory import GB, RamSampler


def server_pid(port: int) -> int:
    """PID que escuta na porta (lsof: psutil.net_connections exige root no macOS)."""
    out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    if not out:
        raise SystemExit(f"Nenhum processo escutando na porta {port}. Rode `uv run refauna serve` antes.")
    return int(out[0])


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=f"http://127.0.0.1:{s.demo.server_port}")
    ap.add_argument("--models", default="qwen_ft,gemma_ft")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()
    models = args.models.split(",")
    pid = server_pid(urlparse(args.url).port or 80)
    oll = ollama.Client(host=s.ollama.host)

    client = Client(args.url, verbose=False)
    sampler = RamSampler(pid, interval=1.0)
    sampler.start()
    rows, model_peak = [], 0.0
    t_all = time.time()
    for i in range(args.n):
        q = s.demo.examples[i % len(s.demo.examples)]
        m = models[i % len(models)]
        t0 = time.time()
        _, _, _, stats = client.predict(q, [], m, True, api_name="/chat")
        dt = time.time() - t0
        loaded = {x.model: x.size for x in oll.ps().models}
        model_gb = sum(loaded.values()) / GB
        model_peak = max(model_peak, model_gb)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        rows.append({"i": i + 1, "model": m, "question": q, "wall_s": round(dt, 1), "stats": stats,
                     "ollama_model_gb": round(model_gb, 2), "loaded": list(loaded),
                     "available_gb": round(vm.available / GB, 2), "swap_gb": round(sw.used / GB, 2)})  # fmt: skip
        print(f"{i + 1:2d}. {m:9s} {dt:5.1f}s · modelo {model_gb:.2f} GB · disponível {vm.available / GB:.2f} GB · "
              f"swap {sw.used / GB:.2f} GB · {q[:50]}")  # fmt: skip
    peak = sampler.stop()

    result = {
        "n": args.n, "models": models, "total_s": round(time.time() - t_all, 1),
        "peak": {"app_gb": round(peak.app_footprint_gb, 2), "ollama_gb": round(peak.ollama_footprint_gb, 2),
                 "total_gb": round(peak.footprint_total_gb, 2), "app_rss_gb": round(peak.app_gb, 2),
                 "ollama_rss_gb": round(peak.ollama_gb, 2), "ollama_ps_gb": round(model_peak, 2),
                 "system_used_gb": round(peak.system_used_gb, 2), "swap_used_gb": round(peak.swap_used_gb, 2),
                 "swap_start_gb": round(peak.swap_start_gb, 2)},
        "single_model_loaded": all(len(r["loaded"]) <= 1 for r in rows),
        "rows": rows,
    }  # fmt: skip
    out = s.local_eval_dir / "ram_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    p = result["peak"]
    print(f"\nPico (phys_footprint) app {p['app_gb']:.2f} GB + Ollama {p['ollama_gb']:.2f} GB → juntos {p['total_gb']:.2f} GB "
          f"{'✅' if p['total_gb'] <= 6 else '❌'} (aceite ≤ 6 GB)")  # fmt: skip
    print(f"Sistema: pico usado {p['system_used_gb']:.2f} GB de {psutil.virtual_memory().total / GB:.0f} GB · "
          f"swap {p['swap_start_gb']:.2f} → pico {p['swap_used_gb']:.2f} GB")  # fmt: skip
    print(f"Um modelo por vez: {'✅' if result['single_model_loaded'] else '❌'} · salvo em {out}")


if __name__ == "__main__":
    main()
