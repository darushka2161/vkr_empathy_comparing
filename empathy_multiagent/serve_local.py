#!/usr/bin/env python3
"""
Скрипт для запуска локального vLLM-сервера с OpenAI-совместимым API.

Использование:
    python serve_local.py --model mistral-small-3.2
    python serve_local.py --model qwen3-32b-local --tensor-parallel-size 2
    python serve_local.py --model llama-3.1-8b-local --port 8001 --max-model-len 4096

После запуска сервера запускай эксперименты как обычно:
    python run_experiment.py --model mistral-small-3.2 --arch empathy_chain --limit 50

Требования:
    pip install vllm
    Добавить в .env: LOCAL_API_KEY=EMPTY
"""

import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import MODEL_REGISTRY

LOCAL_MODELS = {
    k: v for k, v in MODEL_REGISTRY.items()
    if v.get("provider") == "local-vllm"
}


def main():
    if not LOCAL_MODELS:
        print("Нет локальных моделей в MODEL_REGISTRY (provider='local-vllm').")
        sys.exit(1)

    model_list = "\n".join(f"  {k}: {v['model']}" for k, v in LOCAL_MODELS.items())

    parser = argparse.ArgumentParser(
        description="Запуск vLLM-сервера для локального инференса",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Доступные модели:\n{model_list}",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(LOCAL_MODELS.keys()),
        help="Ключ модели из MODEL_REGISTRY (provider=local-vllm)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Порт сервера (default: 8000)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Доля GPU-памяти для vLLM, 0.0–1.0 (default: 0.90)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Число GPU для tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Максимальная длина контекста в токенах (default: из конфига модели)",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
        help="Тип данных весов (default: bfloat16)",
    )
    args = parser.parse_args()

    cfg = LOCAL_MODELS[args.model]
    hf_model = cfg["model"]
    port = args.port

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_model,
        "--served-model-name", hf_model,  # имя модели в API-запросах должно совпадать с cfg["model"]
        "--port", str(port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--trust-remote-code",  # нужно для Qwen и других кастомных архитектур
    ]

    max_len = args.max_model_len or cfg.get("max_model_len")
    if max_len:
        cmd += ["--max-model-len", str(max_len)]

    print("=" * 50)
    print(f"Модель:    {hf_model}")
    print(f"Ключ:      {args.model}")
    print(f"Адрес:     http://localhost:{port}/v1")
    print(f"Размер:    {cfg['size']}")
    print()
    print(f"Команда:\n  {' '.join(cmd)}")
    print()
    print("После запуска:")
    print(f"  python run_experiment.py --model {args.model} --arch empathy_chain --limit 50")
    print()
    print("Остановить: Ctrl+C")
    print("=" * 50)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    except FileNotFoundError:
        print("\n[Ошибка] vllm не установлен. Установи:")
        print("  pip install vllm")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[Ошибка] vLLM завершился с кодом {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
