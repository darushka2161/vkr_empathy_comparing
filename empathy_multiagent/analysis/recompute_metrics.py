# recompute_metrics.py
# Пересчитывает метрики из уже готовых outputs/*.json без повторного запуска архитектур.
#
# Использование:
#   python recompute_metrics.py                        # все файлы в outputs/
#   python recompute_metrics.py --no-bertscore         # пропустить BERTScore
#   python recompute_metrics.py --files outputs/gpt-4o-mini_empathy_chain.json

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

from src.metrics import compute_all_metrics, compute_rag_metrics

PRINT_KEYS = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "ROUGE-1", "ROUGE-2", "ROUGE-L",
    "BERTScore-P", "BERTScore-R", "BERTScore-F",
    "Dist-1", "Dist-2",
    "AvgLen", "Accuracy (%)",
    "Avg calls/example", "Errors",
]


def recompute(path: Path, skip_bertscore: bool) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    arch = data.get("architecture", "")

    hypotheses = [r["generated_response"] for r in results if r.get("generated_response")]
    references = [r["gold_response"] for r in results if r.get("generated_response")]

    if not hypotheses:
        print(f"[SKIP] {path.name} — нет сгенерированных ответов")
        return

    predicted_emotions, gold_emotions = [], []
    for r in results:
        em = r.get("intermediate", {}).get("emotion", {})
        if isinstance(em, dict) and "emotion" in em:
            predicted_emotions.append(em["emotion"])
            gold_emotions.append(r["emotion"])

    print(f"\n{'=' * 55}")
    print(f"File:    {path.name}")
    print(f"Model:   {data.get('model')}  |  Arch: {arch}")
    print(f"Examples: {len(hypotheses)}")
    if skip_bertscore:
        print("BERTScore: SKIPPED")
    print(f"{'=' * 55}")

    metrics = compute_all_metrics(
        hypotheses,
        references,
        predicted_emotions or None,
        gold_emotions or None,
        results,
        skip_bertscore=skip_bertscore,
    )

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    metrics["Avg latency ms"] = round(sum(latencies) / len(latencies)) if latencies else 0
    metrics["Errors"] = sum(1 for r in results if "error" in r)

    if arch == "empathy_rag":
        metrics.update(compute_rag_metrics(results))

    for k in PRINT_KEYS:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")

    data["metrics"] = metrics
    data["metrics_recomputed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute metrics from saved outputs.")
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Конкретные JSON-файлы. По умолчанию — все в outputs/.",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true", default=False,
        help="Пропустить BERTScore.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted((root / "outputs").glob("*.json"))

    if not paths:
        print("Файлы не найдены.")
    else:
        for p in paths:
            recompute(p, skip_bertscore=args.no_bertscore)
        print(f"\nГотово. Обработано файлов: {len(paths)}")
