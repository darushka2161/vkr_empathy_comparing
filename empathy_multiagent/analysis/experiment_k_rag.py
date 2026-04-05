# analysis/experiment_k_rag.py
# K-ablation experiment for EmpathyRAG:
# Tests k=1,2,3,4,5,7,10,15,20 retrieved examples on 20 fixed test dialogues
# Usage: python analysis/experiment_k_rag.py --model qwen-3-32b [--no-bertscore]

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import asyncio
import argparse
import json
import time
from pathlib import Path

# Make sure src/ is importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from src.llm_factory import LLMFactory
from src.load_dataset import prepare_examples
from src.metrics import compute_bleu, compute_rouge, compute_distinct, compute_avg_length, compute_rag_metrics
from architectures.empathy_rag import EmpathyRetriever, empathy_rag


K_VALUES = [1, 2, 3, 4, 5, 7, 10, 15, 20]
N_EXAMPLES = 10


def get_retriever(cache_dir: str = "retriever_cache") -> EmpathyRetriever:
    cache_path = _ROOT / cache_dir
    if (cache_path / "examples.pkl").exists():
        print("Loading cached retriever...")
        return EmpathyRetriever.load(str(cache_path))
    else:
        print("Building retriever from ED train set (~2 min on first run)...")
        train_examples = prepare_examples("train")
        retriever = EmpathyRetriever(train_examples)
        retriever.save(str(cache_path))
        return retriever


async def run_k_experiment(model_key: str, skip_bertscore: bool = False, resume: bool = False):
    os.chdir(_ROOT)

    llm = LLMFactory(model_key)
    retriever = get_retriever()

    # Fix the same 20 test examples for all k values
    test_examples = prepare_examples("test", N_EXAMPLES)
    print(f"\nModel: {model_key}")
    print(f"Examples: {N_EXAMPLES} fixed test dialogues")
    print(f"K values: {K_VALUES}")
    print(f"{'=' * 60}\n")

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{model_key}_rag_k_ablation.json"

    # Load partial results if resuming
    all_k_results = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        all_k_results = {int(k): v for k, v in saved.get("results_by_k", {}).items()}
        done_ks = [k for k in all_k_results if "metrics" in all_k_results[k]]
        if done_ks:
            print(f"Resuming: already done k={done_ks}, skipping them.\n")

    for k in K_VALUES:
        # Skip if already completed (resume mode)
        if k in all_k_results and "metrics" in all_k_results[k]:
            print(f"--- k={k}: already done, skipping ---")
            continue

        print(f"\n--- Running k={k} ---")
        results = []
        for i, ex in enumerate(test_examples):
            try:
                start = time.time()
                result = await empathy_rag(ex["context"], llm, retriever, top_k=k)
                elapsed = (time.time() - start) * 1000
                results.append({
                    "conv_id": ex["conv_id"],
                    "emotion": ex["emotion"],
                    "gold_response": ex["gold_response"],
                    "generated_response": result["response"],
                    "llm_calls": result.get("llm_calls", 3),
                    "intermediate": {k2: v for k2, v in result.items() if k2 not in ("response", "llm_calls")},
                    "latency_ms": round(elapsed),
                })
                print(f"  [{i+1}/{N_EXAMPLES}] ok ({elapsed:.0f}ms)")
            except Exception as e:
                print(f"  [{i+1}/{N_EXAMPLES}] ERROR: {e}")
                results.append({
                    "conv_id": ex["conv_id"],
                    "emotion": ex["emotion"],
                    "gold_response": ex["gold_response"],
                    "generated_response": "",
                    "error": str(e),
                    "latency_ms": 0,
                })

        hypotheses = [r["generated_response"] for r in results if r.get("generated_response")]
        references = [r["gold_response"] for r in results if r.get("generated_response")]

        if not hypotheses:
            print(f"  All examples failed for k={k}, skipping metrics.")
            all_k_results[k] = {"error": "all_failed"}
            continue

        metrics = {}
        metrics.update(compute_bleu(hypotheses, references))
        metrics.update(compute_rouge(hypotheses, references))

        if not skip_bertscore:
            from src.metrics import compute_bertscore
            metrics.update(compute_bertscore(hypotheses, references))

        metrics.update(compute_distinct(hypotheses))
        metrics.update(compute_avg_length(hypotheses))
        metrics.update(compute_rag_metrics(results))

        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        metrics["Avg latency ms"] = round(sum(latencies) / len(latencies)) if latencies else 0
        metrics["Errors"] = sum(1 for r in results if "error" in r)

        all_k_results[k] = {
            "metrics": metrics,
            "results": results,
        }

        print(f"  k={k} metrics: BLEU-1={metrics.get('BLEU-1')}, ROUGE-L={metrics.get('ROUGE-L')}, "
              f"Avg sim={metrics.get('Avg retrieval similarity')}, Novelty={metrics.get('Response novelty')}")

        # Save after each k so we can resume if interrupted
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_key,
                "n_examples": N_EXAMPLES,
                "k_values": K_VALUES,
                "results_by_k": {str(kk): v for kk, v in all_k_results.items()},
            }, f, ensure_ascii=False, indent=2)
        print(f"  (saved checkpoint to {out_path.name})")

    # Final save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_key,
            "n_examples": N_EXAMPLES,
            "k_values": K_VALUES,
            "results_by_k": {str(kk): v for kk, v in all_k_results.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to: {out_path}")

    # Print comparison table
    print_comparison_table(all_k_results, skip_bertscore)

    # Save plot
    try:
        save_k_plots(all_k_results, model_key, out_dir, skip_bertscore)
    except Exception as e:
        print(f"Could not save plots: {e}")


def print_comparison_table(all_k_results: dict, skip_bertscore: bool):
    metrics_to_show = ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-L"]
    if not skip_bertscore:
        metrics_to_show.append("BERTScore-F")
    metrics_to_show += ["Dist-1", "Dist-2", "AvgLen", "Avg retrieval similarity", "Response novelty", "Avg latency ms"]

    col_w = 10
    header = f"{'k':<5}" + "".join(f"{m[:col_w-1]:<{col_w}}" for m in metrics_to_show)
    print("\n" + "=" * len(header))
    print("K-ABLATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for k in K_VALUES:
        entry = all_k_results.get(k, {})
        if "error" in entry:
            print(f"{k:<5}{'ERROR'}")
            continue
        m = entry.get("metrics", {})
        row = f"{k:<5}"
        for metric in metrics_to_show:
            val = m.get(metric, "N/A")
            row += f"{str(val)[:col_w-1]:<{col_w}}"
        print(row)

    print("=" * len(header))


def save_k_plots(all_k_results: dict, model_key: str, out_dir: Path, skip_bertscore: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("BLEU-1", "BLEU-1 (×100)"),
        ("ROUGE-L", "ROUGE-L (×100)"),
        ("Dist-2", "Distinct-2 (×100)"),
        ("Avg retrieval similarity", "Avg Retrieval Similarity"),
        ("Response novelty", "Response Novelty"),
        ("Avg latency ms", "Avg Latency (ms)"),
    ]
    if not skip_bertscore:
        metrics_to_plot.insert(2, ("BERTScore-F", "BERTScore-F"))

    valid_ks = [k for k in K_VALUES if "metrics" in all_k_results.get(k, {})]

    n_plots = len(metrics_to_plot)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[i]
        values = [all_k_results[k]["metrics"].get(metric_key) for k in valid_ks]
        ax.plot(valid_ks, values, marker="o", linewidth=2, markersize=6, color="#2196F3")
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("k (retrieved examples)", fontsize=10)
        ax.set_xticks(valid_ks)
        ax.grid(True, alpha=0.3)
        for x, y in zip(valid_ks, values):
            if y is not None:
                ax.annotate(f"{y:.2f}" if isinstance(y, float) else str(y),
                            (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"EmpathyRAG K-Ablation — {model_key} ({N_EXAMPLES} examples)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"{model_key}_rag_k_ablation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-ablation experiment for EmpathyRAG")
    parser.add_argument("--model", default="qwen-3-32b", help="Model key from config.py")
    parser.add_argument("--no-bertscore", action="store_true", default=False,
                        help="Skip BERTScore computation")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing partial results (skip already completed k values)")
    args = parser.parse_args()

    asyncio.run(run_k_experiment(args.model, skip_bertscore=args.no_bertscore, resume=args.resume))
