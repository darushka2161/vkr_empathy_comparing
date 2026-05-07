# analysis/compare_results.py
# Сводная таблица и графики по всем результатам из outputs/

import json
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]

# Фиксированный набор столбцов для таблицы, CSV и графиков
TABLE_COLS = [
    "file", "model", "architecture",
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "ROUGE-1", "ROUGE-2", "ROUGE-L",
    "BERTScore-P", "BERTScore-R", "BERTScore-F",
    "Dist-1", "Dist-2",
    "AvgLen", "Accuracy (%)",
    "Avg calls/example", "Errors", "Successful",
]

PLOT_METRICS = [
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "ROUGE-1", "ROUGE-2", "ROUGE-L",
    "BERTScore-P", "BERTScore-R", "BERTScore-F",
    "Dist-1", "Dist-2",
    "AvgLen", "Accuracy (%)",
    "Avg calls/example", "Errors",
]


def load_all_results(outputs_dir: str = None) -> list[dict]:
    """Загружает все JSON-файлы из outputs/."""
    if outputs_dir is None:
        outputs_dir = _ROOT / "outputs"
    results = []
    for path in sorted(Path(outputs_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results.append({
            "file": path.name,
            "model": data.get("model"),
            "architecture": data.get("architecture"),
            "num_examples": data.get("num_examples"),
            **data.get("metrics", {}),
        })
    return results


def print_table(rows: list[dict]):
    """Выводит сводную таблицу в консоль."""
    if not rows:
        print("No results found in outputs/")
        return

    cols = [c for c in TABLE_COLS if any(c in r for r in rows)]
    col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    header = "  ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in cols)
        print(line)


def save_csv(rows: list[dict], out_path: str = None):
    """Сохраняет таблицу в CSV."""
    if not rows:
        return
    if out_path is None:
        out_path = _ROOT / "outputs" / "summary.csv"
    import csv
    cols = [c for c in TABLE_COLS if any(c in r for r in rows)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to: {out_path}")


def plot_metrics(rows: list[dict], metrics: list[str] = None):
    """Один subplot на метрику, по оси X — модель+архитектура."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    if not rows:
        return

    if metrics is None:
        metrics = PLOT_METRICS

    # Фильтруем метрики, для которых есть хоть одно ненулевое значение
    metrics = [m for m in metrics if any(r.get(m) for r in rows)]
    if not metrics:
        return

    labels = [f"{r['model']}\n{r['architecture']}" for r in rows]
    x = np.arange(len(labels))

    # Цвет по архитектуре
    arch_list = list(dict.fromkeys(r["architecture"] for r in rows))
    palette = plt.cm.Set2.colors
    arch_colors = {a: palette[i % len(palette)] for i, a in enumerate(arch_list)}
    colors = [arch_colors[r["architecture"]] for r in rows]

    # BERTScore метрики — ось Y от 0.8, остальные от 0
    BERTSCORE_METRICS = {"BERTScore-P", "BERTScore-R", "BERTScore-F"}
    BERTSCORE_YMIN = 0.8

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=arch_colors[a], label=a) for a in arch_list]

    out_dir = _ROOT / "outputs" / "plots"
    out_dir.mkdir(exist_ok=True)

    for metric in metrics:
        values = [float(r.get(metric) or 0) for r in rows]
        is_bert = metric in BERTSCORE_METRICS

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))

        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            if val:
                fmt = f"{val:.6f}" if is_bert else f"{val:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) - (BERTSCORE_YMIN if is_bert else 0)) * 0.01,
                    fmt,
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_title(metric, fontweight="bold", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        if is_bert:
            ymin = BERTSCORE_YMIN
            ymax = max(values) + (max(values) - ymin) * 0.15 if max(values) else ymin + 0.1
        else:
            ymin = 0
            ymax = max(values) * 1.15 if max(values) else 1
        ax.set_ylim(ymin, ymax)
        ax.spines[["top", "right"]].set_visible(False)

        fig.legend(handles=legend_handles, title="Architecture",
                   loc="lower center", ncol=len(arch_list),
                   bbox_to_anchor=(0.5, 0), frameon=False)
        fig.suptitle("EmpatheticDialogues — сравнение архитектур", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        safe_name = metric.replace("/", "_").replace(" ", "_")
        out = out_dir / f"{safe_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {out}")
        plt.close(fig)


if __name__ == "__main__":
    rows = load_all_results()
    rows.sort(key=lambda r: (r.get("model") or "", -float(r.get("BERTScore-F") or 0)))
    print_table(rows)
    save_csv(rows)
    plot_metrics(rows)
