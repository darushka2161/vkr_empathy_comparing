# analysis/plot_architectures.py
# Визуализация четырёх мультиагентных архитектур

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── палитра ──────────────────────────────────────────────────────────────────
C_INPUT    = "#E8EEF4"   # серо-голубой  — Input / Output
C_LLM      = "#D0E8FF"   # синий         — LLM-агент
C_LLM2     = "#B8D4F5"   # темнее синий  — вторичный LLM
C_PARALLEL = "#E8F5E9"   # зелёный фон   — параллельная группа
C_AGENT_A  = "#C8E6C9"   # зелёный       — Comforter
C_AGENT_B  = "#B3E0F7"   # голубой       — Advisor
C_AGENT_C  = "#F9E4B7"   # жёлтый        — Explorer
C_ARBITER  = "#FFD6CC"   # красноватый   — Arbiter
C_LOOP     = "#FFF3CD"   # жёлтый        — Loop/Validate
C_FAISS    = "#EDE7F6"   # фиолетовый    — FAISS
C_OUTPUT   = "#DCEDC8"   # зелёный       — Response
C_BORDER   = "#455A64"
C_ARROW    = "#37474F"
C_LOOP_ARROW = "#E65100"

FONT_MAIN  = dict(fontsize=8.5, fontfamily="DejaVu Sans", color="#1A237E", fontweight="bold")
FONT_SUB   = dict(fontsize=7,   fontfamily="DejaVu Sans", color="#455A64")
FONT_LABEL = dict(fontsize=6.5, fontfamily="DejaVu Sans", color="#546E7A", style="italic")


def box(ax, x, y, w, h, label, sublabel="", color=C_LLM, alpha=1.0, radius=0.04):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.0, alpha=alpha,
        zorder=3,
    )
    ax.add_patch(rect)
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center", zorder=4, **FONT_MAIN)
    if sublabel:
        ax.text(x, y - 0.045, sublabel, ha="center", va="center", zorder=4, **FONT_LABEL)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, label="", style="->", lw=1.3, rad=0.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.02, my, label, zorder=5, **FONT_LABEL)


def brace_group(ax, x1, x2, y_top, y_bot, color=C_PARALLEL, label=""):
    rect = FancyBboxPatch(
        (x1, y_bot), x2-x1, y_top-y_bot,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=color, edgecolor="#81C784", linewidth=1.0, linestyle="--",
        alpha=0.45, zorder=1,
    )
    ax.add_patch(rect)
    if label:
        ax.text((x1+x2)/2, y_top + 0.04, label,
                ha="center", va="bottom", fontsize=6.5,
                color="#2E7D32", style="italic", fontweight="bold", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Архитектура 1: EmpathyChain
# ─────────────────────────────────────────────────────────────────────────────
def draw_chain(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("EmpathyChain", fontsize=11, fontweight="bold", color="#1A237E", pad=6)
    ax.text(0.5, 0.94, "Sequential cascade  •  4 LLM calls",
            ha="center", fontsize=7.5, color="#546E7A", style="italic",
            transform=ax.transAxes)

    nodes = [
        (0.5, 0.82, "Dialogue Input",   "",                  C_INPUT),
        (0.5, 0.66, "Emotion Classifier","emotion + intensity",C_LLM),
        (0.5, 0.50, "Cause Analyzer",   "cause + need",      C_LLM),
        (0.5, 0.34, "Strategy Selector","strategy + tone",   C_LLM2),
        (0.5, 0.18, "Response Generator","final response",    C_LLM2),
        (0.5, 0.05, "Response",         "",                  C_OUTPUT),
    ]
    W, H = 0.42, 0.09
    for x, y, lbl, sub, col in nodes:
        box(ax, x, y, W, H, lbl, sub, col)

    ys = [n[1] for n in nodes]
    for i in range(len(ys)-1):
        arrow(ax, 0.5, ys[i]-H/2, 0.5, ys[i+1]+H/2)

    # номера шагов
    for i, (_, y, *_) in enumerate(nodes[1:-1], 1):
        ax.text(0.795, y, f"Step {i}", va="center", fontsize=6.5,
                color="#90A4AE", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Архитектура 2: EmpathyDebate
# ─────────────────────────────────────────────────────────────────────────────
def draw_debate(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("EmpathyDebate", fontsize=11, fontweight="bold", color="#1A237E", pad=6)
    ax.text(0.5, 0.94, "Parallel agents + arbiter  •  5 LLM calls",
            ha="center", fontsize=7.5, color="#546E7A", style="italic",
            transform=ax.transAxes)

    W, H = 0.38, 0.09
    # Input & emotion
    box(ax, 0.5, 0.84, W, H, "Dialogue Input", "", C_INPUT)
    box(ax, 0.5, 0.70, W, H, "Emotion Classifier", "emotion", C_LLM)
    arrow(ax, 0.5, 0.84-H/2, 0.5, 0.70+H/2)

    # Параллельная зона
    brace_group(ax, 0.04, 0.96, 0.63, 0.38, C_PARALLEL, "parallel")
    agents = [
        (0.18, 0.51, "Comforter",  "emotional\nvalidation", C_AGENT_A),
        (0.50, 0.51, "Advisor",    "constructive\nperspective", C_AGENT_B),
        (0.82, 0.51, "Explorer",   "curious\nquestioning", C_AGENT_C),
    ]
    Wa = 0.26
    for x, y, lbl, sub, col in agents:
        box(ax, x, y, Wa, H+0.02, lbl, sub, col)
        arrow(ax, 0.5, 0.70-H/2, x, y+H/2+0.01, rad=0.1 if x != 0.5 else 0.0)

    # Arbiter
    box(ax, 0.5, 0.28, W, H, "Arbiter", "score + select best", C_ARBITER)
    for x, y, *_ in agents:
        arrow(ax, x, y-H/2-0.01, 0.5, 0.28+H/2, rad=-0.1 if x != 0.5 else 0.0)

    # Output
    box(ax, 0.5, 0.10, W, H, "Best Response", "", C_OUTPUT)
    arrow(ax, 0.5, 0.28-H/2, 0.5, 0.10+H/2)


# ─────────────────────────────────────────────────────────────────────────────
# Архитектура 3: EmpathyLoop
# ─────────────────────────────────────────────────────────────────────────────
def draw_loop(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("EmpathyLoop", fontsize=11, fontweight="bold", color="#1A237E", pad=6)
    ax.text(0.5, 0.94, "Iterative refinement  •  5–11 LLM calls",
            ha="center", fontsize=7.5, color="#546E7A", style="italic",
            transform=ax.transAxes)

    W, H = 0.38, 0.09
    # Input + Planner
    box(ax, 0.5, 0.84, W, H, "Dialogue Input", "", C_INPUT)
    box(ax, 0.5, 0.70, W, H, "Planner", "emotion + strategy\n+ key points", C_LLM)
    arrow(ax, 0.5, 0.84-H/2, 0.5, 0.70+H/2)

    # Generator
    box(ax, 0.5, 0.56, W, H, "Generator", "draft response", C_LLM2)
    arrow(ax, 0.5, 0.70-H/2, 0.5, 0.56+H/2)

    # Validation zone
    brace_group(ax, 0.04, 0.76, 0.49, 0.27, C_LOOP, "validate (parallel)")
    vals = [
        (0.15, 0.38, "Empathy\nValidator",   C_AGENT_A),
        (0.40, 0.38, "Coherence\nValidator", C_AGENT_B),
        (0.65, 0.38, "Safety\nValidator",    C_AGENT_C),
    ]
    Wv = 0.24
    for x, y, lbl, col in vals:
        box(ax, x, y, Wv, H, lbl, "", col)
        arrow(ax, 0.5, 0.56-H/2, x, y+H/2, rad=0.1 if x != 0.5 else 0.0)

    # Decision diamond (симулируем ромб через текст + прямоугольник)
    box(ax, 0.40, 0.19, 0.32, 0.08, "All pass?", "", "#FFEACC", radius=0.06)
    for x, y, *_ in vals:
        arrow(ax, x, y-H/2, 0.40, 0.19+0.04, rad=-0.1 if x != 0.40 else 0.0)

    # Output
    box(ax, 0.40, 0.07, W, H, "Response", "", C_OUTPUT)
    arrow(ax, 0.40, 0.19-0.04, 0.40, 0.07+H/2, label="yes")

    # Feedback loop — П-образный путь снаружи справа
    x_edge = 0.40 + W / 2       # правый край "All pass?" = 0.59
    x_out  = 0.82               # выход вправо, не задевает валидаторы
    y_bot  = 0.19                # центр "All pass?"
    y_top  = 0.56                # центр "Generator"
    x_gen_right = 0.5 + W / 2   # правый край Generator = 0.69
    # три отрезка: вправо → вверх → влево до Generator
    ax.plot([x_edge, x_out, x_out, x_gen_right],
            [y_bot,  y_bot,  y_top, y_top],
            color=C_LOOP_ARROW, lw=1.5, zorder=2, solid_capstyle="round")
    # стрелка в конце (указывает влево, в блок Generator)
    ax.annotate("", xy=(x_gen_right, y_top), xytext=(x_gen_right + 0.001, y_top),
                arrowprops=dict(arrowstyle="-|>", color=C_LOOP_ARROW,
                                lw=1.5, mutation_scale=10),
                zorder=3)
    ax.text(x_out + 0.02, (y_bot + y_top) / 2, "no /\nrefine",
            ha="left", va="center",
            fontsize=6.5, color=C_LOOP_ARROW, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Архитектура 4: EmpathyRAG
# ─────────────────────────────────────────────────────────────────────────────
def draw_rag(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("EmpathyRAG", fontsize=11, fontweight="bold", color="#1A237E", pad=6)
    ax.text(0.5, 0.94, "Retrieval-augmented  •  3 LLM calls + vector search",
            ha="center", fontsize=7.5, color="#546E7A", style="italic",
            transform=ax.transAxes)

    W, H = 0.42, 0.09
    nodes = [
        (0.5, 0.84, "Dialogue Input",     "",                     C_INPUT),
        (0.5, 0.70, "Emotion Classifier", "predicted emotion",     C_LLM),
        (0.5, 0.56, "FAISS Retriever",    "top-3 similar examples\n(no LLM)", C_FAISS),
        (0.5, 0.42, "Example Analyzer",   "patterns + strategies", C_LLM),
        (0.5, 0.26, "Response Generator", "few-shot + analysis",   C_LLM2),
        (0.5, 0.10, "Response",           "",                      C_OUTPUT),
    ]
    for x, y, lbl, sub, col in nodes:
        box(ax, x, y, W, H, lbl, sub, col)

    ys = [n[1] for n in nodes]
    for i in range(len(ys)-1):
        arrow(ax, 0.5, ys[i]-H/2, 0.5, ys[i+1]+H/2)

    # База знаний
    box(ax, 0.87, 0.56, 0.18, 0.08, "Train\nIndex", "", "#EDE7F6", radius=0.05)
    arrow(ax, 0.87-0.09, 0.56, 0.5+W/2, 0.56, label="query", style="-|>")

    ax.text(0.795, 0.70, "LLM ①", va="center", fontsize=6.5, color="#90A4AE", style="italic")
    ax.text(0.795, 0.42, "LLM ②", va="center", fontsize=6.5, color="#90A4AE", style="italic")
    ax.text(0.795, 0.26, "LLM ③", va="center", fontsize=6.5, color="#90A4AE", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────────────────────────────────────
def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 9))
    fig.patch.set_facecolor("#F8F9FA")

    for ax in axes:
        ax.set_facecolor("#FAFBFC")
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_chain(axes[0])
    draw_debate(axes[1])
    draw_loop(axes[2])
    draw_rag(axes[3])

    fig.suptitle(
        "Multiagent Architectures for Empathetic Response Generation",
        fontsize=14, fontweight="bold", color="#1A237E", y=1.01,
    )

    # Легенда
    legend_items = [
        mpatches.Patch(facecolor=C_INPUT,    edgecolor=C_BORDER, label="Input / Output"),
        mpatches.Patch(facecolor=C_LLM,      edgecolor=C_BORDER, label="LLM Agent"),
        mpatches.Patch(facecolor=C_FAISS,    edgecolor=C_BORDER, label="Vector Retriever (no LLM)"),
        mpatches.Patch(facecolor=C_PARALLEL, edgecolor="#81C784", linestyle="--", label="Parallel execution"),
        mpatches.Patch(facecolor=C_LOOP,     edgecolor=C_BORDER, label="Validation / Loop"),
        mpatches.Patch(facecolor=C_OUTPUT,   edgecolor=C_BORDER, label="Final Response"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center", ncol=6,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True, framealpha=0.9,
        fontsize=8.5, edgecolor="#CFD8DC",
    )

    plt.tight_layout(pad=1.5)
    out = Path(__file__).resolve().parents[1] / "outputs" / "architectures.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
