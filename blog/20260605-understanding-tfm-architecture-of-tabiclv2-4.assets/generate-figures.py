"""Generate figures for TabICLv2 post 4 (QASSMax)."""

import os
from pathlib import Path

TMP_DIR = Path(os.environ.get("TMPDIR", "/tmp"))
MPLCONFIG_DIR = TMP_DIR / "matplotlib-cache"
XDG_CACHE_DIR = TMP_DIR / "xdg-cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent


def attention_fading_curve() -> None:
    delta = 2.0
    n = np.linspace(1, 500, 500)
    a_star = 1.0 / (1.0 + (n - 1) * np.exp(-delta))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, a_star, color="#2563eb", linewidth=2)
    ax.set_xlabel("Number of keys $N$")
    ax.set_ylabel(r"Attention mass on relevant key $a_\star$")
    ax.set_title(r"Attention fading (fixed logit gap $\Delta=2$)")
    ax.set_xlim(1, 500)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "Ranking preserved,\nmass diluted",
        xy=(400, a_star[-1]),
        xytext=(280, 0.45),
        arrowprops=dict(arrowstyle="->", color="#64748b"),
        fontsize=10,
        color="#334155",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attention-fading-curve.png", dpi=150)
    plt.close(fig)


def add_box(ax, x: float, y: float, w: float, h: float, label: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#334155",
        facecolor="#f1f5f9",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)


def add_arrow(ax, x_start: float, x_end: float, y: float = 1.6) -> None:
    arrow = FancyArrowPatch(
        (x_start, y),
        (x_end, y),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#64748b",
    )
    ax.add_patch(arrow)


def qassmax_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    boxes = [
        (0.25, 1.1, 1.25, 1.0, r"$q_h$"),
        (2.0, 1.1, 1.55, 1.0, r"$\times\, B_h(n)$"),
        (4.05, 1.1, 1.65, 1.0, r"$\times\, G_h(q_h)$"),
        (6.2, 1.1, 1.3, 1.0, r"$\tilde{q}_h$"),
        (8.0, 1.1, 1.55, 1.0, r"$\tilde{q}_h K_h^\top$"),
        (10.05, 1.1, 1.35, 1.0, r"$\tilde{z}$"),
        (11.9, 1.1, 1.45, 1.0, "softmax"),
        (13.85, 1.1, 0.45, 1.0, r"$a$"),
    ]
    for box in boxes:
        add_box(ax, *box)

    for x_start, x_end in [
        (1.5, 2.0),
        (3.55, 4.05),
        (5.7, 6.2),
        (7.5, 8.0),
        (9.55, 10.05),
        (11.4, 11.9),
        (13.35, 13.85),
    ]:
        add_arrow(ax, x_start, x_end)

    ax.text(2.78, 0.35, "Length-dependent base", ha="center", va="center", fontsize=9, color="#64748b")
    ax.text(4.88, 0.35, "Query-dependent gate", ha="center", va="center", fontsize=9, color="#64748b")
    ax.text(8.78, 0.35, "Query-key dot products", ha="center", va="center", fontsize=9, color="#64748b")
    ax.text(10.72, 0.35, "Attention logits", ha="center", va="center", fontsize=9, color="#64748b")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "qassmax-decomposition.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    attention_fading_curve()
    qassmax_decomposition()
    print("Wrote attention-fading-curve.png and qassmax-decomposition.png")
