"""Generate figures for TabICLv2 post 4 (QASSMax). Run from this directory."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


def qassmax_decomposition() -> None:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1.0, 1.4, 1.0, r"$q_h$"),
        (2.2, 1.0, 1.6, 1.0, r"$\times\, B_h(n)$"),
        (4.3, 1.0, 1.6, 1.0, r"$\times\, G_h(q_h)$"),
        (6.4, 1.0, 1.4, 1.0, r"$\tilde{q}_h$"),
        (8.1, 1.0, 1.5, 1.0, r"$\rightarrow$ softmax"),
    ]
    for x, y, w, h, label in boxes:
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

    for x_start, x_end in [(1.7, 2.2), (3.8, 4.3), (5.9, 6.4), (7.8, 8.1)]:
        arrow = FancyArrowPatch(
            (x_start, 1.5),
            (x_end, 1.5),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#64748b",
        )
        ax.add_patch(arrow)

    ax.text(
        5.0,
        0.35,
        "Length-dependent base          Query-dependent gate",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qassmax-decomposition.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    attention_fading_curve()
    qassmax_decomposition()
    print("Wrote attention-fading-curve.png and qassmax-decomposition.png")
