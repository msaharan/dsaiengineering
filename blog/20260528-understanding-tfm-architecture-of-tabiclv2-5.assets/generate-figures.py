"""Generate figures for TabICLv2 post 5 (many-class classification). Run from this directory."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent


def hierarchy_c57() -> None:
    """Root splits C=57 into six contiguous groups; each group is a leaf (<=10 classes)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    root = FancyBboxPatch(
        (4.5, 5.8),
        3,
        0.7,
        boxstyle="round,pad=0.05",
        facecolor="#e8f4fc",
        edgecolor="#2c5f7a",
        linewidth=1.5,
    )
    ax.add_patch(root)
    ax.text(
        6,
        6.15,
        "Root: 57 classes\n$K=6$ groups",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    groups = [
        (r"$\mathcal{G}_0$", "0–9", 10),
        (r"$\mathcal{G}_1$", "10–19", 10),
        (r"$\mathcal{G}_2$", "20–29", 10),
        (r"$\mathcal{G}_3$", "30–38", 9),
        (r"$\mathcal{G}_4$", "39–47", 9),
        (r"$\mathcal{G}_5$", "48–56", 9),
    ]
    xs = [0.6, 2.4, 4.2, 6.0, 7.8, 9.6]
    y_child = 2.2

    for i, (gname, crange, n) in enumerate(groups):
        x = xs[i]
        ax.annotate(
            "",
            xy=(x + 0.9, y_child + 0.85),
            xytext=(6, 5.75),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
        )
        box = FancyBboxPatch(
            (x, y_child),
            1.8,
            0.85,
            boxstyle="round,pad=0.04",
            facecolor="#f5f5e8",
            edgecolor="#6b6b4a",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.9,
            y_child + 0.55,
            f"{gname}\n{crange}\n({n} classes)",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.text(
            x + 0.9,
            y_child - 0.35,
            "leaf: direct ICL\n($\\leq 10$ classes)",
            ha="center",
            va="center",
            fontsize=8,
            color="#444",
            style="italic",
        )

    ax.text(
        6,
        0.5,
        r"Hierarchy for $C=57$: one root split, then leaf-level classification",
        ha="center",
        fontsize=10,
        color="#333",
    )

    out = OUT_DIR / "tabiclv2-hierarchy-c57.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", out)


if __name__ == "__main__":
    hierarchy_c57()
