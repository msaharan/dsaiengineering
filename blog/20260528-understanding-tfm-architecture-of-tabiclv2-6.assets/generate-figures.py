"""Generate figures for TabICLv2 post 6 (quantile regression). Run from this directory."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent


def pinball_loss(u: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(u >= 0, alpha * u, (alpha - 1) * u)


def pinball_loss_figure() -> None:
    u = np.linspace(-3, 3, 400)
    alphas = [(0.5, "#2563eb"), (0.9, "#dc2626")]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alpha, color in alphas:
        ax.plot(u, pinball_loss(u, alpha), color=color, linewidth=2, label=rf"$\alpha={alpha}$")

    ax.axhline(0, color="#94a3b8", linewidth=0.8)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel(r"Residual $u = y - \hat{q}_\alpha(x)$")
    ax.set_ylabel(r"Pinball loss $\rho_\alpha(u)$")
    ax.set_title("Pinball (quantile) loss — tilted absolute value")
    ax.legend(loc="upper center", frameon=False)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        "underprediction\n($u > 0$)",
        xy=(2.0, pinball_loss(np.array([2.0]), 0.9)[0]),
        xytext=(1.2, 2.2),
        arrowprops=dict(arrowstyle="->", color="#64748b"),
        fontsize=9,
        color="#334155",
    )
    ax.annotate(
        "overprediction\n($u < 0$)",
        xy=(-2.0, pinball_loss(np.array([-2.0]), 0.9)[0]),
        xytext=(-2.8, 1.4),
        arrowprops=dict(arrowstyle="->", color="#64748b"),
        fontsize=9,
        color="#334155",
    )
    ax.text(
        0.02,
        0.02,
        r"For $\alpha=0.9$: slope $0.9$ when $u>0$, slope $0.1$ when $u<0$",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#64748b",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pinball-loss.png", dpi=150)
    plt.close(fig)


def quantile_prediction_interval_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    q05, q50, q95 = 2.2, 5.0, 7.8
    y_line = 1.35

    ax.plot([0.5, 9.5], [y_line, y_line], color="#334155", linewidth=2)
    ax.axvspan(q05, q95, ymin=0.35, ymax=0.65, color="#2563eb", alpha=0.18)

    markers = [
        (q05, r"$\hat{q}_{0.05}$", "5th"),
        (q50, r"$\hat{q}_{0.5}$", "median"),
        (q95, r"$\hat{q}_{0.95}$", "95th"),
    ]
    for x, label, sub in markers:
        ax.plot(x, y_line, "o", color="#2563eb", markersize=9, zorder=3)
        ax.text(x, y_line + 0.55, label, ha="center", va="bottom", fontsize=11)
        ax.text(x, y_line - 0.45, sub, ha="center", va="top", fontsize=8.5, color="#64748b")

    interval_box = FancyBboxPatch(
        (q05, 0.55),
        q95 - q05,
        1.6,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#2563eb",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(interval_box)
    ax.text(
        (q05 + q95) / 2,
        2.55,
        "90% central prediction interval",
        ha="center",
        va="center",
        fontsize=11,
        color="#1e40af",
    )
    ax.text(
        5.0,
        0.2,
        r"Shaded band: $[\hat{q}_{0.05}(x),\ \hat{q}_{0.95}(x)]$",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748b",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "quantile-prediction-interval.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pinball_loss_figure()
    quantile_prediction_interval_figure()
    print("Wrote pinball-loss.png and quantile-prediction-interval.png")
