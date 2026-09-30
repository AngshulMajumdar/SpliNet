#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "final_metrics.json"
OUT = ROOT / "assets" / "plots"

SCALES = {
    ("fineweb_edu", "validation_loss"): (4.9, 7.5),
    ("fineweb_edu", "validation_perplexity"): (145, 1665),
    ("arxiv", "validation_loss"): (3.15, 6.65),
    ("arxiv", "validation_perplexity"): (20, 675),
    ("pg19", "validation_loss"): (4.82, 7.03),
    ("pg19", "validation_perplexity"): (125, 1035),
    ("wikitext103", "validation_loss"): (4.42, 7.17),
    ("wikitext103", "validation_perplexity"): (80, 1180),
}

DISPLAY = {
    "fineweb_edu": "FineWeb-Edu",
    "arxiv": "arXiv",
    "pg19": "PG-19",
    "wikitext103": "WikiText-103",
}


def make_plot(dataset, metric, rows):
    models = ["SpliNet", "FNet", "Transformer"]
    vals = [next(r[metric] for r in rows if r["model"] == m) for m in models]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(models, vals)
    ylabel = "Validation MLM loss" if metric == "validation_loss" else "MLM perplexity"
    title_metric = "Final validation loss" if metric == "validation_loss" else "Final MLM perplexity"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{DISPLAY[dataset]} — {title_metric}")
    ax.set_ylim(*SCALES[(dataset, metric)])
    ax.text(0.01, 0.98, "Lower is better · y-axis truncated", transform=ax.transAxes,
            va="top", ha="left", fontsize=10)
    span = SCALES[(dataset, metric)][1] - SCALES[(dataset, metric)][0]
    for bar, v in zip(bars, vals):
        fmt = f"{v:.3f}" if metric == "validation_loss" else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012*span, fmt,
                ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "validation_loss" if metric == "validation_loss" else "perplexity"
    fig.savefig(OUT / f"{dataset}_{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    for dataset, rows in data.items():
        make_plot(dataset, "validation_loss", rows)
        make_plot(dataset, "validation_perplexity", rows)


if __name__ == "__main__":
    main()
