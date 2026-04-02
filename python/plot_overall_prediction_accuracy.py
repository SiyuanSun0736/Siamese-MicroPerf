#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "docs" / "diagrams"

PAIR_NAMES = ["O1-g", "O2-bolt", "O3-bolt"]
BEST_RUNS = [
    ("CNN", "fixed_time", Path("cnn/fixed_time_best/infer_20260330_213616.log")),
    ("CNN", "fixed_work", Path("cnn/fixed_work_best/infer_20260330_211323.log")),
    ("CNN", "inst", Path("cnn/inst_best/infer_20260330_230630.log")),
    ("LSTM", "fixed_time", Path("lstm/fixed_time_best/infer_20260331_142207.log")),
    ("LSTM", "fixed_work", Path("lstm/fixed_work_best/infer_20260331_145944.log")),
    ("LSTM", "inst", Path("lstm/inst_best/infer_20260331_162147.log")),
    ("Transformer", "fixed_time", Path("transformer/fixed_time_best/infer_20260331_202350.log")),
    ("Transformer", "fixed_work", Path("transformer/fixed_work_best/infer_20260330_154532.log")),
    ("Transformer", "inst", Path("transformer/inst_best/infer_20260331_174437.log")),
]
TRANSFORMER_VARIANTS = {
    "fixed_time": [
        ("v1", Path("transformer/fixed_timev1/infer_20260331_193632.log")),
        ("v2", Path("transformer/fixed_timev2/infer_20260331_201750.log")),
        ("v3", Path("transformer/fixed_timev3/infer_v3.txt")),
        ("best", Path("transformer/fixed_time_best/infer_20260331_202350.log")),
    ],
    "fixed_work": [
        ("v2", Path("transformer/fixed_workv2/infer_v2.log")),
        ("v3", Path("transformer/fixed_workv3/infer_v3.log")),
        ("best", Path("transformer/fixed_work_best/infer_20260330_154532.log")),
    ],
    "inst": [
        ("v1", Path("transformer/inst_retiredv1/infer_20260330_192629.txt")),
        ("v2", Path("transformer/inst_retiredv2/infer_20260331_172047.log")),
        ("v3", Path("transformer/inst_retiredv3/infer_20260331_174437.log")),
        ("best", Path("transformer/inst_best/infer_20260331_174437.log")),
    ],
}
MODEL_COLORS = {
    "CNN": "#1f4e79",
    "LSTM": "#c75b39",
    "Transformer": "#4f7f3f",
}
LABEL_COLORS = {
    "fixed_time": "#6d8ab3",
    "fixed_work": "#d08c60",
    "inst": "#7ea267",
}
PRETTY_LABELS = {
    "fixed_time": "Fixed Time",
    "fixed_work": "Fixed Work",
    "inst": "Inst-Retired",
}


@dataclass
class AccuracyRecord:
    correct: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total


@dataclass
class RunSummary:
    name: str
    pair_results: dict[str, AccuracyRecord]

    @property
    def overall(self) -> AccuracyRecord:
        return AccuracyRecord(
            correct=sum(item.correct for item in self.pair_results.values()),
            total=sum(item.total for item in self.pair_results.values()),
        )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def parse_infer_log(path: Path) -> RunSummary:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"方向准确率\s*=\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)", text)
    if len(matches) != 3:
        raise ValueError(f"Expected 3 direction-accuracy entries in {path}, got {len(matches)}")

    pair_results = {
        pair: AccuracyRecord(correct=int(correct), total=int(total))
        for pair, (correct, total, _pct) in zip(PAIR_NAMES, matches)
    }
    return RunSummary(name=path.stem, pair_results=pair_results)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.svg")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def build_best_summaries() -> dict[tuple[str, str], RunSummary]:
    summaries: dict[tuple[str, str], RunSummary] = {}
    for model, label, rel_path in BEST_RUNS:
        summaries[(model, label)] = parse_infer_log(CHECKPOINTS_DIR / rel_path)
    return summaries


def plot_best_grouped_bars(best_runs: dict[tuple[str, str], RunSummary]) -> None:
    labels = ["fixed_time", "fixed_work", "inst"]
    models = ["CNN", "LSTM", "Transformer"]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for idx, model in enumerate(models):
        values = [best_runs[(model, label)].overall.pct for label in labels]
        offsets = x + (idx - 1) * width
        bars = ax.bar(offsets, values, width=width, label=model, color=MODEL_COLORS[model])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.7,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Overall Prediction Accuracy by Model and Label Mechanism")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_LABELS[label] for label in labels])
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(ncol=3, loc="upper left")

    save_figure(fig, "overall-accuracy-best-grouped")


def plot_best_heatmap(best_runs: dict[tuple[str, str], RunSummary]) -> None:
    row_keys = [
        ("CNN", "fixed_time"),
        ("CNN", "fixed_work"),
        ("CNN", "inst"),
        ("LSTM", "fixed_time"),
        ("LSTM", "fixed_work"),
        ("LSTM", "inst"),
        ("Transformer", "fixed_time"),
        ("Transformer", "fixed_work"),
        ("Transformer", "inst"),
    ]
    matrix = []
    row_labels = []
    for model, label in row_keys:
        summary = best_runs[(model, label)]
        matrix.append([
            summary.pair_results["O1-g"].pct,
            summary.pair_results["O2-bolt"].pct,
            summary.pair_results["O3-bolt"].pct,
            summary.overall.pct,
        ])
        row_labels.append(f"{model} / {PRETTY_LABELS[label]}")

    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    im = ax.imshow(data, cmap="YlGnBu", vmin=50, vmax=90, aspect="auto")

    ax.set_title("Pair-wise and Overall Directional Accuracy (Best Checkpoints)")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["O1-g", "O2-bolt", "O3-bolt", "Overall"])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")

    save_figure(fig, "overall-accuracy-best-heatmap")


def plot_best_model_aggregate(best_runs: dict[tuple[str, str], RunSummary]) -> None:
    models = ["CNN", "LSTM", "Transformer"]
    totals = []
    for model in models:
        correct = sum(best_runs[(model, label)].overall.correct for label in ["fixed_time", "fixed_work", "inst"])
        total = sum(best_runs[(model, label)].overall.total for label in ["fixed_time", "fixed_work", "inst"])
        totals.append(AccuracyRecord(correct=correct, total=total))

    values = [item.pct for item in totals]
    labels = [f"{model}\n{record.correct}/{record.total}" for model, record in zip(models, totals)]

    fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
    bars = ax.bar(models, values, color=[MODEL_COLORS[model] for model in models], width=0.58)
    for bar, value, record in zip(bars, values, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%\n({record.correct}/{record.total})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("Best Checkpoints: Accuracy Aggregated by Model")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))

    save_figure(fig, "overall-accuracy-best-by-model")


def plot_best_label_aggregate(best_runs: dict[tuple[str, str], RunSummary]) -> None:
    labels = ["fixed_time", "fixed_work", "inst"]
    totals = []
    for label in labels:
        correct = sum(best_runs[(model, label)].overall.correct for model in ["CNN", "LSTM", "Transformer"])
        total = sum(best_runs[(model, label)].overall.total for model in ["CNN", "LSTM", "Transformer"])
        totals.append(AccuracyRecord(correct=correct, total=total))

    values = [t.pct for t in totals]
    pretty = [PRETTY_LABELS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    bars = ax.bar(pretty, values, color=[LABEL_COLORS[l] for l in labels], width=0.58)
    for bar, value, record in zip(bars, values, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.2f}%\n({record.correct}/{record.total})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("Best Checkpoints: Accuracy Aggregated by Label Mechanism")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))

    save_figure(fig, "overall-accuracy-best-by-label")


def plot_transformer_variants() -> None:
    families = ["fixed_time", "fixed_work", "inst"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True, sharey=True)

    for ax, family in zip(axes, families):
        entries = TRANSFORMER_VARIANTS[family]
        names = []
        values = []
        for name, rel_path in entries:
            if family == "inst" and name == "best":
                continue
            summary = parse_infer_log(CHECKPOINTS_DIR / rel_path)
            label = "v3 (=best)" if family == "inst" and name == "v3" else name
            names.append(label)
            values.append(summary.overall.pct)

        x = np.arange(len(names))
        ax.plot(
            x,
            values,
            color=LABEL_COLORS[family],
            linewidth=1.8,
            marker="o",
            markersize=6,
        )
        if family == "inst":
            ax.scatter(x[-1], values[-1], s=90, marker="*", color="#2f6f89", zorder=4)
        else:
            ax.scatter(x[-1], values[-1], s=50, color="#2f6f89", zorder=4)

        for xpos, value in zip(x, values):
            ax.text(
                xpos,
                value + 0.6,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(f"Transformer {PRETTY_LABELS[family]}")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xlabel("Run")

    axes[0].set_ylabel("Overall Accuracy (%)")
    fig.suptitle("Transformer Variant Comparison", fontsize=13)

    save_figure(fig, "transformer-variant-accuracy")


def main() -> None:
    configure_style()
    ensure_output_dir()
    best_runs = build_best_summaries()
    plot_best_grouped_bars(best_runs)
    plot_best_heatmap(best_runs)
    plot_best_model_aggregate(best_runs)
    plot_best_label_aggregate(best_runs)
    plot_transformer_variants()
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()