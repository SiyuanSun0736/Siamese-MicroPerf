#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "checkpoints" / "transformer" / "fixed_work_best" / "infer_20260330_154532.log"
V1_NAME = "O3-bolt"
V2_NAME = "O3-bolt-opt"
PROGRAM = "DOE-ProxyApps-C_XSBench"
PAIR_KEY = f"{V1_NAME}_vs_{V2_NAME}"

PMU_FEATURES = {
    "L1-icache-load-misses": "L1I miss MPKI",
    "iTLB-load-misses": "iTLB miss MPKI",
    "branch-misses": "Branch miss MPKI",
    "lbr_log1p_span": "LBR log-span",
}
INST_COL = "inst_retired.any"

DOCS_OUTPUT_DIR = ROOT / "docs" / "diagrams"
PAPER_OUTPUT_DIR = ROOT / "spie-proceedings-style"
STEM = "bolt-case-study-xsbench"


def load_manifest(path: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        entries[record["program"]] = record
    return entries


def parse_prediction(log_path: Path, pair_key: str, program: str) -> tuple[float, float]:
    pair_header = re.compile(r"版本对:\s+([^\s]+)")
    sample_line = re.compile(
        r"INFO:\s+\d+\s+(?P<program>.+?)\s+(?P<pred>-?\d+\.\d+)\s+(?P<true>-?\d+\.\d+)\s+(?P<err>[+-]?\d+\.\d+)\s+"
    )

    current_pair: str | None = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        pair_match = pair_header.search(raw_line)
        if pair_match:
            current_pair = pair_match.group(1)
            continue

        sample_match = sample_line.search(raw_line)
        if sample_match and current_pair == pair_key and sample_match.group("program").strip() == program:
            return float(sample_match.group("true")), float(sample_match.group("pred"))

    raise ValueError(f"Could not find {program} under {pair_key} in {log_path}")


def extract_fixed_work_means(variant: str, program: str, n_ref: int) -> tuple[pd.Series, int]:
    manifest = load_manifest(ROOT / "train_set" / f"manifest_{variant}.jsonl")
    entry = manifest[program]
    df = pd.read_csv(ROOT / entry["csv"])
    df = df[df[INST_COL] > 0].reset_index(drop=True)
    run_count = entry["run_count"]
    effective_len = max(1, min(round(len(df) * n_ref / run_count), len(df)))
    df = df.iloc[:effective_len].copy()

    inst = df[INST_COL].astype(float)
    feature_values = {
        "L1-icache-load-misses": df["L1-icache-load-misses"].astype(float) / inst * 1000.0,
        "iTLB-load-misses": df["iTLB-load-misses"].astype(float) / inst * 1000.0,
        "branch-misses": df["branch-misses"].astype(float) / inst * 1000.0,
        "lbr_log1p_span": df["lbr_log1p_span"].astype(float),
    }
    return pd.DataFrame(feature_values).mean(), effective_len


def build_case_study() -> tuple[float, float, dict[str, float], int, int]:
    manifest_v1 = load_manifest(ROOT / "train_set" / f"manifest_{V1_NAME}.jsonl")
    manifest_v2 = load_manifest(ROOT / "train_set" / f"manifest_{V2_NAME}.jsonl")
    run_count_v1 = manifest_v1[PROGRAM]["run_count"]
    run_count_v2 = manifest_v2[PROGRAM]["run_count"]
    n_ref = min(run_count_v1, run_count_v2)

    true_y, pred_y = parse_prediction(LOG_PATH, PAIR_KEY, PROGRAM)
    mean_v1, len_v1 = extract_fixed_work_means(V1_NAME, PROGRAM, n_ref)
    mean_v2, len_v2 = extract_fixed_work_means(V2_NAME, PROGRAM, n_ref)
    delta_pct = ((mean_v2 / mean_v1) - 1.0) * 100.0
    deltas = {PMU_FEATURES[key]: float(delta_pct[key]) for key in PMU_FEATURES}
    return true_y, pred_y, deltas, len_v1, len_v2


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_outputs(fig: plt.Figure) -> None:
    DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(DOCS_OUTPUT_DIR / f"{STEM}.{suffix}")
    fig.savefig(PAPER_OUTPUT_DIR / f"{STEM}.pdf")


def main() -> None:
    configure_style()
    true_y, pred_y, deltas, len_v1, len_v2 = build_case_study()

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.05, 1.55]},
    )

    labels = ["Ground truth", "Prediction"]
    values = [true_y, pred_y]
    colors = ["#456b8c", "#cf7c4a"]
    bars = ax_left.bar(labels, values, color=colors, width=0.55)
    ax_left.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    ax_left.set_ylim(0.85, 1.05)
    ax_left.set_ylabel(r"Multiplier $Y$ for O3-bolt / O3-bolt-opt")
    ax_left.set_title("Relative Multiplier")
    for bar, value in zip(bars, values):
        speedup = 1.0 / value
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.4f}\n({speedup:.3f}x)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_left.text(
        0.5,
        -0.16,
        f"Effective windows: {len_v1} vs {len_v2}",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        clip_on=False,
    )

    feature_names = list(deltas.keys())
    delta_values = [deltas[name] for name in feature_names]
    bar_colors = ["#4f7f3f" if value < 0 else "#c75b39" for value in delta_values]
    y_pos = list(range(len(feature_names)))
    ax_right.barh(y_pos, delta_values, color=bar_colors, height=0.6)
    ax_right.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(feature_names)
    ax_right.invert_yaxis()
    ax_right.set_xlabel("O3-bolt-opt vs O3-bolt change (%)")
    ax_right.set_title("Key PMU/LBR Shifts")
    for ypos, value in zip(y_pos, delta_values):
        if value >= 0:
            ax_right.text(value + 1.5, ypos, f"{value:+.1f}%", va="center", ha="left", fontsize=8)
        else:
            label_x = value + min(6.0, abs(value) * 0.45)
            ax_right.text(label_x, ypos, f"{value:+.1f}%", va="center", ha="left", fontsize=8)

    fig.suptitle("Case Study: DOE-ProxyApps-C_XSBench under O3-BOLT", fontsize=12)
    save_outputs(fig)


if __name__ == "__main__":
    main()