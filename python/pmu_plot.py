#!/usr/bin/env python3
"""
pmu_plot.py  —  PMU Monitor 采集数据可视化
===========================================
读取 pmu_monitor 输出的 CSV 文件，绘制 7 项关键性能计数器的时间序列图，
并根据 test_workload 的 6 个阶段（P1–P6）自动标注背景色带。

PMU 计数器说明
  inst_retired.any       : 退休指令数（P1 应最高）
  branch-instructions    : 分支指令数（P2 应最高）
  branch-misses          : 分支预测失效数（P3 失效率 ≈ 40-50%）
  L1-icache-load-misses  : L1 指令缓存缺失（P5 > P4 ≫ 其他）
  iTLB-loads             : iTLB 访问次数（P4/P5 应最高）
  iTLB-load-misses       : iTLB 缺失次数（P5 ≫ P4 ≈ 0）
  lbr_avg_span           : LBR 平均跳跃跨度（字节；P6 应最高 ≈ 415B）
  lbr_log1p_span         : ln(1 + lbr_avg_span)（对数变换，含伪影处理）

test_workload 阶段（默认每阶段 5 秒）
  P1 INST-PEAK    纯算术 4 路展开，退休指令率最高
  P2 BRANCH-PRED  密集可预测分支（8 条/迭代）
  P3 BRANCH-RAND  随机不可预测分支（4 条/迭代）
  P4 ITLB-WARM    32 页 JIT 代码（常驻 L1 iTLB，无 iTLB miss）
  P5 ITLB-COLD    512 页 JIT 代码（频繁 iTLB miss）
  P6 LBR-WIDE     16 路展开大循环（lbr_avg_span ≈ 415B）

用法
  python3 pmu_plot.py [csv路径] [-o 输出图片] [--no-phases]
  默认读取 log/pmu_monitor_test_workload.csv，输出 python/pmu_plot.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # 非交互式后端，适合服务器/无桌面环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── 阶段定义（相对于 PMU 采集开始时刻，单位 ms）─────────────────────
# 注：test_workload 在 PMU 监控开始前约 1.5 秒启动，因此 P1 只有 ~3.5s 可见
PHASE_DEFS = [
    #  start_ms  end_ms  label              color
    # 现在以每阶段 5000ms 为一个完整周期（P1..P6 各 5s）
    (     0,   5000, "P1: INST-PEAK",   "#AED6F1"),  # 浅蓝
    (  5000,  10000, "P2: BRANCH-PRED", "#A9DFBF"),  # 浅绿
    ( 10000,  15000, "P3: BRANCH-RAND", "#F9E79F"),  # 浅黄
    ( 15000,  20000, "P4: ITLB-WARM",   "#FAD7A0"),  # 浅橙
    ( 20000,  25000, "P5: ITLB-COLD",   "#F1948A"),  # 浅红
    ( 25000,  30000, "P6: LBR-WIDE",    "#C39BD3"),  # 浅紫
]

# 每阶段默认长度（ms）
PHASE_LEN_MS = 5000
# 完整一轮周期（6 个阶段）
FULL_ROUND_MS = PHASE_LEN_MS * 6
# 首轮 P1 实际可见时间（约 3.5s），后续轮次 P1 为完整 5s
FIRST_P1_VISIBLE_MS = 3500
# 首轮总长度 = 首轮 P1 可见 + 其余 5 个阶段完整时长
FIRST_ROUND_MS = FIRST_P1_VISIBLE_MS + PHASE_LEN_MS * 5

# lbr_avg_span 伪影阈值：P4/P5 阶段 JIT 代码地址跨度噪声极大，截断到此值
LBR_CLIP_BYTES = 700


# ── 辅助函数 ─────────────────────────────────────────────────────────

def si_fmt(x, _pos):
    """将数值格式化为 SI 单位（K / M / B）"""
    abs_x = abs(x)
    if abs_x >= 1e9:
        return f"{x / 1e9:.1f}G"
    if abs_x >= 1e6:
        return f"{x / 1e6:.0f}M"
    if abs_x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def pct_fmt(x, _pos):
    return f"{x:.2f}%"


def build_phases(max_ms: float) -> list:
    """展开多轮阶段，覆盖 [0, max_ms]"""
    phases = []
    round_idx = 0
    while True:
        # round 0 offset = 0; subsequent rounds start at FIRST_ROUND_MS + (n-1)*FULL_ROUND_MS
        if round_idx == 0:
            offset = 0
        else:
            offset = FIRST_ROUND_MS + (round_idx - 1) * FULL_ROUND_MS

        # 首轮需要把 P2..P6 向前平移 first_round_shift，以消除 P1 缩短产生的空隙
        full_p1_len = PHASE_DEFS[0][1] - PHASE_DEFS[0][0]
        first_round_shift = full_p1_len - FIRST_P1_VISIBLE_MS

        for start, end, label, color in PHASE_DEFS:
            # 默认按轮偏移
            s = start + offset
            e = end + offset
            if round_idx == 0:
                if label.startswith("P1"):
                    # 首轮 P1 缩短到 FIRST_P1_VISIBLE_MS
                    s = offset + start
                    e = offset + FIRST_P1_VISIBLE_MS
                else:
                    # 首轮其他阶段向前平移，紧接在缩短的 P1 之后
                    s = start + offset - first_round_shift
                    e = end + offset - first_round_shift

            if s >= max_ms:
                return phases

            phases.append((s, min(e, max_ms), label, color, round_idx))

        round_idx += 1


def add_phase_bands(ax, phases, y_label_pos=0.97):
    """在坐标轴上绘制阶段背景色带，并在第一轮顶部添加简短标签"""
    for s, e, label, color, round_idx in phases:
        ax.axvspan(s / 1000, e / 1000, alpha=0.20, color=color, zorder=0, lw=0)
        if round_idx == 0:          # 只在第一轮添加文字标签
            mid = (s + e) / 2 / 1000
            short = label.split(":")[0].strip()   # "P1" ~ "P6"
            ax.text(
                mid, y_label_pos, short,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=6.5, fontweight="bold", color="#444444",
            )


def mux_correction(df: pd.DataFrame, col: str) -> pd.Series:
    """
    计算每个 500ms 区间的多路复用修正系数
        correction = Δtime_enabled / Δtime_running
    当 time_running 趋近于 time_enabled 时修正系数 ≈ 1（无多路复用）。
    """
    te = df[f"{col}_time_enabled"]
    tr = df[f"{col}_time_running"]
    d_te = te.diff().fillna(te.iloc[0])
    d_tr = tr.diff().fillna(tr.iloc[0])
    return d_te / d_tr.clip(lower=1)   # 防止除以 0


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()

    # 时间（秒）
    df["time_s"] = df["elapsed_ms"] / 1000.0

    # 多路复用修正后的关键计数器
    for col in [
        "inst_retired.any",
        "branch-instructions",
        "branch-misses",
        "L1-icache-load-misses",
        "iTLB-loads",
        "iTLB-load-misses",
    ]:
        df[f"{col}_corr"] = df[col] * mux_correction(df, col)

    # 分支预测失效率（%），分母为修正后的分支指令数
    denom = df["branch-instructions_corr"].replace(0, np.nan)
    df["branch_miss_pct"] = df["branch-misses_corr"] / denom * 100

    # lbr_avg_span：截断极端伪影值，单独标记
    df["lbr_span_clipped"] = df["lbr_avg_span"].clip(upper=LBR_CLIP_BYTES)
    df["lbr_span_artifact"] = df["lbr_avg_span"] > LBR_CLIP_BYTES

    return df


# ── 主绘图函数 ────────────────────────────────────────────────────────

def plot(csv_path: str, output_path: str, show_phases: bool = True):
    df = load_and_prepare(csv_path)
    t = df["time_s"]
    max_ms = float(df["elapsed_ms"].max())
    phases = build_phases(max_ms + 500) if show_phases else []

    # ── 图形布局：4 行 × 2 列 ─────────────────────────────────────────
    fig = plt.figure(figsize=(17, 20))
    fig.suptitle(
        "PMU Monitor — Performance Counters Time Series\n"
        f"File: {Path(csv_path).name}   "
        f"Sample interval: 500 ms   Duration: {max_ms/1000:.0f} s",
        fontsize=12, fontweight="bold", y=0.995,
    )

    gs = GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.28,
                  top=0.965, bottom=0.06, left=0.08, right=0.97)
    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(2)]

    LINE_KW = dict(linewidth=1.0, marker="o", markersize=1.8, color="#1F618D")
    GRID_KW = dict(linestyle="--", alpha=0.35, linewidth=0.55)

    # ── 子图定义 ─────────────────────────────────────────────────────
    def draw(ax_i, y_series, title, ylabel,
             log_y=False, yformatter=None, extra_fn=None):
        ax = axes[ax_i]
        y = np.asarray(y_series, dtype=float)
        ax.plot(t, y, **LINE_KW)
        add_phase_bands(ax, phases)
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.grid(True, which="major", **GRID_KW)
        if log_y:
            ax.set_yscale("log")
            ax.grid(True, which="minor", linestyle=":", alpha=0.18, linewidth=0.4)
        elif yformatter:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(yformatter))
        if extra_fn:
            extra_fn(ax)

    # ① Instructions retired
    draw(0,
        df["inst_retired.any_corr"],
        "① Instructions retired / 500 ms (P1 highest)",
        "Instructions", yformatter=si_fmt)

    # ② Branch instructions
    draw(1,
        df["branch-instructions_corr"],
        "② Branch instructions / 500 ms (P2 highest)",
        "Branch instr.", yformatter=si_fmt)

    # ③ Branch misprediction rate
    draw(2,
        df["branch_miss_pct"],
        "③ Branch misprediction rate (P3 expected highest; modern CPUs may optimize)",
        "Mis-prediction (%)", yformatter=pct_fmt)

    # ④ L1 instruction cache misses (log scale)
    draw(3,
        df["L1-icache-load-misses_corr"].clip(lower=1),
        "④ L1 instruction cache misses / 500 ms (P4/P5 highest)",
        "Misses", log_y=True)
    axes[3].yaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))

    # ⑤ iTLB accesses (log scale)
    draw(4,
        df["iTLB-loads_corr"].clip(lower=1),
        "⑤ iTLB accesses / 500 ms (P4/P5 highest)",
        "Accesses", log_y=True)
    axes[4].yaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))

    # ⑥ iTLB load misses (log scale)
    draw(5,
        df["iTLB-load-misses_corr"].clip(lower=1),
        "⑥ iTLB load misses / 500 ms (P5 ≫ P4 ≈ 0)",
        "Misses", log_y=True)
    axes[5].yaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))

    # ⑦ LBR average jump span (after clipping artifacts)
    ax7 = axes[6]
    ok = ~df["lbr_span_artifact"]
    bad = df["lbr_span_artifact"]
    ax7.plot(t[ok], df["lbr_span_clipped"][ok], **LINE_KW, label="Valid values")
    if bad.any():
        ax7.scatter(
            t[bad], [LBR_CLIP_BYTES * 0.97] * bad.sum(),
            color="#E74C3C", s=14, zorder=5, marker="v",
            label=f"Artifact (clipped to {LBR_CLIP_BYTES}B)\nP4/P5 JIT address noise",
        )
    add_phase_bands(ax7, phases)
    ax7.set_title(
        f"⑦ LBR average jump span (clipped >{LBR_CLIP_BYTES} B) (P6 ≈ 415 B, highest)",
        fontsize=9.5, pad=4,
    )
    ax7.set_ylabel("Bytes", fontsize=8)
    ax7.tick_params(axis="both", labelsize=7.5)
    ax7.grid(True, which="major", **GRID_KW)
    ax7.yaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
    ax7.set_ylim(bottom=0)
    ax7.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # ⑧ lbr_log1p_span (natural log transform, overview)
    ax8 = axes[7]
    ax8.plot(t, df["lbr_log1p_span"], **LINE_KW)
    add_phase_bands(ax8, phases)
    ax8.set_title(
        "⑧ lbr_log1p(span) = ln(1 + avg_span) (log-scale overview)",
        fontsize=9.5, pad=4,
    )
    ax8.set_ylabel("ln(1 + span)", fontsize=8)
    ax8.tick_params(axis="both", labelsize=7.5)
    ax8.grid(True, which="major", **GRID_KW)

    # 在 ⑧ 上标注各典型值
    annot = [
        (4.37,  "P1: span≈78\nln≈4.37"),
        (4.26,  "P2: span≈70\nln≈4.26"),
        (4.69,  "P3: span≈108\nln≈4.69"),
        (32.16, "P4/P5: JIT address artifact\nln≈32.16"),
        (6.03,  "P6: span≈415\nln≈6.03"),
    ]
    for yv, txt in annot:
        ax8.axhline(yv, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
        ax8.text(
            0.995, yv, txt,
            transform=ax8.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6, color="#555555",
        )

    # ── X 轴标签（只在底部两图显示）──────────────────────────────────
    for ax in axes[6:]:
        ax.set_xlabel("Time (s)", fontsize=8.5)
    for ax in axes[:6]:
        ax.tick_params(labelbottom=False)

    # 在每张图上添加自动文本对齐的 y 轴网格线
    for ax in axes:
        ax.set_xlim(left=0, right=max_ms / 1000 * 1.005)

    # ── 底部图例（阶段颜色说明）──────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=color, alpha=0.55, label=label)
        for _, _, label, color, *_ in phases[:6]     # 只取第一轮
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=6,
        fontsize=8,
        title="Workload phases (colors repeat for 2nd round)",
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, 0.002),
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    # 保存
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved: {out.resolve()}")
    plt.close(fig)


# ── 命令行入口 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PMU Monitor CSV 数据可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python3 pmu_plot.py\n"
            "  python3 pmu_plot.py log/pmu_monitor.csv -o python/pmu_plot_short.png\n"
            "  python3 pmu_plot.py --no-phases\n"
        ),
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="log/pmu_monitor_test_workload.csv",
        help="CSV 文件路径（默认: log/pmu_monitor_test_workload.csv）",
    )
    parser.add_argument(
        "-o", "--output",
        default="python/plots/pmu_plot.png",
        help="输出图片路径（默认: python/plots/pmu_plot.png）",
    )
    parser.add_argument(
        "--no-phases",
        action="store_true",
        help="不绘制阶段背景色带",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).exists():
        print(f"[Error] 找不到文件: {csv_path}", file=sys.stderr)
        sys.exit(1)

    plot(csv_path, args.output, show_phases=not args.no_phases)


if __name__ == "__main__":
    main()
