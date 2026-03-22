#!/usr/bin/env python3
"""
filter_lbr.py — 按 LBR 最后三列零值占比筛选 PMU CSV 数据集
=============================================================

PMU 监控窗口固定为 30 s，所有基准都有充足的执行时间。
".runs" 文件记录的是 30 s 内的执行轮数，与 LBR 质量无关。

LBR 最后三列（lbr_samples / lbr_avg_span / lbr_log1p_span）零值过多的根因
-------------------------------------------------------------------------
pmu_monitor 对 PMU 计数器设置了 pe.inherit=1，因此 inst_retired.any /
branch-misses 等计数器可以自动继承并累计子进程的事件。

但 LBR（lbr.c）使用 PERF_SAMPLE_BRANCH_STACK 采样，Linux 内核明确
不支持 PERF_SAMPLE_BRANCH_STACK 与 inherit 同时生效——即使显式设置
inherit，内核也不会将 LBR 采样传播到子进程。

对于通过 fork/exec 把实际计算下放给子进程的工作负载（如 aha、hexxagon、
lambda、JM_lencod 等），父进程几乎不执行任何分支，LBR ring buffer
始终为空，导致三列完全同步地退化为 0。

三列（lbr_samples / lbr_avg_span / lbr_log1p_span）零值完全同步——
只要 lbr_samples==0，另外两列也必为 0，因此用 lbr_log1p_span（模型
直接使用的特征列）的零值占比作为唯一过滤指标即可。

过滤标准（可通过命令行参数调整）
---------------------------------
  --max-zero-rate FLOAT   lbr_log1p_span==0 行的占比上限（默认 0.50）
                           超过此比例则被拒绝
  --min-mean-lbr  FLOAT   非零 lbr_log1p_span 行的均值下限（默认 1.0）

两个条件均满足的数据集才会被复制到输出目录。

用法示例
--------
  # 默认阈值，筛选 O1-g 目录
  python3 train_set/filter_lbr.py \
      --input  train_set/data/O1-g \
      --output train_set/data/O1-g-lbr-filtered

  # 严格模式：零值行不超过 20%，且非零均值 >= 2.0
  python3 train_set/filter_lbr.py \
      --input  train_set/data/O1-g \
      --output train_set/data/O1-g-lbr-filtered \
      --max-zero-rate 0.2 --min-mean-lbr 2.0

  # 只打印统计，不复制文件（dry-run）
  python3 train_set/filter_lbr.py \
      --input  train_set/data/O1-g \
      --output train_set/data/O1-g-lbr-filtered \
      --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


# ── 统计单个 CSV 的 LBR 指标 ──────────────────────────────────────────────────

def lbr_stats(csv_path: Path) -> dict:
    """
    返回字典：
      rows          — 有效行数
      zero_count    — lbr_log1p_span == 0 的行数
      zero_rate     — zero_count / rows
      mean_nonzero  — 仅非零 lbr_log1p_span 行的均值（无非零行则为 0）
      mean_all      — 全部行的 lbr_log1p_span 均值

    三列（lbr_samples / lbr_avg_span / lbr_log1p_span）零值完全同步，
    使用 lbr_log1p_span 作为代表列（即模型直接使用的特征）。
    """
    df = pd.read_csv(csv_path, dtype=str)
    needed = ["lbr_samples", "lbr_avg_span", "lbr_log1p_span"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"CSV '{csv_path}' 缺少列: {col}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    rows = len(df)
    if rows == 0:
        return {"rows": 0, "zero_count": 0, "zero_rate": 0.0,
                "mean_nonzero": 0.0, "mean_all": 0.0}

    nonzero_mask = df["lbr_log1p_span"] > 0
    zero_count   = int((~nonzero_mask).sum())
    zero_rate    = zero_count / rows
    mean_all     = float(df["lbr_log1p_span"].mean())
    mean_nonzero = float(df.loc[nonzero_mask, "lbr_log1p_span"].mean()) if nonzero_mask.any() else 0.0

    return {
        "rows":         rows,
        "zero_count":   zero_count,
        "zero_rate":    zero_rate,
        "mean_nonzero": mean_nonzero,
        "mean_all":     mean_all,
    }


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 LBR 命中质量筛选 PMU CSV 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  required=True,
                        help="输入数据目录（含 *.csv / *.runs 文件）")
    parser.add_argument("--output", required=True,
                        help="输出目录（筛选通过的文件复制至此）")
    parser.add_argument("--max-zero-rate", type=float, default=0.50,
                        help="lbr_log1p_span==0 行的占比上限（默认 0.50，超过则拒绝）")
    parser.add_argument("--min-mean-lbr", type=float, default=1.0,
                        help="非零 lbr_log1p_span 行的均值下限（默认 1.0）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印统计，不复制文件")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] 目录中没有 CSV 文件: {input_dir}", file=sys.stderr)
        sys.exit(0)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    passed = []
    rejected = []

    # 表头
    header = (f"{'文件名':<38} {'rows':>5}  {'零值行':>5}  "
              f"{'零值率':>7}  {'均值(非零)':>10}  {'均值(全部)':>10}  结论")
    print(header)
    print("-" * len(header))

    for csv_path in csv_files:
        try:
            stats = lbr_stats(csv_path)
        except ValueError as e:
            print(f"[SKIP] {csv_path.name}: {e}", file=sys.stderr)
            continue

        ok_zero = stats["zero_rate"]    <= args.max_zero_rate
        ok_mean = stats["mean_nonzero"] >= args.min_mean_lbr

        verdict = "PASS" if (ok_zero and ok_mean) else "REJECT"
        reason  = []
        if not ok_zero:
            reason.append(f"zero_rate={stats['zero_rate']:.2f}>{args.max_zero_rate}")
        if not ok_mean:
            reason.append(f"mean_nonzero={stats['mean_nonzero']:.2f}<{args.min_mean_lbr}")
        reason_str = "  [" + ", ".join(reason) + "]" if reason else ""

        print(
            f"{csv_path.name:<38} {stats['rows']:>5}  {stats['zero_count']:>5}  "
            f"{stats['zero_rate']:>7.2f}  {stats['mean_nonzero']:>10.2f}  "
            f"{stats['mean_all']:>10.2f}  {verdict}{reason_str}"
        )

        if ok_zero and ok_mean:
            passed.append(csv_path)
        else:
            rejected.append(csv_path)

    print("-" * len(header))
    print(f"\n通过: {len(passed)} 个 / 拒绝: {len(rejected)} 个  "
          f"(阈值: zero_rate≤{args.max_zero_rate}, mean_nonzero≥{args.min_mean_lbr})")

    if args.dry_run:
        print("\n[dry-run] 未复制任何文件。")
        return

    # ── 复制通过的 CSV + 对应 .runs 文件 ──────────────────────────────────────
    copied = 0
    for csv_path in passed:
        # 复制 CSV
        dest_csv = output_dir / csv_path.name
        shutil.copy2(csv_path, dest_csv)
        copied += 1

        # 复制对应 .runs 文件（若存在）
        runs_path = csv_path.with_suffix(".runs")
        if runs_path.exists():
            shutil.copy2(runs_path, output_dir / runs_path.name)

    print(f"\n已复制 {copied} 个数据集到: {output_dir}")


if __name__ == "__main__":
    main()
