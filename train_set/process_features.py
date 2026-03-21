#!/usr/bin/env python3
"""
process_features.py — PMU CSV → MPKI 时间序列张量
===================================================

从 collect_dataset.sh 生成的 PMU CSV 文件中提取特征，
计算每个采样周期的 MPKI（Misses Per Kilo Instructions）指标，
并将时间序列归一化、对齐到统一长度，保存为 .npy 文件。

特征维度 D=6：
  [0] icache_mpki       = L1-icache-load-misses  / inst_retired.any * 1000
  [1] itlb_miss_mpki    = iTLB-load-misses        / inst_retired.any * 1000
  [2] branch_miss_mpki  = branch-misses           / inst_retired.any * 1000
  [3] itlb_mpki         = iTLB-loads              / inst_retired.any * 1000
  [4] branch_mpki       = branch-instructions     / inst_retired.any * 1000
  [5] lbr_log1p_span    = log(1 + avg_lbr_span)   （pmu_monitor 已计算）

输出张量形状：(T×D) = (SEQ_LEN × 6)，dtype=float32
  - 不足 SEQ_LEN 行：末尾补零
  - 超过 SEQ_LEN 行：截取前 SEQ_LEN 行

用法：
  # 处理单个 CSV
  python3 process_features.py data/Bubblesort_v1.csv

  # 处理 manifest.jsonl 中所有条目
  python3 process_features.py --manifest manifest.jsonl [--seq-len 60] [--features-dir data/features]
"""

import sys
import json
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── 特征列定义 ────────────────────────────────────────────────────────────────
# (分子列名, 分母列名, 缩放系数, 特征名)
MPKI_DEFS = [
    ("L1-icache-load-misses", "inst_retired.any", 1000.0, "icache_mpki"),
    ("iTLB-load-misses",      "inst_retired.any", 1000.0, "itlb_miss_mpki"),
    ("branch-misses",         "inst_retired.any", 1000.0, "branch_miss_mpki"),
    ("iTLB-loads",            "inst_retired.any", 1000.0, "itlb_mpki"),
    ("branch-instructions",   "inst_retired.any", 1000.0, "branch_mpki"),
]
LBR_COL = "lbr_log1p_span"

FEATURE_NAMES = [d[3] for d in MPKI_DEFS] + ["lbr_log1p_span"]
D = len(FEATURE_NAMES)  # = 6


def load_csv_to_tensor(csv_path: str, seq_len: int = 60) -> np.ndarray:
    """
    读取 pmu_monitor CSV，计算 MPKI 特征，返回 (seq_len × D) float32 数组。

    处理逻辑：
      1. 过滤掉任意关键列为 'N/A' 的行
      2. 计算 MPKI 和 LBR 特征
      3. 对无穷大 / NaN 值替换为 0
      4. 按列做 min-max 归一化（防止量纲差异主导梯度）
      5. 截断或补零到 seq_len
    """
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    # 过滤含 N/A 的行
    key_cols = ["inst_retired.any"] + [d[0] for d in MPKI_DEFS] + [LBR_COL]
    for col in key_cols:
        if col not in df.columns:
            raise ValueError(f"CSV '{csv_path}' 缺少列: {col}")
    df = df[~df[key_cols].isin(["N/A", "n/a", ""]).any(axis=1)].copy()

    if len(df) == 0:
        # 文件无有效行，返回全零张量
        return np.zeros((seq_len, D), dtype=np.float32)

    # 转换为数值，异常值设为 NaN
    for col in key_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=key_cols)
    if len(df) == 0:
        return np.zeros((seq_len, D), dtype=np.float32)

    # ── 计算 MPKI 特征 ────────────────────────────────────────────────────────
    inst = df["inst_retired.any"].values.astype(np.float64)
    # 防止除零：指令数为 0 时设为 1（后续 MPKI=0）
    inst = np.where(inst == 0, 1.0, inst)

    features = []
    for (num_col, denom_col, scale, _) in MPKI_DEFS:
        num = df[num_col].values.astype(np.float64)
        mpki = num / inst * scale
        features.append(mpki)

    # LBR 特征（pmu_monitor 已完成 log(1+avg_span) 变换）
    lbr = df[LBR_COL].values.astype(np.float64)
    features.append(lbr)

    X = np.stack(features, axis=1)  # shape: (T_raw, D)

    # 替换 inf / NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 按列 min-max 归一化 ───────────────────────────────────────────────────
    col_min = X.min(axis=0, keepdims=True)
    col_max = X.max(axis=0, keepdims=True)
    denom = col_max - col_min
    denom = np.where(denom == 0, 1.0, denom)
    X = (X - col_min) / denom

    # ── 对齐到 seq_len ────────────────────────────────────────────────────────
    T_raw = X.shape[0]
    if T_raw >= seq_len:
        X = X[:seq_len]
    else:
        pad = np.zeros((seq_len - T_raw, D), dtype=np.float64)
        X = np.vstack([X, pad])

    return X.astype(np.float32)


def process_manifest(manifest_path: str, features_dir: str, seq_len: int) -> None:
    """
    按照 manifest.jsonl 批量处理所有 CSV，
    将 .npy 特征文件保存到 features_dir。
    """
    manifest_path = Path(manifest_path)
    features_dir  = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    # manifest 路径的基准目录，用于解析相对 CSV 路径
    base_dir = manifest_path.parent.parent  # train_set/../ = project root

    total = ok = skip = 0
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            prog = entry["program"]
            total += 1

            csv_v1 = base_dir / entry["v1_csv"]
            csv_v2 = base_dir / entry["v2_csv"]

            try:
                X_v1 = load_csv_to_tensor(csv_v1, seq_len)
                X_v2 = load_csv_to_tensor(csv_v2, seq_len)
            except Exception as e:
                print(f"[SKIP] {prog}: {e}", file=sys.stderr)
                skip += 1
                continue

            npy_v1 = features_dir / f"{prog}_v1.npy"
            npy_v2 = features_dir / f"{prog}_v2.npy"
            np.save(npy_v1, X_v1)
            np.save(npy_v2, X_v2)
            print(f"[OK]   {prog}  shape={X_v1.shape}  label={entry['label']}")
            ok += 1

    print(f"\n完成: 总计={total}  成功={ok}  跳过={skip}")
    print(f"特征文件保存在: {features_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PMU CSV → MPKI 张量")
    parser.add_argument("csv_or_manifest", nargs="?",
                        help="单个 CSV 文件路径，或省略后用 --manifest 批量处理")
    parser.add_argument("--manifest", default=None,
                        help="manifest.jsonl 文件路径（批量模式）")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="时间序列长度 T（默认 60）")
    parser.add_argument("--features-dir",
                        default=str(Path(__file__).parent / "data" / "features"),
                        help="批量模式下 .npy 输出目录")
    args = parser.parse_args()

    if args.manifest:
        process_manifest(args.manifest, args.features_dir, args.seq_len)
    elif args.csv_or_manifest:
        X = load_csv_to_tensor(args.csv_or_manifest, args.seq_len)
        print(f"张量形状: {X.shape}")
        print(f"特征列: {FEATURE_NAMES}")
        print(f"各列均值: {X.mean(axis=0).tolist()}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
