#!/usr/bin/env python3
"""
build_dataset.py — 异构特征对齐与张量构造
==========================================

根据 README §1–§2 的流程，读取 train_set/data 下的 PMU CSV 数据，
对三组版本对执行特征工程，生成可直接用于 Siamese 网络训练的 PyTorch 张量。

版本对
------
  Pair 1: (O1-g,      O3-g)         — 不同编译优化等级
  Pair 2: (O2-bolt,   O2-bolt-opt)  — BOLT 优化前后
  Pair 3: (O3-bolt,   O3-bolt-opt)  — BOLT 优化前后

特征工程流水线
--------------
  1. PMU 标度统一：将原始计数器转换为 MPKI = C_t / I_t × 1000
     - L1-icache-load-misses  → icache_miss_mpki
     - iTLB-loads             → itlb_load_mpki
     - iTLB-load-misses       → itlb_miss_mpki
     - branch-instructions    → branch_inst_mpki
     - branch-misses          → branch_miss_mpki
  2. LBR 极值压缩：直接使用 CSV 中已计算好的 lbr_log1p_span = log(1 + avg_span)
  3. Z-score 标准化：对全局训练集的每个特征维度计算 μ/σ，逐特征归一化
  4. 拼接输出双塔张量 X_v1, X_v2 ∈ R^{T×D}，标签 Y = N_v1 / N_v2

输出
----
  train_set/tensors/<pair_name>/
    X_v1.pt    — shape (N_samples, T, D) float32
    X_v2.pt    — shape (N_samples, T, D) float32
    Y.pt       — shape (N_samples,) float32      回归标签
    programs.json  — 样本索引 → 程序名映射
    stats.json     — 归一化统计量 { μ, σ, feature_names }

用法
----
  python3 python/build_dataset.py                        # 默认参数
  python3 python/build_dataset.py --seq-len 60           # 指定序列长度
  python3 python/build_dataset.py --no-zscore            # 跳过 Z-score
  python3 python/build_dataset.py --pairs O1-g:O3-g      # 只处理指定对
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import logging
import numpy as np
import pandas as pd
import torch

# ── 常量 ──────────────────────────────────────────────────────────────────────

# 需要转换为 MPKI 的原始 PMU 列 → 输出特征名
PMU_COUNTER_COLS = {
    "L1-icache-load-misses": "icache_miss_mpki",
    "iTLB-loads":            "itlb_load_mpki",
    "iTLB-load-misses":      "itlb_miss_mpki",
    "branch-instructions":   "branch_inst_mpki",
    "branch-misses":         "branch_miss_mpki",
}

# MPKI 除法的分母列
INST_COL = "inst_retired.any"

# LBR 已压缩特征（CSV 中已计算好的 log(1 + avg_span)）
LBR_FEATURE_COL = "lbr_log1p_span"

# 三组默认版本对
DEFAULT_PAIRS = [
    ("O1-g",    "O3-g"),
    ("O2-bolt", "O2-bolt-opt"),
    ("O3-bolt", "O3-bolt-opt"),
]


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """读取 manifest JSONL，返回 { program_name: {variant, csv, run_count, ...} }"""
    entries = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            entries[d["program"]] = d
    return entries


def extract_features(csv_path: Path, seq_len: int) -> tuple[np.ndarray, int] | None:
    """
    从单个 PMU CSV 提取特征矩阵 (T, D) 及有效长度。

    步骤：
      1. 读取 CSV，丢弃 elapsed_ms / timestamp 元数据列
      2. 将 5 个 PMU 计数器转为 MPKI（C_t / I_t × 1000）
      3. 丢弃 I_t=0 的无效行（避免幻影 MPKI 尖刺）
      4. 保留 lbr_log1p_span 作为 LBR 特征
      5. 截断或填充到 seq_len 行，记录有效长度

    返回 (np.ndarray (seq_len, D), valid_len)，失败返回 None。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.getLogger(__name__).warning("无法读取 %s: %s", csv_path, e)
        return None

    # 检查必要列
    required = [INST_COL, LBR_FEATURE_COL] + list(PMU_COUNTER_COLS.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.getLogger(__name__).warning("%s 缺少列: %s", csv_path, missing)
        return None

    # 取指令数作为 MPKI 分母
    inst: np.ndarray = df[INST_COL].to_numpy(dtype=np.float64)

    # 丢弃 I_t=0 的行：这些行代表程序挂起/IO阻塞，MPKI 无物理意义
    valid_mask = inst > 0
    if valid_mask.sum() == 0:
        logging.getLogger(__name__).warning("%s: 所有行 inst=0，跳过", csv_path)
        return None
    df = df.loc[valid_mask].reset_index(drop=True)
    inst = inst[valid_mask]

    features = []

    # PMU → MPKI
    for raw_col, _feat_name in PMU_COUNTER_COLS.items():
        mpki = df[raw_col].values.astype(np.float64) / inst * 1000.0
        features.append(mpki)

    # LBR 压缩特征（已在 CSV 中计算好）
    lbr = df[LBR_FEATURE_COL].values.astype(np.float64)
    features.append(lbr)

    # (T_raw, D)
    feat_matrix = np.column_stack(features)
    T_raw, D = feat_matrix.shape

    # 截断或零填充到 seq_len，记录有效长度
    valid_len = min(T_raw, seq_len)
    if T_raw >= seq_len:
        feat_matrix = feat_matrix[:seq_len]
    else:
        pad = np.zeros((seq_len - T_raw, D), dtype=np.float64)
        feat_matrix = np.concatenate([feat_matrix, pad], axis=0)

    return feat_matrix.astype(np.float32), valid_len


def build_pair_dataset(
    v1_name: str,
    v2_name: str,
    project_root: Path,
    seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[str],
           list[int], list[int]]:
    """
    为一个版本对构建数据集。

    返回:
      X_v1_list  — 每个元素 shape (T, D)
      X_v2_list  — 每个元素 shape (T, D)
      Y_list     — 标签列表 (N_v1 / N_v2)
      programs   — 程序名列表
    """
    manifest_dir = project_root / "train_set"

    m1_path = manifest_dir / f"manifest_{v1_name}.jsonl"
    m2_path = manifest_dir / f"manifest_{v2_name}.jsonl"

    if not m1_path.exists():
        logging.getLogger(__name__).error("manifest 不存在: %s", m1_path)
        return [], [], [], [], [], []
    if not m2_path.exists():
        logging.getLogger(__name__).error("manifest 不存在: %s", m2_path)
        return [], [], [], [], [], []

    m1 = load_manifest(m1_path)
    m2 = load_manifest(m2_path)

    common_programs = sorted(set(m1.keys()) & set(m2.keys()))
    if not common_programs:
        logging.getLogger(__name__).warning("%s 与 %s 无公共程序", v1_name, v2_name)
        return [], [], [], [], [], []

    X_v1_list, X_v2_list, Y_list, programs = [], [], [], []
    len_v1_list, len_v2_list = [], []

    for prog in common_programs:
        entry1, entry2 = m1[prog], m2[prog]

        csv1 = project_root / entry1["csv"]
        csv2 = project_root / entry2["csv"]

        if not csv1.exists() or not csv2.exists():
            logging.getLogger(__name__).warning(
                "%s: CSV 文件缺失 (v1=%s, v2=%s)", prog, csv1.exists(), csv2.exists())
            continue

        result1 = extract_features(csv1, seq_len)
        result2 = extract_features(csv2, seq_len)

        if result1 is None or result2 is None:
            continue

        feat1, vlen1 = result1
        feat2, vlen2 = result2

        # 标签 Y = N_v1 / N_v2
        n1 = entry1.get("run_count", 0)
        n2 = entry2.get("run_count", 0)
        if n2 == 0:
            logging.getLogger(__name__).warning("%s: v2 run_count=0，无法计算标签", prog)
            continue
        if n1 == 0:
            logging.getLogger(__name__).warning("%s: v1 run_count=0，标签为 0（程序可能崩溃/超时），跳过", prog)
            continue

        Y = float(n1) / float(n2)

        X_v1_list.append(feat1)
        X_v2_list.append(feat2)
        Y_list.append(Y)
        programs.append(prog)
        len_v1_list.append(vlen1)
        len_v2_list.append(vlen2)

    return X_v1_list, X_v2_list, Y_list, programs, len_v1_list, len_v2_list


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="异构特征对齐与张量构造 (Feature Engineering)")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="项目根目录（默认自动检测）")
    parser.add_argument(
        "--seq-len", type=int, default=60,
        help="统一序列长度 T（默认 60，即 30s/500ms）")
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help="版本对列表，格式 v1:v2（默认三组）")
    parser.add_argument(
        "--no-zscore", action="store_true",
        help="跳过 Z-score 标准化")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="输出目录（默认 train_set/tensors）")
    args = parser.parse_args()

    project_root = args.project_root
    output_base = args.output_dir or (project_root / "train_set" / "tensors")

    # 配置日志：同时写入文件和输出到控制台
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh = logging.FileHandler(str(log_file), mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    if root_logger.handlers:
        root_logger.handlers = []
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)

    # 解析版本对
    if args.pairs:
        pairs = []
        for p in args.pairs:
            parts = p.split(":")
            if len(parts) != 2:
                print(f"[ERROR] 版本对格式错误: {p}，应为 v1:v2", file=sys.stderr)
                sys.exit(1)
            pairs.append((parts[0], parts[1]))
    else:
        pairs = DEFAULT_PAIRS

    feature_names = list(PMU_COUNTER_COLS.values()) + [LBR_FEATURE_COL]
    D = len(feature_names)

    logging.getLogger(__name__).info("项目根目录: %s", project_root)
    logging.getLogger(__name__).info("序列长度 T: %d", args.seq_len)
    logging.getLogger(__name__).info("特征维度 D: %d  %s", D, feature_names)
    logging.getLogger(__name__).info("Z-score:    %s", '开启' if not args.no_zscore else '关闭')
    logging.getLogger(__name__).info("版本对:     %s", pairs)
    logging.getLogger(__name__).info("")

    for v1_name, v2_name in pairs:
        pair_name = f"{v1_name}_vs_{v2_name}"
        logging.getLogger(__name__).info("%s", '=' * 60)
        logging.getLogger(__name__).info("构建 Pair: %s vs %s", v1_name, v2_name)
        logging.getLogger(__name__).info("%s", '=' * 60)

        X_v1_list, X_v2_list, Y_list, programs, len_v1_list, len_v2_list = \
            build_pair_dataset(v1_name, v2_name, project_root, args.seq_len)

        if not programs:
            logging.getLogger(__name__).warning("无有效样本，跳过此对")
            continue

        N = len(programs)
        logging.getLogger(__name__).info("  有效样本数: %d", N)

        # 堆叠为 (N, T, D)
        X_v1 = np.stack(X_v1_list, axis=0)  # (N, T, D)
        X_v2 = np.stack(X_v2_list, axis=0)  # (N, T, D)
        Y = np.array(Y_list, dtype=np.float32)  # (N,)

        # 构建有效长度数组
        len_v1 = np.array(len_v1_list, dtype=np.int64)
        len_v2 = np.array(len_v2_list, dtype=np.int64)

        # Z-score 标准化：仅在有效（非填充）时间步上计算 μ/σ
        if not args.no_zscore:
            # 掩码感知统计量：只对有效时间步求均值和标准差
            valid_sum = np.zeros(D, dtype=np.float64)
            valid_sq_sum = np.zeros(D, dtype=np.float64)
            valid_count = 0
            for i in range(N):
                valid_sum += X_v1[i, :len_v1[i], :].sum(axis=0)
                valid_sq_sum += (X_v1[i, :len_v1[i], :] ** 2).sum(axis=0)
                valid_count += len_v1[i]
                valid_sum += X_v2[i, :len_v2[i], :].sum(axis=0)
                valid_sq_sum += (X_v2[i, :len_v2[i], :] ** 2).sum(axis=0)
                valid_count += len_v2[i]
            mu = (valid_sum / valid_count).astype(np.float32)  # (D,)
            sigma = np.sqrt(valid_sq_sum / valid_count - mu ** 2).astype(np.float32)  # (D,)
            # 防止除零：标准差为 0 的特征保持原值
            sigma = np.where(sigma == 0, 1.0, sigma)

            # 仅对有效时间步执行归一化，填充区保持为 0
            for i in range(N):
                X_v1[i, :len_v1[i], :] = (X_v1[i, :len_v1[i], :] - mu) / sigma
                X_v2[i, :len_v2[i], :] = (X_v2[i, :len_v2[i], :] - mu) / sigma
        else:
            mu = np.zeros(D, dtype=np.float32)
            sigma = np.ones(D, dtype=np.float32)

        # 转为 PyTorch 张量
        X_v1_tensor = torch.from_numpy(X_v1.astype(np.float32))
        X_v2_tensor = torch.from_numpy(X_v2.astype(np.float32))
        Y_tensor = torch.from_numpy(Y)
        len_v1_tensor = torch.from_numpy(len_v1)
        len_v2_tensor = torch.from_numpy(len_v2)

        # 输出目录
        out_dir = output_base / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(X_v1_tensor, out_dir / "X_v1.pt")
        torch.save(X_v2_tensor, out_dir / "X_v2.pt")
        torch.save(Y_tensor, out_dir / "Y.pt")
        torch.save(len_v1_tensor, out_dir / "len_v1.pt")
        torch.save(len_v2_tensor, out_dir / "len_v2.pt")

        # 程序名映射
        with open(out_dir / "programs.json", "w") as f:
            json.dump(programs, f, indent=2)

        # 归一化统计量（推理时需要复用）
        stats = {
            "feature_names": feature_names,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "seq_len": args.seq_len,
            "v1": v1_name,
            "v2": v2_name,
            "n_samples": N,
        }
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logging.getLogger(__name__).info(
            "  X_v1: %s  X_v2: %s  Y: %s", X_v1_tensor.shape, X_v2_tensor.shape, Y_tensor.shape)
        logging.getLogger(__name__).info(
            "  Y 统计: min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
            Y.min(), Y.max(), Y.mean(), Y.std())
        logging.getLogger(__name__).info("  输出: %s", out_dir)
        logging.getLogger(__name__).info("")

    logging.getLogger(__name__).info("全部完成。")


if __name__ == "__main__":
    main()
