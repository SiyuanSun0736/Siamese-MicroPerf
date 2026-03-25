#!/usr/bin/env python3
"""
build_dataset_instret.py — 退役指令总数比标签机制
=================================================

与 build_dataset.py（固定时间 / 循环次数比标签）的核心区别
-----------------------------------------------------------
  build_dataset.py :
    - 标签 Y = N_v1 / N_v2（固定时间窗口内循环执行次数之比）
    - 循环次数由外部 harness 计数，属于间接度量

  本模块 :
    - 标签 Y = Σinst_v1 / Σinst_v2（退役指令总数之比）
    - 直接从 PMU 硬件计数器 inst_retired.any 的时间序列求和
    - 在固定物理采样窗口（PMU_WINDOW=30s）下，CPU 时钟周期数恒定，
      退役指令总数被等价映射为"工作吞吐量"
    - 无需依赖外部 run_count，标签完全由硬件事件驱动

物理依据
--------
  固定采样窗口 T_total = 30s，采样间隔 Δt = 500ms → ~60 行。

  每行记录 inst_retired.any_t（该 Δt 内退役的指令数），对所有有效行求和：
    Σinst_i = Σ_t inst_retired.any_t   （版本 i 在 30s 内完成的总指令数）

  标签:
    Y = Σinst_v1 / Σinst_v2
    Y > 1 → v1 在相同时间内执行了更多指令 → v1 吞吐量更高
    Y < 1 → v2 吞吐量更高

  关键性质:
    - 与 run_count 标签的方向一致（高性能版本 → 更大的分子）
    - 消除了 run_count 计数中循环开销、harness 噪声等外部因素
    - inst_retired.any 是 PMU 最基础的硬件事件，精度高、抖动小

特征工程流水线（与 build_dataset.py 一致）
------------------------------------------
  1. PMU → MPKI 转换
  2. LBR log1p_span 保留
  3. 截断/填充到 seq_len
  4. Z-score 标准化（仅在有效区间）

输出
----
  train_set/tensors/inst_retired/<pair_name>/
    X_v1.pt        — shape (N_samples, T, D) float32
    X_v2.pt        — shape (N_samples, T, D) float32
    Y.pt           — shape (N_samples,) float32,  Y = Σinst_v1 / Σinst_v2
    len_v1.pt      — shape (N_samples,) int64,    v1 有效序列长度
    len_v2.pt      — shape (N_samples,) int64,    v2 有效序列长度
    programs.json  — 样本索引 → 程序名映射
    stats.json     — 归一化统计量 + 标签机制元数据

用法
----
  python3 python/build_dataset_instret.py
  python3 python/build_dataset_instret.py --seq-len 60
  python3 python/build_dataset_instret.py --pairs O1-g:O3-g
  python3 python/build_dataset_instret.py --no-zscore
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

# 复用 build_dataset 中的常量和基础函数
from build_dataset import (
    PMU_COUNTER_COLS,
    INST_COL,
    LBR_FEATURE_COL,
    DEFAULT_PAIRS,
    load_manifest,
)

logger = logging.getLogger(__name__)


# ── 带指令总数的特征提取 ──────────────────────────────────────────────────────

def extract_features_with_instsum(
    csv_path: Path, seq_len: int
) -> tuple[np.ndarray, int, float] | None:
    """
    从单个 PMU CSV 提取特征矩阵 (T, D)、有效长度、以及退役指令总数。

    与 build_dataset.extract_features 的唯一差异：额外返回 Σinst_retired.any。

    返回 (feat_matrix, valid_len, inst_sum)，失败返回 None。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("无法读取 %s: %s", csv_path, e)
        return None

    required = [INST_COL, LBR_FEATURE_COL] + list(PMU_COUNTER_COLS.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("%s 缺少列: %s", csv_path, missing)
        return None

    inst: np.ndarray = df[INST_COL].to_numpy(dtype=np.float64)

    # 丢弃 I_t=0 的行
    valid_mask = inst > 0
    if valid_mask.sum() == 0:
        logger.warning("%s: 所有行 inst=0，跳过", csv_path)
        return None
    df = df.loc[valid_mask].reset_index(drop=True)
    inst = inst[valid_mask]

    # ── 退役指令总数：对所有有效行的 inst_retired.any 求和 ──
    inst_sum = float(inst.sum())

    # ── 特征工程：PMU → MPKI + LBR ──
    features = []
    for raw_col, _feat_name in PMU_COUNTER_COLS.items():
        mpki = df[raw_col].values.astype(np.float64) / inst * 1000.0
        features.append(mpki)

    lbr = df[LBR_FEATURE_COL].values.astype(np.float64)
    features.append(lbr)

    feat_matrix = np.column_stack(features)
    T_raw, D = feat_matrix.shape

    # 截断或零填充到 seq_len
    valid_len = min(T_raw, seq_len)
    if T_raw >= seq_len:
        feat_matrix = feat_matrix[:seq_len]
    else:
        pad = np.zeros((seq_len - T_raw, D), dtype=np.float64)
        feat_matrix = np.concatenate([feat_matrix, pad], axis=0)

    return feat_matrix.astype(np.float32), valid_len, inst_sum


# ── 数据集构建 ────────────────────────────────────────────────────────────────

def build_pair_dataset_instret(
    v1_name: str,
    v2_name: str,
    project_root: Path,
    seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float],
           list[str], list[int], list[int],
           list[float], list[float]]:
    """
    为一个版本对构建基于退役指令总数比标签的数据集。

    与 build_dataset.build_pair_dataset 的差异:
      - 标签 Y = Σinst_v1 / Σinst_v2（而非 N_v1 / N_v2）
      - 额外返回每个样本的 inst_sum_v1, inst_sum_v2 用于诊断

    Returns
    -------
    X_v1_list, X_v2_list : 特征矩阵列表 (seq_len, D)
    Y_list               : 标签列表（退役指令总数比）
    programs             : 程序名列表
    len_v1_list, len_v2_list : 有效序列长度列表
    instsum_v1_list, instsum_v2_list : 退役指令总数列表
    """
    manifest_dir = project_root / "train_set"
    m1_path = manifest_dir / f"manifest_{v1_name}.jsonl"
    m2_path = manifest_dir / f"manifest_{v2_name}.jsonl"

    if not m1_path.exists():
        logger.error("manifest 不存在: %s", m1_path)
        return [], [], [], [], [], [], [], []
    if not m2_path.exists():
        logger.error("manifest 不存在: %s", m2_path)
        return [], [], [], [], [], [], [], []

    m1 = load_manifest(m1_path)
    m2 = load_manifest(m2_path)

    common_programs = sorted(set(m1.keys()) & set(m2.keys()))
    if not common_programs:
        logger.warning("%s 与 %s 无公共程序", v1_name, v2_name)
        return [], [], [], [], [], [], [], []

    X_v1_list, X_v2_list, Y_list, programs = [], [], [], []
    len_v1_list, len_v2_list = [], []
    instsum_v1_list, instsum_v2_list = [], []

    for prog in common_programs:
        entry1, entry2 = m1[prog], m2[prog]

        csv1 = project_root / entry1["csv"]
        csv2 = project_root / entry2["csv"]

        if not csv1.exists() or not csv2.exists():
            logger.warning(
                "%s: CSV 文件缺失 (v1=%s, v2=%s)", prog, csv1.exists(), csv2.exists())
            continue

        result1 = extract_features_with_instsum(csv1, seq_len)
        result2 = extract_features_with_instsum(csv2, seq_len)

        if result1 is None or result2 is None:
            continue

        feat1, vlen1, inst_sum1 = result1
        feat2, vlen2, inst_sum2 = result2

        # 退役指令总数为零 → 程序未实际执行，跳过
        if inst_sum1 <= 0:
            logger.warning("%s: v1 Σinst=0，跳过", prog)
            continue
        if inst_sum2 <= 0:
            logger.warning("%s: v2 Σinst=0，跳过", prog)
            continue

        # ── 标签: Y = Σinst_v1 / Σinst_v2 ──
        # Y > 1 → v1 退役指令更多 → v1 吞吐量更高
        # Y < 1 → v2 退役指令更多 → v2 吞吐量更高
        Y = inst_sum1 / inst_sum2

        X_v1_list.append(feat1)
        X_v2_list.append(feat2)
        Y_list.append(Y)
        programs.append(prog)
        len_v1_list.append(vlen1)
        len_v2_list.append(vlen2)
        instsum_v1_list.append(inst_sum1)
        instsum_v2_list.append(inst_sum2)

        logger.debug(
            "  %s: Σinst_v1=%.0f  Σinst_v2=%.0f  Y=%.4f",
            prog, inst_sum1, inst_sum2, Y)

    return (X_v1_list, X_v2_list, Y_list, programs,
            len_v1_list, len_v2_list,
            instsum_v1_list, instsum_v2_list)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="退役指令总数比标签 — 特征对齐与张量构造")
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
        help="输出目录（默认 train_set/tensors/inst_retired）")
    args = parser.parse_args()

    project_root = args.project_root
    output_base = args.output_dir or (project_root / "train_set" / "tensors" / "inst_retired")

    # ── 配置日志 ──
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"build_instret_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    # ── 解析版本对 ──
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

    logger.info("=" * 60)
    logger.info("退役指令总数比标签 — 特征对齐与张量构造")
    logger.info("=" * 60)
    logger.info("项目根目录: %s", project_root)
    logger.info("序列长度 T: %d", args.seq_len)
    logger.info("特征维度 D: %d  %s", D, feature_names)
    logger.info("Z-score:    %s", '开启' if not args.no_zscore else '关闭')
    logger.info("版本对:     %s", pairs)
    logger.info("")
    logger.info("标签语义: Y = Σinst_v1 / Σinst_v2（退役指令总数比）")
    logger.info("  Y > 1 → v1 吞吐量更高（退役指令更多）")
    logger.info("  Y < 1 → v2 吞吐量更高（退役指令更多）")
    logger.info("  来源列: %s（对有效行求和）", INST_COL)
    logger.info("")

    for v1_name, v2_name in pairs:
        pair_name = f"{v1_name}_vs_{v2_name}"
        logger.info("%s", '=' * 60)
        logger.info("构建 Pair: %s vs %s  (inst_retired 标签)", v1_name, v2_name)
        logger.info("%s", '=' * 60)

        (X_v1_list, X_v2_list, Y_list, programs,
         len_v1_list, len_v2_list,
         instsum_v1_list, instsum_v2_list) = \
            build_pair_dataset_instret(
                v1_name, v2_name, project_root, args.seq_len)

        if not programs:
            logger.warning("无有效样本，跳过此对")
            continue

        N = len(programs)
        logger.info("  有效样本数: %d", N)

        # 堆叠为 (N, T, D)
        X_v1 = np.stack(X_v1_list, axis=0)
        X_v2 = np.stack(X_v2_list, axis=0)
        Y = np.array(Y_list, dtype=np.float32)
        len_v1 = np.array(len_v1_list, dtype=np.int64)
        len_v2 = np.array(len_v2_list, dtype=np.int64)
        instsum_v1 = np.array(instsum_v1_list, dtype=np.float64)
        instsum_v2 = np.array(instsum_v2_list, dtype=np.float64)

        # ── 标签与 run_count 标签对比诊断 ──
        logger.info("  Y (inst_retired 比) 统计: min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                     Y.min(), Y.max(), Y.mean(), Y.std())
        logger.info("  Σinst_v1 统计: min=%.2e  max=%.2e  mean=%.2e",
                     instsum_v1.min(), instsum_v1.max(), instsum_v1.mean())
        logger.info("  Σinst_v2 统计: min=%.2e  max=%.2e  mean=%.2e",
                     instsum_v2.min(), instsum_v2.max(), instsum_v2.mean())

        # ── Z-score 标准化（仅有效区间） ──
        if not args.no_zscore:
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
            mu = (valid_sum / valid_count).astype(np.float32)
            sigma = np.sqrt(valid_sq_sum / valid_count - mu ** 2).astype(np.float32)
            sigma = np.where(sigma == 0, 1.0, sigma)

            for i in range(N):
                X_v1[i, :len_v1[i], :] = (X_v1[i, :len_v1[i], :] - mu) / sigma
                X_v2[i, :len_v2[i], :] = (X_v2[i, :len_v2[i], :] - mu) / sigma
        else:
            mu = np.zeros(D, dtype=np.float32)
            sigma = np.ones(D, dtype=np.float32)

        # ── 转为 PyTorch 张量 ──
        X_v1_tensor = torch.from_numpy(X_v1.astype(np.float32))
        X_v2_tensor = torch.from_numpy(X_v2.astype(np.float32))
        Y_tensor = torch.from_numpy(Y)
        len_v1_tensor = torch.from_numpy(len_v1)
        len_v2_tensor = torch.from_numpy(len_v2)

        # ── 保存 ──
        out_dir = output_base / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(X_v1_tensor, out_dir / "X_v1.pt")
        torch.save(X_v2_tensor, out_dir / "X_v2.pt")
        torch.save(Y_tensor, out_dir / "Y.pt")
        torch.save(len_v1_tensor, out_dir / "len_v1.pt")
        torch.save(len_v2_tensor, out_dir / "len_v2.pt")

        with open(out_dir / "programs.json", "w") as f:
            json.dump(programs, f, indent=2)

        stats = {
            "mechanism": "inst_retired_ratio",
            "label_semantics": "Y = Σinst_v1 / Σinst_v2 (retired instruction ratio)",
            "label_source_col": INST_COL,
            "feature_names": feature_names,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "seq_len": args.seq_len,
            "v1": v1_name,
            "v2": v2_name,
            "n_samples": N,
            "instsum_v1_mean": float(instsum_v1.mean()),
            "instsum_v2_mean": float(instsum_v2.mean()),
        }
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("  X_v1: %s  X_v2: %s  Y: %s",
                     X_v1_tensor.shape, X_v2_tensor.shape, Y_tensor.shape)
        logger.info("  输出: %s", out_dir)
        logger.info("")

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
