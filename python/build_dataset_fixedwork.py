#!/usr/bin/env python3
"""
build_dataset_fixedwork.py — 固定工作量机制下的特征对齐与张量构造
================================================================

与 build_dataset.py（固定时间机制）的核心区别
---------------------------------------------
  build_dataset.py :
    - 固定时间窗口（30s），双方 PMU 序列长度 T 相同（均截断/填充到 seq_len=60）
    - 标签 Y = N_v1 / N_v2（固定时间运行次数之比）

  本模块 :
    - 固定工作量（运行次数），PMU 序列长度 T_raw 成为 *动态变量*
    - 标签 Y = T_v2 / T_v1（完成同等工作量所需耗时之比）
    - 更快的版本序列更短，更慢的版本序列更长
    - 序列长度自身携带性能信息，与标签物理一致

机制转换原理
------------
  现有数据在固定时间 T_total=30s 下采集，采样间隔 Δt=500ms → ~60 行。
  若 v1 在 30s 完成 N_v1 次，v2 完成 N_v2 次，则每次耗时 t_i = 30/N_i。

  模拟固定工作量 N_ref = min(N_v1, N_v2):
    - v_i 执行 N_ref 次所需时间: T_i = N_ref × (30 / N_i) = 30 × N_ref / N_i
    - 对应 PMU 行数: rows_i = round(60 × N_ref / N_i)
    - 较快版本（N_i 较大）→ rows_i < 60（截断前部）
    - 较慢版本（N_i = N_ref）→ rows_i = 60（保留全部）

  标签: Y = T_v2 / T_v1 = N_v1 / N_v2
        Y > 1 → v1 较快（v2 更慢）；Y < 1 → v2 较快

  关键性质:
        - 数值上 Y 与原始 N_v1/N_v2 一致，但物理定义从吞吐量比切换为耗时比
    - T_raw 不再是常量——它编码了版本速度差异
    - 模型可从序列长度不对称中学习额外信息

特征工程流水线
--------------
  1. PMU → MPKI 转换（与 build_dataset.py 相同）
  2. LBR log1p_span 保留
  3. 按固定工作量比例截断 PMU 序列（核心差异）
  4. Z-score 标准化（仅在有效区间）
  5. 填充到 max_seq_len，记录有效长度

输出
----
  train_set/tensors_fixedwork/<pair_name>/
    X_v1.pt        — shape (N_samples, max_seq_len, D) float32
    X_v2.pt        — shape (N_samples, max_seq_len, D) float32
    Y.pt           — shape (N_samples,) float32,  Y = T_v2 / T_v1
    len_v1.pt      — shape (N_samples,) int64,    v1 有效序列长度
    len_v2.pt      — shape (N_samples,) int64,    v2 有效序列长度
    programs.json  — 样本索引 → 程序名映射
    stats.json     — 归一化统计量 + 固定工作量元数据

用法
----
  python3 python/build_dataset_fixedwork.py
  python3 python/build_dataset_fixedwork.py --max-seq-len 80
  python3 python/build_dataset_fixedwork.py --pairs O1-g:O3-g
  python3 python/build_dataset_fixedwork.py --no-zscore
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


# ── 固定工作量特征提取 ────────────────────────────────────────────────────────

def extract_features_raw(csv_path: Path) -> tuple[np.ndarray, int] | None:
    """
    从 PMU CSV 提取完整特征矩阵，不做截断/填充（保留原始序列长度）。

    返回 (feat_matrix: np.ndarray (T_raw, D), T_raw: int)，失败返回 None。
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

    valid_mask = inst > 0
    if valid_mask.sum() == 0:
        logger.warning("%s: 所有行 inst=0，跳过", csv_path)
        return None
    df = df.loc[valid_mask].reset_index(drop=True)
    inst = inst[valid_mask]

    features = []
    for raw_col, _feat_name in PMU_COUNTER_COLS.items():
        mpki = df[raw_col].values.astype(np.float64) / inst * 1000.0
        features.append(mpki)

    lbr = df[LBR_FEATURE_COL].values.astype(np.float64)
    features.append(lbr)

    feat_matrix = np.column_stack(features).astype(np.float32)
    T_raw = feat_matrix.shape[0]
    return feat_matrix, T_raw


def compute_effective_length(T_raw: int, n_self: int, n_ref: int) -> int:
    """
    计算固定工作量下的有效 PMU 序列长度。

    Parameters
    ----------
    T_raw  : 原始 PMU 行数（固定时间下采集到的行数）
    n_self : 该版本在固定时间内的运行次数
    n_ref  : 参考工作量（= min(N_v1, N_v2)）

    Returns
    -------
    effective_len : 等效于完成 n_ref 次运行所需的 PMU 行数
    """
    if n_self <= 0 or n_ref <= 0:
        return 0
    # T_effective = T_raw × (n_ref / n_self)
    # n_ref ≤ n_self（因为 n_ref = min），所以 effective_len ≤ T_raw
    effective_len = round(T_raw * n_ref / n_self)
    return max(1, min(effective_len, T_raw))


def pad_to_length(feat: np.ndarray, target_len: int) -> np.ndarray:
    """将 (T, D) 特征矩阵截断或零填充到 (target_len, D)。"""
    T, D = feat.shape
    if T >= target_len:
        return feat[:target_len]
    pad = np.zeros((target_len - T, D), dtype=feat.dtype)
    return np.concatenate([feat, pad], axis=0)


# ── 固定工作量数据集构建 ──────────────────────────────────────────────────────

def build_pair_dataset_fixedwork(
    v1_name: str,
    v2_name: str,
    project_root: Path,
    max_seq_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float],
           list[str], list[int], list[int]]:
    """
    为一个版本对构建固定工作量数据集。

    与 build_dataset.build_pair_dataset 的差异:
      - 不使用固定 seq_len 截断，而是按工作量比例计算有效长度
      - 标签 Y = T_v2 / T_v1（耗时比）
      - 更快版本的序列更短

        注意：虽然标签定义变成了耗时比，但数值方向仍保持
            Y > 1 → v1 更快，Y < 1 → v2 更快

    Returns
    -------
    X_v1_list, X_v2_list : 特征矩阵列表 (max_seq_len, D)
    Y_list               : 标签列表（耗时比）
    programs             : 程序名列表
    len_v1_list, len_v2_list : 有效序列长度列表
    """
    manifest_dir = project_root / "train_set"
    m1_path = manifest_dir / f"manifest_{v1_name}.jsonl"
    m2_path = manifest_dir / f"manifest_{v2_name}.jsonl"

    if not m1_path.exists():
        logger.error("manifest 不存在: %s", m1_path)
        return [], [], [], [], [], []
    if not m2_path.exists():
        logger.error("manifest 不存在: %s", m2_path)
        return [], [], [], [], [], []

    m1 = load_manifest(m1_path)
    m2 = load_manifest(m2_path)

    common_programs = sorted(set(m1.keys()) & set(m2.keys()))
    if not common_programs:
        logger.warning("%s 与 %s 无公共程序", v1_name, v2_name)
        return [], [], [], [], [], []

    X_v1_list, X_v2_list, Y_list, programs = [], [], [], []
    len_v1_list, len_v2_list = [], []

    for prog in common_programs:
        entry1, entry2 = m1[prog], m2[prog]

        csv1 = project_root / entry1["csv"]
        csv2 = project_root / entry2["csv"]

        if not csv1.exists() or not csv2.exists():
            logger.warning(
                "%s: CSV 文件缺失 (v1=%s, v2=%s)", prog, csv1.exists(), csv2.exists())
            continue

        # 读取完整 PMU 序列（不截断）
        result1 = extract_features_raw(csv1)
        result2 = extract_features_raw(csv2)
        if result1 is None or result2 is None:
            continue

        feat1, T_raw1 = result1
        feat2, T_raw2 = result2

        n1 = entry1.get("run_count", 0)
        n2 = entry2.get("run_count", 0)
        if n1 == 0:
            logger.warning("%s: v1 run_count=0，跳过", prog)
            continue
        if n2 == 0:
            logger.warning("%s: v2 run_count=0，跳过", prog)
            continue

        # ── 固定工作量截断 ──
        # N_ref = min(N_v1, N_v2): 以较慢版本的运行次数为基准
        n_ref = min(n1, n2)

        # 计算等效有效长度
        eff_len1 = compute_effective_length(T_raw1, n1, n_ref)
        eff_len2 = compute_effective_length(T_raw2, n2, n_ref)

        # 截断到有效长度，然后填充到 max_seq_len
        feat1_eff = feat1[:eff_len1]
        feat2_eff = feat2[:eff_len2]
        feat1_padded = pad_to_length(feat1_eff, max_seq_len)
        feat2_padded = pad_to_length(feat2_eff, max_seq_len)

        # 确保有效长度不超过 max_seq_len
        eff_len1 = min(eff_len1, max_seq_len)
        eff_len2 = min(eff_len2, max_seq_len)

        # ── 标签: Y = T_v2 / T_v1 ──
        # T_i ∝ N_ref / N_i，故 T_v2/T_v1 = N_v1/N_v2
        # Y > 1 → v2 耗时更长 → v1 更快
        # Y < 1 → v1 耗时更长 → v2 更快
        Y = float(n1) / float(n2)

        X_v1_list.append(feat1_padded)
        X_v2_list.append(feat2_padded)
        Y_list.append(Y)
        programs.append(prog)
        len_v1_list.append(eff_len1)
        len_v2_list.append(eff_len2)

        logger.debug(
            "  %s: N_v1=%d N_v2=%d N_ref=%d → eff_len=(%d, %d) Y=%.4f",
            prog, n1, n2, n_ref, eff_len1, eff_len2, Y)

    return X_v1_list, X_v2_list, Y_list, programs, len_v1_list, len_v2_list


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="固定工作量机制 — 特征对齐与张量构造")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="项目根目录")
    parser.add_argument(
        "--max-seq-len", type=int, default=60,
        help="最大序列长度（填充对齐用，默认 60）")
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help="版本对列表，格式 v1:v2（默认三组）")
    parser.add_argument(
        "--no-zscore", action="store_true",
        help="跳过 Z-score 标准化")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="输出目录（默认 train_set/tensors/fixed_work）")
    args = parser.parse_args()

    project_root = args.project_root
    output_base = args.output_dir or (project_root / "train_set" / "tensors" / "fixed_work")

    # ── 配置日志 ──
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"build_fixedwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    logger.info("固定工作量机制 — 特征对齐与张量构造")
    logger.info("=" * 60)
    logger.info("项目根目录:   %s", project_root)
    logger.info("最大序列长度: %d", args.max_seq_len)
    logger.info("特征维度 D:   %d  %s", D, feature_names)
    logger.info("Z-score:      %s", '开启' if not args.no_zscore else '关闭')
    logger.info("版本对:       %s", pairs)
    logger.info("")
    logger.info("标签语义: Y = T_v2 / T_v1（耗时比）")
    logger.info("  Y > 1 → v1 更快（v2 耗时更长）")
    logger.info("  Y < 1 → v2 更快（v1 耗时更长）")
    logger.info("")

    for v1_name, v2_name in pairs:
        pair_name = f"{v1_name}_vs_{v2_name}"
        logger.info("%s", '=' * 60)
        logger.info("构建 Pair: %s vs %s  (固定工作量)", v1_name, v2_name)
        logger.info("%s", '=' * 60)

        X_v1_list, X_v2_list, Y_list, programs, len_v1_list, len_v2_list = \
            build_pair_dataset_fixedwork(
                v1_name, v2_name, project_root, args.max_seq_len)

        if not programs:
            logger.warning("无有效样本，跳过此对")
            continue

        N = len(programs)
        logger.info("  有效样本数: %d", N)

        # 堆叠为 (N, max_seq_len, D)
        X_v1 = np.stack(X_v1_list, axis=0)
        X_v2 = np.stack(X_v2_list, axis=0)
        Y = np.array(Y_list, dtype=np.float32)
        len_v1 = np.array(len_v1_list, dtype=np.int64)
        len_v2 = np.array(len_v2_list, dtype=np.int64)

        # ── 序列长度统计 ──
        logger.info("  v1 有效长度: min=%d  max=%d  mean=%.1f",
                     len_v1.min(), len_v1.max(), len_v1.mean())
        logger.info("  v2 有效长度: min=%d  max=%d  mean=%.1f",
                     len_v2.min(), len_v2.max(), len_v2.mean())
        len_diff = len_v1.astype(float) - len_v2.astype(float)
        logger.info("  长度差 (v1-v2): mean=%.1f  std=%.1f",
                     len_diff.mean(), len_diff.std())

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
            "mechanism": "fixed_workload",
            "label_semantics": "Y = T_v2 / T_v1 (time ratio)",
            "feature_names": feature_names,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "max_seq_len": args.max_seq_len,
            "v1": v1_name,
            "v2": v2_name,
            "n_samples": N,
            "len_v1_mean": float(len_v1.mean()),
            "len_v2_mean": float(len_v2.mean()),
        }
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("  X_v1: %s  X_v2: %s  Y: %s",
                     X_v1_tensor.shape, X_v2_tensor.shape, Y_tensor.shape)
        logger.info("  Y 统计: min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                     Y.min(), Y.max(), Y.mean(), Y.std())
        logger.info("  输出: %s", out_dir)
        logger.info("")

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
