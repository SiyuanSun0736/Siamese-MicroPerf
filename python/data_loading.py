#!/usr/bin/env python3
"""数据加载与预处理：张量加载、多对合并、数据集划分、对称增强。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch


def load_pair_tensors(tensor_dir: Path):
    """加载一组版本对的张量，返回 (X_v1, X_v2, Y, len_v1, len_v2)。"""
    X_v1 = torch.load(tensor_dir / "X_v1.pt", weights_only=True)
    X_v2 = torch.load(tensor_dir / "X_v2.pt", weights_only=True)
    Y = torch.load(tensor_dir / "Y.pt", weights_only=True)

    # 兼容旧数据集（无 len_*.pt 时用序列全长）
    len_v1_path = tensor_dir / "len_v1.pt"
    len_v2_path = tensor_dir / "len_v2.pt"
    if len_v1_path.exists() and len_v2_path.exists():
        len_v1 = torch.load(len_v1_path, weights_only=True)
        len_v2 = torch.load(len_v2_path, weights_only=True)
    else:
        T = X_v1.shape[1]
        len_v1 = torch.full((X_v1.shape[0],), T, dtype=torch.long)
        len_v2 = torch.full((X_v2.shape[0],), T, dtype=torch.long)

    return X_v1, X_v2, Y, len_v1, len_v2


def merge_pairs(tensor_base: Path, pair_names: list[str]):
    """合并多组版本对的数据。"""
    all_v1, all_v2, all_y, all_lv1, all_lv2 = [], [], [], [], []
    for name in pair_names:
        d = tensor_base / name
        if not d.exists():
            logging.getLogger(__name__).warning("跳过不存在的目录: %s", d)
            continue
        x1, x2, y, lv1, lv2 = load_pair_tensors(d)
        all_v1.append(x1)
        all_v2.append(x2)
        all_y.append(y)
        all_lv1.append(lv1)
        all_lv2.append(lv2)
        logging.getLogger(__name__).info("  加载 %s: %d 样本", name, x1.shape[0])
    if not all_v1:
        print("[ERROR] 无可用数据", file=sys.stderr)
        sys.exit(1)
    return (torch.cat(all_v1), torch.cat(all_v2), torch.cat(all_y),
            torch.cat(all_lv1), torch.cat(all_lv2))


def train_val_test_split(X_v1, X_v2, Y, len_v1, len_v2,
                         val_ratio=0.15, test_ratio=0.15, seed=42):
    """按比例随机划分训练集、验证集和测试集。

    验证集用于早停和模型选择，测试集仅用于最终无偏评估。
    """
    N = X_v1.shape[0]
    indices = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)

    test_idx = torch.tensor(indices[:n_test], dtype=torch.long)
    val_idx = torch.tensor(indices[n_test:n_test + n_val], dtype=torch.long)
    train_idx = torch.tensor(indices[n_test + n_val:], dtype=torch.long)

    return (
        X_v1[train_idx], X_v2[train_idx], Y[train_idx],
        len_v1[train_idx], len_v2[train_idx],
        X_v1[val_idx], X_v2[val_idx], Y[val_idx],
        len_v1[val_idx], len_v2[val_idx],
        X_v1[test_idx], X_v2[test_idx], Y[test_idx],
        len_v1[test_idx], len_v2[test_idx],
    )


def augment_pair_swap(X_v1, X_v2, Y, len_v1, len_v2, log_target: bool):
    """对称增强：添加 (v2, v1) 反转对，在 log 空间下标签取反，在原始空间下标签取倒数。

    将数据量翻倍，并让模型学到反对称性质。"""
    if log_target:
        Y_swap = -Y  # log(1/r) = -log(r)
    else:
        Y_swap = 1.0 / Y
    return (
        torch.cat([X_v1, X_v2]),
        torch.cat([X_v2, X_v1]),
        torch.cat([Y, Y_swap]),
        torch.cat([len_v1, len_v2]),
        torch.cat([len_v2, len_v1]),
    )
