#!/usr/bin/env python3
"""
infer.py — 推理与验证阶段 (Inference / Validation)
===================================================

根据 README §5 的设计，加载训练好的 Siamese-MicroPerf 模型，
对配对版本进行推理，输出预测加速比 Ŷ 及判断结论。

支持两种输入模式
----------------
  1. 张量模式（默认）：直接读取 build_dataset.py 生成的 .pt 张量
  2. CSV 模式：读取原始 PMU CSV，实时执行特征工程后推理

输出
----
  对每个程序输出：
    - 预测加速比 Ŷ（标量）
    - 判断结论：Ŷ > 1.0 → v1 优于 v2；Ŷ < 1.0 → v2 优于 v1
    - 若有真实标签 Y，同时输出误差与验证指标

用法
----
  # 对已有张量做推理（自动读取全部版本对）
  python3 python/infer.py --checkpoint python/best_model.pt

  # 只推理指定版本对
  python3 python/infer.py --checkpoint python/best_model.pt \\
      --pairs O2-bolt_vs_O2-bolt-opt

  # 对两个原始 CSV 做单次推理
  python3 python/infer.py --checkpoint python/best_model.pt \\
      --csv-v1 path/to/v1.csv --csv-v2 path/to/v2.csv \\
      --stats train_set/tensors/O2-bolt_vs_O2-bolt-opt/stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import SiameseMicroPerf

# 复用 build_dataset 的特征提取逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import extract_features  # noqa: E402


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device,
               **model_kwargs) -> SiameseMicroPerf:
    """加载训练好的模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SiameseMicroPerf(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def judge(y_hat: float, v1_name: str = "v1", v2_name: str = "v2") -> str:
    """根据预测加速比 Ŷ 输出判断结论。"""
    if y_hat > 1.05:
        pct = (y_hat - 1.0) * 100
        return f"{v1_name} 优于 {v2_name}（快 {pct:.1f}%）"
    elif y_hat < 0.95:
        pct = (1.0 - y_hat) * 100
        return f"{v2_name} 优于 {v1_name}（快 {pct:.1f}%）"
    else:
        return f"{v1_name} ≈ {v2_name}（差异在 ±5% 内）"


# ── 张量模式推理 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_from_tensors(model, tensor_dir: Path, device: torch.device):
    """对已有张量做批量推理，输出逐样本结果。"""
    X_v1 = torch.load(tensor_dir / "X_v1.pt", weights_only=True).to(device)
    X_v2 = torch.load(tensor_dir / "X_v2.pt", weights_only=True).to(device)

    # 标签（可能存在）
    y_path = tensor_dir / "Y.pt"
    Y = torch.load(y_path, weights_only=True) if y_path.exists() else None

    # 程序名映射
    prog_path = tensor_dir / "programs.json"
    programs = json.loads(prog_path.read_text()) if prog_path.exists() else None

    # stats 中记录版本名
    stats_path = tensor_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        v1_name, v2_name = stats.get("v1", "v1"), stats.get("v2", "v2")
    else:
        v1_name, v2_name = "v1", "v2"

    # 批量前向
    Y_hat = model(X_v1, X_v2).cpu()
    N = Y_hat.shape[0]

    return Y_hat, Y, programs, v1_name, v2_name


def print_results(Y_hat, Y, programs, v1_name, v2_name, pair_label: str):
    """打印推理结果表格。"""
    N = Y_hat.shape[0]
    has_label = Y is not None

    print(f"\n{'='*72}")
    print(f"版本对: {pair_label}  ({v1_name} vs {v2_name})  共 {N} 个程序")
    print(f"{'='*72}")

    header = f"  {'#':>4}  {'程序':>30}  {'预测 Ŷ':>8}"
    if has_label:
        header += f"  {'真实 Y':>8}  {'误差':>8}"
    header += f"  {'判断'}"
    print(header)
    print(f"  {'-'*len(header)}")

    all_pred, all_true = [], []
    correct_direction = 0

    for i in range(N):
        y_hat = Y_hat[i].item()
        prog = programs[i] if programs else f"sample_{i}"
        verdict = judge(y_hat, v1_name, v2_name)

        line = f"  {i+1:>4}  {prog:>30}  {y_hat:>8.4f}"
        if has_label:
            y_true = Y[i].item()
            err = y_hat - y_true
            line += f"  {y_true:>8.4f}  {err:>+8.4f}"
            all_pred.append(y_hat)
            all_true.append(y_true)
            # 方向一致性：两者同侧于 1.0
            if (y_hat >= 1.0 and y_true >= 1.0) or (y_hat < 1.0 and y_true < 1.0):
                correct_direction += 1
        line += f"  {verdict}"
        print(line)

    # 汇总统计
    if has_label and len(all_pred) > 0:
        pred = np.array(all_pred)
        true = np.array(all_true)
        mae = np.abs(pred - true).mean()
        mse = ((pred - true) ** 2).mean()
        direction_acc = correct_direction / N * 100

        print(f"\n  ── 验证指标 ──")
        print(f"  MAE  = {mae:.4f}")
        print(f"  MSE  = {mse:.4f}")
        print(f"  RMSE = {np.sqrt(mse):.4f}")
        print(f"  方向准确率 = {correct_direction}/{N} ({direction_acc:.1f}%)")

        # Ŷ > 1 意味着 v1 快；统计预测 v1 优于 v2 的比例
        v1_better_pred = (pred > 1.0).sum()
        v1_better_true = (true > 1.0).sum()
        print(f"  预测 {v1_name} 更优: {v1_better_pred}/{N}  "
              f"(真实: {v1_better_true}/{N})")


# ── CSV 模式推理 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_from_csv(model, csv_v1: Path, csv_v2: Path,
                   stats_path: Path, device: torch.device):
    """对两个原始 PMU CSV 执行实时特征工程并推理。"""
    stats = json.loads(stats_path.read_text())
    seq_len = stats["seq_len"]
    mu = np.array(stats["mu"], dtype=np.float32)
    sigma = np.array(stats["sigma"], dtype=np.float32)
    v1_name = stats.get("v1", "v1")
    v2_name = stats.get("v2", "v2")

    feat1 = extract_features(csv_v1, seq_len)
    feat2 = extract_features(csv_v2, seq_len)
    if feat1 is None or feat2 is None:
        print("[ERROR] 特征提取失败", file=sys.stderr)
        sys.exit(1)

    # Z-score 归一化（复用训练集统计量）
    feat1 = (feat1 - mu) / sigma
    feat2 = (feat2 - mu) / sigma

    # (1, T, D)
    x1 = torch.from_numpy(feat1).unsqueeze(0).to(device)
    x2 = torch.from_numpy(feat2).unsqueeze(0).to(device)

    y_hat = model(x1, x2).item()
    verdict = judge(y_hat, v1_name, v2_name)

    print(f"\n{'='*60}")
    print(f"单次推理结果")
    print(f"{'='*60}")
    print(f"  v1 CSV:  {csv_v1}")
    print(f"  v2 CSV:  {csv_v2}")
    print(f"  预测 Ŷ:  {y_hat:.4f}")
    print(f"  判断:    {verdict}")
    return y_hat


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Siamese-MicroPerf 推理与验证 (§5)")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="模型检查点 .pt 文件")
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="项目根目录")

    # 张量模式参数
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help="版本对目录名（默认全部三组）")

    # CSV 模式参数
    parser.add_argument(
        "--csv-v1", type=Path, default=None,
        help="v1 的原始 PMU CSV 路径")
    parser.add_argument(
        "--csv-v2", type=Path, default=None,
        help="v2 的原始 PMU CSV 路径")
    parser.add_argument(
        "--stats", type=Path, default=None,
        help="归一化统计量 stats.json 路径（CSV 模式必需）")

    # 模型超参（需与训练时一致）
    parser.add_argument("--in-features", type=int, default=6)
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--cnn-out", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(
        args.checkpoint, device,
        in_features=args.in_features,
        cnn_hidden=args.cnn_hidden,
        cnn_out=args.cnn_out,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    )
    print(f"模型加载完成: {args.checkpoint}  (设备: {device})")

    # ── CSV 模式 ──
    if args.csv_v1 and args.csv_v2:
        if not args.stats:
            print("[ERROR] CSV 模式需要 --stats 参数", file=sys.stderr)
            sys.exit(1)
        infer_from_csv(model, args.csv_v1, args.csv_v2, args.stats, device)
        return

    # ── 张量模式 ──
    tensor_base = args.project_root / "train_set" / "tensors"
    pair_names = args.pairs or [
        "O1-g_vs_O3-g",
        "O2-bolt_vs_O2-bolt-opt",
        "O3-bolt_vs_O3-bolt-opt",
    ]

    for pair in pair_names:
        d = tensor_base / pair
        if not d.exists():
            print(f"[WARN] 跳过不存在的目录: {d}", file=sys.stderr)
            continue
        Y_hat, Y, programs, v1_name, v2_name = \
            infer_from_tensors(model, d, device)
        print_results(Y_hat, Y, programs, v1_name, v2_name, pair)

    print()


if __name__ == "__main__":
    main()
