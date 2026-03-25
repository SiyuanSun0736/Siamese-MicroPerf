#!/usr/bin/env python3
"""
train.py — Siamese-MicroPerf 训练 / 验证脚本
=============================================

加载 build_dataset.py 生成的张量，训练 Siamese 1D-CNN + Attention Pooling
网络预测编译版本间的相对加速比。

用法
----
  python3 python/train.py                               # 默认参数
  python3 python/train.py --epochs 200 --lr 1e-3        # 自定义超参
  python3 python/train.py --pairs O1-g_vs_O3-g          # 只用一组对
  python3 python/train.py --eval-only --checkpoint ckpt.pt  # 仅推理
  
说明
----
    默认会将训练过程中表现最好的模型保存到项目根目录下的 `checkpoints/best_model.pt`。
    可以通过 `--checkpoint <path>` 指定要加载或保存的检查点路径以覆盖该默认行为。
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import SiameseMicroPerf

# ── 默认常量 ──────────────────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    "O1-g_vs_O3-g",
    "O2-bolt_vs_O2-bolt-opt",
    "O3-bolt_vs_O3-bolt-opt",
]


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_pair_tensors(tensor_dir: Path):
    """加载一组版本对的张量，返回 (X_v1, X_v2, Y)。"""
    X_v1 = torch.load(tensor_dir / "X_v1.pt", weights_only=True)
    X_v2 = torch.load(tensor_dir / "X_v2.pt", weights_only=True)
    Y = torch.load(tensor_dir / "Y.pt", weights_only=True)
    return X_v1, X_v2, Y


def merge_pairs(tensor_base: Path, pair_names: list[str]):
    """合并多组版本对的数据。"""
    all_v1, all_v2, all_y = [], [], []
    for name in pair_names:
        d = tensor_base / name
        if not d.exists():
            logging.getLogger(__name__).warning("跳过不存在的目录: %s", d)
            continue
        x1, x2, y = load_pair_tensors(d)
        all_v1.append(x1)
        all_v2.append(x2)
        all_y.append(y)
        logging.getLogger(__name__).info("  加载 %s: %d 样本", name, x1.shape[0])
    if not all_v1:
        print("[ERROR] 无可用数据", file=sys.stderr)
        sys.exit(1)
    return torch.cat(all_v1), torch.cat(all_v2), torch.cat(all_y)


def train_val_split(X_v1, X_v2, Y, val_ratio=0.2, seed=42):
    """按比例随机划分训练集和验证集。"""
    N = X_v1.shape[0]
    indices = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(N * (1 - val_ratio))

    train_idx = torch.tensor(indices[:split], dtype=torch.long)
    val_idx = torch.tensor(indices[split:], dtype=torch.long)

    return (
        X_v1[train_idx], X_v2[train_idx], Y[train_idx],
        X_v1[val_idx], X_v2[val_idx], Y[val_idx],
    )


# ── 训练循环 ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x1, x2)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    n = 0
    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        y_hat = model(x1, x2)
        loss = criterion(y_hat, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        all_pred.append(y_hat.cpu())
        all_true.append(y.cpu())

    avg_loss = total_loss / n
    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    mae = (pred - true).abs().mean().item()
    return avg_loss, mae, pred, true


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Siamese-MicroPerf 训练脚本 (§3–§4)")
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="项目根目录")
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help="版本对目录名（默认全部三组）")
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="训练轮数")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="批大小")
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="学习率")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="L2 正则化")
    parser.add_argument(
        "--huber-delta", type=float, default=1.0,
        help="Huber Loss δ 参数")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="验证集比例")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="模型检查点路径（加载/保存）")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="仅评估模式")
    # 模型超参
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--cnn-out", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 配置日志：写入 project_root/log/train_YYYYmmdd_HHMMSS.log
    log_dir = args.project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_file),
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
    )

    logging.getLogger(__name__).info("设备: %s", device)

    # ── 数据加载 ──
    tensor_base = args.project_root / "train_set" / "tensors"
    pair_names = args.pairs or DEFAULT_PAIRS

    logging.getLogger(__name__).info("加载数据...")
    X_v1, X_v2, Y = merge_pairs(tensor_base, pair_names)
    logging.getLogger(__name__).info(
        "总样本数: %d  序列长度 T=%d  特征维度 D=%d",
        X_v1.shape[0], X_v1.shape[1], X_v1.shape[2])

    in_features = X_v1.shape[2]  # D

    # ── 划分训练/验证集 ──
    X_v1_tr, X_v2_tr, Y_tr, X_v1_val, X_v2_val, Y_val = \
        train_val_split(X_v1, X_v2, Y, val_ratio=args.val_ratio, seed=args.seed)

    logging.getLogger(__name__).info("训练集: %d  验证集: %d", Y_tr.shape[0], Y_val.shape[0])

    train_ds = TensorDataset(X_v1_tr, X_v2_tr, Y_tr)
    val_ds = TensorDataset(X_v1_val, X_v2_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── 模型 ──
    model = SiameseMicroPerf(
        in_features=in_features,
        cnn_hidden=args.cnn_hidden,
        cnn_out=args.cnn_out,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
    ).to(device)

    logging.getLogger(__name__).info(
        "\n模型参数: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    logging.getLogger(__name__).info("%s", model)

    # Huber Loss (§4)
    criterion = nn.HuberLoss(delta=args.huber_delta)

    # 加载检查点
    if args.checkpoint and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logging.getLogger(__name__).info("加载检查点: %s", args.checkpoint)

    # ── 评估模式 ──
    if args.eval_only:
        val_loss, val_mae, pred, true = evaluate(model, val_loader, criterion, device)
        logging.getLogger(__name__).info("\n验证集  Loss=%.4f  MAE=%.4f", val_loss, val_mae)
        for i in range(min(10, len(pred))):
            logging.getLogger(__name__).info("  样本 %d: 真实=%.4f  预测=%.4f", i, true[i].item(), pred[i].item())
        return

    # ── 训练 ──
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    # 决定模型保存路径：优先使用 --checkpoint，否则放到 project_root/checkpoints/best_model.pt
    if args.checkpoint:
        save_path = args.checkpoint
    else:
        save_dir = args.project_root / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "best_model.pt"

    logging.getLogger(__name__).info(
        "\n开始训练 (%d epochs, Huber δ=%s)...", args.epochs, args.huber_delta)
    logging.getLogger(__name__).info(
        "%-6s  %-11s  %-10s  %-9s  %-10s  %s",
        'Epoch', 'Train Loss', 'Val Loss', 'Val MAE', 'LR', 'Status')
    logging.getLogger(__name__).info("%s", '-' * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        status = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
            }, save_path)
            status = "← best"

        if epoch % 10 == 0 or epoch == 1 or status:
            logging.getLogger(__name__).info(
                "%6d  %11.6f  %10.6f  %9.4f  %10.2e  %s",
                epoch, train_loss, val_loss, val_mae, lr, status)

    # ── 最终评估 ──
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_mae, pred, true = evaluate(model, val_loader, criterion, device)

    logging.getLogger(__name__).info("%s", '=' * 60)
    logging.getLogger(__name__).info("最佳模型 (epoch %d)", ckpt['epoch'])
    logging.getLogger(__name__).info("  Val Loss = %.6f", val_loss)
    logging.getLogger(__name__).info("  Val MAE  = %.4f", val_mae)
    logging.getLogger(__name__).info("  模型保存: %s", save_path)

    # 输出部分预测样本
    logging.getLogger(__name__).info("\n预测示例 (前 10 个验证样本):")
    logging.getLogger(__name__).info("  %10s  %10s  %10s", '真实 Y', '预测 Ŷ', '误差')
    for i in range(min(10, len(pred))):
        err = pred[i].item() - true[i].item()
        logging.getLogger(__name__).info(
            "  %10.4f  %10.4f  % +10.4f",
            true[i].item(), pred[i].item(), err)


if __name__ == "__main__":
    main()
