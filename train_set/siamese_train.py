#!/usr/bin/env python3
"""
siamese_train.py — Siamese 1D-CNN 微架构性能排序模型
=====================================================

架构（对应 README 第 3-4 节）：
  共享编码器:  Input (D×T) → Conv1D×3 → GlobalMaxPool → FC → scalar y
  排序损失:    p = sigmoid(y_v1 - y_v2),  L = BCE(p, label)

输入数据：
  train_set/manifest.jsonl    每行 JSON: {program, label, v1_csv, v2_csv, ...}
  train_set/data/features/    {program}_v1.npy, {program}_v2.npy  (T×D float32)

用法：
  python3 siamese_train.py [--manifest PATH] [--features-dir DIR]
                            [--seq-len 60] [--d-feat 6]
                            [--epochs 100] [--batch-size 16] [--lr 1e-3]
                            [--save-path model.pt] [--device cpu|cuda]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── 默认超参数 ────────────────────────────────────────────────────────────────
DEFAULT_SEQ_LEN   = 60       # 时间步长 T；须与 process_features.py 一致
DEFAULT_D_FEAT    = 6        # 特征维度 D
DEFAULT_BASE_CH   = 32       # CNN 第一层通道数
DEFAULT_KERNEL    = 7        # 卷积核宽度；应 ≥ PMU 多路复用周期 M
DEFAULT_EPOCHS    = 100
DEFAULT_BATCH     = 16
DEFAULT_LR        = 1e-3
DEFAULT_SPLIT     = 0.8      # 训练集比例

# ── Dataset ──────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    从 manifest.jsonl + .npy 特征文件加载 (X_v1, X_v2, label) 配对。

    X_v{1,2} 形状: (D, T)  ← Conv1d 期望 (channels, length)
    label: float 0.0 或 1.0
    """

    def __init__(self, manifest_path: str, features_dir: str):
        manifest_path = Path(manifest_path)
        features_dir  = Path(features_dir)

        self.pairs = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prog  = entry["program"]
                npy_v1 = features_dir / f"{prog}_v1.npy"
                npy_v2 = features_dir / f"{prog}_v2.npy"
                if not npy_v1.exists() or not npy_v2.exists():
                    continue
                self.pairs.append({
                    "program": prog,
                    "label":   float(entry["label"]),
                    "npy_v1":  str(npy_v1),
                    "npy_v2":  str(npy_v2),
                })

        if len(self.pairs) == 0:
            raise RuntimeError(
                "数据集为空。请先运行:\n"
                "  bash train_set/collect_dataset.sh\n"
                "  python3 train_set/process_features.py --manifest train_set/manifest.jsonl"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        entry = self.pairs[idx]
        X_v1 = np.load(entry["npy_v1"])   # (T, D)
        X_v2 = np.load(entry["npy_v2"])   # (T, D)
        # 转置为 (D, T) 供 Conv1d 使用
        X_v1 = torch.tensor(X_v1.T, dtype=torch.float32)
        X_v2 = torch.tensor(X_v2.T, dtype=torch.float32)
        label = torch.tensor(entry["label"], dtype=torch.float32)
        return X_v1, X_v2, label


# ── Siamese 编码器 ────────────────────────────────────────────────────────────

class SiameseEncoder(nn.Module):
    """
    共享权重的 1D-CNN 编码器 + 全局最大池化 + 全连接投影为标量 y。

    输入形状:  (B, D, T)
    输出形状:  (B,)  ← 表示该版本"执行阻力"的标量

    卷积核宽度 kernel_size 应满足 kernel_size ≥ M（PMU 多路复用轮转周期），
    以在物理层面吸收时分复用带来的时间错位噪声（README §3）。
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_D_FEAT,
        base_ch: int     = DEFAULT_BASE_CH,
        kernel_size: int = DEFAULT_KERNEL,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.cnn = nn.Sequential(
            # 层 1
            nn.Conv1d(in_channels,  base_ch,   kernel_size, padding=pad),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            # 层 2
            nn.Conv1d(base_ch,      base_ch*2, kernel_size, padding=pad),
            nn.BatchNorm1d(base_ch*2),
            nn.ReLU(),
            # 层 3
            nn.Conv1d(base_ch*2,    base_ch*4, kernel_size, padding=pad),
            nn.BatchNorm1d(base_ch*4),
            nn.ReLU(),
        )
        # 全局最大池化：截取时间轴上最恶劣的失速尖峰（README §3）
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc  = nn.Linear(base_ch * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T)  →  scalar: (B,)"""
        h = self.cnn(x)              # (B, base_ch*4, T)
        h = self.gmp(h).squeeze(-1)  # (B, base_ch*4)
        return self.fc(h).squeeze(-1)  # (B,)


# ── 训练 / 验证循环 ──────────────────────────────────────────────────────────

def run_epoch(
    model:     SiameseEncoder,
    loader:    DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
) -> tuple[float, float]:
    """
    执行一个 epoch，返回 (loss, accuracy)。
    optimizer=None 表示验证模式（不更新参数）。
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(train_mode):
        for X_v1, X_v2, labels in loader:
            X_v1   = X_v1.to(device)
            X_v2   = X_v2.to(device)
            labels = labels.to(device)

            y_v1   = model(X_v1)           # (B,)
            y_v2   = model(X_v2)           # (B,)
            logit  = y_v1 - y_v2           # 正值 → v1 更优（README §4）
            loss   = criterion(logit, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            pred        = (logit > 0).float()
            correct    += (pred == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"设备: {device}")

    # ── 数据集 ────────────────────────────────────────────────────────────────
    dataset = PairDataset(args.manifest, args.features_dir)
    print(f"样本对数量: {len(dataset)}")

    n_train = max(1, int(len(dataset) * DEFAULT_SPLIT))
    n_val   = len(dataset) - n_train
    if n_val == 0:
        # 数据极少时全部用于训练，无验证集
        train_set, val_set = dataset, None
    else:
        train_set, val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = (DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
                    if val_set else None)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = SiameseEncoder(
        in_channels=args.d_feat,
        base_ch=DEFAULT_BASE_CH,
        kernel_size=DEFAULT_KERNEL,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}")
    print("-" * 55)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        if val_loader:
            vl_loss, vl_acc = run_epoch(model, val_loader, criterion, None, device)
        else:
            vl_loss, vl_acc = float("nan"), float("nan")

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  "
              f"{vl_loss:>9.4f}  {vl_acc:>8.3f}")

        # 保存最佳验证准确率模型
        if val_loader and vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), args.save_path)

    # 若无验证集，保存最终模型
    if not val_loader:
        torch.save(model.state_dict(), args.save_path)

    print(f"\n模型已保存至: {args.save_path}")
    if val_loader:
        print(f"最佳验证准确率: {best_val_acc:.3f}")


# ── 推理接口（用于验证阶段，对应 README §5-6）────────────────────────────────

def predict(
    model:    SiameseEncoder,
    X_v1:     np.ndarray,
    X_v2:     np.ndarray,
    device:   torch.device,
) -> float:
    """
    输入两个 (T×D) 特征张量，返回 p = P(v1 优于 v2) ∈ (0,1)。
    p > 0.5 → v1 更优；p < 0.5 → v2 更优。
    """
    model.eval()
    with torch.no_grad():
        t1 = torch.tensor(X_v1.T, dtype=torch.float32).unsqueeze(0).to(device)
        t2 = torch.tensor(X_v2.T, dtype=torch.float32).unsqueeze(0).to(device)
        y1 = model(t1)
        y2 = model(t2)
        p  = torch.sigmoid(y1 - y2).item()
    return p


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    train_set_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Siamese MicroPerf 训练")
    parser.add_argument("--manifest",
                        default=str(train_set_dir / "manifest.jsonl"))
    parser.add_argument("--features-dir",
                        default=str(train_set_dir / "data" / "features"))
    parser.add_argument("--seq-len",    type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument("--d-feat",     type=int,   default=DEFAULT_D_FEAT)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--save-path",
                        default=str(train_set_dir / "model.pt"))
    parser.add_argument("--device",     default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
