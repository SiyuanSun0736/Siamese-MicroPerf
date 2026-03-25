#!/usr/bin/env python3
"""
model.py — Siamese 网络主干 (1D-CNN + Attention Pooling + MLP)
==============================================================

按照 README §3–§4 的设计：
  - 共享权重的多层 1D-CNN（带残差连接）提取局部时序特征 H ∈ R^{T×F}
  - Masked Attention Pooling 将变长时序压缩为定长向量 V ∈ R^F
  - MLP 对多种融合特征 [ΔV; V_v1; V_v2] 做回归，输出连续加速比 Ŷ
  - 使用 Huber Loss 作为鲁棒损失函数
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CNNBackbone(nn.Module):
    """多层 1D-CNN 局部特征提取器，带残差连接。

    输入:  (batch, T, D)   — T 时间步, D 原始特征维度
    输出:  (batch, T, F)   — F 为最终卷积通道数
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        # Conv1d 期望 (batch, channels, length)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Conv2 残差连接（hidden→hidden 维度相同，直接相加）
        # Conv3 残差连接（hidden→out 需要投影）
        self.res_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1) \
            if hidden_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, D) → (batch, T, F)"""
        # (batch, T, D) → (batch, D, T) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        # 残差: conv2 输入输出维度相同
        residual = x
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = x + residual

        # 残差: conv3 需要投影
        residual = self.res_proj(x)
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = x + residual

        # (batch, F, T) → (batch, T, F)
        return x.permute(0, 2, 1)


class AttentionPooling(nn.Module):
    """掩码感知的注意力降维：将 (batch, T, F) 压缩为 (batch, F)。

    对填充位置施加 -inf 掩码，使其 softmax 权重为 0，
    防止零填充区的伪特征污染注意力分配。

    A = Softmax(H @ W_att + mask)   — (batch, T, 1)
    V = Σ_t  A_t · H_t             — (batch, F)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, h: torch.Tensor,
                valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """h: (batch, T, F), valid_len: (batch,) → (batch, F)"""
        # (batch, T, 1)
        scores = self.attention(h)

        if valid_len is not None:
            # 构建掩码: (batch, T, 1)，填充位置为 -inf
            batch_size, T, _ = h.shape
            positions = torch.arange(T, device=h.device).unsqueeze(0)  # (1, T)
            mask = positions >= valid_len.unsqueeze(1)  # (batch, T), True=填充
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        weights = F.softmax(scores, dim=1)
        # 安全处理：全部被 mask 时 softmax 产生 nan，替换为 0
        weights = torch.nan_to_num(weights, nan=0.0)
        # 加权求和: (batch, T, F) * (batch, T, 1) → sum → (batch, F)
        return (weights * h).sum(dim=1)


class SiameseMicroPerf(nn.Module):
    """Siamese 网络：共享 CNN+Attention 双塔 + MLP 回归头。

    输入:  x_v1, x_v2  各 (batch, T, D)
           len_v1, len_v2  各 (batch,) 有效序列长度（可选）
    输出:  ŷ           (batch,)  — 预测加速比
    """

    def __init__(self, in_features: int = 6, cnn_hidden: int = 64,
                 cnn_out: int = 128, mlp_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        # 共享权重的特征提取器
        self.backbone = CNNBackbone(
            in_channels=in_features,
            hidden_channels=cnn_hidden,
            out_channels=cnn_out,
            dropout=dropout,
        )
        self.pool = AttentionPooling(feature_dim=cnn_out)

        # MLP 回归头：输入 [V_v1; V_v2; ΔV] — 3×F 维度
        # 保留绝对特征 + 差异特征，信息更丰富
        mlp_input_dim = cnn_out * 3
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),  # 无限制性激活函数
        )

    def encode(self, x: torch.Tensor,
               valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """单塔前向：(batch, T, D) → (batch, F)"""
        h = self.backbone(x)               # (batch, T, F)
        v = self.pool(h, valid_len)        # (batch, F)
        return v

    def forward(self, x_v1: torch.Tensor,
                x_v2: torch.Tensor,
                len_v1: torch.Tensor | None = None,
                len_v2: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_v1: (batch, T, D)
        x_v2: (batch, T, D)
        len_v1, len_v2: (batch,) 有效序列长度（可选，用于掩码注意力）
        → ŷ:  (batch,)
        """
        v1 = self.encode(x_v1, len_v1)   # (batch, F)
        v2 = self.encode(x_v2, len_v2)   # (batch, F)

        # §4: 融合特征 [V_v1; V_v2; ΔV]
        delta = v1 - v2
        fused = torch.cat([v1, v2, delta], dim=-1)  # (batch, 3*F)

        # MLP 回归头输出标量
        y_hat = self.mlp(fused).squeeze(-1)  # (batch,)
        return y_hat


def _configure_logging_for_script(project_root: Path):
    """配置根日志：同时写入文件和控制台（供独立运行模块时使用）。"""
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh = logging.FileHandler(str(log_file), mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    if root_logger.handlers:
        root_logger.handlers = []
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)


def _main():
    parser = argparse.ArgumentParser(description="模型摘要与日志测试")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    _configure_logging_for_script(args.project_root)

    m = SiameseMicroPerf()
    logging.getLogger(__name__).info("模型参数: %s", f"{sum(p.numel() for p in m.parameters()):,}")
    logging.getLogger(__name__).info("%s", m)


if __name__ == "__main__":
    _main()
