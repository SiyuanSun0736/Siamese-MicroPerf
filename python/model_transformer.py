#!/usr/bin/env python3
"""
model_transformer.py — Siamese Transformer 网络 (Transformer Encoder + Attention Pooling + MLP)
================================================================================================

与 model.py 中 CNN 版本具有相同接口，可作为 drop-in 替换。

架构概览
--------
1) 特征编码（共享双塔）
   - 输入投影: Linear(D, d_model) 将原始 PMU 特征映射到模型维度
   - 位置编码: 可学习位置嵌入 (Learnable Positional Embedding)
   - Transformer Encoder: N 层标准 Transformer 编码层
     - Multi-Head Self-Attention + FFN (Pre-Norm / Post-Norm)
     - 支持 padding mask，防止填充位置参与注意力计算
   - Attention Pooling: 与 CNN 版本共用 Masked Attention Pooling
     将 (batch, T, d_model) 压缩为 (batch, d_model)

2) 特征融合
   - delta = v1 - v2
   - fused = [v1, v2, delta] → (batch, 3 * d_model)

3) 回归决策（MLP Head）
   - Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → 标量 Ŷ
"""

import argparse
import logging
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── 位置编码 ──────────────────────────────────────────────────────────────────


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置嵌入，适用于固定最大序列长度的场景。

    输入:  (batch, T, d_model)
    输出:  (batch, T, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model)"""
        T = x.size(1)
        if T > self.pos_embed.size(1):
            raise ValueError(f"序列长度 {T} 超过位置编码上限 {self.pos_embed.size(1)}")
        return self.dropout(x + self.pos_embed[:, :T, :])


class SinusoidalPositionalEncoding(nn.Module):
    """正弦余弦位置编码（不可学习），天然支持外推到更长序列。

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer（不参与梯度计算，但会随模型 .to(device) 迁移）
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model)"""
        T = x.size(1)
        pe = self.get_buffer("pe")
        if T > pe.size(1):
            raise ValueError(f"序列长度 {T} 超过位置编码上限 {pe.size(1)}")
        return self.dropout(x + pe[:, :T, :])


# ── 注意力池化（复用 model.py 的设计） ───────────────────────────────────────


class AttentionPooling(nn.Module):
    """掩码感知的注意力降维：将 (batch, T, F) 压缩为 (batch, F)。

    A = Softmax(H @ W_att + mask)   — (batch, T, 1)
    V = Σ_t  A_t · H_t             — (batch, F)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, h: torch.Tensor,
                valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """h: (batch, T, F), valid_len: (batch,) → (batch, F)"""
        scores = self.attention(h)  # (batch, T, 1)

        if valid_len is not None:
            batch_size, T, _ = h.shape
            positions = torch.arange(T, device=h.device).unsqueeze(0)  # (1, T)
            mask = positions >= valid_len.unsqueeze(1)  # (batch, T), True=填充
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        weights = F.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (weights * h).sum(dim=1)


# ── Transformer 编码器骨干 ────────────────────────────────────────────────────


class TransformerBackbone(nn.Module):
    """Transformer Encoder 局部+全局特征提取器。

    1. 输入投影层: Linear(D, d_model)，将原始 PMU 特征升维
    2. 位置编码: 可学习 / 正弦余弦（由 pos_encoding 参数控制）
    3. Transformer 编码层 × N:
       - Multi-Head Self-Attention (d_model, nhead)
       - Position-wise FFN (d_model → dim_feedforward → d_model)
       - LayerNorm + Dropout + 残差连接
    4. 输出 LayerNorm（稳定梯度）

    输入:  (batch, T, D)          — T 时间步, D 原始特征维度
    输出:  (batch, T, d_model)    — d_model 为模型隐藏维度
    """

    def __init__(self, in_channels: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 max_len: int = 512, pos_encoding: str = "learnable"):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, d_model)

        if pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,         # 使用 (batch, T, d_model) 格式
            norm_first=True,          # Pre-Norm：更稳定的梯度流
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # 最终 LayerNorm
        )

    def forward(self, x: torch.Tensor,
                valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, T, D) → (batch, T, d_model)
        valid_len: (batch,) 有效序列长度（可选），用于构建 padding mask
        """
        # 输入投影: (batch, T, D) → (batch, T, d_model)
        x = self.input_proj(x)

        # 位置编码
        x = self.pos_encoder(x)

        # 构建 padding mask: (batch, T)，True 表示需要忽略的位置
        src_key_padding_mask = None
        if valid_len is not None:
            batch_size, T, _ = x.shape
            positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
            src_key_padding_mask = positions >= valid_len.unsqueeze(1)  # (batch, T)

        # Transformer 编码
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x  # (batch, T, d_model)


# ── Siamese Transformer 主模型 ───────────────────────────────────────────────


class SiameseTransformerMicroPerf(nn.Module):
    """Siamese Transformer 网络：共享 Transformer+Attention 双塔 + MLP 回归头。

    与 SiameseMicroPerf (CNN 版本) 接口完全一致，可直接替换。

    输入:  x_v1, x_v2  各 (batch, T, D)
           len_v1, len_v2  各 (batch,) 有效序列长度（可选）
    输出:  ŷ           (batch,)  — 预测加速比
    """

    def __init__(self, in_features: int = 6, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, mlp_hidden: int = 64,
                 dropout: float = 0.1, max_len: int = 512,
                 pos_encoding: str = "learnable"):
        super().__init__()
        # 共享权重的 Transformer 特征提取器
        self.backbone = TransformerBackbone(
            in_channels=in_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pos_encoding=pos_encoding,
        )
        self.pool = AttentionPooling(feature_dim=d_model)

        # MLP 回归头：输入 [V_v1; V_v2; ΔV] — 3 × d_model 维度
        mlp_input_dim = d_model * 3
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def encode(self, x: torch.Tensor,
               valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """单塔前向：(batch, T, D) → (batch, d_model)"""
        h = self.backbone(x, valid_len)    # (batch, T, d_model)
        v = self.pool(h, valid_len)        # (batch, d_model)
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
        v1 = self.encode(x_v1, len_v1)   # (batch, d_model)
        v2 = self.encode(x_v2, len_v2)   # (batch, d_model)

        # 融合特征 [V_v1; V_v2; ΔV]
        delta = v1 - v2
        fused = torch.cat([v1, v2, delta], dim=-1)  # (batch, 3 * d_model)

        # MLP 回归头输出标量
        y_hat = self.mlp(fused).squeeze(-1)  # (batch,)
        return y_hat


# ── 命令行入口（模型摘要与快速烟雾测试） ─────────────────────────────────────


def _configure_logging_for_script(project_root: Path):
    """配置根日志：同时写入文件和控制台。"""
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"model_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    parser = argparse.ArgumentParser(description="Transformer 模型摘要与烟雾测试")
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    _configure_logging_for_script(args.project_root)

    model = SiameseTransformerMicroPerf(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数: %s", f"{total_params:,}")
    logger.info("%s", model)

    # 烟雾测试：随机输入前向传播
    batch, T, D = 4, 50, 6
    x_v1 = torch.randn(batch, T, D)
    x_v2 = torch.randn(batch, T, D)
    len_v1 = torch.randint(10, T + 1, (batch,))
    len_v2 = torch.randint(10, T + 1, (batch,))

    model.eval()
    with torch.no_grad():
        y_hat = model(x_v1, x_v2, len_v1, len_v2)
    logger.info("烟雾测试输出 shape: %s, values: %s", y_hat.shape, y_hat)
    logger.info("✓ 前向传播成功")


if __name__ == "__main__":
    _main()
