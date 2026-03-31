#!/usr/bin/env python3
"""
model_lstm.py — Siamese LSTM 网络 (Bidirectional LSTM + Attention Pooling + MLP)
=================================================================================

与 model_cnn.py / model_transformer.py 接口完全一致，可作为 drop-in 替换。

架构概览
--------
1) 特征编码（共享双塔）
   - 输入投影: Linear(D, lstm_hidden) 将原始 PMU 特征映射到隐藏维度
   - 双向 LSTM: N 层 Bidirectional LSTM，捕获时序前后依赖
     - 支持 pack_padded_sequence，高效处理变长序列
     - 输出维度: lstm_hidden * 2（双向拼接）
   - 输出投影: Linear(lstm_hidden * 2, out_features) 降维到最终编码维度
   - Attention Pooling: 掩码感知注意力池化
     将 (batch, T, out_features) 压缩为 (batch, out_features)

2) 特征融合
   - delta = v1 - v2
   - fused = [v1, v2, delta] → (batch, 3 * out_features)

3) 回归决策（MLP Head）
   - Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → 标量 Ŷ
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


# ── 注意力池化（复用 model_cnn.py 的设计） ───────────────────────────────────


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


# ── LSTM 编码器骨干 ──────────────────────────────────────────────────────────


class LSTMBackbone(nn.Module):
    """双向 LSTM 时序特征提取器。

    1. 输入投影层: Linear(D, lstm_hidden)，将原始 PMU 特征升维
    2. 双向 LSTM × N 层（_ManualLSTM，兼容 DirectML）:
       - 捕获时序前后文依赖
       - 输出维度: lstm_hidden * 2（双向拼接）
    3. 输出投影层: Linear(lstm_hidden * 2, out_features)
    4. LayerNorm 稳定输出分布

    输入:  (batch, T, D)             — T 时间步, D 原始特征维度
    输出:  (batch, T, out_features)  — out_features 为最终编码维度
    """

    def __init__(self, in_channels: int, lstm_hidden: int = 64,
                 out_features: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, lstm_hidden)
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.output_proj = nn.Linear(lstm_hidden * num_directions, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, T, D) → (batch, T, out_features)
        valid_len: (batch,) 有效序列长度（可选），用于 pack_padded_sequence
        """
        x = self.input_proj(x)  # (batch, T, lstm_hidden)

        if valid_len is not None:
            lengths_cpu = valid_len.clamp(min=1).cpu()
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True,
                                       total_length=x.size(1))
        else:
            x, _ = self.lstm(x)

        x = self.output_proj(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x  # (batch, T, out_features)


# ── Siamese LSTM 主模型 ─────────────────────────────────────────────────────


class SiameseLSTMMicroPerf(nn.Module):
    """Siamese LSTM 网络：共享 BiLSTM+Attention 双塔 + MLP 回归头。

    与 SiameseMicroPerf (CNN) / SiameseTransformerMicroPerf 接口完全一致。

    输入:  x_v1, x_v2  各 (batch, T, D)
           len_v1, len_v2  各 (batch,) 有效序列长度（可选）
    输出:  ŷ           (batch,)  — 预测加速比
    """

    def __init__(self, in_features: int = 6, lstm_hidden: int = 64,
                 lstm_out: int = 128, num_layers: int = 2,
                 bidirectional: bool = True, mlp_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        # 共享权重的 LSTM 特征提取器
        self.backbone = LSTMBackbone(
            in_channels=in_features,
            lstm_hidden=lstm_hidden,
            out_features=lstm_out,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.pool = AttentionPooling(feature_dim=lstm_out)

        # MLP 回归头：输入 [V_v1; V_v2; ΔV] — 3 × lstm_out 维度
        mlp_input_dim = lstm_out * 3
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
        """单塔前向：(batch, T, D) → (batch, lstm_out)"""
        h = self.backbone(x, valid_len)    # (batch, T, lstm_out)
        v = self.pool(h, valid_len)        # (batch, lstm_out)
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
        v1 = self.encode(x_v1, len_v1)   # (batch, lstm_out)
        v2 = self.encode(x_v2, len_v2)   # (batch, lstm_out)

        # 融合特征 [V_v1; V_v2; ΔV]
        delta = v1 - v2
        fused = torch.cat([v1, v2, delta], dim=-1)  # (batch, 3 * lstm_out)

        # MLP 回归头输出标量
        y_hat = self.mlp(fused).squeeze(-1)  # (batch,)
        return y_hat


# ── 命令行入口（模型摘要与快速烟雾测试） ─────────────────────────────────────


def _configure_logging_for_script(project_root: Path):
    """配置根日志：同时写入文件和控制台。"""
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"model_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    parser = argparse.ArgumentParser(description="LSTM 模型摘要与烟雾测试")
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-out", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    _configure_logging_for_script(args.project_root)

    model = SiameseLSTMMicroPerf(
        lstm_hidden=args.lstm_hidden,
        lstm_out=args.lstm_out,
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
