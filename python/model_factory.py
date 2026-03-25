#!/usr/bin/env python3
"""模型工厂：统一构建 CNN / Transformer Siamese 模型。"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from model_cnn import SiameseMicroPerf
from model_lstm import SiameseLSTMMicroPerf
from model_transformer import SiameseTransformerMicroPerf

MODEL_CHOICES = ("cnn", "lstm", "transformer")
INFER_MODEL_CHOICES = ("auto",) + MODEL_CHOICES


def get_model_kwargs(model_name: str, *, in_features: int, args: Any) -> dict[str, Any]:
    """根据模型名称和命令行参数提取模型构造参数。"""
    common_kwargs = {
        "in_features": in_features,
        "mlp_hidden": args.mlp_hidden,
        "dropout": args.dropout,
    }

    if model_name == "cnn":
        return {
            **common_kwargs,
            "cnn_hidden": args.cnn_hidden,
            "cnn_out": args.cnn_out,
        }

    if model_name == "lstm":
        return {
            **common_kwargs,
            "lstm_hidden": args.lstm_hidden,
            "lstm_out": args.lstm_out,
            "num_layers": args.num_layers,
            "bidirectional": args.bidirectional,
        }

    if model_name == "transformer":
        return {
            **common_kwargs,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "max_len": args.max_len,
            "pos_encoding": args.pos_encoding,
        }

    raise ValueError(f"不支持的模型类型: {model_name}")


def build_model(model_name: str, **model_kwargs: Any) -> nn.Module:
    """构建指定类型的模型实例。"""
    if model_name == "cnn":
        return SiameseMicroPerf(**model_kwargs)
    if model_name == "lstm":
        return SiameseLSTMMicroPerf(**model_kwargs)
    if model_name == "transformer":
        return SiameseTransformerMicroPerf(**model_kwargs)
    raise ValueError(f"不支持的模型类型: {model_name}")