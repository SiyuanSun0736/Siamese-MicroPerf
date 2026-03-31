#!/usr/bin/env python3
"""配置管理工具：检查点路径解析、配置持久化、自动微调预设。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from tuned_configs import LABEL_MECHANISMS, TUNED_CONFIGS, BOLT_OPT_VARIANTS
except ImportError:
    from python.tuned_configs import LABEL_MECHANISMS, TUNED_CONFIGS, BOLT_OPT_VARIANTS

DEFAULT_PAIRS = [
    "O1-g_vs_O3-g",
    *BOLT_OPT_VARIANTS,
]


def derive_config_path(save_path: Path, project_root: Path,
                       config_dir: Path | None = None) -> Path:
    """从模型保存路径推导配置 JSON 路径。

    将 checkpoints/ 下的 .pt 路径映射到 configs/ 下的 .json 路径。
    """
    if config_dir is None:
        config_dir = project_root / "configs"
    try:
        rel = save_path.resolve().relative_to(
            (project_root / "checkpoints").resolve())
        return config_dir / rel.with_suffix(".json")
    except ValueError:
        return config_dir / save_path.with_suffix(".json").name


def save_model_config(config_path: Path, *, model_name: str,
                      model_kwargs: dict, log_target: bool,
                      checkpoint_path: Path | str,
                      training_args: dict | None = None) -> None:
    """保存模型配置到 JSON 文件（与 checkpoint 分离存储）。"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config: dict = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "log_target": log_target,
        "checkpoint": str(checkpoint_path),
    }
    if training_args:
        config["training_config"] = training_args
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def collect_training_config(args: argparse.Namespace, *, pair_names: list[str],
                            tensor_base: Path,
                            resolved_device_name: str) -> dict:
    """收集当前运行实际生效的训练配置，便于随模型一起持久化。"""
    return {
        "label_mechanism": args.label_mechanism,
        "auto_tune": args.auto_tune,
        "explicitly_set": sorted(args._explicitly_set),  # noqa: SLF001
        "tensor_base": str(tensor_base),
        "pairs": list(pair_names),
        "device": resolved_device_name,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "huber_delta": args.huber_delta,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "noise_std": args.noise_std,
        "warmup_epochs": args.warmup_epochs,
        "log_target": args.log_target,
        "direction_lambda": args.direction_lambda,
        "pair_swap": args.pair_swap,
        "optimizer": "Adam",
        "scheduler": {
            "name": "LambdaLR",
            "warmup": "linear",
            "decay": "cosine",
        },
    }


def load_model_config(config_path: Path | None) -> dict | None:
    """从 JSON 文件加载模型配置。"""
    if config_path and config_path.exists():
        return json.loads(config_path.read_text())
    return None


def detect_label_mechanism(tensor_base: Path) -> str:
    """从 tensor_base 路径自动推断标签机制。

    匹配规则：路径中包含 'inst_retired' → inst_retired；
    包含 'fixed_work' → fixed_work；其余默认 fixed_time。
    """
    path_str = str(tensor_base)
    if "inst_retired" in path_str or "instret" in path_str:
        return "inst_retired"
    if "fixed_work" in path_str:
        return "fixed_work"
    return "fixed_time"


def apply_tuned_config(args: argparse.Namespace,
                       label_mechanism: str = "fixed_time") -> None:
    """根据 --label-mechanism、--model 和 --pairs 将 TUNED_CONFIGS 中的预设写入 args。

    覆盖优先级：命令行显式值 > 标签对专属配置 > 模型默认配置 > argparse 默认值。
    仅当训练单一标签对时应用该对的专属覆写；多对混合训练时只使用 _default。
    若指定的 label_mechanism 不在 TUNED_CONFIGS 中，回退到 fixed_time。
    """
    model_name = args.model

    # 回退策略：label_mechanism → fixed_time
    if label_mechanism not in TUNED_CONFIGS:
        label_mechanism = "fixed_time"
    mechanism_cfg = TUNED_CONFIGS[label_mechanism]

    if model_name not in mechanism_cfg:
        return

    cfg = mechanism_cfg[model_name]

    # ── 1) 模型架构参数 ──
    for key, val in cfg["model"].items():
        if key in args._explicitly_set:  # noqa: SLF001
            continue
        setattr(args, key, val)

    # ── 2) 训练超参 ──
    training_cfg = cfg["training"]
    base = dict(training_cfg["_default"])

    pair_names = args.pairs or DEFAULT_PAIRS
    # 仅当恰好训练单一标签对时，应用该对的专属覆写
    per_pair_override: dict = {}
    if len(pair_names) == 1 and pair_names[0] in training_cfg:
        per_pair_override = training_cfg[pair_names[0]]
        # 应用 per-pair 模型架构覆盖
        pair_model_ov = cfg.get("model_overrides", {}).get(pair_names[0], {})
        for key, val in pair_model_ov.items():
            if key in args._explicitly_set:  # noqa: SLF001
                continue
            setattr(args, key, val)

    # base + per_pair_override = 最终预设
    final = {**base, **per_pair_override}

    for key, val in final.items():
        if key.startswith("_"):
            continue
        if key in args._explicitly_set:  # noqa: SLF001
            continue
        setattr(args, key, val)


def resolve_checkpoint_file(checkpoint: Path | None,
                            create_dir: bool = False) -> Path | None:
    """将目录/文件形式的 checkpoint 参数解析为具体文件路径。"""
    if checkpoint is None:
        return None

    if checkpoint.exists() and checkpoint.is_dir():
        return checkpoint / "best_model.pt"

    if not checkpoint.exists() and checkpoint.suffix != ".pt":
        if create_dir:
            checkpoint.mkdir(parents=True, exist_ok=True)
        return checkpoint / "best_model.pt"

    return checkpoint
