#!/usr/bin/env python3
"""
train.py — Siamese-MicroPerf 训练 / 验证脚本
=============================================

加载 build_dataset_*.py 生成的张量，训练或评估 Siamese 系列模型（CNN / LSTM / Transformer）。

用法
----
    # 默认（CNN）训练
    python3 python/train.py

    # 指定模型（可选：cnn, lstm, transformer）
    python3 python/train.py --model lstm --lstm-hidden 64 --lstm-out 128

    # 自定义超参
    python3 python/train.py --epochs 200 --lr 1e-3

    # 最佳模型输出位置
    python3 python/train.py --output-model checkpoints/cnn_best.pt

    # 只用一组对
    python3 python/train.py --pairs O1-g_vs_O3-g

    # 指定张量来源（固定工作量示例）
    python3 python/train.py --tensor-base train_set/tensors/fixed_work

    # 仅评估已有 checkpoint
    python3 python/train.py --eval-only --checkpoint checkpoints/best_model.pt

    # 从已有 checkpoint 继续训练，并把最佳模型保存到新位置
    python3 python/train.py --checkpoint checkpoints/best_model.pt \
        --output-model checkpoints/transformer/directml_best.pt

        
说明
----
        - 使用 `--model` 或别名 `--arch` 选择主干：`cnn`（默认），`lstm`，或 `transformer`。
        - LSTM/Transformer 有各自附加超参（参见 --help），训练脚本会把模型类型和构造参数写入 checkpoint 元信息，便于推理端自动恢复。
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

from device_utils import resolve_device
from model_factory import MODEL_CHOICES, build_model, get_model_kwargs
try:
    from tuned_configs import LABEL_MECHANISMS, TUNED_CONFIGS
except ImportError:
    from python.tuned_configs import LABEL_MECHANISMS, TUNED_CONFIGS

# ── 默认常量 ──────────────────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    "O1-g_vs_O3-g",
    "O2-bolt_vs_O2-bolt-opt",
    "O3-bolt_vs_O3-bolt-opt",
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
    当使用多组标签对混合训练时，取各对训练超参的保守合并（数值参数取均值，布尔参数取 OR）。
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
    # 收集每个标签对的覆写
    per_pair_overrides: list[dict] = []
    for pname in pair_names:
        if pname in training_cfg:
            per_pair_overrides.append(training_cfg[pname])

    # 合并标签对专属覆写：数值取均值，布尔取 OR
    merged_override: dict = {}
    if per_pair_overrides:
        all_keys = {k for d in per_pair_overrides for k in d}
        for key in all_keys:
            vals = [d[key] for d in per_pair_overrides if key in d]
            if isinstance(vals[0], bool):
                merged_override[key] = any(vals)
            elif isinstance(vals[0], (int, float)):
                merged_override[key] = sum(vals) / len(vals)
                # 如果原始类型是 int，四舍五入回 int
                if isinstance(vals[0], int):
                    merged_override[key] = round(merged_override[key])

    # base + merged_override = 最终预设
    final = {**base, **merged_override}

    for key, val in final.items():
        if key.startswith("_"):
            continue
        if key in args._explicitly_set:  # noqa: SLF001
            continue
        setattr(args, key, val)


def resolve_checkpoint_file(checkpoint: Path | None, create_dir: bool = False) -> Path | None:
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


# ── 数据加载 ──────────────────────────────────────────────────────────────────

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


def train_val_split(X_v1, X_v2, Y, len_v1, len_v2, val_ratio=0.2, seed=42):
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
        len_v1[train_idx], len_v2[train_idx],
        X_v1[val_idx], X_v2[val_idx], Y[val_idx],
        len_v1[val_idx], len_v2[val_idx],
    )


# ── 数据增强 ─────────────────────────────────────────────────────────────────

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


# ── 训练循环 ──────────────────────────────────────────────────────────────────

def direction_loss_fn(y_hat: torch.Tensor, y: torch.Tensor,
                      threshold: float = 0.0) -> torch.Tensor:
    """方向感知损失：当预测值与真实值在决策边界两侧时施加惩罚。

    使用 ReLU 确保正确方向的样本零惩罚（softplus 有 ln2 基线开销）。
    在 log 空间下，边界为 0；在原始空间下，边界为 1.0。
    """
    # 当 y_hat 和 y 符号相同 → 乘积为正 → margin 为负 → ReLU=0
    # 当符号不同 → 乘积为负 → margin 为正 → 产生惩罚
    margin = -(y_hat - threshold) * (y - threshold)
    return torch.relu(margin).mean()


def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_grad_norm: float = 1.0,
                    noise_std: float = 0.0,
                    direction_lambda: float = 0.0,
                    log_target: bool = False):
    model.train()
    total_loss = 0.0
    n = 0
    for x1, x2, y, lv1, lv2 in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        lv1, lv2 = lv1.to(device), lv2.to(device)

        # 数据增强：对有效区域添加高斯噪声
        if noise_std > 0:
            x1 = x1 + torch.randn_like(x1) * noise_std
            x2 = x2 + torch.randn_like(x2) * noise_std

        optimizer.zero_grad()
        y_hat = model(x1, x2, lv1, lv2)
        loss = criterion(y_hat, y)

        # 方向感知辅助损失
        if direction_lambda > 0:
            threshold = 0.0 if log_target else 1.0
            dir_loss = direction_loss_fn(y_hat, y, threshold=threshold)
            loss = loss + direction_lambda * dir_loss

        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
    for x1, x2, y, lv1, lv2 in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        lv1, lv2 = lv1.to(device), lv2.to(device)
        y_hat = model(x1, x2, lv1, lv2)
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
        "--tensor-base", type=Path, default=None,
        help="张量根目录（默认 train_set/tensors/fixed_time）")
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
        "--output-model", "--save-checkpoint", dest="output_model",
        type=Path, default=None,
        help="最佳模型输出路径（目录或 .pt 文件）；默认复用 --checkpoint 或保存到 checkpoints/best_model.pt")
    parser.add_argument(
        "--device", choices=["auto", "directml", "cuda", "cpu"],
        default="auto",
        help="运行设备（默认 auto，优先 directml，再回退到 cuda/cpu）")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="仅评估模式")
    parser.add_argument(
        "--patience", type=int, default=30,
        help="早停耐心值（验证 loss 连续多少 epoch 不下降时停止）")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="梯度裁剪最大范数")
    parser.add_argument(
        "--noise-std", type=float, default=0.05,
        help="训练时高斯噪声标准差（数据增强）")
    parser.add_argument(
        "--warmup-epochs", type=int, default=10,
        help="学习率线性预热轮数")
    parser.add_argument(
        "--log-target", action="store_true", default=False,
        help="将标签 Y 转为 log(Y) 进行训练（决策边界对称化）")
    parser.add_argument(
        "--direction-lambda", type=float, default=0.0,
        help="方向感知辅助损失权重（0 表示禁用）")
    parser.add_argument(
        "--pair-swap", action="store_true", default=False,
        help="对称增强：添加 (v2,v1) 反转对，数据翻倍")
    parser.add_argument(
        "--model", "--arch", dest="model",
        choices=MODEL_CHOICES, default="cnn",
        help="模型类型")
    # CNN 超参
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--cnn-out", type=int, default=128)
    # LSTM 超参
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-out", type=int, default=128)
    parser.add_argument("--bidirectional", action="store_true", default=True,
                        help="LSTM 使用双向（默认开启）")
    parser.add_argument("--no-bidirectional", dest="bidirectional",
                        action="store_false", help="LSTM 使用单向")
    # Transformer 超参
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--pos-encoding", choices=["learnable", "sinusoidal"],
                        default="learnable")
    # 通用头部超参
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    # 自动微调预设
    parser.add_argument(
        "--auto-tune", action="store_true", default=True,
        help="根据 --model 和 --pairs 自动应用微调超参预设（默认开启）")
    parser.add_argument(
        "--no-auto-tune", dest="auto_tune", action="store_false",
        help="禁用自动微调预设，使用命令行原始默认值")
    # 标签机制
    parser.add_argument(
        "--label-mechanism", dest="label_mechanism",
        choices=["auto"] + list(LABEL_MECHANISMS), default="auto",
        help="标签机制（auto 从 --tensor-base 路径自动推断，默认 auto）")
    # 配置输出目录（auto 信息与模型 checkpoint 分离）
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="模型配置 JSON 输出目录（默认 project_root/configs）")

    # 1) 先获取 argparse 的原始默认值
    arg_defaults = vars(parser.parse_args([]))
    # 2) 正式解析
    args = parser.parse_args()
    # 3) 识别用户在命令行上显式指定的参数
    args._explicitly_set = {
        k for k, v in vars(args).items()
        if k in arg_defaults and v != arg_defaults[k]
    }

    # 4) 解析标签机制（auto 时从 tensor_base 路径推断）
    tensor_base = args.tensor_base or (args.project_root / "train_set" / "tensors" / "fixed_time")
    if args.label_mechanism == "auto":
        args.label_mechanism = detect_label_mechanism(tensor_base)

    # 5) 自动微调：用 TUNED_CONFIGS 覆盖未显式指定的参数
    if args.auto_tune:
        apply_tuned_config(args, label_mechanism=args.label_mechanism)

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        device, resolved_device_name, device_message = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # 配置日志：写入 project_root/log/train_YYYYmmdd_HHMMSS.log
    log_dir = args.project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # 配置根日志：同时写入文件和输出到控制台
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

    logging.getLogger(__name__).info("%s", device_message)
    logging.getLogger(__name__).info("设备: %s", device)

    # 打印自动微调信息
    lm = args.label_mechanism
    logging.getLogger(__name__).info("标签机制: %s", lm)
    if args.auto_tune and lm in TUNED_CONFIGS and args.model in TUNED_CONFIGS[lm]:
        model_cfg = TUNED_CONFIGS[lm][args.model]
        tuned_keys = {
            k for k in vars(args)
            if k in model_cfg.get("model", {})
            or k in model_cfg.get("training", {}).get("_default", {})
        } - args._explicitly_set - {"_explicitly_set", "auto_tune", "label_mechanism"}
        if tuned_keys:
            logging.getLogger(__name__).info(
                "自动微调 [%s/%s]: %s",
                lm, args.model,
                ", ".join(f"{k}={getattr(args, k)}" for k in sorted(tuned_keys)))

    # ── 数据加载 ──
    pair_names = args.pairs or DEFAULT_PAIRS

    logging.getLogger(__name__).info("加载数据...")
    X_v1, X_v2, Y, len_v1, len_v2 = merge_pairs(tensor_base, pair_names)
    logging.getLogger(__name__).info(
        "总样本数: %d  序列长度 T=%d  特征维度 D=%d",
        X_v1.shape[0], X_v1.shape[1], X_v1.shape[2])

    in_features = X_v1.shape[2]  # D
    ckpt_file = resolve_checkpoint_file(args.checkpoint, create_dir=True)
    output_model_file = resolve_checkpoint_file(args.output_model, create_dir=True)
    checkpoint_data = None
    if ckpt_file is not None and ckpt_file.exists():
        checkpoint_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        checkpoint_model_name = checkpoint_data.get("model_name")
        if checkpoint_model_name and checkpoint_model_name != args.model:
            raise ValueError(
                f"检查点模型类型为 {checkpoint_model_name}，但当前 --model={args.model}"
            )

    # ── 划分训练/验证集 ──
    X_v1_tr, X_v2_tr, Y_tr, lv1_tr, lv2_tr, \
        X_v1_val, X_v2_val, Y_val, lv1_val, lv2_val = \
        train_val_split(X_v1, X_v2, Y, len_v1, len_v2,
                        val_ratio=args.val_ratio, seed=args.seed)

    # ── Log-target 变换 ──
    if args.log_target:
        Y_tr = torch.log(Y_tr)
        Y_val = torch.log(Y_val)
        logging.getLogger(__name__).info("已启用 log-target 变换: Y → log(Y)")

    # ── 对称增强: pair-swap ──
    if args.pair_swap:
        X_v1_tr, X_v2_tr, Y_tr, lv1_tr, lv2_tr = augment_pair_swap(
            X_v1_tr, X_v2_tr, Y_tr, lv1_tr, lv2_tr, log_target=args.log_target)
        logging.getLogger(__name__).info("pair-swap 增强后训练集: %d 样本", Y_tr.shape[0])

    logging.getLogger(__name__).info("训练集: %d  验证集: %d", Y_tr.shape[0], Y_val.shape[0])

    train_ds = TensorDataset(X_v1_tr, X_v2_tr, Y_tr, lv1_tr, lv2_tr)
    val_ds = TensorDataset(X_v1_val, X_v2_val, Y_val, lv1_val, lv2_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── 模型 ──
    model_kwargs = get_model_kwargs(args.model, in_features=in_features, args=args)
    model = build_model(args.model, **model_kwargs).to(device)

    logging.getLogger(__name__).info("模型类型: %s", args.model)
    logging.getLogger(__name__).info(
        "\n模型参数: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    logging.getLogger(__name__).info("%s", model)

    # Huber Loss (§4)
    criterion = nn.HuberLoss(delta=args.huber_delta)

    # 加载检查点：支持将 --checkpoint 指定为目录或文件
    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        logging.getLogger(__name__).info("加载检查点: %s", ckpt_file)

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

    # 学习率调度：线性预热 + 余弦退火
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    # 决定模型保存路径：优先使用 --output-model，其次复用 --checkpoint，否则写入默认路径
    if output_model_file is not None:
        save_path = output_model_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
    elif ckpt_file is not None:
        save_path = ckpt_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = args.project_root / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "best_model.pt"

    # 计算配置文件保存路径（configs/ 目录，与 checkpoints 分离）
    config_dir = args.config_dir or (args.project_root / "configs")
    config_save_path = derive_config_path(save_path, args.project_root, config_dir)
    effective_training_config = collect_training_config(
        args,
        pair_names=pair_names,
        tensor_base=tensor_base,
        resolved_device_name=resolved_device_name,
    )

    logging.getLogger(__name__).info("模型保存路径: %s", save_path)
    logging.getLogger(__name__).info("配置保存路径: %s", config_save_path)
    logging.getLogger(__name__).info(
        "\n开始训练 (%d epochs, Huber δ=%s, patience=%d)...",
        args.epochs, args.huber_delta, args.patience)
    logging.getLogger(__name__).info(
        "%-6s  %-11s  %-10s  %-9s  %-10s  %s",
        'Epoch', 'Train Loss', 'Val Loss', 'Val MAE', 'LR', 'Status')
    logging.getLogger(__name__).info("%s", '-' * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=args.grad_clip, noise_std=args.noise_std,
            direction_lambda=args.direction_lambda,
            log_target=args.log_target)
        val_loss, val_mae, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        status = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
            }, save_path)
            # 将模型配置写入 configs/ 目录（与 checkpoint 分离）
            save_model_config(
                config_save_path,
                model_name=args.model,
                model_kwargs=model_kwargs,
                log_target=args.log_target,
                checkpoint_path=save_path,
                training_args=effective_training_config,
            )
            status = "← best"
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or status:
            logging.getLogger(__name__).info(
                "%6d  %11.6f  %10.6f  %9.4f  %10.2e  %s",
                epoch, train_loss, val_loss, val_mae, lr, status)

        # 早停检查
        if patience_counter >= args.patience:
            logging.getLogger(__name__).info(
                "\n早停触发: 验证 loss 连续 %d epoch 未改善", args.patience)
            break

    # ── 最终评估 ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    val_loss, val_mae, pred, true = evaluate(model, val_loader, criterion, device)

    logging.getLogger(__name__).info("%s", '=' * 60)
    logging.getLogger(__name__).info("最佳模型 (epoch %d)", best_epoch)
    logging.getLogger(__name__).info("  Val Loss = %.6f", val_loss)
    logging.getLogger(__name__).info("  Val MAE  = %.4f", val_mae)
    logging.getLogger(__name__).info("  模型保存: %s", save_path)
    logging.getLogger(__name__).info("  配置保存: %s", config_save_path)

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
