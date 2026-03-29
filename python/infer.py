#!/usr/bin/env python3
"""
infer.py — 推理与验证阶段 (Inference / Validation)
===================================================

加载训练好的 Siamese 系列模型（CNN / LSTM / Transformer），对配对版本进行推理，输出预测加速比 Ŷ 及判断结论。

支持两种输入模式
----------------
    1. 张量模式（默认）：直接读取 build_dataset_*.py 生成的 .pt 张量
    2. CSV 模式：读取原始 PMU CSV，实时执行特征工程后推理

输出
----
    对每个程序输出：
        - 预测加速比 Ŷ（标量）
        - 判断结论：Ŷ > 1.0 → v1 优于 v2；Ŷ < 1.0 → v2 优于 v1
        - 若有真实标签 Y，同时输出误差与验证指标

用法
----
    # 对已有张量做推理（默认 fixed_time）
    python3 python/infer.py --checkpoint checkpoints/best_model.pt

    # 指定模型（auto/cnn/lstm/transformer），默认为 auto（优先使用 checkpoint 中的记录）
    python3 python/infer.py --checkpoint checkpoints/best_model.pt --model lstm

    # 使用固定工作量张量
    python3 python/infer.py --checkpoint checkpoints/best_model.pt \
            --tensor-base train_set/tensors/fixed_work

    # 只推理指定版本对
    python3 python/infer.py --checkpoint checkpoints/best_model.pt \
            --pairs O2-bolt_vs_O2-bolt-opt

    # 对两个原始 CSV 做单次推理
    python3 python/infer.py --checkpoint checkpoints/best_model.pt \
            --csv-v1 path/to/v1.csv --csv-v2 path/to/v2.csv \
            --stats train_set/tensors/fixed_time/O2-bolt_vs_O2-bolt-opt/stats.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import logging
import numpy as np
import torch

from device_utils import resolve_device
from model_factory import INFER_MODEL_CHOICES, build_model, get_model_kwargs

# 复用 build_dataset 的特征提取逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_fixedtime import extract_features  # noqa: E402


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device,
               model_name: str, model_kwargs: dict):
    """加载训练好的模型。返回 (model, model_name, model_kwargs, log_target)。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model_name = ckpt.get("model_name")
    checkpoint_model_kwargs = ckpt.get("model_kwargs")
    log_target = ckpt.get("log_target", False)

    if model_name == "auto":
        model_name = checkpoint_model_name or "cnn"
        if checkpoint_model_kwargs:
            model_kwargs = checkpoint_model_kwargs
    elif checkpoint_model_name and checkpoint_model_name != model_name:
        logging.getLogger(__name__).warning(
            "检查点记录的模型类型为 %s，但当前显式指定为 %s；将按显式参数加载",
            checkpoint_model_name, model_name,
        )

    model = build_model(model_name, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_name, model_kwargs, log_target


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


def resolve_label_mode(stats: dict | None) -> tuple[str, str]:
    """从 stats.json 提取标签模式与语义说明。"""
    if not stats:
        return "fixed_time", "Y = N_v1 / N_v2 (throughput ratio)"

    mechanism = stats.get("mechanism", "fixed_time")
    if mechanism == "fixed_workload":
        default_semantics = "Y = T_v2 / T_v1 (time ratio under equal work)"
    else:
        default_semantics = "Y = N_v1 / N_v2 (throughput ratio in a fixed window)"
    return mechanism, stats.get("label_semantics", default_semantics)


# ── 张量模式推理 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_from_tensors(model, tensor_dir: Path, device: torch.device,
                      log_target: bool = False):
    """对已有张量做批量推理，输出逐样本结果。"""
    X_v1 = torch.load(tensor_dir / "X_v1.pt", weights_only=True).to(device)
    X_v2 = torch.load(tensor_dir / "X_v2.pt", weights_only=True).to(device)

    # 有效长度（兼容旧数据集）
    len_v1_path = tensor_dir / "len_v1.pt"
    len_v2_path = tensor_dir / "len_v2.pt"
    if len_v1_path.exists() and len_v2_path.exists():
        len_v1 = torch.load(len_v1_path, weights_only=True).to(device)
        len_v2 = torch.load(len_v2_path, weights_only=True).to(device)
    else:
        T = X_v1.shape[1]
        len_v1 = torch.full((X_v1.shape[0],), T, dtype=torch.long, device=device)
        len_v2 = torch.full((X_v2.shape[0],), T, dtype=torch.long, device=device)

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
        mechanism, label_semantics = resolve_label_mode(stats)
    else:
        v1_name, v2_name = "v1", "v2"
        mechanism, label_semantics = resolve_label_mode(None)

    # 批量前向
    Y_hat = model(X_v1, X_v2, len_v1, len_v2).cpu()

    # 若模型在 log 空间训练，将预测值转回 ratio 空间
    if log_target:
        Y_hat = torch.exp(Y_hat)
        logging.getLogger(__name__).info("已应用 exp() 变换（log-target 模式）")

    N = Y_hat.shape[0]

    return Y_hat, Y, programs, v1_name, v2_name, mechanism, label_semantics


def print_results(Y_hat, Y, programs, v1_name, v2_name,
                  mechanism: str, label_semantics: str, pair_label: str):
    """打印推理结果表格。"""
    N = Y_hat.shape[0]
    has_label = Y is not None
    logger = logging.getLogger(__name__)

    logger.info("%s", "=" * 72)
    logger.info("版本对: %s  (%s vs %s)  共 %d 个程序", pair_label, v1_name, v2_name, N)
    logger.info("标签模式: %s", mechanism)
    logger.info("标签定义: %s", label_semantics)
    logger.info("判定方向: Y > 1 视为 %s 更快，Y < 1 视为 %s 更快", v1_name, v2_name)
    logger.info("%s", "=" * 72)

    header = f"  {'#':>4}  {'程序':>30}  {'预测 Ŷ':>8}"
    if has_label:
        header += f"  {'真实 Y':>8}  {'误差':>8}"
    header += f"  {'判断'}"
    logger.info(header)
    logger.info("  %s", '-' * len(header))

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
            if (y_hat >= 1.0 and y_true >= 1.0) or (y_hat < 1.0 and y_true < 1.0):
                correct_direction += 1
        line += f"  {verdict}"
        logger.info(line)

    # 汇总统计
    if has_label and len(all_pred) > 0:
        pred = np.array(all_pred)
        true = np.array(all_true)
        mae = np.abs(pred - true).mean()
        mse = ((pred - true) ** 2).mean()
        direction_acc = correct_direction / N * 100

        logger.info("\n  ── 验证指标 ──")
        logger.info("  MAE  = %.4f", mae)
        logger.info("  MSE  = %.4f", mse)
        logger.info("  RMSE = %.4f", np.sqrt(mse))
        logger.info("  方向准确率 = %d/%d (%.1f%%)", correct_direction, N, direction_acc)

        v1_better_pred = (pred > 1.0).sum()
        v1_better_true = (true > 1.0).sum()
        logger.info("  预测 %s 更优: %d/%d  (真实: %d/%d)", v1_name, v1_better_pred, N, v1_better_true, N)

        # 方向准确率（排除 ±5% 区间的样本）——聚焦有显著真实差异的样本
        mask = np.abs(true - 1.0) > 0.05
        if mask.sum() > 0:
            correct_filt = (((pred[mask] >= 1.0) & (true[mask] >= 1.0)) |
                            ((pred[mask] <  1.0) & (true[mask] <  1.0))).sum()
            acc_filt = correct_filt / mask.sum() * 100.0
            logger.info("  方向准确率（排除 ±5%%）= %d/%d (%.1f%%)", int(correct_filt), int(mask.sum()), acc_filt)


# ── CSV 模式推理 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_from_csv(model, csv_v1: Path, csv_v2: Path,
                   stats_path: Path, device: torch.device,
                   log_target: bool = False):
    """对两个原始 PMU CSV 执行实时特征工程并推理。"""
    stats = json.loads(stats_path.read_text())
    seq_len = stats.get("seq_len", stats.get("max_seq_len"))
    if seq_len is None:
        logging.error("stats.json 缺少 seq_len/max_seq_len，无法执行 CSV 模式推理")
        sys.exit(1)
    mu = np.array(stats["mu"], dtype=np.float32)
    sigma = np.array(stats["sigma"], dtype=np.float32)
    v1_name = stats.get("v1", "v1")
    v2_name = stats.get("v2", "v2")
    mechanism, label_semantics = resolve_label_mode(stats)

    result1 = extract_features(csv_v1, seq_len)
    result2 = extract_features(csv_v2, seq_len)
    if result1 is None or result2 is None:
        logging.error("特征提取失败")
        sys.exit(1)

    feat1, vlen1 = result1
    feat2, vlen2 = result2

    # Z-score 归一化（复用训练集统计量，仅对有效区域）
    feat1[:vlen1] = (feat1[:vlen1] - mu) / sigma
    feat2[:vlen2] = (feat2[:vlen2] - mu) / sigma

    # (1, T, D)
    x1 = torch.from_numpy(feat1).unsqueeze(0).to(device)
    x2 = torch.from_numpy(feat2).unsqueeze(0).to(device)
    lv1 = torch.tensor([vlen1], dtype=torch.long, device=device)
    lv2 = torch.tensor([vlen2], dtype=torch.long, device=device)

    y_hat = model(x1, x2, lv1, lv2).item()
    if log_target:
        import math
        y_hat = math.exp(y_hat)
    verdict = judge(y_hat, v1_name, v2_name)

    logging.info("%s", "=" * 60)
    logging.info("单次推理结果")
    logging.info("%s", "=" * 60)
    logging.info("  标签模式: %s", mechanism)
    logging.info("  标签定义: %s", label_semantics)
    logging.info("  v1 CSV:  %s", csv_v1)
    logging.info("  v2 CSV:  %s", csv_v2)
    logging.info("  预测 Ŷ:  %.4f", y_hat)
    logging.info("  判断:    %s", verdict)
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
    parser.add_argument(
        "--device", choices=["auto", "directml", "cuda", "cpu"],
        default="auto",
        help="运行设备（默认 auto，优先 directml，再回退到 cuda/cpu）")

    # 张量模式参数
    parser.add_argument(
        "--tensor-base", type=Path, default=None,
        help="张量根目录（默认 train_set/tensors/fixed_time）")
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

    # 模型超参（需与训练时一致；若 checkpoint 中有记录则 infer --model auto 会优先复用）
    parser.add_argument(
        "--model", "--arch", dest="model",
        choices=INFER_MODEL_CHOICES, default="auto")
    parser.add_argument("--in-features", type=int, default=6)
    parser.add_argument("--cnn-hidden", type=int, default=64)
    parser.add_argument("--cnn-out", type=int, default=128)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-out", type=int, default=128)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no-bidirectional", dest="bidirectional",
                        action="store_false")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--pos-encoding", choices=["learnable", "sinusoidal"],
                        default="learnable")
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    try:
        device, resolved_device_name, device_message = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # 配置日志：写入 project_root/log/infer_YYYYmmdd_HHMMSS.log
    log_dir = args.project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"infer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # 配置根日志：同时写入文件和输出到控制台
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    fh = logging.FileHandler(str(log_file), mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    # 清理已有处理器（防止重复添加）
    if root_logger.handlers:
        root_logger.handlers = []
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)

    logging.getLogger(__name__).info("%s", device_message)
    logging.getLogger(__name__).info("设备: %s", device)
    logging.getLogger(__name__).info("设备类型: %s", resolved_device_name)

    fallback_model_name = "cnn" if args.model == "auto" else args.model
    fallback_model_kwargs = get_model_kwargs(
        fallback_model_name,
        in_features=args.in_features,
        args=args,
    )
    model, resolved_model_name, resolved_model_kwargs, log_target = load_model(
        args.checkpoint,
        device,
        model_name=args.model,
        model_kwargs=fallback_model_kwargs,
    )
    logging.getLogger(__name__).info("模型加载完成: %s  (设备: %s)", args.checkpoint, device)
    logging.getLogger(__name__).info("模型类型: %s", resolved_model_name)
    logging.getLogger(__name__).info("模型参数: %s", resolved_model_kwargs)
    if log_target:
        logging.getLogger(__name__).info("log-target 模式: 推理时将自动应用 exp() 变换")

    # ── CSV 模式 ──
    if args.csv_v1 and args.csv_v2:
        if not args.stats:
            logging.error("CSV 模式需要 --stats 参数")
            sys.exit(1)
        infer_from_csv(model, args.csv_v1, args.csv_v2, args.stats, device,
                       log_target=log_target)
        return

    # ── 张量模式 ──
    tensor_base = args.tensor_base or (args.project_root / "train_set" / "tensors" / "fixed_time")
    pair_names = args.pairs or [
        "O1-g_vs_O3-g",
        "O2-bolt_vs_O2-bolt-opt",
        "O3-bolt_vs_O3-bolt-opt",
    ]

    for pair in pair_names:
        d = tensor_base / pair
        if not d.exists():
            logging.getLogger(__name__).warning("跳过不存在的目录: %s", d)
            continue
        Y_hat, Y, programs, v1_name, v2_name, mechanism, label_semantics = \
            infer_from_tensors(model, d, device, log_target=log_target)
        print_results(Y_hat, Y, programs, v1_name, v2_name,
                      mechanism, label_semantics, pair)

    logging.getLogger(__name__).info("")


if __name__ == "__main__":
    main()
