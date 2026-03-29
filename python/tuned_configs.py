#!/usr/bin/env python3
"""训练超参数预设。

集中维护不同标签机制下的模型架构参数与训练超参，供 train.py 导入。
"""

LABEL_MECHANISMS = ("fixed_time", "fixed_work", "inst_retired")

# 模型架构参数（跨标签机制共享）
_CNN_MODEL = {
    "cnn_hidden": 64,
    "cnn_out": 128,
    "mlp_hidden": 64,
    "dropout": 0.10,
}

_LSTM_MODEL = {
    "lstm_hidden": 64,
    "lstm_out": 128,
    "num_layers": 2,
    "bidirectional": True,
    "mlp_hidden": 64,
    "dropout": 0.15,
}

_TRANSFORMER_MODEL = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "max_len": 512,
    "pos_encoding": "learnable",
    "mlp_hidden": 64,
    "dropout": 0.10,
}

TUNED_CONFIGS: dict[str, dict[str, dict]] = {
    # ============================================================
    # fixed_time: 标签 Y = N_v1 / N_v2（固定时间窗口循环次数比）
    # ============================================================
    "fixed_time": {
        "cnn": {
            "model": _CNN_MODEL,
            "training": {
                "_default": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 150,
                    "patience": 30,
                    "warmup_epochs": 10,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.05,
                    "direction_lambda": 0.0,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "noise_std": 0.06,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 8e-4,
                    "direction_lambda": 0.1,
                    "noise_std": 0.03,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 8e-4,
                    "direction_lambda": 0.1,
                    "noise_std": 0.03,
                },
            },
        },
        "lstm": {
            "model": _LSTM_MODEL,
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 200,
                    "patience": 40,
                    "warmup_epochs": 15,
                    "huber_delta": 1.0,
                    "grad_clip": 0.5,
                    "noise_std": 0.03,
                    "direction_lambda": 0.0,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "noise_std": 0.04,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 3e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.02,
                    "patience": 50,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 3e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.02,
                    "patience": 50,
                },
            },
        },
        "transformer": {
            "model": _TRANSFORMER_MODEL,
            "training": {
                "_default": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 200,
                    "patience": 30,
                    "warmup_epochs": 10,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.05,
                    "direction_lambda": 0.0,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "warmup_epochs": 25,
                    "noise_std": 0.03,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 2e-4,
                    "direction_lambda": 0.1,
                    "noise_std": 0.01,
                    "patience": 50,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 2e-4,
                    "direction_lambda": 0.1,
                    "noise_std": 0.01,
                    "patience": 50,
                },
            },
        },
    },

    # ============================================================
    # fixed_work: 标签 Y = T_v2 / T_v1（固定工作量耗时比）
    # ============================================================
    "fixed_work": {
        "cnn": {
            "model": _CNN_MODEL,
            "training": {
                "_default": {
                    "lr": 8e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 180,
                    "patience": 35,
                    "warmup_epochs": 12,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.03,
                    "direction_lambda": 0.0,
                    "pair_swap": False,
                    "log_target": True,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "noise_std": 0.04,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 6e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.02,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 6e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.02,
                },
            },
        },
        "lstm": {
            "model": _LSTM_MODEL,
            "training": {
                "_default": {
                    "lr": 4e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 220,
                    "patience": 45,
                    "warmup_epochs": 18,
                    "huber_delta": 1.0,
                    "grad_clip": 0.5,
                    "noise_std": 0.02,
                    "direction_lambda": 0.0,
                    "pair_swap": False,
                    "log_target": True,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "noise_std": 0.03,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 2e-4,
                    "direction_lambda": 0.20,
                    "noise_std": 0.01,
                    "patience": 55,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 2e-4,
                    "direction_lambda": 0.20,
                    "noise_std": 0.01,
                    "patience": 55,
                },
            },
        },
        "transformer": {
            "model": _TRANSFORMER_MODEL,
            "training": {
                "_default": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 220,
                    "patience": 30,
                    "warmup_epochs": 10,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.05,
                    "direction_lambda": 0.0,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "warmup_epochs": 28,
                    "noise_std": 0.02,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 1.5e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.008,
                    "patience": 55,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 1.5e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.008,
                    "patience": 55,
                },
            },
        },
    },

    # ============================================================
    # inst_retired: 标签 Y = Σinst_v1 / Σinst_v2（退役指令总数比）
    # ============================================================
    "inst_retired": {
        "cnn": {
            "model": _CNN_MODEL,
            "training": {
                "_default": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 130,
                    "patience": 25,
                    "warmup_epochs": 8,
                    "huber_delta": 0.8,
                    "grad_clip": 1.0,
                    "noise_std": 0.03,
                    "direction_lambda": 0.05,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.2,
                    "pair_swap": True,
                    "noise_std": 0.04,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 8e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.02,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 8e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.02,
                },
            },
        },
        "lstm": {
            "model": _LSTM_MODEL,
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 180,
                    "patience": 35,
                    "warmup_epochs": 12,
                    "huber_delta": 0.8,
                    "grad_clip": 0.5,
                    "noise_std": 0.02,
                    "direction_lambda": 0.05,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.2,
                    "pair_swap": True,
                    "noise_std": 0.03,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 3e-4,
                    "direction_lambda": 0.20,
                    "noise_std": 0.01,
                    "patience": 45,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 3e-4,
                    "direction_lambda": 0.20,
                    "noise_std": 0.01,
                    "patience": 45,
                },
            },
        },
        "transformer": {
            "model": _TRANSFORMER_MODEL,
            "training": {
                "_default": {
                    "lr": 3e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 180,
                    "patience": 35,
                    "warmup_epochs": 18,
                    "huber_delta": 0.8,
                    "grad_clip": 1.0,
                    "noise_std": 0.01,
                    "direction_lambda": 0.05,
                    "pair_swap": False,
                    "log_target": False,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.2,
                    "pair_swap": True,
                    "warmup_epochs": 22,
                    "noise_std": 0.02,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 2e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.008,
                    "patience": 45,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 2e-4,
                    "direction_lambda": 0.15,
                    "noise_std": 0.008,
                    "patience": 45,
                },
            },
        },
    },
}