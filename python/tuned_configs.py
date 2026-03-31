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
    "d_model": 64,
    "nhead": 2,
    "num_layers": 3,
    "dim_feedforward": 256,
    "max_len": 128,
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
            "model": {
                "cnn_hidden": 128,
                "cnn_out": 256,
                "mlp_hidden": 256,
                "dropout": 0.25,
            },
            "training": {
                "_default": {
                    "lr": 3e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 180,
                    "patience": 50,
                    "warmup_epochs": 15,
                    "huber_delta": 0.01,
                    "grad_clip": 1.0,
                    "noise_std": 0.01,
                    "direction_lambda": 0.5,
                    "pair_swap": True,
                    "log_target": True,
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
            "model":{
                    "lstm_hidden": 32,
                    "lstm_out": 64,
                    "num_layers": 2,
                    "bidirectional": True,
                    "mlp_hidden": 32,
                    "dropout": 0.20,
                },
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-3,
                    "batch_size": 32,
                    "epochs": 200,
                    "patience": 50,
                    "warmup_epochs": 15,
                    "huber_delta": 0.05,
                    "grad_clip": 1.0,
                    "noise_std": 0.01,
                    "direction_lambda": 0.3,
                    "pair_swap": True,
                    "log_target": True,
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
                    "lr": 2e-4,
                    "weight_decay": 1e-3,
                    "batch_size": 32,
                    "epochs": 250,
                    "patience": 60,
                    "warmup_epochs": 20,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.02,
                    "direction_lambda": 1.0, 
                    "pair_swap": True,
                    "log_target": True,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.5,
                    "pair_swap": True,
                    "warmup_epochs": 25,
                    "noise_std": 0.02,
                    "direction_lambda": 1.0,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 1.5e-4,
                    "direction_lambda": 0.8,
                    "noise_std": 0.01,
                    "patience": 60,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.5,
                    "lr": 1.5e-4,
                    "direction_lambda": 0.8,
                    "noise_std": 0.01,
                    "patience": 60,
                },
            },
        },
    },

    # ============================================================
    # fixed_work: 标签 Y = T_v2 / T_v1（固定工作量耗时比）
    # ============================================================
    "fixed_work": {
        "cnn": {
            "model": {
                "cnn_hidden": 64,
                "cnn_out": 128,
                "mlp_hidden": 128,
                "dropout": 0.10,
            },
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 180,
                    "patience": 50,
                    "warmup_epochs": 15,
                    "huber_delta": 0.05,
                    "grad_clip": 1.0,
                    "noise_std": 0.03,
                    "direction_lambda": 0.2,
                    "pair_swap": True,
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
             "model":{
                    "lstm_hidden": 32,
                    "lstm_out": 64,
                    "num_layers": 2,
                    "bidirectional": True,
                    "mlp_hidden": 32,
                    "dropout": 0.20,
                },
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-3,
                    "batch_size": 32,
                    "epochs": 200,
                    "patience": 50,
                    "warmup_epochs": 15,
                    "huber_delta": 0.05,
                    "grad_clip": 1.0,
                    "noise_std": 0.01,
                    "direction_lambda": 0.3,
                    "pair_swap": True,
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
            "model": {
                    "d_model": 64,
                    "nhead": 2,
                    "num_layers": 3,
                    "dim_feedforward": 256,
                    "max_len": 128,
                    "pos_encoding": "learnable",
                    "mlp_hidden": 64,
                    "dropout": 0.10,
                },
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
            "model": {
                "cnn_hidden": 64,
                "cnn_out": 128,
                "mlp_hidden": 128,
                "dropout": 0.10,
            },
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 180,
                    "patience": 50,
                    "warmup_epochs": 15,
                    "huber_delta": 0.05,
                    "grad_clip": 1.0,
                    "noise_std": 0.005,
                    "direction_lambda": 0.5,
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
            "model":{
                    "lstm_hidden": 64,
                    "lstm_out": 32,
                    "num_layers": 2,
                    "bidirectional": True,
                    "mlp_hidden": 32,
                    "dropout": 0.20,
                },
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 5e-4,
                    "batch_size": 32,
                    "epochs": 200,
                    "patience": 50,
                    "warmup_epochs": 15,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.02,
                    "direction_lambda": 1.0,
                    "pair_swap": True,
                    "log_target": True,
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
            "model": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 3,
                    "dim_feedforward": 256,
                    "max_len": 128,
                    "pos_encoding": "learnable",
                    "mlp_hidden": 64,
                    "dropout": 0.05,
                },
            "training": {
                "_default": {
                    "lr": 5e-4,
                    "weight_decay": 1e-4,
                    "batch_size": 32,
                    "epochs": 220,
                    "patience": 50,
                    "warmup_epochs": 20,
                    "huber_delta": 1.0,
                    "grad_clip": 1.0,
                    "noise_std": 0.02,
                    "direction_lambda": 0.1,
                    "pair_swap": True,
                    "log_target": True,
                },
                "O1-g_vs_O3-g": {
                    "huber_delta": 1.2,
                    "pair_swap": True,
                    "warmup_epochs": 25,
                    "noise_std": 0.02,
                    "direction_lambda": 0.8,
                },
                "O2-bolt_vs_O2-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 1.5e-4,
                    "direction_lambda": 0.6,
                    "noise_std": 0.01,
                    "patience": 60,
                },
                "O3-bolt_vs_O3-bolt-opt": {
                    "huber_delta": 0.4,
                    "lr": 1.5e-4,
                    "direction_lambda": 0.6,
                    "noise_std": 0.01,
                    "patience": 60,
                },
            },
        },
    },
}