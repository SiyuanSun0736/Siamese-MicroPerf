# 硬件前提、训练配置与 LLVM test-suite 数据集说明

本文按当前仓库中的采样脚本、张量构建脚本和训练入口重新整理。重点回答四个问题：

- 这套流程对机器和权限有什么要求
- 模型真正接收的输入、标签和默认训练行为是什么
- O1-g、O3-g、O2-bolt、O2-bolt-opt、O3-bolt、O3-bolt-opt 六类变体分别从哪里来
- 从 llvm-test-suite 到最终训练张量的链路在当前代码里是怎样串起来的

和旧版本文档相比，下面的描述以当前实现为准，尤其修正了两点：

- 训练默认值不再只是 `train.py` 里的 argparse 常数，而是“基础默认值 + auto-tune 预设”的组合
- `train_set/generate_train_set.sh` 目前只负责六类变体的提取、采样和 BOLT 生成，不会自动调用 Python 脚本生成张量

## 1. 运行前提

### 1.1 机器、系统与权限

Siamese-MicroPerf 不是纯离线模型仓库。它依赖真实机器上的 PMU 计数器和 LBR 采样，因此首先要满足数据采集条件。

- 操作系统：Linux
- 架构建议：x86_64
- 内核能力：支持 `perf_event_open`
- PMU 权限：需要能访问硬件性能计数器
- 建议 CPU：支持 LBR 的 Intel 平台，或具备等价能力的 AMD 平台

如果 `/proc/sys/kernel/perf_event_paranoid` 大于 1，`pmu_monitor` 和 `perf record` 往往无法正常工作。当前采集脚本默认就是按需要通过 `sudo` 运行。

另外，`collect_dataset_testbench.sh` 默认开启 `LBR_TID_MON=1`，这意味着它会通过 `tid_monitor` 在子线程创建时补挂 LBR 采样。这个模式除了 PMU 权限外，还依赖 `CAP_NET_ADMIN` 来访问 `NETLINK_CONNECTOR`。

### 1.2 PMU 采样与原始事件

底层 PMU 采样器 `pmu_monitor` 当前会周期性记录 6 个原始计数器：

- `inst_retired.any`
- `L1-icache-load-misses`
- `iTLB-loads`
- `iTLB-load-misses`
- `branch-instructions`
- `branch-misses`

CSV 中还会带上 LBR 统计量，当前特征工程实际使用的是 `lbr_log1p_span`。因此需要区分两层语义：

- 原始采样层：保存 `inst_retired.any`、5 个原始 PMU 计数器，以及 LBR 统计量
- 训练输入层：真正喂给 Siamese 模型的是 6 维特征，即 5 个 MPKI 特征加 1 个 LBR 特征

也就是说，`inst_retired.any` 在训练阶段通常不直接作为输入维度，而是承担两类辅助角色：

- 作为 MPKI 的分母，把原始 PMU 计数器转换成每千条指令事件数
- 在 `inst_retired` 标签机制下，作为标签 `Σinst_v1 / Σinst_v2` 的数据来源

### 1.3 当前脚本里的采样默认值

当前仓库在采集阶段仍然使用固定物理窗口，默认参数如下：

- PMU 采样窗口：30 秒
- 采样间隔：500 ms
- 最小有效行数：50
- 连续循环模式：默认关闭，即 `CONTINUOUS=0`
- LBR 子线程覆盖：默认开启，即 `LBR_TID_MON=1`

这些参数直接决定了后续张量的时间维解释：

- `fixed_time`：默认 `seq_len=60`，对应 `30s / 500ms = 60` 个时间步
- `inst_retired`：同样默认 `seq_len=60`
- `fixed_work`：`max_seq_len` 默认也是 60，但每个版本的有效长度是动态计算的，额外保存 `len_v1` 和 `len_v2`

### 1.4 BOLT 阶段的附加前提

如果只采集 `O1-g` 或 `O3-g`，机器只需要满足 PMU/LBR 采样条件；如果要构建 `O2-bolt-opt` 或 `O3-bolt-opt`，还要额外满足 BOLT 工具链条件：

- 已安装 `llvm-bolt`
- 已安装 `perf2bolt`
- 已安装 `perf`
- 输入二进制以 BOLT-ready 方式链接，带有 `--emit-relocs -no-pie -Wl,-z,now`

默认情况下，`bolt_optimize.sh` 走 LBR 模式；如果机器不支持 LBR，可以把 `USE_INSTRUMENTATION=1`，退回到插桩式 profile 生成。

## 2. 输入张量、特征工程与标签机制

### 2.1 输入形式

Siamese-MicroPerf 的训练样本始终是同一程序两个版本的时序特征对：

- 输入形式：`(Seq_v1, Seq_v2)`
- 单步特征维度：6
- 两塔共享编码器参数

三类标签机制共用同一批 PMU/LBR 原始日志，但张量组织方式不同：

| 标签机制 | 时间维 | 额外长度信息 | 标签 |
| --- | --- | --- | --- |
| `fixed_time` | 固定 60 步 | 否 | `Y = N_v1 / N_v2` |
| `fixed_work` | 填充到 `max_seq_len`，有效长度动态变化 | 是，保存 `len_v1` / `len_v2` | `Y = T_v2 / T_v1` |
| `inst_retired` | 固定 60 步 | 否 | `Y = Σinst_v1 / Σinst_v2` |

三个标签的方向保持一致：预测值大于 1 表示版本 `v1` 更快。

### 2.2 当前实现里的 6 维输入特征

训练脚本不会直接把原始计数器送进模型，而是先做特征工程。当前 6 维输入是：

- `icache_miss_mpki`
- `itlb_load_mpki`
- `itlb_miss_mpki`
- `branch_inst_mpki`
- `branch_miss_mpki`
- `lbr_log1p_span`

其中特征工程规则是：

1. 以 `inst_retired.any` 为分母，把 5 个原始事件转成 MPKI
2. 丢弃 `inst_retired.any == 0` 的无效行，避免产生无物理意义的尖刺
3. 直接使用 CSV 中已经算好的 `lbr_log1p_span`
4. 对有效时间步做 Z-score 标准化

### 2.3 三种标签机制的真实含义

#### `fixed_time`

`fixed_time` 使用固定物理窗口内的完成次数比：

- `Y = N_v1 / N_v2`

其中 `N_v1`、`N_v2` 来自 manifest 中记录的 `run_count`。这反映的是固定 30 秒采样窗口内的吞吐差异。

#### `fixed_work`

`fixed_work` 的物理定义是固定工作量下的耗时比：

- `Y = T_v2 / T_v1`

当前实现并不是重新在线运行固定工作量实验，而是基于固定时间采样结果做重解释：

- 先取 `N_ref = min(N_v1, N_v2)` 作为双方共同完成的工作量
- 再按 `effective_len = round(T_raw * N_ref / N_i)` 从 60 步原始序列中截出有效区间

因此在当前实现里，`fixed_work` 的标签数值仍然等于 `N_v1 / N_v2`，但物理解释从“固定时间下的吞吐比”切换成了“固定工作量下的耗时比”。

#### `inst_retired`

`inst_retired` 不依赖外部 `run_count`，而是直接对每条 CSV 中所有有效行的 `inst_retired.any` 求和：

- `Y = Σinst_v1 / Σinst_v2`

这表示在同一固定采样窗口里，哪个版本完成了更多退役指令，标签来源完全由硬件计数器决定。

### 2.4 最终张量目录

当前 Python 构建脚本会把三类标签机制分别写入：

- `train_set/tensors/fixed_time/<pair_name>`
- `train_set/tensors/fixed_work/<pair_name>`
- `train_set/tensors/inst_retired/<pair_name>`

其中典型文件包括：

- `X_v1.pt`
- `X_v2.pt`
- `Y.pt`
- `programs.json`
- `stats.json`

`fixed_work` 还会额外保存：

- `len_v1.pt`
- `len_v2.pt`

## 3. 模型与训练配置

### 3.1 支持的 Siamese 主干

仓库当前支持三种 Siamese 主干，接口风格保持一致：先分别编码 `v1` 和 `v2`，再把表示向量拼接后送入回归头。

| 主干 | 编码器 | 时序聚合 | 回归输入 |
| --- | --- | --- | --- |
| CNN | 共享 1D-CNN | Mask-aware Attention Pooling | `[V_v1; V_v2; V_v1 - V_v2]` |
| LSTM | 共享 BiLSTM | Mask-aware Attention Pooling | `[V_v1; V_v2; V_v1 - V_v2]` |
| Transformer | 共享 Transformer Encoder | Mask-aware Attention Pooling | `[V_v1; V_v2; V_v1 - V_v2]` |

因此从高层接口看，三种模型的主要差别在编码器内部，不在数据接口和输出头部格式上。

### 3.2 关闭 `auto-tune` 时的基础训练默认值

如果显式传入 `--no-auto-tune`，训练会回到 `train.py` 中的 argparse 基础默认值：

| 参数 | 默认值 |
| --- | --- |
| `epochs` | 150 |
| `batch_size` | 32 |
| `lr` | `1e-3` |
| `weight_decay` | `1e-4` |
| `huber_delta` | 1.0 |
| `val_ratio` | 0.15 |
| `test_ratio` | 0.15 |
| `seed` | 42 |
| `patience` | 30 |
| `grad_clip` | 1.0 |
| `noise_std` | 0.05 |
| `warmup_epochs` | 10 |
| `log_target` | false |
| `direction_lambda` | 0.0 |
| `pair_swap` | false |

优化器仍然是 Adam，学习率调度仍然是“线性预热 + 余弦退火”。

### 3.3 关闭 `auto-tune` 时的主干基础默认值

同样在 `--no-auto-tune` 条件下，各主干的基础结构参数如下：

#### CNN

- `cnn_hidden = 64`
- `cnn_out = 128`
- `mlp_hidden = 64`
- `dropout = 0.1`

#### LSTM

- `lstm_hidden = 64`
- `lstm_out = 128`
- `bidirectional = true`
- `num_layers = 3`
- `mlp_hidden = 64`
- `dropout = 0.1`

这里的 `num_layers=3` 来自训练入口的共享命令行参数；也就是说，关闭 `auto-tune` 时，LSTM 默认会沿用这个共享层数。

#### Transformer

- `d_model = 128`
- `nhead = 4`
- `num_layers = 3`
- `dim_feedforward = 256`
- `max_len = 512`
- `pos_encoding = learnable`
- `mlp_hidden = 64`
- `dropout = 0.1`

### 3.4 默认运行行为：`auto-tune` 处于开启状态

当前代码里，真正的“默认训练行为”并不是上一节那一组常数，而是下面这套规则：

1. `--auto-tune` 默认开启
2. 如果用户没有手动指定 `--label-mechanism`，训练脚本会根据 `tensor_base` 路径自动识别：
   - 路径中包含 `fixed_work`，则使用 `fixed_work`
   - 路径中包含 `inst_retired` 或 `instret`，则使用 `inst_retired`
   - 其余情况默认视为 `fixed_time`
3. 根据“标签机制 + 模型类型”从 `python/tuned_configs.py` 应用预设
4. 如果这次只训练单一版本对，还会继续叠加该版本对的专属覆盖
5. 任何用户显式传入的命令行参数优先级都最高，不会被预设覆盖

这意味着“默认训练参数”在当前仓库里是上下文相关的：

- 训练 `fixed_time` 和训练 `fixed_work`，默认超参数并不一样
- 训练 CNN、LSTM、Transformer，默认结构也不一样
- 单独训练 `O1-g_vs_O3-g` 或单独训练某个 BOLT 对时，默认值还会进一步变化

### 3.5 `auto-tune` 预设的基础轮廓

为了方便理解，可以把 `tuned_configs.py` 看成三层：

- 第一层：按标签机制划分 `fixed_time`、`fixed_work`、`inst_retired`
- 第二层：每个标签机制下再按 `cnn`、`lstm`、`transformer` 划分
- 第三层：先应用该模型的 `_default` 预设；如果只训练一个版本对，再叠加该版本对的 override

不考虑单对数据集的专属覆盖时，三类机制下的基础预设如下。

| 标签机制 | 模型 | 结构预设 | 训练预设摘要 |
| --- | --- | --- | --- |
| `fixed_time` | CNN | `cnn_hidden=128`, `cnn_out=256`, `mlp_hidden=256`, `dropout=0.25` | `epochs=180`, `lr=3e-4`, `patience=50`, `warmup=15`, `log_target=true`, `pair_swap=true` |
| `fixed_time` | LSTM | `lstm_hidden=32`, `lstm_out=64`, `num_layers=2`, `bidirectional=true`, `mlp_hidden=32`, `dropout=0.20` | `epochs=200`, `lr=5e-4`, `weight_decay=1e-3`, `patience=50`, `log_target=true`, `pair_swap=true` |
| `fixed_time` | Transformer | `d_model=32`, `nhead=2`, `num_layers=2`, `dim_feedforward=64`, `max_len=512`, `mlp_hidden=32`, `dropout=0.10` | `epochs=400`, `lr=1e-3`, `patience=80`, `warmup=20`, `log_target=true`, `pair_swap=false` |
| `fixed_work` | CNN | `cnn_hidden=64`, `cnn_out=128`, `mlp_hidden=128`, `dropout=0.10` | `epochs=180`, `lr=5e-4`, `patience=50`, `log_target=true`, `pair_swap=true` |
| `fixed_work` | LSTM | `lstm_hidden=32`, `lstm_out=64`, `num_layers=2`, `bidirectional=true`, `mlp_hidden=32`, `dropout=0.20` | `epochs=200`, `lr=5e-4`, `weight_decay=1e-3`, `patience=50`, `log_target=true`, `pair_swap=true` |
| `fixed_work` | Transformer | `d_model=64`, `nhead=2`, `num_layers=3`, `dim_feedforward=256`, `max_len=128`, `mlp_hidden=64`, `dropout=0.10` | `epochs=220`, `lr=1e-3`, `patience=30`, `warmup=10`, `log_target=false`, `pair_swap=false` |
| `inst_retired` | CNN | `cnn_hidden=64`, `cnn_out=128`, `mlp_hidden=128`, `dropout=0.10` | `epochs=180`, `lr=5e-4`, `patience=50`, `log_target=false`, `pair_swap=false` |
| `inst_retired` | LSTM | `lstm_hidden=64`, `lstm_out=32`, `num_layers=2`, `bidirectional=true`, `mlp_hidden=32`, `dropout=0.20` | `epochs=200`, `lr=5e-4`, `weight_decay=5e-4`, `patience=50`, `log_target=true`, `pair_swap=true` |
| `inst_retired` | Transformer | `d_model=64`, `nhead=2`, `num_layers=2`, `dim_feedforward=256`, `max_len=512`, `mlp_hidden=64`, `dropout=0.0` | `batch_size=16`, `epochs=300`, `lr=5e-4`, `patience=60`, `log_target=true`, `pair_swap=true` |

如果只训练单一版本对，还会再叠加 pair-specific 覆盖。例如：

- `O1-g_vs_O3-g` 会为多个模型额外放宽 `huber_delta`，并调整 `noise_std` 或 `pair_swap`
- 两个 BOLT 对共享一套 `BOLT_OPT_VARIANTS` 覆盖，会额外调整 `lr`、`direction_lambda`、`noise_std` 和 `patience`

因此在当前仓库里，更准确的表述不是“训练脚本只有一组默认超参数”，而是“训练脚本有一组基础 CLI 默认值，但默认运行路径会按标签机制、模型类型和版本对应用预设”。

## 4. 基于 LLVM test-suite 的六类变体

### 4.1 当前数据资产的目录分层

当前仓库把从 llvm-test-suite 提取出来的训练资产组织在 `train_set` 下：

- `bin/<variant>`：该变体对应的可执行文件
- `test/<variant>`：对应的 `.test` 文件和运行时数据
- `data/<variant>`：PMU CSV 时序数据
- `manifest_<variant>.jsonl`：该变体的数据清单
- `bolt_profiles/<variant>`：BOLT 的 `perf.data`、`fdata` 等中间产物
- `tensors/fixed_time`、`tensors/fixed_work`、`tensors/inst_retired`：最终训练张量

manifest 当前每行会记录：

- `program`
- `variant`
- `binary`
- `run_cmd`
- `csv`
- `run_count`

### 4.2 六类变体的来源与命名语义

当前仓库最终围绕六类变体组织数据：

- `O1-g`
- `O3-g`
- `O2-bolt`
- `O2-bolt-opt`
- `O3-bolt`
- `O3-bolt-opt`

其中最重要的命名语义是：

- `-g` 版本表达的是传统编译优化等级差异
- `-bolt` 版本是“BOLT-ready baseline”，不是已经完成 BOLT 优化的结果
- `-bolt-opt` 才是实际执行过 `llvm-bolt` 之后的版本

这六类变体和它们的来源关系如下：

| 变体 | 来源 | 编译 / 优化特征 | 在数据集中的角色 |
| --- | --- | --- | --- |
| `O1-g` | `llvm-test-suite/build-O1-g` | 由 `O1-g.cmake` 以 `-O1 -g` 构建 | 低优化基线，用于和 `O3-g` 对比 |
| `O3-g` | `llvm-test-suite/build-O3-g` | 由 `O3-g.cmake` 以 `-O3 -g` 构建 | 高优化版本，用于和 `O1-g` 对比 |
| `O2-bolt` | `llvm-test-suite/build-O2-bolt` | 由 `O2-bolt.cmake` 以 `-O2` 构建，链接时加入 `--emit-relocs -no-pie -Wl,-z,now` | O2 下的 BOLT 前基线 |
| `O2-bolt-opt` | 由 `O2-bolt` 二次生成 | `bolt_optimize.sh` 通过 `perf record/perf2bolt/llvm-bolt` 生成 | O2 下的 BOLT 后版本 |
| `O3-bolt` | `llvm-test-suite/build-O3-bolt` | 由 `O3-bolt.cmake` 以 `-O3` 构建，链接时加入 `--emit-relocs -no-pie -Wl,-z,now` | O3 下的 BOLT 前基线 |
| `O3-bolt-opt` | 由 `O3-bolt` 二次生成 | `bolt_optimize.sh` 通过 `perf record/perf2bolt/llvm-bolt` 生成 | O3 下的 BOLT 后版本 |

### 4.3 `extract_elf.sh` 当前如何解释“来源”

当前 `train_set/extract_elf.sh` 的默认行为是：

- 默认把版本名映射成 `build-<VERSION>`
- 默认扫描 `llvm-test-suite/build-<VERSION>/MultiSource/Benchmarks`
- 识别其中的 ELF 可执行文件
- 复制到 `train_set/bin/<VERSION>`
- 把对应 `.test` 文件和运行时依赖同步复制到 `train_set/test/<VERSION>`

因此从当前脚本实现上说，这套数据流的直接来源更准确地表述为：

- 先在 llvm-test-suite 中构建不同变体的 `build-*` 目录
- 再从这些 build 目录下的 `MultiSource/Benchmarks` 抽取可执行文件与测试规格

### 4.4 `bolt_optimize.sh` 当前如何生成 `-bolt-opt`

`bolt_optimize.sh` 现在按基准执行三步：

1. 用 `perf record` 对 `O2-bolt` 或 `O3-bolt` 二进制做 profile 采样
2. 用 `perf2bolt` 把采样结果转成 BOLT 可消费的 `fdata`
3. 用 `llvm-bolt` 生成 `OUT_VARIANT=${VARIANT}-opt` 的新二进制

脚本当前默认启用的 BOLT 关键参数包括：

- `-reorder-blocks=ext-tsp`
- `-reorder-functions=hfsort`
- `-split-functions`
- `-split-all-cold`
- `-split-eh`
- `-dyno-stats`
- `-plt=hot`

所以 `O2-bolt-opt` 和 `O3-bolt-opt` 的意义非常明确：它们不是重新从源码编译出来的新优化等级，而是对已经构建好的 `*-bolt` baseline 执行 profile-guided post-link layout optimization 之后的二进制。

## 5. 当前代码里的训练版本对

当前张量构建脚本默认使用三组版本对：

| 训练对 | 物理语义 |
| --- | --- |
| `O1-g_vs_O3-g` | 传统编译优化等级差异 |
| `O2-bolt_vs_O2-bolt-opt` | O2 下的 BOLT 前后差异 |
| `O3-bolt_vs_O3-bolt-opt` | O3 下的 BOLT 前后差异 |

这三组训练对会分别出现在三类张量目录下：

- `train_set/tensors/fixed_time/<pair_name>`
- `train_set/tensors/fixed_work/<pair_name>`
- `train_set/tensors/inst_retired/<pair_name>`

训练脚本里还有一个和数据集选择相关的快捷开关：

- `--bolt-opt`：只保留两个 BOLT 对，即 `O2-bolt_vs_O2-bolt-opt` 和 `O3-bolt_vs_O3-bolt-opt`

## 6. 从 llvm-test-suite 到最终张量的当前链路

按照当前脚本，完整流程应该拆成两个阶段来看。

### 6.1 阶段一：六类变体的提取、采样与 BOLT 生成

这一阶段由 shell 脚本负责，典型顺序如下：

1. 在 `llvm-test-suite` 中准备 `build-O1-g`、`build-O3-g`、`build-O2-bolt`、`build-O3-bolt`
2. 用 `train_set/extract_elf.sh -v <variant>` 提取 ELF 和 `.test` 文件到 `train_set/bin/<variant>` 与 `train_set/test/<variant>`
3. 用 `train_set/collect_dataset_testbench.sh -v <variant>` 采集 PMU CSV，生成 `train_set/data/<variant>` 与 `manifest_<variant>.jsonl`
4. 对 `O2-bolt` 和 `O3-bolt` 运行 `train_set/bolt_optimize.sh -v <variant>`，生成对应的 `*-bolt-opt`
5. 再对 `O2-bolt-opt` 和 `O3-bolt-opt` 执行一次 PMU 采样

`train_set/generate_train_set.sh` 当前实际覆盖的就是这部分流程。它按顺序串起：

- `O1-g`
- `O3-g`
- `O2-bolt` → `O2-bolt-opt`
- `O3-bolt` → `O3-bolt-opt`

### 6.2 阶段二：从原始 CSV 构建训练张量

在原始数据已经准备好之后，再由 Python 脚本分别构建三类标签机制下的张量：

1. `python/build_dataset_fixedtime.py`
2. `python/build_dataset_fixedwork.py`
3. `python/build_dataset_instret.py`

这一步才会真正生成 `train_set/tensors/...` 下的 `X_v1.pt`、`X_v2.pt`、`Y.pt` 和统计文件。

换句话说，当前仓库里的顶层数据链路不是“一个脚本从头包到尾”，而是：

- shell 脚本负责构建六类原始变体及其 PMU 数据
- Python 脚本负责把这些原始数据转成三类训练张量

## 7. 结论

从建模接口上看，Siamese-MicroPerf 做的是“同一程序两个版本的时序特征回归”；从当前代码实现看，它实际上建立在一条比较清晰的数据工程链路之上：

- `O1-g` 对 `O3-g` 表示传统编译优化等级差异
- `O2-bolt` 对 `O2-bolt-opt`、`O3-bolt` 对 `O3-bolt-opt` 表示 BOLT 前后差异
- 三组版本对共享同样的 PMU/LBR 原始采样机制，再分别投影成 `fixed_time`、`fixed_work` 和 `inst_retired` 三种标签语义

如果只记一个关键结论，那就是：

- `-g` 系列是编译优化等级对比
- `-bolt` 是 BOLT-ready baseline
- `-bolt-opt` 才是 BOLT 实际优化后的版本

理解这三个层次，再去看训练配置、张量目录和实验结果，就不会把六类变体误解成六个互不相关的类别。