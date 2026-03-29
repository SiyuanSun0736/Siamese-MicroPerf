# 硬件环境、模型参数与 LLVM test-suite 变体数据集说明

本文集中说明 Siamese-MicroPerf 的运行硬件前提、模型默认参数，以及基于 LLVM test-suite 构建的数据集变体。重点放在 O1-g、O3-g、O2-bolt、O2-bolt-opt、O3-bolt、O3-bolt-opt 这六类版本的来源、命名含义和它们在训练数据中的角色。

## 1. 硬件与运行环境

### 1.1 基本硬件前提

项目不是纯离线模型仓库，它依赖真实机器上的 PMU 和 LBR 采样。因此这里的“硬件环境”更准确地说是数据采集与 BOLT 优化的运行前提。

- 操作系统：Linux
- 架构建议：x86_64
- 内核能力：支持 perf_event_open
- PMU 权限：需要能够访问硬件性能计数器
- LBR 支持：建议使用支持 LBR 的 CPU，仓库说明中给出的典型范围是 Intel Skylake 及以上，或支持等价能力的 AMD Zen3+

如果机器上 `/proc/sys/kernel/perf_event_paranoid` 大于 1，跨进程 PMU 采样和 perf record 可能失败。仓库里的采集脚本默认按需要使用 sudo 运行。

### 1.2 采集阶段依赖的硬件事件

PMU 采样器 `pmu_monitor` 会周期性采集以下原始事件，并在 CSV 中写出配套的 LBR 统计量：

- inst_retired.any
- L1-icache-load-misses
- iTLB-loads
- iTLB-load-misses
- branch-instructions
- branch-misses
- lbr_avg_span / lbr_log1p_span

其中训练时真正作为输入特征使用的是 6 维：5 个 MPKI 特征加 1 个 LBR 特征 `lbr_log1p_span`。

### 1.3 采样窗口与默认运行参数

仓库里的数据采集脚本使用的是固定物理窗口采样，关键默认值如下：

- PMU 采样窗口：30 秒
- 采样间隔：500 ms
- 固定时间机制下的默认序列长度：60 个时间步
- 最小有效行数：50
- 线程监控：默认开启 LBR TID 监控，用于覆盖子线程创建后的 LBR 挂载场景

这组设置直接影响数据集的时间分辨率，也决定了固定时间和退役指令机制下张量的默认时间维大小。

### 1.4 BOLT 阶段额外依赖

如果要生成 O2-bolt-opt 和 O3-bolt-opt，还需要额外满足：

- 系统安装 llvm-bolt
- 系统安装 perf2bolt
- 系统安装 perf
- CPU 能提供 LBR 分支栈采样，或者退回到插桩模式

也就是说，O1-g 或 O3-g 只需要 PMU 采样环境，而 O2-bolt-opt / O3-bolt-opt 还额外依赖 BOLT 的 profile-guided 二进制重排工具链。

## 2. 模型默认参数

### 2.1 输入与标签

Siamese-MicroPerf 的输入始终是同一程序两个版本的时序特征对：

- 输入形式：`(Seq_v1, Seq_v2)`
- 单步特征维度：6
- 固定时间 / 退役指令机制默认序列长度：60
- 固定工作量机制：序列长度可变，同时额外记录有效长度 `len_v1` 和 `len_v2`

三种标签机制分别是：

- 固定时间：`Y = N_v1 / N_v2`
- 固定工作量：`Y = T_v2 / T_v1`
- 退役指令总数：`Y = Σinst_v1 / Σinst_v2`

虽然物理定义不同，但标签方向保持一致：预测值大于 1 表示版本 v1 更快。

### 2.2 支持的模型主干

仓库当前支持三种 Siamese 主干：

| 主干 | 编码器 | 池化 | 回归头 |
| --- | --- | --- | --- |
| CNN | 共享 1D-CNN | Mask-aware Attention Pooling | MLP |
| LSTM | 共享 BiLSTM | Mask-aware Attention Pooling | MLP |
| Transformer | 共享 Transformer Encoder | Mask-aware Attention Pooling | MLP |

三个模型在高层接口上保持一致，都会把两个版本分别编码成定长向量，然后拼接 `V_v1`、`V_v2` 以及 `V_v1 - V_v2` 送入 MLP 回归头。

### 2.3 训练脚本默认超参数

`python/train.py` 中的默认训练参数如下：

| 参数 | 默认值 |
| --- | --- |
| epochs | 150 |
| batch_size | 32 |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| huber_delta | 1.0 |
| val_ratio | 0.2 |
| seed | 42 |
| patience | 30 |
| grad_clip | 1.0 |
| noise_std | 0.05 |
| warmup_epochs | 10 |
| mlp_hidden | 64 |
| dropout | 0.1 |

损失函数使用 Huber Loss，优化器使用 Adam，学习率调度采用“线性预热 + 余弦退火”。

### 2.4 各主干默认结构参数

#### CNN

- cnn_hidden = 64
- cnn_out = 128

CNN 主干本质上是共享的 1D 卷积编码器，编码结果经过注意力池化后得到版本级表示。

#### LSTM

- lstm_hidden = 64
- lstm_out = 128
- num_layers = 3
- bidirectional = true

这里的 `num_layers` 是训练脚本里统一暴露的命令行参数，对 LSTM 来说表示堆叠的 LSTM 层数。

#### Transformer

- d_model = 128
- nhead = 4
- num_layers = 3
- dim_feedforward = 256
- max_len = 512
- pos_encoding = learnable

同样的 `num_layers` 参数在 Transformer 中表示 encoder layer 的层数。

## 3. 基于 LLVM test-suite 的变体数据集

### 3.1 数据集来源

项目中的原始程序来源是仓库内嵌的 LLVM test-suite。采集流程并不是直接对源码做在线编译，而是先在 llvm-test-suite 下生成不同 build 目录，再把 ELF 可执行文件和对应测试输入抽取到 `train_set` 中，随后执行 PMU 采样和张量构建。

数据资产在 `train_set` 下分成几层：

- `bin/<variant>`：该变体对应的可执行文件
- `test/<variant>`：该变体对应的 `.test` 规格与运行时文件
- `data/<variant>`：该变体的 PMU CSV 时序日志
- `manifest_<variant>.jsonl`：该变体的样本清单
- `tensors/fixed_time`、`tensors/fixed_work`、`tensors/inst_retired`：最终训练张量

每条 manifest 记录会保存程序名、变体名、二进制路径、运行命令、CSV 路径和固定窗口内的运行次数 `run_count`。

### 3.2 先澄清命名语义

这里最容易误解的是 `-bolt` 后缀。

在当前仓库里：

- `O2-bolt` 和 `O3-bolt` 不是“已经做完 BOLT 优化”的最终版本
- 它们更准确地说是“为 BOLT 准备好的 baseline binary”
- 真正应用了 profile-guided BOLT 重排后的版本是 `O2-bolt-opt` 和 `O3-bolt-opt`

因此，训练对里真正代表“BOLT 前后对比”的是：

- `O2-bolt` 对 `O2-bolt-opt`
- `O3-bolt` 对 `O3-bolt-opt`

### 3.3 六类变体的构建方式

| 变体 | llvm-test-suite build 目录 | cache 文件 / 来源 | 主要编译或优化特征 | 在数据集中的角色 |
| --- | --- | --- | --- | --- |
| O1-g | build-O1-g | cmake/caches/O1-g.cmake | `-O1 -g` | 优化等级对比中的低优化版本 |
| O3-g | build-O3-g | cmake/caches/O3-g.cmake | `-O3 -g` | 优化等级对比中的高优化版本 |
| O2-bolt | build-O2-bolt | cmake/caches/O2-bolt.cmake | `-O2`，链接时加 `--emit-relocs -no-pie -Wl,-z,now` | BOLT 前的 baseline |
| O2-bolt-opt | 由 O2-bolt 再生成 | train_set/bolt_optimize.sh | 基于 perf/LBR profile 的 BOLT 重排 | BOLT 后版本 |
| O3-bolt | build-O3-bolt | cmake/caches/O3-bolt.cmake | `-O3`，链接时加 `--emit-relocs -no-pie -Wl,-z,now` | BOLT 前的 baseline |
| O3-bolt-opt | 由 O3-bolt 再生成 | train_set/bolt_optimize.sh | 基于 perf/LBR profile 的 BOLT 重排 | BOLT 后版本 |

### 3.4 O1-g、O3-bolt、O3-bolt-opt 的重点说明

#### O1-g

O1-g 来自 `llvm-test-suite/build-O1-g`，使用 `O1-g.cmake` 以 `-O1 -g` 编译。它的作用不是和 BOLT 系列直接对比，而是与 O3-g 组成“传统编译优化等级差异”这一组训练数据。

这组数据回答的是：只改变前端编译优化等级时，PMU/LBR 时序是否足以预测性能倍率。

#### O3-bolt

O3-bolt 来自 `llvm-test-suite/build-O3-bolt`，编译时使用 `-O3`，同时在链接阶段加入：

- `-Wl,--emit-relocs`
- `-no-pie`
- `-Wl,-z,now`

这些设置的目的不是直接提速，而是让后续 BOLT 能够读取足够的重定位信息并稳定重排代码布局。因此 O3-bolt 是 O3 级别下的“BOLT-ready baseline”。

#### O3-bolt-opt

O3-bolt-opt 不是独立从源码再次编译出来的，它是从 O3-bolt 出发，经 `train_set/bolt_optimize.sh` 二次生成的。脚本对每个 benchmark 执行三步：

1. 用 perf record 和 LBR 采样运行中的 O3-bolt 可执行文件
2. 用 perf2bolt 把 perf.data 转成 BOLT 可消费的 fdata
3. 用 llvm-bolt 生成新的 O3-bolt-opt 二进制

默认启用的 BOLT 关键参数包括：

- `-reorder-blocks=ext-tsp`
- `-reorder-functions=hfsort`
- `-split-functions`
- `-split-all-cold`
- `-split-eh`
- `-dyno-stats`
- `-plt=hot`

因此 O3-bolt-opt 对应的是“同样的 O3 编译前提下，再做一次 profile-guided post-link layout optimization 后”的版本。它和 O3-bolt 配对后，能较直接地反映 BOLT 对指令缓存、iTLB 和分支局部性的影响。

### 3.5 变体到训练张量的映射关系

张量构建脚本并不是把六个变体混在一起训练，而是组织成三个固定版本对：

| 训练对 | 物理含义 |
| --- | --- |
| O1-g_vs_O3-g | 传统编译优化等级差异 |
| O2-bolt_vs_O2-bolt-opt | O2 下的 BOLT 前后差异 |
| O3-bolt_vs_O3-bolt-opt | O3 下的 BOLT 前后差异 |

这三组版本对会分别被写入三类张量目录：

- `train_set/tensors/fixed_time/<pair_name>`
- `train_set/tensors/fixed_work/<pair_name>`
- `train_set/tensors/inst_retired/<pair_name>`

也就是说，同一组版本对会对应三种标签语义，但底层 PMU / LBR 原始日志来源一致。

### 3.6 从 LLVM test-suite 到最终张量的完整链路

完整流程可以概括为：

1. 在 `llvm-test-suite` 中按不同 cache 构建变体目录，例如 `build-O1-g`、`build-O3-bolt`
2. 用 `train_set/extract_elf.sh` 从 build 目录提取 ELF、`.test` 文件和运行时数据到 `train_set/bin/<variant>` 与 `train_set/test/<variant>`
3. 用 `train_set/collect_dataset_testbench.sh` 对每个变体执行 PMU 采样，输出 `train_set/data/<variant>` 与 `manifest_<variant>.jsonl`
4. 对 `O2-bolt` 和 `O3-bolt` 运行 `train_set/bolt_optimize.sh`，生成 `O2-bolt-opt` 与 `O3-bolt-opt`
5. 再次对 `*-bolt-opt` 版本执行 PMU 采样
6. 用 `python/build_dataset_fixedtime.py`、`python/build_dataset_fixedwork.py`、`python/build_dataset_instret.py` 生成最终训练张量

顶层脚本 `train_set/generate_train_set.sh` 已经把这个流程按变体顺序串起来了。

## 4. 结论

如果只看建模接口，Siamese-MicroPerf 像是在比较两个时间序列；但从数据工程角度看，它比较的其实是 LLVM test-suite 中同一程序的不同编译和后链接优化版本。理解 O1-g、O3-g、O2-bolt、O2-bolt-opt、O3-bolt、O3-bolt-opt 之间的来源关系，是理解整个训练集语义的前提。

其中最关键的一点是：

- `-g` 系列表达的是传统编译优化等级差异
- `-bolt` 系列表达的是“为 BOLT 准备的基线”
- `-bolt-opt` 才是实际应用了 BOLT profile-guided 布局优化后的版本

这也是为什么仓库最终训练的不是六个独立类别，而是三个有明确物理语义的版本对。
