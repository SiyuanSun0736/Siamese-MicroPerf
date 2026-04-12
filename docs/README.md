# 文档导航

这份索引页按“先看什么、再看什么”的顺序组织当前 `docs/` 目录，方便从总览一路跳到数据、模型、标签和结果细节。

## 快速入口

- 项目总览与运行方法： [../README.md](../README.md)
- 仓库里的文档目录： [README.md](README.md)
- 训练与数据集总览： [hardware-model-dataset.md](hardware-model-dataset.md)
- 标签语义与三种构建脚本差异： [label-mechanisms.md](label-mechanisms.md)
- 模型结构详解： [model-architecture.md](model-architecture.md)

## 按主题阅读

### 1. 系统与采集链路

- PMU、LBR、特征工程全链路： [data-collection-feature-engineering.md](data-collection-feature-engineering.md)
- BOLT 相关背景与生成流程： [bolt.md](bolt.md)
- llvm-test-suite 数据来源与变体关系： [llvm-test-suite.md](llvm-test-suite.md)

### 2. 数据集与标签

- 三种标签机制、物理定义与脚本入口： [label-mechanisms.md](label-mechanisms.md)
- 当前硬件前提、训练默认行为、六类变体来源： [hardware-model-dataset.md](hardware-model-dataset.md)

### 3. 模型与算法

- Siamese 架构、共享编码器、融合方式： [model-architecture.md](model-architecture.md)
- 更偏算法表述的说明： [algorithm.md](algorithm.md)

### 4. 结果与分析

- 汇总后的 overall prediction accuracy： [overall-prediction-accuracy.md](overall-prediction-accuracy.md)

## 图示与可视化

`docs/diagrams/` 下保留了 Mermaid 源文件、SVG 和部分 HTML 可交互图。常用入口包括：

- 前向链路交互图： [diagrams/forward_sequence.html](diagrams/forward_sequence.html)
- 模型结构图： [diagrams/model-architecture.svg](diagrams/model-architecture.svg)
- overall accuracy 热力图： [diagrams/overall-accuracy-best-heatmap.svg](diagrams/overall-accuracy-best-heatmap.svg)
- Transformer 试验对比图： [diagrams/transformer-variant-accuracy.svg](diagrams/transformer-variant-accuracy.svg)

## 推荐阅读顺序

如果你是第一次接触这个仓库，建议按下面顺序阅读：

1. 先看 [../README.md](../README.md) 了解代码入口、环境要求和训练/推理命令。
2. 再看 [hardware-model-dataset.md](hardware-model-dataset.md) 了解当前默认训练行为、张量来源和六类变体。
3. 接着看 [label-mechanisms.md](label-mechanisms.md) 和 [data-collection-feature-engineering.md](data-collection-feature-engineering.md)，把标签语义和输入特征对齐方式看清楚。
4. 最后看 [model-architecture.md](model-architecture.md) 与 [overall-prediction-accuracy.md](overall-prediction-accuracy.md)，对应模型结构和最终结果。

## 当前内置实验产物

仓库根目录目前还保留了一组可以直接用于推理的 Transformer 固定时间实验产物：

- 配置文件： [../configs/trans_best_time.json](../configs/trans_best_time.json)
- checkpoint： [../checkpoints/trans_best_time.pt](../checkpoints/trans_best_time.pt)

对应推理命令见 [../README.md](../README.md)。