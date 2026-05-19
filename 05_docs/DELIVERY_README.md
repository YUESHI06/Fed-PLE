# 代码与实验数据交付说明

## 1. 交付范围

本次交付内容基于当前 `CEGT_Fed` 项目的稳定版本整理，包含：

- 完整代码
- 已完成实验产生的原始结果
- 面向二次统计分析与绘图的数据导出文件
- 对应的结果图
- 技术说明文档

当前交付中的实验结果均来自已执行完成的主实验流程，结果文件总数为 `216`。

## 2. 项目技术目标

该项目面向联邦学习场景下的智能合约漏洞检测任务，核心目标是：

- 以图结构形式建模智能合约
- 使用 `CEGT` 提取图级语义表示
- 在联邦学习框架下进行多客户端训练
- 在存在假阴标签噪声的情况下完成鲁棒训练
- 对不同噪声机制和不同方法进行系统对比

## 3. 代码结构说明

### 3.1 顶层脚本

- `run_experiments.py`  
  主实验入口。负责统一调度 `Motivation`、`Baseline`、均匀假阴噪声、非对称噪声和 `Ablation` 消融实验，并将结果写入 `results/`。

- `export_paper_outputs.py`  
  结果导出脚本。负责将 `results/` 中的原始 JSON 结果汇总为 CSV、Markdown 和 PNG 图。

- `collect_results.py`  
  控制台结果汇总脚本。用于按方法、漏洞类型、噪声设置快速打印结果表。

- `run_all_experiments.ps1`  
  Windows 一键运行脚本。

- `run_all_experiments.sh`  
  Linux / shell 一键运行脚本。

- `non_Fed_Train.py`  
  非联邦训练相关逻辑与动机实验辅助流程。

- `Ablation.py`  
  消融实验相关定义。

- `Fed_RESCUER.py` / `Fed_FedAvg.py` / `Fed_CL.py` / `Fed_CLC.py` / `Fed_FedCorr.py` / `Fed_ARFL.py`  
  各联邦方法对应的实验实现入口。

- `options.py`  
  参数定义。

- `utils.py`  
  通用辅助函数。

### 3.2 模型目录 `models`

- `cegt.py`  
  `CEGT` 主模型定义，包含图级表示学习主结构。

- `layers.py`  
  图卷积、特征变换与基础层定义。

- `lcn.py`  
  `LCN` 标签修正网络定义，用于噪声标签建模。

### 3.3 数据处理目录 `data_processing`

- `dappscan_processor.py`  
  `DAppSCAN` 数据处理逻辑。

- `smartbugs_processor.py`  
  `SmartBugs` 数据处理脚本。

- `dataloader_manager.py`  
  客户端数据读取、划分、噪声注入、测试集组织与 dataloader 生成。

- `graph_dataset.py`  
  图数据集定义及 batch 拼接逻辑。

### 3.4 训练目录 `trainers`

- `clients.py`  
  本地客户端训练、标签处理、局部更新等逻辑。

- `server.py`  
  联邦服务端聚合逻辑。

- `evaluation.py`  
  评估指标计算、结果打印与结果保存。

## 4. 实验输出目录说明

### 4.1 原始结果目录 `results`

该目录存放主实验的原始输出结果，每个结果文件均为一个 JSON 文件。

目录层级结构：

```text
results/
  0.0/
  0.1/
  0.2/
  0.3/
  diff/
```

含义如下：

- `0.0 / 0.1 / 0.2 / 0.3`  
  表示均匀假阴噪声比例。

- `diff/`  
  表示非对称噪声实验结果。

每个子目录下再按方法分目录，例如：

- `RESCUER`
- `FedAvg`
- `CL`
- `CLC`
- `FedCorr`
- `ARFL`
- `Ablation_Full`
- `Ablation_woLCN`
- `Motivation_Exp1`

每个 JSON 文件对应一个具体实验条件，例如：

- `pure_reentrancy_result.json`
- `fn_integer_overflow_result.json`
- `diff_noise_time_dependency_result.json`

文件名中字段含义：

- `pure`：无噪声
- `fn`：均匀假阴噪声
- `diff_noise`：非对称噪声
- `reentrancy / integer_overflow / time_dependency / dos_failed_call`：漏洞任务

### 4.2 导出结果目录 `paper_outputs`

该目录存放从原始结果中进一步导出的 CSV、图和总结性文档。

## 5. 数据文件详细说明

### 5.1 长表数据

- `all_results_long.csv`  
  所有实验结果的长表格式汇总文件。  
  每一行对应一个实验条件，字段包括：
  - `method`
  - `vulnerability`
  - `noise_type`
  - `noise_rate`
  - `noise_bucket`
  - `F1 score`
  - `Precision`
  - `Recall`
  - `Accuracy`
  - `TP`
  - `TN`
  - `FP`
  - `FN`

该文件适用于：

- 任意自定义分组统计
- 二次绘图
- 重新生成论文图表
- 计算均值、标准差、排名和趋势

### 5.2 Baseline 对比结果

- `baseline_f1_score.csv`  
  各 baseline 方法在四类漏洞、四种均匀噪声设置下的 F1 表。

- `baseline_accuracy.csv`  
  与上表结构相同，但数值为 Accuracy。

- `baseline_precision.csv`  
  与上表结构相同，但数值为 Precision。

- `baseline_recall.csv`  
  与上表结构相同，但数值为 Recall。

这些文件的列结构统一为：

- 行：方法
- 列：`漏洞类型 + 噪声设置`

### 5.3 非对称噪声对比结果

- `baseline_asymmetric_f1_score.csv`
- `baseline_asymmetric_accuracy.csv`
- `baseline_asymmetric_precision.csv`
- `baseline_asymmetric_recall.csv`

用于表示 baseline 方法在非对称噪声条件下的指标结果。

### 5.4 Ablation 消融结果

- `ablation_f1_score.csv`
- `ablation_accuracy.csv`
- `ablation_precision.csv`
- `ablation_recall.csv`

用于表示完整模型及各消融变体在不同漏洞、不同均匀噪声条件下的指标结果。

主要变体包括：

- `Ablation_Full`
- `Ablation_woLCN`
- `Ablation_woWarmup`
- `Ablation_woOrtho`
- `Ablation_woTransformer`

### 5.5 Motivation 结果

- `motivation_f1_score.csv`
- `motivation_accuracy.csv`
- `motivation_precision.csv`
- `motivation_recall.csv`

用于表示四组 `Motivation` 实验在四类漏洞任务上的结果。

### 5.6 数据集统计

- `dataset_statistics.csv`  
  记录各漏洞任务对应的数据规模信息。

### 5.7 面向画图的聚合数据

以下文件属于已经聚合好的二次绘图数据：

- `baseline_average_f1_by_noise.csv`  
  各 baseline 方法在 `0.0 / 0.1 / 0.2 / 0.3` 下的平均 F1。

- `baseline_robustness_summary.csv`  
  各 baseline 方法从 `0.0` 噪声到 `0.3` 噪声的整体变化情况。

- `ablation_average_f1_by_noise.csv`  
  完整模型与各消融变体在不同噪声下的平均 F1。

- `ablation_delta_vs_full.csv`  
  各消融变体相对 `Ablation_Full` 的平均差值。

- `ablation_delta_by_vulnerability.csv`  
  各漏洞任务维度上的消融对比结果。

- `best_baseline_by_task_noise.csv`  
  每个漏洞任务、每种噪声设置下表现最好的 baseline 方法。

## 6. 图文件说明

### 6.1 对比实验图

- `baseline_average_f1_by_noise.png`  
  baseline 方法整体平均 F1 随均匀假阴噪声变化的折线图。

- `baseline_grouped_by_noise.png`  
  不同噪声水平下 baseline 方法平均 F1 的分组柱状图。

- `baseline_robustness_drop.png`  
  baseline 方法从 `0.0` 到 `0.3` 噪声条件下性能变化图。

### 6.2 消融实验图

- `ablation_average_f1.png`  
  完整模型与各消融变体的平均 F1 对比图。

- `ablation_full_gain.png`  
  完整模型相对于各消融变体的性能差值图。

### 6.3 Motivation 图

- `motivation_average_f1.png`  
  四组 `Motivation` 实验的平均 F1 图。

### 6.4 其他统计图

- `best_method_counts.png`  
  统计不同 baseline 方法在各任务与噪声设置下成为最优方法的次数。

## 7. 运行链路说明

### 7.1 主实验运行

执行：

```powershell
python run_experiments.py
```

运行结果会写入：

```text
results/
```

### 7.2 导出 CSV 与图

执行：

```powershell
python export_paper_outputs.py
```

导出结果会写入：

```text
paper_outputs/
```

### 7.3 控制台表格查看

执行：

```powershell
python collect_results.py --result_dir ./results
```

## 8. 交付包目录说明

整理后的交付目录为：

```text
delivery_package/
  01_code/
  02_results_raw/
  03_plot_data/
  04_best_figures/
  05_docs/
```

各目录含义如下：

- `01_code`  
  存放运行主实验和导出结果所需的核心代码。

- `02_results_raw`  
  存放原始实验结果 JSON。

- `03_plot_data`  
  存放适合做二次统计和绘图的 CSV 文件。

- `04_best_figures`  
  存放已导出的重点图。

- `05_docs`  
  存放交付说明和实验总结文档。

## 9. 说明

交付包中的代码目录不包含人为拼接的“效果数值文件”，实验结果来自：

1. `run_experiments.py` 实际运行
2. `results/` 原始 JSON 写出
3. `export_paper_outputs.py` 从原始 JSON 自动汇总生成 CSV、图和文档

因此，图表与 CSV 均可以从原始结果重新导出。
