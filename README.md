# Predicting Airline Satisfaction

[Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10) 的航空旅客满意度预测项目。目标是预测 `satisfaction` 的正类概率，评估指标为 ROC-AUC。

当前已完成数据读取与基础探索性分析（EDA），尚未训练模型或生成提交预测。

## 项目结构

```text
read_airline_data.ipynb    数据读取、分节图表与中文分析
scipy_feature_correlation.ipynb  全特征关联、分层/组合、冗余及建模建议
code/scipy_feature_correlation/ SciPy 统计计算、PDF 绘图与公式检查
code/plot/airline_eda.py   数据检查、统计汇总与绘图
code/plot/check_eda.py     汇总边界与数据完整性检查
code/plot/README.md        统计方法、图表说明与 QA 记录
image/                    9 份 PDF 图表
```

EDA 覆盖全部 21 个业务特征：连续变量、服务评分、文字类别，以及单独的 ID 排序诊断。图表展示标签分布、缺失情况、各组满意率及 95% Wilson 区间、单变量关联与 Spearman 相关性。所有图片文件仅导出 PDF。

## 数据准备

从比赛页面下载并解压数据，将以下文件放在项目根目录：

```text
train.csv
test.csv
sample_submission.csv
```

原始数据、运行生成的 `image/source_data/` 统计表与运行信息，以及本地 `AGENTS.md` 均不提交到 GitHub。Notebook 保留探索分析的内嵌图表和汇总结果。

## 运行

需要 Python、NumPy、pandas、Matplotlib、SciPy（Spearman 相关计算）以及 Jupyter/IPython（运行 notebook）。

在项目根目录打开 `read_airline_data.ipynb`，选择对应 Python 环境并依次执行全部单元。也可直接运行 EDA：

```python
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path.cwd() / "code"))
from plot.airline_eda import run_eda

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
eda = run_eda(train, test, output="image")
```

PDF 写入 `image/`，可追溯的统计表和运行信息写入 `image/source_data/`。完成 EDA 后检查汇总边界、特征覆盖及样本数/正类数守恒：

```bash
python code/plot/check_eda.py
```

本机指定解释器为 `/Users/zhaobo/anaconda3/envs/torch/bin/python`；其他机器使用自己的 Python 环境即可。

## SciPy 特征关联分析

从根目录依次执行 `scipy_feature_correlation.ipynb`，即可计算全部 21 个业务特征与标签的关联，并单独检查 ID。Notebook 分 block 解释公式、适用范围与结果，计算逻辑保存在 `code/scipy_feature_correlation/`，不需要复制计算代码到 notebook。

分析包括 Cramér’s V、卡方及 BY 多重检验调整、经验互信息与标签熵降低比例、点二列/Spearman/Kendall 相关、原始值 AUC/Somers’ D，以及连续变量分箱敏感性。还提供按舱位/出行目的分层的条件互信息、四组特征组合和特征间的层次聚类。

图表仅导出至 `image/scipy_feature_correlation/*.pdf`；全量统计表与运行版本保存到被忽略的 `image/scipy_feature_correlation/source_data/`。执行公式与边界检查：

```bash
python code/scipy_feature_correlation/check_analysis.py
```

这些全训练集统计量帮助提出编码、非线性和交互假设；不能替代控制全部其他变量后的独立贡献或交叉验证 AUC，也不能用作验证集的目标编码。

## 分析说明

这些图表是全训练集上的描述性分析，不代表因果关系、模型重要性或验证集性能。后续建模应在训练折内完成编码与特征处理，并使用独立验证集或交叉验证评估 ROC-AUC。

比赛提交格式为 `id,satisfaction`，其中 `satisfaction` 为预测概率。详细方法与图表说明见 [code/plot/README.md](code/plot/README.md)。
