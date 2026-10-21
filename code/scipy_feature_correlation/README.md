# SciPy 特征关联分析

入口：[scipy_feature_correlation.ipynb](../../scipy_feature_correlation.ipynb)。在项目根目录使用指定 Python 环境顺序执行即可，数据文件由用户本地准备。

- `analysis.py`：语义类型识别、固定分组、计数表、V/MI、数值方向、分层条件信息、组合与特征间矩阵。
- `figures.py`：排名、方向、分箱敏感性、分层、组合、层次结构及全特征条件满意率；复用 `plot.airline_eda` 的样式和分面图，仅保存 PDF。
- `check_analysis.py`：可运行的公式/边界验证，覆盖独立、完美、U 形、并列值、缺失、Wilson 区间及 XOR 条件信息。

## 统计约定

全训练集 21 个业务特征进入排名，ID 单独等宽分箱诊断，label 不参与自身排名。连续变量默认 20 分位数箱，重复切点合并，延误零值独立；缺失是显式组。10/20/40 分箱敏感性另外列出。所有分层共用全训练集的特征分组。数值方向指标仅按当前特征非缺失行计算，并报告有效 n；文字类别不强行赋数值。

V 使用未经 Yates 连续性校正的 Pearson 卡方。MI 通过 `scipy.stats.entropy(base=2)` 从计数表计算，标签熵降低比例为 `I(Y;G)/H(Y)`。这两项可识别非单调关联，但受样本内偏差及分组方式影响。V² 等于二元标签的 grouped eta²；不能将 V、相关系数、熵降低比例或 AUC 的大小直接作同尺度比较。

原始值 AUC 来自 `mannwhitneyu(positive_values, negative_values)` 的 U/(n1*n0)，并列值按半分计。`Somers D = 2*AUC-1`；SciPy 对应 `somersd(label, score)`，并非反向参数顺序。低基数特征用 SciPy 直接核对；高基数特征由 U 等价计算，避免巨大列联表。

21 个业务特征的卡方检验组成一个族，使用 BY 调整，其他分析为探索性描述，不额外报告显著性。p=0 表示数值下溢，不能用于排序。记录独立性在合成数据中未得到证明，推断结果谨慎解释；不凭 p 值删列。报告最小期望频数及小于 5 的期望格比例。

分层条件互信息是每个层的 MI 按层概率加权，分别控制 Class 或 Type of Travel 一个变量；不是控制全部其他变量。组合增量通过互信息链式法则计算，不声称联合 MI 增大证明交互。特征层次树使用 average linkage 和 1−V 差异度，不使用 Ward，不声称存在真实航班层级。

Wilson 区间是逐点 95% 区间，假设独立 Bernoulli 观测；不是多重比较同时区间。所有组进入 V/MI，`supported_rate_gap` 单独限制到 n≥max(100, ceil(0.001N))。不删除弱信号、极端值或评分 0。

## 产物与验证

`image/scipy_feature_correlation/` 内生成 12 份 PDF。`source_data/` 保存全特征汇总、全部分组计数/率/区间、分层与组合结果、分箱敏感性、特征矩阵/树及运行版本参数；整个目录被 gitignore 排除。Notebook 保留执行输出和内嵌预览。

```bash
MPLCONFIGDIR=/tmp/airline-matplotlib /Users/zhaobo/anaconda3/envs/torch/bin/python code/scipy_feature_correlation/check_analysis.py
```

Notebook 末尾还检查实际数据列覆盖、样本/正类数守恒、数值有效 n、分层期望频数和仅 PDF 输出。没有训练模型、抽样、目标编码或使用 test 标签。模型编码、缺失处理及特征选择应在训练折内拟合，增量价值需要独立验证/交叉验证 AUC。

官方来源和指标公式见 notebook 的 Block 2 及末尾资料链接。
