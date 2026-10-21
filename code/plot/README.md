# Airline satisfaction 基础 EDA

入口是项目根目录的 `read_airline_data.ipynb`。原有数据读取部分保留，新增 EDA 分节导入 `plot.airline_eda`，展示图与中文解读。主模块只依赖现有 numpy、pandas、matplotlib；不安装包，不训练模型。

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'code'))
from plot.airline_eda import run_eda, grouped_rates, plot_rate_pages

eda = run_eda(train, test, output='image')
# 单独重画某个特征，例如改变连续变量分箱数：
age = grouped_rates(train, 'Age', 'continuous', bins=30)
figs = plot_rate_pages({'Age': age}, ['Age'], train.satisfaction.mean(),
                       prefix='age_30_bins', output='image')
```

使用 `/Users/zhaobo/anaconda3/envs/torch/bin/python`。模块 `run_eda` 返回 profile、各特征 rates、associations、correlations 和 figures；不修改传入 DataFrame。每张图仅保存 PDF，不导出 PNG 或 JPG；notebook 内嵌预览保留。

## 图与源数据

| 输出文件前缀 | 目的 | 统计来源 |
|---|---|---|
| 01_target_and_missingness | 标签分布与 train/test 非零缺失率 | train.satisfaction；两份数据全部列 |
| 02_continuous_rates_01 | 4 个连续特征的条件满意率 | 每箱满意数/记录数 |
| 03_rating_rates_01–03 | 全部 13 个服务评分，含 0 | 每个原始等级 |
| 04_category_rates_01 | 全部 4 个文字类别特征 | 每个原始类别 |
| 05_feature_associations | 全部 21 个业务特征的边际关联 | grouped eta-squared |
| 06_spearman_correlations | 数值特征冗余与标签的秩相关 | 全训练集成对非缺失记录 |
| 07_id_diagnostic_01 | ID 区间是否存在标签顺序漂移 | ID 的 20 个等宽箱 |

`image/source_data/feature_rates.csv` 提供每个点的 feature、kind、group、n、positive、rate、x（数值组中位数）、x_min、x_max、low、high。类别组的 x 只是绘图位置。缺失组中位数为空，图上的 NA 是单独的排版位置，无数值解释。所有分箱包含边界和重复值，长尾无截断。连续箱的宽度不同，下方灰条表示计数而非概率密度。

其余文件：`data_profile.csv`（train 699,635 行，test 299,844 行，全部列）、`feature_associations.csv`、`spearman_correlations.csv`、`correlation_pair_counts.csv`、`id_rates.csv`、`runtime.json`（运行时版本与参数）。没有读取或更改 sample_submission 中的预测值。

## 统计与复现报告

- 观测单位：CSV 的一条记录，不声称是独立旅客。全部训练记录参与每个业务特征的满意率汇总，不抽样、不全表删除缺失行。到达延误有 292 条训练缺失记录，单独作为一组；test 同列 130 条缺失。
- 二分类：False/0 为不满意，True/1 为满意。全训练集满意率为 310,339/699,635 ≈ 44.36%。灰虚线为这个描述性参考值。
- 中心：各组二元标签均值（满意率）。区间：95% Wilson 二项比例区间，z=1.959963984540054，逐点计算，非同时区间、非 SD/SEM。它假设记录相互独立，合成数据依赖可能使其过窄。
- 分箱：Age、Flight Distance 默认 20 个分位数箱；延误零值独立成组，剩余数值做 20 分位数箱，重复切点合并。图中线连接中位数上的箱均值，不是拟合回归曲线。评分和类别按原始等级完整展开；不解释评分 0 的语义。延误轴用 symlog 压缩，标注 0–1 区间线性；这种尺度不能直接用来判断原始尺度的曲率。
- 关联：`eta² = Σ(n_g/N)(p_g-p)² / [p(1-p)]`，含缺失组。它衡量样本内分组间满意率变异；受分箱方式、组数、共线性、混杂影响，不是泛化性能、因果效应或多变量特征重要性。`supported_rate_gap` 的摘要只使用 n≥max(100, ceil(0.001N)) 的组；图及 eta² 不作此过滤。
- 相关：Spearman 基于成对非缺失记录。涉及到达延误时，仅该配对计算不含其 292 条缺失行（n=699,343）；其他配对 n=699,635。ID 为标识符，单独诊断；4 个文字类别不强行赋数值进入相关矩阵，但全部包含在类别图与关联排序中。
- 检验：未做假设检验，无 p 值或显著性星号，多重比较校正不适用。探索性曲线中的阈值不能作为已验证机制。
- ML：没有训练/验证/测试划分，没有模型拟合、随机种子或交叉验证；没有报告预测 AUC。后续建模的编码、特征选择和参数拟合应在训练折内完成，验证集评估 ROC-AUC。不要直接使用本 EDA 全训练集的分组标签均值作验证集目标编码。

## QA 记录

问题：哪些特征与二分类满意度有关，哪些呈非单调或阈值关系？采用等大小 quantitative grid 分面；按照 Nature-family 风格设置 183 mm 双栏、单特征 89 mm 单栏画布。PDF tight 裁边加留白后约 186/92 mm，并非期刊投稿文件。当前用途为基础 EDA。

资产核对：已读取技能 LineTrend/plot_trend.py、BarCategorical/plot_correctness_by_category.py、CorrelationMatrix/plot_CorrelationHeatmap_v1.R 及其预览。前两者分别是累积时间序列和多方法基准汇总，无法只替换路径得到本任务的条件比例图；第三者为 ggpairs，且本环境无 R。采用 Python 参数继承：等大小分面、0–1 比例范围、去除上右边框、无框图例原则；实际字体/颜色/导出使用技能的三个原样 baseline。没有声称运行了这些生产脚本或复现其完整视觉系统。

预先防范的排版问题：长列名换行；13 个评分分页；大样本散点改为全数据分组统计与计数条。单特征页用单栏，避免空半页。缺失点不与数值点连线；延误零点不与正值趋势线连接。

- Pass 0（AP-0…AP-7）：三个原样 baseline 存在；显式语义配色、无彩虹、无四面边框、无图内图例遮挡、仅导出 PDF。计数条展示记录数，不是小 n 实验均值，逐个观测叠加不适用。
- Pass 1（CL-1…CL-7）：实际最小显式文字为 6 pt；画布宽度 89/183 mm、高度≤219 mm；PDF 嵌入字体；边框 0.6 pt；向外刻度。最终 PDF 解析检查文本、字体、单页和矢量图元。
- Pass 2（VI-1…VI-7）：图与问题对应；统一蓝色代表满意率，灰色为计数/参考线；无红绿两色分类；概率轴统一 0–1，计数条从 0 开始；区间定义与 n 可追溯；分面标签位置一致；已核对 revision-cases 中热图、分面一致性与小 n 的常见问题。
- Pass 3（VV-1…VV-5，初版历史检查）：逐张查看当时生成的 9 张 PNG，另核对单栏和热图 PDF 渲染；无文字/数据明显遮挡、截断或重叠；坐标尺度有标注；弱关联保留原样。Gender 和 ID 曲线接近平坦是真实结果，不能为满足技能的“最低信号幅度”检查而删除或改写数据。

技能自带 `qa_validator.py` 的文件没有 CLI 入口，因此直接导入调用各 check 函数。它有两项误报：颜色检查只识别 `DIVERGING = [`，不识别原样 baseline 的多个空格；字号正则把 `capsize=2` 的误差线帽宽误认为 2 pt 字号。另外“无图外图例”“柱图无个体点”“未识别尺寸表达式”产生提示。已逐一人工核对：本代码没有图例，柱图为大样本计数，尺寸为 mm/25.4，最小文字 6 pt；不为迎合检查器改写原样 baseline。

完整 notebook 在指定 torch Python 中按顺序逐单元执行，含 9 个内嵌图像输出，无单元错误。当前沙箱限制 Jupyter localhost 端口绑定，因此验收使用同一解释器的 IPython 进程内执行与富输出捕获；不是一个已启动的外部 Jupyter kernel 会话。

导出格式更新：共享保存函数仅写入 PDF，已清理 `image/` 中初版生成的 9 张 PNG。

运行检查：

```bash
MPLCONFIGDIR=/tmp/airline-matplotlib /Users/zhaobo/anaconda3/envs/torch/bin/python code/plot/check_eda.py
```

检查覆盖缺失、零值、重复切点、常数特征、全 0/1 标签的汇总边界，以及实际全部 21 个业务特征的样本数、正类数守恒。对真实数据中弱信号不设人为最低幅度。

## 建模启示

优先关注 Online boarding、Class、Inflight wifi service、Inflight entertainment 和 Type of Travel。特别检查评分 0、评分 3→4 的跳变，年龄先升后降，以及 Gate location 的 U 形（线性相关约 0.023，不能据此认为无信号）。比较可表达非线性的模型与逻辑回归基线，交互和实际预测增益留到独立验证阶段确认。
