# opencode

个人学习代码仓库：面向金融产品场景的数据分析与关联规则挖掘（Market Basket / Association Rule Mining），并包含行为金融与投资相关的笔记/幻灯片。

## 目录结构

```
opencode/
├── association_rules/              # 关联规则分析项目
│   ├── data/                       # 数据文件（示例 CSV）
│   ├── docs/                       # 文档（指南/报告）
│   ├── images/                     # 输出图表（脚本生成）
│   ├── notebooks/                  # Jupyter 笔记本
│   └── scripts/                    # Python 脚本（数据生成/分析/可视化）
├── behavioral_finance/             # 行为金融
│   ├── docs/                       # 读书笔记
│   └── slides/                     # Marp 幻灯片（md/html）
├── investment/                     # 投资相关笔记/材料
│   └── （md/html/pdf 等文件）
├── regression_analysis/            # 回归分析项目
│   ├── data/                       # 数据文件
│   ├── docs/                       # 文档（回归分析指南）
│   ├── notebooks/                  # Jupyter 笔记本
│   └── scripts/                    # Python 脚本
├── cluster_analysis/               # 聚类分析项目
│   ├── docs/                       # 文档（聚类算法指南）
│   └── （其他目录待创建）
├── requirements.txt                # Python 依赖（用于 association_rules）
└── CLAUDE.md                       # 项目运行入口索引（面向 AI 助手）
```

## 快速开始

建议使用独立环境（conda/venv 均可）。

```bash
# 进入项目目录
cd opencode

# 安装依赖
pip install -r requirements.txt
```

### 关联规则项目（association_rules）

```bash
# 1) 生成模拟客户-产品交易数据
python association_rules/scripts/01_data/generate_customer_data.py

# 2) 金融产品关联分析（Apriori/规则生成）
python association_rules/scripts/02_analysis/finance_analysis.py

# 3) FP-Growth 分析示例
python association_rules/scripts/02_analysis/fpgrowth_analysis.py

# 4) 生成可视化图表（输出到 association_rules/images）
python association_rules/scripts/03_visualization/visualize_rules.py
```

### Notebook

```bash
jupyter notebook association_rules/notebooks/algorithm_comparison.ipynb
```

## 主要依赖

见 [requirements.txt](./requirements.txt)，主要包含：`pandas`、`numpy`、`mlxtend`、`scikit-learn`、`matplotlib`、`networkx`、`jupyter` 等。

## 聚类分析项目

聚类分析项目包含常用聚类算法的通俗解释和实现，包括：

### 文档
- **聚类算法通俗指南**：[cluster_analysis/docs/聚类算法通俗指南.md](./cluster_analysis/docs/聚类算法通俗指南.md)
  - DBSCAN密度聚类算法的通俗理解
  - 层次聚类算法的通俗理解
  - 算法对比和选择指南

### 特点
- **生活化比喻**：用城市人群分布、家族树、班级合并等例子解释算法
- **实际案例**：客户细分、社交圈层等实际应用场景
- **参数选择指南**：提供不同数据特征的参数选择建议
- **代码示例**：Python实现示例

### 适用场景
- **DBSCAN**：不规则形状的簇、有噪声的数据、不知道簇数量的情况
- **层次聚类**：需要层次结构、需要可视化、小数据集

## 说明

- 本仓库偏学习与复现实验，脚本输出默认在各子目录（如 `association_rules/images`）。
- 更完整的脚本入口清单见 [CLAUDE.md](./CLAUDE.md)。
