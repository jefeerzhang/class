# CLAUDE.md

> 关联规则分析学习代码库 - Association Rule Mining for Financial Products

本文档为 AI 助手提供项目指导信息。

## Overview

This is a personal learning codebase focused on **association rule mining** (关联规则) algorithms for financial product analysis. The work explores market basket analysis techniques applied to banking/finance scenarios (products like 信用卡, 储蓄账户, 基金, 贷款, 理财产品, 保险).

## Project Structure

```
opencode/
├── association_rules/          # 关联规则分析项目
│   ├── scripts/
│   │   ├── 01_data/            # 数据生成
│   │   ├── 02_analysis/        # 关联分析
│   │   ├── 03_visualization/   # 可视化
│   │   └── 04_utils/           # 工具脚本
│   ├── notebooks/              # Jupyter 笔记本
│   ├── data/                   # 数据集
│   ├── docs/                   # 文档
│   └── images/                 # 生成的图表
├── behavioral_finance/         # 行为公司金融项目
│   ├── docs/                   # 文档
│   └── slides/                 # Marp 幻灯片
├── CLAUDE.md                   # 项目说明
├── README.md                   # 项目介绍
└── requirements.txt            # Python 依赖
```

## Running the Scripts

### 关联规则项目

所有脚本在 `association_rules/scripts/` 目录，输出为中文。

```bash
# 安装依赖
pip install mlxtend pandas numpy scikit-learn matplotlib networkx jupyter ipywidgets moviepy pillow

# 生成客户数据
python association_rules/scripts/01_data/generate_customer_data.py

# 金融产品关联分析
python association_rules/scripts/02_analysis/finance_analysis.py

# Apriori 测试
python association_rules/scripts/02_analysis/apriori_test.py

# 金融示例
python association_rules/scripts/02_analysis/finance_example.py

# FP-Growth 性能对比
python association_rules/scripts/02_analysis/fpgrowth_analysis.py

# 生成可视化图表
python association_rules/scripts/03_visualization/visualize_rules.py

# Jupyter Notebook
jupyter notebook association_rules/notebooks/algorithm_comparison.ipynb

# 生成视频演示
python association_rules/scripts/04_utils/generate_video.py
```

### 行为公司金融项目

```bash
# 转换 Marp 幻灯片为 HTML
marp behavioral_finance/slides/prospect_theory_slides.md --output behavioral_finance/slides/prospect_theory_slides.html

# 转换为 PDF
marp behavioral_finance/slides/prospect_theory_slides.md --pdf
```

## Key Libraries

- `mlxtend.frequent_patterns` - Apriori and FP-Growth algorithms
- `mlxtend.preprocessing.TransactionEncoder` - Convert transactions to boolean matrix
- `pandas` - Data manipulation
- `marp-cli` - Markdown to slide conversion

## Core Algorithm Patterns

```python
# 1. Encode transactions
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

# 2. Mine frequent itemsets (FP-Growth preferred over Apriori for performance)
frequent_itemsets = fpgrowth(df_encoded, min_support=0.3, use_colnames=True)

# 3. Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 4. Filter quality rules
quality_rules = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.7)]
```

## Key Metrics

- **Support** (支持度): Frequency of itemset occurrence - filter threshold ≥0.3
- **Confidence** (置信度): Rule reliability/predictive power - filter threshold ≥0.7
- **Lift** (提升度): Correlation strength - filter threshold >1.2 (positive correlation)

## Tips

- 默认参数：`min_support=0.3`, `min_confidence=0.7`, `min_lift=1.2`
- 优先使用 FP-Growth 算法（性能优于 Apriori）
- 所有输出为中文

## Marp 幻灯片规范

创建 Marp 幻灯片时，**必须**在文件开头添加以下配置：

```yaml
---
marp: true
theme: gaia
class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.jpg')
style: |
  section {
    font-family: 'Times New Roman', 'SimSun';
  }
  h1 {
    color: #2c3e50;
    font-size: 1.5em;
  }
  h2 {
    color: #34495e;
    border-bottom: 2px solid #3498db;
  }
  footer {
    font-size: 0.5em;
    color: #7f8c8d;
  }
---
```

**转换命令：**
```bash
# 转 HTML
marp path/to/slides.md --output path/to/slides.html

# 转 PDF
marp path/to/slides.md --pdf
```
