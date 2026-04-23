# CLAUDE.md

> 关联规则分析学习代码库 - Association Rule Mining for Financial Products

本文档为 AI 助手提供项目指导信息。

## Karpathy 指导原则

行为准则，减少常见的 LLM 编码错误。与项目特定说明合并使用。

**权衡：** 这些准则偏向谨慎而非速度。对于琐碎任务，请自行判断。

### 1. 先思考后编码

**不要假设。不要隐藏困惑。揭示权衡。**

在实现之前：
- 明确陈述你的假设。如果不确定，请提问。
- 如果存在多种解释，请呈现它们——不要默默选择。
- 如果存在更简单的方法，请说出来。在适当时提出反对意见。
- 如果某些内容不清晰，请停止。指出令人困惑的地方。询问。

### 2. 简洁优先

**解决问题所需的最小代码。无投机性代码。**

- 不要超出请求的功能。
- 不要为一次性使用的代码创建抽象。
- 不要添加未请求的"灵活性"或"可配置性"。
- 不要为不可能的场景添加错误处理。
- 如果写了200行代码，但50行就能解决，请重写。

自问："高级工程师会说这过于复杂吗？"如果是，请简化。

### 3. 精准修改

**只修改必须修改的部分。只清理自己的混乱。**

编辑现有代码时：
- 不要"改进"相邻的代码、注释或格式。
- 不要重构未损坏的代码。
- 匹配现有风格，即使你会以不同方式编写。
- 如果发现无关的死代码，请提及——不要删除。

当你的修改产生孤儿代码时：
- 移除你的修改导致未使用的导入/变量/函数。
- 除非被要求，否则不要移除预先存在的死代码。

测试：每一行更改都应直接追溯到用户的请求。

### 4. 目标驱动执行

**定义成功标准。循环直到验证。**

将任务转化为可验证的目标：
- "添加验证" → "为无效输入编写测试，然后使它们通过"
- "修复错误" → "编写重现错误的测试，然后使它通过"
- "重构X" → "确保测试在重构前后都通过"

对于多步骤任务，陈述简要计划：
```
1. [步骤] → 验证：[检查]
2. [步骤] → 验证：[检查]
3. [步骤] → 验证：[检查]
```

强有力的成功标准允许你独立循环。弱标准（"使其工作"）需要持续澄清。

---

**这些准则生效时：** 差异中的不必要更改更少，因过度复杂化而进行的重写更少，澄清问题在实现之前而非错误之后提出。

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
├── regression_analysis/        # 回归分析项目
│   ├── data/                   # 数据集
│   ├── docs/                   # 文档
│   ├── notebooks/              # Jupyter 笔记本
│   └── scripts/                # Python 脚本
├── cluster_analysis/           # 聚类分析项目
│   ├── docs/                   # 文档（聚类算法指南）
│   └── （其他目录待创建）
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

### 聚类分析项目

聚类分析项目包含常用聚类算法的通俗解释和实现，文档在 `cluster_analysis/docs/` 目录。

```bash
# 查看聚类算法通俗指南
cat cluster_analysis/docs/聚类算法通俗指南.md

# 或在浏览器中查看（如果支持）
# code cluster_analysis/docs/聚类算法通俗指南.md
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
