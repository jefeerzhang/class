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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a personal learning codebase focused on **association rule mining** (关联规则) algorithms for financial product analysis. The work explores market basket analysis techniques applied to banking/finance scenarios (products like 信用卡, 储蓄账户, 基金, 贷款, 理财产品, 保险).

## Running the Scripts

All scripts are in the `关联算法/` directory and use Chinese-language output.

```bash
# Install dependencies
pip install mlxtend pandas numpy scikit-learn matplotlib networkx jupyter ipywidgets moviepy pillow

# Generate synthetic customer data
python 关联算法/generate_customer_data.py

# Run financial product association analysis
python 关联算法/finance_analysis.py

# Run basic Apriori test
python 关联算法/apriori_test.py

# Run finance example
python 关联算法/finance_example.py

# FP-Growth algorithm with performance comparison
python 关联算法/fpgrowth_analysis.py

# Generate visualization charts
python 关联算法/visualize_rules.py

# Interactive Jupyter Notebook
jupyter notebook 关联算法/algorithm_comparison.ipynb

# Generate video presentation
python 关联算法/generate_video.py
```

## Architecture

The codebase uses a layered approach:

1. **Data Generation** (`generate_customer_data.py`) - Creates synthetic customer datasets with product holdings
2. **Core Analysis** (`finance_analysis.py`, `finance_example.py`, `apriori_test.py`) - Implements association rule mining using mlxtend
3. **FP-Growth Analysis** (`fpgrowth_analysis.py`) - FP-Growth algorithm with Apriori performance comparison
4. **Visualization** (`visualize_rules.py`) - Generates charts: penetration pie chart, itemsets bar chart, rules scatter plot, network graph, lift distribution
5. **Interactive Notebook** (`algorithm_comparison.ipynb`) - Jupyter Notebook with interactive widgets for parameter tuning
6. **Video Generation** (`generate_video.py`) - Creates video presentation with subtitles
7. **Documentation** (`.md` files) - Comprehensive guides and reports

## Key Libraries

- `mlxtend.frequent_patterns` - Apriori and FP-Growth algorithms
- `mlxtend.preprocessing.TransactionEncoder` - Convert transactions to boolean matrix
- `pandas` - Data manipulation

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
