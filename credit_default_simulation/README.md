# 信贷违约模拟案例

## 目的

基于正则化、降维、定性响应变量评估指标，模拟一个信贷违约预测案例，演示以下知识点：

1. **正则化回归**：Ridge、Lasso、ElasticNet 在高维特征下的应用
2. **降维方法**：主成分回归（PCR）与偏最小二乘（PLS）
3. **分类评估指标**：混淆矩阵、准确率、精确率、召回率、F1、AUC
4. **交叉验证**：用于超参数调优与模型选择

## 模拟数据特征

模拟 1000 个客户的信贷数据，包含以下特征：

| 特征 | 描述 | 数据类型 |
|------|------|----------|
| `age` | 年龄 | 连续 |
| `income` | 年收入（万元） | 连续 |
| `debt_ratio` | 负债率（负债/收入） | 连续 |
| `credit_score` | 信用评分（300-850） | 连续 |
| `months_employed` | 在职月数 | 整数 |
| `num_credit_lines` | 信贷账户数 | 整数 |
| `num_late_payments` | 逾期次数 | 整数 |
| `loan_amount` | 贷款金额（万元） | 连续 |
| `savings_balance` | 储蓄余额（万元） | 连续 |
| `employment_type` | 就业类型（全职/兼职/自雇） | 分类 |

**目标变量**：`default`（是否违约，0/1）

## 数据生成逻辑

违约概率基于以下逻辑生成：
- 信用评分越低，违约概率越高
- 负债率越高，违约概率越高
- 收入越低，违约概率越高
- 逾期次数越多，违约概率越高
- 加入随机噪声

## 文件说明

- `credit_data.csv`：模拟的信贷数据（1000个样本，10个特征）
- `credit_default_analysis.ipynb`：Jupyter Notebook 分析脚本（主要分析文件）
- `credit_default_analysis.py`：Python 版本的分析脚本
- `generate_data_v2.py`：数据生成脚本（简化版，违约率24.2%）
- `generate_data.py`：数据生成脚本（原始版）
- `README.md`：本文件

## 分析流程

1. **数据预处理**：缺失值处理、标准化、分类变量编码
2. **探索性分析**：特征分布、相关性、违约率
3. **正则化回归**：Ridge、Lasso、ElasticNet 模型训练与对比
4. **降维方法**：PCR、PLS 与正则化方法的对比
5. **评估指标**：混淆矩阵、准确率、精确率、召回率、F1、AUC
6. **交叉验证**：超参数调优、模型选择
7. **结论与建议**：模型对比、业务启示

## 依赖库

```
pandas
numpy
scikit-learn
matplotlib
seaborn
mlxtend
jupyter
```

## 使用方法

1. 运行 `credit_default_analysis.ipynb` 中的数据生成单元格，生成 `credit_data.csv`
2. 按照 Notebook 中的步骤进行分析
3. 查看结果与可视化