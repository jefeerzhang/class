---
marp: true
theme: gaia
class: lead
paginate: true
backgroundColor: #fff
header: '数据报告标题'
style: |
  section {
    font-family: 'Times New Roman', 'SimSun', serif;
  }
  h1 {
    color: #2c3e50;
    font-size: 1.5em;
  }
  h2 {
    color: #34495e;
    border-bottom: 2px solid #3498db;
  }
  h3 {
    color: #34495e;
    font-size: 1.1em;
  }
  img {
    display: block !important;
    margin: 0 auto !important;
  }
  section table {
    display: table !important;
    width: auto !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
  table {
    font-size: 0.85em;
  }
  th {
    background-color: #ecf0f1;
  }
  tr:nth-child(even) {
    background-color: #f8f9fa;
  }
  blockquote {
    border-left: 4px solid #3498db;
    padding-left: 1rem;
    color: #2c3e50;
    background-color: #ecf0f1;
  }
  .highlight {
    color: #e74c3c;
    font-weight: bold;
  }
  footer {
    font-size: 0.5em;
    color: #7f8c8d;
  }
---

# 数据报告标题

**副标题：分析目标与范围**

报告人 | 日期

---

## 分析概览

**核心发现：**

| 指标 | 数值 | 变化 |
|------|------|------|
| 总样本量 | 10,000 | +15% |
| 关键指标A | 85.3% | +5.2% |
| 关键指标B | 2.4 | -0.8 |

**结论先行：** 本季度整体表现稳健，A 指标提升明显。

---

# Part 1: 数据来源与方法

---

## 数据说明

**数据集基本信息：**

- **时间范围：** 2024年1月 - 2024年12月
- **样本量：** 10,000 条记录
- **特征维度：** 20 个变量
- **数据质量：** 缺失率 < 3%

**处理方法：**

1. 缺失值：中位数填充
2. 异常值：IQR 方法剔除
3. 标准化：Z-score 标准化

---

## 分析框架

$$\text{目标变量} = f(\text{特征}_1, \text{特征}_2, ..., \text{特征}_p) + \epsilon$$

**采用方法：**

- 描述性统计
- 相关性分析
- 回归建模
- 交叉验证

---

# Part 2: 核心发现

---

## 发现一：趋势变化

**关键洞察：**

- 指标A 呈上升趋势，Q3 达到峰值
- 指标B 波动较大，季节性明显
- 两者存在负相关关系（$r = -0.62$）

> 建议：关注 Q3 峰值背后的驱动因素。

---

## 发现二：分组对比

| 组别 | 样本量 | 均值 | 标准差 | 显著性 |
|------|--------|------|--------|--------|
| 组A | 5,200 | 85.3 | 12.1 | *** |
| 组B | 3,100 | 78.6 | 15.4 | *** |
| 组C | 1,700 | 91.2 | 8.7 | *** |

**结论：** 组C 表现最优，组B 仍有提升空间。

---

## 发现三：回归结果

**模型性能：**

```
R² = 0.847,  Adjusted R² = 0.842
RMSE = 2.34,  MAE = 1.87
```

**显著特征（Top 5）：**

| 特征 | 系数 | 标准误 | t值 | p值 |
|------|------|--------|-----|-----|
| 特征X | 3.24 | 0.31 | 10.45 | <0.001 |
| 特征Y | -1.87 | 0.28 | -6.68 | <0.001 |
| 特征Z | 1.56 | 0.25 | 6.24 | <0.001 |

---

# Part 3: 结论与建议

---

## 主要结论

1. **趋势向好** - 核心指标持续改善
2. **差异显著** - 不同组别表现分化
3. **模型可靠** - 预测精度达到实用标准

---

## 行动建议

| 优先级 | 建议 | 预期效果 | 负责人 |
|--------|------|----------|--------|
| **高** | 优化组B 的运营策略 | +8% 提升 | 运营团队 |
| **中** | 深入分析特征X 的影响机制 | 优化模型 | 数据团队 |
| **低** | 扩大组C 的成功经验 | 长期增长 | 管理层 |

---

## 局限性

- 数据时间跨度有限（1年）
- 部分特征存在多重共线性
- 因果推断需要进一步验证

> **下一步：** 收集更长时间序列数据，建立面板模型。

---

# 附录

---

## 技术细节

**模型选择过程：**

```python
# 比较多个模型
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.5)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 完整结果表

（详细结果可放在此处或单独附件）

| 指标 | 训练集 | 测试集 | 交叉验证 |
|------|--------|--------|----------|
| R² | 0.852 | 0.847 | 0.844 |
| RMSE | 2.28 | 2.34 | 2.38 |
| MAE | 1.82 | 1.87 | 1.91 |
