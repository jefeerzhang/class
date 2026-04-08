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

# 关联规则挖掘完整指南

从基础到时序，四类关联规则算法的理论与实践

---

## 目录

1. 概述：什么是关联规则
2. 基本关联规则 (Apriori)
3. 进阶算法 (FP-Growth)
4. 多维关联规则
5. 时序关联规则
6. 算法对比与选择
7. 业务应用场景

---

## 1. 概述：什么是关联规则

### 核心思想

关联规则挖掘发现数据中item之间的共现关系

**典型问题**：顾客买了A商品，还会买什么？

---

## 三个核心指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **支持度** | P(A ∩ B) | A和B同时出现的概率 |
| **置信度** | P(B\|A) | 买了A的人，买B的概率 |
| **提升度** | P(B\|A) / P(B) | 关联强度，排除随机性 |

---

## 指标解读

```
支持度 30%：100个顾客中，30人同时买了A和B
置信度 70%：买了A的顾客中，70%也买了B
提升度 1.5：买了A的顾客买B的可能性是平均水平的1.5倍
```

---

## 阈值建议

| 场景 | 最小支持度 | 最小置信度 | 最小提升度 |
|------|-----------|-----------|-----------|
| 探索性分析 | 5% | 50% | 1.0 |
| 常规分析 | 10% | 60% | 1.1 |
| 高价值规则 | 25% | 70% | 1.2 |

---

## 2. 基本关联规则 (Apriori)

### 目标

发现交易数据中item同时出现的高频模式

### 作用

- 识别顾客购买行为中的共现规律
- 发现产品之间的正向关联
- 为交叉销售、货架摆放提供数据支持

---

## Apriori 核心原理

**Apriori Principle**：如果一个项集是频繁的，那么它的所有子集也必须是频繁的

**需要多次扫描数据库，产生大量候选项集**

---

## Apriori 算法流程

```
1. 扫描数据库，计算每个1-项集的支持度
2. 剪枝：移除低于阈值的项集
3. 使用频繁1-项集生成候选2-项集
4. 再次扫描数据库，验证候选项集
5. 重复直到没有新的频繁项集
```

---

## Apriori 代码示例

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = [
    ["面包", "牛奶", "尿布"],
    ["面包", "牛奶", "啤酒", "尿布"],
    ["牛奶", "尿布", "鸡蛋"],
]

te = TransactionEncoder()
df_encoded = pd.DataFrame(
    te.fit_transform(transactions),
    columns=te.columns_
)

frequent_itemsets = apriori(df_encoded, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
```

---

## 3. 进阶算法：FP-Growth

### 目标

解决Apriori算法多次扫描数据库的问题，更高效地挖掘频繁项集

---

## FP-Growth 核心优势

- **只需扫描数据库2次**（而非N次）
- **不生成候选项集**，直接挖掘
- 比Apriori快 **3-10倍**
- 大幅提升挖掘效率，尤其在大规模数据集上

---

## FP-Tree 结构

**FP-Tree (Frequent Pattern Tree)**：一种压缩表示事务数据库的树结构

**算法流程**：

```
1. 第一次扫描：统计每个item频率，移除低频item
2. 第二次扫描：构建FP-Tree，按频率排序插入事务
3. 递归挖掘：从FP-Tree条件模式基递归挖掘频繁项集
```

---

## FP-Growth vs Apriori 性能对比

```python
# FP-Growth 推荐用于大规模数据
start_fpg = time.time()
fpg_itemsets = fpgrowth(df_encoded, min_support=0.25, use_colnames=True)
time_fpg = time.time() - start_fpg

start_apr = time.time()
apr_itemsets = apriori(df_encoded, min_support=0.25, use_colnames=True)
time_apr = time.time() - start_apr

speedup = time_apr / time_fpg
print(f"FP-Growth 比 Apriori 快: {speedup:.2f} 倍")
```

---

## 4. 多维关联规则

### 目标

在关联规则中引入多个维度（用户属性、时间、场景），发现更精细的关联模式

---

## 单维 vs 多维规则

| 类型 | 示例 |
|------|------|
| **单维规则** | 储蓄账户 → 信用卡 (只看产品维度) |
| **多维规则** | 年龄=30-40 ∧ 收入=高 → 持有理财产品 |

---

## 维度类型

| 维度 | 示例 | 类型 |
|------|------|------|
| 人口属性 | 年龄、性别、地区 | 类别型 |
| 消费能力 | 收入等级、信用评分 | 数值型/类别型 |
| 行为特征 | 购买频率、渠道偏好 | 数值型 |
| 时间特征 | 季度、节假日 | 时间型 |

---

## 多维关联代码

```python
# 将年龄和收入等级编码为产品维度
df["年龄_青年"] = df["年龄"].apply(lambda x: 1 if x < 35 else 0)
df["年龄_中年"] = df["年龄"].apply(lambda x: 1 if 35 <= x < 50 else 0)
df["收入_高"] = df["收入等级"].apply(lambda x: 1 if x == "高" else 0)

# 构建多维交易列表
def build_multi_dimensional_transaction(row):
    items = row["持有产品"].split("|")
    if row["年龄"] < 35:
        items.append("青年客户")
    ...
    return items

frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)
```

---

## 5. 时序关联规则

### 目标

发现事件发生的**先后顺序**规律，不仅问"哪些产品一起出现"，更问"**先买X后买Y**"的时序模式

---

## 核心概念

**时序模式 (Sequential Pattern)**：`[A] → [B]` 表示买了A之后倾向于买B

**关键指标**：

| 指标 | 含义 |
|------|------|
| 支持度 | 序列在所有客户中出现的频率 |
| 置信度 | 给定前缀序列后，期望项出现的条件概率 |
| 平均间隔 | 模式中相邻产品开通的平均时间间隔 |

---

## PrefixSpan 算法

**PrefixSpan** (Prefix-Projected Sequential Pattern Mining)

- 输入：客户产品序列数据库
- 输出：频繁顺序模式
- 优势：通过投影前缀，只需扫描数据库一次

---

## 时序规则示例

```
Frequent Sequential Patterns (support >= 5 customers)
------------------------------------------------------------
  1. 储蓄账户 -> 基金
     Support: 57 customers (57.0%)
  2. 储蓄账户 -> 信用卡
     Support: 50 customers (50.0%)
  3. 储蓄账户 -> 信用卡 -> 基金
     Support: 26 customers (26.0%)
```

---

## 典型产品开通路径

```
First product (initial adoption):
  储蓄账户: 93 customers (93.0%)
  信用卡: 4 customers (4.0%)
  基金: 2 customers (2.0%)

Average Time Gaps Between Products:
  储蓄账户 -> 信用卡: 102 days (3.4 months)
  信用卡 -> 贷款: 215 days (7.2 months)
```

---

## 6. 算法对比与选择

| 特性 | Apriori | FP-Growth | 多维关联 | 时序关联 |
|------|---------|-----------|---------|---------|
| 考虑顺序 | 否 | 否 | 否 | **是** |
| 数据库扫描 | N次 | **2次** | N次 | N次 |
| 候选项集 | 生成 | **不生成** | 生成 | 生成 |
| 适合数据量 | 小(<10K) | **大(>10K)** | 中 | 中 |

---

## 选择指南

```
问题：哪些产品经常一起被购买？
  → FP-Growth (高效，推荐)

问题：什么样的人买什么样的产品组合？
  → 多维关联规则

问题：客户买了A之后，一般多久会买B？
  → 时序关联规则 (PrefixSpan)
```

---

## 性能建议

| 数据规模 | 推荐算法 | 预期耗时 |
|---------|---------|---------|
| < 1,000 条 | Apriori / FP-Growth | < 1秒 |
| 1,000 - 10,000 | FP-Growth | 1-10秒 |
| 10,000 - 100,000 | FP-Growth | 10秒-1分钟 |
| > 100,000 | FP-Growth (分布式) | 1分钟+ |

---

## 7. 业务应用场景

### 场景1：交叉销售推荐

**问题**：如何向现有客户推荐新产品？

**解决方案**：基于关联规则发现"持有产品A的客户也倾向持有产品B"

---

### 场景2：客户分层运营

**问题**：如何针对不同客户群体制定营销策略？

**解决方案**：多维关联规则发现不同人群的产品偏好差异

---

### 场景3：客户生命周期预测

**问题**：客户开通第一个产品后，下一步会开通什么？

**解决方案**：时序关联规则预测客户行为路径

---

### 场景4：产品组合定价

**问题**：如何设计有竞争力的产品组合套餐？

**解决方案**：基于频繁项集发现高支持度的产品组合

---

## 代码文件索引

| 文件 | 功能 | 算法 |
|------|------|------|
| `apriori_test.py` | Apriori算法简单演示 | Apriori |
| `finance_analysis.py` | 金融产品关联规则分析 | Apriori |
| `fpgrowth_analysis.py` | FP-Growth性能对比 | FP-Growth |
| `sequential_analysis.py` | 时序关联规则挖掘 | PrefixSpan |

---

## 快速运行

```bash
# 安装依赖
pip install mlxtend pandas numpy prefixspan

# 基本关联规则
python 关联算法/finance_analysis.py

# FP-Growth对比
python 关联算法/fpgrowth_analysis.py

# 时序关联规则
python 关联算法/generate_customer_data.py
python 关联算法/sequential_analysis.py
```

---

## 总结

1. **Apriori**：基础算法，适合小数据集
2. **FP-Growth**：推荐算法，高效处理大规模数据
3. **多维关联规则**：结合用户画像，精准分析
4. **时序关联规则**：发现先后顺序，预测客户路径

---

*文档生成日期：2026-04-08*

*基于 mlxtend + prefixspan 库实现*
