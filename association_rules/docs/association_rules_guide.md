# 关联规则挖掘完整指南

从基础到时序，四类关联规则算法的理论与实践

---

## 目录

1. [概述：什么是关联规则](#1-概述什么是关联规则)
2. [第一类：基本关联规则 (Apriori)](#2-第一类基本关联规则-apriori)
3. [第二类：进阶算法 (FP-Growth)](#3-第二类进阶算法-fp-growth)
4. [第三类：多维关联规则](#4-第三类多维关联规则)
5. [第四类：时序关联规则](#5-第四类时序关联规则)
6. [算法对比与选择](#6-算法对比与选择)
7. [业务应用场景](#7-业务应用场景)

---

## 1. 概述：什么是关联规则

### 核心思想

关联规则挖掘发现数据中item之间的共现关系。

**典型问题**：顾客买了A商品，还会买什么？

### 三个核心指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **支持度 (Support)** | `P(A ∩ B)` | A和B同时出现的概率 |
| **置信度 (Confidence)** | `P(B\|A)` | 买了A的人，买B的概率 |
| **提升度 (Lift)** | `P(B\|A) / P(B)` | 关联强度，排除随机性 |

### 指标解读

```
支持度 30%：100个顾客中，30人同时买了A和B
置信度 70%：买了A的顾客中，70%也买了B
提升度 1.5：买了A的顾客买B的可能性是平均水平的1.5倍
```

### 阈值建议

| 场景 | 最小支持度 | 最小置信度 | 最小提升度 |
|------|-----------|-----------|-----------|
| 探索性分析 | 0.05 (5%) | 0.5 (50%) | 1.0 |
| 常规分析 | 0.10 (10%) | 0.6 (60%) | 1.1 |
| 高价值规则 | 0.25 (25%) | 0.7 (70%) | 1.2 |

---

## 2. 第一类：基本关联规则 (Apriori)

### 目标

发现交易数据中item同时出现的高频模式

### 作用

- 识别顾客购买行为中的共现规律
- 发现产品之间的正向关联（买A的人也会买B）
- 为交叉销售、货架摆放提供数据支持

### 核心算法：Apriori

** Apriori Principle **：如果一个项集是频繁的，那么它的所有子集也必须是频繁的

**算法流程**：

```
1. 扫描数据库，计算每个1-项集的支持度
2. 剪枝：移除低于阈值的项集
3. 使用频繁1-项集生成候选2-项集
4. 再次扫描数据库，验证候选项集
5. 重复直到没有新的频繁项集
```

**关键问题**：需要多次扫描数据库，产生大量候选项集

### 应用场景

- **零售**：购物篮分析，商品关联推荐
- **电商**："购买此商品的人也购买"推荐
- **金融**：客户产品持有组合分析

### 代码示例

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# 交易数据
transactions = [
    ["面包", "牛奶", "尿布"],
    ["面包", "牛奶", "啤酒", "尿布"],
    ["牛奶", "尿布", "鸡蛋"],
    ["面包", "鸡蛋"],
    ["牛奶", "尿布", "啤酒"],
]

# 编码为布尔矩阵
te = TransactionEncoder()
df_encoded = pd.DataFrame(
    te.fit_transform(transactions),
    columns=te.columns_
)

# 挖掘频繁项集
frequent_itemsets = apriori(df_encoded, min_support=0.4, use_colnames=True)

# 生成关联规则
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7
)

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

**输出示例**：

```
   antecedents consequents   support  confidence  lift
0    (面包,)    (牛奶,)      0.6       0.75       1.25
1    (尿布,)    (牛奶,)      0.6       0.75       1.25
2    (面包,)    (鸡蛋,)      0.4       0.50       1.67
```

### 金融产品分析代码

**文件**：`finance_analysis.py`

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# 加载客户数据
csv_file = "customer_products.csv"
df = pd.read_csv(csv_file)

# 提取交易列表
transactions = df["持有产品"].apply(lambda x: x.split("|")).tolist()

# 编码
te = TransactionEncoder()
df_encoded = pd.DataFrame(
    te.fit_transform(transactions),
    columns=te.columns_
)

# 挖掘频繁项集 (最小支持度 25%)
frequent_itemsets = apriori(df_encoded, min_support=0.25, use_colnames=True)

# 生成关联规则 (最小置信度 60%)
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6,
    num_itemsets=len(frequent_itemsets)
)

# 筛选高质量规则
quality_rules = rules[(rules["lift"] > 1.1) & (rules["confidence"] > 0.7)]

# 输出结果
for idx, row in quality_rules.head(10).iterrows():
    ant = ", ".join(list(row["antecedents"]))
    con = ", ".join(list(row["consequents"]))
    print(f"{ant} -> {con}")
    print(f"  支持度: {row['support']:.1%} | 置信度: {row['confidence']:.1%} | 提升度: {row['lift']:.2f}")
```

**运行**：

```bash
python 关联算法/finance_analysis.py
```

---

## 3. 第二类：进阶算法 (FP-Growth)

### 目标

解决Apriori算法多次扫描数据库的问题，更高效地挖掘频繁项集

### 作用

- 减少数据库扫描次数（从N次减少到2次）
- 大幅提升挖掘效率，尤其在大规模数据集上
- 保持与Apriori相同的结果质量

### 核心算法：FP-Growth

** FP-Tree (Frequent Pattern Tree) **：一种压缩表示事务数据库的树结构

**算法流程**：

```
1. 第一次扫描：统计每个item的出现频率，移除低于阈值的item
2. 第二次扫描：构建FP-Tree，按频率排序插入事务
3. 递归挖掘：从FP-Tree的条件模式基递归挖掘频繁项集
```

**核心优势**：

- 不生成候选项集，直接挖掘
- 只需扫描数据库2次
- 比Apriori快3-10倍

### 应用场景

- **大规模数据集**：千万级交易记录
- **实时分析**：需要快速响应的场景
- **资源受限环境**：内存和计算资源有限

### 代码示例

**文件**：`fpgrowth_analysis.py`

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import time

# 数据编码
te = TransactionEncoder()
df_encoded = pd.DataFrame(
    te.fit_transform(transactions),
    columns=te.columns_
)

# FP-Growth 挖掘
start = time.time()
frequent_itemsets = fpgrowth(
    df_encoded,
    min_support=0.25,
    use_colnames=True
)
elapsed = time.time() - start

print(f"FP-Growth 耗时: {elapsed:.4f} 秒")
print(f"发现频繁项集: {len(frequent_itemsets)} 个")

# 添加长度列便于分析
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
```

**FP-Growth vs Apriori 性能对比**：

```python
# 对比两种算法
start_fpg = time.time()
fpg_itemsets = fpgrowth(df_encoded, min_support=0.25, use_colnames=True)
time_fpg = time.time() - start_fpg

start_apr = time.time()
apr_itemsets = apriori(df_encoded, min_support=0.25, use_colnames=True)
time_apr = time.time() - start_apr

speedup = time_apr / time_fpg
print(f"FP-Growth 比 Apriori 快: {speedup:.2f} 倍")
```

### 金融产品分析代码

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

df = pd.read_csv("customer_products.csv")
transactions = df["持有产品"].apply(lambda x: x.split("|")).tolist()

te = TransactionEncoder()
df_encoded = pd.DataFrame(
    te.fit_transform(transactions),
    columns=te.columns_
)

# 使用FP-Growth（推荐）
frequent_itemsets = fpgrowth(df_encoded, min_support=0.25, use_colnames=True)

# 添加项集长度
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

# 按长度分组查看
for length in sorted(frequent_itemsets["length"].unique()):
    subset = frequent_itemsets[frequent_itemsets["length"] == length].sort_values(
        "support", ascending=False
    )
    print(f"\n{length}-项集:")
    for idx, row in subset.head(5).iterrows():
        items = ", ".join(list(row["itemsets"]))
        print(f"  - [{items}]: {row['support']:.1%}")
```

**运行**：

```bash
python 关联算法/fpgrowth_analysis.py
```

---

## 4. 第三类：多维关联规则

### 目标

在关联规则中引入多个维度（用户属性、时间、场景等），发现更精细的关联模式

### 作用

- 分析"什么样的人"会购买"什么样的产品组合"
- 支持客户分群的精细化运营
- 发现跨维度的条件关联规则

### 核心概念

**单维规则**：`储蓄账户 → 信用卡` (只看产品维度)

**多维规则**：`年龄=30-40 ∧ 收入=高 → 持有理财产品`

### 维度类型

| 维度 | 示例 | 类型 |
|------|------|------|
| 人口属性 | 年龄、性别、地区 | 类别型 |
| 消费能力 | 收入等级、信用评分 | 数值型/类别型 |
| 行为特征 | 购买频率、渠道偏好 | 数值型 |
| 时间特征 | 季度、节假日 | 时间型 |

### 应用场景

- **精准营销**：按客户画像推荐产品
- **风险评估**：多维条件下的违约关联
- **个性化推荐**：结合用户画像的千人千面推荐

### 代码示例

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 加载数据
df = pd.read_csv("customer_products.csv")

# ========== 方法1：扩展产品维度 ==========
# 将年龄和收入等级编码为产品
df["年龄_青年"] = df["年龄"].apply(lambda x: 1 if x < 35 else 0)
df["年龄_中年"] = df["年龄"].apply(lambda x: 1 if 35 <= x < 50 else 0)
df["收入_高"] = df["收入等级"].apply(lambda x: 1 if x == "高" else 0)

# 构建多维交易列表
def build_multi_dimensional_transaction(row):
    items = row["持有产品"].split("|")
    if row["年龄"] < 35:
        items.append("青年客户")
    else:
        items.append("中老年客户")
    if row["收入等级"] == "高":
        items.append("高收入")
    return items

transactions = df.apply(build_multi_dimensional_transaction, axis=1).tolist()

# 编码并挖掘
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# 筛选包含客户属性的规则
multi_dimensional_rules = rules[
    rules["antecedents"].apply(lambda x: any(
        item in ["青年客户", "中老年客户", "高收入"] for item in x
    ))
]

print("多维关联规则示例:")
for idx, row in multi_dimensional_rules.head(10).iterrows():
    ant = ", ".join(list(row["antecedents"]))
    con = ", ".join(list(row["consequents"]))
    print(f"{ant} -> {con}")
    print(f"  置信度: {row['confidence']:.1%} | 支持度: {row['support']:.1%}")
```

### 金融产品多维分析代码

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("customer_products.csv")

print("=" * 60)
print("多维关联规则分析")
print("=" * 60)

# 按收入等级分层分析
print("\n【按收入等级分层的产品偏好】")

for income_level in ["低", "中", "高"]:
    subset = df[df["收入等级"] == income_level]
    transactions = subset["持有产品"].apply(lambda x: x.split("|")).tolist()

    te = TransactionEncoder()
    df_encoded = pd.DataFrame(
        te.fit_transform(transactions),
        columns=te.columns_
    )

    frequent_itemsets = fpgrowth(df_encoded, min_support=0.15, use_colnames=True)

    print(f"\n{income_level}收入客户 (n={len(subset)}):")
    if len(frequent_itemsets) > 0:
        top_itemsets = frequent_itemsets.sort_values("support", ascending=False).head(5)
        for idx, row in top_itemsets.iterrows():
            items = ", ".join(list(row["itemsets"]))
            print(f"  - {items}: {row['support']:.1%}")
```

---

## 5. 第四类：时序关联规则

### 目标

发现事件发生的先后顺序规律，不仅问"哪些产品一起出现"，更问"先买X后买Y"的时序模式

### 作用

- 理解客户产品开通的生命周期路径
- 预测客户未来的购买意向
- 发现产品之间的因果/时序依赖关系

### 核心概念

**时序模式 (Sequential Pattern)**：`[A] → [B]` 表示买了A之后倾向于买B

**关键指标**：

| 指标 | 含义 |
|------|------|
| 支持度 | 序列在所有客户中出现的频率 |
| 置信度 | 给定前缀序列后，期望项出现的条件概率 |
| 平均间隔 | 模式中相邻产品开通的平均时间间隔 |

### 算法：PrefixSpan

**PrefixSpan** (Prefix-Projected Sequential Pattern Mining)：

- 输入：客户产品序列数据库
- 输出：频繁顺序模式

**算法优势**：

- 通过投影前缀，只需扫描数据库一次
- 时间复杂度优于类Apriori的暴力方法

### 应用场景

- **客户旅程分析**：识别典型的产品开通路径
- **交叉销售时机**：预测客户下一步可能需要什么
- **生命周期价值预测**：根据早期行为预测客户价值

### 代码示例

**文件**：`sequential_analysis.py`

```python
from prefixspan import PrefixSpan
import pandas as pd
import numpy as np

# 读取时序数据
df = pd.read_csv("customer_products_temporal.csv")
df["开通日期"] = pd.to_datetime(df["开通日期"])

# 按客户聚合，按日期排序得到产品序列
customer_sequences = (
    df.sort_values(["客户ID", "开通日期"])
    .groupby("客户ID")["产品"]
    .apply(list)
    .tolist()
)

print(f"客户数量: {len(customer_sequences)}")

# ========== 挖掘频繁顺序模式 ==========
ps = PrefixSpan(customer_sequences)
patterns = ps.frequent(5)  # 支持度 >= 5

patterns.sort(key=lambda x: -x[0])

print("\n频繁顺序模式:")
for sup, pattern in patterns:
    if len(pattern) >= 2:
        support_pct = sup / len(customer_sequences) * 100
        print(f"  {' -> '.join(pattern)}: {support_pct:.1f}%")

# ========== 生成时序关联规则 ==========
def generate_temporal_rules(customer_sequences, patterns):
    """从顺序模式生成时序规则"""
    rules = []

    for sup, pattern in patterns:
        if len(pattern) >= 2:
            # 生成 A -> B 规则
            for i in range(1, len(pattern)):
                antecedent = tuple(pattern[:i])
                consequent = pattern[i]

                # 计算置信度
                conf_count = 0
                total_count = 0

                for seq in customer_sequences:
                    for j in range(len(seq) - len(antecedent) + 1):
                        if tuple(seq[j:j + len(antecedent)]) == antecedent:
                            total_count += 1
                            # 检查consequent是否在antecedent之后
                            if consequent in seq[j + len(antecedent):]:
                                conf_count += 1
                            break

                if total_count > 0:
                    confidence = conf_count / total_count
                    rules.append({
                        "antecedent": " -> ".join(antecedent),
                        "consequent": consequent,
                        "support": sup / len(customer_sequences),
                        "confidence": confidence
                    })

    return rules

rules = generate_temporal_rules(customer_sequences, patterns)
rules.sort(key=lambda x: -x["confidence"])

print("\n时序关联规则:")
print("规则解读: 如果客户【先】购买了 X，那么【后续】会购买 Y")
for rule in rules[:10]:
    print(f"  {rule['antecedent']} -> {rule['consequent']}")
    print(f"    支持度: {rule['support']:.1%} | 置信度: {rule['confidence']:.1%}")
```

### 时序数据生成代码

**文件**：`generate_customer_data.py` (已修改)

```python
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# 产品开通的典型时间间隔（月）
product_intervals = {
    "储蓄账户": (0, 0),      # 基准产品，通常最先开通
    "信用卡": (1, 6),        # 开卡后1-6个月内
    "基金": (2, 8),          # 2-8个月
    "理财产品": (3, 10),     # 3-10个月
    "贷款": (6, 18),         # 6-18个月（较晚）
    "保险": (4, 12),         # 4-12个月
}

def generate_opening_date(base_date, product):
    """生成产品开通日期"""
    min_months, max_months = product_intervals[product]
    months_delay = random.randint(min_months, max_months)
    return base_date + timedelta(days=months_delay * 30)

# 生成时序数据
base_date = datetime(2023, 1, 1)
customer_records = []

for i in range(1, 101):
    customer_id = f"C{str(i).zfill(4)}"
    products_held = random.choice(product_combinations).copy()
    age = random.randint(22, 65)
    income_tier = random.choices(["低", "中", "高"], weights=[0.3, 0.5, 0.2])[0]

    # 为每个产品生成开通日期
    opening_dates = {p: generate_opening_date(base_date, p) for p in products_held}
    sorted_products = sorted(products_held, key=lambda p: opening_dates[p])

    for product in sorted_products:
        customer_records.append({
            "客户ID": customer_id,
            "年龄": age,
            "收入等级": income_tier,
            "产品": product,
            "开通日期": opening_dates[product].strftime("%Y-%m-%d"),
        })

df = pd.DataFrame(customer_records)
df.to_csv("customer_products_temporal.csv", index=False, encoding="utf-8-sig")

print(f"生成 {len(df)} 条产品开通记录")
```

### 典型输出示例

```
============================================================
Sequential Pattern Mining - Temporal Association Rules
============================================================

Customer count: 100
Avg products per customer: 3.37

Frequent Sequential Patterns (support >= 5 customers)
------------------------------------------------------------
  1. 储蓄账户 -> 基金
     Support: 57 customers (57.0%)
  2. 储蓄账户 -> 信用卡
     Support: 50 customers (50.0%)
  3. 储蓄账户 -> 信用卡 -> 基金
     Support: 26 customers (26.0%)

Temporal Association Rules
------------------------------------------------------------
Rule interpretation: If customer buys X FIRST, then buys Y LATER
------------------------------------------------------------
  Rule 1: 储蓄账户 -> 理财产品 -> 基金
    Support: 6.0% | Confidence: 83.3%
  Rule 2: 储蓄账户 -> 信用卡
    Support: 57.0% | Confidence: 61.3%

Typical Product Adoption Paths
------------------------------------------------------------
First product (initial adoption):
  储蓄账户: 93 customers (93.0%)
  信用卡: 4 customers (4.0%)
  基金: 2 customers (2.0%)

Average Time Gaps Between Products
------------------------------------------------------------
  储蓄账户 -> 信用卡: 102 days (3.4 months)
  信用卡 -> 贷款: 215 days (7.2 months)
```

**运行**：

```bash
python 关联算法/generate_customer_data.py  # 生成时序数据
python 关联算法/sequential_analysis.py     # 分析时序规则
```

---

## 6. 算法对比与选择

### 特性对比

| 特性 | Apriori | FP-Growth | 多维关联 | 时序关联 |
|------|---------|-----------|---------|---------|
| **考虑顺序** | 否 | 否 | 否 | 是 |
| **多维支持** | 可扩展 | 可扩展 | 原生 | 可扩展 |
| **算法复杂度** | O(2^n) | O(n×m) | 取决于实现 | O(n×m) |
| **数据库扫描次数** | N次 | 2次 | N次 | N次 |
| **候选项集** | 生成 | 不生成 | 生成 | 生成 |
| **适合数据量** | 小 (<10K) | 大 (>10K) | 中 | 中 |

### 选择指南

```
问题：哪些产品经常一起被购买？
  → FP-Growth (高效，推荐)

问题：买A的人会不会买B？
  → FP-Growth + 关联规则

问题：什么样的人买什么样的产品组合？
  → 多维关联规则

问题：客户买了A之后，一般多久会买B？
  → 时序关联规则 (PrefixSpan)

问题：客户的典型产品开通路径是什么？
  → 时序关联规则 (PrefixSpan)
```

### 性能建议

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

**业务价值**：
- 提高客户产品渗透率
- 降低获客成本
- 提升客户生命周期价值

**代码实现**：

```python
# 发现高置信度规则
high_conf_rules = rules[rules["confidence"] > 0.7].sort_values("confidence", ascending=False)

# 转化为推荐策略
for idx, row in high_conf_rules.head(5).iterrows():
    ant = ", ".join(list(row["antecedents"]))
    con = ", ".join(list(row["consequents"]))
    print(f"推荐策略: 向已持有[{ant}]的客户推荐[{con}]")
    print(f"预期转化率: {row['confidence']:.1%}")
```

### 场景2：客户分层运营

**问题**：如何针对不同客户群体制定营销策略？

**解决方案**：多维关联规则发现不同人群的产品偏好差异

**代码实现**：

```python
# 按收入等级分层
for income in ["低", "中", "高"]:
    subset_rules = rules[income_level_rules[income]]
    top_rule = subset_rules.sort_values("confidence", ascending=False).iloc[0]
    print(f"{income}收入客户: {top_rule['antecedents']} -> {top_rule['consequents']}")
```

### 场景3：客户生命周期预测

**问题**：客户开通第一个产品后，下一步会开通什么？

**解决方案**：时序关联规则预测客户行为路径

**代码实现**：

```python
# 预测下一个产品
def predict_next_product(customer_sequence, rules):
    """基于时序规则预测下一个产品"""
    last_product = customer_sequence[-1]

    # 查找以last_product为前提的规则
    candidates = [
        rule for rule in rules
        if rule["antecedent"].endswith(last_product)
    ]

    return sorted(candidates, key=lambda x: -x["confidence"])[0] if candidates else None
```

### 场景4：产品组合定价

**问题**：如何设计有竞争力的产品组合套餐？

**解决方案**：基于频繁项集发现高支持度的产品组合

**代码实现**：

```python
# 发现高支持度的产品组合
combo_rules = rules[
    (rules["antecedent_len"] + rules["consequent_len"]) >= 3
].sort_values("support", ascending=False)

for idx, row in combo_rules.head(3).iterrows():
    products = list(row["antecedents"]) + list(row["consequent"])
    print(f"产品组合: {' + '.join(products)}")
    print(f"支持度: {row['support']:.1%} (约{int(row['support']*10000)}人)")
```

---

## 附录：代码文件索引

| 文件 | 功能 | 算法 |
|------|------|------|
| `apriori_test.py` | Apriori算法简单演示 | Apriori |
| `finance_analysis.py` | 金融产品关联规则分析 | Apriori |
| `fpgrowth_analysis.py` | FP-Growth性能对比 | FP-Growth |
| `generate_customer_data.py` | 生成模拟数据(含时序) | - |
| `sequential_analysis.py` | 时序关联规则挖掘 | PrefixSpan |
| `visualize_rules.py` | 规则可视化 | - |

## 附录：快速运行

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

*文档生成日期：2026-04-08*
*基于 mlxtend + prefixspan 库实现*
