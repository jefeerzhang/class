# 关联规则分析详解

## 1. 核心思想

关联规则分析的核心思想是：**发现数据中项与项之间的隐藏关联关系**

经典案例：啤酒与尿布
- 超市发现购买尿布的年轻父亲经常同时购买啤酒
- 这种隐藏模式可指导商品摆放、促销策略

## 2. 核心目的

**应用场景：**
- 购物篮分析：商品推荐、货架布局
- 网页点击流：用户行为预测
- 医疗诊断：症状与疾病关联
- 保险欺诈：异常行为检测

**目标：** 从海量交易数据中提取有价值的关联模式

## 3. 核心概念

```python
# 三个关键指标
支持度 (Support) = 同时包含A和B的交易数 / 总交易数
置信度 (Confidence) = 包含A且包含B的交易数 / 包含A的交易数  
提升度 (Lift) = 置信度 / B的支持度
```

**解读：**
- **支持度**：项集出现的频率（重要性）
- **置信度**：A→B的可信程度（规则强度）
- **提升度**：A对B的提升作用
  - lift > 1：正相关（A促进B）
  - lift = 1：无关联
  - lift < 1：负相关

## 4. Apriori算法原理

**核心思想：先验原理**
```
如果一个项集是频繁的，则它的所有子集也是频繁的
如果一个项集是不频繁的，则它的所有超集也是不频繁的
```

**算法步骤：**
```python
# 伪代码
1. 扫描数据库，统计1-项集支持度，筛选频繁项集
2. 连接步：用频繁k项集生成候选k+1项集
3. 剪枝步：删除包含非频繁子集的候选项集
4. 扫描数据库计算候选支持度，筛选频繁项集
5. 重复2-4直到无法生成新的频繁项集
```

## 5. 代码逐行解析

```python
# 第一步：数据准备
transactions = [
    ['面包', '牛奶', '尿布'],        # 交易1
    ['面包', '牛奶', '啤酒', '尿布'], # 交易2
    ['牛奶', '尿布', '鸡蛋'],         # 交易3
    ['面包', '鸡蛋'],                 # 交易4
    ['牛奶', '尿布', '啤酒']          # 交易5
]

# 第二步：数据编码（转换为布尔矩阵）
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 转换结果示例：
#    尿布   牛奶   面包   啤酒   鸡蛋
# 1  True  True  True  False False
# 2  True  True  True  True  False
# ...

# 第三步：挖掘频繁项集
frequent_itemsets = apriori(df, 
                            min_support=0.4,  # 最小支持度40%
                            use_colnames=True)

# 第四步：生成关联规则
rules = association_rules(frequent_itemsets, 
                         metric="confidence",    # 评估指标
                         min_threshold=0.7)      # 最小置信度70%
```

## 6. 运行结果解读

**频繁项集分析：**
```
support=0.8  尿布, 牛奶          # 出现在80%的交易中（最频繁）
support=0.6  面包                # 出现在60%的交易中
support=0.4  牛奶+尿布+啤酒      # 3项组合
```

**关联规则分析：**
```
规则：啤酒 → 尿布
- 支持度：0.4（同时出现概率40%）
- 置信度：1.0（买啤酒100%买尿布）
- 提升度：1.25（比随机购买高25%）

商业建议：将啤酒和尿布货架靠近摆放
```

```
规则：牛奶 → 尿布
- 支持度：0.8（最高频组合）
- 置信度：1.0（绝对关联）
- 提升度：1.25

商业建议：这两个商品可以捆绑促销
```

## 7. 算法优化

**Apriori的缺点：**
- 多次扫描数据库（效率低）
- 产生大量候选项集

**改进算法：**
- **FP-Growth**：只需扫描2次，无需生成候选集
- **Eclat**：基于垂直数据格式

## 8. 参数调优建议

```python
# 支持度设置
min_support = 0.4  # 太低→规则太多噪音大
                    # 太高→漏掉有价值模式

# 置信度设置  
min_confidence = 0.7  # 根据业务需求调整

# 提升度筛选
rules = rules[rules['lift'] > 1]  # 只保留正相关规则
```

## 9. 实际应用技巧

```python
# 过滤高质量规则
quality_rules = rules[
    (rules['support'] >= 0.4) & 
    (rules['confidence'] >= 0.7) & 
    (rules['lift'] > 1.2)
]

# 按提升度排序
top_rules = quality_rules.sort_values('lift', ascending=False)

# 可视化
import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('支持度')
plt.ylabel('置信度')
plt.show()
```

## 10. 完整代码示例

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# 数据准备
transactions = [
    ['面包', '牛奶', '尿布'],
    ['面包', '牛奶', '啤酒', '尿布'],
    ['牛奶', '尿布', '鸡蛋'],
    ['面包', '鸡蛋'],
    ['牛奶', '尿布', '啤酒']
]

# 数据编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 挖掘频繁项集
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("频繁项集:")
print(frequent_itemsets)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\n关联规则:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

## 总结

关联规则通过**支持度-置信度-提升度**三个维度衡量项集关联性，Apriori算法利用先验原理高效剪枝。代码结果显示牛奶与尿布是最强关联组合，可直接指导商业决策。