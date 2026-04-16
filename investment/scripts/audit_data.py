# -*- coding: utf-8 -*-
"""数据审核脚本"""

import pandas as pd
from collections import Counter

# 加载数据
df = pd.read_csv('data/bank_transactions.csv')

print("=" * 60)
print("数据完整性检查")
print("=" * 60)

# 基本信息
print(f"\n总记录数: {len(df)}")
print(f"唯一客户数: {df['customer_id'].nunique()}")
print(f"产品种类数: {len(set([p for prods in df['products'] for p in prods.split(',')]))}")
print(f"时间范围: {df['transaction_date'].min()} ~ {df['transaction_date'].max()}")

# 字段检查
print('\n字段列表:', list(df.columns))

# 各字段分布
print("\n" + "=" * 60)
print("字段分布检查")
print("=" * 60)

print("\n[年龄段分布]")
print(df['age_group'].value_counts().sort_index())

print("\n[收入水平分布]")
print(df['income_level'].value_counts())

print("\n[职业类型分布]")
print(df['occupation'].value_counts())

print("\n[风险偏好分布]")
print(df['risk_tolerance'].value_counts())

print("\n[交易月份分布]")
print(df['transaction_month'].value_counts().sort_index())

# 产品统计
print("\n" + "=" * 60)
print("产品统计")
print("=" * 60)

all_products = []
for products in df['products']:
    all_products.extend(products.split(','))
product_counts = Counter(all_products)

print("\n各产品购买频率:")
for product, count in sorted(product_counts.items(), key=lambda x: -x[1]):
    print(f"  {product}: {count} 次 ({count/len(df)*100:.1f}%)")

# 产品类别统计
all_categories = []
for cats in df['product_categories']:
    all_categories.extend(cats.split(','))
cat_counts = Counter(all_categories)
print("\n产品类别分布:")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count} 次")

# 每笔交易产品数量
df['product_count'] = df['products'].apply(lambda x: len(x.split(',')))
print("\n每笔交易产品数量分布:")
print(df['product_count'].value_counts().sort_index())

# 金额统计
print("\n" + "=" * 60)
print("金额统计")
print("=" * 60)
print(f"均值: {df['total_amount'].mean():,.0f} 元")
print(f"中位数: {df['total_amount'].median():,.0f} 元")
print(f"最小值: {df['total_amount'].min():,} 元")
print(f"最大值: {df['total_amount'].max():,} 元")
print(f"标准差: {df['total_amount'].std():,.0f} 元")

# 客户交易次数分布
print("\n" + "=" * 60)
print("客户交易频次分布")
print("=" * 60)
trans_per_customer = df.groupby('customer_id').size()
print(f"人均交易次数: {trans_per_customer.mean():.1f}")
print(f"最少交易次数: {trans_per_customer.min()}")
print(f"最多交易次数: {trans_per_customer.max()}")
print("\n交易次数分布:")
print(trans_per_customer.value_counts().sort_index().head(15))

# 关联规则挖掘可行性检查
print("\n" + "=" * 60)
print("关联规则挖掘可行性检查")
print("=" * 60)

# 检查是否有足够的共现产品组合
from itertools import combinations
combo_counts = Counter()
for products in df['products']:
    prod_list = products.split(',')
    if len(prod_list) >= 2:
        for combo in combinations(sorted(prod_list), 2):
            combo_counts[combo] += 1

print(f"\n产品对共现次数 Top 10:")
for combo, count in combo_counts.most_common(10):
    print(f"  {combo[0]} + {combo[1]}: {count} 次")

# 按维度分组检查
print("\n按年龄段分组样本数:")
print(df.groupby('age_group').size())

print("\n按收入水平分组样本数:")
print(df.groupby('income_level').size())

print("\n按风险偏好分组样本数:")
print(df.groupby('risk_tolerance').size())

# 时序分析可行性
print("\n" + "=" * 60)
print("时序分析可行性检查")
print("=" * 60)

# 客户多次交易情况
multi_trans_customers = (trans_per_customer > 1).sum()
print(f"有多次交易的客户数: {multi_trans_customers}")
print(f"单次交易客户数: {(trans_per_customer == 1).sum()}")

# 时间分布
print("\n按月交易分布:")
print(df.groupby('transaction_month').size())

print("\n" + "=" * 60)
print("审核完成")
print("=" * 60)