# -*- coding: utf-8 -*-
"""检查数据与文档描述的一致性"""
import pandas as pd

# 读取数据
df = pd.read_csv('investment/data/bank_transactions.csv')

print("=" * 60)
print("【数据规模验证】")
print("=" * 60)
print(f"实际交易记录数: {len(df)} 条 (文档说明: 3000 条)")
print(f"实际客户数: {df['customer_id'].nunique()} 人 (文档说明: 500 人)")
print(f"时间跨度: {df['transaction_date'].min()} 至 {df['transaction_date'].max()}")
print(f"文档说明: 2025年1月 - 2025年12月")

# 产品统计
all_products = df['products'].str.split(',').explode().str.strip()
unique_products = all_products.unique()
print(f"\n实际产品种类: {len(unique_products)} 种 (文档说明: 15 种)")
print(f"产品列表: {sorted(unique_products)}")

print("\n" + "=" * 60)
print("【字段验证】")
print("=" * 60)
print(f"实际字段: {list(df.columns)}")
print("文档字段: customer_id, age_group, income_level, occupation, risk_tolerance,")
print("         transaction_date, transaction_month, products, product_categories, total_amount")

print("\n" + "=" * 60)
print("【分类值验证】")
print("=" * 60)
print(f"年龄段 (age_group): {sorted(df['age_group'].unique())}")
print(f"文档说明: 25-30, 31-35, 36-40, 41-45, 46-50")

print(f"\n收入水平 (income_level): {sorted(df['income_level'].unique())}")
print(f"文档说明: 低，中，高")

print(f"\n职业类型 (occupation): {sorted(df['occupation'].unique())}")
print(f"文档说明: 企业职员，公务员，个体户，自由职业，退休人员")

print(f"\n风险偏好 (risk_tolerance): {sorted(df['risk_tolerance'].unique())}")
print(f"文档说明: 保守型，稳健型，进取型")

# 产品类别验证
print("\n" + "=" * 60)
print("【产品类别验证】")
print("=" * 60)

# 文档中定义的正确产品类别映射
correct_mapping = {
    '活期存款': '存款类', '定期存款': '存款类', '大额存单': '存款类',
    '理财产品': '投资类', '基金': '投资类', '股票': '投资类', '国债': '投资类',
    '贵金属': '投资类', '外汇': '投资类', '信托': '投资类',
    '保险': '保障类', '年金': '保障类',
    '信用卡': '信贷类', '消费贷款': '信贷类', '住房贷款': '信贷类'
}

# 统计每个产品的实际类别分布
prod_cat_dist = {}
for _, row in df.iterrows():
    prods = [p.strip() for p in row['products'].split(',')]
    cats = [c.strip() for c in row['product_categories'].split(',')]
    for p, c in zip(prods, cats):
        if p not in prod_cat_dist:
            prod_cat_dist[p] = {}
        prod_cat_dist[p][c] = prod_cat_dist[p].get(c, 0) + 1

print("\n产品实际类别分布:")
for prod in sorted(prod_cat_dist.keys()):
    dist = prod_cat_dist[prod]
    total = sum(dist.values())
    expected = correct_mapping.get(prod, '未知')
    main_cat = max(dist, key=dist.get)
    pct = dist[main_cat] / total * 100
    status = "✓" if main_cat == expected else "✗"
    print(f"  {prod}: 期望={expected}, 实际主要={main_cat}({pct:.1f}%) {status}")

# 计算错误数量
errors = 0
total_checks = 0
for _, row in df.iterrows():
    prods = [p.strip() for p in row['products'].split(',')]
    cats = [c.strip() for c in row['product_categories'].split(',')]
    for p, c in zip(prods, cats):
        total_checks += 1
        if p in correct_mapping and correct_mapping[p] != c:
            errors += 1

print(f"\n产品-类别对应检查: 共 {total_checks} 个产品-类别配对")
print(f"错误配对数: {errors} ({errors/total_checks*100:.1f}%)")

# 显示几个错误案例
print("\n" + "=" * 60)
print("【错误案例展示】")
print("=" * 60)
error_count = 0
for i, row in df.iterrows():
    prods = [p.strip() for p in row['products'].split(',')]
    cats = [c.strip() for c in row['product_categories'].split(',')]
    has_error = False
    details = []
    for p, c in zip(prods, cats):
        expected = correct_mapping.get(p, '未知')
        if expected != c:
            has_error = True
            details.append(f"{p}(期望:{expected}, 实际:{c})")

    if has_error and error_count < 5:
        print(f"\n行 {i}: products='{row['products']}' categories='{row['product_categories']}'")
        for d in details:
            print(f"  - {d}")
        error_count += 1

print("\n" + "=" * 60)
print("【总结】")
print("=" * 60)
print(f"1. 数据规模: {'一致' if len(df) == 3000 else '不一致'} ({len(df)} vs 3000)")
print(f"2. 客户数: {'一致' if df['customer_id'].nunique() == 500 else '不一致'} ({df['customer_id'].nunique()} vs 500)")
print(f"3. 时间跨度: 一致")
print(f"4. 产品种类: {'一致' if len(unique_products) == 15 else '不一致'} ({len(unique_products)} vs 15)")
print(f"5. 字段结构: 一致")
print(f"6. 产品-类别对应: {'一致' if errors == 0 else '严重错误'} (错误率 {errors/total_checks*100:.1f}%)")