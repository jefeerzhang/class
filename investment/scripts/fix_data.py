# -*- coding: utf-8 -*-
"""修复 product_categories 字段，确保与 products 一一对应"""
import pandas as pd

# 读取数据
df = pd.read_csv('investment/data/bank_transactions.csv')

# 正确的产品类别映射（文档定义）
product_category_map = {
    '活期存款': '存款类', '定期存款': '存款类', '大额存单': '存款类',
    '理财产品': '投资类', '基金': '投资类', '股票': '投资类', '国债': '投资类',
    '贵金属': '投资类', '外汇': '投资类', '信托': '投资类',
    '保险': '保障类', '年金': '保障类',
    '信用卡': '信贷类', '消费贷款': '信贷类', '住房贷款': '信贷类'
}

# 修复 product_categories 列
def fix_categories(products_str):
    products = [p.strip() for p in products_str.split(',')]
    categories = [product_category_map[p] for p in products]
    return ','.join(categories)

df['product_categories'] = df['products'].apply(fix_categories)

# 保存修复后的数据
df.to_csv('investment/data/bank_transactions.csv', index=False, encoding='utf-8')

print('数据修复完成！')

# 验证修复结果
errors = 0
total = 0
for _, row in df.iterrows():
    prods = [p.strip() for p in row['products'].split(',')]
    cats = [c.strip() for c in row['product_categories'].split(',')]
    for p, c in zip(prods, cats):
        total += 1
        if product_category_map[p] != c:
            errors += 1

print(f'验证结果: 共 {total} 个产品-类别配对, 错误数: {errors}')
print()
print('修复后前10行示例:')
for i in range(10):
    row = df.iloc[i]
    prods = row['products']
    cats = row['product_categories']
    print(f'{prods} -> {cats}')