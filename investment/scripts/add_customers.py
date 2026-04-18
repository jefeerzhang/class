# -*- coding: utf-8 -*-
"""补充客户数据，使客户数达到500人"""
import pandas as pd
import random
from datetime import datetime, timedelta

# 读取现有数据
df = pd.read_csv('investment/data/bank_transactions.csv')

# 分析现有客户
print('=== 现有客户分析 ===')
print(f'当前客户数: {df["customer_id"].nunique()}')

# 分析分布
print('\n年龄段分布:')
print(df['age_group'].value_counts().to_dict())

print('\n收入水平分布:')
print(df['income_level'].value_counts().to_dict())

print('\n职业分布:')
print(df['occupation'].value_counts().to_dict())

print('\n风险偏好分布:')
print(df['risk_tolerance'].value_counts().to_dict())

# 每位客户交易数量
cust_trans = df.groupby('customer_id').size()
trans_counts = cust_trans.value_counts().to_dict()
print(f'\n每位客户交易数量分布: {dict(sorted(trans_counts.items()))}')

# 产品和类别映射
product_category_map = {
    '活期存款': '存款类', '定期存款': '存款类', '大额存单': '存款类',
    '理财产品': '投资类', '基金': '投资类', '股票': '投资类', '国债': '投资类',
    '贵金属': '投资类', '外汇': '投资类', '信托': '投资类',
    '保险': '保障类', '年金': '保障类',
    '信用卡': '信贷类', '消费贷款': '信贷类', '住房贷款': '信贷类'
}
products_list = list(product_category_map.keys())

# 现有客户ID
existing_ids = set(df['customer_id'].unique())
print(f'\n客户ID范围: {min(existing_ids)} ~ {max(existing_ids)}')

# 需要补充的客户数量
need_customers = 500 - len(existing_ids)
print(f'\n需要补充: {need_customers} 个客户')

if need_customers > 0:
    # 现有分布统计
    age_groups = ['25-30', '31-35', '36-40', '41-45', '46-50']
    income_levels = ['低', '中', '高']
    occupations = ['企业职员', '公务员', '个体户', '自由职业', '退休人员']
    risk_tolerances = ['保守型', '稳健型', '进取型']

    # 从现有数据中获取分布权重
    age_weights = df['age_group'].value_counts(normalize=True).to_dict()
    income_weights = df['income_level'].value_counts(normalize=True).to_dict()
    occupation_weights = df['occupation'].value_counts(normalize=True).to_dict()
    risk_weights = df['risk_tolerance'].value_counts(normalize=True).to_dict()

    # 生成新客户数据
    new_records = []

    # 找到缺失的客户ID
    all_possible_ids = [f'C{str(i).zfill(4)}' for i in range(1, 501)]
    missing_ids = [id for id in all_possible_ids if id not in existing_ids]

    print(f'\n缺失的客户ID: {missing_ids[:10]}...')

    for cust_id in missing_ids[:need_customers]:
        # 随机选择客户属性（按现有分布权重）
        age = random.choices(age_groups, weights=[age_weights.get(a, 0.2) for a in age_groups])[0]
        income = random.choices(income_levels, weights=[income_weights.get(i, 0.33) for i in income_levels])[0]
        occupation = random.choices(occupations, weights=[occupation_weights.get(o, 0.2) for o in occupations])[0]
        risk = random.choices(risk_tolerances, weights=[risk_weights.get(r, 0.33) for r in risk_tolerances])[0]

        # 生成1-6笔交易
        num_trans = random.choices(list(trans_counts.keys()), weights=list(trans_counts.values()))[0]

        for _ in range(num_trans):
            # 随机日期
            start_date = datetime(2025, 1, 1)
            random_days = random.randint(0, 364)
            trans_date = start_date + timedelta(days=random_days)
            trans_month = trans_date.strftime('%Y-%m')

            # 随机选择2-4个产品
            num_products = random.randint(2, 4)
            selected_products = random.sample(products_list, num_products)
            products_str = ','.join(selected_products)
            categories_str = ','.join([product_category_map[p] for p in selected_products])

            # 随机金额（参考现有数据分布）
            amount = random.randint(10000, 1000000)

            new_records.append({
                'customer_id': cust_id,
                'age_group': age,
                'income_level': income,
                'occupation': occupation,
                'risk_tolerance': risk,
                'transaction_date': trans_date.strftime('%Y-%m-%d'),
                'transaction_month': trans_month,
                'products': products_str,
                'product_categories': categories_str,
                'total_amount': amount
            })

    # 添加新记录
    new_df = pd.DataFrame(new_records)
    df = pd.concat([df, new_df], ignore_index=True)

    # 保存
    df.to_csv('investment/data/bank_transactions.csv', index=False, encoding='utf-8')

    print(f'\n已添加 {len(new_records)} 条新交易记录')
    print(f'新客户数: {df["customer_id"].nunique()}')
    print(f'总交易记录数: {len(df)}')

    # 显示新增数据示例
    print('\n新增数据示例:')
    print(new_df.head(10).to_string())

else:
    print('\n客户数已达到500，无需补充')