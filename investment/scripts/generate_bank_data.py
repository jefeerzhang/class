# -*- coding: utf-8 -*-
"""
生成银行金融产品交易模拟数据
用于关联规则挖掘作业
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 设置随机种子保证可复现
np.random.seed(42)
random.seed(42)

# ============== 基础配置 ==============

# 客户数量
N_CUSTOMERS = 500

# 交易记录数量
N_TRANSACTIONS = 3000

# 时间范围
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

# 年龄段
AGE_GROUPS = ['25-30', '31-35', '36-40', '41-45', '46-50']

# 收入水平
INCOME_LEVELS = ['低', '中', '高']

# 职业类型
OCCUPATIONS = ['企业职员', '公务员', '个体户', '自由职业', '退休人员']

# 风险偏好
RISK_TOLERANCES = ['保守型', '稳健型', '进取型']

# 产品及其类别
PRODUCTS = {
    # 存款类
    '活期存款': '存款类',
    '定期存款': '存款类',
    '大额存单': '存款类',
    # 投资类
    '理财产品': '投资类',
    '基金': '投资类',
    '股票': '投资类',
    '国债': '投资类',
    '贵金属': '投资类',
    '外汇': '投资类',
    '信托': '投资类',
    # 保障类
    '保险': '保障类',
    '年金': '保障类',
    # 信贷类
    '信用卡': '信贷类',
    '消费贷款': '信贷类',
    '住房贷款': '信贷类',
}

PRODUCT_NAMES = list(PRODUCTS.keys())


# ============== 客户画像生成 ==============

def generate_customer_profiles(n_customers):
    """生成客户基本信息"""
    
    # 年龄段分布（中年人较多）
    age_weights = [0.15, 0.25, 0.30, 0.20, 0.10]
    
    # 根据年龄段设定收入和职业的关联
    profiles = []
    
    for i in range(n_customers):
        customer_id = f'C{str(i+1).zfill(4)}'
        age_group = np.random.choice(AGE_GROUPS, p=age_weights)
        
        # 年龄段影响收入分布
        if age_group in ['25-30']:
            income_level = np.random.choice(INCOME_LEVELS, p=[0.4, 0.45, 0.15])
            occupation = np.random.choice(OCCUPATIONS, p=[0.5, 0.2, 0.1, 0.15, 0.05])
        elif age_group in ['31-35', '36-40']:
            income_level = np.random.choice(INCOME_LEVELS, p=[0.2, 0.5, 0.3])
            occupation = np.random.choice(OCCUPATIONS, p=[0.4, 0.25, 0.15, 0.15, 0.05])
        elif age_group in ['41-45']:
            income_level = np.random.choice(INCOME_LEVELS, p=[0.15, 0.45, 0.4])
            occupation = np.random.choice(OCCUPATIONS, p=[0.35, 0.25, 0.2, 0.1, 0.1])
        else:  # 46-50
            income_level = np.random.choice(INCOME_LEVELS, p=[0.2, 0.4, 0.4])
            occupation = np.random.choice(OCCUPATIONS, p=[0.3, 0.2, 0.15, 0.1, 0.25])
        
        # 收入和年龄影响风险偏好
        if income_level == '低':
            risk_tolerance = np.random.choice(RISK_TOLERANCES, p=[0.5, 0.35, 0.15])
        elif income_level == '中':
            risk_tolerance = np.random.choice(RISK_TOLERANCES, p=[0.25, 0.5, 0.25])
        else:  # 高收入
            risk_tolerance = np.random.choice(RISK_TOLERANCES, p=[0.15, 0.35, 0.5])
        
        profiles.append({
            'customer_id': customer_id,
            'age_group': age_group,
            'income_level': income_level,
            'occupation': occupation,
            'risk_tolerance': risk_tolerance
        })
    
    return pd.DataFrame(profiles)


# ============== 产品购买偏好设置 ==============

def get_product_preferences(profile):
    """根据客户画像生成产品偏好权重"""
    
    weights = {}
    
    # 基础权重：存款类产品大众化
    weights['活期存款'] = 0.8
    weights['定期存款'] = 0.5
    weights['大额存单'] = 0.2
    
    # 投资类产品
    weights['理财产品'] = 0.5
    weights['基金'] = 0.4
    weights['股票'] = 0.25
    weights['国债'] = 0.3
    weights['贵金属'] = 0.15
    weights['外汇'] = 0.1
    weights['信托'] = 0.08
    
    # 保障类
    weights['保险'] = 0.4
    weights['年金'] = 0.15
    
    # 信贷类
    weights['信用卡'] = 0.6
    weights['消费贷款'] = 0.25
    weights['住房贷款'] = 0.15
    
    # 根据风险偏好调整
    if profile['risk_tolerance'] == '保守型':
        weights['活期存款'] *= 1.3
        weights['定期存款'] *= 1.5
        weights['大额存单'] *= 1.5
        weights['国债'] *= 1.5
        weights['保险'] *= 1.3
        weights['股票'] *= 0.3
        weights['外汇'] *= 0.2
        weights['信托'] *= 0.2
    elif profile['risk_tolerance'] == '进取型':
        weights['股票'] *= 2.0
        weights['基金'] *= 1.5
        weights['理财产品'] *= 1.3
        weights['外汇'] *= 1.5
        weights['贵金属'] *= 1.5
        weights['信托'] *= 1.5
        weights['定期存款'] *= 0.5
        weights['大额存单'] *= 0.5
    
    # 根据收入水平调整
    if profile['income_level'] == '高':
        weights['理财产品'] *= 1.5
        weights['基金'] *= 1.5
        weights['信托'] *= 2.0
        weights['大额存单'] *= 1.5
        weights['年金'] *= 1.5
    elif profile['income_level'] == '低':
        weights['信用卡'] *= 1.3
        weights['消费贷款'] *= 1.3
        weights['信托'] *= 0.3
    
    # 根据年龄段调整
    if profile['age_group'] in ['25-30', '31-35']:
        weights['信用卡'] *= 1.3
        weights['消费贷款'] *= 1.5
        weights['住房贷款'] *= 1.5
        weights['股票'] *= 1.3
        weights['基金'] *= 1.3
    elif profile['age_group'] in ['46-50']:
        weights['保险'] *= 1.5
        weights['年金'] *= 1.5
        weights['国债'] *= 1.3
        weights['股票'] *= 0.7
    
    return weights


# ============== 交易生成 ==============

def generate_transactions(customers_df, n_transactions):
    """生成交易记录"""
    
    transactions = []
    
    # 生成日期范围
    date_range = (END_DATE - START_DATE).days
    
    for _ in range(n_transactions):
        # 随机选择客户
        customer_idx = np.random.randint(0, len(customers_df))
        profile = customers_df.iloc[customer_idx]
        
        # 生成交易日期
        random_days = np.random.randint(0, date_range)
        trans_date = START_DATE + timedelta(days=random_days)
        trans_month = trans_date.strftime('%Y-%m')
        
        # 根据偏好选择产品
        preferences = get_product_preferences(profile)
        
        # 每笔交易购买 1-5 种产品
        n_products = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.3, 0.3, 0.15, 0.1])
        
        # 加权随机选择产品
        product_list = list(preferences.keys())
        weights_list = [preferences[p] for p in product_list]
        weights_list = np.array(weights_list) / sum(weights_list)  # 归一化
        
        selected_products = np.random.choice(
            product_list, 
            size=min(n_products, len(product_list)), 
            replace=False, 
            p=weights_list
        )
        selected_products = list(selected_products)
        
        # 生成产品类别
        categories = list(set([PRODUCTS[p] for p in selected_products]))
        
        # 生成交易金额
        base_amount = {
            '存款类': 50000,
            '投资类': 80000,
            '保障类': 20000,
            '信贷类': 100000
        }
        
        total_amount = 0
        for p in selected_products:
            category = PRODUCTS[p]
            # 金额有较大波动
            amount = np.random.lognormal(
                mean=np.log(base_amount[category]), 
                sigma=0.8
            )
            total_amount += amount
        
        # 根据收入调整金额
        if profile['income_level'] == '低':
            total_amount *= 0.5
        elif profile['income_level'] == '高':
            total_amount *= 2.0
        
        total_amount = int(total_amount)
        
        transactions.append({
            'customer_id': profile['customer_id'],
            'age_group': profile['age_group'],
            'income_level': profile['income_level'],
            'occupation': profile['occupation'],
            'risk_tolerance': profile['risk_tolerance'],
            'transaction_date': trans_date.strftime('%Y-%m-%d'),
            'transaction_month': trans_month,
            'products': ','.join(selected_products),
            'product_categories': ','.join(categories),
            'total_amount': total_amount
        })
    
    return pd.DataFrame(transactions)


# ============== 主程序 ==============

if __name__ == '__main__':
    print("=" * 50)
    print("开始生成银行金融产品交易模拟数据")
    print("=" * 50)
    
    # 生成客户画像
    print("\n[1/3] 生成客户画像...")
    customers_df = generate_customer_profiles(N_CUSTOMERS)
    print(f"  - 客户数量: {len(customers_df)}")
    print(f"  - 年龄段分布:\n{customers_df['age_group'].value_counts().sort_index()}")
    
    # 生成交易记录
    print("\n[2/3] 生成交易记录...")
    transactions_df = generate_transactions(customers_df, N_TRANSACTIONS)
    print(f"  - 交易记录数: {len(transactions_df)}")
    print(f"  - 时间范围: {transactions_df['transaction_date'].min()} ~ {transactions_df['transaction_date'].max()}")
    
    # 保存数据
    print("\n[3/3] 保存数据...")
    output_path = '../data/bank_transactions.csv'
    transactions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  - 数据已保存至: {output_path}")
    
    # 数据概览
    print("\n" + "=" * 50)
    print("数据概览")
    print("=" * 50)
    print(f"\n总交易记录数: {len(transactions_df)}")
    print(f"唯一客户数: {transactions_df['customer_id'].nunique()}")
    print(f"月份数: {transactions_df['transaction_month'].nunique()}")
    
    print("\n产品购买频率:")
    # 统计每个产品的出现次数
    all_products = []
    for products in transactions_df['products']:
        all_products.extend(products.split(','))
    from collections import Counter
    product_counts = Counter(all_products)
    for product, count in product_counts.most_common():
        print(f"  {product}: {count} 次 ({count/len(transactions_df)*100:.1f}%)")
    
    print("\n交易金额统计:")
    print(f"  均值: {transactions_df['total_amount'].mean():,.0f} 元")
    print(f"  中位数: {transactions_df['total_amount'].median():,.0f} 元")
    print(f"  最小值: {transactions_df['total_amount'].min():,} 元")
    print(f"  最大值: {transactions_df['total_amount'].max():,} 元")
    
    print("\n" + "=" * 50)
    print("数据生成完成！")
    print("=" * 50)