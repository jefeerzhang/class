import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

products = ["信用卡", "储蓄账户", "基金", "贷款", "理财产品", "保险"]

product_combinations = [
    ["储蓄账户"],
    ["储蓄账户", "信用卡"],
    ["储蓄账户", "基金"],
    ["储蓄账户", "信用卡", "基金"],
    ["储蓄账户", "信用卡", "基金", "理财产品"],
    ["信用卡", "储蓄账户", "贷款"],
    ["储蓄账户", "基金", "保险"],
    ["信用卡", "储蓄账户", "基金", "保险"],
    ["储蓄账户", "贷款", "保险"],
    ["信用卡", "储蓄账户", "贷款", "保险"],
    ["储蓄账户", "基金", "理财产品"],
    ["信用卡", "储蓄账户", "基金", "理财产品", "保险"],
    ["储蓄账户", "保险"],
    ["信用卡", "基金", "理财产品"],
    ["储蓄账户", "贷款"],
    ["信用卡", "储蓄账户", "基金", "贷款", "保险"],
    ["储蓄账户", "理财产品", "保险"],
    ["信用卡", "储蓄账户", "保险"],
    ["基金", "理财产品"],
    ["储蓄账户", "基金", "贷款"],
]

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
    """生成产品开通日期，基于基础日期和产品类型"""
    min_months, max_months = product_intervals[product]
    months_delay = random.randint(min_months, max_months)
    return base_date + timedelta(days=months_delay * 30)

# 基准日期：2023年1月1日
base_date = datetime(2023, 1, 1)

customer_records = []
for i in range(1, 101):
    customer_id = f"C{str(i).zfill(4)}"

    products_held = random.choice(product_combinations).copy()

    if random.random() < 0.3:
        additional = random.choice(["保险", "理财产品", "基金", "贷款"])
        if additional not in products_held:
            products_held.append(additional)

    age = random.randint(22, 65)
    income_tier = random.choices(["低", "中", "高"], weights=[0.3, 0.5, 0.2])[0]

    # 为每个产品生成开通日期
    # 先按时间顺序排列产品（模拟客户旅程）
    opening_dates = {}
    for product in products_held:
        opening_dates[product] = generate_opening_date(base_date, product)

    # 按开通日期排序（确定产品开通顺序）
    sorted_products = sorted(products_held, key=lambda p: opening_dates[p])

    # 为每个产品创建记录
    for product in sorted_products:
        # 添加一些随机性到日期（同一天内可能有多个产品）
        date_noise = random.randint(0, 5)
        actual_date = opening_dates[product] + timedelta(days=date_noise)

        customer_records.append({
            "客户ID": customer_id,
            "年龄": age,
            "收入等级": income_tier,
            "产品": product,
            "开通日期": actual_date.strftime("%Y-%m-%d"),
        })

    # 调整基准日期，让不同客户的开通时间有差异
    base_date = base_date + timedelta(days=random.randint(0, 30))

df = pd.DataFrame(customer_records)

# 保存时序数据
csv_file = r"C:\Users\jefeer\Downloads\opencode\关联算法\data\customer_products_temporal.csv"
df.to_csv(csv_file, index=False, encoding="utf-8-sig")

print(f"成功生成 {len(df)} 条产品开通记录（{len(df['客户ID'].unique())} 位客户）")
print(f"数据已保存到：{csv_file}")
print(f"\n数据预览（前15行）：")
print(df.head(15))
print(f"\n产品开通频次：")
print(df["产品"].value_counts())
print(f"\n客户平均产品数：{df.groupby('客户ID')['产品'].count().mean():.2f}")
