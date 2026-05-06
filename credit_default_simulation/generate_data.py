"""
生成模拟的信贷违约数据
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# 样本数量
n = 1000

# 生成特征
age = np.random.randint(22, 65, n)  # 年龄 22-65
income = np.random.lognormal(mean=10.8, sigma=0.5, size=n)  # 年收入，对数正态分布
debt_ratio = np.random.beta(a=2, b=5, size=n)  # 负债率，Beta分布
credit_score = np.clip(np.random.normal(loc=680, scale=50, size=n), 300, 850)  # 信用评分
months_employed = np.random.randint(0, 360, n)  # 在职月数
num_credit_lines = np.random.poisson(lam=5, size=n)  # 信贷账户数
num_late_payments = np.random.poisson(lam=1, size=n)  # 逾期次数
loan_amount = np.random.lognormal(mean=3.0, sigma=0.8, size=n)  # 贷款金额
savings_balance = np.random.lognormal(mean=4.0, sigma=1.0, size=n)  # 储蓄余额

# 就业类型
employment_type = np.random.choice(
    ['全职', '兼职', '自雇'],
    size=n,
    p=[0.7, 0.2, 0.1]
)

# 生成违约概率（逻辑函数）
logit = (
    0.0  # 基础截距（调整后，目标违约率约30%）
    - 0.012 * (credit_score - 650)  # 信用评分影响（负相关）
    + 4.5 * debt_ratio  # 负债率影响（正相关）
    + 0.008 * (age - 40)  # 年龄影响（正相关）
    - 0.003 * (months_employed / 12)  # 工作年限影响（负相关）
    + 0.15 * num_late_payments  # 逾期次数影响（正相关）
    - 0.0005 * income  # 收入影响（负相关）
    + 0.3 * (employment_type == '自雇').astype(int)  # 自雇影响（正相关）
    + np.random.normal(0, 0.4, n)  # 随机噪声
)

prob = 1 / (1 + np.exp(-logit))  # 转换为概率
default = (np.random.rand(n) < prob).astype(int)  # 生成违约标签

# 创建 DataFrame
df = pd.DataFrame({
    'age': age,
    'income': np.round(income, 2),
    'debt_ratio': np.round(debt_ratio, 4),
    'credit_score': np.round(credit_score, 0),
    'months_employed': months_employed,
    'num_credit_lines': num_credit_lines,
    'num_late_payments': num_late_payments,
    'loan_amount': np.round(loan_amount, 2),
    'savings_balance': np.round(savings_balance, 2),
    'employment_type': employment_type,
    'default': default
})

# 保存到 CSV
df.to_csv('credit_data.csv', index=False, encoding='utf-8-sig')

# 显示统计信息
print("数据生成完成！")
print(f"样本数量: {len(df)}")
print(f"违约比例: {df['default'].mean():.2%}")
print(f"\n特征统计:")
print(df.describe().round(2))
print(f"\n就业类型分布:")
print(df['employment_type'].value_counts())
print(f"\n前5行数据预览:")
print(df.head())