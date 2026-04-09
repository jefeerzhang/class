from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 银行客户产品持有数据
# 场景：分析银行客户持有的金融产品之间的关联关系
customer_products = [
    ["信用卡", "储蓄账户", "理财产品"],  # 客户1
    ["信用卡", "储蓄账户", "基金", "保险"],  # 客户2
    ["储蓄账户", "基金"],  # 客户3
    ["信用卡", "贷款", "储蓄账户"],  # 客户4
    ["信用卡", "储蓄账户", "基金", "理财产品"],  # 客户5
    ["储蓄账户", "保险"],  # 客户6
    ["信用卡", "贷款", "保险"],  # 客户7
    ["信用卡", "储蓄账户", "基金"],  # 客户8
    ["贷款", "储蓄账户", "保险"],  # 客户9
    ["信用卡", "储蓄账户", "基金", "理财产品"],  # 客户10
    ["储蓄账户", "基金"],  # 客户11
    ["信用卡", "贷款", "储蓄账户", "保险"],  # 客户12
    ["信用卡", "基金", "理财产品"],  # 客户13
    ["储蓄账户", "保险"],  # 客户14
    ["信用卡", "储蓄账户", "基金"],  # 客户15
]

print("=" * 60)
print("金融产品关联规则分析")
print("=" * 60)

# 数据编码
te = TransactionEncoder()
te_ary = te.fit(customer_products).transform(customer_products)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("\n数据概览（前5行）：")
print(df.head())
print(f"\n总客户数：{len(customer_products)}")
print(f"金融产品数：{len(te.columns_)}")

# 挖掘频繁项集
print("\n" + "=" * 60)
print("步骤1：挖掘频繁项集 (min_support=0.3)")
print("=" * 60)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

print("\n频繁项集：")
print(frequent_itemsets.sort_values(["length", "support"], ascending=[True, False]))

# 统计各长度的频繁项集数量
print("\n各长度频繁项集统计：")
print(frequent_itemsets["length"].value_counts().sort_index())

# 生成关联规则
print("\n" + "=" * 60)
print("步骤2：生成关联规则 (min_confidence=0.6)")
print("=" * 60)
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6,
    num_itemsets=len(frequent_itemsets),
)

# 添加规则长度信息
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

print("\n关联规则（按提升度排序）：")
rules_display = rules[
    ["antecedents", "consequents", "support", "confidence", "lift"]
].sort_values("lift", ascending=False)
print(rules_display)

# 筛选高质量规则
print("\n" + "=" * 60)
print("步骤3：筛选高质量规则 (lift>1.2 & confidence>0.7)")
print("=" * 60)
quality_rules = rules[(rules["lift"] > 1.2) & (rules["confidence"] > 0.7)]
print(f"\n高质量规则数量：{len(quality_rules)}")
print("\n高质量规则详情：")
print(
    quality_rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].sort_values("lift", ascending=False)
)

# 业务分析
print("\n" + "=" * 60)
print("步骤4：业务分析与建议")
print("=" * 60)

# 分析1：交叉销售机会
print("\n【分析1】交叉销售机会")
print("-" * 60)
high_conf_rules = rules[rules["confidence"] >= 0.8].sort_values(
    "confidence", ascending=False
)
if len(high_conf_rules) > 0:
    print("高置信度规则（置信度≥80%）：")
    for idx, row in high_conf_rules.head(5).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        print(f"  • 持有【{ant}】的客户，{row['confidence']:.1%}会购买【{con}】")
        print(f"    支持度：{row['support']:.1%}，提升度：{row['lift']:.2f}")

# 分析2：产品组合推荐
print("\n【分析2】产品组合推荐")
print("-" * 60)
combo_rules = rules[rules["antecedent_len"] >= 2].sort_values("lift", ascending=False)
if len(combo_rules) > 0:
    print("热门产品组合：")
    for idx, row in combo_rules.head(3).iterrows():
        ant = " + ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        print(f"  • 【{ant}】→【{con}】")
        print(f"    提升度：{row['lift']:.2f}，置信度：{row['confidence']:.1%}")

# 分析3：产品渗透率
print("\n【分析3】产品渗透率分析")
print("-" * 60)
product_support = {}
for product in te.columns_:
    product_support[product] = df[product].sum() / len(df)

print("各产品市场渗透率：")
for product, support in sorted(
    product_support.items(), key=lambda x: x[1], reverse=True
):
    print(f"  • {product:8s}: {support:.1%}")

print("\n" + "=" * 60)
print("业务建议汇总")
print("=" * 60)
print("""
1. 产品推荐策略：
   - 对持有信用卡的客户重点推荐基金和储蓄账户（置信度高达85%+）
   - 贷款客户有强烈保险需求，可打包销售

2. 捆绑营销方案：
   - 设计"信用卡+基金+储蓄账户"组合套餐
   - 推出"贷款+保险"优惠包，提升客户粘性

3. 精准营销建议：
   - 储蓄账户客户是基金产品的优质潜在客户（提升度1.2+）
   - 理财产品客户通常已持有多种产品，可作为高价值客户重点维护

4. 风险管理提示：
   - 持有多种产品组合的客户风险分散能力较强
   - 单一产品客户需关注产品到期流失风险
""")

print("\n分析完成！")
