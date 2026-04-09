from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

csv_file = r"C:\Users\jefeer\Downloads\opencode\关联算法\data\customer_products_static.csv"
df = pd.read_csv(csv_file)

print("=" * 80)
print("金融产品关联规则分析报告")
print("=" * 80)

print(f"\n数据概况：")
print(f"  - 总客户数：{len(df)} 人")
print(f"  - 年龄范围：{df['年龄'].min()} - {df['年龄'].max()} 岁")
print(f"  - 平均年龄：{df['年龄'].mean():.1f} 岁")
print(f"  - 平均持有产品数：{df['产品数量'].mean():.1f} 个")

print(f"\n收入等级分布：")
income_dist = df["收入等级"].value_counts()
for tier, count in income_dist.items():
    print(f"  - {tier}收入：{count} 人 ({count / len(df) * 100:.1f}%)")

print(f"\n产品数量分布：")
product_count_dist = df["产品数量"].value_counts().sort_index()
for count, num in product_count_dist.items():
    print(f"  - 持有{count}个产品：{num} 人 ({num / len(df) * 100:.1f}%)")

transactions = df["持有产品"].apply(lambda x: x.split("|")).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("\n" + "=" * 80)
print("步骤1：产品渗透率分析")
print("=" * 80)

product_support = df_encoded.sum() / len(df_encoded)
product_support_sorted = product_support.sort_values(ascending=False)

print("\n各产品市场渗透率：")
for product, support in product_support_sorted.items():
    count = int(support * len(df_encoded))
    print(f"  {product:8s}: {support:.1%} ({count} 人)")

print("\n" + "=" * 80)
print("步骤2：挖掘频繁项集 (min_support=0.25)")
print("=" * 80)

frequent_itemsets = apriori(df_encoded, min_support=0.25, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

print(f"\n发现频繁项集 {len(frequent_itemsets)} 个")
print("\n频繁项集详情（按长度和支持度排序）：")
for length in sorted(frequent_itemsets["length"].unique()):
    print(f"\n{length}-项集：")
    subset = frequent_itemsets[frequent_itemsets["length"] == length].sort_values(
        "support", ascending=False
    )
    for idx, row in subset.iterrows():
        items = ", ".join(list(row["itemsets"]))
        count = int(row["support"] * len(df_encoded))
        print(f"  - [{items}]: {row['support']:.1%} ({count} 人)")

print("\n" + "=" * 80)
print("步骤3：生成关联规则 (min_confidence=0.6)")
print("=" * 80)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6,
    num_itemsets=len(frequent_itemsets),
)

rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

print(f"\n生成关联规则 {len(rules)} 条")

print("\n" + "=" * 80)
print("步骤4：规则质量分析")
print("=" * 80)

print("\n【高质量规则】(lift > 1.1 & confidence > 0.7)：")
quality_rules = rules[(rules["lift"] > 1.1) & (rules["confidence"] > 0.7)]
if len(quality_rules) > 0:
    quality_rules_sorted = quality_rules.sort_values(
        ["lift", "confidence"], ascending=[False, False]
    )
    print(f"共 {len(quality_rules)} 条高质量规则\n")
    for idx, row in quality_rules_sorted.head(10).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        count = int(row["support"] * len(df_encoded))
        print(f"  {ant} -> {con}")
        print(
            f"    支持度: {row['support']:.1%} ({count} 人) | 置信度: {row['confidence']:.1%} | 提升度: {row['lift']:.2f}"
        )
else:
    print("未找到满足条件的规则")

print("\n【高置信度规则】(confidence > 0.75)：")
high_conf_rules = rules[rules["confidence"] > 0.75].sort_values(
    "confidence", ascending=False
)
if len(high_conf_rules) > 0:
    print(f"共 {len(high_conf_rules)} 条高置信度规则\n")
    for idx, row in high_conf_rules.head(8).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        count = int(row["support"] * len(df_encoded))
        print(f"  {ant} -> {con}")
        print(
            f"    置信度: {row['confidence']:.1%} | 支持度: {row['support']:.1%} ({count} 人) | 提升度: {row['lift']:.2f}"
        )

print("\n【高提升度规则】(lift > 1.15)：")
high_lift_rules = rules[rules["lift"] > 1.15].sort_values("lift", ascending=False)
if len(high_lift_rules) > 0:
    print(f"共 {len(high_lift_rules)} 条高提升度规则\n")
    for idx, row in high_lift_rules.head(8).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        print(f"  {ant} -> {con}")
        print(
            f"    提升度: {row['lift']:.2f} | 置信度: {row['confidence']:.1%} | 支持度: {row['support']:.1%}"
        )

print("\n" + "=" * 80)
print("步骤5：业务洞察与建议")
print("=" * 80)

print("\n【洞察1】核心产品分析")
print("-" * 80)
top_products = product_support_sorted.head(3)
print("最热门的3个产品：")
for product, support in top_products.items():
    print(f"  - {product}：渗透率 {support:.1%}，是核心基础产品")

print("\n【洞察2】交叉销售机会")
print("-" * 80)
if len(quality_rules) > 0:
    print("最佳交叉销售组合：")
    for idx, row in quality_rules_sorted.head(3).iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))
        print(f"  - 持有【{ant}】的客户，有 {row['confidence']:.1%} 概率购买【{con}】")
        print(f"    建议：在{ant}办理页面或APP中推荐{con}")
else:
    print("  建议提高最小支持度或置信度阈值以发现更强关联")

print("\n【洞察3】产品组合策略")
print("-" * 80)
combo_3plus = rules[rules["antecedent_len"] + rules["consequent_len"] >= 3]
if len(combo_3plus) > 0:
    print("推荐的产品组合套餐：")
    combo_3plus_sorted = combo_3plus.sort_values("support", ascending=False)
    for idx, row in combo_3plus_sorted.head(3).iterrows():
        products_list = list(row["antecedents"]) + list(row["consequents"])
        combo_str = " + ".join(products_list)
        count = int(row["support"] * len(df_encoded))
        print(f"  - {combo_str}（支持度 {row['support']:.1%}，{count} 人持有）")
else:
    print("  暂未发现3个产品以上的强关联组合")

print("\n【洞察4】客户分层建议")
print("-" * 80)
avg_products = df["产品数量"].mean()
high_value_threshold = int(avg_products) + 1
high_value_customers = df[df["产品数量"] >= high_value_threshold]
print(f"高价值客户（持有{high_value_threshold}个及以上产品）：")
print(
    f"  - 数量：{len(high_value_customers)} 人 ({len(high_value_customers) / len(df) * 100:.1f}%)"
)
print(f"  - 建议：重点维护，提供VIP服务和专属理财顾问")

low_value_customers = df[df["产品数量"] <= 2]
print(f"\n潜力客户（持有2个及以下产品）：")
print(
    f"  - 数量：{len(low_value_customers)} 人 ({len(low_value_customers) / len(df) * 100:.1f}%)"
)
print(f"  - 建议：交叉销售，提升产品渗透率")

print("\n" + "=" * 80)
print("步骤6：营销策略建议")
print("=" * 80)

print("\n【短期策略】（1-3个月）")
print("-" * 80)
if len(high_conf_rules) > 0:
    top_rule = high_conf_rules.iloc[0]
    ant = ", ".join(list(top_rule["antecedents"]))
    con = ", ".join(list(top_rule["consequents"]))
    print(f"1. 针对{ant}客户，推送{con}营销信息")
    print(f"   预期转化率：{top_rule['confidence']:.1%}")
else:
    print("1. 根据产品渗透率，针对低渗透率产品开展推广活动")

print("\n【中期策略】（3-6个月）")
print("-" * 80)
print("2. 设计产品组合优惠方案")
if len(quality_rules) > 0:
    for idx, row in quality_rules_sorted.head(2).iterrows():
        products_list = list(row["antecedents"]) + list(row["consequents"])
        combo_str = " + ".join(products_list)
        print(f"   - {combo_str} 组合优惠包")

print("\n【长期策略】（6-12个月）")
print("-" * 80)
print("3. 建立客户流失预警机制")
print("   - 监控单一产品客户，防止流失")
print("   - 对高价值客户提供个性化理财建议")
print("4. 开发智能推荐系统")
print("   - 基于关联规则自动推荐产品")
print("   - 实时计算客户价值评分")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print(f"\n数据文件：{csv_file}")
print(f"分析客户数：{len(df)} 人")
print(f"发现频繁项集：{len(frequent_itemsets)} 个")
print(f"生成关联规则：{len(rules)} 条")
print(f"高质量规则：{len(quality_rules)} 条")
