"""
FP-Growth 算法独立脚本
===================
对比 FP-Growth 与 Apriori 算法的性能，并生成关联规则

FP-Growth 优势：
- 只需扫描数据库2次（Apriori需要多次）
- 不生成候选项集，直接挖掘频繁项集
- 通常比 Apriori 快3-5倍
"""

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 数据路径
CSV_FILE = r"C:\Users\jefeer\Downloads\opencode\关联算法\data\customer_products_static.csv"


def load_and_encode_data():
    """加载数据并编码为布尔矩阵"""
    print("=" * 70)
    print("步骤1：加载客户数据")
    print("=" * 70)

    df = pd.read_csv(CSV_FILE)
    print(f"加载 {len(df)} 条客户记录")

    # 转换为交易列表
    transactions = df["持有产品"].apply(lambda x: x.split("|")).tolist()

    # 编码
    te = TransactionEncoder()
    df_encoded = pd.DataFrame(
        te.fit_transform(transactions), columns=te.columns_
    )

    print(f"产品种类：{len(te.columns_)}")
    print(f"列名：{list(te.columns_)}")

    return df, df_encoded


def analyze_product_penetration(df_encoded):
    """分析产品渗透率"""
    print("\n" + "=" * 70)
    print("步骤2：产品渗透率分析")
    print("=" * 70)

    product_support = df_encoded.sum() / len(df_encoded)
    product_support_sorted = product_support.sort_values(ascending=False)

    print("\n各产品渗透率：")
    for product, support in product_support_sorted.items():
        count = int(support * len(df_encoded))
        bar = "█" * int(support * 30)
        print(f"  {product:8s}: {support:5.1%} {bar} ({count}人)")


def mine_frequent_itemsets_fpgrowth(df_encoded, min_support=0.25):
    """使用 FP-Growth 挖掘频繁项集"""
    print("\n" + "=" * 70)
    print("步骤3：FP-Growth 挖掘频繁项集")
    print("=" * 70)
    print(f"最小支持度：{min_support}")

    start_time = time.time()
    frequent_itemsets = fpgrowth(
        df_encoded, min_support=min_support, use_colnames=True
    )
    elapsed = time.time() - start_time

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

    print(f"\nFP-Growth 耗时：{elapsed:.4f} 秒")
    print(f"发现频繁项集：{len(frequent_itemsets)} 个")

    print("\n频繁项集详情：")
    for length in sorted(frequent_itemsets["length"].unique()):
        subset = frequent_itemsets[frequent_itemsets["length"] == length].sort_values(
            "support", ascending=False
        )
        print(f"\n  {length}-项集 ({len(subset)} 个)：")
        for idx, row in subset.head(5).iterrows():
            items = ", ".join(list(row["itemsets"]))
            count = int(row["support"] * len(df_encoded))
            print(f"    - [{items}]: {row['support']:.1%} ({count}人)")

    return frequent_itemsets, elapsed


def mine_frequent_itemsets_apriori(df_encoded, min_support=0.25):
    """使用 Apriori 挖掘频繁项集（用于对比）"""
    print("\n" + "=" * 70)
    print("步骤4：Apriori 挖掘频繁项集（性能对比）")
    print("=" * 70)
    print(f"最小支持度：{min_support}")

    start_time = time.time()
    frequent_itemsets = apriori(
        df_encoded, min_support=min_support, use_colnames=True
    )
    elapsed = time.time() - start_time

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

    print(f"\nApriori 耗时：{elapsed:.4f} 秒")

    return frequent_itemsets, elapsed


def generate_association_rules(frequent_itemsets, min_confidence=0.6):
    """生成关联规则"""
    print("\n" + "=" * 70)
    print("步骤5：生成关联规则")
    print("=" * 70)
    print(f"最小置信度：{min_confidence}")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets),
    )

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

    print(f"生成关联规则：{len(rules)} 条")

    return rules


def filter_quality_rules(rules, min_lift=1.1, min_confidence=0.7):
    """筛选高质量规则"""
    print("\n" + "=" * 70)
    print("步骤6：筛选高质量规则")
    print("=" * 70)
    print(f"筛选条件：lift > {min_lift} 且 confidence > {min_confidence:.0%}")

    quality_rules = rules[
        (rules["lift"] > min_lift) & (rules["confidence"] > min_confidence)
    ]

    print(f"\n高质量规则数量：{len(quality_rules)} 条")

    if len(quality_rules) > 0:
        quality_rules_sorted = quality_rules.sort_values(
            ["lift", "confidence"], ascending=[False, False]
        )

        print("\n高质量规则详情（按提升度排序）：")
        for idx, row in quality_rules_sorted.head(15).iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            print(f"\n  {ant} → {con}")
            print(
                f"    支持度: {row['support']:.1%} | "
                f"置信度: {row['confidence']:.1%} | "
                f"提升度: {row['lift']:.2f}"
            )

    return quality_rules


def analyze_high_confidence_rules(rules):
    """分析高置信度规则"""
    print("\n" + "=" * 70)
    print("步骤7：高置信度规则分析")
    print("=" * 70)

    high_conf_rules = rules[rules["confidence"] > 0.75].sort_values(
        "confidence", ascending=False
    )

    if len(high_conf_rules) > 0:
        print(f"\n高置信度规则（confidence > 75%）：{len(high_conf_rules)} 条")
        for idx, row in high_conf_rules.head(8).iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            print(f"\n  {ant} → {con}")
            print(f"    置信度: {row['confidence']:.1%} | 提升度: {row['lift']:.2f}")

    return high_conf_rules


def performance_comparison(fpgrowth_time, apriori_time):
    """性能对比"""
    print("\n" + "=" * 70)
    print("步骤8：FP-Growth vs Apriori 性能对比")
    print("=" * 70)

    print(f"\n  FP-Growth 耗时：{fpgrowth_time:.4f} 秒")
    print(f"  Apriori 耗时：{apriori_time:.4f} 秒")

    if apriori_time > 0:
        speedup = apriori_time / fpgrowth_time
        print(f"\n  FP-Growth 比 Apriori 快：{speedup:.2f} 倍")

    if fpgrowth_time < 0.01:
        print("\n  注：数据量较小，两者差异不明显")
        print("  大规模数据（>10000条）时，FP-Growth 优势显著")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("       FP-Growth 关联规则分析")
    print("       对比 FP-Growth 与 Apriori 算法性能")
    print("=" * 70)

    # 1. 加载数据
    df, df_encoded = load_and_encode_data()

    # 2. 产品渗透率
    analyze_product_penetration(df_encoded)

    # 3. FP-Growth 挖掘
    fpgrowth_itemsets, fpgrowth_time = mine_frequent_itemsets_fpgrowth(df_encoded)

    # 4. Apriori 挖掘（对比用）
    apriori_itemsets, apriori_time = mine_frequent_itemsets_apriori(df_encoded)

    # 5. 生成关联规则
    rules = generate_association_rules(fpgrowth_itemsets)

    # 6. 筛选高质量规则
    quality_rules = filter_quality_rules(rules)

    # 7. 高置信度规则
    high_conf_rules = analyze_high_confidence_rules(rules)

    # 8. 性能对比
    performance_comparison(fpgrowth_time, apriori_time)

    # 完成
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    print(f"\n运行可视化脚本查看图表：")
    print(f"  python 关联算法/visualize_rules.py")

    return {
        "frequent_itemsets": fpgrowth_itemsets,
        "rules": rules,
        "quality_rules": quality_rules,
        "high_conf_rules": high_conf_rules,
        "fpgrowth_time": fpgrowth_time,
        "apriori_time": apriori_time,
    }


if __name__ == "__main__":
    main()
