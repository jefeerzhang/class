"""
关联规则可视化脚本
================
生成以下图表：
1. 产品渗透率饼图
2. 频繁项集支持度柱状图
3. 关联规则散点图（support vs confidence，颜色=lift）
4. 高质量规则网络图
"""

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pandas as pd
import sys
import io
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CSV_FILE = r"C:\Users\jefeer\Downloads\opencode\关联算法\data\customer_products_static.csv"
OUTPUT_DIR = r"C:\Users\jefeer\Downloads\opencode\关联算法\images"


def load_and_mine_rules(min_support=0.25, min_confidence=0.6):
    """加载数据并挖掘规则"""
    print("加载数据...")

    df = pd.read_csv(CSV_FILE)
    transactions = df["持有产品"].apply(lambda x: x.split("|")).tolist()

    te = TransactionEncoder()
    df_encoded = pd.DataFrame(
        te.fit_transform(transactions), columns=te.columns_
    )

    # 挖掘频繁项集和规则
    frequent_itemsets = fpgrowth(
        df_encoded, min_support=min_support, use_colnames=True
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets),
    )
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

    # 筛选高质量规则
    quality_rules = rules[(rules["lift"] > 1.1) & (rules["confidence"] > 0.7)]

    return df, df_encoded, frequent_itemsets, rules, quality_rules


def plot_product_penetration(df_encoded):
    """绘制产品渗透率饼图"""
    print("绘制产品渗透率饼图...")

    product_support = df_encoded.sum() / len(df_encoded)
    product_support_sorted = product_support.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set3(range(len(product_support_sorted)))
    wedges, texts, autotexts = ax.pie(
        product_support_sorted.values,
        labels=product_support_sorted.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75,
    )

    ax.set_title('金融产品渗透率分布', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/product_penetration.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存至：{OUTPUT_DIR}/product_penetration.png")


def plot_frequent_itemsets(frequent_itemsets):
    """绘制频繁项集支持度柱状图"""
    print("绘制频繁项集柱状图...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # 按长度分组显示
    for length in sorted(frequent_itemsets["length"].unique()):
        subset = frequent_itemsets[frequent_itemsets["length"] == length].sort_values(
            "support", ascending=True
        )

        labels = [", ".join(list(x)) for x in subset["itemsets"]]
        values = subset["support"].values

        # 横向柱状图
        y_pos = range(len(labels))
        bars = ax.barh(
            [f"{length}项集: {l}" for l in labels],
            values,
            height=0.6,
            label=f'{length}-项集',
        )

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.1%}', va='center', fontsize=8)

    ax.set_xlabel('支持度 (Support)', fontsize=11)
    ax.set_title('频繁项集支持度分布', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(frequent_itemsets["support"]) * 1.15)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/frequent_itemsets.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存至：{OUTPUT_DIR}/frequent_itemsets.png")


def plot_rules_scatter(rules):
    """绘制关联规则散点图"""
    print("绘制关联规则散点图...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # 全部规则
    scatter = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        s=100,
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('提升度 (Lift)', fontsize=11)

    # 标注高质量规则
    quality_rules = rules[(rules["lift"] > 1.1) & (rules["confidence"] > 0.7)]
    if len(quality_rules) > 0:
        for idx, row in quality_rules.head(5).iterrows():
            ant = ", ".join(list(row["antecedents"])[:2])
            con = ", ".join(list(row["consequents"])[:2])
            ax.annotate(
                f"{ant}→{con}",
                (row["support"], row["confidence"]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                alpha=0.8,
            )

    ax.set_xlabel('支持度 (Support)', fontsize=11)
    ax.set_ylabel('置信度 (Confidence)', fontsize=11)
    ax.set_title('关联规则分布 (颜色=提升度)', fontsize=16, fontweight='bold')

    # 添加参考线
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='confidence=0.7')
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='support=0.25')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rules_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存至：{OUTPUT_DIR}/rules_scatter.png")


def plot_rules_network(quality_rules, top_n=10):
    """绘制高质量规则网络图"""
    print("绘制规则网络图...")

    if len(quality_rules) < 1:
        print("  警告：高质量规则数量不足，跳过网络图")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    G = nx.DiGraph()

    # 取 top_n 规则
    top_rules = quality_rules.nlargest(top_n, 'lift')

    # 添加节点和边
    for idx, row in top_rules.iterrows():
        ant = ", ".join(list(row["antecedents"]))
        con = ", ".join(list(row["consequents"]))

        G.add_node(ant, node_type='antecedent')
        G.add_node(con, node_type='consequent')
        G.add_edge(ant, con, weight=row['lift'], confidence=row['confidence'])

    # 设置布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 绘制节点
    node_colors = ['#FF6B6B' if G.nodes[n].get('node_type') == 'antecedent' else '#4ECDC4'
                   for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    edge_widths = [w * 2 for w in weights]

    nx.draw_networkx_edges(
        G, pos,
        edge_color='#888888',
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )

    # 添加边标签（lift值）
    edge_labels = {(u, v): f"L={G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    ax.set_title(f'高质量关联规则网络图 (Top {top_n})', fontsize=16, fontweight='bold')
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=12, label='前置条件'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=12, label='推荐结果'),
    ], loc='upper left')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rules_network.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存至：{OUTPUT_DIR}/rules_network.png")


def plot_lift_distribution(rules):
    """绘制提升度分布直方图"""
    print("绘制提升度分布图...")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(rules["lift"], bins=30, color='#5DADE2', edgecolor='white', alpha=0.8)

    # 标注分位数
    lift_75 = rules["lift"].quantile(0.75)
    lift_90 = rules["lift"].quantile(0.90)

    ax.axvline(x=lift_75, color='orange', linestyle='--', linewidth=2, label=f'75%分位: {lift_75:.2f}')
    ax.axvline(x=lift_90, color='red', linestyle='--', linewidth=2, label=f'90%分位: {lift_90:.2f}')

    ax.set_xlabel('提升度 (Lift)', fontsize=11)
    ax.set_ylabel('规则数量', fontsize=11)
    ax.set_title('提升度分布', fontsize=16, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lift_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存至：{OUTPUT_DIR}/lift_distribution.png")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("关联规则可视化")
    print("=" * 60)

    # 加载数据并挖掘规则
    df, df_encoded, frequent_itemsets, rules, quality_rules = load_and_mine_rules()

    print(f"\n数据概览：")
    print(f"  - 客户数量：{len(df)}")
    print(f"  - 频繁项集：{len(frequent_itemsets)} 个")
    print(f"  - 关联规则：{len(rules)} 条")
    print(f"  - 高质量规则：{len(quality_rules)} 条")

    # 生成图表
    print("\n生成图表...")
    print("-" * 40)

    plot_product_penetration(df_encoded)
    plot_frequent_itemsets(frequent_itemsets)
    plot_rules_scatter(rules)
    plot_rules_network(quality_rules)
    plot_lift_distribution(rules)

    print("\n" + "=" * 60)
    print("可视化完成！所有图表已保存至：")
    print(f"  {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
