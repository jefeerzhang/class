"""
基金产品 K-Means 聚类分析脚本
结合聚类分析讲义知识点，对基金数据进行聚类分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(filepath):
    """加载基金数据"""
    print("=" * 60)
    print("第一步：数据加载")
    print("=" * 60)
    
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n数据预览:\n{df.head()}")
    
    return df


def prepare_features(df):
    """
    准备聚类特征
    选择6个数值特征：年化收益率、波动率、夏普比率、最大回撤、管理费率、基金规模
    """
    print("\n" + "=" * 60)
    print("第二步：特征选择")
    print("=" * 60)
    
    # 定义特征列
    feature_cols = ['年化收益率(%)', '波动率(%)', '夏普比率', '最大回撤(%)', '管理费率(%)', '基金规模(亿元)']
    
    X = df[feature_cols].copy()
    
    print(f"\n选择的特征: {feature_cols}")
    print(f"\n特征描述性统计:\n{X.describe()}")
    
    # 检查量纲差异（讲义1.6节：为什么需要标准化）
    print("\n【知识点：数据标准化必要性】")
    print("-" * 50)
    print("特征量纲差异分析（讲义1.6节）：")
    print(f"  基金规模范围: {X['基金规模(亿元)'].min():.2f} ~ {X['基金规模(亿元)'].max():.2f}（极差近万倍）")
    print(f"  年化收益率范围: {X['年化收益率(%)'].min():.2f} ~ {X['年化收益率(%)'].max():.2f}（个位数到几十）")
    print(f"  波动率范围: {X['波动率(%)'].min():.2f} ~ {X['波动率(%)'].max():.2f}")
    print("\n  ⚠️  如果不做标准化，'基金规模'将在距离计算中占据绝对主导地位！")
    print("  → 因此必须使用Z-Score标准化，让所有人站在同一起跑线上")
    
    return X, feature_cols


def standardize_data(X, feature_cols):
    """
    数据标准化（讲义1.6节）
    使用Z-Score标准化：x' = (x - μ) / σ
    转换后：均值为0，标准差为1
    """
    print("\n" + "=" * 60)
    print("第三步：数据标准化（Z-Score标准化）")
    print("=" * 60)
    print("\n【知识点：Z-Score标准化（讲义1.6节）】")
    print("-" * 50)
    print("公式: x' = (x - μ) / σ")
    print("  - μ 是特征均值")
    print("  - σ 是特征标准差")
    print("  - 转换后：均值为0，标准差为1")
    print("\n目的：消除量纲影响，让不同特征具有可比性")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print(f"\n标准化后特征描述性统计:\n{X_scaled_df.describe().round(4)}")
    
    # 对比标准化前后
    print("\n标准化前后对比（以基金规模为例）：")
    print(f"  标准化前: 均值={X['基金规模(亿元)'].mean():.2f}, 标准差={X['基金规模(亿元)'].std():.2f}")
    print(f"  标准化后: 均值={X_scaled_df['基金规模(亿元)'].mean():.4f}, 标准差={X_scaled_df['基金规模(亿元)'].std():.4f}")
    
    return X_scaled, scaler, X_scaled_df


def elbow_method(X_scaled, max_k=10):
    """
    肘部法则确定K值（讲义2.1.6节）
    原理：随着K增大，SSE必然减小。但当K增加到某个值后，SSE下降幅度显著变缓，
          这个转折点就是"肘部"，对应的K值就是最佳选择。
    """
    print("\n" + "=" * 60)
    print("第四步：K值选择 - 肘部法则")
    print("=" * 60)
    print("\n【知识点：肘部法则（讲义2.1.6节）】")
    print("-" * 50)
    print("核心思想：找一个'性价比最高'的K值")
    print("  - K值增加 → SSE必然减小（簇越多，每个簇越集中）")
    print("  - 但超过某个点后，SSE下降幅度显著变缓 → 这个转折点就是'肘部'")
    print("\n类比：切蛋糕")
    print("  - K=1：整个蛋糕一块，SSE很大")
    print("  - K=2：切成两半，SSE大幅下降")
    print("  - K=3：切成三块，SSE又下降一些")
    print("  - K=4+：SSE几乎不降了 → 停止")
    
    sse = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)  # inertia_就是SSE
    
    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('K值（簇数）', fontsize=12)
    plt.ylabel('SSE（簇内平方和）', fontsize=12)
    plt.title('肘部法则确定K值\n（讲义2.1.6节：肘部法则原理）', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 标记K=3的位置
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='建议K=3')
    plt.legend()
    plt.tight_layout()
    plt.savefig('C:\\Users\\jefeer\\Downloads\\opencode\\investment\\assets\\elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSSE值随K变化:")
    for k, s in zip(k_range, sse):
        indicator = " ← 肘部建议位置" if k == 3 else ""
        print(f"  K={k}: SSE={s:.2f}{indicator}")
    
    return sse


def silhouette_analysis(X_scaled, k_range=range(2, 11)):
    """
    轮廓系数分析（讲义2.1.6节）
    轮廓系数综合比较"到同簇的距离" vs "到最近其他簇的距离"
    取值范围[-1, 1]：接近1表示聚类效果好，接近-1表示聚类效果差
    """
    print("\n" + "=" * 60)
    print("第五步：K值验证 - 轮廓系数")
    print("=" * 60)
    print("\n【知识点：轮廓系数（讲义2.1.6节）】")
    print("-" * 50)
    print("轮廓系数回答两个问题：")
    print("  1. '我和自己簇的兄弟姐妹有多亲近？'（内部亲密度a）")
    print("  2. '我和隔壁簇的人有多疏远？'（外部疏远度b）")
    print("\n公式: s(i) = (b - a) / max(a, b)")
    print("  - s(i) ≈ 1：被正确分类（内部亲近，外部疏远）✓")
    print("  - s(i) ≈ 0：在两个簇边界上（两边都不亲不疏）")
    print("  - s(i) ≈ -1：可能被错误分类（跟外面更亲近）✗")
    
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  K={k}: 平均轮廓系数={score:.4f}")
    
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('K值（簇数）', fontsize=12)
    plt.ylabel('平均轮廓系数', fontsize=12)
    plt.title('轮廓系数选择K值\n（讲义2.1.6节：轮廓系数原理）', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 标记最大值
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'最佳K={best_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('C:\\Users\\jefeer\\Downloads\\opencode\\investment\\assets\\silhouette_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n→ 轮廓系数建议选择 K={best_k}（轮廓系数最大）")
    
    return silhouette_scores, best_k


def perform_kmeans(X_scaled, k=3):
    """
    执行K-Means聚类（讲义2.1节）
    K-Means核心思想："找中心、分阵营、再找中心、再分阵营...循环直到稳定"
    优化目标：最小化SSE（簇内平方和），即最大化簇内相似度
    """
    print("\n" + "=" * 60)
    print(f"第六步：K-Means聚类（K={k}）")
    print("=" * 60)
    print("\n【知识点：K-Means算法（讲义2.1节）】")
    print("-" * 50)
    print("核心思想：找中心、分阵营、循环直到稳定")
    print("\n算法步骤：")
    print("  1. 初始化：随机选择K个点作为初始质心")
    print("  2. 分配：每个点分配给距离最近的质心")
    print("  3. 更新：重新计算每个簇的质心（均值点）")
    print("  4. 迭代：重复2-3，直到质心不再变化")
    print("\n优化目标：最小化SSE（簇内平方和）")
    print("  SSE = Σ(点到所属簇中心的距离²)")
    print("  → 簇内点越集中，SSE越小，聚类效果越好")
    
    print("\n【知识点：聚类目标（讲义1.3节）】")
    print("-" * 50)
    print("聚类的核心目标：最大化簇内相似度，最小化簇间相似度")
    print("  - 簇内相似度高：同一簇内的点紧密聚集")
    print("  - 簇间相似度低：不同簇之间分离清楚")
    print("  - 理想状态：内紧外松")
    
    # 执行K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"\n聚类完成！")
    print(f"  最终SSE: {kmeans.inertia_:.4f}")
    print(f"  迭代次数: {kmeans.n_iter_}")
    
    # 统计各簇样本数
    print(f"\n各簇样本分布:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  簇{cluster_id}: {count}个样本")
    
    return kmeans, labels


def calculate_cluster_statistics(X, labels, feature_cols):
    """计算各簇的统计特征"""
    print("\n" + "=" * 60)
    print("第七步：各簇特征分析")
    print("=" * 60)
    
    X_with_labels = X.copy()
    X_with_labels['预测簇标签'] = labels
    
    # 各簇均值
    cluster_means = X_with_labels.groupby('预测簇标签')[feature_cols].mean()
    print("\n各簇特征均值:")
    print(cluster_means.round(4))
    
    # 各簇标准差
    cluster_stds = X_with_labels.groupby('预测簇标签')[feature_cols].std()
    print("\n各簇特征标准差:")
    print(cluster_stds.round(4))
    
    return X_with_labels, cluster_means, cluster_stds


def visualize_clusters_pca(X_scaled, labels, k):
    """使用PCA降维可视化聚类结果"""
    print("\n" + "=" * 60)
    print("第八步：聚类结果可视化")
    print("=" * 60)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 绘制散点图
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    cluster_names = ['稳健型', '平衡型', '进取型']
    
    for i in range(k):
        mask = labels == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=f'簇{i} ({cluster_names[i]})', 
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    plt.xlabel(f'第一主成分 (贡献率: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'第二主成分 (贡献率: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.title('K-Means聚类结果可视化（PCA降维）\n讲义1.3节：簇内相似 vs 簇间相异', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('C:\\Users\\jefeer\\Downloads\\opencode\\investment\\assets\\cluster_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA降维信息:")
    print(f"  第一主成分贡献率: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  第二主成分贡献率: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  累计贡献率: {sum(pca.explained_variance_ratio_):.2%}")
    
    return X_pca, pca


def create_radar_chart(cluster_means, feature_cols):
    """创建各簇特征雷达图"""
    # 标准化到0-1范围用于雷达图
    from sklearn.preprocessing import MinMaxScaler
    
    scaler_radar = MinMaxScaler()
    cluster_means_scaled = scaler_radar.fit_transform(cluster_means)
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cluster_names = ['稳健型', '平衡型', '进取型']
    
    for i in range(len(cluster_means)):
        values = cluster_means_scaled[i].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=f'簇{i} ({cluster_names[i]})', color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('各簇特征雷达图\n（归一化后的特征对比）', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('C:\\Users\\jefeer\\Downloads\\opencode\\investment\\assets\\cluster_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n雷达图已生成")


def calculate_silhouette_detail(X_scaled, labels):
    """计算详细的轮廓系数"""
    print("\n" + "=" * 60)
    print("第九步：轮廓系数详细分析")
    print("=" * 60)
    
    # 总体轮廓系数
    overall_score = silhouette_score(X_scaled, labels)
    print(f"\n整体轮廓系数: {overall_score:.4f}")
    
    # 各簇轮廓系数
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    
    print("\n各簇轮廓系数:")
    for i in range(len(np.unique(labels))):
        cluster_silhouette = sample_silhouette_values[labels == i]
        print(f"  簇{i}: 平均={cluster_silhouette.mean():.4f}, 最小={cluster_silhouette.min():.4f}, 最大={cluster_silhouette.max():.4f}")
    
    # 绘制轮廓系数分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_lower = 10
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cluster_names = ['稳健型', '平衡型', '进取型']
    
    for i in range(len(np.unique(labels))):
        cluster_silhouette = sample_silhouette_values[labels == i]
        cluster_silhouette.sort()
        
        size_cluster = len(cluster_silhouette)
        y_upper = y_lower + size_cluster
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                        facecolor=colors[i], alpha=0.7, edgecolor='white')
        ax.text(-0.05, y_lower + 0.5 * size_cluster, f'簇{i}')
        y_lower = y_upper + 10
    
    ax.set_xlabel('轮廓系数', fontsize=12)
    ax.set_ylabel('样本', fontsize=12)
    ax.set_title(f'各簇轮廓系数分布\n整体轮廓系数={overall_score:.4f}（讲义2.1.6节）', fontsize=14)
    ax.axvline(x=overall_score, color='r', linestyle='--', label=f'平均={overall_score:.3f}')
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('C:\\Users\\jefeer\\Downloads\\opencode\\investment\\assets\\silhouette_detail.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return overall_score


def compare_with_original_labels(df, labels):
    """与原始标签对比"""
    print("\n" + "=" * 60)
    print("第十步：与原始标签对比验证")
    print("=" * 60)
    
    if '簇标签' in df.columns:
        original_labels_str = df['簇标签'].values
        print("\n原始标签分布:")
        print(df['簇标签'].value_counts())
        
        # 将原始字符串标签转换为数值（提取簇号）
        label_map = {'簇1_稳健': 0, '簇2_平衡': 1, '簇3_进取': 2}
        original_labels = np.array([label_map.get(x, -1) for x in original_labels_str])
        
        print("\n预测标签 vs 原始标签交叉表:")
        comparison = pd.crosstab(labels, original_labels, rownames=['预测簇'], colnames=['原始簇'])
        print(comparison)
        
        # 计算匹配率（由于标签编号可能不同，需要找到最佳映射）
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(original_labels, labels)
        row_ind, col_ind = linear_sum_assignment(-cm)  # 最大化匹配
        match_count = cm[row_ind, col_ind].sum()
        accuracy = match_count / len(labels)
        
        print(f"\n与原始标签的匹配度: {accuracy:.2%}")
        
        return comparison
    else:
        print("数据中无原始标签列，跳过对比")
        return None


def save_results(df, X, labels, cluster_means, output_path):
    """保存聚类结果"""
    print("\n" + "=" * 60)
    print("第十一步：保存结果")
    print("=" * 60)
    
    # 添加聚类结果到原始数据
    result_df = df.copy()
    result_df['预测簇'] = labels
    
    # 映射到业务标签
    cluster_names = {0: '稳健型', 1: '平衡型', 2: '进取型'}
    result_df['预测簇名称'] = result_df['预测簇'].map(cluster_names)
    
    # 保存CSV
    result_path = output_path.replace('.csv', '_cluster_result.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"聚类结果已保存: {result_path}")
    
    # 保存各簇统计
    stats_path = output_path.replace('.csv', '_cluster_stats.csv')
    cluster_means.to_csv(stats_path, encoding='utf-8-sig')
    print(f"各簇统计已保存: {stats_path}")
    
    return result_df


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("基金产品 K-Means 聚类分析")
    print("结合《聚类分析讲义》知识点")
    print("=" * 80)
    
    # 设置路径
    data_path = 'C:\\Users\\jefeer\\Downloads\\opencode\\investment\\data\\聚类分析_金融数据演示.csv'
    output_path = 'C:\\Users\\jefeer\\Downloads\\opencode\\investment\\data\\cluster_result.csv'
    
    # 1. 加载数据
    df = load_data(data_path)
    
    # 2. 准备特征
    X, feature_cols = prepare_features(df)
    
    # 3. 标准化
    X_scaled, scaler, X_scaled_df = standardize_data(X, feature_cols)
    
    # 4. 肘部法则
    sse = elbow_method(X_scaled)
    
    # 5. 轮廓系数
    silhouette_scores, best_k = silhouette_analysis(X_scaled)
    
    # 6. K-Means聚类（使用K=3，与原始数据一致）
    kmeans, labels = perform_kmeans(X_scaled, k=3)
    
    # 7. 各簇统计
    X_with_labels, cluster_means, cluster_stds = calculate_cluster_statistics(X, labels, feature_cols)
    
    # 8. 可视化
    X_pca, pca = visualize_clusters_pca(X_scaled, labels, k=3)
    create_radar_chart(cluster_means, feature_cols)
    
    # 9. 轮廓系数详细分析
    overall_silhouette = calculate_silhouette_detail(X_scaled, labels)
    
    # 10. 与原始标签对比
    comparison = compare_with_original_labels(df, labels)
    
    # 11. 保存结果
    result_df = save_results(df, X, labels, cluster_means, output_path)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - investment/assets/elbow_method.png (肘部法则图)")
    print("  - investment/assets/silhouette_score.png (轮廓系数图)")
    print("  - investment/assets/silhouette_detail.png (轮廓系数分布图)")
    print("  - investment/assets/cluster_pca.png (聚类结果PCA可视化)")
    print("  - investment/assets/cluster_radar.png (各簇特征雷达图)")
    print("  - investment/data/cluster_result.csv (聚类结果)")
    print("  - investment/data/cluster_stats.csv (各簇统计)")
    
    return result_df, cluster_means, kmeans


if __name__ == '__main__':
    result_df, cluster_means, kmeans = main()
