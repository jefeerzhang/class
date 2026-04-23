"""
经典决策树算法演示：ID3、C4.5、CART
展示三大算法的核心原理与实现差异
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import math
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================
# ID3算法实现（信息增益）
# ============================================

class ID3DecisionTree:
    """
    ID3决策树算法实现
    使用信息增益作为分裂准则
    仅支持类别特征，构建多叉树
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def entropy(self, y):
        """计算信息熵 H(D)"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def information_gain(self, X, y, feature_idx):
        """计算信息增益 IG(D, A)"""
        total_entropy = self.entropy(y)
        
        # 获取特征的唯一值
        feature_values = set(X[:, feature_idx])
        
        # 计算条件熵
        conditional_entropy = 0
        for value in feature_values:
            subset_y = y[X[:, feature_idx] == value]
            weight = len(subset_y) / len(y)
            conditional_entropy += weight * self.entropy(subset_y)
        
        return total_entropy - conditional_entropy
    
    def choose_best_feature(self, X, y, available_features):
        """选择信息增益最大的特征"""
        best_gain = -1
        best_feature = None
        
        for feature_idx in available_features:
            gain = self.information_gain(X, y, feature_idx)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
        
        return best_feature, best_gain
    
    def majority_vote(self, y):
        """多数投票确定类别"""
        counts = Counter(y)
        return counts.most_common(1)[0][0]
    
    def build_tree(self, X, y, available_features, depth=0):
        """递归构建决策树"""
        # 停止条件1：所有样本属于同一类
        if len(set(y)) == 1:
            return {'leaf': True, 'class': y[0]}
        
        # 停止条件2：没有可用特征或样本数太少
        if len(available_features) == 0 or len(y) < self.min_samples_split:
            return {'leaf': True, 'class': self.majority_vote(y)}
        
        # 停止条件3：达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return {'leaf': True, 'class': self.majority_vote(y)}
        
        # 选择最佳分裂特征
        best_feature, best_gain = self.choose_best_feature(X, y, available_features)
        
        if best_feature is None or best_gain <= 0:
            return {'leaf': True, 'class': self.majority_vote(y)}
        
        # 构建分支
        tree = {
            'leaf': False,
            'feature': best_feature,
            'gain': best_gain,
            'children': {}
        }
        
        feature_values = set(X[:, best_feature])
        remaining_features = [f for f in available_features if f != best_feature]
        
        for value in feature_values:
            subset_X = X[X[:, best_feature] == value]
            subset_y = y[X[:, best_feature] == value]
            
            if len(subset_y) == 0:
                tree['children'][value] = {'leaf': True, 'class': self.majority_vote(y)}
            else:
                tree['children'][value] = self.build_tree(
                    subset_X, subset_y, remaining_features, depth + 1
                )
        
        return tree
    
    def fit(self, X, y):
        """训练模型"""
        available_features = list(range(X.shape[1]))
        self.tree = self.build_tree(X, y, available_features)
        return self
    
    def predict_one(self, x, tree):
        """预测单个样本"""
        if tree['leaf']:
            return tree['class']
        
        feature_value = x[tree['feature']]
        if feature_value in tree['children']:
            return self.predict_one(x, tree['children'][feature_value])
        else:
            # 未见过的特征值，返回多数类
            return self.majority_vote([tree['children'][c]['class'] 
                                      for c in tree['children'] 
                                      if tree['children'][c]['leaf']])
    
    def predict(self, X):
        """预测多个样本"""
        return [self.predict_one(x, self.tree) for x in X]
    
    def print_tree(self, tree=None, depth=0, feature_names=None):
        """打印树结构"""
        if tree is None:
            tree = self.tree
        
        indent = "  " * depth
        
        if tree['leaf']:
            print(f"{indent}→ 叶节点: 类别 {tree['class']}")
        else:
            feature_name = feature_names[tree['feature']] if feature_names else f"特征{tree['feature']}"
            print(f"{indent}特征 {feature_name} (增益={tree['gain']:.4f})")
            for value, child in tree['children'].items():
                print(f"{indent}  分支 [{value}]:")
                self.print_tree(child, depth + 2, feature_names)


# ============================================
# C4.5算法实现（信息增益率）
# ============================================

class C45DecisionTree:
    """
    C4.5决策树算法实现
    使用信息增益率作为分裂准则
    支持类别特征和连续特征
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.continuous_features = []  # 记录连续特征
        self.split_points = {}         # 记录分裂点
        
    def entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def intrinsic_value(self, X, feature_idx):
        """计算固有值 IV(A)"""
        counts = Counter(X[:, feature_idx])
        probs = [count / len(X) for count in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)
    
    def information_gain(self, X, y, feature_idx):
        """计算信息增益"""
        total_entropy = self.entropy(y)
        feature_values = set(X[:, feature_idx])
        
        conditional_entropy = 0
        for value in feature_values:
            subset_y = y[X[:, feature_idx] == value]
            weight = len(subset_y) / len(y)
            conditional_entropy += weight * self.entropy(subset_y)
        
        return total_entropy - conditional_entropy
    
    def gain_ratio(self, X, y, feature_idx):
        """计算信息增益率"""
        ig = self.information_gain(X, y, feature_idx)
        iv = self.intrinsic_value(X, feature_idx)
        return ig / iv if iv > 0 else 0
    
    def find_best_split_point(self, X, y, feature_idx):
        """为连续特征找最优分裂点"""
        # 获取特征值并排序
        values = X[:, feature_idx]
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_y = y[sorted_indices]
        
        # 候选分裂点
        unique_values = np.unique(sorted_values)
        if len(unique_values) <= 1:
            return None, 0
        
        split_points = (unique_values[:-1] + unique_values[1:]) / 2
        
        best_gain = -1
        best_point = None
        
        for point in split_points:
            left_y = sorted_y[sorted_values <= point]
            right_y = sorted_y[sorted_values > point]
            
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            
            # 计算信息增益
            left_entropy = self.entropy(left_y)
            right_entropy = self.entropy(right_y)
            
            weighted_entropy = (len(left_y) / len(y)) * left_entropy + \
                              (len(right_y) / len(y)) * right_entropy
            
            gain = self.entropy(y) - weighted_entropy
            
            if gain > best_gain:
                best_gain = gain
                best_point = point
        
        return best_point, best_gain
    
    def choose_best_feature(self, X, y, available_features, is_continuous=None):
        """选择信息增益率最大的特征"""
        # 首先计算所有特征的信息增益
        gains = []
        for feature_idx in available_features:
            if is_continuous and is_continuous[feature_idx]:
                point, gain = self.find_best_split_point(X, y, feature_idx)
                gains.append((feature_idx, gain, point, True))
            else:
                gain = self.information_gain(X, y, feature_idx)
                gains.append((feature_idx, gain, None, False))
        
        # 计算平均信息增益
        avg_gain = sum(g[1] for g in gains) / len(gains) if gains else 0
        
        # 只考虑信息增益高于平均的特征
        candidates = [g for g in gains if g[1] >= avg_gain]
        
        if not candidates:
            candidates = gains
        
        # 在候选中选择增益率最高的
        best_ratio = -1
        best_feature = None
        best_point = None
        is_cont = False
        
        for feature_idx, gain, point, continuous in candidates:
            if continuous:
                # 对于连续特征，使用二分后的增益率
                if point is not None:
                    # 创建临时二元特征
                    temp_X = (X[:, feature_idx] <= point).astype(int)
                    iv = self.intrinsic_value(temp_X.reshape(-1, 1), 0)
                    ratio = gain / iv if iv > 0 else 0
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_feature = feature_idx
                        best_point = point
                        is_cont = True
            else:
                iv = self.intrinsic_value(X, feature_idx)
                ratio = gain / iv if iv > 0 else 0
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_feature = feature_idx
                    best_point = None
                    is_cont = False
        
        return best_feature, best_ratio, best_point, is_cont
    
    def majority_vote(self, y):
        """多数投票"""
        counts = Counter(y)
        return counts.most_common(1)[0][0]
    
    def build_tree(self, X, y, available_features, is_continuous, depth=0):
        """构建决策树"""
        # 停止条件
        if len(set(y)) == 1:
            return {'leaf': True, 'class': y[0]}
        
        if len(available_features) == 0 or len(y) < self.min_samples_split:
            return {'leaf': True, 'class': self.majority_vote(y)}
        
        if self.max_depth is not None and depth >= self.max_depth:
            return {'leaf': True, 'class': self.majority_vote(y)}
        
        # 选择最佳特征
        best_feature, best_ratio, split_point, is_cont = self.choose_best_feature(
            X, y, available_features, is_continuous
        )
        
        if best_feature is None:
            return {'leaf': True, 'class': self.majority_vote(y)}
        
        # 记录分裂点
        if is_cont:
            self.split_points[(depth, best_feature)] = split_point
        
        # 构建分支
        tree = {
            'leaf': False,
            'feature': best_feature,
            'ratio': best_ratio,
            'split_point': split_point,
            'is_continuous': is_cont,
            'children': {}
        }
        
        if is_cont:
            # 连续特征：二分
            left_mask = X[:, best_feature] <= split_point
            right_mask = X[:, best_feature] > split_point
            
            remaining_features = [f for f in available_features if f != best_feature]
            
            if len(y[left_mask]) > 0:
                tree['children']['<='] = self.build_tree(
                    X[left_mask], y[left_mask], remaining_features, is_continuous, depth + 1
                )
            else:
                tree['children']['<='] = {'leaf': True, 'class': self.majority_vote(y)}
            
            if len(y[right_mask]) > 0:
                tree['children']['>'] = self.build_tree(
                    X[right_mask], y[right_mask], remaining_features, is_continuous, depth + 1
                )
            else:
                tree['children']['>'] = {'leaf': True, 'class': self.majority_vote(y)}
        else:
            # 类别特征：多分
            feature_values = set(X[:, best_feature])
            remaining_features = [f for f in available_features if f != best_feature]
            
            for value in feature_values:
                mask = X[:, best_feature] == value
                if len(y[mask]) > 0:
                    tree['children'][value] = self.build_tree(
                        X[mask], y[mask], remaining_features, is_continuous, depth + 1
                    )
                else:
                    tree['children'][value] = {'leaf': True, 'class': self.majority_vote(y)}
        
        return tree
    
    def fit(self, X, y, continuous_features=None):
        """训练模型"""
        available_features = list(range(X.shape[1]))
        
        # 标记连续特征
        is_continuous = [False] * X.shape[1]
        if continuous_features:
            for idx in continuous_features:
                is_continuous[idx] = True
        
        self.tree = self.build_tree(X, y, available_features, is_continuous)
        return self
    
    def predict_one(self, x, tree):
        """预测单个样本"""
        if tree['leaf']:
            return tree['class']
        
        if tree['is_continuous']:
            if x[tree['feature']] <= tree['split_point']:
                return self.predict_one(x, tree['children']['<='])
            else:
                return self.predict_one(x, tree['children']['>'])
        else:
            feature_value = x[tree['feature']]
            if feature_value in tree['children']:
                return self.predict_one(x, tree['children'][feature_value])
            else:
                return self.majority_vote([tree['children'][c]['class'] 
                                          for c in tree['children']])
    
    def predict(self, X):
        """预测"""
        return [self.predict_one(x, self.tree) for x in X]


# ============================================
# CART算法（使用sklearn实现）
# ============================================

def cart_demo(X, y, feature_names, class_names):
    """
    CART算法演示（使用sklearn）
    """
    print("\n" + "=" * 60)
    print("CART算法演示（基尼系数）")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # CART分类树
    cart = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    cart.fit(X_train, y_train)
    
    y_pred = cart.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nCART准确率: {accuracy:.4f}")
    print(f"树深度: {cart.get_depth()}")
    print(f"叶节点数: {cart.get_n_leaves()}")
    
    # 特征重要性（基于基尼系数）
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': cart.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性（基尼减少量）:")
    print(importance)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    plot_tree(cart, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True)
    plt.title('CART决策树（基尼系数）')
    plt.tight_layout()
    plt.savefig('cart_tree.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return cart


# ============================================
# 算法对比演示
# ============================================

def compare_algorithms():
    """
    对比ID3、C4.5和CART算法
    """
    print("=" * 60)
    print("经典决策树算法对比演示")
    print("=" * 60)
    
    # 使用鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # 为ID3准备离散化数据
    from sklearn.preprocessing import KBinsDiscretizer
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_discrete = discretizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train_discrete = discretizer.transform(X_train)
    X_test_discrete = discretizer.transform(X_test)
    
    print(f"\n数据集信息:")
    print(f"样本数: {len(X)}")
    print(f"特征数: {X.shape[1]}")
    print(f"类别数: {len(class_names)}")
    print(f"特征: {feature_names}")
    
    # 1. ID3算法
    print("\n--- ID3算法 ---")
    id3 = ID3DecisionTree(max_depth=3)
    id3.fit(X_train_discrete, y_train)
    
    y_pred_id3 = id3.predict(X_test_discrete)
    acc_id3 = accuracy_score(y_test, y_pred_id3)
    
    print(f"\nID3准确率: {acc_id3:.4f}")
    print("\nID3树结构:")
    id3.print_tree(feature_names=[f"{f}_离散" for f in feature_names])
    
    # 2. C4.5算法
    print("\n--- C4.5算法 ---")
    c45 = C45DecisionTree(max_depth=3)
    c45.fit(X_train, y_train, continuous_features=list(range(4)))  # 所有特征都是连续的
    
    y_pred_c45 = c45.predict(X_test)
    acc_c45 = accuracy_score(y_test, y_pred_c45)
    
    print(f"\nC4.5准确率: {acc_c45:.4f}")
    
    # 3. CART算法
    print("\n--- CART算法 ---")
    cart = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    cart.fit(X_train, y_train)
    
    y_pred_cart = cart.predict(X_test)
    acc_cart = accuracy_score(y_test, y_pred_cart)
    
    print(f"\nCART准确率: {acc_cart:.4f}")
    print(f"树深度: {cart.get_depth()}")
    print(f"叶节点数: {cart.get_n_leaves()}")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("算法对比总结")
    print("=" * 60)
    
    results = pd.DataFrame({
        '算法': ['ID3', 'C4.5', 'CART'],
        '分裂准则': ['信息增益', '信息增益率', '基尼系数'],
        '准确率': [acc_id3, acc_c45, acc_cart],
        '树结构': ['多叉树', '多叉树', '二叉树'],
        '支持连续特征': ['否（需离散化）', '是', '是'],
        'sklearn支持': ['否', '否', '是']
    })
    
    print(results)
    
    # 可视化对比
    plt.figure(figsize=(10, 6))
    
    algorithms = ['ID3', 'C4.5', 'CART']
    accuracies = [acc_id3, acc_c45, acc_cart]
    
    bars = plt.bar(algorithms, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.ylabel('准确率')
    plt.title('经典决策树算法准确率对比')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def demo_split_criteria():
    """
    演示不同分裂准则的计算过程
    """
    print("\n" + "=" * 60)
    print("分裂准则计算演示")
    print("=" * 60)
    
    # 创建简单示例数据
    data = pd.DataFrame({
        '天气': ['晴', '晴', '阴', '雨', '雨', '雨', '阴', '晴', '晴', '雨'],
        '温度': ['热', '热', '热', '适中', '冷', '冷', '冷', '适中', '冷', '适中'],
        '活动': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是']
    })
    
    print("\n示例数据:")
    print(data)
    
    y = data['活动'].values
    
    # 计算根节点熵
    id3 = ID3DecisionTree()
    root_entropy = id3.entropy(y)
    print(f"\n根节点熵 H(D): {root_entropy:.4f}")
    
    # 计算各特征的信息增益
    features = ['天气', '温度']
    for feature in features:
        X_feature = data[feature].values.reshape(-1, 1)
        
        ig = id3.information_gain(X_feature, y, 0)
        iv = id3.intrinsic_value(X_feature, 0)
        gr = ig / iv if iv > 0 else 0
        
        print(f"\n特征 '{feature}':")
        print(f"  信息增益: {ig:.4f}")
        print(f"  固有值: {iv:.4f}")
        print(f"  信息增益率: {gr:.4f}")
        
        # 详细计算过程
        print(f"  详细计算:")
        for value in set(X_feature):
            subset = y[X_feature.flatten() == value]
            subset_entropy = id3.entropy(subset)
            weight = len(subset) / len(y)
            print(f"    {feature}={value}: 熵={subset_entropy:.4f}, 权重={weight:.2f}")
    
    return data


def demo_gini_vs_entropy():
    """
    对比基尼系数和信息熵在不同分布下的表现
    """
    print("\n" + "=" * 60)
    print("基尼系数 vs 信息熵对比")
    print("=" * 60)
    
    # 不同类别分布
    distributions = [
        ([1, 0], "纯节点（100%正类）"),
        ([0.9, 0.1], "90%正类, 10%负类"),
        ([0.8, 0.2], "80%正类, 20%负类"),
        ([0.7, 0.3], "70%正类, 30%负类"),
        ([0.6, 0.4], "60%正类, 40%负类"),
        ([0.5, 0.5], "均匀分布"),
    ]
    
    results = []
    
    for probs, description in distributions:
        # 信息熵
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        # 基尼系数
        gini = 1 - sum(p ** 2 for p in probs)
        
        results.append({
            '分布': description,
            '信息熵': entropy,
            '基尼系数': gini
        })
        
        print(f"{description}:")
        print(f"  信息熵: {entropy:.4f}")
        print(f"  基尼系数: {gini:.4f}")
    
    # 可视化
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 信息熵曲线
    axes[0].plot(df['信息熵'], 'o-', label='信息熵')
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df['分布'], rotation=45, ha='right')
    axes[0].set_ylabel('信息熵')
    axes[0].set_title('不同分布下的信息熵')
    axes[0].grid(True, alpha=0.3)
    
    # 基尼系数曲线
    axes[1].plot(df['基尼系数'], 's-', label='基尼系数', color='orange')
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels(df['分布'], rotation=45, ha='right')
    axes[1].set_ylabel('基尼系数')
    axes[1].set_title('不同分布下的基尼系数')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gini_entropy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df


def demo_continuous_feature_split():
    """
    演示连续特征的分裂点选择
    """
    print("\n" + "=" * 60)
    print("连续特征分裂点选择演示")
    print("=" * 60)
    
    # 创建示例数据
    data = pd.DataFrame({
        '年龄': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        '购买': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  # 0:不购买, 1:购买
    })
    
    print("\n示例数据:")
    print(data)
    
    X = data['年龄'].values.reshape(-1, 1)
    y = data['购买'].values
    
    # C4.5分裂点选择
    c45 = C45DecisionTree()
    best_point, best_gain = c45.find_best_split_point(X, y, 0)
    
    print(f"\n最优分裂点: {best_point}")
    print(f"信息增益: {best_gain:.4f}")
    
    # 展示所有候选分裂点的信息增益
    print("\n所有候选分裂点:")
    unique_values = np.unique(X.flatten())
    split_points = (unique_values[:-1] + unique_values[1:]) / 2
    
    for point in split_points:
        left_y = y[X.flatten() <= point]
        right_y = y[X.flatten() > point]
        
        if len(left_y) > 0 and len(right_y) > 0:
            left_entropy = c45.entropy(left_y)
            right_entropy = c45.entropy(right_y)
            
            weighted_entropy = (len(left_y) / len(y)) * left_entropy + \
                              (len(right_y) / len(y)) * right_entropy
            
            gain = c45.entropy(y) - weighted_entropy
            
            print(f"  分裂点={point:.1f}: 左熵={left_entropy:.4f}, 右熵={right_entropy:.4f}, "
                  f"增益={gain:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制数据点
    colors = ['red' if v == 0 else 'blue' for v in y]
    plt.scatter(data['年龄'], data['购买'], c=colors, s=100)
    
    # 绘制分裂点
    plt.axvline(best_point, color='green', linestyle='--', linewidth=2,
                label=f'最优分裂点 ({best_point:.1f})')
    
    plt.xlabel('年龄')
    plt.ylabel('购买意愿')
    plt.title('连续特征分裂点选择')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('continuous_split.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return data


def main():
    """
    主函数
    """
    print("经典决策树算法完整演示")
    print("ID3、C4.5、CART三大算法原理与实现")
    print("=" * 60)
    
    # 1. 分裂准则对比
    demo_gini_vs_entropy()
    
    # 2. 分裂准则计算过程
    demo_split_criteria()
    
    # 3. 连续特征分裂
    demo_continuous_feature_split()
    
    # 4. 三大算法对比
    results = compare_algorithms()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("生成的图片:")
    print("  - gini_entropy_comparison.png")
    print("  - continuous_split.png")
    print("  - algorithm_comparison.png")
    print("  - cart_tree.png")
    print("=" * 60)


if __name__ == '__main__':
    main()