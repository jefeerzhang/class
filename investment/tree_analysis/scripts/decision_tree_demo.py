"""
决策树方法演示脚本
包含分类决策树和回归决策树的基础示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            mean_squared_error, r2_score, roc_auc_score)
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demo_classification_tree():
    """
    分类决策树演示
    使用鸢尾花数据集
    """
    print("=" * 60)
    print("分类决策树演示 - 鸢尾花数据集")
    print("=" * 60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建模型
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )
    
    # 训练
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 交叉验证
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n特征重要性:")
    print(importance)
    
    # 可视化决策树
    plt.figure(figsize=(12, 8))
    plot_tree(clf, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('分类决策树可视化 - 鸢尾花数据集')
    plt.tight_layout()
    plt.savefig('decision_tree_iris.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.show()
    
    return clf


def demo_regression_tree():
    """
    回归决策树演示
    使用加州房价数据集
    """
    print("\n" + "=" * 60)
    print("回归决策树演示 - 加州房价数据集")
    print("=" * 60)
    
    # 加载数据
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建模型
    reg = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    
    # 训练
    reg.fit(X_train, y_train)
    
    # 预测
    y_pred = reg.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 交叉验证
    cv_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
    print(f"交叉验证R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': reg.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n特征重要性:")
    print(importance)
    
    # 预测值vs实际值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 实际值')
    plt.tight_layout()
    plt.show()
    
    # 残差图
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.tight_layout()
    plt.show()
    
    return reg


def demo_hyperparameter_tuning():
    """
    超参数调优演示
    """
    print("\n" + "=" * 60)
    print("超参数调优演示 - GridSearchCV")
    print("=" * 60)
    
    # 创建合成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 定义参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    # 网格搜索
    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 在测试集上评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 可视化调参结果
    results = pd.DataFrame(grid_search.cv_results_)
    
    plt.figure(figsize=(10, 6))
    for criterion in ['gini', 'entropy']:
        subset = results[results['param_criterion'] == criterion]
        plt.plot(subset['param_max_depth'], subset['mean_test_score'],
                'o-', label=f'{criterion}')
    
    plt.xlabel('最大深度')
    plt.ylabel('交叉验证准确率')
    plt.title('参数调优结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return grid_search


def demo_pruning():
    """
    剪枝效果演示
    """
    print("\n" + "=" * 60)
    print("剪枝效果演示")
    print("=" * 60)
    
    # 创建数据
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 不剪枝的树
    clf_full = DecisionTreeClassifier(random_state=42)
    clf_full.fit(X_train, y_train)
    
    # 预剪枝
    clf_pre = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    clf_pre.fit(X_train, y_train)
    
    # 后剪枝 (CCP)
    path = clf_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # 排除最后一个（空树）
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    
    # 找出最佳ccp_alpha
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    best_idx = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]
    
    clf_post = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    clf_post.fit(X_train, y_train)
    
    # 结果对比
    print(f"\n不剪枝 - 训练准确率: {clf_full.score(X_train, y_train):.4f}, "
          f"测试准确率: {clf_full.score(X_test, y_test):.4f}, "
          f"节点数: {clf_full.tree_.node_count}")
    
    print(f"预剪枝 - 训练准确率: {clf_pre.score(X_train, y_train):.4f}, "
          f"测试准确率: {clf_pre.score(X_test, y_test):.4f}, "
          f"节点数: {clf_pre.tree_.node_count}")
    
    print(f"后剪枝 - 训练准确率: {clf_post.score(X_train, y_train):.4f}, "
          f"测试准确率: {clf_post.score(X_test, y_test):.4f}, "
          f"节点数: {clf_post.tree_.node_count}, "
          f"最佳alpha: {best_alpha:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率vs alpha
    axes[0].plot(ccp_alphas, train_scores, 'o-', label='训练集')
    axes[0].plot(ccp_alphas, test_scores, 'o-', label='测试集')
    axes[0].axvline(best_alpha, color='r', linestyle='--', label=f'最佳alpha={best_alpha:.4f}')
    axes[0].set_xlabel('CCP Alpha')
    axes[0].set_ylabel('准确率')
    axes[0].set_title('后剪枝: 准确率 vs Alpha')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 树深度vs节点数
    depths = [clf.tree_.max_depth for clf in clfs]
    nodes = [clf.tree_.node_count for clf in clfs]
    axes[1].plot(ccp_alphas, depths, 'o-', label='深度')
    axes[1].plot(ccp_alphas, nodes, 's-', label='节点数')
    axes[1].set_xlabel('CCP Alpha')
    axes[1].set_ylabel('深度/节点数')
    axes[1].set_title('后剪枝: 树复杂度 vs Alpha')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return clf_full, clf_pre, clf_post


def main():
    """
    主函数
    """
    print("决策树方法完整演示")
    print("=" * 60)
    
    # 1. 分类决策树
    clf = demo_classification_tree()
    
    # 2. 回归决策树
    reg = demo_regression_tree()
    
    # 3. 超参数调优
    grid_search = demo_hyperparameter_tuning()
    
    # 4. 剪枝效果
    clf_full, clf_pre, clf_post = demo_pruning()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()