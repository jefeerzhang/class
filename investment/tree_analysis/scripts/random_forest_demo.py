"""
随机森林与集成方法演示脚本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            mean_squared_error, r2_score, roc_curve, auc)
import seaborn as sns

# 尝试导入可选库
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost未安装，跳过相关演示")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM未安装，跳过相关演示")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demo_random_forest_classification():
    """
    随机森林分类演示
    """
    print("=" * 60)
    print("随机森林分类演示 - 乳腺癌数据集")
    print("=" * 60)
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建模型
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        max_features='sqrt',
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = auc(*roc_curve(y_test, y_proba)[:2])
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"OOB得分: {rf.oob_score_:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['恶性', '良性']))
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特征:")
    print(importance.head(10))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 特征重要性条形图
    top_features = importance.head(15)
    axes[0].barh(range(len(top_features)), top_features['importance'])
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('重要性')
    axes[0].set_title('随机森林特征重要性 (Top 15)')
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('假阳性率')
    axes[1].set_ylabel('真阳性率')
    axes[1].set_title('ROC曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return rf, importance


def demo_ensemble_comparison():
    """
    集成方法对比演示
    """
    print("\n" + "=" * 60)
    print("集成方法对比演示")
    print("=" * 60)
    
    # 创建数据
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 定义模型
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
    
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbose=-1
        )
    
    # 训练和评估
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = auc(*roc_curve(y_test, y_proba)[:2])
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'model': model,
            'y_proba': y_proba
        }
        
        print(f"  准确率: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率对比
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    aucs = [results[n]['auc'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracies, width, label='准确率')
    axes[0].bar(x + width/2, aucs, width, label='AUC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('得分')
    axes[0].set_title('模型性能对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # ROC曲线对比
    for name in names:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC = {results[name]['auc']:.4f})")
    
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('假阳性率')
    axes[1].set_ylabel('真阳性率')
    axes[1].set_title('ROC曲线对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demo_oob_vs_cv():
    """
    OOB误差 vs 交叉验证对比
    """
    print("\n" + "=" * 60)
    print("OOB误差 vs 交叉验证对比")
    print("=" * 60)
    
    # 创建数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    
    n_estimators_list = [10, 25, 50, 100, 150, 200]
    oob_scores = []
    cv_scores = []
    
    for n_est in n_estimators_list:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        oob_scores.append(rf.oob_score_)
        cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()
        cv_scores.append(cv)
        
        print(f"n_estimators={n_est}: OOB={rf.oob_score_:.4f}, CV={cv:.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, oob_scores, 'o-', label='OOB得分')
    plt.plot(n_estimators_list, cv_scores, 's-', label='交叉验证得分')
    plt.xlabel('树的数量 (n_estimators)')
    plt.ylabel('准确率')
    plt.title('OOB vs 交叉验证得分对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return oob_scores, cv_scores


def demo_feature_importance_comparison():
    """
    特征重要性方法对比
    """
    print("\n" + "=" * 60)
    print("特征重要性方法对比")
    print("=" * 60)
    
    from sklearn.inspection import permutation_importance
    
    # 加载数据
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练模型
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # MDI重要性 (平均不纯度减少)
    mdi_importance = pd.DataFrame({
        'feature': feature_names,
        'mdi_importance': rf.feature_importances_
    }).sort_values('mdi_importance', ascending=False)
    
    # 置换重要性
    perm_result = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=42
    )
    perm_importance = pd.DataFrame({
        'feature': feature_names,
        'perm_importance': perm_result.importances_mean,
        'perm_std': perm_result.importances_std
    }).sort_values('perm_importance', ascending=False)
    
    # 合并结果
    comparison = mdi_importance.merge(perm_importance, on='feature')
    comparison['mdi_rank'] = comparison['mdi_importance'].rank(ascending=False)
    comparison['perm_rank'] = comparison['perm_importance'].rank(ascending=False)
    
    print("\nTop 15 特征重要性对比:")
    print(comparison.head(15))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MDI重要性
    top_mdi = mdi_importance.head(15)
    axes[0].barh(range(len(top_mdi)), top_mdi['mdi_importance'])
    axes[0].set_yticks(range(len(top_mdi)))
    axes[0].set_yticklabels(top_mdi['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('重要性')
    axes[0].set_title('MDI特征重要性')
    
    # 置换重要性
    top_perm = perm_importance.head(15)
    axes[1].barh(range(len(top_perm)), top_perm['perm_importance'],
                xerr=top_perm['perm_std'])
    axes[1].set_yticks(range(len(top_perm)))
    axes[1].set_yticklabels(top_perm['feature'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('重要性')
    axes[1].set_title('置换重要性 (带误差条)')
    
    plt.tight_layout()
    plt.show()
    
    return comparison


def demo_boosting_learning_curve():
    """
    梯度提升学习曲线演示
    """
    print("\n" + "=" * 60)
    print("梯度提升学习曲线演示")
    print("=" * 60)
    
    # 创建数据
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=15,
        n_redundant=3, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练不同数量的树
    n_estimators_list = list(range(10, 201, 10))
    train_scores = []
    test_scores = []
    
    gb = GradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, random_state=42
    )
    
    for n_est in n_estimators_list:
        gb_temp = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=3, learning_rate=0.1, random_state=42
        )
        gb_temp.fit(X_train, y_train)
        
        train_scores.append(accuracy_score(y_train, gb_temp.predict(X_train)))
        test_scores.append(accuracy_score(y_test, gb_temp.predict(X_test)))
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, 'o-', label='训练集')
    plt.plot(n_estimators_list, test_scores, 's-', label='测试集')
    plt.xlabel('树的数量 (n_estimators)')
    plt.ylabel('准确率')
    plt.title('梯度提升学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 学习率影响
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for lr in learning_rates:
        gb_lr = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=lr, random_state=42
        )
        gb_lr.fit(X_train, y_train)
        
        # 训练曲线
        train_score = [accuracy_score(y_train, y_pred) 
                       for y_pred in gb_lr.staged_predict(X_train)]
        test_score = [accuracy_score(y_test, y_pred) 
                     for y_pred in gb_lr.staged_predict(X_test)]
        
        axes[0].plot(range(1, 201), train_score, label=f'lr={lr}')
        axes[1].plot(range(1, 201), test_score, label=f'lr={lr}')
    
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('训练准确率')
    axes[0].set_title('不同学习率的训练曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('测试准确率')
    axes[1].set_title('不同学习率的测试曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return train_scores, test_scores


def main():
    """
    主函数
    """
    print("随机森林与集成方法完整演示")
    print("=" * 60)
    
    # 1. 随机森林分类
    rf, importance = demo_random_forest_classification()
    
    # 2. 集成方法对比
    results = demo_ensemble_comparison()
    
    # 3. OOB vs CV
    oob_scores, cv_scores = demo_oob_vs_cv()
    
    # 4. 特征重要性对比
    comparison = demo_feature_importance_comparison()
    
    # 5. 梯度提升学习曲线
    train_scores, test_scores = demo_boosting_learning_curve()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()