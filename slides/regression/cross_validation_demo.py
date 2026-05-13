#!/usr/bin/env python3
"""
交叉验证演示脚本
生成 K-Fold 示意图、学习曲线、验证曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, cross_val_score,
    learning_curve, validation_curve
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 图1: K-Fold 交叉验证示意图
# ============================================================

n_samples = 20
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
indices = np.arange(n_samples)

fig, axes = plt.subplots(k, 1, figsize=(12, 6))

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
    ax = axes[fold_idx]
    
    # 绘制所有样本点
    ax.scatter(range(n_samples), [0]*n_samples, c='lightgray', s=100, zorder=2)
    
    # 训练集
    ax.scatter(train_idx, [0]*len(train_idx), c='#2196F3', s=100, zorder=3, label='训练集')
    
    # 验证集
    ax.scatter(val_idx, [0]*len(val_idx), c='#FF5722', s=100, zorder=3, label='验证集')
    
    ax.set_xlim(-1, n_samples)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(range(1, n_samples+1), fontsize=8)
    ax.set_ylabel(f'Fold {fold_idx+1}', fontsize=10)
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#ddd')
    
    if fold_idx == 0:
        ax.legend(loc='upper right', fontsize=9, ncol=2)

axes[-1].set_xlabel('样本编号', fontsize=11)
fig.suptitle('5-Fold 交叉验证数据划分示意图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'kfold_diagram.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 图2: 不同 K 值的交叉验证结果比较
# ============================================================

X, y = make_regression(n_samples=200, n_features=10, noise=20, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = [3, 5, 10, 20, 50, 100]
cv_means = []
cv_stds = []

for k in k_values:
    if k > len(X):
        continue
    scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y, cv=k, scoring='r2')
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(k_values, cv_means, yerr=cv_stds, fmt='o-', capsize=5,
            color='#2196F3', linewidth=2, markersize=8, label='CV R² ± 标准差')
ax.axhline(y=cv_means[0], color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('K 值（折数）', fontsize=12)
ax.set_ylabel('交叉验证 R²', fontsize=12)
ax.set_title('不同 K 值对交叉验证结果的影响', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(k_values)

# 添加注释
ax.annotate('K=5: 常用选择\n偏差-方差平衡好', xy=(5, cv_means[1]),
            xytext=(7, cv_means[1]-0.02),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'k_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 图3: 学习曲线（训练集大小 vs 得分）
# ============================================================

X, y = make_regression(n_samples=500, n_features=15, noise=30, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

train_sizes, train_scores, val_scores = learning_curve(
    Ridge(alpha=1.0), X_scaled, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.1, color='#2196F3')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.1, color='#FF5722')

ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', linewidth=2,
        markersize=6, label='训练集得分')
ax.plot(train_sizes, val_mean, 'o-', color='#FF5722', linewidth=2,
        markersize=6, label='验证集得分')

ax.set_xlabel('训练样本数', fontsize=12)
ax.set_ylabel('R² 得分', fontsize=12)
ax.set_title('学习曲线：训练集大小 vs 模型表现', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

# 添加注释
gap = train_mean[-1] - val_mean[-1]
if gap > 0.05:
    ax.annotate(f'差距 = {gap:.2f}\n（过拟合信号）',
                xy=(train_sizes[-1], (train_mean[-1]+val_mean[-1])/2),
                xytext=(train_sizes[-3], val_mean[-1]-0.1),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 图4: 验证曲线（参数 vs 得分）
# ============================================================

X, y = make_regression(n_samples=200, n_features=10, noise=20, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

alphas = np.logspace(-3, 3, 20)
train_scores, val_scores = validation_curve(
    Ridge(), X_scaled, y,
    param_name='alpha', param_range=alphas,
    cv=5, scoring='r2', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(alphas, train_mean - train_std, train_mean + train_std,
                alpha=0.1, color='#2196F3')
ax.fill_between(alphas, val_mean - val_std, val_mean + val_std,
                alpha=0.1, color='#FF5722')

ax.plot(alphas, train_mean, 'o-', color='#2196F3', linewidth=2,
        markersize=5, label='训练集得分')
ax.plot(alphas, val_mean, 'o-', color='#FF5722', linewidth=2,
        markersize=5, label='验证集得分')

# 标记最优 alpha
best_idx = np.argmax(val_mean)
best_alpha = alphas[best_idx]
ax.axvline(x=best_alpha, color='green', linestyle='--', alpha=0.7)
ax.annotate(f'最优 α = {best_alpha:.3f}',
            xy=(best_alpha, val_mean[best_idx]),
            xytext=(best_alpha*3, val_mean[best_idx]-0.02),
            fontsize=11, ha='left',
            arrowprops=dict(arrowstyle='->', color='green'))

ax.set_xscale('log')
ax.set_xlabel('正则化参数 α', fontsize=12)
ax.set_ylabel('R² 得分', fontsize=12)
ax.set_title('验证曲线：正则化参数 α 的选择', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'validation_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 图5: 不同交叉验证策略对比
# ============================================================

X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                           n_classes=2, weights=[0.9, 0.1], random_state=42)
X_scaled = StandardScaler().fit_transform(X)

cv_strategies = {
    'K-Fold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'LeaveOneOut': LeaveOneOut(),
}

results = {}
for name, cv in cv_strategies.items():
    scores = cross_val_score(LogisticRegression(max_iter=1000), X_scaled, y,
                            cv=cv, scoring='accuracy')
    results[name] = scores

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for idx, (name, scores) in enumerate(results.items()):
    ax = axes[idx]
    ax.bar(range(len(scores)), scores, color=['#2196F3', '#4CAF50', '#FF9800'][idx],
           alpha=0.8)
    ax.axhline(y=scores.mean(), color='red', linestyle='--', linewidth=2,
               label=f'均值: {scores.mean():.3f}')
    ax.set_xlabel('折数', fontsize=10)
    ax.set_ylabel('准确率', fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('不同交叉验证策略的对比（不平衡数据集）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cv_strategies.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"所有图片已保存到 {OUTPUT_DIR}")
