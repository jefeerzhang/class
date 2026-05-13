#!/usr/bin/env python3
"""
分类模型评估指标 - 信用卡违约预测示例
生成混淆矩阵热力图、ROC曲线，并输出准确率、AUC等指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)
n = 500

# 模拟客户特征
age = np.random.randint(22, 65, n)                        # 年龄
income = np.random.lognormal(10.8, 0.5, n)                # 年收入
debt_ratio = np.random.beta(2, 5, n)                      # 负债率
credit_score = np.clip(np.random.normal(680, 50, n), 300, 850)  # 信用评分
months_employed = np.random.randint(0, 360, n)            # 在职月数

# 违约概率（与信用评分负相关，与负债率正相关）
logit = (
    -2.5
    - 0.012 * (credit_score - 650)
    + 4.5 * debt_ratio
    + 0.008 * (age - 40)
    - 0.003 * (months_employed / 12)
)
prob = 1 / (1 + np.exp(-logit))
default = (np.random.rand(n) < prob).astype(int)

print(f"违约比例: {default.mean():.2%}")
print(f"违约人数: {default.sum()}, 未违约人数: {(1-default).sum()}")

X = np.column_stack([age, income, debt_ratio, credit_score, months_employed])
feature_names = ['年龄', '年收入', '负债率', '信用评分', '在职月数']
y = default

# 标准化特征
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 输出指标
output_lines = []
output_lines.append("=== 分类模型评估指标 ===\n")
output_lines.append(f"准确率 (Accuracy):   {accuracy:.4f}")
output_lines.append(f"精确率 (Precision):  {precision:.4f}")
output_lines.append(f"召回率 (Recall):     {recall:.4f}")
output_lines.append(f"F1 分数:             {f1:.4f}")
output_lines.append(f"AUC:                 {roc_auc:.4f}\n")

output_lines.append("=== 混淆矩阵 ===")
output_lines.append(f"True Negatives  (TN): {tn}")
output_lines.append(f"False Positives (FP): {fp}")
output_lines.append(f"False Negatives (FN): {fn}")
output_lines.append(f"True Positives  (TP): {tp}")

output = "\n".join(output_lines)
print(output)

with open(OUTPUT_DIR / 'evaluation_output.txt', 'w', encoding='utf-8') as f:
    f.write(output)

# 图1: 混淆矩阵热力图
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

classes = ['未违约', '违约']
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='真实值',
    xlabel='预测值',
    title='混淆矩阵（信用卡违约预测）'
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center', color=color, fontsize=16)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
plt.close()

# 图2: ROC 曲线
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='随机猜测')

ax.set_xlabel('假正率 (FPR)', fontsize=13)
ax.set_ylabel('真正率 (TPR)', fontsize=13)
ax.set_title('ROC 曲线（信用卡违约预测）', fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=150)
plt.close()

print(f"\n图片已保存到 {OUTPUT_DIR}")
