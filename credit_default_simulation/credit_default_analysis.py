# 信贷违约分析 Jupyter Notebook
# 在 Jupyter 中打开此文件，然后运行所有单元格

# %% [markdown]
# # 信贷违约预测：正则化、降维与评估指标
# 
# 本 Notebook 演示基于信贷违约数据的机器学习分析流程，涵盖：
# 1. **数据探索与预处理**
# 2. **正则化回归**（Ridge、Lasso、ElasticNet）
# 3. **降维方法**（PCR、PLS）
# 4. **分类评估指标**（混淆矩阵、准确率、精确率、召回率、F1、AUC）
# 5. **交叉验证与模型选择**

# %%
# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# 设置可视化样式与中文字体
import matplotlib.font_manager as fm
plt.style.use('seaborn-v0_8-darkgrid')
# 检查可用中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist if any(keyword in f.name.lower() for keyword in ['simhei', 'microsoft yahei', 'arial unicode', 'sim sun', 'microsoftyahei'])]
if available_fonts:
    plt.rcParams['font.sans-serif'] = available_fonts[:3] + ['DejaVu Sans']  # 备选字体
    print(f'已设置中文字体: {available_fonts[:3]}')
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    print('未找到中文字体，使用默认字体（中文可能显示为方框）')
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_palette("husl")

# %%
# 1. 加载数据
print("1. 加载数据")
df = pd.read_csv('credit_data.csv')
print(f"数据集形状: {df.shape}")
print(f"特征数量: {df.shape[1] - 1}")
print(f"样本数量: {df.shape[0]}")
print(f"\n目标变量分布:")
print(df['default'].value_counts())
print(f"\n违约率: {df['default'].mean():.2%}")

# %%
# 2. 探索性数据分析
print("\n2. 探索性数据分析")
print("\n特征统计摘要:")
print(df.describe().round(2))

print("\n分类特征分布:")
print(df['employment_type'].value_counts())

# 可视化：违约率 vs 非违约率
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 违约分布
ax1 = axes[0, 0]
default_counts = df['default'].value_counts()
ax1.bar(['非违约 (0)', '违约 (1)'], default_counts.values, color=['green', 'red'])
ax1.set_title('违约分布')
ax1.set_ylabel('数量')
for i, v in enumerate(default_counts.values):
    ax1.text(i, v + 10, str(v), ha='center')

# 信用评分分布
ax2 = axes[0, 1]
ax2.hist(df['credit_score'], bins=30, alpha=0.7, color='blue')
ax2.axvline(df['credit_score'].mean(), color='red', linestyle='--', label=f'均值: {df["credit_score"].mean():.0f}')
ax2.set_title('信用评分分布')
ax2.set_xlabel('信用评分')
ax2.set_ylabel('频数')
ax2.legend()

# 负债率分布
ax3 = axes[1, 0]
ax3.hist(df['debt_ratio'], bins=30, alpha=0.7, color='orange')
ax3.set_title('负债率分布')
ax3.set_xlabel('负债率')
ax3.set_ylabel('频数')

# 年收入分布
ax4 = axes[1, 1]
ax4.hist(df['income'], bins=30, alpha=0.7, color='green')
ax4.set_title('年收入分布')
ax4.set_xlabel('年收入（万元）')
ax4.set_ylabel('频数')

plt.tight_layout()
plt.show()

# 相关性热力图
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

# %%
# 3. 数据预处理
print("\n3. 数据预处理")

# 分离特征和目标
X = df.drop('default', axis=1)
y = df['default']

# 识别数值特征和分类特征
numeric_features = ['age', 'income', 'debt_ratio', 'credit_score', 
                    'months_employed', 'num_credit_lines', 'num_late_payments',
                    'loan_amount', 'savings_balance']
categorical_features = ['employment_type']

# 预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集违约率: {y_train.mean():.2%}")
print(f"测试集违约率: {y_test.mean():.2%}")

# 预处理数据
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 获取特征名称
feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat']
                      .get_feature_names_out(categorical_features)))
print(f"\n处理后的特征数量: {X_train_processed.shape[1]}")

# %%
# 4. 正则化回归
print("\n4. 正则化回归")

# 转换为 DataFrame 便于查看
X_train_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed,
                          columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
                         columns=feature_names)

# 4.1 Ridge 回归
print("\n4.1 Ridge 回归")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_processed, y_train)
ridge_pred = ridge.predict(X_test_processed)
ridge_pred_binary = (ridge_pred > 0.5).astype(int)
print(f"Ridge 测试集 R²: {ridge.score(X_test_processed, y_test):.4f}")
print(f"Ridge 测试集 RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred)):.4f}")

# 4.2 Lasso 回归
print("\n4.2 Lasso 回归")
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_processed, y_train)
lasso_pred = lasso.predict(X_test_processed)
lasso_pred_binary = (lasso_pred > 0.5).astype(int)
print(f"Lasso 测试集 R²: {lasso.score(X_test_processed, y_test):.4f}")
print(f"Lasso 测试集 RMSE: {np.sqrt(mean_squared_error(y_test, lasso_pred)):.4f}")
print(f"Lasso 非零系数个数: {np.sum(lasso.coef_ != 0)}")

# 4.3 Elastic Net
print("\n4.3 Elastic Net")
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_processed, y_train)
elastic_pred = elastic.predict(X_test_processed)
elastic_pred_binary = (elastic_pred > 0.5).astype(int)
print(f"Elastic Net 测试集 R²: {elastic.score(X_test_processed, y_test):.4f}")
print(f"Elastic Net 测试集 RMSE: {np.sqrt(mean_squared_error(y_test, elastic_pred)):.4f}")

# 可视化系数比较
plt.figure(figsize=(12, 6))
coef_df = pd.DataFrame({
    '特征': feature_names,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_,
    'Elastic Net': elastic.coef_
})
x = np.arange(len(feature_names))
width = 0.25
plt.bar(x - width, ridge.coef_, width, label='Ridge', alpha=0.8)
plt.bar(x, lasso.coef_, width, label='Lasso', alpha=0.8)
plt.bar(x + width, elastic.coef_, width, label='Elastic Net', alpha=0.8)
plt.xlabel('特征')
plt.ylabel('系数值')
plt.title('正则化回归系数比较')
plt.xticks(x, feature_names, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 5. 降维方法
print("\n5. 降维方法")

# 5.1 主成分回归 (PCR)
print("\n5.1 主成分回归 (PCR)")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)

print(f"解释方差比例: {pca.explained_variance_ratio_.round(3)}")
print(f"累积解释方差: {np.sum(pca.explained_variance_ratio_):.3f}")

# 用主成分进行回归
pcr = LogisticRegression(max_iter=1000)
pcr.fit(X_train_pca, y_train)
pcr_pred = pcr.predict(X_test_pca)
print(f"PCR 准确率: {accuracy_score(y_test, pcr_pred):.4f}")

# 5.2 偏最小二乘 (PLS)
print("\n5.2 偏最小二乘 (PLS)")
pls = PLSRegression(n_components=5)
pls.fit(X_train_processed, y_train)
pls_pred = pls.predict(X_test_processed)
pls_pred_binary = (pls_pred > 0.5).astype(int)
print(f"PLS R²: {pls.score(X_test_processed, y_test):.4f}")
print(f"PLS RMSE: {np.sqrt(mean_squared_error(y_test, pls_pred)):.4f}")

# %%
# 6. 分类评估指标
print("\n6. 分类评估指标")

# 使用逻辑回归作为基准模型
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_processed, y_train)
y_pred = logreg.predict(X_test_processed)
y_prob = logreg.predict_proba(X_test_processed)[:, 1]

# 混淆矩阵
print("\n6.1 混淆矩阵")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测未违约', '预测违约'],
            yticklabels=['实际未违约', '实际违约'])
plt.title('混淆矩阵')
plt.ylabel('真实值')
plt.xlabel('预测值')
plt.show()

# 6.2 分类指标
print("\n6.2 分类指标")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数: {f1:.4f}")

# 6.3 ROC 曲线和 AUC
print("\n6.3 ROC 曲线和 AUC")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 6.4 精确率-召回率曲线
print("\n6.4 精确率-召回率曲线")
from sklearn.metrics import precision_recall_curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='purple', lw=2, label=f'PR 曲线 (AUC = {pr_auc:.3f})')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()

# %%
# 7. 交叉验证与模型选择
print("\n7. 交叉验证与模型选择")

# 7.1 不同模型的交叉验证对比
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'Elastic Net': ElasticNet(alpha=0.01, l1_ratio=0.5)
}

cv_results = {}
for name, model in models.items():
    # 对于回归模型，转换为分类评估
    if name in ['Ridge', 'Lasso', 'Elastic Net']:
        scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='r2')
        cv_results[name] = scores.mean()
        print(f"{name}: R² = {scores.mean():.4f} (±{scores.std():.4f})")
    else:
        scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='accuracy')
        cv_results[name] = scores.mean()
        print(f"{name}: Accuracy = {scores.mean():.4f} (±{scores.std():.4f})")

# 7.2 超参数调优：Lasso 的 alpha 选择
print("\n7.2 Lasso 超参数调优")
alphas = np.logspace(-4, 1, 20)
lasso_scores = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=1000)
    scores = cross_val_score(lasso, X_train_processed, y_train, cv=5, scoring='r2')
    lasso_scores.append(scores.mean())

best_alpha = alphas[np.argmax(lasso_scores)]
print(f"最优 alpha: {best_alpha:.4f}")
print(f"最优 CV R²: {max(lasso_scores):.4f}")

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, lasso_scores, 'b-', linewidth=2)
plt.xlabel('Alpha (正则化强度)')
plt.ylabel('交叉验证 R²')
plt.title('Lasso 回归：交叉验证得分 vs Alpha')
plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'最优 alpha = {best_alpha:.4f}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 7.3 降维方法对比：PCR vs PLS
print("\n7.3 降维方法对比：PCR vs PLS")
n_components_range = range(1, 11)
pcr_scores = []
pls_scores = []

for n_comp in n_components_range:
    # PCR
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_processed)
    pcr = LogisticRegression(max_iter=1000)
    pcr_cv = cross_val_score(pcr, X_train_pca, y_train, cv=5, scoring='accuracy')
    pcr_scores.append(pcr_cv.mean())
    
    # PLS
    pls = PLSRegression(n_components=n_comp)
    pls_cv = cross_val_score(pls, X_train_processed, y_train, cv=5, scoring='r2')
    pls_scores.append(pls_cv.mean())

plt.figure(figsize=(10, 6))
plt.plot(n_components_range, pcr_scores, 'b-o', label='PCR (Accuracy)')
plt.plot(n_components_range, pls_scores, 'r-s', label='PLS (R²)')
plt.xlabel('成分数')
plt.ylabel('交叉验证得分')
plt.title('PCR vs PLS：不同成分数的性能对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# 8. 模型对比总结
print("\n8. 模型对比总结")
print("\n各模型性能对比：")
comparison_df = pd.DataFrame({
    '模型': ['Logistic Regression', 'Ridge', 'Lasso', 'Elastic Net', 'PCR (5成分)', 'PLS (5成分)'],
    '测试集得分': [
        accuracy_score(y_test, y_pred),
        ridge.score(X_test_processed, y_test),
        lasso.score(X_test_processed, y_test),
        elastic.score(X_test_processed, y_test),
        accuracy_score(y_test, pcr_pred),
        pls.score(X_test_processed, y_test)
    ],
    '指标类型': ['Accuracy', 'R²', 'R²', 'R²', 'Accuracy', 'R²']
})
print(comparison_df.round(4))

# 选择最佳模型
print("\n最佳模型选择建议：")
print("1. 如果需要可解释性：Logistic Regression（系数可解释）")
print("2. 如果需要特征选择：Lasso（自动选择重要特征）")
print("3. 如果需要稳定性：Ridge（处理共线性）")
print("4. 如果需要降维：PLS（有监督降维，通常优于PCR）")

# %%
# 9. 业务建议
print("\n9. 业务建议")
print("""
基于本分析，对银行信贷风控提出以下建议：

1. **关键风险因素**：
   - 信用评分（最重要）
   - 负债率
   - 逾期次数
   - 就业类型（自雇风险较高）

2. **模型选择**：
   - 生产环境：优先使用逻辑回归或 Ridge 回归
   - 特征探索：使用 Lasso 识别关键风险因素
   - 高维数据：考虑 PLS 降维

3. **阈值调整**：
   - 风控部门：降低阈值，提高召回率（宁可误报，不可漏报）
   - 营销部门：提高阈值，提高精确率（精准定位优质客户）

4. **监控指标**：
   - 定期监控模型性能（AUC、KS值）
   - 关注特征漂移（客户结构变化）
   - 定期重新训练模型
""")

print("\n分析完成！")