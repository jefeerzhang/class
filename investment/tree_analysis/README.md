# 树类分析方法 (Tree-based Methods)

本目录包含树类分析方法的完整学习资料，涵盖理论基础、代码实现和金融领域应用。

## 目录结构

```
tree_analysis/
├── docs/           # 文档资料
│   └── 树类分析方法完整指南.md   # 详细理论文档
├── notebooks/      # Jupyter notebooks (待添加)
├── scripts/        # Python脚本
│   ├── decision_tree_demo.py            # 决策树基础演示
│   ├── random_forest_demo.py            # 随机森林与集成方法演示
│   └ financial_applications_demo.py     # 金融应用演示
└── data/           # 数据文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm shap  # 可选，用于高级模型
```

### 2. 运行示例脚本

```bash
# 决策树基础演示
python scripts/decision_tree_demo.py

# 随机森林演示
python scripts/random_forest_demo.py

# 金融应用演示
python scripts/financial_applications_demo.py
```

## 学习内容

### 文档内容概览

`docs/树类分析方法完整指南.md` 包含以下章节：

1. **概述** - 树类方法发展历程与优势
2. **决策树基本原理** - 分裂准则、剪枝策略
3. **经典决策树算法详解** - ID3、C4.5、CART三大经典算法
4. **分类决策树** - CART算法、多类别处理
5. **回归决策树** - MSE分裂准则、预测方法
6. **随机森林** - Bagging、OOB误差、特征重要性
7. **其他集成方法** - GBDT、XGBoost、LightGBM、CatBoost
8. **模型评价指标** - 分类/回归指标、交叉验证
9. **模型比较与选择** - 各方法优缺点对比
10. **金融领域应用** - 信用评分、股票预测、欺诈检测
11. **代码实战** - 完整示例代码
12. **最佳实践** - 数据预处理、超参数调优、部署建议

### 核心知识点

| 方法 | 核心特点 | 适用场景 |
|------|----------|----------|
| 决策树 | 可解释性强 | 需要决策规则的场景 |
| 随机森林 | 高准确率、抗过拟合 | 中等数据规模、特征重要分析 |
| GBDT | 高精度、顺序训练 | 追求精度的竞赛/项目 |
| XGBoost | 正则化、并行化 | 大数据、高精度需求 |
| LightGBM | 高效率、大数据处理 | 大规模数据、快速训练 |

### 金融应用案例

| 应用 | 问题类型 | 核心指标 |
|------|----------|----------|
| 信用评分 | 二分类 | AUC、召回率 |
| 股票预测 | 回归/分类 | R²、准确率 |
| 欺诈检测 | 异常检测 | 精确率、召回率 |
| 期权定价 | 回归 | MSE |

## 推荐学习顺序

1. 阅读文档中的理论部分
2. 运行 `decision_tree_demo.py` 理解基础概念
3. 运行 `random_forest_demo.py` 学习集成方法
4. 运行 `financial_applications_demo.py` 了解实际应用
5. 尝试调整参数，观察模型变化

## 常用参数速查

### 决策树

```python
DecisionTreeClassifier(
    criterion='gini',       # 'gini' 或 'entropy'
    max_depth=5,            # 最大深度，防止过拟合
    min_samples_leaf=5,     # 叶节点最小样本数
    class_weight='balanced' # 类别不平衡时使用
)
```

### 随机森林

```python
RandomForestClassifier(
    n_estimators=100,       # 树的数量
    max_depth=10,           # 每棵树深度
    max_features='sqrt',    # 每次分裂考虑的特征数
    oob_score=True,         # 袋外得分评估
    n_jobs=-1               # 并行计算
)
```

### XGBoost

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,          # 数据采样比例
    colsample_bytree=0.8,   # 特征采样比例
    reg_alpha=0.1,          # L1正则化
    reg_lambda=1.0          # L2正则化
)
```

## 参考资料

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.

---

**更新日期：** 2026-04-21