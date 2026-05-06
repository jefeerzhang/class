---
marp: true
theme: graph_paper
paginate: true
header: '技术分享标题'
backgroundColor: #fff
style: |
  section {
    font-family: "Helvetica Neue", "Helvetica", "Arial", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  }
  h1, h2, h3, h4, h5, h6 {
    font-family: "Helvetica Neue", "Helvetica", "Arial", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  }
  /* 图片和表格居中 */
  img {
    display: block !important;
    margin: 0 auto !important;
  }
  section table {
    display: table !important;
    width: auto !important;
    margin-left: auto !important;
    margin-right: auto !important;
  }
  /* 代码块样式 */
  pre {
    background-color: #f8f9fa;
    border-radius: 6px;
    padding: 0.8em;
    font-size: 0.85em;
    line-height: 1.5;
    overflow-x: auto;
  }
  code {
    background-color: #f5f5f5;
    padding: 0.15em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
  }
  /* 行内代码深色 */
  :not(pre) > code {
    color: #c7254e;
    background-color: #f9f2f4;
  }
  /* 强调文本样式 */
  strong {
    color: #333;
  }
  blockquote {
    color: #0066cc;
    border-left: 4px solid #0066cc;
    padding-left: 1rem;
    margin: 1rem 0;
    background-color: transparent;
  }
  blockquote p {
    color: #0066cc;
    margin: 0;
  }
  /* 列表样式优化 */
  ul, ol {
    margin-left: 1.5em;
  }
  li {
    margin-bottom: 0.3em;
  }
  /* 步骤标记 */
  .step {
    display: inline-block;
    background: #0066cc;
    color: white;
    border-radius: 50%;
    width: 1.5em;
    height: 1.5em;
    text-align: center;
    line-height: 1.5em;
    font-size: 0.8em;
    margin-right: 0.3em;
  }
  /* MathJax 公式大小 */
  .katex {
    font-size: 1em;
  }
---

# 技术分享标题

**核心主题说明**

演讲人 / 日期

---

## 今日内容

1. **背景与问题** - 为什么需要这个方案
2. **核心原理** - 关键技术点
3. **代码实践** - 核心实现
4. **对比与总结** - 方案评估

---

# Part 1: 背景与问题

---

## 问题场景

**实际遇到的挑战：**

- 问题一：描述具体场景
- 问题二：现有方案的不足
- 问题三：为什么需要新方案

**关键数据：** 用数据说明问题的严重性

---

## 现有方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 方案A | ... | ... | ... |
| 方案B | ... | ... | ... |
| 我们的方案 | ... | ... | ... |

---

# Part 2: 核心原理

---

## 关键概念

**核心公式：**

$$L = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}\beta_j^2$$

- 第一项：数据拟合程度
- 第二项：正则化惩罚项
- $\lambda$：控制惩罚强度

---

## 算法流程

**三步走：**

1. **初始化** - 设置参数和初始值
2. **迭代优化** - 核心计算逻辑
3. **收敛判断** - 停止条件

> 关键洞察：算法的复杂度是 $O(n \cdot p)$，适合中等规模数据。

---

# Part 3: 代码实践

---

## 核心实现

```python
from sklearn.linear_model import Ridge

# 初始化模型
ridge = Ridge(alpha=1.0)

# 训练
ridge.fit(X_train, y_train)

# 预测与评估
print(f"R²: {ridge.score(X_test, y_test):.4f}")
```

---

## 关键参数调优

```python
# 网格搜索最优参数
alphas = np.logspace(-3, 3, 100)
best_alpha = None
best_score = -np.inf

for alpha in alphas:
    model = Ridge(alpha=alpha)
    score = cross_val_score(model, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_alpha = alpha
```

**要点：** 交叉验证是选择超参数的最靠谱方法。

---

## 可视化结果

```python
# 系数路径图
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('系数值')
plt.title('正则化强度对系数的影响')
```

> 观察：随着 alpha 增大，系数平滑地向 0 收缩。

---

# Part 4: 对比与总结

---

## 方法对比

| 方法 | 稀疏性 | 计算速度 | 适用场景 |
|------|--------|----------|----------|
| Ridge | 否 | 快 | 共线性严重 |
| Lasso | 是 | 中等 | 特征筛选 |
| Elastic Net | 是 | 中等 | 综合场景 |

---

## 最佳实践

1. **先标准化** - 正则化对特征尺度敏感
2. **交叉验证** - 用 CV 选择超参数
3. **从简单开始** - 先试试 Ridge，不够用再加复杂度
4. **可视化** - 画系数路径图帮助理解

---

## 常见问题

**Q: 特征数远大于样本数怎么办？**

A: 用 Lasso 或 Elastic Net，或者直接上 PLS。

**Q: 怎么判断过拟合？**

A: 训练集和测试集的性能差距过大。

---

# 总结

**核心要点：**

- 正则化防止过拟合，提升泛化能力
- L2 收缩系数，L1 产生稀疏解
- 交叉验证选择超参数

**下一步：**

动手跑一遍代码，改改参数看看效果。

---

# 参考资料

- 论文/书籍链接
- 代码仓库地址
- 推荐阅读列表
