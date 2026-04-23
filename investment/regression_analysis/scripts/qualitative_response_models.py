"""
定性响应变量回归模型演示
包含：Logistic回归、多项Logistic、有序Logistic、Probit回归
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report,
                            log_loss)
import statsmodels.api as sm
from scipy import stats

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demo_binary_logistic():
    """
    二分类Logistic回归演示
    """
    print("=" * 60)
    print("二分类Logistic回归演示")
    print("=" * 60)
    
    # 生成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 1. sklearn版本（预测为主）
    print("\n--- sklearn Logistic回归 ---")
    log_sk = LogisticRegression(max_iter=1000, random_state=42)
    log_sk.fit(X_train, y_train)
    
    y_pred_sk = log_sk.predict(X_test)
    y_proba_sk = log_sk.predict_proba(X_test)[:, 1]
    
    print(f"准确率: {accuracy_score(y_test, y_pred_sk):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba_sk):.4f}")
    
    # 2. statsmodels版本（统计推断）
    print("\n--- statsmodels Logistic回归（统计推断）---")
    X_sm = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_sm)
    logit_result = logit_model.fit()
    
    print(logit_result.summary())
    
    # 发生比比解释
    odds_ratios = np.exp(logit_result.params[1:])  # 排除常数项
    print("\n发生比比（Odds Ratios）：")
    for i, (feature, or_val) in enumerate(zip(feature_names, odds_ratios)):
        if or_val > 1:
            print(f"  {feature}: {or_val:.4f} → 增加{(or_val-1)*100:.1f}%")
        else:
            print(f"  {feature}: {or_val:.4f} → 减少{(1-or_val)*100:.1f}%")
    
    # 系数显著性检验
    print("\n显著性检验（Wald检验）：")
    for i, (feature, coef, pval) in enumerate(zip(feature_names, 
                                                   logit_result.params[1:],
                                                   logit_result.pvalues[1:])):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feature}: β={coef:.4f}, p={pval:.4f} {sig}")
    
    # 置信区间
    conf_int = logit_result.conf_int()[1:]  # 排除常数项
    print("\n系数95%置信区间：")
    for feature, (lower, upper) in zip(feature_names, conf_int.values):
        print(f"  {feature}: [{lower:.4f}, {upper:.4f}]")
    
    # 3. ROC曲线
    y_proba_sm = logit_result.predict(sm.add_constant(X_test))
    
    fpr_sk, tpr_sk, _ = roc_curve(y_test, y_proba_sk)
    fpr_sm, tpr_sm, _ = roc_curve(y_test, y_proba_sm)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sk, tpr_sk, label=f'sklearn (AUC={roc_auc_score(y_test, y_proba_sk):.4f})')
    plt.plot(fpr_sm, tpr_sm, label=f'statsmodels (AUC={roc_auc_score(y_test, y_proba_sm):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic回归ROC曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logistic_roc.png', dpi=150)
    plt.show()
    
    # 4. Logit函数可视化
    z = np.linspace(-10, 10, 100)
    logistic = 1 / (1 + np.exp(-z))
    
    plt.figure(figsize=(8, 6))
    plt.plot(z, logistic, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='P=0.5')
    plt.axvline(x=0, color='r', linestyle='--', label='Logit=0')
    plt.xlabel('z = β₀ + β₁X₁ + ...')
    plt.ylabel('P(Y=1)')
    plt.title('Logistic函数（Sigmoid）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logistic_function.png', dpi=150)
    plt.show()
    
    return logit_result


def demo_multinomial_logistic():
    """
    多项Logistic回归演示
    """
    print("\n" + "=" * 60)
    print("多项Logistic回归演示（多分类）")
    print("=" * 60)
    
    # 使用鸢尾花数据集（3类）
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. sklearn多项Logistic
    print("\n--- sklearn多项Logistic ---")
    multi_log = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    multi_log.fit(X_train_scaled, y_train)
    
    y_pred = multi_log.predict(X_test_scaled)
    y_proba = multi_log.predict_proba(X_test_scaled)
    
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 查看系数（每个类别一组）
    print("\n各类别系数（相对于最后一个类别）：")
    for i, class_name in enumerate(class_names[:-1]):
        print(f"\n{class_name} vs {class_names[-1]}:")
        for j, feature in enumerate(feature_names):
            print(f"  {feature}: {multi_log.coef_[i, j]:.4f}")
    
    # 2. 概率预测可视化
    plt.figure(figsize=(10, 6))
    
    # 对测试集的预测概率
    proba_df = pd.DataFrame(y_proba, columns=class_names)
    proba_df['true_class'] = [class_names[i] for i in y_test]
    
    # 各类别平均预测概率
    for class_name in class_names:
        mask = proba_df['true_class'] == class_name
        avg_proba = proba_df.loc[mask, class_names].mean()
        plt.bar(range(len(class_names)), avg_proba, alpha=0.5, label=f'真实={class_name}')
    
    plt.xticks(range(len(class_names)), class_names)
    plt.xlabel('预测类别')
    plt.ylabel('平均预测概率')
    plt.title('多项Logistic各类别预测概率')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('multinomial_proba.png', dpi=150)
    plt.show()
    
    # 3. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('多项Logistic混淆矩阵')
    plt.tight_layout()
    plt.savefig('multinomial_cm.png', dpi=150)
    plt.show()
    
    return multi_log


def demo_ordered_logistic():
    """
    有序Logistic回归演示
    """
    print("\n" + "=" * 60)
    print("有序Logistic回归演示")
    print("=" * 60)
    
    # 创建有序数据（模拟信用评级）
    np.random.seed(42)
    n = 500
    
    # 特征
    income = np.random.lognormal(mean=10, sigma=1, size=n)
    credit_score = np.random.randint(300, 850, n)
    debt_ratio = np.random.uniform(0.1, 0.8, n)
    
    # 有序响应变量（信用评级：1=差, 2=中, 3=良, 4=优）
    # 基于特征的组合生成
    score = (credit_score - 600) / 200 - debt_ratio + np.log(income) / 10
    rating = pd.cut(score, bins=[-np.inf, -0.5, 0.5, 1.5, np.inf], 
                   labels=[1, 2, 3, 4]).astype(int)
    
    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'debt_ratio': debt_ratio,
        'rating': rating
    })
    
    print(f"\n评级分布:")
    print(df['rating'].value_counts().sort_index())
    
    # 使用statsmodels的OrderedModel
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        
        features = ['income', 'credit_score', 'debt_ratio']
        X = df[features]
        y = df['rating']
        
        # 有序Logistic模型
        ordered_model = OrderedModel(y, X, distr='logit')
        ordered_result = ordered_model.fit()
        
        print("\n有序Logistic回归结果:")
        print(ordered_result.summary())
        
        # 阈值（截断点）
        print("\n阈值参数:")
        for i, threshold in enumerate(ordered_result.params[:3]):
            print(f"  θ_{i+1}: {threshold:.4f}")
        
        # 系数
        print("\n回归系数:")
        for feature, coef in zip(features, ordered_result.params[3:]):
            print(f"  {feature}: {coef:.4f}")
        
        # 预测
        pred_probs = ordered_result.predict(X)
        pred_rating = pred_probs.idxmax(axis=1) + 1  # 选择概率最大的类别
        
        accuracy = accuracy_score(y, pred_rating)
        print(f"\n准确率: {accuracy:.4f}")
        
        # 预测概率可视化
        plt.figure(figsize=(10, 6))
        for i in range(4):
            plt.hist(pred_probs.iloc[:, i], bins=30, alpha=0.5, 
                    label=f'评级{i+1}')
        plt.xlabel('预测概率')
        plt.ylabel('频数')
        plt.title('有序Logistic各类别预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ordered_logistic_proba.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("statsmodels OrderedModel不可用，使用替代方法")
        
        # 使用sklearn的LogisticRegression（非最优）
        from sklearn.linear_model import LogisticRegression
        
        X_train, X_test, y_train, y_test = train_test_split(
            df[['income', 'credit_score', 'debt_ratio']], 
            df['rating'], 
            test_size=0.3, 
            random_state=42
        )
        
        multi_log = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        multi_log.fit(X_train, y_train)
        
        y_pred = multi_log.predict(X_test)
        print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    
    return df


def demo_probit_regression():
    """
    Probit回归演示
    """
    print("\n" + "=" * 60)
    print("Probit回归演示")
    print("=" * 60)
    
    # 使用之前的数据
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # statsmodels Probit
    print("\n--- Probit回归 ---")
    X_sm = sm.add_constant(X_train)
    probit_model = sm.Probit(y_train, X_sm)
    probit_result = probit_model.fit()
    
    print(probit_result.summary())
    
    # 预测
    y_proba_probit = probit_result.predict(sm.add_constant(X_test))
    
    # 对比Logistic
    logit_model = sm.Logit(y_train, X_sm)
    logit_result = logit_model.fit()
    y_proba_logit = logit_result.predict(sm.add_constant(X_test))
    
    print("\n模型对比:")
    print(f"Probit AUC: {roc_auc_score(y_test, y_proba_probit):.4f}")
    print(f"Logit AUC: {roc_auc_score(y_test, y_proba_logit):.4f}")
    
    # Logistic vs Probit函数对比
    z = np.linspace(-4, 4, 100)
    logistic = 1 / (1 + np.exp(-z))
    probit = stats.norm.cdf(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, logistic, 'b-', linewidth=2, label='Logistic (σ(z))')
    plt.plot(z, probit, 'r--', linewidth=2, label='Probit (Φ(z))')
    plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('z = β₀ + β₁X₁ + ...')
    plt.ylabel('P(Y=1)')
    plt.title('Logistic vs Probit函数对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logistic_vs_probit.png', dpi=150)
    plt.show()
    
    # 预测概率对比散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_proba_logit, y_proba_probit, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Logistic预测概率')
    plt.ylabel('Probit预测概率')
    plt.title('Logistic vs Probit预测概率对比')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logit_probit_comparison.png', dpi=150)
    plt.show()
    
    return probit_result, logit_result


def demo_odds_ratio_calculation():
    """
    发生比比（Odds Ratio）计算与解释演示
    """
    print("\n" + "=" * 60)
    print("发生比比（Odds Ratio）详解")
    print("=" * 60)
    
    # 不同概率下的发生比和对数发生比
    probs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    
    odds_ratios_table = pd.DataFrame({
        'P(Y=1)': probs,
        'P(Y=0)': [1-p for p in probs],
        'Odds': [p/(1-p) for p in probs],
        'Log Odds': [np.log(p/(1-p)) for p in probs]
    })
    
    print("\n概率、发生比、对数发生比对照表:")
    print(odds_ratios_table.round(4))
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 概率vs发生比
    axes[0].plot(probs, [p/(1-p) for p in probs], 'o-')
    axes[0].set_xlabel('P(Y=1)')
    axes[0].set_ylabel('Odds')
    axes[0].set_title('概率 vs 发生比')
    axes[0].grid(True, alpha=0.3)
    
    # 概率vs对数发生比
    axes[1].plot(probs, [np.log(p/(1-p)) for p in probs], 'o-')
    axes[1].set_xlabel('P(Y=1)')
    axes[1].set_ylabel('Log Odds')
    axes[1].set_title('概率 vs 对数发生比')
    axes[1].grid(True, alpha=0.3)
    
    # 系数解释示例
    betas = np.linspace(-2, 2, 9)
    or_values = np.exp(betas)
    
    axes[2].bar(range(len(betas)), or_values)
    axes[2].set_xticks(range(len(betas)))
    axes[2].set_xticklabels([f'β={b:.1f}' for b in betas])
    axes[2].set_xlabel('系数β')
    axes[2].set_ylabel('Odds Ratio = exp(β)')
    axes[2].set_title('系数 vs 发生比比')
    axes[2].axhline(y=1, color='r', linestyle='--', label='OR=1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('odds_ratio_explanation.png', dpi=150)
    plt.show()
    
    # 系数解释规则
    print("\n系数解释规则:")
    print("  β > 0 → OR > 1 → X增加使P(Y=1)增加")
    print("  β = 0 → OR = 1 → X不影响P(Y=1)")
    print("  β < 0 → OR < 1 → X增加使P(Y=1)减少")
    print("\n具体解释:")
    print("  β = 0.5 → OR = 1.65 → X增加1单位，发生比增加65%")
    print("  β = -0.5 → OR = 0.61 → X增加1单位，发生比减少39%")
    
    return odds_ratios_table


def demo_statistical_inference():
    """
    Logistic回归统计推断演示
    """
    print("\n" + "=" * 60)
    print("Logistic回归统计推断")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n = 500
    
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    
    # 真实系数
    beta_true = np.array([-1, 0.5, 0])
    z = 1 + beta_true[0]*X1 + beta_true[1]*X2 + beta_true[2]*X3
    prob = 1 / (1 + np.exp(-z))
    y = np.random.binomial(1, prob)
    
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})
    
    # Logistic回归
    X = df[['X1', 'X2', 'X3']]
    X_sm = sm.add_constant(X)
    
    logit_model = sm.Logit(df['y'], X_sm)
    result = logit_model.fit()
    
    print(result.summary())
    
    # Wald检验详解
    print("\n--- Wald检验 ---")
    for i, (feature, coef, se, pval) in enumerate(zip(
        ['const', 'X1', 'X2', 'X3'],
        result.params,
        result.bse,
        result.pvalues
    )):
        z_stat = coef / se
        print(f"{feature}:")
        print(f"  系数: {coef:.4f}")
        print(f"  标准误: {se:.4f}")
        print(f"  z统计量: {z_stat:.4f}")
        print(f"  p值: {pval:.4f}")
        print(f"  显著性: {'显著' if pval < 0.05 else '不显著'}")
    
    # 似然比检验
    print("\n--- 似然比检验 ---")
    # 全模型
    ll_full = result.llf
    
    # 空模型（只有常数项）
    null_model = sm.Logit(df['y'], sm.add_constant(df[['y']]))
    null_result = null_model.fit()
    ll_null = null_result.llf
    
    LR_stat = 2 * (ll_full - ll_null)
    LR_pval = 1 - stats.chi2.cdf(LR_stat, df=3)
    
    print(f"全模型对数似然: {ll_full:.4f}")
    print(f"空模型对数似然: {ll_null:.4f}")
    print(f"LR统计量: {LR_stat:.4f}")
    print(f"LR检验p值: {LR_pval:.4f}")
    
    # 拟合优度
    print("\n--- 拟合优度指标 ---")
    print(f"Log-Likelihood: {result.llf:.4f}")
    print(f"AIC: {result.aic:.4f}")
    print(f"BIC: {result.bic:.4f}")
    print(f"McFadden R²: {result.prsquared:.4f}")
    
    # 伪R²计算
    R2_CS = 1 - np.exp(-2*(ll_full - ll_null)/n)
    R2_N = (ll_full - ll_null) / (ll_full + n)
    
    print(f"Cox-Snell R²: {R2_CS:.4f}")
    print(f"Nagelkerke R²: {R2_N:.4f}")
    
    # Hosmer-Lemeshow检验
    print("\n--- Hosmer-Lemeshow检验 ---")
    try:
        from statsmodels.stats.diagnostic import linear_lmtest
        # 手动实现HL检验
        y_pred = result.predict(X_sm)
        
        # 分成10组
        df['pred_prob'] = y_pred
        df['group'] = pd.qcut(df['pred_prob'], 10, labels=False, duplicates='drop')
        
        hl_data = df.groupby('group').agg({
            'y': ['sum', 'count'],
            'pred_prob': 'mean'
        })
        hl_data.columns = ['observed', 'total', 'expected_prob']
        hl_data['expected'] = hl_data['expected_prob'] * hl_data['total']
        hl_data['expected_not'] = hl_data['total'] - hl_data['expected']
        hl_data['observed_not'] = hl_data['total'] - hl_data['observed']
        
        # 计算HL统计量
        hl_stat = ((hl_data['observed'] - hl_data['expected'])**2 / hl_data['expected'] +
                   (hl_data['observed_not'] - hl_data['expected_not'])**2 / hl_data['expected_not']).sum()
        
        hl_pval = 1 - stats.chi2.cdf(hl_stat, df=len(hl_data)-2)
        
        print(f"HL统计量: {hl_stat:.4f}")
        print(f"HL检验p值: {hl_pval:.4f}")
        print(f"拟合优度: {'良好' if hl_pval > 0.05 else '拟合不足'}")
        
    except Exception as e:
        print(f"HL检验计算失败: {e}")
    
    return result


def main():
    """
    主函数
    """
    print("定性响应变量回归模型完整演示")
    print("=" * 60)
    
    # 1. 二分类Logistic
    logit_result = demo_binary_logistic()
    
    # 2. 多项Logistic
    multi_log = demo_multinomial_logistic()
    
    # 3. 有序Logistic
    df_ordered = demo_ordered_logistic()
    
    # 4. Probit回归
    probit_result, logit_result2 = demo_probit_regression()
    
    # 5. 发生比比解释
    odds_table = demo_odds_ratio_calculation()
    
    # 6. 统计推断
    inference_result = demo_statistical_inference()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("生成的图片:")
    print("  - logistic_roc.png")
    print("  - logistic_function.png")
    print("  - multinomial_proba.png")
    print("  - multinomial_cm.png")
    print("  - ordered_logistic_proba.png")
    print("  - logistic_vs_probit.png")
    print("  - logit_probit_comparison.png")
    print("  - odds_ratio_explanation.png")
    print("=" * 60)


if __name__ == '__main__':
    main()