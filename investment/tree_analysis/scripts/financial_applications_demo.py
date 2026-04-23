"""
树类方法金融应用演示脚本
包含：信用评分、股票预测模拟、欺诈检测示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                             GradientBoostingClassifier, IsolationForest)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, mean_squared_error, r2_score)
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_credit_data(n_samples=5000):
    """
    生成模拟信用评分数据
    """
    np.random.seed(42)
    
    # 特征生成
    data = {
        'annual_income': np.random.lognormal(mean=10, sigma=1, size=n_samples),  # 年收入
        'debt_to_income': np.random.uniform(0.1, 0.5, n_samples),                # 负债收入比
        'credit_history_years': np.random.exponential(scale=5, size=n_samples),   # 信用历史年限
        'num_credit_lines': np.random.randint(1, 20, n_samples),                  # 信用额度数量
        'num_late_payments': np.random.choice([0, 1, 2, 3, 5, 10], n_samples,     # 逾期次数
                                               p=[0.5, 0.25, 0.15, 0.05, 0.04, 0.01]),
        'credit_utilization': np.random.beta(2, 5, n_samples),                    # 信用使用率
        'employment_years': np.random.exponential(scale=3, size=n_samples),       # 工作年限
        'loan_amount': np.random.lognormal(mean=9, sigma=1, size=n_samples),      # 贷款金额
        'age': np.random.randint(18, 75, n_samples),                              # 年龄
        'home_ownership': np.random.choice(['OWN', 'MORTGAGE', 'RENT'], n_samples, # 房产状况
                                           p=[0.2, 0.5, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # 类别编码
    df['home_ownership_encoded'] = df['home_ownership'].map({
        'OWN': 2, 'MORTGAGE': 1, 'RENT': 0
    })
    
    # 生成目标变量（违约与否）基于特征
    # 违约概率与负债收入比、逾期次数正相关，与收入、信用历史负相关
    prob_default = (
        0.1 + 
        0.3 * df['debt_to_income'] + 
        0.05 * df['num_late_payments'] - 
        0.01 * np.log(df['annual_income']) - 
        0.01 * df['credit_history_years'] +
        0.2 * df['credit_utilization'] +
        np.random.normal(0, 0.1, n_samples)
    )
    
    prob_default = np.clip(prob_default, 0, 1)
    df['default'] = (prob_default > 0.5).astype(int)
    
    return df


def demo_credit_scoring():
    """
    信用评分模型演示
    """
    print("=" * 60)
    print("信用评分模型演示")
    print("=" * 60)
    
    # 生成数据
    df = generate_credit_data(5000)
    
    print(f"\n数据集统计:")
    print(f"总样本数: {len(df)}")
    print(f"违约率: {df['default'].mean():.2%}")
    
    # 特征选择
    feature_cols = [
        'annual_income', 'debt_to_income', 'credit_history_years',
        'num_credit_lines', 'num_late_payments', 'credit_utilization',
        'employment_years', 'loan_amount', 'age', 'home_ownership_encoded'
    ]
    
    X = df[feature_cols]
    y = df['default']
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 模型列表
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 处理类别不平衡（针对GBDT）
        if name == 'Gradient Boosting':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 评估
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        results[name] = {
            'metrics': metrics,
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 性能对比
    names = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_names))
    width = 0.25
    
    for i, name in enumerate(names):
        values = [results[name]['metrics'][m] for m in metrics_names]
        axes[0, 0].bar(x + i * width, values, width, label=name)
    
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(['准确率', '精确率', '召回率', 'F1', 'AUC'])
    axes[0, 0].set_ylabel('得分')
    axes[0, 0].set_title('信用评分模型性能对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # ROC曲线
    from sklearn.metrics import roc_curve
    for name in names:
        fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
        axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={results[name]['metrics']['auc']:.4f})")
    
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_xlabel('假阳性率')
    axes[0, 1].set_ylabel('真阳性率')
    axes[0, 1].set_title('ROC曲线对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 特征重要性
    rf_model = results['Random Forest']['model']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1, 0].barh(range(len(importance)), importance['importance'])
    axes[1, 0].set_yticks(range(len(importance)))
    axes[1, 0].set_yticklabels(importance['feature'])
    axes[1, 0].set_xlabel('重要性')
    axes[1, 0].set_title('特征重要性 (随机森林)')
    
    # 混淆矩阵（随机森林）
    cm = confusion_matrix(y_test, results['Random Forest']['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['正常', '违约'],
                yticklabels=['正常', '违约'])
    axes[1, 1].set_xlabel('预测标签')
    axes[1, 1].set_ylabel('真实标签')
    axes[1, 1].set_title('混淆矩阵 (随机森林)')
    
    plt.tight_layout()
    plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 详细分类报告
    print("\n" + "=" * 60)
    print("随机森林分类报告:")
    print("=" * 60)
    print(classification_report(y_test, results['Random Forest']['y_pred'],
                               target_names=['正常', '违约']))
    
    return results, df


def generate_stock_data(n_days=1000):
    """
    生成模拟股票价格数据
    """
    np.random.seed(42)
    
    # 基础参数
    initial_price = 100
    volatility = 0.02
    drift = 0.0001
    
    # 生成价格序列（几何布朗运动）
    returns = np.random.normal(drift, volatility, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # 创建DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.lognormal(mean=15, sigma=1, size=n_days)
    })
    
    # 添加技术指标
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_60'] = df['close'].rolling(60).mean()
    
    df['ma_cross_5_20'] = (df['ma_5'] > df['ma_20']).astype(int)
    df['ma_cross_20_60'] = (df['ma_20'] > df['ma_60']).astype(int)
    
    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 成交量比率
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_5']
    
    # 目标变量：未来5日收益率
    df['target_return'] = df['return_1d'].shift(-5)
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    return df.dropna()


def demo_stock_prediction():
    """
    股票收益预测演示
    """
    print("\n" + "=" * 60)
    print("股票收益预测演示")
    print("=" * 60)
    
    # 生成数据
    df = generate_stock_data(1000)
    
    print(f"\n数据集统计:")
    print(f"总天数: {len(df)}")
    print(f"上涨天数比例: {df['target_direction'].mean():.2%}")
    
    # 特征选择
    feature_cols = [
        'return_1d', 'return_5d', 'return_20d',
        'ma_cross_5_20', 'ma_cross_20_60',
        'volatility_5d', 'volatility_20d',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'volume_ratio'
    ]
    
    X = df[feature_cols]
    
    # 回归目标：未来收益率
    y_return = df['target_return']
    
    # 分类目标：涨跌方向
    y_direction = df['target_direction']
    
    # 时间序列分割（使用前70%训练，后30%测试）
    split_idx = int(len(df) * 0.7)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_return_train = y_return.iloc[:split_idx]
    y_return_test = y_return.iloc[split_idx:]
    y_direction_train = y_direction.iloc[:split_idx]
    y_direction_test = y_direction.iloc[split_idx:]
    
    # 1. 回归模型：预测收益率
    print("\n--- 收益率预测 (回归) ---")
    
    reg_models = {
        'Decision Tree': DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=20, random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
    }
    
    reg_results = {}
    
    for name, model in reg_models.items():
        model.fit(X_train, y_return_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_return_test, y_pred)
        r2 = r2_score(y_return_test, y_pred)
        
        reg_results[name] = {
            'mse': mse,
            'r2': r2,
            'y_pred': y_pred,
            'model': model
        }
        
        print(f"{name}: MSE={mse:.6f}, R²={r2:.4f}")
    
    # 2. 分类模型：预测涨跌方向
    print("\n--- 涨跌方向预测 (分类) ---")
    
    clf_models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
    }
    
    clf_results = {}
    
    for name, model in clf_models.items():
        model.fit(X_train, y_direction_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_direction_test, y_pred)
        auc = roc_auc_score(y_direction_test, y_proba)
        
        clf_results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'model': model
        }
        
        print(f"{name}: 准确率={accuracy:.4f}, AUC={auc:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 股票价格走势
    axes[0, 0].plot(df['date'], df['close'], label='收盘价')
    axes[0, 0].axvline(df['date'].iloc[split_idx], color='r', linestyle='--',
                       label='训练/测试分割')
    axes[0, 0].set_xlabel('日期')
    axes[0, 0].set_ylabel('价格')
    axes[0, 0].set_title('股票价格走势')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 收益率预测 vs 实际
    axes[0, 1].scatter(y_return_test, reg_results['Random Forest']['y_pred'], 
                       alpha=0.5, s=10)
    axes[0, 1].plot([y_return_test.min(), y_return_test.max()],
                    [y_return_test.min(), y_return_test.max()], 'r--')
    axes[0, 1].set_xlabel('实际收益率')
    axes[0, 1].set_ylabel('预测收益率')
    axes[0, 1].set_title(f'收益率预测 (R²={reg_results["Random Forest"]["r2"]:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 分类准确率对比
    names = list(clf_results.keys())
    accuracies = [clf_results[n]['accuracy'] for n in names]
    aucs = [clf_results[n]['auc'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, accuracies, width, label='准确率')
    axes[1, 0].bar(x + width/2, aucs, width, label='AUC')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_ylabel('得分')
    axes[1, 0].set_title('涨跌方向预测性能')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 预测信号叠加
    test_dates = df['date'].iloc[split_idx:]
    pred_signal = clf_results['Random Forest']['y_pred']
    
    axes[1, 1].plot(test_dates, df['close'].iloc[split_idx:], label='收盘价')
    
    # 标记预测上涨的点
    up_points = df['close'].iloc[split_idx:].iloc[pred_signal == 1]
    up_dates = test_dates.iloc[pred_signal == 1]
    axes[1, 1].scatter(up_dates, up_points, c='green', marker='^', s=20, 
                       label='预测上涨', alpha=0.5)
    
    # 标记预测下跌的点
    down_points = df['close'].iloc[split_idx:].iloc[pred_signal == 0]
    down_dates = test_dates.iloc[pred_signal == 0]
    axes[1, 1].scatter(down_dates, down_points, c='red', marker='v', s=20,
                       label='预测下跌', alpha=0.5)
    
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('价格')
    axes[1, 1].set_title('预测信号叠加')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stock_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 特征重要性
    rf_clf = clf_results['Random Forest']['model']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排序:")
    print(importance)
    
    return reg_results, clf_results, df


def generate_fraud_data(n_samples=10000):
    """
    生成模拟欺诈交易数据
    """
    np.random.seed(42)
    
    # 正常交易
    n_normal = int(n_samples * 0.99)
    n_fraud = n_samples - n_normal
    
    data = {
        'transaction_amount': np.concatenate([
            np.random.exponential(scale=100, size=n_normal),  # 正常交易金额
            np.random.exponential(scale=500, size=n_fraud)    # 欺诈交易金额（较高）
        ]),
        'transaction_time': np.concatenate([
            np.random.uniform(6, 22, n_normal),               # 正常交易时间（白天）
            np.random.uniform(0, 6, n_fraud)                  # 欺诈交易时间（夜间）
        ]),
        'merchant_category': np.concatenate([
            np.random.randint(1, 20, n_normal),
            np.random.choice([1, 5, 10], n_fraud)              # 欺诈集中在特定类别
        ]),
        'distance_from_home': np.concatenate([
            np.random.exponential(scale=10, n_normal),
            np.random.exponential(scale=100, n_fraud)          # 欺诈交易距离较远
        ]),
        'distance_from_last_transaction': np.concatenate([
            np.random.exponential(scale=5, n_normal),
            np.random.exponential(scale=50, n_fraud)
        ]),
        'ratio_to_median_price': np.concatenate([
            np.random.beta(2, 5, n_normal),
            np.random.beta(5, 2, n_fraud)                      # 欺诈交易金额异常
        ]),
        'repeat_card_use': np.concatenate([
            np.random.randint(0, 10, n_normal),
            np.random.randint(0, 2, n_fraud)                   # 欺诈交易少重复使用
        ]),
        'card_age_days': np.concatenate([
            np.random.randint(30, 3650, n_normal),
            np.random.randint(0, 30, n_fraud)                  # 欺诈多发生在新卡
        ])
    }
    
    df = pd.DataFrame(data)
    
    # 标签
    df['is_fraud'] = np.concatenate([
        np.zeros(n_normal),
        np.ones(n_fraud)
    ])
    
    return df


def demo_fraud_detection():
    """
    欺诈检测演示
    """
    print("\n" + "=" * 60)
    print("欺诈检测演示")
    print("=" * 60)
    
    # 生成数据
    df = generate_fraud_data(10000)
    
    print(f"\n数据集统计:")
    print(f"总交易数: {len(df)}")
    print(f"欺诈交易数: {df['is_fraud'].sum()}")
    print(f"欺诈比例: {df['is_fraud'].mean():.2%}")
    
    feature_cols = [
        'transaction_amount', 'transaction_time', 'merchant_category',
        'distance_from_home', 'distance_from_last_transaction',
        'ratio_to_median_price', 'repeat_card_use', 'card_age_days'
    ]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n测试集欺诈交易数: {y_test.sum()}")
    
    # 1. 监督学习方法
    print("\n--- 监督学习方法 ---")
    
    # 随机森林（类别平衡）
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    print("\n随机森林结果:")
    print(f"准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"精确率: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1: {f1_score(y_test, y_pred_rf):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
    
    # 2. 异常检测方法
    print("\n--- 异常检测方法 (Isolation Forest) ---")
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=df['is_fraud'].mean(),  # 预期异常比例
        random_state=42
    )
    
    # 只用正常交易训练
    X_train_normal = X_train[y_train == 0]
    iso_forest.fit(X_train_normal)
    
    y_pred_iso = iso_forest.predict(X_test)
    y_pred_iso = (y_pred_iso == -1).astype(int)  # -1表示异常
    
    print("\nIsolation Forest结果:")
    print(f"准确率: {accuracy_score(y_test, y_pred_iso):.4f}")
    print(f"精确率: {precision_score(y_test, y_pred_iso):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_iso):.4f}")
    print(f"F1: {f1_score(y_test, y_pred_iso):.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 混淆矩阵 - 随机森林
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['正常', '欺诈'],
                yticklabels=['正常', '欺诈'])
    axes[0, 0].set_xlabel('预测标签')
    axes[0, 0].set_ylabel('真实标签')
    axes[0, 0].set_title('混淆矩阵 - 随机森林')
    
    # 混淆矩阵 - Isolation Forest
    cm_iso = confusion_matrix(y_test, y_pred_iso)
    sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1],
                xticklabels=['正常', '欺诈'],
                yticklabels=['正常', '欺诈'])
    axes[0, 1].set_xlabel('预测标签')
    axes[0, 1].set_ylabel('真实标签')
    axes[0, 1].set_title('混淆矩阵 - Isolation Forest')
    
    # ROC曲线
    from sklearn.metrics import roc_curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    axes[1, 0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.4f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_xlabel('假阳性率')
    axes[1, 0].set_ylabel('真阳性率')
    axes[1, 0].set_title('ROC曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1, 1].barh(range(len(importance)), importance['importance'])
    axes[1, 1].set_yticks(range(len(importance)))
    axes[1, 1].set_yticklabels(importance['feature'])
    axes[1, 1].set_xlabel('重要性')
    axes[1, 1].set_title('特征重要性 (欺诈检测)')
    
    plt.tight_layout()
    plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 详细分类报告
    print("\n" + "=" * 60)
    print("随机森林分类报告:")
    print("=" * 60)
    print(classification_report(y_test, y_pred_rf, target_names=['正常', '欺诈']))
    
    return rf, iso_forest, df


def main():
    """
    主函数
    """
    print("树类方法金融应用完整演示")
    print("=" * 60)
    
    # 1. 信用评分
    credit_results, credit_df = demo_credit_scoring()
    
    # 2. 股票预测
    stock_reg_results, stock_clf_results, stock_df = demo_stock_prediction()
    
    # 3. 欺诈检测
    rf_fraud, iso_fraud, fraud_df = demo_fraud_detection()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("结果图片已保存:")
    print("  - credit_scoring_results.png")
    print("  - stock_prediction_results.png")
    print("  - fraud_detection_results.png")
    print("=" * 60)


if __name__ == '__main__':
    main()