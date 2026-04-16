# opencode

个人学习代码仓库：面向金融产品场景的数据分析与关联规则挖掘。

## 目录结构

```
opencode/
├── association_rules/      # 关联规则分析（Apriori / FP-Growth）
│   ├── data/              # 数据文件
│   ├── docs/              # 文档
│   ├── images/            # 图片
│   ├── notebooks/         # Jupyter notebooks
│   └── scripts/           # Python 脚本
├── behavioral_finance/     # 行为金融学
│   ├── assets/            # 资源文件（图片、视频等）
│   ├── data/              # 数据文件
│   ├── docs/              # 文档
│   ├── notebooks/         # Jupyter notebooks
│   └── scripts/           # Python 脚本
└── investment/            # 投资分析
    ├── assets/            # 资源文件（PDF、HTML 等）
    ├── data/              # 数据文件
    ├── docs/              # 文档
    ├── notebooks/         # Jupyter notebooks
    └── scripts/           # Python 脚本
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook
```

## 模块说明

| 模块 | 说明 |
|------|------|
| `association_rules` | 关联规则挖掘：数据生成、Apriori、FP-Growth、可视化 |
| `behavioral_finance` | 行为金融学理论与案例分析 |
| `investment` | 投资分析：基金投资、聚类分析、IPO 实务等 |
