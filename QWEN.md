# QWEN.md - 项目配置

## 项目概述

个人学习代码仓库，主要研究金融产品场景下的关联规则分析（Association Rule Mining）。

## 技术栈

- **语言**: Python 3.x
- **核心库**: pandas, numpy, scikit-learn, mlxtend
- **可视化**: matplotlib, networkx
- **开发环境**: Jupyter Notebook

## 代码风格

- 使用有意义的变量名（中文注释可接受）
- 函数添加简短 docstring
- 代码单元格保持简洁，便于在 Jupyter 中运行

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook

# 运行 Python 脚本
python scripts/<script_name>.py
```

## 项目结构

```
opencode/
├── association_rules/    # 关联规则分析主目录
│   ├── data/            # 数据文件
│   ├── docs/            # 文档
│   ├── images/          # 图片
│   ├── notebooks/       # Jupyter notebooks
│   └── scripts/         # Python 脚本
├── behavioral_finance/   # 行为金融学
│   ├── assets/          # 资源文件
│   ├── data/            # 数据文件
│   ├── docs/            # 文档
│   ├── notebooks/       # Jupyter notebooks
│   └── scripts/         # Python 脚本
└── investment/          # 投资分析
    ├── assets/          # 资源文件
    ├── data/            # 数据文件
    ├── docs/            # 文档
    ├── notebooks/       # Jupyter notebooks
    └── scripts/         # Python 脚本
```

## 偏好设置

- 使用中文进行交流和注释
- 代码输出保持简洁
- 优先使用 pandas 进行数据处理