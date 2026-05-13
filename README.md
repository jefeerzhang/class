# opencode

个人学习代码仓库，主要放金融数据分析、投资学、行为金融、计量方法和课程幻灯片相关材料。仓库里既有可运行脚本，也有 Markdown 讲义、PDF/HTML 幻灯片和少量 AI 助手技能文件。

## 目录结构

```text
opencode/
├── association_rules/                  # 金融产品关联规则分析
│   ├── data/                           # 示例交易数据
│   ├── docs/                           # 指南与分析报告
│   ├── images/                         # 可视化输出
│   ├── notebooks/                      # Jupyter Notebook
│   └── scripts/                        # 数据生成、规则挖掘、可视化脚本
├── behavioral_finance/                 # 行为金融讲义、论文笔记与幻灯片
├── credit_default_simulation/          # 信用违约模拟数据与分析脚本
├── investment/                         # 投资学资料
│   ├── assets/                         # 图片、PDF、HTML 资源
│   ├── cluster_analysis/               # 聚类分析讲义
│   ├── credit_rationing/               # 信贷配给材料
│   ├── data/                           # 投资分析示例数据
│   ├── docs/                           # 投资学笔记与讲义
│   ├── regression_analysis/            # 回归分析材料
│   ├── scripts/                        # 数据处理和分析脚本
│   └── tree_analysis/                  # 树模型讲义与演示脚本
├── slides/                             # 课程幻灯片项目
│   ├── bond-price-yield/
│   ├── clustering-guide/
│   ├── credit-rationing/
│   ├── regression/
│   └── regression-analysis-guide/
├── skills/                             # 本仓库保存的 Codex/Claude 技能包
├── reports/                            # 生成过程中的审查记录和报告
├── requirements.txt                    # Python 依赖
├── CLAUDE.md                           # 面向 Claude 的项目索引
└── QWEN.md                             # 面向 Qwen 的项目索引
```

## 快速开始

建议使用独立环境。

```bash
cd opencode
pip install -r requirements.txt
```

## 常用入口

关联规则项目：

```bash
python association_rules/scripts/01_data/generate_customer_data.py
python association_rules/scripts/02_analysis/finance_analysis.py
python association_rules/scripts/02_analysis/fpgrowth_analysis.py
python association_rules/scripts/03_visualization/visualize_rules.py
```

投资学聚类分析：

```bash
python investment/scripts/generate_bank_data.py
python investment/scripts/kmeans_analysis.py
```

信用违约模拟：

```bash
python credit_default_simulation/generate_data.py
python credit_default_simulation/credit_default_analysis.py
```

Notebook：

```bash
jupyter notebook association_rules/notebooks/algorithm_comparison.ipynb
```

## 本地工具与忽略目录

以下目录属于本地工具、缓存或外部项目副本，不作为主仓库内容维护：

- `.learnings/`
- `.statamcp/`
- `.uploads/`
- `ppt-master/`

这些目录已写入 `.gitignore`。其中 `ppt-master/` 体量较大，当前保留在工作区，作为独立工具使用。

## 说明

- 本仓库偏学习、备课和复现实验，脚本输出通常保存在各自子目录中。
- 更完整的脚本入口和资料索引见 [CLAUDE.md](./CLAUDE.md)。
