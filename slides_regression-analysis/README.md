# 回归分析及其应用 - Marp 演示文稿项目

## 项目概述

本项目将《回归分析方法完整指南.md》转换为基于 Marp 的演示文稿，采用 gaia 主题（Times New Roman + 宋体，蓝色下划线装饰）。

## 目录结构

```
slides_regression-analysis/
├── 05_final/              # 最终产出文件
│   ├── presentation.md    # Marp Markdown 源文件（22页）
│   ├── slides.html        # HTML 预览版本
│   └── slides.pdf         # PDF 版本
└── README.md              # 本文件
```

## 文件说明

### 1. `presentation.md` - 源文件
- **格式**：Marp Markdown
- **页数**：22页
- **主题**：gaia
- **内容**：回归分析的完整知识体系，包括：
  - 概述与分类
  - 定性响应变量回归（Logistic回归）
  - 传统多元线性回归
  - 正则化方法（Ridge、Lasso、Elastic Net）
  - 降维方法（PCR、PLS）
  - 模型选择策略
  - 金融领域应用
  - 代码实战概述
  - 最佳实践与注意事项

### 2. `slides.html` - HTML 预览
- **用途**：在浏览器中查看演示文稿
- **使用方式**：双击文件或使用浏览器打开
- **功能**：支持全屏模式、键盘导航

### 3. `slides.pdf` - PDF 版本
- **用途**：打印、分发或离线查看
- **特点**：静态格式，保留所有样式

## 原始来源

- **原始文档**：`investment/regression_analysis/docs/回归分析方法完整指南.md`
- **内容长度**：3404行，涵盖回归分析的完整理论、方法和应用

## 演示文稿特点

1. **内容精简**：从3404行原文精简为22页幻灯片
2. **结构清晰**：按照教学逻辑组织内容
3. **视觉优化**：
   - 每页严格控制内容密度
   - 使用表格对比关键概念
   - 添加生活比喻和金融应用案例
4. **技术特色**：
   - 支持数学公式（LaTeX语法）
   - 中英文混排优化
   - 响应式布局

## 预览与编辑

### 预览演示文稿
```bash
# 使用 Marp CLI 预览
npx @marp-team/marp-cli -s slides_regression-analysis/05_final/presentation.md

# 直接打开 HTML 文件
start slides_regression-analysis/05_final/slides.html
```

### 编辑源文件
```bash
# 使用任何文本编辑器编辑
code slides_regression-analysis/05_final/presentation.md
```

### 重新生成输出
```bash
# 重新导出 HTML
npx @marp-team/marp-cli slides_regression-analysis/05_final/presentation.md -o slides_regression-analysis/05_final/slides.html --html

# 重新导出 PDF
npx @marp-team/marp-cli slides_regression-analysis/05_final/presentation.md -o slides_regression-analysis/05_final/slides.pdf --pdf
```

## 最近更新

- **日期**：2026年4月23日
- **版本**：v2.0
- **更新内容**：
  - 从17页扩展到22页，优化内容密度
  - 添加CSS样式调整表格显示
  - 补充统计推断、交叉验证等关键内容
  - 优化视觉层次和强调元素

## 技术支持

- **Marp**：[https://marp.app/](https://marp.app/)
- **Marp CLI**：[https://github.com/marp-team/marp-cli](https://github.com/marp-team/marp-cli)
- **gaia主题**：自定义CSS主题，包含Times New Roman字体和蓝色下划线装饰