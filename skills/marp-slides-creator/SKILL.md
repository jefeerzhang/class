---
name: marp-slides-creator
description: 专业Marp演示文稿制作助手。提供简洁工作流程：内容分析、slides制作、多维度审阅、终稿输出。所有产出物保存在项目工作文件夹的05_final目录中。当用户提到"制作slides"、"做PPT"、"演示文稿"、"Marp"、"幻灯片"、"presentation"等关键词时自动启用。Professional Marp presentation assistant with streamlined workflow: content analysis, slide creation, multi-dimensional review, and final output. All outputs saved in 05_final directory.
allowed-tools: Read, Write, Edit, Bash, Task, Glob, question
---

# Marp Slides 制作助手

## 技能概述

本技能提供简洁的Marp演示文稿制作流程：内容分析、slides制作、多维度审阅、终稿输出。对于中文演示文稿，额外包含中文语言规范审阅阶段。所有产出物保存在项目工作文件夹的05_final目录中。

**支持多种主题风格**：内置 15 款精选主题，涵盖学术、商务、创意等场景。详见 `themes/README.md`。

## 核心原则

### Slides设计原则

| 原则 | 说明 | 要点 |
|------|------|------|
| 一页一点 | 每页只讲一个核心观点 | 避免信息过载 |
| 视觉层次 | 标题→要点→细节 | 使用粗体、列表、引用区分 |
| 留白充足 | 内容不超过页面60% | 给观众呼吸空间 |
| 文字精简 | 每页文字不超过8行 | 关键词优于完整句子 |

### 内容密度控制

- **标题页**：主标题 + 副标题 + 作者信息
- **内容页**：标题 + 3-5个要点（每点1-2行）
- **对比页**：两栏对比，每栏不超过4点
- **总结页**：3-5个核心要点

### 字体大小规范

不同主题字体大小差显著，制作时需根据主题调整：

| 元素 | 推荐范围 | 说明 |
|------|----------|------|
| 标题页标题 | 40-60px / 2.5-3em | 主标题醒目 |
| 内容页标题 | 24-32px / 1.5-2em | 避免过大 |
| 正文内容 | 18-24px / 1-1.4em | 阅读舒适 |
| 要点列表 | 16-20px / 0.9-1.1em | 略小于正文 |
| 引用块 | 16-20px / 0.9-1.1em | 与列表相近 |
| 注释/页脚 | 12-14px / 0.7-0.8em | 最小字号 |

**各主题字体大小参考**：

| 主题 | 正文字号 | 标题字号 | 备注 |
|------|----------|----------|------|
| companySZ | 23-24px | 36-40px | 企业正式风格 |
| companyLightBlue | 21-23px | 36-40px | 企业清新风格 |
| zju | 21-24px | 36px | 浙大官方风格 |
| jobs | 34px | 48px+ | 发布会风格，字号偏大 |
| turing | 27-32px | 46px | 技术风格 |
| einstein | 27px | 32px | 科学风格 |
| academic | 0.9-1em | - | 使用相对单位 |
| beam | 0.6-0.65em | - | 最小字号主题 |
| graph_paper | 0.65-0.9em | - | 手写笔记感 |
| simple | 100% | 200% | 极简风格 |
| gaia | 1-1.5em | 1.5em | 自定义主题 |

**安全值**：内容页正文不低于 `0.9em`（约14-16px），确保可读性。

**警告值**：低于 `0.7em`（约11px）可能导致阅读困难。

> 注：Marp 默认 1em ≈ 20px。实际像素值取决于主题基础设置。

## 四阶段工作流程

严格按照以下流程执行，每阶段结束后询问用户确认：

### 阶段一：工作空间初始化与主题选择

**触发**：用户请求制作slides时立即执行

**执行步骤**：

1. **环境检查**：
   在继续之前，先检查 marp-cli 是否可用：
   ```bash
   npx @marp-team/marp-cli --version 2>/dev/null || echo "需要安装 Node.js 和 marp-cli"
   ```
   如果不可用，提示用户运行：
   ```bash
   npm install -g @marp-team/marp-cli
   # 或使用 npx（无需全局安装）
   ```

2. **确定工作模式**：
   询问用户选择工作模式：
   - **标准模式**（默认）：完整五阶段流程，包含内容分析确认和多维度审阅
   - **快速模式**：跳过内容分析确认和审阅阶段，直接出稿。适合 5-10 页的简单演示

3. **确定项目名称**：
   - 根据用户提供的主题/文件名确定项目名称
   - 使用简短的英文或拼音命名，用连字符连接
   - 示例：`academic-writing`、`ai-introduction`、`product-launch`
   - 如果用户已有项目目录（如 `slides/regression`），优先复用现有目录

4. **创建工作文件夹结构**：
   ```bash
   # 创建主工作文件夹
   mkdir -p slides/[项目名]/05_final
   mkdir -p slides/[项目名]/images    # 图片资源存放目录
   ```

5. **文件夹结构说明**：

   ```text
   slides/[项目名]/
   ├── images/                # 图片资源（与 .md 同级引用）
   │   └── figure_1.png
   └── 05_final/              # 最终产出
       ├── presentation.md       # 最终Markdown
       ├── slides.html           # HTML版本
       └── slides.pdf            # PDF版本
   ```

6. **主题选择**：

   使用 question 工具让用户选择主题风格：

   ```
   questions:
     - question: "请选择演示文稿的主题风格"
       header: "主题风格"
       multiSelect: false
       options:
         - label: "academic (学术风格)"
           description: "适合学术报告、论文答辩，maroon红色标题栏"
         - label: "beam (Beamer风格)"
           description: "仿LaTeX Beamer，适合学术演讲、技术研讨"
         - label: "jobs (乔布斯风格)"
           description: "Apple发布会风格，适合产品发布、商业演示"
         - label: "graph_paper (方格纸风格)"
           description: "技术分享、教学演示，手写笔记感"
         - label: "gaia (自定义主题)"
           description: "商务风格，Times New Roman + 宋体，蓝色下划线"
         - label: "companyLightBlue (企业浅蓝)"
           description: "清新浅蓝配色，专业商务风格"
         - label: "companySZ (企业深色)"
           description: "深色企业风格，正式庄重"
         - label: "zju (浙大风格)"
           description: "浙江大学官方配色，高校学术演示"
         - label: "turing (图灵风格)"
           description: "计算机科学、技术主题"
         - label: "simple (极简风格)"
           description: "极简设计，最少装饰"
         - label: "gradient (渐变背景)"
           description: "现代感渐变背景，创意展示"
         - label: "socrates (苏格拉底风格)"
           description: "哲学、人文学科演示"
         - label: "einstein (爱因斯坦风格)"
           description: "科学、物理学演示"
         - label: "border (边框简约)"
           description: "简约边框装饰，通用演示"
         - label: "academic-lightblue (学术浅蓝)"
           description: "学术风格浅蓝配色"
   ```

   **主题文件位置**：`themes/` 目录下所有 `.css` 文件

**输出**：工作文件夹路径、选择的主题名称，告知用户所有产出物将保存在此文件夹中

---

### 阶段二：内容分析与消化

**触发**：工作空间创建完成后，用户提供输入文件（PDF、论文、文档、笔记等）

**执行步骤**：

1. **初步阅读**：使用 Read 工具读取输入文件，了解整体结构
2. **深度分析**：
   - 提取核心论点和关键信息
   - 识别逻辑结构（问题-方案-结论 / 现状-分析-建议 等）
   - 标记重要数据、引用、案例
3. **内容分类**：
   - 核心观点（必须包含）
   - 支撑证据（选择性包含）
   - 背景信息（简化或省略）
4. **大纲生成**：
   - 确定slides数量（建议10-20页）
   - 规划每页主题和内容密度
   - 设计叙事流程

**输出**：直接向用户展示内容分析和大纲，请用户确认后继续

### 阶段三：Slides制作

**触发**：用户确认内容分析和大纲

**执行步骤**：

1. **读取模板**：使用 Read 工具读取 `marp-template.md`
2. **应用主题**：
   - 在 YAML frontmatter 中设置用户选择的主题：
     ```yaml
     ---
     marp: true
     theme: [用户选择的主题名]
     ---
     ```
   - 不同主题可能支持特殊的 class，参考 `themes/README.md`

3. **主题适配（自动调整）**：

   不同主题对标题格式有不同要求，制作 slides 时必须根据主题自动调整：

   | 主题 | 内容页标题格式 | 说明 |
   |------|----------------|------|
   | **beam** | `#` (h1) | h1 显示在顶部蓝色条中 |
   | **academic** | `#` (h1) | h1 用于页面标题 |
   | **jobs** | `#` (h1) | h1 有特殊下划线样式 |
   | **gaia** | `##` (h2) | h2 带蓝色下划线装饰 |
   | graph_paper | `##` (h2) | 标准 h2 标题 |
   | simple | `##` (h2) | 标准 h2 标题 |
   | companyLightBlue | `##` (h2) | 企业浅蓝风格 |
   | companySZ | `##` (h2) | 企业深色风格 |
   | zju | `##` (h2) | 浙大官方风格 |
   | turing | `##` (h2) | 图灵风格 |
   | gradient | `##` (h2) | 渐变背景风格 |
   | socrates | `##` (h2) | 苏格拉底风格 |
   | einstein | `##` (h2) | 爱因斯坦风格 |
   | border | `##` (h2) | 边框简约风格 |
   | academic-lightblue | `##` (h2) | 学术浅蓝风格 |

   **beam 主题特殊处理**：
   - 每页第一个 `#` (h1) 会自动显示在顶部蓝色装饰条中
   - 标题页使用 `<!-- _class: title -->` 指令
   - 内容页直接使用 `#` 标题（不是 `##`）
   - 示例：
     ```markdown
     ---
     # 页面标题在蓝色条中

     正文内容...
     ```

   **academic/jobs 主题特殊处理**：
   - 同样使用 `#` (h1) 作为页面标题
   - 支持 `class: lead` 用于特殊页面

   **gaia 主题特殊处理**：
   - 使用 `##` (h2) 作为内容页标题，带蓝色下划线装饰
   - 字体：Times New Roman（英文）、SimSun（中文）
   - 支持 `class: lead` 用于标题页
   - 适合商务演示、学术报告

4. **必须包含居中样式**：frontmatter 的 `style` 中必须包含以下 CSS，否则图片和表格会左对齐：
     ```css
     /* 图片居中 */
     img {
       display: block !important;
       margin: 0 auto !important;
     }
     /* 表格居中（覆盖主题的 width:100% 和 display:block） */
     section table {
       display: table !important;
       width: auto !important;
       margin-left: auto !important;
       margin-right: auto !important;
     }
     ```
     > **原因**：Marp 基础 CSS 设置 `section table { display: block; width: max-content }`，多数自定义主题又设置 `table { width: 100% }`，两者都会导致表格无法居中。必须用 `!important` 同时覆盖 `display`、`width` 和 `margin` 三个属性才能生效。`marp-template.md` 已包含这些样式。

5. **修改Header**：根据演讲主题调整
6. **逐页制作**：
   - 遵循模板中的页面类型和格式
   - 每页严格控制内容量
   - 使用适当的Markdown格式（粗体、引用、列表）

**页面类型选择**（参考 `references/slide-types.md`）：

- 标题页：开场
- 目录页：内容预告
- 单点页：核心观点展示
- 列表页：多要点展示
- 对比页：两种观点/方案对比
- 代码页：关键代码片段展示（技术演示专用）
- 公式页：数学公式推导（学术演示专用）
- 数据页：表格和统计结果（数据报告专用）
- 引用页：重要引用展示
- 分隔页：章节过渡
- 总结页：要点回顾

**代码块处理原则**：
- 单页代码不超过 15 行，超出则拆分为多页或只展示关键片段
- 用 ` ```python ` 标注语言以获得语法高亮
- 复杂代码用注释标注核心逻辑，删除无关细节
- 可用伪代码替代完整实现

**数学公式支持**：
- Marp 支持 MathJax：`$...$` 行内公式，`$$...$$` 块级公式
- 示例：`$\lambda$` 渲染为 λ，`$$L = \sum (y_i - \hat{y}_i)^2$$` 渲染为展开式
- 公式较多时，每页不超过 2-3 个块级公式

**图片引用规范**：
- 图片统一放在 `slides/[项目名]/images/` 目录
- 引用方式：`![描述](./images/figure_1.png)`
- 导出 PDF 时必须使用 `--allow-local-files` 参数

**文件保存**：

- 保存初稿：`slides/[项目名]/05_final/presentation.md`

### 阶段四：审阅与优化

**触发**：初稿完成后自动进入（快速模式跳过此阶段）

**执行方式**：使用 Task 工具派遣 1 个综合审阅 agent，一次性完成内容、格式、密度、视觉四个维度的检查。

**审阅 Agent 调用**：

```
Task 工具参数：
- subagent_type: "general"
- description: "Slides综合审阅"
- prompt:

你是一个 Slides 综合审阅专家。请对以下 Marp Slides 进行全面审阅：

## 原始材料
[插入用户提供的原始文件内容或摘要]

## 用户要求
[插入用户提出的具体要求和偏好]

## Slides 初稿
[插入 slides/[项目名]/05_final/presentation.md 的内容]

## 审阅维度

1. **内容完整性**
   - 核心论点是否都已覆盖？
   - 是否有重要信息遗漏？
   - 逻辑顺序是否合理？

2. **格式规范性**
   - YAML frontmatter 是否正确？
   - 分页符 `---` 使用是否规范？
   - 标题层级是否符合主题要求？

3. **内容密度**
   - 每页文字行数是否 ≤ 8 行？
   - 列表项数量是否 ≤ 6 项？
   - 是否有需要拆分的页面？

4. **视觉层次**
   - 每页是否有清晰的标题？
   - 粗体、引用是否使用得当？
   - 整体风格是否一致？

5. **中文规范**（如果是中文演示文稿）
   - 标点符号是否为中文标点？
   - 引号使用是否精简？
   - 中英文混排是否加空格？

## 输出要求

1. 逐项列出发现的问题（标注页码和行号）
2. 按优先级排序（高/中/低）
3. 给出可直接应用的修改建议
4. 如果问题较少，可直接输出修正后的完整 slides 内容

将审阅报告保存到 slides/[项目名]/review_report.txt
```

**快速模式说明**：
- 如果用户在阶段一选择了快速模式，跳过阶段四，直接进入阶段五
- 快速模式适合内容简单、页数较少（≤10页）的演示文稿

**输出**：向用户展示审阅报告，按优先级排列的修改清单

---

### 阶段五：生成终稿

**触发**：审阅修改完成后（或快速模式下阶段三结束后）

**执行步骤**：

1. **确认图片路径**：
   如果 slides 中引用了本地图片，确保图片放在 `slides/[项目名]/images/` 目录下，且引用路径为相对路径：
   ```markdown
   ![描述](./images/figure_1.png)
   ```

2. **导出HTML和PDF**：

   ```bash
   # 切换到项目目录，确保相对路径正确解析
   cd slides/[项目名]

   # 导出HTML预览版本
   npx @marp-team/marp-cli 05_final/presentation.md -o 05_final/slides.html --html --theme-set ./themes/

   # 导出PDF版本（必须加 --allow-local-files 以加载本地图片）
   npx @marp-team/marp-cli 05_final/presentation.md -o 05_final/slides.pdf --theme-set ./themes/ --allow-local-files
   ```

   **图片未显示排查**：
   - 检查 `--allow-local-files` 是否已添加
   - 检查图片路径是否为相对路径（如 `./images/xxx.png`）
   - 检查运行命令时的工作目录是否与 .md 文件所在目录一致

**最终产出**：

- `slides/[项目名]/05_final/presentation.md` - 最终 Markdown
- `slides/[项目名]/05_final/slides.html` - HTML 预览版本
- `slides/[项目名]/05_final/slides.pdf` - PDF 版本

**输出**：告知用户终稿文件位置和预览方式

## 常见问题处理

### 文字溢出

**原因**：单页内容过多

**解决方案**：
1. **拆分法**：将内容拆分为多页
   ```markdown
   ## 原始页面：五个要点

   拆分为：

   ## 要点概览（1/2）
   1. 要点一
   2. 要点二
   3. 要点三

   ---

   ## 要点概览（2/2）
   4. 要点四
   5. 要点五
   ```

2. **精简法**：删除次要信息
3. **层次法**：使用子列表缩短主列表

### 代码块过长

**解决方案**：
- 只展示关键代码片段
- 使用伪代码替代完整代码
- 分多页展示，每页一个逻辑单元

### 表格过大

**解决方案**：
- 拆分为多个小表格
- 转换为列表形式
- 只展示关键数据行

### 字体过小

**判断标准**：正文字号小于 0.7em 时多数人会阅读困难

**解决方案**：
1. **拆分法**：将内容拆分为多页，减少每页信息量
2. **精简法**：删除次要要点，保留核心内容
3. **视觉辅助法**：用图表、图标代替文字说明

## 模板使用说明

模板文件：
- **`marp-template.md`** - 英文通用模板（graph_paper 主题）
- **`marp-template-zh.md`** - 中文通用模板（graph_paper 主题，带中文注释和示例）
- **`marp-template-tech.md`** - 技术分享/代码演示专用模板（含代码块样式和 MathJax 支持）
- **`marp-template-data.md`** - 数据报告专用模板（含表格样式和统计结果展示）

**模板选择指南**：

| 演示类型 | 推荐模板 | 主题建议 |
|----------|----------|----------|
| 学术报告/论文答辩 | marp-template-zh.md | academic, beam, gaia |
| 技术分享/代码演示 | marp-template-tech.md | graph_paper, turing, beam |
| 数据报告/分析汇报 | marp-template-data.md | gaia, companyLightBlue |
| 教学课件/培训 | marp-template-zh.md | simple, graph_paper |
| 产品发布/商业演示 | marp-template-zh.md | jobs, gradient |

**gaia 主题模板配置示例**：
```yaml
---
marp: true
theme: gaia
class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.jpg')
style: |
  section {
    font-family: 'Times New Roman', 'SimSun', serif;
  }
  h1 {
    color: #2c3e50;
    font-size: 1.5em;
  }
  h2 {
    color: #34495e;
    border-bottom: 2px solid #3498db;
  }
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
---

# 标题页

副标题

---

## 内容页标题

正文内容...
```

**修改Header**：
```yaml
header: '你的课程/演讲标题'
```

## 资源引用

### 参考文件
- **`marp-template.md`** - Marp英文通用模板
- **`marp-template-zh.md`** - Marp中文通用模板
- **`marp-template-tech.md`** - 技术分享/代码演示专用模板
- **`marp-template-data.md`** - 数据报告专用模板
- **`references/slide-types.md`** - 页面类型详解（含代码页、公式页、数据页）
- **`references/review-checklist.md`** - 审阅检查清单
- **`themes/README.md`** - 主题说明文档（含快速适配对照表）
- **`themes/*.css`** - 15款精选主题文件

## Marp-CLI 命令参考

```bash
# 先进入项目目录（确保相对路径正确）
cd slides/[项目名]

# 预览（启动本地服务器）
npx @marp-team/marp-cli -s 05_final/presentation.md --theme-set ./themes/

# 导出HTML版本
npx @marp-team/marp-cli 05_final/presentation.md -o 05_final/slides.html --html --theme-set ./themes/

# 导出PDF版本（含本地图片）
npx @marp-team/marp-cli 05_final/presentation.md -o 05_final/slides.pdf --theme-set ./themes/ --allow-local-files
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--theme-set ./themes/` | 加载本地 themes 文件夹中的所有主题 |
| `--allow-local-files` | 允许加载本地图片和资源（PDF 导出必需） |
| `--html` | 启用 HTML 标签支持 |
| `-s` | 启动预览服务器 |

**导出格式说明**：

| 格式 | 用途 | 文件扩展名 |
|------|------|-----------|
| HTML | 网页展示、在线分享 | `.html` |
| PDF | 打印、正式分发 | `.pdf` |

## 交互规范

1. 每阶段结束后询问用户是否继续
2. 内容分析后让用户确认大纲再继续
3. 检查发现问题时，说明问题和建议的修复方案
4. 提供选择时给出推荐选项和理由

## Windows 路径注意

Windows 环境下路径使用反斜杠或正斜杠均可，但建议统一使用正斜杠以兼容 WSL：

```bash
# Windows - 统一使用正斜杠
slides/project-name/05_final/presentation.md
```

## 禁止事项

- 不在单页放置过多内容（每页 ≤ 8 行文字）
- 不使用过小的字体或过密的排版（正文 ≥ 0.9em）
- 不输出未经验证的终稿（至少检查分页和密度）
- 不遗漏 `--allow-local-files` 参数（当使用了本地图片时）
