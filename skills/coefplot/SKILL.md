---
name: coefplot
description: Stata coefplot 回归系数作图专家。当用户需要在 Stata 中绘制回归系数图、多模型对比图、边际效应图、系数排序图时自动启用。覆盖 coefplot 全部功能：基本用法、keep/drop、多模型绘制、偏移调整、多轴、子图、bycoefs、系数匹配、排序、矩阵绘图、recast 图形类型、连续轴、eform/odds ratio、rescale/transform、标准化系数、margins、置信区间类型与层级、bootstrap CI、平滑CI、截断CI、标签系统(headings/groups/coeflabels/eqlabels)、标记样式(mlabel/mlabels/加权标记)、条形图、堆叠条形图、箭头变化图、addplot、子图专属选项、模型名转系数名等。触发关键词：coefplot、系数图、回归系数、多模型对比、estimates、子图、bycoefs、森林图、coefficient plot。
---

# Stata coefplot 作图技能

## 技能概述

本技能提供 Stata `coefplot` 命令的完整作图指导，涵盖从基础用法到高级多模型、多子图的所有功能。基于 Ben Jann 官方文档（repec.sowi.unibe.ch/stata/coefplot），所有示例可直接运行。

安装：`ssc install coefplot`，或 `net install gr0059.pkg, from(http://repec.sowi.unibe.ch/stata/coefplot/)`

## 预备知识

coefplot 的基本流程：运行估计模型 → 存储结果（`estimates store`）→ 调用 `coefplot` 绘制系数点估计及其置信区间。

- 点估计从 `e(b)` 的第一个方程提取，置信区间从 `e(V)` 计算出
- 默认水平布局（系数在 Y 轴，估计值在 X 轴），用 `vertical` 翻转为竖直布局
- 选项分四层：modelopts → plotopts → subgropts → globalopts，上层选项作为下层默认值

---

## 一、基本用法

### 1.0 estout/esttab 工作流衔接

实际研究中常用 `eststo`/`estout` 管理模型，再喂给 coefplot：

```stata
ssc install estout  // 首次使用

sysuse auto, clear
eststo clear

eststo m1: regress price mpg
eststo m2: regress price mpg trunk
eststo m3: regress price mpg trunk length

* 先用 esttab 看表格
esttab, se star(* 0.1 ** 0.05 *** 0.01)

* 再用 coefplot 作图（eststo 存储的模型直接用名字调用）
coefplot m1 m2 m3, drop(_cons) xline(0)
```

**关键点**：
- `eststo` 存储的模型名可以直接用于 `coefplot`，无需额外 `estimates store`
- `estadd` 可添加自定义统计量到已存储的模型，`coefplot` 通过 `matrix()` 读取
- `estimates dir` 查看已存储的模型列表

### 1.1 单模型

```stata
sysuse auto, clear
regress price mpg trunk length turn
coefplot, drop(_cons) xline(0)
```

- `drop(_cons)`：排除常数项
- `xline(0)`：在 0 处画参考线
- 竖直布局：`coefplot, vertical drop(_cons) yline(0)`（参考线改为 `yline(0)`）

### 1.2 keep / drop

| 选项 | 作用 |
|------|------|
| `keep(pattern)` | 保留匹配模式的系数，`*` `?` 通配符 |
| `drop(pattern)` | 排除匹配模式的系数 |
| `keep(*:)` | 选择所有方程（多方程模型如 `mlogit`） |
| `omitted` | 显示被省略的系数 |
| `baselevels` | 显示基准水平系数 |

```stata
coefplot, keep(mpg trunk)                    // 只保留指定系数
coefplot, nolabel keep(*:) omitted baselevels // 显示全部方程含基准
coefplot, keep(3:*.foreign 4:mpp 5:mpp _cons) // 按方程筛选
```

---

## 二、多模型绘制

### 2.1 多模型独立系列

```stata
coefplot D F, drop(_cons) xline(0)

coefplot (D, label(Domestic Cars) pstyle(p3))       ///
         (F, label(Foreign Cars)  pstyle(p4))       ///
       , drop(_cons) xline(0) msymbol(S)

* 用 p1() p2() 替代括号写法
coefplot D F, drop(_cons) xline(0) msymbol(S)       ///
    p1(label(Domestic Cars) pstyle(p3))             ///
    p2(label(Foreign Cars)  pstyle(p4))
```

### 2.2 偏移控制

系数间距为 1 单位，偏移宜在 -0.5 到 0.5 之间：

```stata
coefplot D F, drop(_cons) xline(0) nooffsets           // 关闭自动偏移
coefplot (D, offset(0.05)) (F, offset(-0.05)), drop(_cons) xline(0)
```

### 2.3 多轴

因变量量纲不同时：

```stata
coefplot Price (Weight, axis(2)), drop(_cons)         ///
    xtitle(Price) xtitle(Weight, axis(2))
```

### 2.4 合并模型到同一系列（Appending）

```stata
coefplot (mpg trunk length turn, label(bivariate))    ///
         (multivariate)                               ///
       , drop(_cons) xline(0)
```

语法：`(namelist [, modelopts] \ namelist [, modelopts] \ ... [, plotopts])`

---

## 三、子图（Subgraphs）

### 3.1 基本语法

```
coefplot plotlist [, subgropts] || plotlist [, subgropts] || ... [, globalopts]
```

```stata
coefplot D, bylabel(Domestic Cars) || F, bylabel(Foreign Cars) ||, drop(_cons) xline(0)
```

### 3.2 每子图多模型 + byopts

```stata
coefplot (D, label(Domestic)) (F, label(Foreign)), bylabel(Price)  ///
    || (D_weight) (F_weight), bylabel(Weight)                       ///
    ||, drop(_cons) xline(0) byopts(xrescale)
```

子图模型数不一致时插入 `_skip` 对齐。`byopts(compact cols(1))` 控制排列方式。

### 3.3 norecycle — 每子图独立样式

```stata
coefplot (rep2, label(rep78=2)) (rep3, label(rep78=3)), bylabel(Low record)  ///
    || (rep4, label(rep78=4)) (rep5, label(rep78=5)), bylabel(High record)    ///
    ||, drop(_cons) xline(0) norecycle legend(colfirst)
```

### 3.4 bycoefs — 按系数拆分面板

将系数视为"子图"，原子图视为"系数"：

```stata
coefplot rep78_3 || rep78_4 || rep78_5, drop(_cons) xline(0) ///
    bycoefs byopts(xrescale)

* 竖直版
coefplot rep78_3 || rep78_4 || rep78_5, drop(_cons) yline(0) ///
    bycoefs byopts(yrescale) vertical
```

---

## 四、通配符匹配模型名

```stata
coefplot (d*, label(domestic)) (f*, label(foreign)) ///
    , drop(_cons) xline(0)

* 不合并（独立系列）
coefplot est0* || est1*, drop(_cons) xline(0) ///
    plotlabels("White" "Black") bylabels("North" "South")
```

---

## 五、系数匹配规则

默认使用第一个（非零）方程，按系数名称匹配（忽略方程名）。

| 选项 | 作用 |
|------|------|
| `eqstrict` | 严格按方程名匹配 |
| `keep(*:)` | 显示所有方程 |
| `asequation(name)` | 为模型指定方程名 |
| `eqrename(old = new)` | 重命名方程，可用 `regex` |
| `rename(old = new)` | 重命名系数，可用 `regex` |
| `noeqlabels` | 隐藏方程标签 |

```stata
* 匹配不同名称的系数
coefplot (m1, label(Without error)) (m2, label(With error)) ///
    , xline(1) rename(x1err = x1)
```

### 模型名作为系数名

```stata
coefplot (industry_*), keep(grade) asequation swapnames ///
    title("Effect of grade on wages by industry")

* 修改后自动获取值标签
coefplot (industry_*), keep(grade) asequation swapnames ///
    eqrename(^industry_(.*)$ = \1.industry, regex)
```

---

## 六、系数排序

### 6.1 指定顺序

```stata
coefplot m1 || m2 || m3, ... orderby(3:)           // 以第三个子图为参照
coefplot m1 || m2 || m3, ... order(mpg trunk length) // 显式顺序
coefplot m1 || m2 || m3, ... order(. mpg . t* .)     // 通配符 + 间隔
coefplot m1 || m2, ... order(5: 4:)                  // 按方程排序
```

### 6.2 按数值排序

```stata
coefplot, sort                                     // 升序
coefplot, sort(, descending)                       // 降序
coefplot, sort(, by(se))                           // 按标准误排序
coefplot ... sort(1, descending)                   // 按特定系列排序
coefplot ... sort(2:1, descending)                 // 含子图：子图号:系列号
```

---

## 七、估计值变换

### 7.1 eform — 指数化（Odds Ratio / Hazard Ratio）

```stata
logit foreign mpg trunk length turn
coefplot, drop(_cons) xline(1) eform xtitle("Odds ratio")
```

`eform` 自动处理 logit (OR)、stcox (HR)、mlogit (RRR)、poisson (IRR)。

### 7.2 rescale — 乘法缩放

```stata
* 比例 → 百分比
coefplot, rescale(100) xtitle(Percent) recast(bar) barwidth(0.5) finten(60) ///
    citop citype(logit) ciopts(recast(rcap))

* 选择特定系数缩放
coefplot, drop(_cons) xline(0) rescale(weight = 100 gpm = .01) ///
    coeflabels(weight = "Weight (in 100 lbs.)" gpm = "Gallon per 100 miles")

* 标准化系数（用标准差缩放）
coefplot, drop(_cons) xline(0) ///
    rescale(mpg = `sd_mpg' weight = `sd_weight' ///
            length = `sd_length' turn = `sd_turn')
```

### 7.3 transform — 任意变换

`@` 为值的占位符：

```stata
* 等同于 eform
coefplot (., eform label(eform)) (., transform(* = exp(@)) label(transform)) ///
    , drop(_cons) xline(1) xtitle("Odds ratio")

* 混合效应模型：方差 → 标准差，协方差 → 相关系数
coefplot, noeqlabels keep(ln*: at*:) ///
    transform(ln*: = exp(@) at*: = tanh(@)) ///
    coeflabels(ln*1: = "se(week)" ln*2: = "se(_cons)" ///
               at*: = "corr(week,_cons)" ln*e: = "sd(Residual)")
```

### 7.4 if() — 按值选择系数

```stata
* 显著/不显著不同样式（@ll, @ul 为 CI 下限/上限）
coefplot (., if(@ll<0 & @ul>0)) (., if(@ll>0 | @ul<0)) ///
    , drop(_cons) nooffset xline(0) legend(off)

* 截断宽 CI 并用不同箭头标注
coefplot (., pstyle(p1) if(@ll>2&@ul<12))                ///
         (., pstyle(p1) if(@ll>2&@ul>=12) ciopts(recast(pcarrow)))   ///
         (., pstyle(p1) if(@ll<=2&@ul<12) ciopts(recast(pcrarrow)))  ///
         (., pstyle(p1) if(@ll<=2&@ul>=12) ciopts(recast(pcbarrow))) ///
    , nooffset transform(* = min(max(@,2),12)) legend(off)
```

### 7.5 标准化系数

三种方式：

1. **`center` 标准化变量**（`ssc install center`）：
   ```stata
   preserve
   center price mpg weight length turn foreign, inplace standardize
   regress price mpg weight length turn, noconstant
   restore
   coefplot, xline(0) xtitle("Standardized Coefficients")
   ```

2. **`sem` + `b_std`/`V_std`**：
   ```stata
   sem (price <- mpg weight length turn)
   coefplot, drop(_cons) xline(0) b(b_std) v(V_std)
   ```

3. **`rescale()` 手动缩放**（见 7.2）。

---

## 八、margins 结果绘制

必须加 `post` 选项确保结果存入 `e()`：

```stata
logit foreign mpg trunk length turn
margins, dydx(*) post
coefplot, xline(0) xtitle("Average marginal effects")
```

### mlogit 按方程分别绘制

```stata
mlogit insure i.male i.nonwhite i.site
margins, dydx(*) post

coefplot (, keep(*:1._predict) label(Indemnity)) ///
         (, keep(*:2._predict) label(Prepaid))   ///
         (, keep(*:3._predict) label(Uninsure))  ///
       , swapnames xline(0) legend(rows(1))
```

---

## 九、置信区间

### 9.1 改变 CI 图形类型

```stata
coefplot domestic foreign, drop(_cons) xline(0) ciopts(recast(rcap))   // 带帽线
```

### 9.2 多层级置信区间

```stata
* 99.9%, 99%, 95%
coefplot, drop(_cons) xline(0) msymbol(s) mfcolor(white)  ///
    levels(99.9 99 95) legend(order(1 "99.9" 2 "99" 3 "95") rows(1))

* 控制线宽
coefplot, ... levels(99.9 99 95) ciopts(lwidth(*1 *3 *6))

* 从浅到深的多层级（Harrell 风格）
coefplot, drop(_cons) xline(0) msymbol(d) mcolor(white)  ///
    levels(99 95 90 80 70) ciopts(lwidth(3 ..) lcolor(*.2 *.4 *.6 *.8 *1)) ///
    legend(order(1 "99" 2 "95" 3 "90" 4 "80" 5 "70") rows(1))

* 95% + 50%（Cleveland 风格）
coefplot domestic foreign, drop(_cons) xline(0) levels(95 50) ciopts(recast(. rcap))
```

### 9.3 自定义 CI 来源

| 选项 | 作用 |
|------|------|
| `v(name)` | 自定义方差矩阵 |
| `se(name)` | 自定义标准误向量 |
| `ci(name)` | 预计算的置信区间矩阵 |
| `df(#)` | 自定义自由度 |

```stata
* survey 估计：比较设计效应 CI 与 SRS CI
coefplot (., label(design-based)) (., v(V_srs) label(SRS-based)) ///
    , keep(female black orace rural) xlabel(,grid)

* 用不同 df
local df_r = e(N) - e(df_m) - 1
coefplot (., label(design-based)) (., v(V_srs) df(`df_r') label(SRS-based)) ///
    , keep(female black orace rural) xlabel(,grid)
```

### 9.4 Bootstrap 置信区间

```stata
regress price mpg trunk length turn, vce(bootstrap)

coefplot (., ci(ci_normal)    label(normal))     ///
         (., ci(ci_percentile) label(percentile)) ///
         (., ci(ci_bc)        label(bc))          ///
       , drop(_cons) xline(0) legend(rows(1))
```

### 9.5 平滑置信区间

```stata
coefplot domestic foreign, drop(_cons) xline(0) cismooth grid(none)
```

`cismooth` 生成 50 级渐变 CI（1%, 3%, ..., 99%），有专属子选项控制外观，独立于 `levels()` / `ciopts()`。

### 9.6 比例数据的 CI

```stata
coefplot domestic foreign, xtitle("Repair Record 1978") ytitle("Proportion") ///
    vertical recast(bar) barwidth(0.25) finten(60) ///
    citop citype(logit) ciopts(recast(rcap)) rename(*.rep78 = "")
```

`citype(logit)` 确保比例 CI 在 0-1 范围内。

### 9.7 截断宽置信区间

```stata
coefplot, transform(* = min(max(@,1.5),12.5)) ///
    xscale(range(1.5 12.5)) plotregion(margin(zero))

* 带背景色区分
coefplot, transform(* = min(max(@,2),12)) ///
    plotregion(color(gray) icolor(white)) grid(nogextend)
```

---

## 十、标签系统

### 10.1 自动标签 vs 系数名

coefplot 自动从变量标签/值标签生成标签。用 `nolabels` 显示原始名称，用 `regress, coeflegend` 查看系数名。

```stata
coefplot, xline(0) nolabels
```

### 10.2 coeflabels — 自定义系数标签

```stata
coefplot, xline(0) coeflabels(1.foreign = "Foreign Car" _cons = "Constant")

* 换行：wrap(20) 自动在第 20 字符处换行
coefplot, xline(0) coeflabels(, wrap(20))

* 截断：truncate(20)
coefplot, xline(0) coeflabels(, truncate(20))

* 手动换行（复合双引号）
coefplot, xline(0) coeflabels(4.rep78 = `""Repair Record" "1978 = 4""', wrap(20))

* 修改交互项连接符
coefplot, xline(0) coeflabels(, interaction(" x "))
```

### 10.3 headings — 系数间添加标题

```stata
coefplot, xline(0) omitted baselevels drop(_cons) ///
    headings(3.rep78 = "{bf:Repair Record}" ///
             0.foreign = "{bf:Car Type}" ///
             3.rep78#0.foreign = "{bf:Interaction Effects}")
```

### 10.4 groups — 分组标签

```stata
coefplot, xline(0) omitted baselevels drop(_cons) ///
    groups(?.rep78 = `""{bf:Repair}" "{bf:Record}""' ///
           ?.foreign = "{bf:Car Type}" ///
           ?.rep78#?.foreign = "{bf:Interaction Effects}")

* headings + groups 组合
coefplot, xline(0) omitted baselevels drop(_cons) ///
    headings(3.rep78 = "{it:Repair record:}" ///
             0.foreign = "{it:Car type:}", nogap) ///
    groups(headroom 1.foreign = "{bf:Main Effects}" ///
           ?.rep78#?.foreign = "{bf:Interaction Effects}")
```

### 10.5 bycoefs 下的 headings/groups

因 bycoefs 用数字标识元素，用数字：

```stata
coefplot Domestic || Foreign || Total, drop(_cons) yline(0) ///
    bycoefs byopts(yrescale) vertical ///
    group(1 2 = "{bf:Subgroup results}", nogap) ylabel(0, add)
```

### 10.6 方程标签（eqlabels）

```stata
* 自定义方程标签
coefplot, omitted keep(*:) coeflabels(mpp = "Milage") ///
    eqlabels("Equation 1" "Equation 2" "Equation 3")

* 作为标题插入（asheadings）
coefplot, omitted keep(*:) eqlabels("{bf:Eq 1}" "{bf:Eq 2}" "{bf:Eq 3}", asheadings)

* 从数据获取标签
coefplot, keep(*:) eqlabels(, labels)
```

### 10.7 标签移到右侧

```stata
coefplot, xline(0) omitted baselevels drop(_cons) yscale(alt) ///
    headings(...)

* group 标签也移到右侧（axis(2)）
coefplot, xline(0) omitted baselevels drop(_cons) yscale(alt axis(2)) ///
    groups(..., angle(rvertical))
```

### 10.8 左对齐标签

```stata
coefplot, xline(0) drop(_cons) omitted baselevels yscale(noline alt) ///
    graphregion(margin(l=65)) coeflabels(, notick labgap(-125)) ///
    headings(..., labgap(-130))

* 或使用 gr_edit（记录图形编辑器操作）
gr_edit .move yaxis1 leftof 8 5
```

### 10.9 网格线

```stata
* 自定义网格线样式
coefplot, xline(0) xlabel(, grid) ///
    grid(between glpattern(dash) glwidth(*2) glcolor(gray))

* 手动定义刻度线
coefplot, xline(0) ytick(1.5 3.5 4.5 6.5, notick glstyle(refline))
```

---

## 十一、标记与标记标签

### 11.1 标记样式

```stata
coefplot (m1, msymbol(D) mlcolor(magenta) mfcolor(magenta*.3)) ///
         (m2, msymbol(S)) ///
       , mfcolor(white) msize(large)

* 用 pstyle 快速设置
coefplot (m1, pstyle(p3)) (m2, pstyle(p4))

* CI 单独设 pstyle
coefplot (m1, ciopts(pstyle(p2))) (m2, pstyle(p3) ciopts(pstyle(p4)))
```

### 11.2 仅标记 / 仅 CI

```stata
coefplot (., noci label(Markers only)) (., cionly label(CIs only) key(ci)) ///
    , drop(_cons)
```

### 11.3 标记标签显示点估计值

```stata
coefplot, xline(0) mlabel format(%9.2g) mlabposition(12) mlabgap(*2)
```

**标签后加白色背景盒**：
```stata
* 方法一：手动添加矩阵
mata: st_matrix("e(box)", (st_matrix("e(b)"):-2 \ st_matrix("e(b)"):+2))
coefplot, xline(0) mlabel format(%9.2g) mlabposition(0) msymbol(i) ///
    ci(95 box) ciopts(recast(. rbar) barwidth(. 0.35) color(. white))

* 方法二：transform 动态生成
coefplot, xline(0) mlabel format(%9.2g) mlabposition(0) msymbol(i) ///
    ci(95 99) transform(* = "cond(@==@ll2, @b-2, cond(@==@ul2, @b+2, @))") ///
    ciopts(recast(. rbar) barwidth(. 0.35) fcolor(. white) lwidth(. medium))
```

### 11.4 字符串表达式作为标记标签

```stata
* 显示 p 值
coefplot, xline(0) mlabposition(1) mlabgap(*2) ///
    mlabel("{it:p} = " + string(@pval,"%9.3f"))

* 显著性星号
coefplot, xline(0) mlabposition(1) ///
    mlabel(cond(@pval<.001, "***", cond(@pval<.01, "**", ///
           cond(@pval<.05, "*", cond(@pval<.1, "+", ""))))) ///
    note("+ p < .1, * p < .05, ** p < .01, *** p < .001")
```

### 11.5 自定义特定系数的标签

```stata
coefplot (domestic, mlabels(length = 1 "+" * = 11 "0")) ///
         (foreign,  mlabels(trunk length = 1 "+" * = 11 "0")) ///
       , drop(_cons) xline(0) note("Hypotheses: + positive; 0 no effect")

* 渲染选项
coefplot (domestic, mlabels(trunk = 12 "comment") ///
    mlabangle(45) mlabgap(2) mlabsize(medium) mlabcolor(red))
```

### 11.6 标签放在 CI 末端

```stata
* 用 addplot + 内部变量 @at @ul
coefplot domestic foreign, drop(_cons) xline(0) ///
    addplot(scatter @at @ul, ms(i) mlabel(@b) mlabformat("%9.1f") ///
            mlabcolor(black) mlabpos(2))

* 复杂标签：先 mlabel(..., mlabcolor(none)) 生成 @mlbl 再 addplot 调用
coefplot domestic foreign, drop(_cons) xline(0) ///
    mlabel("{it:p} = " + string(@pval,"%9.3f")) mlabcolor(none) ///
    addplot(scatter @at @ul, ms(i) mlabel(@mlbl) mlabcolor(black) mlabpos(2))
```

### 11.7 标记标签作为轴标签

详见 help file 的 `coefplot_mlbl` / `coefplot_ymlbl` 自定义程序。

### 11.8 加权标记

```stata
* 按精度（1/se）缩放
coefplot, weight(1/@se) ms(oh) drop(_cons) xline(0)

* 用辅助变量缩放（如组样本量）
coefplot, ciopts(recast(rcap)) aux(_N) weight(@aux1) mfcolor(*.6)
```

跨系列可比用 `asequation swapnames` 合并为单系列。

---

## 十二、图形类型（recast）

### 12.1 条形图 + 带帽 CI

```stata
coefplot (D, label(Domestic Cars)) (F, label(Foreign Cars)) ///
    , drop(_cons) xline(0) recast(bar) ciopts(recast(rcap))  ///
    citop barwidth(0.3)
```

### 12.2 比例 + 均值混合

```stata
coefplot (prop, recast(bar) noci barwidth(0.5) color(*.6))        ///
         (mean, recast(connected) ciopts(recast(rcap)) axis(2))    ///
       , vertical nooffsets plotlabels("Proportion" "Price")       ///
       xtitle("Repair record") ytitle("Proportion")                ///
       ytitle("Price", axis(2)) rename(^.*([0-9])\..+$ = \1, regex)
```

### 12.3 连续轴（折线图）

```stata
logit foreign mpg
margins, at(mpg=(10(2)40)) post
estimates store bivariate

logit foreign mpg turn price
margins, at(mpg=(10(2)40)) post
estimates store multivariate

coefplot bivariate multivariate, ytitle(Pr(foreign=1)) xtitle(Miles per Gallon) ///
    at recast(line) lwidth(*2) ciopts(recast(rline) lpattern(dash))
```

`at()` 自动关闭偏移。

### 12.4 条形图 — 每柱不同样式

```stata
coefplot (., keep(3.rep78)) (., keep(4.rep78)) (., keep(5.rep78)) ///
    , vertical legend(off) nooffsets recast(bar) barwidth(0.8) fcolor(*.8) ///
    citop ciopts(recast(rcap)) citype(logit) ///
    coeflabels(, notick labgap(2)) plotregion(margin(b=0))
```

### 12.5 条形图加数值标签

```stata
* Stata 17+ 可直接用 mlabel
coefplot domestic foreign, vertical recast(bar) barwidth(0.3) fcolor(*.5) ///
    ciopts(recast(rcap)) citop citype(logit) format(%9.2f) ///
    addplot(scatter @b @at, ms(i) mlabel(@b) mlabpos(2) mlabcolor(black))

* 标签在柱内
coefplot domestic foreign, vertical noci format(%9.1f) rescale(100) ///
    recast(bar) barwidth(0.3) fcolor(*.5) plotregion(margin(b=0)) ///
    coeflabels(, notick labgap(2)) ///
    ylabel(0(10)70, angle(horizontal) format(%9.0f)) ytitle(Percent) ///
    addplot(scatter @b @at if @plot==1, ms(i) mlabel(@b) mlabpos(6) pstyle(p1) ///
    || scatter @b @at if @plot==2, ms(i) mlabel(@b) mlabpos(6) pstyle(p2))
```

---

## 十三、矩阵绘图

```stata
* 中位数 + 置信区间
coefplot matrix(median), ci(CI)

* 矩阵与模型混合
coefplot (., label(mean) rename(^.*([0-9])\..+$ = \1, regex))    ///
         (matrix(R[,1]), ci((2 3)) label(median))                 ///
       , ytitle(Repair Record 1978) xtitle(Price)
```

---

## 十四、Varia 高级技巧

### 14.1 箭头表示变化

```stata
webuse nlswork, clear
mean ln_wage if year==88, over(ind_code)
matrix b88 = e(b)
mean ln_wage if year==78, over(ind_code)
mata: assert(st_matrixcolstripe("b88")==st_matrixcolstripe("e(b)"))
quietly estadd matrix b88

* rcap 显示起点，pcarrow 显示变化方向
coefplot, ci((b b) (b b88)) ciopts(recast(rcap pcarrow)) cionly ///
    vertical sort rename(^.+@([0-9]+)\..+$ = \1, regex) ///
    xtitle("Industry code") ytitle("Change in ln(wage) from 78 to 88")

* 按终点排序
coefplot, ci((b b) (b b88)) ciopts(recast(rcap pcarrow)) cionly ///
    vertical sort(, by(ul 2)) rename(...)
```

### 14.2 堆叠条形图

需要 `moremata`，详见帮助文件的完整示例。

### 14.3 addplot — 子图专属选项

coefplot 不原生支持子图专属 `xline()`、轴标题、图例，用 `addplot` 补充：

```stata
* 安装：ssc install addplot

* 子图专属 xline
coefplot ., bylabel(Log odds) || ., bylabel(Odds ratios) eform ///
    || , drop(_cons) nolabel byopts(xrescale)
addplot 1: , xline(0) norescaling
addplot 2: , xline(1) norescaling

* 子图专属轴标题
coefplot Price || Weight, drop(_cons) xline(0) byopts(xrescale)
addplot 1: , b1title("Dollars") norescaling
addplot 2: , b1title("Pounds") norescaling

* 子图专属图例
coefplot rep2 rep3, bylabel(Low record) || rep4 rep5, bylabel(High record) ///
    || , drop(_cons) xline(0) norecycle byopts(legend(off))
addplot 1: , legend(order(2 "rep78=2" 4 "rep78=3") on) norescaling
addplot 2: , legend(order(6 "rep78=4" 8 "rep78=5") on) norescaling
```

`norescaling` 必须加。

### 14.4 不同大小的子图

```stata
coefplot price, drop(_cons) subtitle(Price, box bexpand lstyle(none)) ///
    name(price) nodraw
coefplot weight, drop(_cons) subtitle(Weight, box bexpand lstyle(none)) ///
    name(weight) nodraw yscale(off) fxsize(40)
graph combine price weight, imargin(small)
graph drop price weight
```

---

## 十五、选项层级速查

| 层级 | 作用范围 | 常用选项 |
|------|----------|----------|
| `modelopts` | 单个模型/矩阵 | `keep()`, `asequation()`, `rename()`, `eform`, `rescale()`, `transform()` |
| `plotopts` | 单个系列 | `label()`, `msymbol()`, `recast()`, `offset()`, `noci`, `cionly` |
| `subgropts` | 单个子图 | `bylabel()`, `title()`, `norecycle` |
| `globalopts` | 整张图 | `levels()`, `xline()`, `vertical`, `byopts()`, `sort()` |

嵌套规则：上层作为默认，下级可覆盖。

**位置细节**：

- `coefplot m1, opts1 \|\| m2, opts2 opts3` → opts2 和 opts3 是全局
- 只作用于 m2：`coefplot m1, opts1 \|\| m2, opts2 \|\|, opts3`
- `(m1, opts1 \\ m2, opts2)` → opts2 作用于两个模型
- 只作用于 m2：`(m1, opts1 \\ m2, opts2 \\)`

---

## 十六、常见模式速查

```stata
* 双变量 vs 多变量
coefplot (b1 b2 ..., label(Bivariate)) (m, label(Multivariate)), drop(_cons) xline(0)

* 分组子图
coefplot g1, bylabel(G1) || g2, bylabel(G2) ||, drop(_cons) xline(0)

* 按系数子图
coefplot m1 || m2 || m3, drop(_cons) xline(0) bycoefs byopts(xrescale)

* 森林图（降序）
coefplot, sort(, descending) xline(0)

* 边际效应折线
coefplot m1 m2, at recast(line) lwidth(*2) ciopts(recast(rline) lpattern(dash))

* Odds Ratio
coefplot, drop(_cons) xline(1) eform xtitle("Odds Ratio")

* 多层级 CI
coefplot, levels(99 95 90) ciopts(lwidth(*1 *3 *6))

* 条形 + 数值标签
coefplot, vertical recast(bar) mlabel format(%9.2g) mlabposition(12)
```

---

## 十七、注意事项

1. 先 `estimates store` 再 `coefplot`
2. `vertical` 后用 `yline(0)` 替代 `xline(0)`
3. 系数间距为 1，`offset()` 取值 -0.5~0.5
4. 多方程模型用 `keep(*:)` 显示全部
5. `rename()` / `eqrename()` 支持 `regex` 子选项
6. `levels()` 用空格分隔多个层级
7. `addplot` 必须加 `norescaling`
8. 安装：`ssc install coefplot`，安装 addplot：`ssc install addplot`

---

## 参考文献

- Ben Jann (2014). "Plotting regression coefficients and other estimates." *The Stata Journal*, 14(4), 708-737.
- 官方文档：https://repec.sowi.unibe.ch/stata/coefplot/getting-started.html
- 内部临时变量列表：`help coefplot` → Accessing internal temporary variables
