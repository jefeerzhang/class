"""
债券久期可视化脚本 — Bond Duration Visualization
生成 4 张示意图，帮助理解久期概念
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ========== 中文支持配置 ==========
import os
import matplotlib.font_manager as fm

# 使用 matplotlib 内置字体管理器检测系统可用中文字体
font_found = False
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC']
selected_font = None

# 从系统已安装字体中查找中文字体
for font_name in chinese_fonts:
    try:
        path = fm.findfont(fm.FontProperties(family=font_name))
        # findfont 返回 DejaVu Sans 的路径时表示字体未找到
        if 'DejaVu' not in path and os.path.exists(path):
            selected_font = font_name
            font_found = True
            break
    except Exception:
        continue

if font_found:
    rcParams['font.sans-serif'] = [selected_font]
    rcParams['axes.unicode_minus'] = False
    print(f'使用字体: {selected_font}')
else:
    # 如果找不到标准中文字体，使用 matplotlib 自带的 SimSun
    # matplotlib 2.x+ 自带 SimSun 字体文件
    print('警告: 未找到系统安装的中文字体，尝试使用 matplotlib 内置 SimSun')
    rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

rcParams['font.size'] = 12
rcParams['figure.dpi'] = 150

# 输出目录
OUTPUT_DIR = 'C:/Users/jefeer/Downloads/opencode/investment/assets/duration_viz'

# ========== 辅助函数 ==========

def bond_price(face_value, coupon_rate, ytm, years, freq=1):
    """计算债券价格"""
    coupon = face_value * coupon_rate / freq
    n = int(years * freq)
    r = ytm / freq
    t = np.arange(1, n + 1)
    pv_coupons = np.sum(coupon / (1 + r) ** t)
    pv_face = face_value / (1 + r) ** n
    return pv_coupons + pv_face


def macaulay_duration(face_value, coupon_rate, ytm, years, freq=1):
    """计算麦考利久期"""
    coupon = face_value * coupon_rate / freq
    n = int(years * freq)
    r = ytm / freq
    t = np.arange(1, n + 1)
    pv_coupons = coupon / (1 + r) ** t
    pv_face = face_value / (1 + r) ** n
    price = np.sum(pv_coupons) + pv_face
    weighted_time = np.sum(t * pv_coupons) + n * pv_face
    duration = weighted_time / price / freq
    return duration


def modified_duration(face_value, coupon_rate, ytm, years, freq=1):
    """计算修正久期"""
    mac_dur = macaulay_duration(face_value, coupon_rate, ytm, years, freq)
    return mac_dur / (1 + ytm / freq)


# ========== 图1: 价格-收益率曲线与切线 ==========

def plot_price_yield_curve():
    """
    展示债券价格与收益率的关系，
    在当前位置画切线（久期的几何意义）
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    face = 100
    coupon = 0.05
    years = 10
    current_ytm = 0.05

    yields = np.linspace(0.01, 0.12, 200)
    prices = [bond_price(face, coupon, y, years) for y in yields]

    # 主曲线
    ax.plot(yields * 100, prices, 'b-', linewidth=2.5, label='债券价格')

    # 当前价格点
    current_price = bond_price(face, coupon, current_ytm, years)
    ax.plot(current_ytm * 100, current_price, 'ro', markersize=10, zorder=5)
    ax.annotate(f'当前点\nYTM={current_ytm*100:.0f}%\n价格={current_price:.2f}',
                xy=(current_ytm * 100, current_price),
                xytext=(current_ytm * 100 + 2, current_price + 8),
                fontsize=11, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

    # 切线（修正久期的几何意义）
    mod_dur = modified_duration(face, coupon, current_ytm, years)
    tangent_slope = -mod_dur * current_price / 100
    y_tangent = current_price + tangent_slope * (yields * 100 - current_ytm * 100)
    ax.plot(yields * 100, y_tangent, 'r--', linewidth=1.5,
            label=f'切线（斜率 = -修正久期×价格）\n修正久期 ≈ {mod_dur:.2f}')

    # 标注久期含义
    delta_y = 0.01
    price_down = bond_price(face, coupon, current_ytm + delta_y, years)
    price_change_pct = (price_down - current_price) / current_price * 100
    ax.annotate(f'收益率+1%\n价格变化≈{price_change_pct:.2f}%',
                xy=((current_ytm + delta_y) * 100, price_down),
                xytext=((current_ytm + delta_y) * 100 + 1.5, price_down - 3),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('到期收益率 (%)', fontsize=13)
    ax.set_ylabel('债券价格 (元)', fontsize=13)
    ax.set_title('图1: 债券价格-收益率曲线与久期的几何意义', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 13)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/price_yield_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] 图1: 价格-收益率曲线已生成')


# ========== 图2: 不同票面利率下久期 vs 期限 ==========

def plot_duration_vs_maturity():
    """
    展示不同票面利率下，久期如何随期限变化
    体现: 期限越长、久期越大（但边际递减）
          票面利率越高、久期越短
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    face = 100
    ytm = 0.05
    maturities = np.arange(1, 31)
    coupon_rates = [0.00, 0.03, 0.05, 0.08]
    colors = ['red', 'green', 'blue', 'purple']
    labels = ['0% (零息债券)', '3%', '5%', '8%']

    for cr, color, label in zip(coupon_rates, colors, labels):
        durations = [macaulay_duration(face, cr, ytm, m) for m in maturities]
        ax.plot(maturities, durations, color=color, linewidth=2, label=label)

    # 对角线: y = x (零息债券的久期 = 期限)
    ax.plot(maturities, maturities, 'k--', linewidth=1, alpha=0.4, label='久期 = 期限（参考线）')

    ax.set_xlabel('债券期限 (年)', fontsize=13)
    ax.set_ylabel('麦考利久期 (年)', fontsize=13)
    ax.set_title('图2: 不同票面利率下久期 vs 期限', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 31)

    # 添加注释说明
    ax.annotate('利率越高→久期越短', xy=(15, 5), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.annotate('期限越长→久期越大\n但增速递减 (Malkiel定理四)', xy=(5, 10), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/duration_vs_maturity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] 图2: 久期 vs 期限已生成')


# ========== 图3: 久期 vs 票面利率 ==========

def plot_duration_vs_coupon():
    """
    展示不同期限下，久期如何随票面利率变化
    体现: 票面利率越高，久期越短
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    face = 100
    ytm = 0.05
    coupon_rates = np.linspace(0, 0.12, 50)
    years_list = [3, 5, 10, 20, 30]
    colors = ['orange', 'green', 'blue', 'purple', 'red']

    for years, color in zip(years_list, colors):
        durations = [macaulay_duration(face, cr, ytm, years) for cr in coupon_rates]
        ax.plot(coupon_rates * 100, durations, color=color, linewidth=2,
                label=f'{years}年期')

    ax.set_xlabel('票面利率 (%)', fontsize=13)
    ax.set_ylabel('麦考利久期 (年)', fontsize=13)
    ax.set_title('图3: 久期 vs 票面利率（不同期限）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax.annotate('票面利率↑ → 久期↓\n（高息票=提前收回更多现金流）',
                xy=(10, 5), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/duration_vs_coupon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] 图3: 久期 vs 票面利率已生成')


# ========== 图4: 线性近似 vs 实际价格变化 ==========

def plot_linear_approximation():
    """
    展示久期线性近似与实际价格变化的差异
    体现凸性的价值
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    face = 100
    coupon = 0.05
    years = 10
    current_ytm = 0.05

    # === 左图: 价格变化对比 ===
    ax = axes[0]
    current_price = bond_price(face, coupon, current_ytm, years)
    mod_dur = modified_duration(face, coupon, current_ytm, years)

    yield_changes = np.linspace(-0.05, 0.05, 100)
    actual_prices = np.array([bond_price(face, coupon, current_ytm + dy, years) for dy in yield_changes])
    actual_changes = (actual_prices - current_price) / current_price * 100

    # 久期线性近似
    linear_changes = -mod_dur * yield_changes * 100

    ax.plot(yield_changes * 100, actual_changes, 'b-', linewidth=2.5, label='实际价格变化')
    ax.plot(yield_changes * 100, linear_changes, 'r--', linewidth=2, label='久期线性近似')

    # 标注误差区域
    where_pos = actual_changes > linear_changes
    ax.fill_between(yield_changes * 100,
                     actual_changes, linear_changes,
                     where=where_pos,
                     color='green', alpha=0.15, label='凸性收益（实际 > 近似）')
    where_neg = actual_changes < linear_changes
    ax.fill_between(yield_changes * 100,
                     actual_changes, linear_changes,
                     where=where_neg,
                     color='red', alpha=0.1, label='凸性损失（实际 < 近似）')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')

    ax.set_xlabel('收益率变化 (bp, 基点)', fontsize=12)
    ax.set_ylabel('价格变化 (%)', fontsize=12)
    ax.set_title('线性近似 vs 实际价格变化', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # === 右图: 近似误差 ===
    ax = axes[1]
    error = actual_changes - linear_changes
    ax.plot(yield_changes * 100, error, 'g-', linewidth=2)
    ax.fill_between(yield_changes * 100, 0, error,
                     where=(error > 0), color='green', alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='-')

    ax.set_xlabel('收益率变化 (bp, 基点)', fontsize=12)
    ax.set_ylabel('近似误差 (百分点)', fontsize=12)
    ax.set_title('久期近似误差 = 凸性效应', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.annotate('利率下降时\n实际跌更少 ▲', xy=(-4, max(error)*0.7),
                fontsize=10, color='green', fontweight='bold')
    ax.annotate('利率上升时\n实际涨更多 ▲', xy=(3, max(error)*0.7),
                fontsize=10, color='green', fontweight='bold')

    fig.suptitle('图4: 凸性的价值 — 久期线性近似的局限', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/convexity_approximation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] 图4: 凸性近似已生成')


# ========== 图5: Python 代码截图 — 久期计算流程 ==========

def plot_duration_calculation_demo():
    """
    一张信息图，展示久期计算的完整流程
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # 表格数据：计算过程
    face = 100
    coupon_rate = 0.05
    ytm = 0.06
    years = 3
    freq = 1

    coupon = face * coupon_rate
    n = int(years * freq)
    r = ytm

    col_labels = ['期数 t', '现金流', '折现因子', '现值', 't × 现值']
    data = []
    total_pv = 0
    total_weighted = 0

    for t in range(1, n + 1):
        cf = coupon if t < n else coupon + face
        discount = 1 / (1 + r) ** t
        pv = cf * discount
        weighted = t * pv
        total_pv += pv
        total_weighted += weighted
        data.append([f'{t}', f'{cf:.2f}', f'{discount:.4f}', f'{pv:.2f}', f'{weighted:.2f}'])

    data.append(['合计', '', '', f'{total_pv:.2f}', f'{total_weighted:.2f}'])

    table = ax.table(cellText=data, colLabels=col_labels,
                     loc='center', cellLoc='center',
                     colWidths=[0.08, 0.15, 0.18, 0.15, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # 设置样式 — 确保所有单元格使用中文字体
    font_family = rcParams['font.sans-serif'][0]
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:  # header row
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold', family=font_family)
        elif row == len(data):  # total row
            cell.set_facecolor('#D6E4F0')
            cell.set_text_props(fontweight='bold', family=font_family)
        elif col == 0:  # first column
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(family=font_family)
        else:
            cell.set_text_props(family=font_family)

    # 添加计算结果
    mac_dur = total_weighted / total_pv
    mod_dur = mac_dur / (1 + r)

    result_text = (
        f'债券价格 = {total_pv:.2f} 元\n'
        f'麦考利久期 = {total_weighted:.2f} / {total_pv:.2f} = {mac_dur:.4f} 年\n'
        f'修正久期 = {mac_dur:.4f} / (1 + {ytm*100:.0f}%) = {mod_dur:.4f}\n'
        f'→ 收益率上升1%，价格约下跌 {mod_dur:.2f}%'
    )
    ax.text(0.5, -0.15, result_text, ha='center', va='top',
            fontsize=12, family=rcParams['font.sans-serif'][0],
            bbox=dict(boxstyle='round', facecolor='#FFF2CC', alpha=0.8))

    ax.set_title('久期计算完整流程演示', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/duration_calculation_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('[OK] 图5: 久期计算流程已生成')


# ========== 主函数 ==========

if __name__ == '__main__':
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'图片输出目录: {OUTPUT_DIR}')

    plot_price_yield_curve()
    plot_duration_vs_maturity()
    plot_duration_vs_coupon()
    plot_linear_approximation()
    plot_duration_calculation_demo()

    print(f'\n全部图片已生成到: {OUTPUT_DIR}')
    print('文件列表:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f'  {f}  ({size:.1f} KB)')
