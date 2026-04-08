"""
视频生成脚本
==========
使用 MoviePy 将图表图片合成为讲解视频
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from moviepy import *
from PIL import Image, ImageDraw, ImageFont
import os

# 路径配置
OUTPUT_DIR = r"C:\Users\jefeer\Downloads\opencode\关联算法"
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")

# 确保目录存在
os.makedirs(VIDEO_DIR, exist_ok=True)

# 视频配置
WIDTH = 1280
HEIGHT = 720
DURATION_PER_SLIDE = 8  # 每页停留秒数
FPS = 24

# 讲解内容（字幕）
SUBTITLES = [
    {
        "time": 0,
        "end": 5,
        "text": "FP-Growth 与 Apriori 算法对比分析"
    },
    {
        "time": 5,
        "end": 12,
        "text": "本报告基于银行客户金融产品持有数据，分析关联规则挖掘算法"
    },
    {
        "time": 12,
        "end": 20,
        "text": "实验参数：100位客户，6种金融产品，最小支持度25%"
    },
    {
        "time": 20,
        "end": 28,
        "text": "产品渗透率分析：储蓄账户渗透率最高达91%，是最核心的基础产品"
    },
    {
        "time": 28,
        "end": 36,
        "text": "基金渗透率74%，位居第二，显示较强的投资理财需求"
    },
    {
        "time": 36,
        "end": 44,
        "text": "理财产品渗透率最低36%，属于高净值客户产品"
    },
    {
        "time": 44,
        "end": 52,
        "text": "频繁项集分析：2项集中，储蓄账户+基金组合最常见达65%"
    },
    {
        "time": 52,
        "end": 60,
        "text": "3项集中，储蓄账户+基金+保险组合覆盖42%客户"
    },
    {
        "time": 60,
        "end": 68,
        "text": "FP-Growth算法只需扫描数据库2次，不生成候选项集"
    },
    {
        "time": 68,
        "end": 76,
        "text": "Apriori算法需要多次扫描数据库，生成大量候选项集"
    },
    {
        "time": 76,
        "end": 84,
        "text": "性能对比：FP-Growth比Apriori快约1.26倍"
    },
    {
        "time": 84,
        "end": 92,
        "text": "大规模数据时，FP-Growth优势会更加显著"
    },
    {
        "time": 92,
        "end": 100,
        "text": "高质量规则：基金+信用卡推荐保险，提升度最高达1.31"
    },
    {
        "time": 100,
        "end": 108,
        "text": "信用卡+保险推荐基金，置信度高达91.4%"
    },
    {
        "time": 108,
        "end": 116,
        "text": "贷款客户100%持有储蓄账户，是强关联规则"
    },
    {
        "time": 116,
        "end": 124,
        "text": "规则网络图清晰展示：前置条件到推荐结果的关联路径"
    },
    {
        "time": 124,
        "end": 132,
        "text": "业务建议：对基金+信用卡客户推荐保险，预期转化率80%"
    },
    {
        "time": 132,
        "end": 140,
        "text": "对信用卡+保险客户推荐基金，预期转化率91.4%"
    },
    {
        "time": 140,
        "end": 148,
        "text": "算法选择建议：数据量小于1000条，两者皆可"
    },
    {
        "time": 148,
        "end": 156,
        "text": "数据量1000到10000条，推荐使用FP-Growth"
    },
    {
        "time": 156,
        "end": 164,
        "text": "数据量超过10000条，必须使用FP-Growth"
    },
    {
        "time": 164,
        "end": 172,
        "text": "结论：FP-Growth在保持相同结果的同时，性能更优"
    },
    {
        "time": 172,
        "end": 180,
        "text": "感谢观看！关联规则挖掘，挖掘数据中的隐藏价值"
    },
]


def create_title_slide():
    """创建标题页"""
    img = Image.new('RGB', (WIDTH, HEIGHT), color=(30, 60, 90))
    draw = ImageDraw.Draw(img)

    # 标题
    try:
        font_large = ImageFont.truetype("arial.ttf", 60)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    title = "FP-Growth 与 Apriori"
    subtitle = "算法对比分析报告"

    # 居中绘制
    bbox = draw.textbbox((0, 0), title, font=font_large)
    title_width = bbox[2] - bbox[0]
    draw.text(((WIDTH - title_width) // 2, HEIGHT // 3), title, fill='white', font=font_large)

    bbox = draw.textbbox((0, 0), subtitle, font=font_small)
    subtitle_width = bbox[2] - bbox[0]
    draw.text(((WIDTH - subtitle_width) // 2, HEIGHT // 2), subtitle, fill=(200, 200, 200), font=font_small)

    return img


def add_subtitle(img, text):
    """为图片添加字幕条"""
    draw = ImageDraw.Draw(img)

    # 底部字幕条
    bar_height = 60
    draw.rectangle([(0, HEIGHT - bar_height), (WIDTH, HEIGHT)], fill=(0, 0, 0))

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text(((WIDTH - text_width) // 2, HEIGHT - bar_height + 15), text, fill='white', font=font)

    return img


def generate_srt():
    """生成字幕文件"""
    srt_content = ""
    for i, sub in enumerate(SUBTITLES, 1):
        start = f"{int(sub['time'] // 3600):02d}:{int((sub['time'] % 3600) // 60):02d}:{int(sub['time'] % 60):02d},000"
        end = f"{int(sub['end'] // 3600):02d}:{int((sub['end'] % 3600) // 60):02d}:{int(sub['end'] % 60):02d},000"
        srt_content += f"{i}\n{start} --> {end}\n{sub['text']}\n\n"

    with open(os.path.join(VIDEO_DIR, "subtitles.srt"), "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"[OK] 字幕文件已保存: {VIDEO_DIR}/subtitles.srt")


def create_video():
    """创建视频"""
    print("=" * 60)
    print("开始生成视频")
    print("=" * 60)

    clips = []

    # 1. 标题页
    print("创建标题页...")
    title_img = create_title_slide()
    title_img = add_subtitle(title_img, "FP-Growth 与 Apriori 算法对比分析")
    title_path = os.path.join(VIDEO_DIR, "title.png")
    title_img.save(title_path)
    title_clip = ImageClip(title_path).with_duration(5)
    clips.append(title_clip)

    # 2. 数据概况页
    print("创建数据概况页...")
    data_img = Image.new('RGB', (WIDTH, HEIGHT), color=(250, 250, 250))
    draw = ImageDraw.Draw(data_img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 40)
        font_text = ImageFont.truetype("arial.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    draw.text((50, 50), "1. 实验概述", fill=(30, 60, 90), font=font_title)

    info = [
        "客户数量: 100 人",
        "产品种类: 6 种",
        "最小支持度: 0.25 (25%)",
        "最小置信度: 0.6 (60%)",
        "",
        "产品列表:",
        "储蓄账户、基金、保险、信用卡、贷款、理财产品"
    ]

    for i, line in enumerate(info):
        draw.text((80, 150 + i * 35), line, fill=(50, 50, 50), font=font_text)

    data_img = add_subtitle(data_img, "实验参数设置：100位客户，6种金融产品，最小支持度25%")
    data_path = os.path.join(VIDEO_DIR, "data_overview.png")
    data_img.save(data_path)
    data_clip = ImageClip(data_path).with_duration(8)
    clips.append(data_clip)

    # 3. 产品渗透率饼图
    print("添加产品渗透率饼图...")
    pie_path = os.path.join(OUTPUT_DIR, "product_penetration.png")
    if os.path.exists(pie_path):
        pie_img = Image.open(pie_path)
        pie_img = pie_img.resize((WIDTH, HEIGHT - 60))
        pie_with_sub = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        pie_with_sub.paste(pie_img, (0, 0))
        pie_with_sub = add_subtitle(pie_with_sub, "产品渗透率分析：储蓄账户91%最高，理财产品36%最低")
        pie_path_out = os.path.join(VIDEO_DIR, "pie_chart.png")
        pie_with_sub.save(pie_path_out)
        pie_clip = ImageClip(pie_path_out).with_duration(10)
        clips.append(pie_clip)

    # 4. 频繁项集柱状图
    print("添加频繁项集柱状图...")
    bar_path = os.path.join(OUTPUT_DIR, "frequent_itemsets.png")
    if os.path.exists(bar_path):
        bar_img = Image.open(bar_path)
        bar_img = bar_img.resize((WIDTH, HEIGHT - 60))
        bar_with_sub = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        bar_with_sub.paste(bar_img, (0, 0))
        bar_with_sub = add_subtitle(bar_with_sub, "频繁项集分析：2项集12个，3项集7个，4项集1个")
        bar_path_out = os.path.join(VIDEO_DIR, "bar_chart.png")
        bar_with_sub.save(bar_path_out)
        bar_clip = ImageClip(bar_path_out).with_duration(10)
        clips.append(bar_clip)

    # 5. 算法对比页
    print("创建算法对比页...")
    algo_img = Image.new('RGB', (WIDTH, HEIGHT), color=(240, 240, 240))
    draw = ImageDraw.Draw(algo_img)

    draw.text((50, 50), "4. 算法性能对比", fill=(30, 60, 90), font=font_title)

    comparison = [
        "",
        "算法          FP-Growth        Apriori",
        "扫描次数       2次             多次",
        "候选项集       无             大量",
        "执行速度       快             慢",
        "",
        "FP-Growth 比 Apriori 快约 1.26 倍",
        "（当前数据量差异不明显，大规模数据优势显著）"
    ]

    for i, line in enumerate(comparison):
        draw.text((80, 140 + i * 40), line, fill=(50, 50, 50), font=font_text)

    algo_img = add_subtitle(algo_img, "FP-Growth只需2次扫描，不生成候选项集，性能更优")
    algo_path = os.path.join(VIDEO_DIR, "algo_comparison.png")
    algo_img.save(algo_path)
    algo_clip = ImageClip(algo_path).with_duration(10)
    clips.append(algo_clip)

    # 6. 规则散点图
    print("添加规则散点图...")
    scatter_path = os.path.join(OUTPUT_DIR, "rules_scatter.png")
    if os.path.exists(scatter_path):
        scatter_img = Image.open(scatter_path)
        scatter_img = scatter_img.resize((WIDTH, HEIGHT - 60))
        scatter_with_sub = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        scatter_with_sub.paste(scatter_img, (0, 0))
        scatter_with_sub = add_subtitle(scatter_with_sub, "关联规则分布：支持度、置信度、提升度三维度分析")
        scatter_path_out = os.path.join(VIDEO_DIR, "scatter_chart.png")
        scatter_with_sub.save(scatter_path_out)
        scatter_clip = ImageClip(scatter_path_out).with_duration(10)
        clips.append(scatter_clip)

    # 7. 提升度分布图
    print("添加提升度分布图...")
    lift_path = os.path.join(OUTPUT_DIR, "lift_distribution.png")
    if os.path.exists(lift_path):
        lift_img = Image.open(lift_path)
        lift_img = lift_img.resize((WIDTH, HEIGHT - 60))
        lift_with_sub = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        lift_with_sub.paste(lift_img, (0, 0))
        lift_with_sub = add_subtitle(lift_with_sub, "提升度分布：75%分位1.08，90%分位1.17")
        lift_path_out = os.path.join(VIDEO_DIR, "lift_dist.png")
        lift_with_sub.save(lift_path_out)
        lift_clip = ImageClip(lift_path_out).with_duration(8)
        clips.append(lift_clip)

    # 8. 规则网络图
    print("添加规则网络图...")
    network_path = os.path.join(OUTPUT_DIR, "rules_network.png")
    if os.path.exists(network_path):
        network_img = Image.open(network_path)
        network_img = network_img.resize((WIDTH, HEIGHT - 60))
        network_with_sub = Image.new('RGB', (WIDTH, HEIGHT), color=(255, 255, 255))
        network_with_sub.paste(network_img, (0, 0))
        network_with_sub = add_subtitle(network_with_sub, "规则网络：红色=前置条件，蓝色=推荐结果")
        network_path_out = os.path.join(VIDEO_DIR, "network_chart.png")
        network_with_sub.save(network_path_out)
        network_clip = ImageClip(network_path_out).with_duration(10)
        clips.append(network_clip)

    # 9. 高质量规则页
    print("创建高质量规则页...")
    rules_img = Image.new('RGB', (WIDTH, HEIGHT), color=(250, 250, 250))
    draw = ImageDraw.Draw(rules_img)

    draw.text((50, 50), "高质量规则 (lift > 1.1 且 confidence > 70%)", fill=(30, 60, 90), font=font_title)

    rules_text = [
        "",
        "规则                                    支持度  置信度  提升度",
        "基金+信用卡 → 保险                      32%     80%    1.31",
        "基金+信用卡+储蓄账户 → 保险              26%     76.5%  1.25",
        "信用卡+保险 → 基金                      32%     91.4%  1.24",
        "信用卡+储蓄账户+保险 → 基金              26%     89.7%  1.21",
        "理财产品 → 基金                          30%     83.3%  1.13",
        "",
        "最强关联：基金+信用卡 → 保险（lift=1.31）",
        "最高置信度：信用卡+保险 → 基金（91.4%）"
    ]

    for i, line in enumerate(rules_text):
        color = (50, 50, 50) if i > 0 else (30, 30, 30)
        draw.text((60, 130 + i * 35), line, fill=color, font=font_text)

    rules_img = add_subtitle(rules_img, "高质量规则：基金+信用卡推荐保险，提升度最高")
    rules_path = os.path.join(VIDEO_DIR, "quality_rules.png")
    rules_img.save(rules_path)
    rules_clip = ImageClip(rules_path).with_duration(12)
    clips.append(rules_clip)

    # 10. 业务建议页
    print("创建业务建议页...")
    biz_img = Image.new('RGB', (WIDTH, HEIGHT), color=(240, 248, 255))
    draw = ImageDraw.Draw(biz_img)

    draw.text((50, 50), "5. 业务应用建议", fill=(30, 60, 90), font=font_title)

    biz_text = [
        "",
        "交叉销售策略：",
        "客户现状              推荐产品       预期转化率",
        "已购买基金+信用卡     推荐保险       80.0%",
        "已购买信用卡+保险     推荐基金       91.4%",
        "已购买理财产品       推荐基金       83.3%",
        "已购买贷款           推荐储蓄账户    100%",
        "",
        "产品组合套餐：",
        "基础套餐：储蓄账户 + 基金（覆盖率65%）",
        "标准套餐：储蓄账户 + 基金 + 保险（覆盖率42%）"
    ]

    for i, line in enumerate(biz_text):
        draw.text((60, 130 + i * 32), line, fill=(50, 50, 50), font=font_text)

    biz_img = add_subtitle(biz_img, "业务建议：基于高质量规则设计交叉销售策略")
    biz_path = os.path.join(VIDEO_DIR, "business_advice.png")
    biz_img.save(biz_path)
    biz_clip = ImageClip(biz_path).with_duration(12)
    clips.append(biz_clip)

    # 11. 结论页
    print("创建结论页...")
    conclusion_img = Image.new('RGB', (WIDTH, HEIGHT), color=(30, 60, 90))
    draw = ImageDraw.Draw(conclusion_img)

    draw.text((50, 80), "6. 结论", fill='white', font=font_title)

    conclusion_text = [
        "",
        "1. FP-Growth在保持相同结果的同时，性能更优",
        "",
        "2. 最强关联：基金+信用卡 → 保险（lift=1.31）",
        "",
        "3. 最高置信度：信用卡+保险 → 基金（91.4%）",
        "",
        "4. 储蓄账户是核心产品（91%渗透率）",
        "",
        "算法选择：",
        "< 1,000条：两者皆可",
        "1,000-10,000条：推荐FP-Growth",
        "> 10,000条：必须使用FP-Growth"
    ]

    for i, line in enumerate(conclusion_text):
        draw.text((80, 150 + i * 35), line, fill=(220, 220, 220), font=font_text)

    conclusion_img = add_subtitle(conclusion_img, "感谢观看！关联规则挖掘，挖掘数据中的隐藏价值")
    conclusion_path = os.path.join(VIDEO_DIR, "conclusion.png")
    conclusion_img.save(conclusion_path)
    conclusion_clip = ImageClip(conclusion_path).with_duration(10)
    clips.append(conclusion_clip)

    # 拼接视频
    print("\n拼接视频片段...")
    final_clip = concatenate_videoclips(clips)

    # 保存视频
    video_path = os.path.join(VIDEO_DIR, "algorithm_comparison.mp4")
    print(f"\n保存视频: {video_path}")
    final_clip.write_videofile(video_path, fps=FPS, codec='libx264', audio=False)

    print("\n" + "=" * 60)
    print("视频生成完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  视频: {video_path}")
    print(f"  字幕: {VIDEO_DIR}/subtitles.srt")

    return video_path


if __name__ == "__main__":
    # 先确保图表已生成
    print("检查图表文件...")

    charts = [
        "product_penetration.png",
        "frequent_itemsets.png",
        "rules_scatter.png",
        "rules_network.png",
        "lift_distribution.png"
    ]

    missing = []
    for chart in charts:
        path = os.path.join(OUTPUT_DIR, chart)
        if not os.path.exists(path):
            missing.append(chart)

    if missing:
        print(f"警告：以下图表缺失，需要先运行 visualize_rules.py")
        for m in missing:
            print(f"  - {m}")
    else:
        print("所有图表已就绪")

    # 生成字幕
    generate_srt()

    # 创建视频
    create_video()
