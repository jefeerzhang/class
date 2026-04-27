import re

with open('../05_final/presentation.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到第一个---之后的内容（跳过frontmatter）
# frontmatter以---开始和结束
first_separator = content.find('---')
if first_separator == -1:
    print("没有找到分隔符")
    exit()

# 跳过第一个frontmatter
after_frontmatter = content[first_separator+3:]  # 跳过---

# 现在分割页面（每个页面以---开始）
# 使用正则表达式：\n---\n
pages_raw = re.split(r'\n---\n', after_frontmatter)
print(f"原始页面数: {len(pages_raw)}")

# 合并可能被错误分割的页面（例如表格内的---）
pages = []
current_page = []
for part in pages_raw:
    part = part.strip()
    if not part:
        continue
    # 检查是否是新的页面开始（通常以##开头）
    if part.startswith('## ') or part.startswith('# ') or part.startswith('**重点'):
        if current_page:
            pages.append('\n'.join(current_page))
            current_page = []
        current_page.append(part)
    else:
        # 继续当前页面
        current_page.append(part)
if current_page:
    pages.append('\n'.join(current_page))

print(f"合并后页面数: {len(pages)}")

# 分析每页
results = []
for i, page in enumerate(pages, 1):
    lines = page.strip().split('\n')
    
    # 统计有内容的行（非空行）
    content_lines = [line for line in lines if line.strip() != '']
    
    # 统计列表项
    list_items = 0
    nested_lists = 0
    for line in content_lines:
        # 无序列表
        if re.match(r'^\s*[-*]\s', line):
            list_items += 1
            # 检查嵌套（缩进）
            if re.match(r'^\s{2,}[-*]\s', line):
                nested_lists += 1
        # 有序列表
        elif re.match(r'^\s*\d+\.\s', line):
            list_items += 1
            if re.match(r'^\s{2,}\d+\.\s', line):
                nested_lists += 1
    
    # 计算最长行的字符数
    max_line_len = 0
    for line in content_lines:
        line_len = len(line)
        if line_len > max_line_len:
            max_line_len = line_len
    
    # 计算段落长度（连续非列表非空行）
    paragraphs = []
    current_para = []
    for line in content_lines:
        # 跳过列表项、表格行、标题
        if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+\.\s', line) or line.startswith('|') or line.startswith('#') or line.startswith('$$') or line.startswith('**') or line.startswith('---'):
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        paragraphs.append('\n'.join(current_para))
    
    # 检查过长段落（>3行）
    long_paragraphs = 0
    for para in paragraphs:
        para_lines = para.split('\n')
        if len(para_lines) > 3:
            long_paragraphs += 1
    
    # 密度评级
    density_rating = '低'
    issues = []
    if len(content_lines) > 8:
        density_rating = '高'
        issues.append(f'行数{len(content_lines)}')
    elif len(content_lines) > 6:
        density_rating = '中'
        issues.append(f'行数{len(content_lines)}')
    
    if max_line_len > 45:
        density_rating = '高' if density_rating != '高' else density_rating
        issues.append(f'最长行{max_line_len}字符')
    elif max_line_len > 35 and density_rating == '低':
        density_rating = '中'
        issues.append(f'最长行{max_line_len}字符')
    
    if list_items > 6:
        density_rating = '高' if density_rating != '高' else density_rating
        issues.append(f'列表项{list_items}项')
    elif list_items > 5 and density_rating == '低':
        density_rating = '中'
        issues.append(f'列表项{list_items}项')
    
    if long_paragraphs > 0:
        density_rating = '高' if density_rating != '高' else density_rating
        issues.append(f'{long_paragraphs}个过长段落')
    
    results.append({
        'page': i,
        'lines': len(content_lines),
        'max_len': max_line_len,
        'list_items': list_items,
        'density': density_rating,
        'issues': issues
    })

# 输出表格
print('| 页码 | 行数 | 最长行 | 列表项 | 密度评级 | 状态 |')
print('|------|------|--------|--------|----------|------|')
for r in results:
    status = 'OK' if r['density'] == '低' else ('WARN' if r['density'] == '中' else 'ERROR')
    print(f"| {r['page']} | {r['lines']} | {r['max_len']}字符 | {r['list_items']}项 | {r['density']} | {status} |")

print("\n高密度页面（需处理）:")
for r in results:
    if r['density'] == '高':
        print(f"- 第{r['page']}页：{', '.join(r['issues'])}")

# 特别检查：段落长度
print("\n段落长度检查（>3行）:")
for i, page in enumerate(pages, 1):
    lines = page.strip().split('\n')
    content_lines = [line for line in lines if line.strip() != '']
    paragraphs = []
    current_para = []
    for line in content_lines:
        if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+\.\s', line) or line.startswith('|') or line.startswith('#') or line.startswith('$$') or line.startswith('**') or line.startswith('---'):
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        paragraphs.append('\n'.join(current_para))
    
    for j, para in enumerate(paragraphs):
        para_lines = para.split('\n')
        if len(para_lines) > 3:
            print(f"- 第{i}页，段落{j+1}：{len(para_lines)}行")
            print(f"  内容：{para[:100]}...")