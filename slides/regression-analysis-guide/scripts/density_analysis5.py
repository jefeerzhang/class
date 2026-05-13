import re

with open('../05_final/presentation.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到所有分隔符行
separator_lines = []
for i, line in enumerate(lines):
    line_stripped = line.strip()
    if line_stripped == '---':
        separator_lines.append(i)

# 页面从第一个分隔符后开始
pages = []
start_idx = 0
for sep_idx in separator_lines:
    if sep_idx > start_idx:
        page_lines = lines[start_idx:sep_idx]
        pages.append(page_lines)
    start_idx = sep_idx + 1
if start_idx < len(lines):
    pages.append(lines[start_idx:])

print("页面数量:", len(pages))

# 分析每页
results = []
for page_num, page_lines in enumerate(pages, 1):
    content_lines = [line.rstrip('\n') for line in page_lines if line.strip() != '']
    
    # 统计列表项
    list_items = 0
    nested_lists = 0
    for line in content_lines:
        if re.match(r'^\s*[-*]\s', line):
            list_items += 1
            if re.match(r'^\s{2,}[-*]\s', line):
                nested_lists += 1
        elif re.match(r'^\s*\d+\.\s', line):
            list_items += 1
            if re.match(r'^\s{2,}\d+\.\s', line):
                nested_lists += 1
    
    # 最长行字符数
    max_line_len = 0
    for line in content_lines:
        line_len = len(line)
        if line_len > max_line_len:
            max_line_len = line_len
    
    # 段落长度
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
    
    long_paragraphs = 0
    for para in paragraphs:
        para_lines = para.split('\n')
        if len(para_lines) > 2:  # 用户定义：>2行为过长
            long_paragraphs += 1
    
    # 密度评级（基于用户标准）
    density_rating = '低'
    issues = []
    
    # 每页文字行数
    if len(content_lines) > 8:
        density_rating = '高'
        issues.append('行数' + str(len(content_lines)))
    elif len(content_lines) > 6:
        if density_rating != '高':
            density_rating = '中'
        issues.append('行数' + str(len(content_lines)))
    
    # 每行字符数
    if max_line_len > 45:
        if density_rating != '高':
            density_rating = '高'
        issues.append('最长行' + str(max_line_len) + '字符')
    elif max_line_len > 35:
        if density_rating == '低':
            density_rating = '中'
        issues.append('最长行' + str(max_line_len) + '字符')
    
    # 列表项数量
    if list_items > 6:
        if density_rating != '高':
            density_rating = '高'
        issues.append('列表项' + str(list_items) + '项')
    elif list_items > 5:
        if density_rating == '低':
            density_rating = '中'
        issues.append('列表项' + str(list_items) + '项')
    
    # 嵌套列表深度
    if nested_lists > 0:
        if density_rating != '高':
            density_rating = '中'
        issues.append('嵌套列表深度>1层')
    
    # 段落长度
    if long_paragraphs > 0:
        if density_rating != '高':
            density_rating = '高'
        issues.append(str(long_paragraphs) + '个过长段落')
    
    results.append({
        'page': page_num,
        'lines': len(content_lines),
        'max_len': max_line_len,
        'list_items': list_items,
        'nested': nested_lists,
        'density': density_rating,
        'issues': issues
    })

# 输出表格
print("页码 行数 最长行 列表项 嵌套 密度评级 状态")
for r in results:
    status = 'OK' if r['density'] == '低' else ('WARN' if r['density'] == '中' else 'ERROR')
    print(str(r['page']) + ' ' + str(r['lines']) + ' ' + str(r['max_len']) + ' ' + str(r['list_items']) + ' ' + str(r['nested']) + ' ' + r['density'] + ' ' + status)

print("\n高密度页面（需处理）:")
for r in results:
    if r['density'] == '高':
        print('第' + str(r['page']) + '页：' + ', '.join(r['issues']))

print("\n中密度页面（需关注）:")
for r in results:
    if r['density'] == '中':
        print('第' + str(r['page']) + '页：' + ', '.join(r['issues']))