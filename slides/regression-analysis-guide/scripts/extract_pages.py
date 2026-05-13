import re

with open('../05_final/presentation.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

separator_lines = []
for i, line in enumerate(lines):
    line_stripped = line.strip()
    if line_stripped == '---':
        separator_lines.append(i)

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

for page_num, page_lines in enumerate(pages, 1):
    print("\n=== 第" + str(page_num) + "页 ===")
    content_lines = [line.rstrip('\n') for line in page_lines if line.strip() != '']
    # 打印前几行
    for i, line in enumerate(content_lines[:5]):
        print(line)
    if len(content_lines) > 5:
        print("... (共" + str(len(content_lines)) + "行)")
    
    # 找到标题
    for line in content_lines:
        if line.startswith('## ') or line.startswith('# '):
            print("标题: " + line)
            break