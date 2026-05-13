import re
with open('../05_final/presentation.md', 'r', encoding='utf-8') as f:
    content = f.read()
# 找到第一个---之后的内容
parts = content.split('---')
print(f'部分数: {len(parts)}')
for i, part in enumerate(parts[:5]):
    lines = part.strip().split('\n')
    print(f'部分{i}: {len(lines)}行，第一行: {lines[0] if lines else "空"}')