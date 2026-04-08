"""
Sequential Pattern Mining - Temporal Association Rules

Analyze temporal patterns: If customer buys X first, then buys Y later.
"""

import pandas as pd
import numpy as np
from prefixspan import PrefixSpan
from datetime import datetime
from collections import defaultdict

# Read temporal data
csv_file = r"C:\Users\jefeer\Downloads\opencode\关联算法\data\customer_products_temporal.csv"
df = pd.read_csv(csv_file)

print("=" * 60)
print("Sequential Pattern Mining - Temporal Association Rules")
print("=" * 60)

# ============================================================
# Step 1: Aggregate customer product sequences
# ============================================================

# Convert date to datetime for sorting
df["开通日期"] = pd.to_datetime(df["开通日期"])

# Group by customer, sort by date, extract product sequences
customer_sequences = df.sort_values(["客户ID", "开通日期"]).groupby("客户ID")["产品"].apply(list).tolist()

print(f"\nCustomer count: {len(customer_sequences)}")
print(f"Avg products per customer: {np.mean([len(s) for s in customer_sequences]):.2f}")

# ============================================================
# Step 2: Mine frequent sequential patterns
# ============================================================

ps = PrefixSpan(customer_sequences)

# Find patterns with support >= 5 customers
min_support = 5
patterns = ps.frequent(min_support)

# Sort by support descending
patterns.sort(key=lambda x: -x[0])

print(f"\n" + "=" * 60)
print(f"Frequent Sequential Patterns (support >= {min_support} customers)")
print("=" * 60)

# Filter patterns with 2+ items
meaningful_patterns = [(sup, pat) for sup, pat in patterns if len(pat) >= 2]

print(f"\nFound {len(meaningful_patterns)} meaningful patterns:\n")

for i, (sup, pattern) in enumerate(meaningful_patterns[:15], 1):
    support_pct = sup / len(customer_sequences) * 100
    print(f"  {i}. {' -> '.join(pattern)}")
    print(f"     Support: {sup} customers ({support_pct:.1f}%)")
    print()

# ============================================================
# Step 3: Generate temporal association rules
# ============================================================

print("\n" + "=" * 60)
print("Temporal Association Rules")
print("=" * 60)

# Generate rules from each pattern
rules_found = []
for sup, pattern in meaningful_patterns:
    if len(pattern) >= 2:
        # Generate A -> B rules
        for i in range(1, len(pattern)):
            antecedent = tuple(pattern[:i])
            consequent = pattern[i]

            # Calculate confidence
            conf_count = 0
            total_count = 0

            for seq_items in customer_sequences:
                # Check if antecedent appears in sequence
                ant_found = False
                ant_len = len(antecedent)

                for j in range(len(seq_items) - ant_len + 1):
                    if tuple(seq_items[j:j + ant_len]) == antecedent:
                        ant_found = True
                        break

                if ant_found:
                    total_count += 1
                    # Check if consequent appears after antecedent
                    for item in seq_items[j + ant_len:]:
                        if item == consequent:
                            conf_count += 1
                            break

            if total_count > 0:
                confidence = conf_count / total_count
                support_pct = sup / len(customer_sequences) * 100

                rules_found.append({
                    "antecedent": " -> ".join(antecedent),
                    "consequent": consequent,
                    "support": support_pct,
                    "confidence": confidence * 100,
                })

# Sort by confidence
rules_found.sort(key=lambda x: -x["confidence"])

print("\nRule interpretation: If customer buys X FIRST, then buys Y LATER")
print("-" * 60)

for i, rule in enumerate(rules_found[:10], 1):
    print(f"\n  Rule {i}: {rule['antecedent']} -> {rule['consequent']}")
    print(f"    Support: {rule['support']:.1f}% (pattern frequency)")
    print(f"    Confidence: {rule['confidence']:.1f}% (prob of consequent given antecedent)")

# ============================================================
# Step 4: Product adoption path analysis
# ============================================================

print("\n" + "=" * 60)
print("Typical Product Adoption Paths")
print("=" * 60)

# First product distribution
first_products = defaultdict(int)
for seq in customer_sequences:
    if len(seq) > 0:
        first_products[seq[0]] += 1

print("\nFirst product (initial adoption):")
for product, count in sorted(first_products.items(), key=lambda x: -x[1]):
    pct = count / len(customer_sequences) * 100
    print(f"  {product}: {count} customers ({pct:.1f}%)")

# Second product distribution (conditional on first)
print("\nSecond product (given first product):")
second_products = defaultdict(lambda: defaultdict(int))
for seq in customer_sequences:
    if len(seq) >= 2:
        second_products[seq[0]][seq[1]] += 1

for first_prod in sorted(second_products.keys()):
    print(f"\n  After first buying [{first_prod}], second product distribution:")
    total = sum(second_products[first_prod].values())
    for second_prod, count in sorted(second_products[first_prod].items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    -> {second_prod}: {count} times ({pct:.1f}%)")

# ============================================================
# Step 5: Time gap analysis
# ============================================================

print("\n" + "=" * 60)
print("Average Time Gaps Between Products")
print("=" * 60)

# Calculate adjacent product time gaps
adjacent_gaps = defaultdict(list)

for cid, group in df.groupby("客户ID"):
    sorted_group = group.sort_values("开通日期")
    products = sorted_group["产品"].tolist()
    dates = sorted_group["开通日期"].tolist()

    for i in range(len(dates) - 1):
        gap_days = (dates[i + 1] - dates[i]).days
        adjacent_gaps[f"{products[i]} -> {products[i + 1]}"].append(gap_days)

print("\nAverage gap between adjacent products:")
for path, gaps in sorted(adjacent_gaps.items(), key=lambda x: np.mean(x[1])):
    avg_gap = np.mean(gaps)
    print(f"  {path}: {avg_gap:.0f} days ({avg_gap / 30:.1f} months)")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
