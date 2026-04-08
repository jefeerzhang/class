from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = [
    ["面包", "牛奶", "尿布"],
    ["面包", "牛奶", "啤酒", "尿布"],
    ["牛奶", "尿布", "鸡蛋"],
    ["面包", "鸡蛋"],
    ["牛奶", "尿布", "啤酒"],
]

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("频繁项集:")
print(frequent_itemsets)
print("\n关联规则:")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
