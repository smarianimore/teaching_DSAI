import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [['Milk', 'Cereals'],
           ['Milk', 'Biscuits', 'Tea'],
           ['Milk', 'Cereals', 'Biscuits'],
           ['Biscuits', 'Tea', 'Ham'],
           ['Milk', 'Cereals', 'Biscuits', 'Tea']]

encoder = TransactionEncoder()
transactions = encoder.fit(dataset).transform(dataset)
df = pd.DataFrame(transactions, columns=encoder.columns_)
print(df)

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, df.shape[0], metric="lift", min_threshold=1.5)
sorted_rules = rules.sort_values('lift', ascending=False)
print(sorted_rules[["antecedents", "consequents", "lift"]])

rules = association_rules(frequent_itemsets, df.shape[0], metric="confidence", min_threshold=0.75)
sorted_rules = rules.sort_values('confidence', ascending=False)
print(sorted_rules[["antecedents", "consequents", "confidence"]])
