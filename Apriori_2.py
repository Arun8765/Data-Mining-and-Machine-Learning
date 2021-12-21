import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('store_data.csv', header=None)

print(store_data.head())
print("Count: ",store_data[0].count())


records = []
# for i in range(0, 7501):
#     records.append([str(store_data.values[i,j]) for j in range(0, 20)])
n_rows=store_data[0].count()
n_columns=len(store_data.columns)
# print(n_rows,n_columns)
for i in range(0, n_rows):
    records.append([str(store_data.values[i,j]) for j in range(0, n_columns)])

# records= store_data.tolist()
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=2, min_length=2)
# association_rules = apriori(records, min_support=0.0045, min_confidence=0.1, min_lift=3, min_length=2)
association_results = list(association_rules)

# print(association_results)

print(len(association_results))
# for k in association_rules:
#     print(k)
# print(association_results[0],"\n\n\n\n\n")

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[1] + " / " + items[0])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")