"""Importing pandas and apriori packages"""
import pandas as pd
from apyori import apriori

'''Reading the csv dataset'''
store_data = pd.read_csv('store_data.csv', header=None)

'''Inserting the dataset into a list'''
records = []
n_rows=store_data[0].count()
n_columns=len(store_data.columns)
for i in range(0, n_rows):
    records.append([str(store_data.values[i,j]) for j in range(0, n_columns)])

'''Recieving the results for the interested association thresholds'''
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

''' Printing the association rules '''
print("The threshold for Association Rules set are:\n"
      " Min support: 0.0045\n Min confidence: 0.2\n Min Lift: 3\n Min Length: 2")
print("The number of assosiations in the data that satisfy the rules is: ",len(association_results),"\n\n====")

'''Printing the resulting Rule, Support, Confidence and Lift'''
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