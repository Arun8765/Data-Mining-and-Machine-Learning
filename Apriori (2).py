from apyori import apriori

records=[
    ['milk','shrimp','almonds','beer'],
    ['chicken','chilli','cheese','carrot'],
    ['milk','chilli','carrot','beer'],
    ['shrimp','mutton','butter','bread','cabbage'],
    ['orange','sausage','tomato','almonds'],
    ['orange','mutton','butter','carrot']
]

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=2, min_length=2)

association_results = list(association_rules)

print(len(association_results))

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
    print("=====================================\n")
