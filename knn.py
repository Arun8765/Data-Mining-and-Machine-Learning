#knn classifier 

import sklearn
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
# from CHAID import Tree
# from chefboost import Chefboost as chef
import matplotlib.pyplot as plt


col_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

hd = pd.read_csv(r"heart_Disease.csv")

hd['age'] = hd['age'].astype(float)
# print(hd)

#split dataset in features and target variable
feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X = hd[feature_cols] # Features
y = hd.target # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


'''Knn algorithm'''

x=[]
y=[]
# Create Decision Tree classifer object
for i in range(3,100):
    clf = KNeighborsClassifier(n_neighbors=i)#euclidean distance measure

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("****\nKNN accuracy:",metrics.accuracy_score(y_test, y_pred),"\n****\n\n  ")
    y.append(metrics.accuracy_score(y_test, y_pred))
    x.append(i)

plt.plot(x,y)
plt.show()