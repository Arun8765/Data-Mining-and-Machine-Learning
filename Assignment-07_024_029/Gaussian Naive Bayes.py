# Gaussian Naive Bayes Classifier

import sklearn
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
from sklearn import metrics 



col_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

hd = pd.read_csv(r"heart_Disease.csv")
hd['age'] = hd['age'].astype(float)
# print(hd)

#split dataset in features and target variable
feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X = hd[feature_cols] # Features
y = hd.target # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None) # 70% training and 30% test


'''Knn algorithm'''


# Create Decision Tree classifer object

clf =GaussianNB()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("****\nnaive bayes accuracy:",metrics.accuracy_score(y_test, y_pred),"\n****\n\n  ")


