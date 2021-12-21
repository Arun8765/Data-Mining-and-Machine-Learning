import pandas as pd
import numpy as np
import mord as m
from sklearn.metrics import r2_score
# import sklearn as sk
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import linear_model
reg=linear_model.LinearRegression()

c= m.OrdinalRidge()
df= pd.read_csv("tips.csv")

# print(df)
df['sex']=df['sex'].map({'Female':0,'Male':1})
df['smoker']=df['smoker'].map({'Yes':1,'No':0})
df['day']=df['day'].map({'Sun':3,'Thur':0,'Fri':1,'Sat':2})
df['time']=df['time'].map({'Lunch':0,'Dinner':1})
df['size']=df['size'].astype(int)
df['total_bill']=df['total_bill'].astype(float)
df['tip']=df['tip'].astype(float)
# print(df)
# print(df['tip'].astype(int))
# print(np.array(df['tip'].astype(int)))

# print(np.array(df[['total_bill','sex','smoker','day','time','size']]))
# lst=list(map(int,"1  1  3  3  3  4  2  3  1  3  1  5  1  3  3  3  1  3  3  3  4  2  2  7  3  2  2  2  4  3  1  2  3  2  3  3  2  3  2  5  2  2  3  1  5  3  5  6  2  3  2  2  5  1  4  3  3  1  1  6  3  2  1  3  2  3  2  1  2  2  1  3  3  5  2  1  3  4  3  2  3  3  1  5  2  5  2  4  5  3  3  3  1  4  3  4  4  1  3  1  2  3  2  3  4  1  4  4  3  4  3  1  4  2  4  3  5  1  1  2  2  1  2  2  2  4  1  2  2  2  1  2  1  2  3  1  2  2  2  2  3  6  5  5  2  1  1  1  1  2  2  2  2  2  2  5  5  3  2  2  3  2  2  2  3  3  2  4  1  2 10  3  5  3  4  3  2  2  4  3  3  5  3  6  3  5  3  2  3  4  1  4  2  2  4  1  2  5  2  2  4  2  2  2  4  3  3  3  2  2  2  5  9  2  6  1  3  1  1  3  2  3  1  3  1  2  2  3  2  2  2  3  3  1  3  1  1  1  4  5  2  2  1  3".split()))
# print(lst)
X_train, X_test, y_train, y_test = train_test_split(df[['total_bill','sex','smoker','day','time','size']], df['tip'].astype(float), train_size=0.8)

# slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(X_train),np.array(y_train))
# print(r_value)

c.fit(np.array(X_train),np.array(y_train))
reg.fit(np.array(X_train),np.array(y_train))
# print(c.predict(np.array(X_test)))
# print(reg.predict(np.array(X_test)))
print("The coeffecients and intercept of multivariate linear regression: ")
print(reg.coef_,",",reg.intercept_,"\n")
r2_lin= r2_score(np.array(y_test),reg.predict(np.array(X_test)))
r2_ord = r2_score(np.array(y_test),c.predict(np.array(X_test)))
print("Similarity score of Multivariate Linear Regression:",r2_lin)
print("Similarity score of Ordinal Regression:",r2_ord)
# 28.55	2.05	Male	No	Sun	Dinner	3

# from pylab import *
# scatter(df["tip"],df["total_bill"])
# print(df.columns[1],"\n",df.columns[0])
# print(df[['tip','total_bill']].loc[[4]])
# print(df[['total_bill','sex','smoker','day','time','size']])
# for i in range(len(df.columns)):
#     for j in range(i+1, len(df.columns)):
#         scatter(df[df.columns[i]], df[df.columns[j]])

