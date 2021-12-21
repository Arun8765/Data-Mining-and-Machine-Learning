import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
reg=linear_model.LinearRegression()

df= pd.read_csv("../tips.csv")

df['sex']=df['sex'].map({'Female':0,'Male':1})
df['smoker']=df['smoker'].map({'Yes':1,'No':0})
df['day']=df['day'].map({'Sun':3,'Thur':0,'Fri':1,'Sat':2})
df['time']=df['time'].map({'Lunch':0,'Dinner':1})
df['size']=df['size'].astype(int)
df['total_bill']=df['total_bill'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(df[['total_bill','sex','smoker','day','time','size']], df['tip'].astype(float), train_size=0.8)


reg.fit(np.array(X_train),np.array(y_train))
# print("The coeffecients and intercept of multivariate linear regression: ")
# print(reg.coef_,",",reg.intercept_,"\n")
print("Coefficient for total bill:",reg.coef_[0])
print("Coefficient for sex:",reg.coef_[1])
print("Coefficient for smoker:",reg.coef_[2])
print("Coefficient for day:",reg.coef_[3])
print("Coefficient for time:",reg.coef_[4])
print("Coefficient for size:",reg.coef_[5])
print("Intercept:", reg.intercept_)
r2_lin= r2_score(np.array(y_test),reg.predict(np.array(X_test)))
print("R-squared score of Multivariate Linear Regression:",r2_lin)
