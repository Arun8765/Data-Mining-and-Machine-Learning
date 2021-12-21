import pandas as pd
import numpy as np
import mord as m
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

c= m.OrdinalRidge()
df= pd.read_csv("tips.csv")

df['sex']=df['sex'].map({'Female':0,'Male':1})
df['smoker']=df['smoker'].map({'Yes':1,'No':0})
df['day']=df['day'].map({'Sun':3,'Thur':0,'Fri':1,'Sat':2})
df['time']=df['time'].map({'Lunch':0,'Dinner':1})
df['size']=df['size'].astype(int)
df['total_bill']=df['total_bill'].astype(float)
df['tip']= df['tip'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(df[['total_bill','sex','smoker','day','time','size']], df['tip'], train_size=0.8)


c.fit(np.array(X_train),np.array(y_train))
r2_ord = r2_score(np.array(y_test),c.predict(np.array(X_test)))
print("R-squared measure of Ordinal Regression:",r2_ord)