# kmedoids
import matplotlib.style as style
from sklearn_extra.cluster import KMedoids
import numpy as np
import pandas as pd


df = pd.read_csv(r'diabetes.csv')

features = df.columns
cols = (df[features] == 0).sum()
# print(cols)


df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)


# handle blood pressure, glucose and BMI values

df['Glucose'].fillna(df['Glucose'].median(), inplace=True)

df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)

df['BMI'].fillna(df['BMI'].median(), inplace=True)


# handle insulin based on glucose

by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])


def fill_Insulin(series):
    return series.fillna(series.median())


df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())


# skin_thickness wrt BMI

by_BMI_Insulin = df.groupby(['BMI'])


def fill_Skinthickness(series):
    return series.fillna(series.mean())


df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(
    fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)


kmedoids = KMedoids(n_clusters=2, random_state=0).fit(df)
print(kmedoids.labels_)
print('cluster centers: ', kmedoids.cluster_centers_)
# print(df)

l = []
l.extend(df['Outcome'])

cnt = 0
for i, j in zip(kmedoids.labels_, df['Outcome']):
    if i == j:
        cnt += 1

print('accuracy: ', cnt/768 * 100)
