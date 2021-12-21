# mean shift clustering
import matplotlib.style as style
from sklearn_extra.cluster import KMedoids
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth


df = pd.read_csv(r'diabetes.csv')

features = df.columns
cols = (df[features] == 0).sum()
# print(cols)


# handling all the inconsistencies in the dataset
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


# mean-shift clustering
df2 = df.copy()
# print(df.pop('Outcome'))

bandwidth = estimate_bandwidth(df, quantile=0.2, n_samples=768)

clustering = MeanShift(bandwidth=bandwidth).fit(df)
print('number of clusters detected: ', len(np.unique(clustering.labels_)))
# print('cluster centers: ', clustering.cluster_centers_)

df1 = pd.DataFrame([i for i in clustering.cluster_centers_], index=[
                   np.unique(clustering.labels_)], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
print(df1)
# print(df)


# for measuring the accuracy of the clustering
l = []
l.extend(df2['Outcome'])

cnt = 0
for i, j in zip(clustering.labels_, df2['Outcome']):
    if i == j:
        cnt += 1

print('accuracy: ', cnt/768 * 100)
