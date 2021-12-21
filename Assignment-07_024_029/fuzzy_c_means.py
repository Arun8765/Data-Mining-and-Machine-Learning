import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fcmeans import FCM
from tabulate import tabulate
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r"diabetes.csv")

# Cleaning the dataset by replacing 0 by median
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df['Glucose'].fillna(df['Glucose'].median(), inplace=True)

df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)

df['BMI'].fillna(df['BMI'].median(), inplace=True)

# Grouping insulin base on Glucose level and replacing 0 by the mean of the medians of the groups
by_Glucose_Age_Insulin_Grp = df.groupby(['Glucose'])


def fill_Insulin(series):
    return series.fillna(series.median())


df['Insulin'] = by_Glucose_Age_Insulin_Grp['Insulin'].transform(fill_Insulin)
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())

# Grouping Skin thickness based on BMI and replacing 0 by the mean of the medians of the groups
by_BMI_Insulin = df.groupby(['BMI'])


def fill_Skinthickness(series):
    return series.fillna(series.mean())


df['SkinThickness'] = by_BMI_Insulin['SkinThickness'].transform(fill_Skinthickness)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)


# Plotting the dataframe using PCA method
def plot_df(df):
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    x = df.loc[:, features].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    y = df.loc[:, ['Outcome']].values

    # print(x, y)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=[
        'principal component 1', 'principal component 2'])

    # print(principalDf)

    finalDf = pd.concat([principalDf, df[['Outcome']]], axis=1)

    print(finalDf)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['Outcome'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()

    plt.show()


plot_df(df)

fcm = FCM(n_clusters=2)
fcm.fit(df)

fcm_centers = fcm.centers
fcm_labels = fcm.predict(df)

print("The Coordinates of the cluster centers are:\n", fcm_centers, "\n\n\n================================\n\n")

tp = 0
tn = 0
fp = 0
fn = 0
for i, j in zip(fcm_labels, df['Outcome']):
    if i == j:
        if i == 0:
            tn += 1
        else:
            tp += 1
    else:
        if j == 0:
            fp += 1
        else:
            fn += 1

results = [["Actual Positive:", tp, fn],
           ["Actual Negative:", fp, tn]]
results = tabulate(results, headers=["Confusion Matrix", "Predicted Positive", "Predicted Negative"])

print(results)
print("\n\n\n=================================\n\n")
print('Accuracy:', (tp + tn) / (tp + tn + fp + fn) * 100, "%")
print('Precision:', tp / (tp + fp) * 100, "%")
print('Recall:', tp / (tp + fn) * 100, "%")
print('F-measure:', tp / (tp + (fp + fn) / 2))

# Plotting the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(df['Outcome'], fcm_labels)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Fuzzy c-means')
display.plot()
plt.show()
print("\nROC Area Under Curve Score:", roc_auc_score(df['Outcome'], fcm_labels))
