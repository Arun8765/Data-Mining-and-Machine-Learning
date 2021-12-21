# Support Vector Machines

import numpy as np
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, datasets

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    np.random.seed(1234)
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y


# Creating 2 Clusters with a total of 100 points
(X, y) = createClusteredData(100, 2)

# Plot to display the distribution of Income Vs Age
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.title('Income Vs Age')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()

# Scaling the Cluster
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X = scaling.transform(X)

# Display the Scaled Cluster
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
plt.title('Scaled income vs age')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()


# Create the SVM model
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)

# Prediction of the Cluster of the given points
def plotPredictions(clf):
    # Create a dense grid of points to sample
    xx, yy = np.meshgrid(np.arange(-1, 1, .001),
                         np.arange(-1, 1, .001))

    # Convert to Numpy arrays
    npx = xx.ravel()
    npy = yy.ravel()

    # Convert to a list of 2D (income, age) points
    samplePoints = np.c_[npx, npy]

    # Generate predicted labels (cluster numbers) for each point
    Z = clf.predict(samplePoints)

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)  # Reshape results to match xx dimension
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)  # Draw the contour
    plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))  # Draw the points
    plt.title('SVM output of income vs age')
    plt.xlabel('Income')
    plt.ylabel('Age')
    plt.show()


plotPredictions(svc)

print("A person with income 200000 and age 40 belongs to class:",svc.predict(scaling.transform([[200000, 40]])))

print("A person with income 50000 and age 65 belongs to class:",svc.predict(scaling.transform([[50000, 65]])))