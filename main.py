from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale data
X = StandardScaler().fit(X).transform(X)

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Compute the probability function and perform classification
for i in range(len(X_test)):
    r1 = X_test[i]
    r12 = np.sum((X_train - r1) ** 2, axis=1)
