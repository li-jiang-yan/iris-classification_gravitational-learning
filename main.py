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

# Compute the mass factor for the training data
n = np.bincount(y_train)
m = np.array([1/n[i] for i in y_train])

# Compute the probability function and perform classification
for i in range(len(X_test)):
    r1 = X_test[i]
    r12 = np.sum((X_train - r1) ** 2, axis=1)
    v = m / r12
    p = np.array([np.sum(np.where(y_train == target, v, 0)) for target in set(y_train)])
    P = p / np.sum(p)
