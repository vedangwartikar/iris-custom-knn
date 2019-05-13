import numpy as np
import pandas as pd
from scipy.spatial import distance

class CustomKNN:
	def fit(self, train_data, train_target):
		self.train_data = train_data
		self.train_target = train_target

	def predict(self, test_data):
		predictions =[]
		for row in test_data:
			label = self.closest(row)
			predictions.append(label)
			return predictions

	def closest(self, row):
		closest_distance = distance.euclidean(row, self.train_data[0])
		closest_index = 0
		for i in range(1, len(self.train_data)):
			dist = distance.euclid(row, self.train_data[i])
			if dist < closest_distance:
				closest_distance = dist
				closest_index = i
		return self.train_target[closest_index]

from sklearn.datasets import load_iris
iris_dataset = load_iris()

#Splitting dataset into 75% train set and 25% test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

#Train, Test set sizes
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#Data visualization using pandas scatter_matrix
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

#Building K-nearest neighbor model
# from sklearn.neighbors import KNeighborsClassifier
knn = CustomKNN(n_neighbors = 1)

knn.fit(X_train, y_train)

#Custom iris length inputs
a = input('Enter sepal length(cm): ')
b = input('Enter sepal width(cm): ')
c = input('Enter petal length(cm): ')
d = input('Enter petal width(cm): ')
X_new = np.array([[a,b,c,d]])

#Actual prediction of knn model
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

#Accuracy of knn model
print("Test set score: {}".format(knn.score(X_test, y_test)))
