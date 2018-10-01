# iris-prediction-knn
__Predicting the species of iris using K-Nearest Neighbor.__

This machine learning model focuses on classifying an unknown iris into three species(setosa, versicolor and virginica) considering the parameters such as sepal length, sepal width, petal length and petal width. The confidence(score) for the model is 97.36%

Required dependencies:
* Bullet list
numpy - http://www.numpy.org/
* Bullet list
pandas - https://pandas.pydata.org/
* Bullet list
sci-kit learn - http://scikit-learn.org/stable/
* Bullet list
mglearn - https://pypi.org/project/mglearn/

Above dependencies can be installed using pip command in the python shell.

Information about the dataset can be found in dataset_info.py file in the repository.

Refer to the image iris_petal_sepal.png

Algorithm used: K-Nearest Neighbor Classifier
KNN finds the point in the training set that is closest to the new point. Then it assigns the label of this training point to the new data point. The k in k-nearest neighbors signifies that instead of using only the closest neighbor to the new data point, we can consider any fixed number k of neighbors in the training set.





