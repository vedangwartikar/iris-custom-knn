# iris-custom-knn
__Predicting the species of iris using custom built K-Nearest Neighbor.__
- - - -
This machine learning model focuses on classifying an unknown iris into three species(setosa, versicolor and virginica) considering the parameters such as sepal length, sepal width, petal length and petal width. The confidence(score) for the model is 94 - 97.36%(The score may vary due to the model getting trained differently with each iteration).

Required dependencies:
* [numpy](http://www.numpy.org/) - Numerical computation
* [pandas](https://pandas.pydata.org/) - Data Analysis and manipulation
* [sci-kit learn](http://scikit-learn.org/stable/) - Scientific computation
* [mglearn](https://pypi.org/project/mglearn/) - Data visualization

Above dependencies can be installed using pip command in the python shell.

Information about the dataset can be found in dataset_info.py file in the repository.

Refer to the image iris_petal_sepal.png

Algorithm used: Custom built K-Nearest Neighbor Classifier
> KNN finds the point in the training set that is closest to the new point. Then it assigns the label of this training point to the new data point. The k in k-nearest neighbors signifies that instead of using only the closest neighbor to the new data point, we can consider any fixed number k of neighbors in the training set.
