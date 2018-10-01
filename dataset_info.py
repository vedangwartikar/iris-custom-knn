from sklearn.datasets import load_iris
iris_dataset = load_iris()

#Keys of iris_dataset
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()) + "\n")

#Information regarding Target in dataset
print("Target names: {}".format(iris_dataset['target_names']) + "\n")

print("Type of target: {}".format(type(iris_dataset['target'])) + "\n")

print("Shape of target: {}".format(iris_dataset['target'].shape) + "\n")

print("Target:\n{}".format(iris_dataset['target']))

#Features in dataset
print("Feature names: {}".format(iris_dataset['feature_names']) + "\n")

#Actual data information
print("Type of data: {}".format(type(iris_dataset['data'])) + "\n")

print("Shape of data: {}".format(iris_dataset['data'].shape) + "\n")

print("First 5 colomns of dataset:\n {}".format(iris_dataset['data'][:5]))

#Generelized information regarding iris_dataset
print(iris_dataset['DESCR'][:1180] + "\n")
