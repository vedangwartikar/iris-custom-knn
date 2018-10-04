import mglearn

#make_forge() dataset
#Prediction based on the nearest data point
mglearn.plots.plot_knn_classification(n_neighbors = 1)

#Prediction based on the most frequently occured class considering k-neighbors
mglearn.plots.plot_knn_classification(n_neighbors = 2)

mglearn.plots.plot_knn_classification(n_neighbors = 3)
