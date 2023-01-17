# lib package

## Submodules

## lib.coarsening module


### lib.coarsening.coarsen(A, levels, self_connections=False)
Coarsen a graph, represented by its adjacency matrix A, at multiple
levels.


### lib.coarsening.compute_perm(parents)
Return a list of indices to reorder the adjacency and data matrices so
that the union of two neighbors from layer to layer forms a binary tree.


### lib.coarsening.metis(W, levels, rid=None)
Coarsen a graph multiple times using the METIS algorithm.

INPUT
W: symmetric sparse weight (adjacency) matrix
levels: the number of coarsened graphs

OUTPUT
graph[0]: original graph of size N_1
graph[2]: coarser graph of size N_2 < N_1
graph[levels]: coarsest graph of Size N_levels < … < N_2 < N_1
parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}

> which indicate the parents in the coarser graph[i+1]

nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

NOTE
if “graph” is a list of length k, then “parents” will be a list of length k-1


### lib.coarsening.metis_one_level(rr, cc, vv, rid, weights)

### lib.coarsening.perm_adjacency(A, indices)
Permute adjacency matrix, i.e. exchange node ids,
so that binary unions form the clustering tree.


### lib.coarsening.perm_data(x, indices)
Permute data matrix, i.e. exchange node ids,
so that binary unions form the clustering tree.


### lib.coarsening.perm_data_back(x, indices, feature_num)
Permute data from the binary tree order to the original order of data.
:param x: data with rows to permute
:param indices: permutation
:param feature_num: number of features in data in total
:return: x with features in original order


### lib.coarsening.perm_matrix_back(mat, indices, feature_num)
Permute adjacency matrix from the binary tree order to the original order of features.
:param mat: adjacency mat
:param indices:
:param feature_num:
:return: mat with rows and columns in original order

## lib.data_handling module


### _class_ lib.data_handling.DataPreprocessor(path_to_feature_values, path_to_labels, path_to_feature_graph=None)
Bases: `object`

Performs converting of the dataset to NumPy format for feeding into the machine learning algorithm.


#### _static_ generate_Q(transition_matrix, q_power)
Generate Q^{q_power} for n variables.


#### get_adj_feature_graph_as_np_array()

#### get_data_frame_for_mRMR_method()
Returns the data frame of feature values and labels.
The first raw: “class” “feat1” “feat2”…
The second raw: “label” “value of feat1” values of feat2”


#### get_feature_values_as_np_array(columns=None)
Can extract values from fixed columns.


#### get_labels_as_np_array()

#### _static_ normalize_data(X, eps=5e-07)
Z score calculation, each column of X is a feature, each row is a sample.
Returns three variables:
X normalized,
array of mean values
array of std values.


#### _static_ normalize_data_0_1(X, eps=5e-07)
Normalize in 0_1 interval, each column of X is a feature, each row is a sample.
Returns three variables:
X normalized,
array of min
array of max values.


#### _static_ scale_data(X_val, column_mean, column_std, non_zero_ind)
Scaling the validation data according to the mean and std of features in the training data.
Returns:
X_val scaled.


#### _static_ scale_data_0_1(X_val, column_min, column_max, non_zero_ind)
Scaling the validation data according to the min and max of features in the training data.
Returns:
X_val scaled.

## lib.graph module


### lib.graph.adjacency(dist, idx)
Return the adjacency matrix of a kNN graph.


### lib.graph.chebyshev(L, X, K)
Return T_k X where T_k are the Chebyshev polynomials of order up to K.
Complexity is O(KMN).


### lib.graph.distance_lshforest(z, k=4, metric='cosine')
Return an approximation of the k-nearest cosine distances.


### lib.graph.distance_scipy_spatial(z, k=4, metric='euclidean')
Compute exact pairwise distances.


### lib.graph.distance_sklearn_metrics(z, k=4, metric='euclidean')
Compute exact pairwise distances.


### lib.graph.fourier(L, algo='eigh', k=1)
Return the Fourier basis, i.e. the EVD of the Laplacian.


### lib.graph.grid(m, dtype=<class 'numpy.float32'>)
Return the embedding of a grid graph.


### lib.graph.lanczos(L, X, K)
Given the graph Laplacian and a data matrix, return a data matrix which can
be multiplied by the filter coefficients to filter X using the Lanczos
polynomial approximation.


### lib.graph.laplacian(W, normalized=True)
Return the Laplacian of the weigth matrix.


### lib.graph.lmax(L, normalized=True)
Upper-bound on the spectrum.


### lib.graph.plot_spectrum(L, algo='eig')
Plot the spectrum of a list of multi-scale Laplacians L.


### lib.graph.replace_random_edges(A, noise_level)
Replace randomly chosen edges by random edges.


### lib.graph.rescale_L(L, lmax=2)
Rescale the Laplacian eigenvalues in [-1,1].

## lib.visualize_mnist module


### lib.visualize_mnist.get_heatmap(digit)
Computes the heatmap of the out of the digit.


### lib.visualize_mnist.plot_numbers(data_to_test, rel, labels_data_to_test, labels_by_network, results_dir)
Plots the numbers and the relevance heatmap of it.

## Module contents
