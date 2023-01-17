# Graph Convolutional Networks
+ Convolutional Network operating on a graph $G = (V, E)$
  + feature matrix $N * F^0$
  + adjacency matrix $A$ of $G$
+ learn features of neighboring nodes
+ Hidden Layer $H^i = f(H^{(i-1)}, A)$
+ forward propagation $f(H^i, A) = \sigma(AH^iW^i)$
+ add self loops:
  + add the identity matrix $I$ to the adjacency matrix $A$
  + $\hat{A} = A + I$
+ normalize feature representations
  + calculate the degree matrix $D = deg(v_i)$ if i == j else 0
  + use the inverse degree matrix: $f(X, A) = D^{-1}AX$
  + symmetric normalization: $f(X, A) = D^{-1/2}AD^{-1/2}$
+ this results in the spectral propagation rule: $f(X,A) = \sigma(D^{-1/2}\hat{A}D^{-1/2}XW)$

# GCN with Fast Localized Spectral Filtering
+ filtering rule: $y = \sum^{K}_{k = 0}{w_k T_k(\hat{L}) f_{\Theta}(X)}$
+ $f_{\Theta}$: MLP of X
+ $\hat{L}$: scaled Laplacian

# Chebnet revisited
+ New filtering rule: $y = 2 / K + 1 \sum^{K}_{k = 0}{\sum^{K}_{j = 0}{\gamma_j T_k(x_j) T_k(\hat{L})f_{\Theta}(X)}}$
  + with $x_j = cos((j + 1/2) \pi / (K + 1))$
  + $\gamma_j$: learnable parameters
+ The new filtering rule reparameterizes $w_k$, using Chebyshev interpolation
