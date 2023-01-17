# Graph layer-wise relevance propagation
+ LRP as postprocessind step for the trained ML model
+ Molecular network $G = (V, E, A)$ ,V/E: Vertices/Edges, A: adjacency matrix
+ Calculate the relevances to quantify a relevance score for each gene g:
  + $\forall x : f(x) = \sum_{g}{R_g(x)}$
+ convolution on graphs:
  + $L = D - A$ can be diagonalized s.t. $L = U \Lambda U^T$ with $\Lambda = diag([\lambda_1,...,\lambda_m])$
  + $U$ and $U^T$ define the fourier/inv. fourier transform
  + a convolution on a graph can be viewed as a filtering operation, can be calculated recursively (chebyshe expansion)
  + Chebyshev expansion: $T_k(x) = 2xT_{k-1}(x)-T_{k-2}(x)$ with $T_0=1$ and $T_1=x$
  + neighborhood determined by shortest path
  + filtering operation: $y=h_{\Theta}(\Lambda)x = \sum_{k=0}^{K-1}{\Theta_kT_k(L)x = [\bar x_0,...,\bar x_{K-1}]\Theta}$
+ lrp is based on deep taylor decomposition
  + function $f(x)$ can be decomposed with a root point $x^*$ s.t. $f(x^*)=0$
  + first order Taylor expansion of $f(x)$: $f(x) = f(x^*) + \sum_{g=1}^{m}{R_g(x) + \epsilon} = 0 + \sum_{g=1}^{m}R_g(x) + \epsilon$
  + use LRP-$\epsilon$ with $\epsilon=1^{-10}$, $z^+$ rule coming from deep taylor decomposition
  + rewrite the filtering rule to propagate relevance $y=\sum_{k=0}^{K-1}{\Theta_k T_k (L)x = [\bar L_0,...,\bar L_{K-1}]\Theta_x = W_x}$, weight matrix $W\in R^{mxm}$ with $W=[\bar L_0,...,\bar L_{K-1}]\Theta$
    + $\bar L_k = 2L\bar L_{k-1} - \bar L_{k-2}$, $\bar L_0 = I$ and $\bar L_1 =L$ (chebyshev polynomials)
  + $y_j = \hat{W_j} \times \hat{x} \in R^m$ compute the $j^{th}$ channel of the output feature map
  + overall: $\R_{\hat x} = \sum_{j=1}^{F_{out}}{R_{\hat x}^j \in R_+^{m*F_{in}}}$

## GLRP on gene expression data
+ data was not standardized, trained directly on quantile normalized data

## GLRP on Chebnet revisited
+ Basically the same, but re-parameterize $\Theta$ as shown in gcn.md
