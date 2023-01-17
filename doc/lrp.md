# Layer-Wise Relevance Propagation
+ use network weights + activations and propagate them back through the network
+ retrieve which input neurons contributed "how much" to the output
+ LRP as Deep Taylor Decomposition
  + Taylor series: $\sum^{\inf}_{n=0}{\frac{f^{(n)}(a)}{n!}(x-a)^n}$ 
  + relevance score $R_k$ is expressed as function of the lower-level activations $(a_j)_j)$ denoted by vector $a$
  + $R_k(a) = R_k(\widetilde{a}) + \sum_{0,j}{a_j - \widetilde{a_j}} * [\nabla R_k(\widetilde{a})]_j + ...$
+ use LRP-0 for upper layers, LRP-$\epsilon$ for middle layers, LRP-$\gamma$ for lower layers

## Rules
+ simple/LRP-0:
  + $R^{(l)}_i = \sum_{j}\frac{z_{ij}}{\sum_{i'}z_{i'j}}$ with $z_ij = x^{l}_i w^{(l,l+1)}_{ij}$
  + $i$ indexes a neuron at layer $l$
+ $\epsilon$-rule/LRP-$\epsilon$
  + $R_j = \sum{\frac{a_j w_{jk}}{\epsilon + \sum_{0,j}{a_j w_{jk}}}} R_k$
  + $\epsilon$ should absorb relevance if the activation of neuron $k$ are weak/contradictory
+ $\gamma$-rule/LRP-$\gamma$
  + $R_j = \sum_{k}{\frac{a_j * (w_{jk} + \gamma w^{+}_{jk})}{\sum_{0,j}{a_j * (w_{jk} + \gamma w^{+}_{jk})}}}$
  + $\gamma$ controls positive contributions

## Implementation
+ express the backward pass as the gradient
  + $c_j = [\nabla(\sum_{k}{z_k(a) * s_k}]_j$

```
def relprop(a, layer, R):
  z = epsilon + rho(layer).forward(a)
  s = R / (z+1e-9)
  (z * s.data).sum().backward()
  c = a.grad
  R = a * c
```

+ Sources:
  + http://iphome.hhi.de/samek/pdf/MonXAI19.pdf
  + http://iphome.hhi.de/samek/pdf/BinICISA16.pdf
