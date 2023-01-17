import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

G = nx.Graph(name='G')

for i in range(6):
    G.add_node(i, name=i)

edges = [(0,1), (0,2), (1,2), (0,3), (3,4), (3,5), (4,5)]
G.add_edges_from(edges)

print("Graph info:\n", nx.info(G))

print("\nGraph Nodes: ", G.nodes.data())

#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()

A = np.array(nx.attr_matrix(G, node_attr='name')[0])
X = np.array(nx.attr_matrix(G, node_attr='name')[1])
X = np.expand_dims(X, axis=1)

print('Shape of A: ', A.shape)
print('\nShape of X: ', X.shape)
print('\nAdjacency Matrix (A):\n', A)
print('\nNode Features Matrix (X):\n', X)

AX = np.dot(A, X)
#print("SUM of neighor node features ", AX)

# self-loops + normalization

G_self_loops = G.copy()
self_loops = []
for i in range(G.number_of_nodes()):
    self_loops.append((i, i))

G_self_loops.add_edges_from(self_loops)

A_hat = np.array(nx.attr_matrix(G_self_loops, node_attr='name')[0])
AX = np.dot(A_hat, X)
#print("AX: \n", AX)

# normalize
Deg_Mat = G_self_loops.degree()
D = np.diag([deg for (n, deg) in list(Deg_Mat)])
D_inv = np.linalg.inv(D)

DAX = np.dot(D_inv, AX)
print("DAX: \n", DAX)

D_half_norm = fractional_matrix_power(D, -0.5)
DADX = D_half_norm.dot(A_hat).dot(D_half_norm).dot(X)
print("DADX: \n", DADX)

# add weights
np.random.seed(123456)
n_h = 4
n_y = 2
W0 = np.random.randn(X.shape[1], n_h) * 0.01
W1 = np.random.randn(n_h, n_y) * 0.01

def relu(x):
    return np.maximum(0,x)

def gcn(A, H, W):
    I = np.identity(A.shape[0])
    # self loop
    A_hat = A + I
    # degree matr/normalize
    D = np.diag(np.sum(A_hat, axis=0))
    D_half_norm = fractional_matrix_power(D, -0.5)
    eq = D_half_norm.dot(A_hat).dot(D_half_norm).dot(H).dot(W)
    return relu(eq)

H1 = gcn(A, X, W0)
H2 = gcn(A, H1, W1)
print("Feature repr.: \n", H2)

def plot_features(H2):
    x = H2[:,0]
    y = H2[:,1]
    size = 1000
    plt.scatter(x, y, size)
    plt.xlim([np.min(x)*0.9, np.max(x) * 1.1])
    plt.ylim([-1, 1])
    plt.xlabel("dim0")
    plt.ylabel("dim1")
    plt.title("repr")

    for i, row in enumerate(H2):
        str = "{}".format(i)
        plt.annotate(str, (row[0], row[1]), fontsize=18, fontweight='bold')
    plt.show()

plot_features(H2)
