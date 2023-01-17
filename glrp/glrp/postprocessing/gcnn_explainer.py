import matplotlib.pyplot as plt
from glrp.base.base_explainer import BaseExplainer
from spektral.models import GNNExplainer
from spektral.utils.convolution import chebyshev_polynomial, chebyshev_filter, chebyshev_reparam_weights
import scipy
import time
import math

from enum import Enum
from spektral.data.loaders import MixedLoader
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from lib import coarsening
import numpy as np
import tensorflow as tf


class LRPRule(Enum):
    DEFAULT = 0
    GAMMA = 1
    EPSILON = 2

class GCNNExplainer(BaseExplainer):
    """
    The GCNExplainer from the paper:
    > [ Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer](https://pubmed.ncbi.nlm.nih.gov/33706810/)
    > Hryhorii Chereda, Annalen Bleckmann, Kerstin Menck, Júlia Perera-Bel, Philip Stegmaier, Florian Auer, Frank Kramer, Andreas Leha, Tim Beißbarth

    The model can be used to explain the predictions for a graph of a graph convolutional neural network using
    the proposed GLRP approach to work with the spektral framework. Original implementation: https://gitlab.gwdg.de/UKEBpublic/graph-lrp.

    https://mdpi-res.com/d_attachment/sensors/sensors-21-04536/article_deploy/sensors-21-04536-v2.pdf?version=1625210886
    """
    def __init__(self, conf, model, data, mode='uniform'):
        super(GCNNExplainer, self).__init__(conf, model, data)
        self.model = model
        self.conf = conf
        self.test_patient_ids = data.PI_test
        self.orig_data = data
        self.data = data.test_data
        self.mode = mode
        # choose different lrp rules for different layers or one rule for
        # every layer: http://iphome.hhi.de/samek/pdf/MonXAI19.pdf
        self.epsilon = conf["epsilon"]
        self.gamma = conf["gamma"]

    def explain(self):
        #explainer = GCNExplainer(self.config, self.model, self.data.test_data, mode='uniform')
        relevances = self.explain_graph()
        return relevances

    def explain_graph(self):
        # read model
        #self.model.summary()

        # TODO: increase batch size??
        b = self.data.n_graphs

        # load data
        loader = MixedLoader(self.data, batch_size=b, shuffle=False, epochs=1)
        R = []
        # TODO: determine rules
        rule = LRPRule.EPSILON
        
        rev_layers = list(reversed(self.model.layers))
        layer_names = [x.name for x in rev_layers]
        trainable_layers = [x for x in layer_names if "dense" in x or "conv" in x]
        num_trainable_layers = len(trainable_layers)
        num_split = math.floor(num_trainable_layers / 2)
        rule_array = [LRPRule.EPSILON, LRPRule.GAMMA, LRPRule.GAMMA]
        
        # lrp-0 -> lrp-e -> lrp-y
        # !don't use lrp-0
        rules = {}
        count = 0
        i = 0
        for x in rev_layers:
            if x.name in trainable_layers:
                rules[x] = rule_array[i]
                if count != 0 and count % num_split == 0:
                    i += 1
                count += 1
            else:
                rules[x] = LRPRule.DEFAULT
            

        # TODO: partition layers for different rules
        #
        # IMPORTANT: Mind the first layer, because relevances should not depend
        # on the inputs to avoid biases!!!
        relevance_batches = []
        print("CALCULATE polys")
        polynomials = []
        for graph in self.orig_data.adj_coarsened[0:len(self.conf["F"])]:
            polys = chebyshev_polynomial(graph, int(self.conf["K"][0]) - 1)
            polynomials.append(polys)
        #polynomials = [chebyshev_polynomial(x, self.conf["K"] - 1) for x in self.orig_data.adj_coarsened]
        # TODO: will be adapted to gracul
        #polynomials = np.array(chebyshev_polynomial(self.data.a, 7))
        pools = self.conf["p"]
        print(polynomials)
        for batch in loader:
            inputs, target = batch
            inputs = (inputs[0], sp_matrix_to_sp_tensor(self.orig_data.adj_coarsened[0]), sp_matrix_to_sp_tensor(self.orig_data.adj_coarsened[1]))
            print(inputs)
            self.model.summary()
            preds = self.model(inputs, training=False)
            # 3 different rules, 3 input layers, 2 max pool layers
            #parts = math.ceil((len(rev_layers) - 5) / 3)
            #count = 0
            #for idx, layer in rev_layers.enumerate():
            print(preds)
            R = tf.one_hot(np.argmax(preds, axis=1), depth = 2)
            print(R)
            for i, layer in enumerate(rev_layers):
                rule = rules[layer]
                print(layer.name)
                print(rule)
                if "input" in layer.name:
                    continue

                #print("\n\n\n ++++ CUR RELEVANCES +++ \n")
                #print(R)
                #print("\n\n\n\n")
                # get activations (a) from previous layer
                # if the next one is one of the input layers
                # use the other one to get the inputs instead
                # of the adjacency matrix
                if "input" in rev_layers[i+1].name:
                    act_model = tf.keras.Model(self.model.input, rev_layers[i+2].output)
                else:
                    act_model = tf.keras.Model(self.model.input, rev_layers[i+1].output)
                # print(act_model.summary())
                activations = act_model(inputs, training=False)
                weights = layer.get_weights()
                print(activations.shape)
                if len(weights) > 0:
                    weights = weights[0]
                    if len(weights) >= 2:
                        bias = weights[1]
                #continue
                if "dense" in layer.name:
                    print("MACHE DENSE THINGS")
                    R = self.prop_dense(activations, weights, R, rule)
                    print("NACH DENSE THINGS")
                    print("\n\n\n")
                if "conv" in layer.name:
                    R = self.prop_chebconv(activations, weights, R, polynomials.pop(), rule, chebnet_type=self.conf["chebnet_type"])
                if "pool" in layer.name:
                    p = pools.pop()
                    R = self.prop_pool(activations, R, ksize=[1, p, 1, 1], strides=[1, p, 1, 1],type=self.conf["pool"])
                if "flatten" in layer.name:
                    print("MACHE FLATTEN THINGS")
                    R = self.prop_flatten(activations, R)
                    print("NACH FLATTEN THINGS")
                    print("\n\n\n")
            #return R
            R = coarsening.perm_data_back(R, self.orig_data.perm, self.orig_data.feature_graph.shape[0])
            print(R)
            relevance_batches.extend(R)

        return relevance_batches
        #return R
        # get layers
        # select correct propagation function
        # determine suiting propagation rules for each layer
        # propagate output back through the layers
        # postprocess data

    def subgraph_to_cx(self):
        """
        Convert the resulting subgraphs to the cx format
        to prepare it for submission to NDEx.
        :return:
            `net_cx`: cx graph
        """
        pass

    def rho(self, w, rule=LRPRule.EPSILON):
        if rule == LRPRule.GAMMA:
            return w + self.gamma * np.clip(w, 0.0, np.max(w))
        else:
            # lrp-0 -> epsilon=0
            return w

    def prop_dense(self, a, w, R, rule=LRPRule.EPSILON):
        """
        Propagate outputs through a dense layer.
        :params:
            `a`: vector of lower-layer activations
            `layer`: copy of the current layer
            `R`: relevances
        """
        #if (rule == LRPRule.DEFAULT):
        #    self.epsilon = 0.0
        # z: forward pass
        # s: R/z+1e-9
        # c: Backward pass
        # R: a * c
        rho_w = tf.maximum(0.0, w)
        rho_w = self.rho(rho_w, rule)
        z = self.epsilon
        if rule == LRPRule.DEFAULT:
            z = 0.0
        z += tf.matmul(a, rho_w)
        s = R / (z + 1e-9)
        c = tf.matmul(s, tf.transpose(rho_w))
        return c * a

    def prop_flatten(self, a, R):
        """
        Propagate outputs through a flatten layer.
        """
        return tf.reshape(R, tf.shape(a))

    def prop_chebconv(self, a, w, R, polynomials, rule=LRPRule.EPSILON, first_layer=False, chebnet_type="chebnet"):
        """
        Propagate outputs through a cheb conv layer.
        """
        # calculate polynomials
        # IMPORTANT: function calculates K+1 polynomials
        # TODO: get it to work with coarsening
        #print(polynomials)
        #print(polynomials.shape)
        polys = polynomials[0].reshape(polynomials[0].shape[0]*polynomials[0].shape[0],1)
        for p in polynomials[1:]:
            print(p.shape[0] * p.shape[0])
            polys = scipy.sparse.hstack([polys, p.reshape(p.shape[0]*p.shape[0], 1)])
        #polynomials = self.calc_Laplace_Polynom(self.data.a, 2)
        print(polys)
        print(polys.shape)
        # get input channels
        print(w.shape)
        print(a.shape)
        N, M, F_in = a.shape
        F_out = w.shape[-1]
        print(N, M, F_in, F_out)
        # get output channels

        # Reparametrize weights if chebnet2
        if chebnet_type == "chebnet2":
            new_w = np.zeros_like(w)
            for k in range(0, w.shape[0]):
                new_w[k] = chebyshev_reparam_weights(w.shape[0], w, k)
            w = new_w

        # calculate relevances for each filter R_x
        if first_layer:
            R_x = [np.zeros(shape=[N, M], dtype=np.float32)]
        else:
            R_x = [np.zeros(shape=[N, M * F_in], dtype=np.float32)]

        print("\n\n STARTE DURCH FILTER \n")
        start = time.time()
        for i in range(F_out):
            inner_s = time.time()
            print("BIN BEI FILTER " + str(i) + " mit Ges.Zeit: " + str(time.time() - start))
            if first_layer:
                rho_w = polys.dot(w[:, i])
            else:
                rho_w = polys.dot(w[:, :, i])
            print("NACH poly dot mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
            #print(rho_w)
            #print("Max", rho_w.shape)
            rho_w = np.maximum(0.0, rho_w)
            #print("rule", rho_w.shape)
            rho_w = self.rho(rho_w, rule)
            print("NACH rho/max mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
            #print("reshape1", rho_w.shape)
            if first_layer:
                rho_w = np.reshape(rho_w, [M, M])
            else:
                rho_w = np.reshape(rho_w, [M, M, F_in])
                print("NACH reshape1 mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
                #print("reshape2", rho_w.shape)
                rho_w = np.transpose(rho_w, axes=[0, 2, 1])
                print("NACH transpose mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
                #print("reshape3", rho_w.shape)
                rho_w = np.reshape(rho_w, [M * F_in, M])
                print("NACH reshape2 mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
                #print("reshape4", rho_w.shape)
            act = np.reshape(a, [N, F_in * M])
            print("NACH reshape3 mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
            #print(act.shape)
            # only positive values
            # forward pass
            z = self.epsilon + np.matmul(act, rho_w)
            print(z)
            print(z.shape)
            print("NACH z mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
            #print(z.shape)
            #print(R.shape)
            s = R[:, :, i] / (z + 1e-9)
            print("NACH s mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
            c = np.matmul(s, np.transpose(rho_w))
            print("NACH c mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
            R_j = act * c
            R_x += R_j
            print("NACH finale mit loopZeit: " + str(time.time() - inner_s) + " Ges.Zeit " + str(time.time() - start))
        if not first_layer:
            R_x = np.reshape(R_x, [N, M, F_in])
        print("NACH RELPROP CHEBCONV")
        return R_x

    def prop_glob_max_pool(self, a, R):
        """"""
        pass

    def prop_pool(self, a, R, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', type='max'):
        """
        Propagate outputs through a max pooling layer.
        """
        if "max" in type:
            z = tf.nn.max_pool(tf.expand_dims(a, 3), ksize, strides, padding) + self.epsilon
            s = tf.expand_dims(R, 3) / z
            c = tf.raw_ops.MaxPoolGradV2(orig_input=tf.expand_dims(a, 3), orig_output=z, grad=s, ksize=ksize, strides=strides, padding=padding)
        elif "avg" in type:
            z = tf.nn.avg_pool(tf.expand_dims(a, 3), ksize, strides, padding) + self.epsilon
            s = tf.expand_dims(R, 3) / z
            c = tf.raw_ops.AvgPoolGrad(orig_input=tf.expand_dims(a, 3), orig_output=z, grad=s, ksize=ksize, strides=strides, padding=padding)
        else:
            raise Exception("Error: Pooling type not found!")
        return tf.squeeze(c * tf.expand_dims(a, 3), [3])

    def assign_rules_to_layers(self):
        """
        Assign a suiting LRP-rule to each model layer.
        """
        pass
