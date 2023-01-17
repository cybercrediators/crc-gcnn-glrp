import tensorflow as tf
import numpy as np
from glrp.base.base_model import BaseModel
from glrp.data.data_model import DataModel
from spektral.layers import ChebConv, GlobalMaxPool, GlobalSumPool, GlobalAvgPool
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from spektral.data import MixedLoader

class MnistModel(BaseModel):
    '''testing spektral model with the spektral example model for gcn with chebconv'''
    def __init__(self, config, data):
        super(MnistModel, self).__init__(config)
        self.data = data
        self.a_dtype = self.data.a.dtype
        self.n_nodes = self.data.n_nodes
        self.n_node_features = self.data.n_node_features
        self.loss_fn = SparseCategoricalCrossentropy()
        self.optimizer = Adam()
        self.model = None

    class Net(Model):
        def __init__(self, config, **kwargs):
            super().__init__(**kwargs)
            channels = config["F"]
            out_pols = config["K"]
            reg = config["regularization"]
            self.cc_1 = ChebConv(
                    channels=channels[0],
                    K=out_pols[0],
                    activation='relu',
                    kernel_regularizer=l2(reg),
                    use_bias=True
            )

            # chebconv2
            self.cc_2 = ChebConv(
                    channels=channels[1],
                    K=out_pols[1],
                    activation='relu',
                    kernel_regularizer=l2(reg),
                    use_bias=True
            )
            # flatten
            self.f_1 = GlobalSumPool()
            # fc1
            fc_size = config["M"]
            self.fc_1 = Dense(fc_size[0], activation='relu')
            # output (softmax)
            self.out = Dense(fc_size[1], activation='softmax')

        def call(self, inputs):
            x_in, a_in = inputs
            x = self.cc_1([x_in, a_in])
            x = self.cc_2([x, a_in])
            out = self.f_1(x)
            out = self.fc_1(out)
            out = self.out(out)

            return out

        def add_dropout(self, layer, dropout):
            if (dropout) > 0.0:
                return Dropout(dropout)(layer)
            return layer

    def build_model(self):
        self.model = self.Net(self.config)
        return self.model

    def required_params(self):
        """return the required parameters needed from a config file"""
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, loader):
        step = 0
        results = []
        for batch in loader:
            step += 1
            inputs, target = batch
            preds = self.model(inputs, training=False)
            loss = self.loss_fn(target, preds)
            acc = tf.reduce_mean(sparse_categorical_accuracy(target, preds))
            results.append((loss, acc, len(target)))
            if step == loader.steps_per_epoch:
                results = np.array(results)
                return np.average(results[:, :-1], 0, weights=results[:, -1])

    @tf.function
    def train_on_batch(self, inputs, target):
        with tf.GradientTape() as tape:
            pred = self.model(inputs, training=True)
            loss = self.loss_fn(target, pred) + sum(self.model.losses)
            acc = tf.reduce_mean(sparse_categorical_accuracy(target, pred))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, acc

    def train(self, train_loader, test_loader, val_loader):
        # TODO: make configurable
        best_val_loss = 99999
        patience = 10
        cur_patients = patience
        step = 0

        train_results = []
        for batch in train_loader:
            step += 1
            
            inputs, target = batch
            loss, acc = self.train_on_batch(inputs, target)
            print(self.model.summary())
            train_results.append((loss, acc, len(target)))
            #print(train_loader.steps_per_epoch)

            if step == train_loader.steps_per_epoch:
                val_results = self.evaluate(val_loader)
                if val_results[0] < best_val_loss:
                    best_val_loss = val_results[0]
                    cur_patients = patience
                    test_results = self.evaluate(test_loader)
                else:
                    cur_patients -= 1
                    if cur_patients == 0:
                        print("Early stopping")
                        break

                train_results = np.array(train_results)
                train_results = np.average(train_results[:, :-1], 0, weights=train_results[:, -1])

                print(
                    "Train loss: {:.4f}, acc: {:.4f} | "
                    "Valid loss: {:.4f}, acc: {:.4f} | ".format(
                        *train_results, *val_results
                    )
                )

                # Reset epoch
                train_results = []
                step = 0
