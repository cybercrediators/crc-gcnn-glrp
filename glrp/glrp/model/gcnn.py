import tensorflow as tf
import numpy as np
import time
import copy
from glrp.base.base_model import BaseModel
#from spektral.layers import ChebConv, GlobalMaxPool, GlobalSumPool, GlobalAvgPool
from spektral.layers import ChebConv, ChebConv2, GlobalMaxPool, GlobalSumPool, GlobalAvgPool
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, MaxPooling1D, AveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, binary_accuracy
from spektral.data import MixedLoader

from spektral.utils import convolution
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from scipy.sparse import csr_matrix

from glrp.model.callbacks.auc import roc_callback


class GCNNModel(BaseModel):
    """
    The GCNN model class. 
    """
    
    def __init__(self, config, data):
        super(GCNNModel, self).__init__(config)
        #self.build_model()
        self.data = data
        self.a_dtype = self.data.dataset.a.dtype
        self.n_nodes = self.data.dataset.n_nodes
        self.n_node_features = self.data.dataset.n_node_features

        self.loss_fn = SparseCategoricalCrossentropy()
        self.optimizer = Adam(self.config["learning_rate"])
        self.model = None
        self.shape_x = data.adj_coarsened[0].shape[0]
        self.shape_a_l2 = data.adj_coarsened[1].shape[0]
        print(self.shape_x, self.shape_a_l2)

        # use tf callbacks in the custom training loop
        self._callbacks = []
        self.callbacks = None
        self.logs = {}
        self.patience = config["patience"]
        # attention: predictions will be max_reduce before
        self.train_metrics = {
            "Accuracy": tf.keras.metrics.BinaryAccuracy(),
            "Precision": tf.keras.metrics.Precision(),
            "AUC": tf.keras.metrics.AUC(num_thresholds=100),
            "Recall": tf.keras.metrics.Recall()
        }
        self.test_metrics = {
            "Accuracy": tf.keras.metrics.BinaryAccuracy(),
            "Precision": tf.keras.metrics.Precision(),
            "AUC": tf.keras.metrics.AUC(num_thresholds=100),
            "Recall": tf.keras.metrics.Recall()
        }

    class Net(Model):
        """
        Subclassed gcnn model from a given config file.
        """
        def __init__(self, config, **kwargs):
            super().__init__(**kwargs)
            channels = config["F"]
            out_pols = config["K"]
            pool_sizes = config["p"]
            self.shape_x = config["shape_x"]
            self.shape_a_l2 = config["shape_a_l2"]
            self.pool_type = config["pool"]
            self.chebnet_type = config["chebnet_type"]
            if config["chebnet_type"] == "chebnet2":
                self.chebnet = ChebConv2
            else:
                self.chebnet = ChebConv
            reg = config["regularization"]
            self.test_activations = []
            self.cc_1 = self.chebnet(
                    channels=channels[0],
                    K=out_pols[0],
                    activation='relu',
                    kernel_regularizer=l2(reg),
                    use_bias=True
            )
            # maxpool
            self.max_p_1 = MaxPooling1D(pool_size=pool_sizes[0], strides=pool_sizes[0])
            self.max_p_2 = MaxPooling1D(pool_size=pool_sizes[1], strides=pool_sizes[1])
            # avgpool
            self.avg_p_1 = AveragePooling1D(pool_size=pool_sizes[0], strides=pool_sizes[0])
            self.avg_p_2 = AveragePooling1D(pool_size=pool_sizes[1], strides=pool_sizes[1])
            # chebconv2
            self.cc_2 = self.chebnet(
                    channels=channels[1],
                    K=out_pols[1],
                    activation='relu',
                    kernel_regularizer=l2(reg),
                    use_bias=True
            )
            # flatten
            #self.f_1 = GlobalSumPool()
            self.f_1 = Flatten()
            # fc1
            fc_size = config["M"]
            self.fc_1 = Dense(fc_size[0], activation='relu')
            # output (softmax)
            self.fc_2 = Dense(fc_size[1], activation='relu')
            self.out = Dense(fc_size[2], activation='softmax')
            #self.out = Dense(fc_size[2], activation='sigmoid')
        def call(self, inputs):
            x_in, a_l1, a_l2 = inputs
            x = self.cc_1([x_in, a_l1])

            if "avg" in self.pool_type:
                x = self.avg_p_1(x)
            else:
                x = self.max_p_1(x)
            x = self.cc_2([x, a_l2])

            if "avg" in self.pool_type:
                x = self.avg_p_2(x)
            else:
                x = self.max_p_2(x)
            # x = self.cc_1([x_in, a_in])
            # x = self.cc_2([x, a_in])
            # TODO: pooling layers
            out = self.f_1(x)
            out = self.fc_1(out)
            out = self.fc_2(out)
            out = self.out(out)
            tf.summary.histogram('outputs', out)
            return out

        def model(self):
            x = Input(shape=[self.shape_x, 1])
            a_l1 = Input(shape=[self.shape_x])
            a_l2 = Input(shape=[self.shape_a_l2])
            return tf.keras.Model(inputs=[[x, a_l1, a_l2]], outputs=self.call([x, a_l1, a_l2]))

        def add_dropout(self, layer, dropout):
            """
            add a dropout layer to the current model.
            """
            if (dropout) > 0.0:
                return Dropout(dropout)(layer)
            return layer

    def build_model(self):
        """
        Return the current model as a keras Model.
        """
        conf = self.config.copy()
        conf["shape_x"] = self.shape_x
        conf["shape_a_l2"] = self.shape_a_l2
        self.model = self.Net(conf)
        return self.model.model()

    def evaluate(self, loader, test=False):
        """
        evaluate the current model with the given data from
        the spektral loader.
        """
        step = 0
        results = []
        for batch in loader:
            step += 1
            inputs, target = batch
            inputs = self.conv_inputs(inputs)
            preds = self.model(inputs, training=False)
            target = target.reshape((target.shape[0], 1))
            pred_cats = tf.argmax(preds, 1)
            if not test:
                for _, metric in self.test_metrics.items():
                    # tf.print("PREDICTION: ", pred)
                    # tf.print("ARGMAX: ", pred_cats)
                    # tf.print("TARGET: ", target)
                    metric.update_state(target, pred_cats)
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
        pred_cats = tf.argmax(pred, 1)
        for _, metric in self.train_metrics.items():
            # tf.print("PREDICTION: ", pred)
            # tf.print("ARGMAX: ", pred_cats)
            # tf.print("TARGET: ", target)
            metric.update_state(target, pred_cats)
        return loss, acc

    def conv_inputs(self, inp):
        """
        prepare inputs with coarsened adj matrix
        """
        a_l1 = sp_matrix_to_sp_tensor(self.data.adj_coarsened[0])
        a_l2 = sp_matrix_to_sp_tensor(self.data.adj_coarsened[1])
        #print(self.data.adj_coarsened[0].shape)
        #print(self.data.adj_coarsened[1].shape)
        inputs = (inp[0], a_l1, a_l2)
        return inputs

    def train(self, train_loader, test_loader, val_loader):
        """
        Train the current model on the given train, test
        and validation loader.
        """
        best_val_loss = 99999
        patience = self.patience
        cur_patients = patience
        step = 0
        epoch = 0
        self.add_callbacks()

        self.callbacks = tf.keras.callbacks.CallbackList(
            self._callbacks,
            add_history=True,
            model=self.model
        )
        
        self.callbacks.on_train_begin(logs=self.logs)

        train_results = []
        test_results = []
        self.callbacks.on_epoch_begin(epoch, logs=self.logs)
        for batch in train_loader:
            self.callbacks.on_batch_begin(step, logs=self.logs)
            step += 1

            inputs, target = batch
            inputs = self.conv_inputs(inputs)
            loss, acc = self.train_on_batch(inputs, target)
            train_results.append((loss, acc, len(target)))

            if step == train_loader.steps_per_epoch:
                epoch += 1
                self.callbacks.on_batch_end(step, logs=self.logs)
                self.callbacks.on_epoch_end(epoch, logs=self.logs)
                val_results = self.evaluate(val_loader)
                if val_results[0] < best_val_loss:
                    best_val_loss = val_results[0]
                    cur_patients = patience
                    test_results = self.evaluate(test_loader, True)
                else:
                    cur_patients -= 1
                    if cur_patients == 0:
                        print("Early stopping")
                        break

                train_results = np.array(train_results)
                train_results = np.average(train_results[:, :-1], 0, weights=train_results[:, -1])
                tf.summary.scalar('train accuracy', train_results[1], epoch)
                tf.summary.scalar('train loss', train_results[0], epoch)
                tf.summary.scalar('test accuracy', test_results[1], epoch)
                tf.summary.scalar('test loss', test_results[0], epoch)

                self.display_metrics(epoch)

                print(
                    "Train loss: {:.4f}, acc: {:.9f} | "
                    "Val loss: {:.4f}, acc: {:.9f} | "
                    "Test loss: {:.4f}, acc: {:.9f} | ".format(
                        *train_results, *val_results, *test_results
                    )
                )
        
                # Reset epoch
                for _, metric in self.train_metrics.items():
                    metric.reset_states()
                for _, metric in self.test_metrics.items():
                    metric.reset_states()
                train_results = []
                step = 0
                self.callbacks.on_epoch_begin(epoch, logs=self.logs)
        self.callbacks.on_train_end(logs=self.logs)
        # TODO: retrieve tf training history from callbacks
        return test_results

    def display_metrics(self, epoch):
        """
        Pretty print the epoch metrics
        """
        outp_str = "+++ Metrics +++\n"
        precision = 0.0
        recall = 0.0
        for name, metric in self.train_metrics.items():
            outp_str += "{}: {:.5f} | ".format(name, float(metric.result()))
            tf.summary.scalar(name, float(metric.result()), epoch)
            if name == "Precision":
                precision = float(metric.result())
            if name == "Recall":
                recall = float(metric.result())
        outp_str += "F1: {:.5f}\n".format(self.calculate_f1(recall, precision))
        for name, metric in self.test_metrics.items():
            outp_str += "(VAL) {}: {:.5f} | ".format(name, float(metric.result()))
            tf.summary.scalar(name, float(metric.result()), epoch)
        print(outp_str)

    def calculate_f1(self, recall, precision):
        """
        Calculate the F1 Score
        """
        if recall <= 0.0 and precision <= 0.0:
            return 0.0
        return 2 * ((precision * recall) / (precision + recall))

    def add_callbacks(self):
        """add tensorflow callbacks to the custom training loop"""
        # checkpoint callback
        #self._callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        #    filepath=self.config["path_to_results"] + "gcnn_checkpoint.ckpt",
        #    save_weights_only=True,
        #    verbose=1
        #))
        # roc/auc
        # tensorboard callback
        self._callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=self.config["path_to_log"] + str(round(time.time() * 1000)), update_freq=1)
        )

    def save(self, fname="gcnn_checkpoint.ckpt"):
        """
        Save the current model.
        """
        try:
            print("Saving trained model...")
            #save_path = self.config["path_to_results"]
            ##fname = "model" + round(time.time() * 1000)
            self.model.save(fname, save_format="tf")
        except Exception as e:
            raise e

    def load(self, fname="gcnn_checkpoint.ckpt"):
        """
        Load the current model.
        """
        print("Loading checkpoint weights...")
        try:
            #save_path = self.config["path_to_results"]
            self.model = tf.keras.models.load_model(fname)
            return self.model
        except Exception as e:
            raise e
