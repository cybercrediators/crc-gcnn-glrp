import time
import numpy as np
from glrp.base.base_trainer import BaseTrainer
from spektral.data.loaders import BatchLoader, MixedLoader, DisjointLoader, SingleLoader
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten
from spektral.layers import ChebConv, GlobalMaxPool, GCNConv
from spektral.datasets.mnist import MNIST

from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy, binary_accuracy
from tensorflow.keras import Model
import tensorflow as tf

class Net(Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        channels = self.config["F"]
        reg = self.config["regularization"]
        out_polynomials = self.config["K"]
        self.cc_1 = ChebConv(channels=channels[0], K=out_polynomials[0], activation='relu', kernel_regularizer=l2(reg), use_bias=True)
        self.cc_2 = ChebConv(channels=channels[1], K=out_polynomials[1], activation='relu', kernel_regularizer=l2(reg), use_bias=True)
        self.f_1 = Flatten()
        fc_size= self.config["M"]
        self.fc_1 = Dense(fc_size[0], activation='relu')
        self.fc_2 = Dense(fc_size[1], activation='relu')
        self.fc_3 = Dense(fc_size[2], activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.cc_1([x, a])
        x = self.cc_2([x, a])
        outp = self.f_1(x)
        outp = self.fc_1(outp)
        outp = self.fc_2(outp)
        outp = self.fc_3(outp)
        return outp


class TestingRunner(BaseTrainer):
    def __init__(self, config, model, data):
        super(TestingRunner, self).__init__(config, model, data)
        self.loss_fn = BinaryCrossentropy(from_logits=False)
        if self.config["momentum"]:
            self.optimizer = SGD(learning_rate=self.config["learning_rate"], momentum=self.config["momentum"])
        else:
            self.optimizer = Adam(learning_rate=self.config["learning_rate"])

    def summarize(self):
        pass

    def run(self):
        start = time.time()
        print("Load data...")
        #data = MNIST()
        #data.a = GCNConv.preprocess(data.a)
        #data.a = sp_matrix_to_sp_tensor(data.a)
        #data_tr, data_te = data[:-10000], data[-10000:]
        #np.random.shuffle(data_tr)
        #train_loader = MixedLoader(data_tr, batch_size=self.config["batch_size"], epochs=self.config["num_epochs"])
        #test_loader = MixedLoader(data_te, batch_size=self.config["batch_size"])

        # convert dataset to mixed mode
        train_loader = MixedLoader(self.data.train_data, batch_size=self.config["batch_size"], epochs=self.config["num_epochs"], shuffle=False)
        test_loader = MixedLoader(self.data.test_data, batch_size=self.config["batch_size"], shuffle=False)

        print(self.data.dataset[0].x.shape)
        print(self.data.dataset[0].x)
        print(self.data.dataset.a.shape)
        print(self.data.dataset.a)

        print("Start Training...")

        self.model = Net(self.config)

        step = 0
        results_tr = []
        patience = 100000
        current_patience = patience
        best_val_loss = 99999

        for batch in train_loader:
            step += 1

            inputs, target = batch
            loss, acc = self.train_step(inputs, target)
            results_tr.append((loss, acc, len(target)))
            
            if step == train_loader.steps_per_epoch:
                results_va = self.evaluate(test_loader)
                if results_va[0] < best_val_loss:
                    best_val_loss = results_va[0]
                    current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience == 0:
                        print("Early stopping")
                        break

                # Print results
                results_tr = np.array(results_tr)
                results_tr = np.average(results_tr[:, :-1], 0, weights=results_tr[:, -1])
                print(
                    "Train loss: {:.4f}, acc: {:.4f} | "
                    "Valid loss: {:.4f}, acc: {:.4f} | ".format(
                        *results_tr, *results_va
                    )
                )

                # Reset epoch
                results_tr = []
                step = 0

        #self.model.fit(
        #    train_loader.load(),
        #    steps_per_epoch=train_loader.steps_per_epoch,
        #    validation_data=test_loader.load(),
        #    validation_steps=test_loader.steps_per_epoch,
        #    epochs=self.config["num_epochs"],
        #    callbacks=[EarlyStopping(patience=self.config["early_stopping"], restore_best_weights=True)]
        #)

        print("Evaluate: ")
        eval_results = self.model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)


        print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

        end = time.time()
        print("Time: {} seconds".format(end - start))

    @tf.function
    def train_step(self, inputs, target):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(target, predictions) + sum(self.model.losses)
            acc = tf.reduce_mean(binary_accuracy(target, predictions))
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, acc

    def evaluate(self, loader):
        step = 0
        results = []
        for batch in loader:
            step += 1
            inputs, target = batch
            predictions = self.model(inputs, training=False)
            target = target.reshape((-1, 1))
            loss = self.loss_fn(target, predictions)
            acc = tf.reduce_mean(binary_accuracy(target, predictions))
            results.append((loss, acc, len(target)))  # Keep track of batch size
            if step == loader.steps_per_epoch:
                results = np.array(results)
                return np.average(results[:, :-1], 0, weights=results[:, -1])
