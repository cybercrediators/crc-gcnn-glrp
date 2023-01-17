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


class TestingModel(BaseModel):
    '''testing spektral model with the spektral example model for gcn with chebconv'''
    def __init__(self, config, data):
        super(TestingModel, self).__init__(config)
        self.data = data
        self.a_dtype = self.data.dataset.a.dtype
        self.n_nodes = self.data.dataset.n_nodes
        self.n_node_features = self.data.dataset.n_node_features

    def build_model(self):
        # input layers
        x_in = Input(shape=(self.n_nodes, self.n_node_features))
        a_in = Input((self.n_nodes, self.n_nodes), sparse=True, dtype=self.a_dtype)
        print("\n\n\n")
        print(x_in.shape)
        print(self.n_node_features)
        print(self.data.dataset[0])

        # dropout if configured (no dropout in orig)
        x_in = self.add_dropout(x_in, self.config["dropout"])

        # 2 chebconv layers
        out_polynomials = self.config["K"]
        channels = self.config["F"]
        regularization = self.config["regularization"]
        cc_1 = ChebConv(
            channels=channels[0],
            K=out_polynomials[0],
            activation='relu',
            kernel_regularizer=l2(regularization),
            use_bias=True
        )([x_in, a_in])
        
        # droupout if configured
        cc_1 = self.add_dropout(cc_1, self.config["dropout"])

        cc_2 = ChebConv(
            channels=channels[1],
            K=out_polynomials[1],
            activation='relu',
            kernel_regularizer=l2(regularization),
            use_bias=True
        )([cc_1, a_in])

        # 2 max pooling layers (if pooling config exists)
        layer_3 = cc_2
        #if "p" in self.config:
        #    pooling = self.config["p"]
        #    mp_1 = tf.nn.max_pool(cc_2,
        #                          ksize=[1, pooling[0], 1, 1], 
        #                          strides=[1, pooling[0], 1, 1],
        #                          padding='SAME'
        #                          )
        #    mp_2 = tf.nn.max_pool(mp_1,
        #                          ksize=[1, pooling[0], 1, 1], 
        #                          strides=[1, pooling[0], 1, 1],
        #                          padding='SAME'
        #                          )
        #    layer_3 = mp_2

        # flatten layer a bit problematic
        f_1 = Flatten()(layer_3)
        #f_1 = GlobalSumPool()(layer_3)
        # fc 512
        fc_size = self.config["M"]
        fc_1 = Dense(fc_size[0], activation='relu')(f_1)
        # fc 128
        fc_2 = Dense(fc_size[1], activation='relu')(fc_1)
        # fc output
        out = Dense(fc_size[2], activation='softmax')(fc_2)

        # build model
        model = Model(inputs=[x_in, a_in], outputs=out)
        if self.config["momentum"]:
            optimizer = SGD(learning_rate=self.config["learning_rate"], momentum=self.config["momentum"])
        else:
            optimizer = Adam(learning_rate=self.config["learning_rate"])

        model.compile(
                optimizer=optimizer,
                loss=BinaryCrossentropy(from_logits=False),
                weighted_metrics=["acc"]
            )

        #model.summary()
        return model

    def add_dropout(self, layer, dropout):
        if (dropout) > 0.0:
            return Dropout(dropout)(layer)
        return layer

    def required_params(self):
        """return the required parameters needed from a config file"""
        pass

    def save(self):
        pass

    def load(self):
        pass
