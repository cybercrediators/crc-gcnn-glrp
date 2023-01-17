import time
import numpy as np
from spektral.data.loaders import MixedLoader
from tensorflow.keras.callbacks import EarlyStopping
from glrp.base.base_trainer import BaseTrainer


class MnistRunner(BaseTrainer):
    def __init__(self, config, model, data):
        super(MnistRunner, self).__init__(config, model, data)

    def summarize(self):
        pass

    def run(self):
        start = time.time()


        # train/test_split
        train_data, test_data = self.data[:-10000], self.data[-10000:]
        np.random.shuffle(train_data)
        train_data, val_data = train_data[:-10000], train_data[-1000:]
        train_loader = MixedLoader(train_data, batch_size=self.config["batch_size"], epochs=self.config["num_epochs"])
        val_loader = MixedLoader(val_data, batch_size=self.config["batch_size"])
        test_loader = MixedLoader(test_data, batch_size=self.config["batch_size"])
        print(self.data.a)

        self.model.build_model()
        self.model.train(train_loader, test_loader, val_loader)

        #print("Evaluate...")
        #self.model.evaluate()
        end = time.time()
        print("Total time: ", end - start)
