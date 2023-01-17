import time
import numpy as np
import copy
from glrp.base.base_trainer import BaseTrainer
from glrp.model.gcnn import GCNNModel
from spektral.data.loaders import MixedLoader
from lib import coarsening
from lib import graph
from sklearn.model_selection import KFold

from spektral.utils import convolution
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from scipy.sparse import csr_matrix


class GCNNRunner(BaseTrainer):
    def __init__(self, config, model, data):
        super(GCNNRunner, self).__init__(config, model, data)
        self.config = config
        self.model = model
        self.data = data

    def summarize(self):
        pass

    def run(self):
        start = time.time()
        #self.config["batch_size"] = 10

        rng = np.random.default_rng()
        idx = rng.permutation(len(self.data.train_data))

        #train_loader = MixedLoader(self.data.train_data, batch_size=self.config["batch_size"], epochs=self.config["num_epochs"], shuffle=True)
        test_loader = MixedLoader(self.data.test_data, batch_size=self.config["batch_size"], shuffle=True)
        # val_loader = MixedLoader(self.data.test_data, batch_size=self.config["batch_size"], shuffle=True)
        #m = self.model.build_model()
        #m.summary()
        #self.model.train(train_loader, test_loader, val_loader)


        # create 10 folds
        kf = KFold(n_splits=10)
        kf.get_n_splits(self.data.train_data)
        fold_num = 0
        best_test_res = [9999., 0.]
        all_acc = []
        for train_index, test_index in kf.split(self.data.train_data):
            self.model = GCNNModel(self.config, self.data)
            train_loader = MixedLoader(self.data.train_data[train_index], batch_size=self.config["batch_size"], epochs=self.config["num_epochs"], shuffle=True)
            val_loader = MixedLoader(self.data.train_data[test_index], batch_size=self.config["batch_size"], shuffle=True)
            m = self.model.build_model()
            m.summary()
            test_res = self.model.train(train_loader, test_loader, val_loader)
            print("\n\n\n FOLD: {}".format(fold_num))
            print(test_res)
            all_acc.append(test_res[1] * 100.0)
            print(all_acc)
            fold_num += 1
            if test_res[0] < best_test_res[0] and test_res[1] > best_test_res[1]:
                print("NEUE BESTLEISTUNG!\n\n")
                best_test_res = copy.copy(test_res)
                self.model.save(self.config["model_path"])
        tf_model = self.model.load(self.config["model_path"])
        a = tf_model.get_weights()
        self.model.model.set_weights(a)
        print("BEST MODEL SCORE: {}, {}", best_test_res[0], best_test_res[1])
        print("AVG.: {} (+- {})".format(np.mean(all_acc), np.std(all_acc)))

        end = time.time()
        print("Total time: ", end - start)
