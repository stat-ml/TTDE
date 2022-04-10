from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ttde.dl_routine import TensorDatasetX


__all__ = ['load_resnet_embeddings_dataset']


class _EMBEDDINGS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, train_path, test_path):

        trn, val, tst = _load_data_normalised(train_path, test_path)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        # util.plot_hist_marginals(data_split.x)
        plt.show()


def _load_data(train_path, test_path):

    return np.load(train_path), np.load(test_path)


def _load_data_split(train_path, test_path):

    rng = np.random.RandomState(42)

    train_data, test_data = _load_data(train_path, test_path)
    rng.shuffle(train_data)
    N = train_data.shape[0]

    N_validate = int(0.1*train_data.shape[0])
    data_validate = train_data[-N_validate:]
    data_train = train_data[0:-N_validate]

    return data_train, data_validate, test_data


def _load_data_normalised(train_path, test_path):

    data_train, data_validate, data_test = _load_data_split(train_path, test_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test


def load_resnet_embeddings_dataset(train_path: str, test_path: str, to_jax: bool = True):

    embs = _EMBEDDINGS(train_path, test_path)

    data_train = TensorDatasetX(embs.trn.x)
    data_val = TensorDatasetX(embs.val.x)
    data_test = TensorDatasetX(embs.tst.x)

    if to_jax:
        data_train = data_train.to_jax()
        data_val = data_val.to_jax()
        data_test = data_test.to_jax()

    return data_train, data_val, data_test
