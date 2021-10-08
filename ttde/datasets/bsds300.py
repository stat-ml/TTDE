from pathlib import Path

import numpy as np
import h5py

from ttde.dl_routine import TensorDatasetX


class BSDS300:
    """
    A dataset of patches from BSDS300.
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):

            self.x = data[:]
            self.N = self.x.shape[0]

    def __init__(self, root):

        # load dataset
        f = h5py.File(root, 'r')

        self.trn = self.Data(f['train'])
        self.val = self.Data(f['validation'])
        self.tst = self.Data(f['test'])

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2

        f.close()


def load_bsds300_dataset(path: Path, to_jax: bool = True):
    power = BSDS300(path / 'BSDS300' / 'BSDS300.hdf5')

    data_train = TensorDatasetX(power.trn.x)
    data_val = TensorDatasetX(power.val.x)

    if to_jax:
        data_train = data_train.to_jax()
        data_val = data_val.to_jax()

    return data_train, data_val
