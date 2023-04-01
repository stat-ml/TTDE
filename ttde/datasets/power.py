from pathlib import Path

import numpy as np

from ttde.dl_routine import TensorDatasetX


__all__ = ['load_power_dataset']


class _POWER:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, root):

        trn, val, tst = _load_data_normalised(root)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def _load_data(root):
    return np.load(root / 'power' / 'data.npy')


def _load_data_split_with_noise(root):

    rng = np.random.RandomState(42)

    data = _load_data(root)
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    ############################
    # Add noise
    ############################
    # global_intensity_noise = 0.1*rng.rand(N, 1)
    voltage_noise = 0.01*rng.rand(N, 1)
    # grp_noise = 0.001*rng.rand(N, 1)
    gap_noise = 0.001*rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def _load_data_normalised(root):

    data_train, data_validate, data_test = _load_data_split_with_noise(root)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test


def load_power_dataset(path: Path, to_jax: bool = True):
    power = _POWER(path)

    data_train = TensorDatasetX(power.trn.x)
    data_val = TensorDatasetX(power.val.x)

    if to_jax:
        data_train = data_train.to_jax()
        data_val = data_val.to_jax()

    return data_train, data_val
