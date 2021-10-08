Tensor-Train Density Estimation
-------------------------------

Code for reproducing experiments from this paper:
>Novikov, Georgii S., Maxim E. Panov, and Ivan V. Oseledets. "Tensor-Train Density Estimation." arXiv preprint arXiv:2108.00089 (2021). 
>[[arxiv]](https://arxiv.org/abs/2108.00089#:~:text=We%20propose%20a%20new%20efficient,also%20has%20very%20intuitive%20hyperparameters.)


Installation
------------

All requirements can be installed via [poetry](https://github.com/python-poetry/poetry):
```
pip install poetry
poetry install git+https://github.com/stat-ml/TTDE
```
WARNING: The `jax` library with poetry is installed without GPU support. We recommend that you install the correct version of `jax` separately after installing packages with `poetry`, following the recommendations from [repository](https://github.com/google/jax).


Data preparation
----------------

Download tabular data from the UCI dataset collection as described in [MAF](https://github.com/gpapamak/maf). The following examples assume that the data are unzipped into the `~/from-MAF-paper` folder:
```
ls ~/from-MAF-paper

from_MAF_paper/
|-- BSDS300
|   `-- BSDS300.hdf5
|-- cifar10
|   |-- data_batch_1
|   |-- data_batch_2
|   |-- data_batch_3
|   |-- data_batch_4
|   |-- data_batch_5
|   `-- test_batch
|-- gas
|   `-- ethylene_CO.pickle
|-- hepmass
|   |-- 1000_test.csv
|   `-- 1000_train.csv
|-- miniboone
|   `-- data.npy
|-- mnist
|   `-- mnist.pkl.gz
`-- power
    `-- data.npy
```


Training
--------

Use the `ttde/train.py` script to start training: 
```
Usage: python -m ttde.train [OPTIONS]

Options:
  --dataset [Power|Gas|Hepmass|Miniboone|BSDS300]
                                  Name of the dataset. Choose one of Power,
                                  Gas, Hepmass, Miniboone, BSDS300  [required]
  --q INTEGER                     degree of splines  [required]
  --m INTEGER                     number of basis functions  [required]
  --rank INTEGER                  rank of tensor-train decomposition
                                  [required]
  --n-comps INTEGER               number of components in the mixture
                                  [required]
  --em-steps INTEGER              number of EM steps for model initializaion
                                  [required]
  --noise FLOAT                   magnitude of Gaussian noise for model
                                  initializatoin  [required]
  --batch-sz INTEGER              batch size  [required]
  --train-noise FLOAT             Gaussian noise to add to samples during
                                  training  [required]
  --lr FLOAT                      learning rate for Adam optimizer  [required]
  --train-steps INTEGER           number of train steps  [required]
  --data-dir PATH                 directory with MAF datasets  [required]
  --work-dir PATH                 directory where to store checkpoints and
                                  tensorboard plots  [required]
  --help                          Show this message and exit.
```

Reproduce the results from the article (Table 3) as follows:
```
power:
	python -m ttde.train --dataset power --q 2 --m 256 --rank 16 --n-comps 32 --em-steps 10 --noise 0.01 --batch-sz 8192 --train-noise 0.01 --lr 0.001 --train-steps 10000 --data-dir ~/from-MAF-paper --work-dir ~/workdir
gas:
	python -m ttde.train --dataset gas --q 2 --m 512 --rank 32 --n-comps 32 --em-steps 10 --noise 0.01 --batch-sz 1024 --train-noise 0.01 --lr 0.001 --train-steps 100000 --data-dir ~/from-MAF-paper --work-dir ~/workdir
hepmass:
	python -m ttde.train --dataset hepmass --q 2 --m 128 --rank 32 --n-comps 32 --em-steps 10 --noise 0.01 --batch-sz 2048 --train-noise 0.01 --lr 0.001 --train-steps 10000 --data-dir ~/from-MAF-paper --work-dir ~/workdir
miniboone:
	python -m ttde.train --dataset miniboone --q 2 --m 64 --rank 32 --n-comps 32 --em-steps 10 --noise 0.01 --batch-sz 1024 --train-noise 0.08 --lr 0.001 --train-steps 10000 --data-dir ~/from-MAF-paper --work-dir ~/workdir
bsds300:
	python -m ttde.train --dataset bsds300 --q 2 --m 256 --rank 16 --n-comps 32 --em-steps 10 --noise 0.01 --batch-sz 512 --train-noise 0.01 --lr 0.001 --train-steps 100000 --data-dir ~/from-MAF-paper --work-dir ~/workdir
```
After the training is over, the results can be viewed using the `tensorboard`:
```
tensorboard --logdir ~/workdir
```

In all examples, replace the path `~/from-MAF-paper` with the one where you put the dataset data, and replace `~/workdir/` with the folder where you want the results to be placed in.
