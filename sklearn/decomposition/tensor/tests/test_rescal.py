from scipy.io.matlab import loadmat
from scipy.sparse import coo_matrix
from sklearn.decomposition.tensor.rescal import RESCAL
from sklearn.decomposition.tensor.preprocessing import train_test_split, trial_tensor
import numpy as np

import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)



def test_cpurescal():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]).tocsr() for i in range(k)]
    idxs = train_test_split(T, n_folds=10, n_tp=100, n_tn=1000)
    TT = trial_tensor(T, idxs[0])

    R = RESCAL(45, verbose=True)
    R.fit(TT, val_data=(idxs[0][1], idxs[0][3]))
