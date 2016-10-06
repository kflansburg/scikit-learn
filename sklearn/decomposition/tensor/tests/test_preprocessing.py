from nose.tools import *
from scipy.io.matlab import loadmat
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.decomposition.tensor.preprocessing import train_test_split, trial_tensor

def test_trial_tensor():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]).tocsr() for i in range(k)]
    idxs = train_test_split(T, n_folds=10, n_tp=100, n_tn=1000)
    
    X_train = idxs[0][0] 
    X_test = idxs[0][1]
    y_train = idxs[0][2]
    y_test = idxs[0][3]

    TT = trial_tensor(T, idxs[0])
    for slice in range(k):
        x = X_test[X_test[:,0]==slice]
        assert np.count_nonzero(TT[slice][x[:,1], x[:,2]]) == 0
        x = X_train[X_train[:,0]==slice]
        assert np.count_nonzero(TT[slice][x[:,1], x[:,2]]) > 0 or np.count_nonzero(T[slice][x[:,1], x[:,2]]) == 0

def test_train_test_split():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]) for i in range(k)]
    FOLDS = 6

    idxs = train_test_split(T, n_folds=FOLDS, n_tp=50, n_tn=500)

    assert len(idxs)==FOLDS, "Does not return correct number of folds. %d != %d"%(len(idxs), FOLDS)
    assert len(idxs[0])==4, "Does not return information for split."

    for fold in idxs:
        X_train, X_test, y_train, y_test = fold
        assert X_train.shape==((FOLDS-1)*550, 3), X_train.shape
        assert X_test.shape==(550,3), X_test.shape
        assert y_train.shape==((FOLDS-1)*550,1), y_train.shape
        assert y_test.shape==(550,1), y_test.shape


def test_train_test_split_slice():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]) for i in range(k)]
    FOLDS = 6
    idxs = train_test_split(T, n_folds=FOLDS, n_tp=50, n_tn=500, val_slice=[3,5])
    for fold in idxs:
        X_train, X_test, y_train, y_test = fold
        nnz = np.count_nonzero((X_train[:,0]==3) | (X_train[:,0]==5))
        assert nnz==(FOLDS-1)*550, nnz


@raises(AssertionError)
def test_train_test_split_insufficient_tp():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]) for i in range(k)]
    FOLDS = 6
    idxs = train_test_split(T, n_folds=FOLDS, n_tp=10000, n_tn=500)


@raises(AssertionError)
def test_train_test_split_insufficient_tn():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]) for i in range(k)]
    FOLDS = 6
    idxs = train_test_split(T, n_folds=FOLDS, n_tp=50, n_tn=100000)
