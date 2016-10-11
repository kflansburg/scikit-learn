from scipy.io.matlab import loadmat
from scipy.sparse import coo_matrix
from sklearn.decomposition.tensor.rescal import RESCAL
from sklearn.decomposition.tensor.preprocessing import train_test_split, trial_tensor
from sklearn.metrics import precision_recall_curve, auc
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

def predict(X_test, A, R):
    y_pred = []
    for row in X_test:
        slice, row, col = row
        y_pred.append(A[row,:].dot(R[slice]).dot(A[col,:].T))
    return np.array(y_pred)

def compute_auc(X_test, y_test, A, R): 
        y_pred = predict(X_test, A, R)
        prec, recall, _ = precision_recall_curve(y_test, y_pred)
        return auc(recall, prec)

def test_gpurescal():
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]).tocsr() for i in range(k)]
    idxs = train_test_split(T, n_folds=10, n_tp=500, n_tn=5000)
    TT = trial_tensor(T, idxs[0])

    RCPU = RESCAL(50)
    RCPU.fit(TT, val_data=(idxs[0][1], idxs[0][3]), lambda_A=0.0, lambda_R=0.0, history=True, conv=0.0, max_iter=50)

    Ainit = RCPU.R.A[0].astype(np.float32)

    RGPU = RESCAL(50, gpu=True)
    RGPU.fit(TT, val_data=(idxs[0][1], idxs[0][3]), lambda_A=0.0, lambda_R=0.0, A=Ainit)

    for sym in ["A","R","F","E","S","U","Vt","Shat"]:
        XGPU = eval("RGPU.R.%s"%(sym))
        XCPU = eval("RCPU.R.%s"%(sym))
        print sym
        print "GPU", " x ".join(map(str, XGPU.shape))
        print "CPU", " x ".join(map(str, XCPU.shape))

        assert XGPU.shape==XCPU.shape, "Iterations between CPU and GPU mismatch. %s\t%s"%(repr(XGPU.shape), repr(XCPU.shape))
        for i in range(XGPU.shape[0]):
            if XGPU.ndim==3:
                print "\t",i,np.linalg.norm(XGPU[i,:,:] - XCPU[i,:,:], 'fro')
            elif XGPU.ndim==2:
                print "\t",i,np.linalg.norm(XGPU[i,:] - XCPU[i,:], 2)
            elif XGPU.ndim==4:
                nrm = 0.0
                for j in range(XGPU.shape[1]):
                    nrm+= np.linalg.norm(XGPU[i,j,:,:] - XCPU[i,j,:,:], 'fro')
                print "\t",i,nrm

    for i in range(RGPU.R.A.shape[0]):
        A = RGPU.R.A[i,:,:]
        R = RGPU.R.R[i,:,:,:]
        aucGPU = compute_auc(idxs[0][1], idxs[0][3], A, R)

        A = RCPU.R.A[i,:,:]
        R = RCPU.R.R[i,:,:,:]
        aucCPU = compute_auc(idxs[0][1], idxs[0][3], A, R)
        print i, "\tCPU", aucCPU, "\tGPU", aucGPU 

#def test_cpurescal():
#    mat = loadmat('tests/alyawarradata.mat')
#    K = np.array(mat['Rs'], np.float32)
#    e, k = K.shape[0], K.shape[2]
#    T = [coo_matrix(K[:, :, i]).tocsr() for i in range(k)]
#    idxs = train_test_split(T, n_folds=10, n_tp=500, n_tn=5000)
#    TT = trial_tensor(T, idxs[0])
#    R = RESCAL(25)
#    R.fit(TT, val_data=(idxs[0][1], idxs[0][3]), lambda_A=10.0, lambda_R=10.0)
