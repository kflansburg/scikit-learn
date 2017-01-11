from scipy.io.matlab import loadmat
from scipy.sparse import coo_matrix
from sklearn.decomposition.tensor.rescal import RESCAL
from sklearn.decomposition.tensor.preprocessing import train_test_split, trial_tensor
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
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
        #prec, recall, _ = precision_recall_curve(y_test, y_pred)
        #return auc(recall, prec)
        return roc_auc_score(y_test, y_pred)

def test_gpurescal():
    mat = loadmat('sklearn/decomposition/tensor/tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    T = [coo_matrix(K[:, :, i]).tocsr() for i in range(k)]
    idxs = train_test_split(T, n_folds=10, n_tp=500, n_tn=5000)
    TT = trial_tensor(T, idxs[0])

    RCPU = RESCAL(50)
    RCPU.fit(TT, val_data=(idxs[0][1], idxs[0][3]), lambda_A=0.0, lambda_R=0.0, history=True, conv=0.0, max_iter=30)

    Ainit = RCPU.R.A[0].astype(np.float32)

    RGPU = RESCAL(50, gpu=True)
    RGPU.fit(TT, val_data=(idxs[0][1], idxs[0][3]), lambda_A=0.0, lambda_R=0.0, A=Ainit)

    results = {}
    dims = {}
    for sym in ["A","R","F","E","S","U","Vt","Shat"]:
        results[sym] = []

        XGPU = eval("RGPU.R.%s"%(sym))
        XCPU = eval("RCPU.R.%s"%(sym))
        assert XGPU.shape==XCPU.shape, "Iterations between CPU and GPU mismatch. %s\t%s"%(repr(XGPU.shape), repr(XCPU.shape))
        dims[sym] = " x ".join(map(str, XGPU.shape[1:]))

        for i in range(XGPU.shape[0]):
            if XGPU.ndim==3:
                nrm = np.linalg.norm(XGPU[i,:,:] - XCPU[i,:,:], 'fro')
            elif XGPU.ndim==2:
                nrm = np.linalg.norm(XGPU[i,:] - XCPU[i,:], 2)
            elif XGPU.ndim==4:
                nrm = 0.0
                for j in range(XGPU.shape[1]):
                    nrm+= np.linalg.norm(XGPU[i,j,:,:] - XCPU[i,j,:,:], 'fro')
            results[sym].append(nrm)

    results['AUC'] = []
    dims['AUC'] = ""
    for i in range(RGPU.R.A.shape[0]):
        A = RGPU.R.A[i,:,:]
        R = RGPU.R.R[i,:,:,:]
        aucGPU = compute_auc(idxs[0][1], idxs[0][3], A, R)

        A = RCPU.R.A[i,:,:]
        R = RCPU.R.R[i,:,:,:]
        aucCPU = compute_auc(idxs[0][1], idxs[0][3], A, R)
        results['AUC'].append(aucCPU-aucGPU)

    
    print "{: ^5s}".format("ITER") + "||" + "|".join(map(lambda s: "{: ^12s}".format(s), dims.keys()))
    print "{: ^5s}".format("") + "||" + "|".join(map(lambda s: "{: ^12s}".format(s), dims.values()))
    print "="*123 
    N = max(map(lambda l: len(l), results.values()))
    for k,v in results.iteritems():
        if len(v)<N:
            results[k] = [ np.Inf ] * ( N - len(v) ) + v

    for i in range(N):
        r = map(lambda l: l[i], results.values())
        print "{: ^5d}".format(i) + "||" +"|".join(map(lambda s: "{: ^12.2e}".format(s), r))
        



#def test_cpurescal():
#    mat = loadmat('tests/alyawarradata.mat')
#    K = np.array(mat['Rs'], np.float32)
#    e, k = K.shape[0], K.shape[2]
#    T = [coo_matrix(K[:, :, i]).tocsr() for i in range(k)]
#    idxs = train_test_split(T, n_folds=10, n_tp=500, n_tn=5000)
#    TT = trial_tensor(T, idxs[0])
#    R = RESCAL(25)
#    R.fit(TT, val_data=(idxs[0][1], idxs[0][3]), lambda_A=10.0, lambda_R=10.0)
