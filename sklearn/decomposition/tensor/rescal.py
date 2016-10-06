from cpurescal import als

class RESCAL:
    """RESCAL Tensor Decomposition

    Introduced by Nickel et. al 
    http://www.cip.ifi.lmu.de/~nickel/data/paper-icml2011.pdf

    Factors a three way tensor of size E x E x K, representing the adjacency
    matrix of K relationship types on an E node graph into:

    X_k = A * R_k * A'

    Where A is E x r and R is r x r x K and r is the selected rank of the 
    decomposition. 


    See
    ---
    For a full description of the algorithm see:
    .. [1] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
        "A Three-Way Model for Collective Learning on Multi-Relational Data",
        ICML 2011, Bellevue, WA, USA

    .. [2] Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel,
        "Factorizing YAGO: Scalable Machine Learning for Linked Data"
        WWW 2012, Lyon, France

    """
    def __init__(self, rank, use_gpu=False, max_iter=25, verbose=False):
        self.rank = rank
        self.use_gpu = use_gpu
        self.max_iter=max_iter
        self.verbose=verbose

    def fit(self, X, val_data=None):
        self.A, self.R, _, _, _ = als(X, 
                                      self.rank, 
                                      val_data = val_data, 
                                      verbose=self.verbose)

    def predict(self):
        pass

from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def predict_rescal_als(T):
    A, R, _, _, _ = als(
        T, 100, init='nvecs', conv=1e-3,
        lambda_A=10, lambda_R=10
    )
    n = A.shape[0]
    P = np.zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = np.dot(A, np.dot(R[k], A.T))
    return P


def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = np.linalg.norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P


def innerfold(T, mask_idx, target_idx, e, k, sz):
    Tc = [Ti.copy() for Ti in T]
    mask_idx = np.unravel_index(mask_idx, (e, e, k))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    # set values to be predicted to zero
    for i in range(len(mask_idx[0])):
        Tc[mask_idx[2][i]][mask_idx[0][i], mask_idx[1][i]] = 0

    # predict unknown values
    P = predict_rescal_als(Tc)
    P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[target_idx], P[target_idx])
    return auc(recall, prec)





if __name__=="__main__":
    mat = loadmat('tests/alyawarradata.mat')
    K = np.array(mat['Rs'], np.float32)
    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k
    GROUND_TRUTH = K.copy()
    T = [lil_matrix(K[:, :, i]) for i in range(k)]
    print T[0]
    FOLDS = 10
    IDX = list(range(SZ))
    np.random.shuffle(IDX)
    fsz = int(SZ / FOLDS)
    offset = 0
    idx_test = IDX[offset:offset + fsz]
    idx_train = np.setdiff1d(IDX, idx_test)
    np.random.shuffle(idx_train)
    idx_train = idx_train[:fsz].tolist()
    print('Train Fold %d' % 0)
    print innerfold(T, idx_train + idx_test, idx_train, e, k, SZ)
    print ('Test Fold %d' % 0)
    print innerfold(T, idx_test, idx_test, e, k, SZ)

