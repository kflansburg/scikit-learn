from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
import numpy as np

def trial_tensor(*args, **options):
    """Given a Tensor and fold indecies, produce a copy of the tensor with
    test indices zeroed out. 

    Parameters
    ----------
    slices : [scipy.sparse.spmatrix]
        List of sparse matrices representing each tensor slice.

    indexes : (np.array, np.array, np.array, np.array)
        X_train, X_test, y_train, y_test

    Returns
    -------
    slices : [scipy.sparse.spmatrix]
        List of sparse matrices with test indecies zeroed out.  
    """

    if len(args)==0: return

    T = args[0]
    X_train, X_test, y_train, y_test = args[1]
    
    TT = [s.copy().tolil() for s in T]
    idxs = np.unique(X_test[:,0])
    for i in idxs:
        rows = X_test[X_test[:,0]==i,1]
        cols = X_test[X_test[:,0]==i,2]
        data = y_test[X_test[:,0]==i]
        TT[i][rows,cols]=0
    TT = [s.tocsr() for s in TT]
    for s in TT: s.eliminate_zeros()
    return TT


def train_test_split(*slices, **options):
    """Generate training and testing indecies for fast tensor validation.

    Parameters
    ----------
    slices : [scipy.sparse.spmatrix]
        List of sparse matrices representing each tensor slice.

    n_folds : int
        Number of training folds.

    n_tp : int
        Number of true positive indecies in each fold.

    n_tf : int
        Number of true negative indecies in each fold. 

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    Returns
    -------
    fold_idxs : [[(int,int,float)]]
        List of folds, each fold is a list of tuples containing, 
        row, column, and value for validation. 
    """
    if len(slices)==0: return
    slices = slices[0]
    K = len(slices)
    E = slices[0].shape[0]

    n_folds = options.pop('n_folds', 10)
    n_tp = options.pop('n_tp', 100)
    n_tn = options.pop('n_tn', 1000)
    random_state = options.pop('random_state', None)
    val_slice = options.pop('val_slice', list(range(K)))
    rng = check_random_state(random_state)

    tot_idxs = []
    for i in range(K):
        if i in val_slice:
            s = slices[i]
            s = s.tocoo()

            nnz = s.nnz

            rows = s.row
            cols = s.col
            slis = i * np.ones(nnz, dtype=np.int)
            data = s.data

            tot_idxs += zip(slis, rows, cols, data) 

    # Check that there are enough TP indexes
    N_TP = len(tot_idxs)
    assert N_TP >= n_folds*n_tp, "Insufficient TP Relationships %d/%d"%(N_TP, n_folds*n_tp)

    # Shuffle TP indexes
    rng.shuffle(tot_idxs)
    tp_idxs = tot_idxs[:n_tp*n_folds]
    tot_idxs = set(map(lambda i: (i[0], i[1], i[2]), tot_idxs))

    # Check that there are enough TN indexes
    tn_idxs = set([])
    Kv = len(val_slice)
    assert Kv*E*E - N_TP - n_tn*n_folds >= 0, "Insufficient TF Relationships %d/%d"%(Kv*E*E - N_TP, n_folds*n_tn)
   
    # Select TN Indexes 
    while len(tn_idxs) < n_tn * n_folds:
        t = (rng.choice(val_slice), rng.randint(E), rng.randint(E))

        # Check that this is a new index
        n1 = len(tot_idxs)
        tot_idxs.add(t)
        if len(tot_idxs)>n1:
            tn_idxs.add(t)

    X_TP = np.array((list(tp_idxs)))
    y_TP = X_TP[:,-1].reshape(-1,1)
    X_TP = X_TP[:,:-1]
   
    X_TN = np.array((list(tn_idxs))) 
    y_TN = np.zeros(X_TN.shape[0]).reshape(-1,1)

    X = np.vstack([X_TP, X_TN]).astype(np.int)
    y = np.vstack([y_TP, y_TN])

    ret = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train, test in kf.split(X, y=y):
        ret.append((X[train], X[test], y[train], y[test]))
    return ret
