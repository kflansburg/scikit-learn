import logging
import time
import numpy as np
import scipy.sparse
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils.validation import check_random_state

_log = logging.getLogger('RESCAL')

class CPURESCAL:
    """Computes RESCAL decomposition on CPU.

    Parameters
    ----------
    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.

    Attributes
    -------
    A : [np.array] if history else np.array
        Array of shape (N, rank) representing the factor matrix A.

    R : [[np.array]] if history else [np.array]
        list of K arrays of shape (rank, rank) corresponding to the
        factor matrices R_k. 

    U : [np.array] if history else None
    S : [np.array] if history else None
    Vt : [np.array] if history else None
    Shat : [np.array] if history else None
    F : [np.array] if history else None
    E : [np.array] if history else None
        Histories of various intermediate values when updating A and R.

    fit : [float]
        List of AUC fits for each iteration. 

    execttimes : [float]
        List of times per iteration.

    itr : int
        Number of iterations until convergence.
    """

    def __init__(self, **kwargs):
        self.rng = check_random_state(kwargs.pop('random_state', None))
        self.init = kwargs.pop('init', 'eigs')

        if not len(kwargs) == 0:
            raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    def __validate_tensor(self):
        X = self.X
        assert type(X)==list, "X must be a list of sparse matrices"

        for i in range(self.K):
            assert X[i].ndim == 2, 'Frontal slices of X must be matrices.'
            assert X[i].shape == (self.N, self.N), 'Frontal slices of X must be all of same shape and square.'

        self.X = [ x.tocsr() for x in self.X ]
        for x in self.X: x.sort_indices() 

    def __initialize_A(self):

        tic = time.time()
        if self.init=='rand':
            _log.debug('Initializing A Randomly')
            A = self.rng.rand(self.N, self.max_rank)

        elif self.init=='eigs':
            _log.debug('Initializing A By Eigendecompostion')
            S = scipy.sparse.csr_matrix((self.N, self.N))
            for i in range(self.K):
                S += self.X[i]
                S += self.X[i].T
            _, A = scipy.sparse.linalg.eigsh(S, self.max_rank)
    
        else:
            raise ValueError('Unknown init option.')

        _log.debug("DONE: %f Seconds."%(time.time() - tic))
        self.init_A = A

    def set_x(self, X, max_rank, **kwargs):
        """Sets X, and initializes A. 
        
        This is a seperate function so that multiple runs do not have to 
        reinitialize A.

        Parameters
        ----------
        X : [scipy.sparse.spmatrix]
            List of frontal slices X_k of the tensor X.
            The shape of each X_k is (N, N).
   
        max_rank : int
            The largest rank you expect to use. This controls how large
            A is when it is initialized.
 
        init : string, optional
            Initialization method of the factor matrices. 'eigs' (default)
            initializes A based on the eigenvectors of X. 'random' initializes
            the factor matrices randomly.
        """
        
        self.max_rank = max_rank

        self.X = X
        self.K = len(X)
        self.N = X[0].shape[0]
        self.__validate_tensor()
        
        self.__initialize_A()

    def __update_A(self):
        if self.history: A = self.A[-1]
        else: A = self.A
        if self.history: R = self.R[-1]
        else: R = self.R

        F = np.zeros((self.N, self.rank))
        E = np.zeros((self.rank, self.rank))

        AtA = np.dot(A.T, A)

        for i in range(self.K):
            F += self.X[i].dot( np.dot(A, R[i].T)) + self.X[i].T.dot(np.dot(A, R[i]))
            E += np.dot(R[i], np.dot(AtA, R[i].T)) + np.dot(R[i].T, np.dot(AtA, R[i]))

        I = self.lambda_A * np.eye(self.rank)

        A = np.linalg.solve(I + E.T, F.T).T
        
        if self.history:
            self.A.append(A)
            self.F.append(F)
            self.E.append(E)
        else:
            self.A = A

    def __update_R(self):
        if self.history: A = self.A[-1]
        else: A = self.A

        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        
        Shat = np.kron(S, S)
        Shat = (Shat / (Shat ** 2 + self.lambda_R)).reshape(self.rank, self.rank)

        R = []
        for i in range(self.K):
            Rn = Shat * np.dot(U.T, self.X[i].dot(U))
            Rn = np.dot(Vt.T, np.dot(Rn, Vt))
            R.append(Rn)

        if self.history:
            self.U.append(U)
            self.S.append(S)
            self.Vt.append(Vt)
            self.Shat.append(Shat)
            self.R.append(R)
        else:
            self.R = R

    def predict(self, X_test):
        if self.history: 
            A = self.A[-1]
            R = self.R[-1]
        else:
            A = self.A
            R = self.R

        y_pred = []
        for row in X_test:
            slice, row, col = row
            y_pred.append(A[row,:].dot(R[slice]).dot(A[col,:].T))
        return np.array(y_pred)

    def __compute_auc(self): 
        if self.X_test is not None and self.y_test is not None:
            y_pred = self.predict(self.X_test)
            try:
                prec, recall, _ = precision_recall_curve(self.y_test, y_pred)
            except ValueError as e:
                print self.y_test
                print y_pred
                raise ValueError(e)
            self.fits.append(auc(recall, prec))
            self.fitchange = abs(self.fits[-1] - self.fits[-2])
        else:
            self.fits.append(np.Inf)
            self.fitchange = np.Inf

    def __one_step(self):
        self.__update_A()
        self.__update_R()
        self.__compute_auc()

    def fit(self, rank, **kwargs):
        """
        lambda_A : float, optional
            Regularization parameter for A factor matrix. 0.0 by default.
    
        lambda_R : float, optional
            Regularization parameter for R_k factor matrices. 0.0 by default.
    
        val_data: (X: np.array, y: np.array)
            Indecies to use for validating the tensor. 
            First array, X, has rows of [slice_idx, row_idx, col_idx], and 
            the corresponding row in the second array, y, has the target 
            value for that index. If not specified, fit is not calculated. 
    
        max_iter : int, optional
            Maximium number of iterations of the ALS algorithm. 50 by default.
    
        conv : float, optional
            Stop when residual of factorization is less than conv. 1e-5 by default.
    
        history : boolean, optional
            Returns a copy of results for each iteration to allow validation. 
        """    

        self.rank = rank
        self.max_iter = kwargs.pop('max_iter', 50)
        self.conv = kwargs.pop('conv', 10e-6)
        self.lambda_A = kwargs.pop('lambda_A', 0.0)
        self.lambda_R = kwargs.pop('lambda_R', 0.0)
        self.history = kwargs.pop('history', False)
        self.X_test, self.y_test = kwargs.pop('val_data',(None, None))

        if self.history:
            self.U = []
            self.S = []
            self.Vt = []
            self.Shat = []
            self.R = []
            self.A = []
            self.F = []
            self.E = []

        _log.debug(
            '[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
            (self.rank, self.max_iter, self.conv, self.lambda_A)
        )

        if self.history: self.A = [self.init_A[:,:rank].copy()]
        else: self.A = self.init_A[:,:rank].copy()
        self.__update_R()
            
        self.exectimes = []
        self.fits = [np.Inf]
        
        for i in range(self.max_iter):
            tic = time.time()
            self.__one_step()
            self.exectimes.append(time.time()-tic)
            
            _log.debug('[%3d] fval: %0.5f | delta: %7.1e | secs: %.5f' % (
                i, self.fits[-1], self.fitchange, self.exectimes[-1]
            ))

            if i > 0 and self.fitchange < self.conv:
                self.itr=i
                break

        if self.history:
            self.A = np.array(self.A)
            self.R = np.array(self.R)
            self.F = np.array(self.F)
            self.E = np.array(self.E)
            self.U = np.array(self.U)
            self.S = np.array(self.S)
            self.Vt = np.array(self.Vt)
            self.Shat = np.array(self.Shat)
