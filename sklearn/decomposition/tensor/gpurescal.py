import logging
_log = logging.getLogger('RESCAL')

from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils.validation import check_random_state

import sys
import time
import numpy as np

from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh

from theano import shared, function, config, sandbox, scan, sparse
from theano import tensor as TH
from theano.gof import Apply
import theano.sparse.basic as THS
from theano.tensor import nlinalg, slinalg
from theano.sandbox.cuda import GpuOp 
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_from_host, host_from_gpu
from theano.misc.pycuda_utils import to_cudandarray, to_gpuarray

from pycuda import gpuarray
from pycuda import curandom
from pycuda.driver import mem_alloc, memcpy_htod, from_device

from skcuda.linalg import svd, column_major, row_major
from skcuda.linalg import transpose as gpu_transpose

from theano.sandbox.cuda.blas import gpucsrmm2, csr_gpu, gpu_svd, gpucsrmm2_2

config.optimizer='fast_run'
config.exception_verbosity='high'
config.floatX = 'float32'


class RESCAL:
    def __init__(self, maxIter=50, maxVec=100):
        self.maxIter = maxIter
        self.lambda_A = shared(np.float32(0.0))
        self.lambda_R = shared(np.float32(0.0))
        self.history = False
        self.maxVec=maxVec

    def _update_R(self, Ap):

        U, S, Vt = gpu_svd(Ap)

        Shat = TH.outer(S,S)
        Shat = Shat / ((Shat ** 2) + self.lambda_R) 

        R = TH.stack([TH.dot(Vt.T, TH.dot((Shat * TH.dot(U.T,gpucsrmm2(x,U))),Vt)) for x in self.X])

        return R, U, S, Vt, Shat
    
    def _update_A(self, Ap, Rp):
        AtA = TH.dot(Ap.T, Ap)
        F = TH.zeros((self.n, self.rank))
        for i in range(len(self.X)): 
            F += gpucsrmm2_2(self.X[i], Ap, Rp[i].T)  
            F += gpucsrmm2_2(self.X[i], Ap, Rp[i], transposeA = True) 

        E = TH.zeros((self.rank, self.rank))
        for i in range(self.k):
            E += TH.dot(Rp[i], TH.dot(AtA, Rp[i].T)) + TH.dot(Rp[i].T, TH.dot(AtA, Rp[i]))

        I = self.lambda_A * TH.eye(self.rank)

        return slinalg.solve(I+E.T, F.T).T, F, E
        
    def _check_input_tensor(self,X):
        """ Check that the tensor X is well formed:
            1. All slices are the same shape
            2. All slices are two dimensional
        """
        sz = X[0].shape
        for i in range(len(X)):
            if X[i].ndim != 2:
                raise ValueError('Frontal slices of X must be matrices')
            if X[i].shape != sz:
                raise ValueError('Frontal slices of X must be all of same shape')
    
    def setX(self, X):
        self._check_input_tensor(X)

        self.X = [x.tocsr().sorted_indices() for x in X]
        dtype = X[0].dtype
        n = self.X[0].shape[0]

        self.n = n
        self.k = len(self.X)
       
        _log.debug("Initializing A")
        sys.stdout.flush()
        tic = time.time()
        S = csr_matrix((n, n), dtype=dtype)
        for i in range(self.k):
            S = S + X[i]
            S = S + X[i].T
        _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), self.maxVec)
        self.A_full = A.astype(np.float32)
        _log.debug("Completed in %f Seconds!"%(time.time()-tic))
        self.X = [shared(csr_gpu(x)) for x in self.X]   

    def set_rank(self, rank, history=False):
        self.history = history
        self.rank = rank
        self.A = self.A_full[:, :rank].copy()

        def oneStep(F, E, U, S, Vt, Shat, A, R):
            A, F, E = self._update_A(A, R)
            R, U, S, Vt, Shat = self._update_R(A)
            return F, E, U, S, Vt, Shat, A, R
       
        Ap = TH.matrix("A")
        Rp = TH.tensor3("R")
        Fp = TH.zeros((self.n, self.rank), dtype=np.float32)
        Ep = TH.zeros((self.rank, self.rank), dtype=np.float32)
        Up = TH.zeros((self.n, self.rank), dtype=np.float32)
        Vtp = TH.zeros((self.rank, self.rank), dtype=np.float32)
        Shatp = TH.zeros((self.rank, self.rank), dtype=np.float32)
        Sp = TH.zeros((self.rank,), dtype=np.float32)

        [F, E, U, S, Vt, Shat, A, R], updates = scan(fn=oneStep, outputs_info=[Fp, Ep, Up, Sp, Vtp, Shatp, Ap, Rp], n_steps=self.maxIter)

        _log.debug("Compiling to GPU")
        sys.stdout.flush()
        tic = time.time()
        if self.history:
            self.f = function(inputs=[Ap,Rp], outputs=(F, E, U, S, Vt, Shat, A, R), updates=updates)
        else:
            self.f = function(inputs=[Ap,Rp], outputs=(A[-1],R[-1]), updates=updates)
        _log.debug("Completed in %f Seconds!"%(time.time()-tic))
       
        _log.debug("Initializing R")
        sys.stdout.flush()
        tic = time.time()
        Ap = TH.matrix()
        if self.history:
            self.R, self.U, self.S, self.Vt, self.Shat = function([Ap], self._update_R(Ap))(self.A)
        else:
            self.R = function([Ap], self._update_R(Ap)[0])(self.A)
        _log.debug("Completed in %f Seconds!"%(time.time()-tic))
 
    def fit(self, lambda_A, lambda_R, val_data=(None, None)):
        self.lambda_A.set_value(np.float32(lambda_A))
        self.lambda_R.set_value(np.float32(lambda_R))
        Ain = self.A
        Rin = self.R
        _log.debug("Computing".ljust(30))
        tic = time.time()
        if self.history:
            self.F, self.E, U, S, Vt, Shat, A, R = self.f(Ain, Rin)

            # Copy from GPU and Extend Axis
            self.A = self.A[np.newaxis, :, :]
            self.R = self.R[np.newaxis, :, :, :]
            self.U = np.asarray(self.U)[np.newaxis, :, :]
            self.S = np.asarray(self.S)[np.newaxis, :]
            self.Vt = np.asarray(self.Vt)[np.newaxis, :, :]
            self.Shat = np.asarray(self.Shat)[np.newaxis, :, :]

            # Stack Results
            self.A = np.append(self.A, A, axis=0)
            self.U = np.append(self.U, U, axis=0)
            self.S = np.append(self.S, S, axis=0)
            self.Vt = np.append(self.Vt, Vt, axis=0)
            self.Shat = np.append(self.Shat, Shat, axis=0)
            self.R = np.append(self.R, R, axis=0)

        else:
            self.A, self.R = self.f(Ain, Rin)
        _log.debug("Completed in %f Seconds!"%(time.time()-tic))
