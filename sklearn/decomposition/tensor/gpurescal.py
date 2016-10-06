import logging
logger = logging.getLogger()
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
    def __init__(self, maxIter=25, maxVec=100):
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
        if self.history:
            return R, Shat, S
        else:
            return R
    
    def _update_A(self, Ap, Rp):
        AtA = TH.dot(Ap.T, Ap)
        F = TH.zeros((self.n, self.rank))
        for i in range(len(self.X)): 
            #F += gpucsrmm2(self.X[i], TH.dot(Ap, Rp[i].T))  
            F += gpucsrmm2_2(self.X[i], Ap, Rp[i].T)  
            #F += gpucsrmm2(self.X[i], TH.dot(Ap, Rp[i]), transposeA = True) 
            F += gpucsrmm2_2(self.X[i], Ap, Rp[i], transposeA = True) 

        E = TH.zeros((self.rank, self.rank))
        for i in range(self.k):
            E += TH.dot(Rp[i], TH.dot(AtA, Rp[i].T)) + TH.dot(Rp[i].T, TH.dot(AtA, Rp[i]))

        I = self.lambda_A * TH.eye(self.rank)

        if self.history:
            return slinalg.solve(I+E.T, F.T).T, F, E
        else:
            return slinalg.solve(I+E.T, F.T).T
        
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
       
        print "\tInitializing A".ljust(30),
        sys.stdout.flush()
        tic = time.time()
        S = csr_matrix((n, n), dtype=dtype)
        for i in range(self.k):
            S = S + X[i]
            S = S + X[i].T
        _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), self.maxVec)
        self.A_full = A.astype(np.float32)
        print time.time()-tic

        self.X = [shared(csr_gpu(x)) for x in self.X]   

    def set_rank(self, rank):
        self.rank = rank
        self.A = self.A_full[:, :rank].copy()

        def oneStep(A, R):
            A = self._update_A(A, R)
            R = self._update_R(A)
            return A,R
       
        Ap = TH.matrix("A")
        Rp = TH.tensor3("R")
       
        [A,R], updates = scan(fn=oneStep, outputs_info=[Ap,Rp], n_steps=self.maxIter)

        print "\tCompiling to GPU".ljust(30),
        sys.stdout.flush()
        tic = time.time()
        self.f = function(inputs=[Ap,Rp], outputs=(A[-1],R[-1]), updates=updates)
        print time.time()-tic
       
        print "\tInitializing R".ljust(30),
        sys.stdout.flush()
        tic = time.time()
        Ap = TH.matrix()
        self.R = function([Ap], self._update_R(Ap))(self.A)
        print time.time()-tic

 
    def fit(self, lambda_A, lambda_R, history=False):
        self.lambda_A.set_value(np.float32(lambda_A))
        self.lambda_R.set_value(np.float32(lambda_R))

        self.history = history

        Ain = self.A
        Rin = self.R

        #Ap = TH.matrix("A")
        #Rp = TH.tensor3("R")
        #update_A = function([Ap, Rp], self._update_A(Ap, Rp))
        #update_R = function([Ap], self._update_R(Ap))
        #
        #print "\tInitializing R".ljust(30),
        #sys.stdout.flush()
        #tic = time.time()
        #if history:
        #    R,Shat,S = update_R(A)
        #else:
        #    R = update_R(A)
        #print time.time()-tic    

        #if history:
        #    A_hist = [A]
        #    R_hist = [R]
        #    Shat_hist = [Shat]           
        #    S_hist = [S] 
        #    E_hist = [np.zeros((self.rank, self.rank))]            
        #    F_hist = [np.zeros((self.n, self.rank))]            


        #for i in range(self.maxIter):
        #    tic = time.time()
        #    
        #    
        #    if history:
        #        A, F, E = update_A(A,R)
        #        R, Shat, S = update_R(A)
        #        A_hist.append(A)
        #        R_hist.append(R)
        #        F_hist.append(F)
        #        E_hist.append(E)
        #        Shat_hist.append(Shat)
        #        S_hist.append(S)
        #    else: 
        #        A = update_A(A, R)
        #        R = update_R(A)

        #    print "\tITER %d\t%f"%(i,time.time()-tic)
        #   
        #if history:
        #    self.F = F_hist
        #    self.E = E_hist
        #    self.Shat = Shat_hist
        #    self.S = S_hist
        #    self.A = A_hist
        #    self.R = R_hist
        #else: 
        #    self.A = A
        #    self.R = R
    
        
       
        print "\tComputing".ljust(30),
        sys.stdout.flush()
        tic = time.time()
        self.A, self.R = self.f(Ain, Rin)
        print time.time()-tic

