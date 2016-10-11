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
    def __init__(self, rank, gpu=False, random_state=None):
        self.rank = rank
        self.gpu = gpu
        if gpu:
            from gpurescal import RESCAL
            from skcuda.misc import init as ginit
            ginit()
            self.R = RESCAL()
            
        else:
            from cpurescal import CPURESCAL as RESCAL
            self.R = RESCAL(random_state=random_state)

    def fit(self, X, **kwargs):
    
        if self.gpu:
            self.R.setX(X)
            if "A" in kwargs:
                self.R.A_full = kwargs["A"]
            self.R.set_rank(self.rank, history=True)
            self.R.fit(0.0,0.0)    
        else:
            self.R.set_x(X, self.rank)
            self.R.fit(self.rank, **kwargs)

    def predict(self, X_test):
        self.R.predict(X_test)
