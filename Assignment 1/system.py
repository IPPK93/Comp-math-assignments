import numpy as np

class System():
    def __init__(self, matrix: np.ndarray, rhs: np.ndarray = None):
        assert matrix.shape[0] == matrix.shape[1], "matrix is not square"
        if rhs is None:
            rhs = np.zeros(matrix.shape[0])
        else:
            assert rhs.shape[0] == matrix.shape[0], "right-hand side does not combine with the matrix"
        self.matrix = matrix
        self.rhs = rhs

    def solve_gauss(self):
        A = self.matrix.copy()
        b = self.rhs.copy()
        nrows, ncols = A.shape
        
#       Forward pass
        for i in range(nrows):
            sorted_indices = i + np.abs(A[i:, i]).argsort()[::-1]
            A[i:] = A[sorted_indices]
            b[i:] = b[sorted_indices]
            diag_elem = A[i][i]
            for j in range(i + 1, nrows):
                if A[j][i] != 0:
                    b[j] = b[j] * diag_elem - b[i] * A[j][i]
                    A[j] = A[j] * diag_elem - A[i] * A[j][i]  
        
#       Backward pass
        for i in reversed(range(nrows)):
            diag_elem = A[i][i]
            for j in range(i):
                if A[j][i] != 0:
                    b[j] = b[j] * diag_elem - b[i] * A[j][i]
                    A[j] = A[j] * diag_elem - A[i] * A[j][i]
                
        return 1/np.diag(A) * b

    def solve_ortogonalization(self):
        A = self.matrix.copy()
        b = self.rhs.copy()
        nrows, ncols = A.shape
        
        C = np.zeros_like(A, dtype = np.longdouble)
        d = np.zeros_like(b, dtype = np.longdouble)
        C[0] = A[0]/np.sqrt(np.dot(A[0], A[0]))
        d[0] = b[0]/np.sqrt(np.dot(A[0], A[0]))
        
        for k in range(1, nrows):
            ort_ccomp = 0
            ort_dcomp = 0
            for m in range(k):
                ort_ccomp += np.dot(A[k], C[m]) * C[m]
                ort_dcomp += np.dot(A[k], C[m]) * d[m]
            
            c_ = A[k] - ort_ccomp
            norm = np.sqrt(np.dot(c_, c_))
            
            C[k] = c_/norm
            d[k] = (b[k] - ort_dcomp)/norm
        
        return C.T @ d
    
    def solve_seidel(self, x_0, iter_num):
        A = self.matrix.copy()
        b = self.rhs.copy()
        nrows, ncols = A.shape
        
        L = np.tril(A, -1)
        D = np.diag(np.diag(A))
        U = np.triu(A, 1)
        
        x = x_0
        for i in range(iter_num):
            x = -np.linalg.inv(L + D) @ (U @ x - b)
        return x
            