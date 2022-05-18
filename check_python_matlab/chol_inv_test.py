# Finally, this works -
# https://download.lfd.uci.edu/pythonlibs/x6hvwk7i/numpy-1.22.3+mkl-cp39-cp39-win_amd64.whl
# MUST be run with mkl_numpy and no_mkl_scipy

import numpy as np
import scipy
from scipy import linalg

from scipy.io import savemat
from scipy.io import loadmat

from pprint import pprint
from math import sqrt

def cholesky_decomposition(M):
    """
    https://www.intel.com/content/www/us/en/developer/articles/technical/using-intel-distribution-for-python-with-anaconda.html
    conda activate idp
    conda deactivate

    https://github.com/TayssirDo/Cholesky-decomposition
    Compute the cholesky decomposition of a SPD matrix M.
    :param M: (N, N) real valued matrix.
    :return: R: (N, N) upper triangular matrix with positive diagonal entries if M is SPD.
    
    https://mathoverflow.net/questions/153926/perturbation-of-cholesky-decomposition-for-matrix-inversion    
    http://www.cs.umd.edu/~oleary/tr/tr4807.pdf
    https://mcsweeney90.github.io/files/modified-cholesky-decomposition-and-applications.pdf
    https://nhigham.com/2020/12/22/what-is-a-modified-cholesky-factorization/
    
    https://stackoverflow.com/questions/39574924/why-is-inverting-a-positive-definite-matrix-via-cholesky-decomposition-slower-th
    """

    A = np.copy(M)
    n = A.shape[0]
    R = np.zeros_like(A)

    for k in range(n):
        #print(f"{k}: {A[k, k]} {M[k,k]}")
        if A[k, k] < 0:
            A[k, k] = abs(A[k, k])
        R[k, k] = sqrt(A[k, k])
        R[k, k + 1:] = A[k, k + 1:] / R[k, k]
        for j in range(k + 1, n):
            A[j, j:] = A[j, j:] - R[k, j] * R[k, j:]

    return R

def main():
    # must be run with mkl_numpy and no_mkl_scipy
    
    print("scipy:")
    print(scipy.show_config())
    print("\nnumpy:")
    print(np.show_config())
    
    mat = loadmat('A_L_R_iL.mat',squeeze_me=True)
    A = mat['A']
    mL = mat['L']
    mR = mat['R']
    miL = mat['iL']

    L = np.linalg.cholesky(A)
    iL = np.linalg.inv(L)

    L_no_mkl = scipy.linalg.inv(L)
    iL_no_mkl = scipy.linalg.inv(L)

    d = np.max(np.abs((L_no_mkl-mL)))
    print(f"\nnorm(scipy_no_mkl.linalg.cholesky() - Matlab chol()) = {d}")
    d = np.max(np.abs((iL_no_mkl-miL)))
    print(f"norm(scipy_no_mkl.linalg.inv() - Matlab inv()) = {d}\n")

    d = np.max(np.abs((mL-L)))
    print(f"norm(np_mkl.linalg.cholesky() - Matlab chol()) = {d}")
    d = np.max(np.abs((iL-miL)))
    print(f"norm(np_mkl.linalg.inv() - Matlab inv()) = {d}\n")
  

if __name__ == '__main__':
  main()