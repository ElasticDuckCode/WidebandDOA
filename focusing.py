#!/usr/bin/env python3
import numpy as np

from utils import manifold

def focussing_matrix_rss(A_fi, A_f0):
    '''
    Compute the RSS focusing matrix which would convert array manifolds
    in A(f_i) to A(f_0) according to Hung's and Kaveh's method proposed in
    1988, "Focussing Matricies for Coherent Signal-Subspace Processing"

    https://doi.org/10.1109/29.1655

    T(f_i) = V(f_i) U(f_i)^H

    where U, V come from the SVD of M = A(f_i)A(f_0)^H
    '''
    M = A_fi @ A_f0.conj().T
    U, _, VH = np.linalg.svd(M)
    return VH.conj().T @ U.conj().T

if __name__ == '__main__':
    A = manifold(1, np.asarray([0, 10]) * np.pi/180, np.arange(3))
    B = manifold(10, np.asarray([0, 10]) * np.pi/180, np.arange(3))
    T = focussing_matrix_rss(A, B)
    print(f"{T = }")
    print(f"{A = }")
    print(f"{B = }")
    print(f"{T @ A = }")


