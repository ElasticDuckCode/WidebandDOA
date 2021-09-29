#!/usr/bin/env python3

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from utils import manifold, fill_hankel_grid, solve_l1, get_largest_k_peaks


def dynamic_dictionary(measurements, matricies, ref_matrix, f_0, freqs, grid, sensors, n_theta, err=1e-2, it=2, r=1e-5):
    '''
    Solve Wideband DOA problem using the Dynamic Dictionary method
    presented in 2019 "Low Complexity DOA Estimation for Wideband Off-Grid 
    Sources Based on Re-Focused Compressive Sensing With Dynamic Dictionary"

    https://doi.org/10.1109/JSTSP.2019.2932973

    This function will solve the ULA, on-grid version, meaning B(l_r, theta)
    is our original manifold and step 2 is omitted.
    '''

    sups = []

    # Focus w/ RSS initially using coarse grid
    n_grid=100
    #print(ref_matrix[:,0:2:].shape)
    idd = np.arange(50)
    idd*=2
    #print(idd)
    temp = matricies[0,:,idd]
    #print(matricies[0,:,idd].shape)

    focus_matrix = np.asarray([
        focussing_matrix_rss(A_fi=matricies[i,:,idd].T, A_f0=ref_matrix[:,idd])
        for i in range(matricies.shape[0])
    ])

    # Focus Measurements
    efocus = np.asarray([
        focus_matrix[i] @ measurements[i]
        for i in range(focus_matrix.shape[0]) if freqs[i] != f_0
    ])
    mfocus = efocus.mean(axis=0)

    # Solve Initial L1
    A_f0 = manifold(f_0, grid, sensors)
    sup = solve_l1(mfocus, A_f0, err=err)
    sups.append(sup)

    # Initial DOA Estimates
    theta_k = np.asarray(grid[get_largest_k_peaks(sup, k=n_theta)])
    #print(theta_k)
    #plt.figure()
    #plt.plot(mfocus)
    # plt.figure()
    
    # plt.stem(np.linspace(0, 60, n_grid) * np.pi/180,sup)
    
    #print(theta_k * 180/np.pi)
    r=3/180*np.pi
    for i in range(it):
        #print("This is r")
        #print(r)
        A_f0 = manifold(f_0, theta_k, sensors)
        A_fi = np.asarray([
            manifold(freqs[i], theta_k, sensors)
            for i in range(len(freqs))
        ])
        focus_matrix = np.asarray([
            focussing_matrix_rss(A_fi=A_fi[i], A_f0=A_f0)
            for i in range(A_fi.shape[0])
        ])
        efocus = np.asarray([
            focus_matrix[i] @ measurements[i]
            for i in range(measurements.shape[0]) if freqs[i] != f_0
        ])
        mfocus = efocus.mean(axis=0)
        A_f0 = manifold(f_0, grid, sensors)
        #print(A_f0.shape,grid)
        # plt.figure()
        # plt.plot(mfocus)
        sup = solve_l1(mfocus, A_f0, err=err)

        #print(sup)
        # plt.figure()
        # plt.stem(np.linspace(0, 60, n_grid) * np.pi/180,sup)
        # plt.show()
        sups.append(sup)
        theta_k = np.asarray(grid[get_largest_k_peaks(sup, k=n_theta)])
        #print(theta_k)
        theta_k = np.concatenate([
            theta_k - r, theta_k, theta_k + r
        ])
        r /= 2
        #print(theta_k * 180/np.pi)

    return sup, sups, efocus


def focussing_matrix_rss(A_fi, A_f0):
    '''
    Compute the RSS focusing matrix which would convert array manifolds
    in A(f_i) to A(f_0) according to Hung's and Kaveh's method proposed in
    1988, "Focussing Matricies for Coherent Signal-Subspace Processing"

    https://doi.org/10.1109/29.1655

    T(f_i) = V(f_i) U(f_i)^H

    where U, V come from the SVD of C = A(f_i)A(f_0)^H
    '''
    U, s, VH = linalg.svd(A_fi @ A_f0.conj().T)
    return VH.conj().T @ U.conj().T


def interpolate(bins, ii, i0, n_meas, meas, doa_grid):
    if i0 == ii:
        return meas # no need to extrapolate
    gcd = np.gcd(ii, i0)
    factor = ii // gcd
    syn = np.zeros(factor*(len(meas)-1)+1, dtype=complex)
    syn[::factor] = meas
    n_hankel = len(syn) // 2 + 1
    H = linalg.hankel(syn[:n_hankel], syn[n_hankel-1:])
    AL = manifold(bins[i0], doa_grid, np.arange(H.shape[0]))
    AR = manifold(bins[i0], doa_grid, np.arange(H.shape[1]))
    H_fill = fill_hankel_grid(H, AL, AR)
    syn = np.concatenate([H_fill[:, 1], H_fill[-1, 1:]])
    sample = i0 // gcd
    new = syn[::sample]
    new = new[:n_meas]
    return new


def extrapolate(bins, ii, i0, meas, doa_grid, step=1):
    highest_i0 = (len(meas)-1) * i0
    highest_ii = int(np.ceil(highest_i0 / ii) * ii)
    n_syn_meas = highest_ii // ii + 1
    need_ii = n_syn_meas - len(meas)
    syn = meas
    for i in range(need_ii // step + 1):
        syn = np.append(syn, np.zeros(step))
        M  = len(syn) // 2 + 1
        H = linalg.hankel(syn[:M], syn[M-1:])
        AL = manifold(bins[i0], doa_grid, np.arange(H.shape[0]))
        AR = manifold(bins[i0], doa_grid, np.arange(H.shape[1]))
        H_fill = fill_hankel_grid(H, AL, AR)
        syn = np.concatenate([H_fill[:, 0], H_fill[-1, 1:]])
    return syn
