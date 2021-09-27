#!/usr/bin/env python3

import numpy as np
from numpy import random
from numpy import pi
from scipy import linalg, fft
from matplotlib import pyplot as plt
from utils import solve_l1


def manifold(f_i, doa_list, sensor_list, d = 1):
    return np.exp(2*np.pi*1j * f_i * d * np.outer(sensor_list, np.sin(doa_list)))


def manifold_tensor(f_list, doa_list, sensor_list, d=1):
    tensor = np.zeros([len(f_list), len(sensor_list), len(doa_list)], dtype=complex)
    for i, f_i in enumerate(f_list):
        tensor[i] = manifold(f_i, doa_list, sensor_list, d)
    return tensor


def music(R_yy, sensors, n_sources, ref_f, N):
    U, _, _ = linalg.svd(R_yy)
    U2 = U[:, n_sources:]
    G = manifold(ref_f, np.linspace(0, 60, N) * np.pi/180, sensors)
    projection = linalg.norm(U2.conj().T @ G, axis=0)
    pseudospectrum = 1 / projection
    pseudospectrum /= pseudospectrum.max()
    return pseudospectrum


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


def main():
    n_sensors = 15
    n_inner = n_sensors // 2
    n_outer = n_sensors - n_inner
    sensors = np.arange(n_sensors)
    n_bins = 64
    f_grid = 1/n_bins * np.arange(n_bins)
    frequencies = np.arange(700, 10_000)
    #frequencies = np.asarray([3_000])
    n_frequencies = len(frequencies)
    doas = np.asarray([10, 20]) * np.pi/180
    n_doa = len(doas)
    n_focus_grid = 100

    ''' Create Synthetic Measurements '''
    signals =  np.sign(np.random.randn(n_frequencies, n_doa)) + random.rand(n_frequencies, n_doa)
    matrix = manifold_tensor(f_grid[frequencies], doas, sensors)
    measurements = np.zeros([n_bins, n_sensors], dtype=complex)
    for i, indx in enumerate(frequencies):
        measurements[indx] = matrix[i] @ signals[i]
    #measurements = fft.ifft(measurements, axis=0, norm='forward')

    ''' Focus Manifolds ''' 
    focussed_matrix = np.zeros([n_frequencies, n_sensors, n_sensors], dtype=complex)
    focussed_meas = np.zeros_like(measurements)
    ref_indx = -1


    ref_dict = manifold(f_grid[frequencies[ref_indx]], np.linspace(0, 60, n_focus_grid) * pi/180, sensors)
    print(np.linspace(0, 60, n_focus_grid))
    for i, indx in enumerate(frequencies):
        f_dict = manifold(f_grid[indx], np.linspace(0, 60, n_focus_grid) * pi/180, sensors)
        focussed_matrix[i] = focussing_matrix_rss(f_dict, ref_dict)
        focussed_meas[indx] = focussed_matrix[i] @ measurements[indx]
    
    asdf = focussed_matrix[0] @ matrix[0]
    #print(matrix.shape)
    print(np.abs(matrix[ref_indx] - asdf))

    #plt.plot(np.abs(focussed_meas[3_000]))
    #plt.plot(np.abs(measurements[3_000]))
    #plt.show()

    ''' Solver w/ Averaged Measurements'''
    f_meas = focussed_meas.sum(axis=0)
    sigs = signals.sum(axis=0)
    true_meas = matrix[ref_indx] @ sigs
    plt.plot(f_meas)
    plt.plot(true_meas)
    plt.show()
    dict = manifold(f_grid[frequencies[ref_indx]], np.linspace(0, 60, 1000) * pi/180, sensors)
    u_g = solve_l1(f_meas, dict, err=linalg.norm(f_meas - true_meas)*1.1)
    #f_h = solve_l1(measurements[frequencies[-1]], dict, err=1e-3 * np.sqrt(n_sensors))
    plt.plot(np.linspace(0, 60, 1000), u_g)
    plt.show()


    return 0


if __name__ == '__main__':
    main()
