#!/usr/bin/env python3
import numpy as np
from numpy import random
from scipy import linalg, fft
from matplotlib import pyplot as plt


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
    n_sensors = 32
    n_inner = n_sensors // 2
    n_outer = n_sensors - n_inner
    inner = np.arange(n_inner)
    outer = np.arange(1, n_outer+1)*n_inner
    sensors = np.concatenate([inner, outer])
    sensors = np.arange(n_sensors)
    n_bins = 100_000
    f_grid = 1/n_bins * np.arange(n_bins)
    #frequencies = np.arange(60_000, 70_000)
    frequencies = np.asarray([30_000])
    n_frequencies = len(frequencies)
    doas = np.asarray([40]) * np.pi/180
    n_doa = len(doas)

    ''' Create Synthetic Measurements '''
    signals = np.sign(random.randn(n_frequencies, n_doa)) * (1 + random.rand(n_frequencies, n_doa))
    matrix = manifold_tensor(f_grid[frequencies], doas, sensors)
    measurements = np.zeros([n_bins, n_sensors], dtype=complex)
    for i, indx in enumerate(frequencies):
        measurements[indx] = matrix[i] @ signals[i]
    measurements = fft.ifft(measurements, axis=0, norm='forward')

    ''' Split into Groups for Auto-Correlations '''
    n_groups = 10
    n_new_bins = n_bins // n_groups
    measurement_groups = np.zeros([n_groups, n_new_bins, n_sensors], dtype=complex)
    for i in range(n_groups):
        measurement_groups[i, ...] = measurements[i*n_new_bins:(i+1)*n_new_bins]
    measurement_groups = fft.fft(measurement_groups, axis=1, norm='forward')

    auto_correlations = np.zeros([n_new_bins, n_sensors, n_sensors], dtype=complex)
    for i in range(n_new_bins):
        Y_i = measurement_groups[:, i, :].T
        auto_correlations[i] = 1/n_groups * Y_i @ Y_i.conj().T

    ''' Arbitrarily Pick Bin 0 to Focus to By Cheating'''
    focussed_auto_correlations = np.zeros_like(auto_correlations)
    f_matrices = np.zeros([n_new_bins, n_sensors, n_sensors], dtype=complex)
    A_ref = manifold(f_grid[frequencies[-1]], doas, sensors)
    for i in range(n_new_bins):
        f_matrices[i] = focussing_matrix_rss(manifold(f_grid[i], doas, sensors), A_ref)
        focussed_auto_correlations[i] = f_matrices[i] @ auto_correlations[i] @ f_matrices[i].conj().T

    ''' Blindly do Average MUSIC '''
    n_spec = 1000
    spectrums = np.zeros([n_new_bins, n_spec], dtype=complex)
    for i in range(n_new_bins):
        spectrums[i] = music(focussed_auto_correlations[i], sensors, n_doa, f_grid[frequencies[-1]], n_spec)

    spec = music(auto_correlations[3000], sensors, n_doa, f_grid[3000], n_spec)
    plt.plot(np.linspace(0, 60, n_spec), np.abs(spec))
    plt.show()

    return 0


if __name__ == '__main__':
    main()
