#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, random, fft
from scipy import linalg

from utils import manifold, manifold_tensor, solve_l1, solve_mmv, get_largest_k_peaks
from focus import interpolate, focussing_matrix_rss, dynamic_dictionary


def main() -> int:
    n_bins = 32
    n_sensors = 15
    n_theta = 3
    n_grid = 100

    ''' Build Measurements '''
    sensors = np.arange(n_sensors)
    bins = 1/n_bins * np.arange(n_bins)
    f_bins = np.arange(1, 4)
    freqs = bins[f_bins]
    grid = np.linspace(0, 60, n_grid) * pi/180
    id_th = [10, 30, 50]
    theta = grid[id_th]
    print("True DOAs:")
    print(theta)
    #theta = random.uniform(0, 60, size=n_theta) * pi/180
    #theta = np.asarray([15, 30]) * pi/180
    #theta = random.choice(np.linspace(-20, 20, 1000), size=n_theta, replace=False) * pi/180
    n_freq = len(freqs)
    signal = 1+np.random.rand(n_freq, n_theta)
    #signal = np.ones([n_freq, n_theta])
    matrix = manifold_tensor(freqs, theta, sensors)
    measurements = np.asarray([matrix[i] @ signal[i] for i in range(n_freq)])
    
    ''' Focus Measurements to Low Frequency Manifolds (Our Technique) '''
    f_0 = 1
    
    ifocus = np.asarray([ # implicitly focus
        interpolate(bins, f_i, f_0, n_sensors, measurements[i], grid)
        for i, f_i in enumerate(f_bins)
    ])
    #mfocus = ifocus[1]
    A_f0 = manifold(bins[f_0], grid, sensors)
    #sup = solve_mmv(ifocus.T, A_f0, err=1e-3 * np.sqrt(n_freq))
    
    # compare w/ truth
    rmatrix = manifold(bins[f_0], theta, sensors)
    true = np.asarray([
        rmatrix @ signal[i] for i in range(n_freq)
    ])
    #A_f0 = manifold(bins[f_0], grid, sensors)
    #sup = solve_mmv(true.T, A_f0, err=1e-1 * np.sqrt(n_freq * n_sensors))

    print(f"NMSE (Ours): {linalg.norm(true - ifocus) / linalg.norm(true)}")
    # plt.stem(grid * 180/pi, np.abs(linalg.norm(sup, axis=0)), label=f'Ours')
    # plt.tight_layout()
    # plt.show()

    
    # #''' Run Dynamic Dictionary '''
    # f_0 = f_bins[-2]
    # A_f0 = matrix[-2]
    # A_f0 = manifold(bins[f_0], grid, sensors)
    # print(A_f0.shape)
    # A_fi = np.asarray([
    #    manifold(freqs[i], grid, sensors)
    #    for i in range(len(freqs)) if freqs[i] != bins[f_0]
    # ])
    # print(A_fi.shape,len(freqs))
    # sup, sups, efocus = dynamic_dictionary(measurements, A_fi, A_f0, bins[f_0], freqs, grid, sensors, n_theta, it=2)

    # # compare w/ truth
    # rmatrix = manifold(bins[f_0], theta, sensors)
    # true = np.asarray([
    #    rmatrix @ signal[i] for i in range(n_freq) if f_bins[i] != f_0
    # ])
    # print(f"NMSE (DD): {linalg.norm(true - efocus) / linalg.norm(true)}")
    # #print(f"{theta * 180/pi = }")
    # plt.figure()
    # for i, sup in enumerate(sups):
    #    plt.plot(grid * 180/pi, np.abs(sup), linewidth=1.5, label=f'DD-F it-{i}')
    # plt.vlines(theta * 180/pi, ymin=0.0, ymax=plt.ylim()[1], linestyles='--', colors='k')
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.show()


    #''' Run Dynamic Dictionary '''
    #f_0 = f_bins[n_freq-1]
    #A_f0 = matrix[n_freq-1]
    #A_f0 = manifold(bins[f_0], grid, sensors)
    #print(A_f0.shape)
    #A_fi = np.asarray([
    #   manifold(freqs[i], grid, sensors)
    #   for i in range(len(freqs)) if freqs[i] != bins[f_0]
    #])
    #print(A_fi.shape,len(freqs))
    #sup, sups, efocus = dynamic_dictionary(measurements, A_fi, A_f0, bins[f_0], freqs, grid, sensors, n_theta, it=3)

    ## compare w/ truth
    #rmatrix = manifold(bins[f_0], theta, sensors)
    #true = np.asarray([
    #   rmatrix @ signal[i] for i in range(n_freq) if f_bins[i] != f_0
    #])
    #print(f"NMSE (DD): {linalg.norm(true - efocus) / linalg.norm(true)}")
    ##print(f"{theta * 180/pi = }")
    #plt.figure()
    #for i, sup in enumerate(sups):
    #   plt.plot(grid * 180/pi, np.abs(sup), linewidth=1.5, label=f'DD-F it-{i}')
    #plt.vlines(theta * 180/pi, ymin=0.0, ymax=plt.ylim()[1], linestyles='--', colors='k')
    #plt.legend(loc='upper left')
    #plt.tight_layout()
    #plt.show()


    plt.figure() # Implicit
    plt.subplot(121)
    plt.plot(true.T.real)
    plt.plot(ifocus.T.real, '--')
    plt.grid()
    plt.xlabel('Sensor')
    plt.title('Real')
    plt.subplot(122)
    plt.plot(true.T.imag)
    plt.plot(ifocus.T.imag, '--')
    plt.xlabel('Sensor')
    plt.title('Imag')
    plt.grid()
    plt.tight_layout()
    plt.show()

    #plt.figure() # RSS
    #plt.subplot(121)
    #plt.plot(true[0].T.real)
    #plt.plot(efocus[0].T.real, '--')
    #plt.grid()
    #plt.xlabel('Sensor')
    #plt.title('Real')
    #plt.subplot(122)
    #plt.plot(true[0].T.imag)
    #plt.plot(efocus[0].T.imag, '--')
    #plt.xlabel('Sensor')
    #plt.title('Imag')
    #plt.grid()
    #plt.tight_layout()
    #plt.show()

    #print(efocus.shape)
    return 0


if __name__ == '__main__':
    main()

