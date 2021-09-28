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
    f_bins = np.arange(5, 8)
    freqs = bins[f_bins]
    grid = np.linspace(0, 60, n_grid) * pi/180
    id_th = [10,30,50]
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
    f_0 = 5
    
    ifocus = np.asarray([ # implicitly focus
        interpolate(bins, f_i, f_0, n_sensors, measurements[i], grid)
        for i, f_i in enumerate(f_bins)
    ])
    A_f0 = manifold(bins[f_0], grid, sensors)
    sup = solve_mmv(ifocus.T, A_f0, err=1e-3 * np.sqrt(n_freq))
    
    # compare w/ truth
    rmatrix = manifold(bins[f_0], theta, sensors)
    suppp = np.array([3, 10, 20, 25])
    true = np.asarray([
        rmatrix @ signal[i] for i in range(n_freq)
    ])
    A_f0 = manifold(bins[f_0], grid, sensors)
    sup = solve_mmv(true.T, A_f0, err=1e-1 * np.sqrt(n_freq * n_sensors))

    print(f"NMSE (Ours): {linalg.norm(true - ifocus) / linalg.norm(true)}")
    plt.stem(grid * 180/pi, np.abs(linalg.norm(sup, axis=0)), label=f'Ours')
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == '__main__':
    main()

