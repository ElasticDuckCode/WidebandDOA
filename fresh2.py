#!/usr/bin/env python3

import time
from multiprocessing import Pool, Process, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, random
from scipy import linalg

from tqdm import trange

from utils import manifold, manifold_tensor, solve_mmv, fill_hankel_grid


#def apply_focus(bins, f_bin, f_0, measurement, grid, n_sensors):
#    st = time.time()
#    #m = extrapolate(bins, f_bin, f_0, measurement, grid, step=2)
#    m = interpolate(bins, f_bin, f_0, n_sensors, measurement, grid)
#    et = time.time() - st
#    print("Focused {} to {} in {:.3f}s".format(f_bin, f_0, et))
#    return m

def interpolate(measurement, bins, grid, factor):
    if factor == 1:
        return measurement
    n_sensors = measurement.size
    fmeas = np.zeros(factor*(n_sensors-1) + 1, dtype=complex)
    fmeas[::factor] = measurement
    n_hankel = fmeas.size // 2 + 1
    H = linalg.hankel(fmeas[:n_hankel], fmeas[n_hankel-1:])
    A = manifold(bins[1], grid, np.arange(n_hankel))
    H_fill = fill_hankel_grid(H, A, A)
    results = np.concatenate([H_fill[:, 0], H_fill[-1, 1:]])
    return results

def extrapolate(results, bins, grid, n_sensors, f_0, step=1):
    highest = f_0 * (n_sensors - 1)
    need = highest - len(results)
    if need == 0:
        return results
    for i in range(step):
        results = np.append(results, np.zeros(need//step))
        n_hankel = results.size // 2 + 1
        H = linalg.hankel(results[:n_hankel], results[n_hankel-1:])
        A = manifold(bins[1], grid, np.arange(n_hankel))
        H_fill = fill_hankel_grid(H, A, A)
        results = np.concatenate([H_fill[:, 0], H_fill[-1, 1:]])
    return results

if __name__ == "__main__":

    ''' Parameters '''
    n_bins = 16
    n_sensors = 10
    n_theta = 4
    n_grid = 100


    ''' Build Measurements '''
    sensors = np.arange(n_sensors)
    bins = 1/n_bins * np.arange(n_bins)
    f_bins = np.arange(5, 10)
    freqs = bins[f_bins]
    grid = np.linspace(0, 60, n_grid) * pi/180
    id_th = [10, 40, 70, 95]
    theta = grid[id_th]
    n_freq = len(freqs)
    signal = 1 + np.random.rand(n_freq, n_theta)
    matrix = manifold_tensor(freqs, theta, sensors)
    measurements = np.asarray([matrix[i] @ signal[i] for i in range(n_freq)])


    ''' Focus Measurements to Low Frequency Manifolds (Our Technique) '''
    f_0 = 10
    #fmeasure = apply_focus(bins, f_bins[0], f_0, measurements[0], grid, n_sensors)

    
    results = []
    for i in trange(n_freq):
        result = interpolate(measurements[i], bins, grid, f_bins[i])
        result = extrapolate(result, bins, grid, n_sensors, f_0, step=1)
        results.append(result)
    results = np.asarray(results)
    results = results[:, ::f_0]
    rmatrix = manifold(bins[f_0], theta, np.arange(results.shape[-1]))
    true = np.asarray([
       rmatrix @ signal[i] for i in range(n_freq)
    ])

    # cheat for verification
    eps = linalg.norm(true - results)
    print("MSE: {}".format(linalg.norm(true - results)**2 / len(true) / n_freq))
    plt.figure(1)
    plt.plot(true.T.real)
    plt.plot(results.T.real, '--')
    #plt.legend(f_bins)
    plt.tight_layout()

    #st = time.time()
    #print("{} CPU Cores".format(cpu_count()))
    #with Pool(cpu_count()) as p:
    #    results = p.starmap(apply_focus, [(bins, f_bins[i], f_0, measurements[i], grid, n_sensors) for i in range(n_freq)])
    #results = np.asarray(results)
    #et = time.time() - st
    #print("--- Focused in {:.3f}s ---".format(et))


    st = time.time()
    U, s, Vh = linalg.svd(results.T, full_matrices=True)
    U2 = U[:, n_theta:]
    shared_matrix = manifold(bins[f_0], grid, sensors) 
    proj = linalg.norm(U2.conj().T @ shared_matrix, axis=0)
    pseudospectrum = 1/proj
    pseudospectrum /= pseudospectrum.max()
    x = pseudospectrum

    #st = time.time()
    #shared_matrix = manifold(bins[f_0], grid, sensors) 
    #x = solve_mmv(results.T, shared_matrix, err=2 * eps * np.sqrt(n_freq * n_sensors))
    #et = time.time() - st
    #print("--- Solved MMV in {:.3f}s ---".format(et))

    plt.figure(2)
    #plt.stem(grid * 180/pi, linalg.norm(x, axis=0))
    plt.plot(grid * 180/pi, x)
    plt.vlines(theta * 180/pi, ymin=0.0, ymax=plt.ylim()[1], linestyles='--', colors='k')
    plt.tight_layout()
    plt.show()

