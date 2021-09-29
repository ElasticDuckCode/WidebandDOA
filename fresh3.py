#!/usr/bin/env python3

import time
import json
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, random
from scipy import linalg

from tqdm import trange

from utils import manifold, manifold_tensor, fill_hankel_grid_noise
from focus import focussing_matrix_rss, dynamic_dictionary


def apply_focus(bins, grid, measurement, f_i, f_0, n_sensors, err=0):
    st = time.time()
    #n_sensors = measurement.size
    result = interpolate(measurement, bins, grid, f_i, err)
    result = extrapolate(result, bins, grid, n_sensors, f_0, step=1, err=err)
    result = result[::f_0]
    et = time.time() - st
    print("Focused {} to {} in {:.3f}s".format(f_i, f_0, et))
    return result


def interpolate(measurement, bins, grid, factor, err=0):
    if factor == 1:
        return measurement
    n_sensors = measurement.size
    fmeas = np.zeros(factor*(n_sensors-1) + 1, dtype=complex)
    fmeas[::factor] = measurement
    n_hankel = fmeas.size // 2 + 1
    H = linalg.hankel(fmeas[:n_hankel], fmeas[n_hankel-1:])
    A = manifold(bins[1], grid, np.arange(n_hankel))
    H_fill = fill_hankel_grid_noise(H, A, A, err=err)
    results = np.concatenate([H_fill[:, 0], H_fill[-1, 1:]])
    return results


def extrapolate(results, bins, grid, n_sensors, f_0, step=1, err=0):
    highest = f_0 * (n_sensors - 1)
    need = highest - len(results)
    if need == 0:
        return results
    for i in range(step):
        results = np.append(results, np.zeros(need//step))
        n_hankel = results.size // 2 + 1
        H = linalg.hankel(results[:n_hankel], results[n_hankel-1:])
        A = manifold(bins[1], grid, np.arange(n_hankel))
        H_fill = fill_hankel_grid_noise(H, A, A, err=err)
        results = np.concatenate([H_fill[:, 0], H_fill[-1, 1:]])
    return results


if __name__ == "__main__":

    ''' Parameters '''
    n_bins = 16
    n_sensors = 10
    n_theta = 4
    n_grid = 100

    sensors = np.arange(n_sensors)
    bins = 1/n_bins * np.arange(n_bins)
    f_bins = np.arange(3, 10)
    freqs = bins[f_bins]
    grid = np.linspace(0, 60, n_grid) * pi/180
    id_th = [10, 40, 70, 95]
    theta = grid[id_th]
    n_freq = len(freqs)
    f_0 = 10
    matrix = manifold_tensor(freqs, theta, sensors)
    n_monte_carlo = 100

    ''' Store Settings '''
    settings = {
        'n_monte_carlo': n_monte_carlo,
        'n_bins': n_bins,
        'n_sensors': n_sensors,
        'n_theta': n_theta,
        'n_grid' : n_grid,
        'f_bins' : f_bins.tolist(),
        'theta': theta.tolist(),
        'n_freq': n_freq,
        'f_0': f_0
    }
    tf = open("results/settings.json", "w")
    json.dump(settings, tf, indent=4)
    tf.close()

    ''' Run Monte Carlo Experiments '''
    #snrs = np.logspace(-2.0, 1.5, 8)
    #print(1/snrs)
    sigs = np.logspace(-2.0, 0.5, 8)
    print(sigs)
    
    MSE_prop = np.empty(0)
    MSE_rss = np.empty(0)
    MSE_dd = np.empty(0)
    for j, sig in enumerate(sigs):
        for i in trange(n_monte_carlo):
            ''' Build Measurements '''
            signal = 1 + random.rand(n_freq, n_theta)
            measurements = np.asarray([matrix[i] @ signal[i] for i in range(n_freq)])
            noise = np.sqrt(sigs[0] / 2) * np.random.uniform(-1, 1, size=(n_freq, n_sensors, 2)).view(np.complex128).squeeze()
            err = linalg.norm(noise)
            measurements += noise

            ''' Build True for Comparision '''
            rmatrix = manifold(bins[f_0], theta, sensors)
            true = np.asarray([
               rmatrix @ signal[i] for i in range(n_freq)
            ])

            ''' Focus Measurements to Low Frequency Manifolds (Our Technique) '''
            st = time.time()
            print("{} CPU Cores".format(cpu_count()))
            try:
                with Pool(cpu_count()) as p:
                    # Feeding in exactly the noise norm to get best performance
                    results = p.starmap(apply_focus, [(bins, grid, measurements[i], f_bins[i], f_0, n_sensors, err) for i in range(n_freq)])
            except:
                # WARNING, THIS MAKES IT SO YOU HAVE TO KEYBOARD INTERUP MANY TIMES
                print("ERROR: Failed to solve, skipping MC iteration...")
                continue 
            results = np.asarray(results)
            et = time.time() - st
            print("--- Focused in {:.3f}s (IFOCUS) ---".format(et))

            # SAVE MSE
            MSE = linalg.norm(true - results)
            MSE_prop = np.append(MSE_prop, MSE)
            print(f"MSE (IFOCUS): {MSE}")
            np.save(f"results/MSE_IFOCUS_sig{j}.npy", MSE_prop)

            #plt.figure()
            #plt.plot(true.T.real)
            #plt.plot(results.T.real, '--')
            #plt.tight_layout()
            #plt.show()

            ''' Focus Measurements w/ RSS (Using known DOA's)'''
            st = time.time()
            focus_matrix = np.asarray([
                focussing_matrix_rss(A_fi=matrix[i], A_f0=rmatrix)
                for i in range(n_freq)
            ])
            results = np.asarray([
                focus_matrix[i] @ measurements[i]
                for i in range(n_freq)
            ])
            et = time.time() - st
            print("--- Focused in {:.3f}s (RSS) ---".format(et))

            # SAVE MSE
            MSE = linalg.norm(true - results)
            MSE_rss = np.append(MSE_rss, MSE)
            print(f"MSE (Ideal RSS): {MSE}")
            np.save(f"results/MSE_RSS_sig{j}.npy", MSE_rss)

            #plt.figure()
            #plt.plot(true.T.real)
            #plt.plot(results.T.real, '--')
            #plt.tight_layout()
            #plt.show()

            ''' Focus Measurements w/ Dynamic Dictionary '''
            DD_it = 2
            A_f0 = manifold(bins[f_0], grid, sensors)
            A_fi = np.asarray([
               manifold(freqs[i], grid, sensors)
               for i in range(n_freq)
            ])
            sup, sups, results = dynamic_dictionary(measurements, A_fi, A_f0, bins[f_0], freqs, grid, sensors, n_theta, err=err, it=DD_it)
            print(results.shape)

            # SAVE MSE
            MSE = linalg.norm(true - results)
            MSE_dd = np.append(MSE_dd, MSE)
            print(f"MSE (DD): {MSE}")
            np.save(f"results/MSE_DD_it{DD_it}_sig{j}.npy", MSE_dd)

            #plt.figure()
            #plt.plot(true.T.real)
            #plt.plot(results.T.real, '--')
            #plt.tight_layout()
            #plt.show()
