#!/usr/bin/env python3

''' Python Modules '''
import sys

''' Computational Modules'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, random, fft
from scipy import linalg
from utils import fill_hankel_by_rank_minimization, manifold, manifold_tensor


def main() -> int:
    n_sensors = 15
    n_bins = 1000
    f_grid = 1/n_bins * np.arange(n_bins)
    frequencies = [10, 20]
    n_frequencies = len(frequencies)
    doas = np.asarray([0, 10, 20, 30]) * pi/180
    #doas = np.asarray([-60, -35, -15, 5, 30, 45, 60]) * pi/180
    n_doa = len(doas)
    
    '''
    Simulate Time-Domain Measurements
    '''
    signals = random.randn(n_frequencies, n_doa)
    #measurements = np.zeros([n_sensors, n_bins])
    matrix = manifold_tensor(f_grid[frequencies], doas, n_sensors)
    measurements = np.zeros([n_bins, n_sensors], dtype=complex)
    for i, indx in enumerate(frequencies):
        measurements[indx] = matrix[i] @ signals[i]
    measurements = fft.ifft(measurements, axis=0)
    
    '''
    Simulate Processing of Measurements
    '''
    n_syn_sensors = n_frequencies*n_sensors - n_frequencies + 1
    syn_measurements = np.zeros([n_frequencies, n_syn_sensors], dtype=complex)
    hankels = np.zeros([n_frequencies, n_syn_sensors//2 + 1, n_syn_sensors - n_syn_sensors//2], dtype=complex)

    # After taking 1000 time samples, need to computer DFT of measurements
    measurements = fft.fft(measurements, axis=0)
    
    # We must know which frequencies we are measuring prior
    syn_measurements[0][0:n_sensors] = measurements[frequencies[0]]
    syn_measurements[1][0:2*n_sensors:2] = measurements[frequencies[1]]

    hankels[0] = linalg.hankel(syn_measurements[0][:n_syn_sensors//2+1], syn_measurements[0][n_syn_sensors//2:])
    hankels[1] = linalg.hankel(syn_measurements[1][:n_syn_sensors//2+1], syn_measurements[1][n_syn_sensors//2:])

    # Search on DOA grid
    doa_grid = np.arange(-60, 60, 1) * pi/180
    shared_matrix = manifold(f_grid[frequencies[0]], doa_grid, np.arange(n_syn_sensors))
    print(shared_matrix.shape[:hankels.shape[1]])
    filled_hankels = np.zeros_like(hankels)
    filled_hankels[0] = fill_hankel_by_rank_minimization(hankels[0], shared_matrix[:hankels.shape[1]])
    filled_hankels[1] = fill_hankel_by_rank_minimization(hankels[1], shared_matrix[:hankels.shape[1]])

    syn_measurement0 = np.zeros(n_syn_sensors, dtype=complex)
    syn_measurement0 = np.concatenate([filled_hankels[0][:, 0], filled_hankels[0][-1, 1:]])

    syn_measurement1 = np.zeros(n_syn_sensors, dtype=complex)
    syn_measurement1 = np.concatenate([filled_hankels[1][:, 0], filled_hankels[1][-1, 1:]])

    _, s0, _ = linalg.svd(filled_hankels[0])
    _, s1, _ = linalg.svd(filled_hankels[1])

    ''' Compare to Truths and Plot '''
    true_matrix = manifold(f_grid[frequencies[0]], doas, np.arange(n_syn_sensors))
    true_measurement0 = true_matrix @ signals[0]
    true_hankel0 = linalg.hankel(true_measurement0[:n_syn_sensors//2+1], true_measurement0[n_syn_sensors//2:])
    true_measurement1 = true_matrix @ signals[1]
    true_hankel1 = linalg.hankel(true_measurement1[:n_syn_sensors//2+1], true_measurement1[n_syn_sensors//2:])

    print("MSE Measurement 0: {}".format(1/n_syn_sensors * linalg.norm(syn_measurement0 -true_measurement0)))
    print("MSE Measurement 1: {}".format(1/n_syn_sensors * linalg.norm(syn_measurement1 -true_measurement1)))
    
    fig, ax = plt.subplots(1,3)
    ax[0].axis('off')
    ax[0].imshow(np.abs(hankels[0]), cmap='magma', vmin=0)
    ax[0].set_title('Measurements 0')
    ax[1].axis('off')
    ax[1].imshow(np.abs(filled_hankels[0]), cmap='magma', vmin=0)
    ax[1].set_title('Predicted')
    ax[2].axis('off')
    ax[2].imshow(np.abs(true_hankel0), cmap='magma', vmin=0)
    ax[2].set_title('Truth')
    plt.suptitle('Hankel 0')
    plt.tight_layout()

    fig, ax = plt.subplots(1,3)
    ax[0].axis('off')
    ax[0].imshow(np.abs(hankels[1]), cmap='magma', vmin=0)
    ax[0].set_title('Measurements 1')
    ax[1].axis('off')
    ax[1].imshow(np.abs(filled_hankels[1]), cmap='magma', vmin=0)
    ax[1].set_title('Predicted')
    ax[2].axis('off')
    ax[2].imshow(np.abs(true_hankel1), cmap='magma', vmin=0)
    ax[2].set_title('Truth')
    plt.suptitle('Hankel 1')
    plt.tight_layout()

    plt.figure()
    plt.plot(np.abs(true_measurement0 - syn_measurement0), linewidth=2, label='0')
    plt.plot(np.abs(true_measurement1 - syn_measurement1), linewidth=2, label='1')
    plt.xlabel('Sensor Index')
    plt.ylabel('Absolute Error')
    plt.title('Predicted VS True Measurement 0 Error')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(s0, label='Singular Values 0')
    plt.plot(s1, label='Singular Values 1')
    plt.legend()
    plt.tight_layout()

    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
