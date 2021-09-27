#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, random, fft

    
def A(theta_list, f_i, N) -> np.ndarray:
    manifold = np.exp(-1j * 2*pi * f_i * np.outer(np.arange(N), np.sin(theta_list)))
    return manifold


def A_tensor(f_list, N, theta_list) -> np.ndarray:
    K = len(theta_list)
    F = len(f_list)
    A_tensor = np.zeros([F, N, K], dtype=complex)
    for i, f_i in enumerate(f_list):
        A_tensor[i] = A(theta_list, f_i, N)
    return A_tensor


def example() -> int:
    n_sensors = 100
    n_bins = 1000
    n_doa = 7
    n_snapshots = 1
    #angle = pi/2
    #doa_array = np.asarray([angle])
    doa_array = np.asarray([-60, -35, -15, 5, 30, 45, 60]) * pi/180
    #doa_array = np.asarray([-60, 0, 60]) * pi/180
    f_grid = 1/n_bins * np.arange(n_bins)
    #f_i = 1/100
    #signal = np.sin(pi * 2/1000 * np.arange(n_snapshots))[None]
    #signal = np.ones([1, n_snapshots])
    signal = random.rand(n_doa, n_snapshots) - 1/2
    measurements = np.zeros([n_sensors, n_snapshots, n_bins], dtype=complex)
    for i in range(n_bins):
        if i == 10 or i == 20:
            matrix = A(doa_array, f_grid[i], n_sensors)
            measurements[:, :, i] = matrix @ signal
    #measurements = fft.ifft(measurements, axis=-1, norm='forward')
    #measurement = np.sum(measurements, axis=-1)
    measurements = measurements.reshape(n_sensors, n_bins)

    plt.rcParams.update({'font.size': 14})
    plt.figure()
    plt.plot(measurements[0, :100].real, label='Sensor 0 Real')
    plt.plot(measurements[10, :100].real, label='Sensor 10 Real')
    plt.xlabel('Time')
    plt.title('Amplitude VS Time')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(color='#99AABB', linestyle=':')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_facecolor('#E0F0FF')
    plt.figure()
    plt.plot(measurements[:, 0].real, label='Time 0 Real')
    plt.plot(measurements[:, 10].real, label='Time 10 Real')
    plt.xlabel('Sensor')
    plt.title('Amplitude vs Sensor')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(color='#99AABB', linestyle=':')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_facecolor('#E0F0FF')
    plt.tight_layout()
    plt.show()
    return 0

def main() -> int:
    n_doas = 7
    n_bins = 1000
    n_sensors = 20
    n_snapshots = 1000
    f_carrier = 400
    bandwidth = 80
    freq_grid = 1/n_bins * fft.fftfreq(n_bins)
    doa_array = np.asarray([-60, -35, -15, 5, 30, 45, 60]) * pi/180

    # Get mask for possible frequency content of the signals on carriers
    freq_mask = np.zeros(n_bins, dtype=bool)
    freq_mask[f_carrier-bandwidth:f_carrier+bandwidth] = True
    
    # Signal have DOA's, with changing amplitude over snapshots
    signal = np.zeros([n_doas, n_snapshots])
    #signal = random.randn(n_doas, n_snapshots)
    signal[-1, :] = np.sin(2*pi * 1/100000 *n_snapshots)
    
    # for each frequency, there is an n_sensors x n_doas matrix
    manifold_matrix = A_tensor(freq_grid, n_sensors, doa_array)
    
    # measurements
    measurements = np.zeros([n_sensors, n_snapshots, n_bins], dtype=complex)
    for i in range(n_bins):
        if i == 400:
            measurements[:, :, i] = (manifold_matrix[i] @ signal)
    #for i in range(n_sensors):
    #    measurements[i] = ifft(measurements[i], axis=-1)
    measurements = np.sum(measurements, axis=-1)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(measurements[0, :].real)
    ax[0, 0].plot(measurements[0, :].imag)
    ax[0, 0].set_title('Sensor 0 over Time')
    ax[0, 0].set_xlabel('Snapshot Index')
    ax[0, 1].plot(measurements[5, :].real)
    ax[0, 1].plot(measurements[5, :].imag)
    ax[0, 1].set_title('Sensor 5 over Time')
    ax[0, 1].set_xlabel('Snapshot Index')
    ax[1, 0].stem(measurements[:, 0].real)
    ax[1, 0].set_title('All Sensors, Snapshot 0')
    ax[1, 0].set_xlabel('Sensor Index')
    ax[1, 1].stem(measurements[:, 70].real)
    ax[1, 1].set_title('All Sensors, Snapshot 70')
    ax[1, 1].set_xlabel('Sensor Index')
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    #main()
    example()

