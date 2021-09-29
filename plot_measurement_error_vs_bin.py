#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def measurement_error_vs_bin() -> int:
    
    n_monte_carlo = 100
    n_sensors = 10

    results_prop = np.asarray([
        np.load(f"results/mse_vs_freq/MSE_IFOCUS_{i}.npy")
        for i in range(n_monte_carlo)
    ])

    results_rss = np.asarray([
        np.load(f"results/mse_vs_freq/MSE_RSS_{i}.npy")
        for i in range(n_monte_carlo)
    ])

    results_dd = np.asarray([
        np.load(f"results/mse_vs_freq/MSE_DD_it2_{i}.npy")
        for i in range(n_monte_carlo)
    ])
        
    plt.rc('font', size=14) #controls default text size
    #plt.rc('axes', titlesize=10) #fontsize of the title
    #plt.rc('axes', labelsize=10) #fontsize of the x and y labels
    #plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
    #plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
    plt.rc('legend', fontsize=11) #fontsize of the legend

    prop_db = 20 * np.log10(results_prop.mean(axis=0) / n_sensors)
    rss_db = 20 * np.log10(results_rss.mean(axis=0) / n_sensors)
    dd_db = 20 * np.log10(results_dd.mean(axis=0) / n_sensors)
    f_bins = np.arange(3, 10)
    plt.plot(f_bins, prop_db, label='Proposed', lw=2, color='b', marker='o', ms=7, fillstyle='none')
    plt.plot(f_bins, dd_db, label='DD', color='r', lw=2, marker='D', ms=7, fillstyle='none')
    plt.plot(f_bins, rss_db, label='RSS (True DOA)', lw=2, color='k', marker='s', ms=7, fillstyle='none')
    plt.xlabel('Frequency Bin')
    plt.ylabel('MSE / (dB)')
    plt.title('MSE Focussing to Bin 10')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    measurement_error_vs_bin()
