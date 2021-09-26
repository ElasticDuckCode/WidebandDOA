#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import scipy.linalg as linalg
import scipy.signal as sps


def manifold(f_i, doa_list, sensor_list, d = 1):
    return np.exp(2*np.pi*1j * f_i * d * np.outer(sensor_list, np.sin(doa_list)))


def manifold_tensor(f_list, doa_list, N):
    tensor = np.zeros([len(f_list), N, len(doa_list)], dtype=complex)
    for i, f_i in enumerate(f_list):
        tensor[i] = manifold(f_i, doa_list, np.arange(N))
    return tensor


def fill_hankel_by_rank_minimization(hankel_measurements: np.ndarray, manifold_matrix: np.ndarray,
        max_iter: int = 1, gamma: float = 0.1, err: float = 0) -> np.ndarray:
    sensor_count, grid_size = manifold_matrix.shape
    predicted_signal = cp.Variable(shape=grid_size)
    hankel_indx = np.nonzero(hankel_measurements)
    objective = cp.Minimize(cp.norm1(predicted_signal))
    hankel_matrix = manifold_matrix @ cp.diag(predicted_signal) @ manifold_matrix.T
    constraint = [
        #cp.norm2(hankel_matrix[hankel_indx] - hankel_measurements[hankel_indx]) <= err,
        hankel_matrix[hankel_indx] == hankel_measurements[hankel_indx]
    ]
    problem = cp.Problem(objective, constraint)
    problem.solve(verbose=True)
    predicted_signal = predicted_signal.value
    hankel_matrix = manifold_matrix @ np.diag(predicted_signal) @ manifold_matrix.T
    return hankel_matrix


def solve_mmv(measurements: np.ndarray, manifold_matrix: np.ndarray, err: float = 1e-10) -> np.ndarray:
    '''
    Solve multi-measurement optimization problem of the form

        min_X || X ||_2,1
        s.t   || Y - AX ||_F < err

        where Y = [y_1 ... y_t]^T
        and   X = [x_1 ... x_t]^T 

        for t signals sharing the same sparse support.

    '''
    sensor_count, grid_size = manifold_matrix.shape
    _, num_signals = measurements.shape
    predicted_signals = cp.Variable(shape=(grid_size, num_signals))
    objective = cp.Minimize(cp.mixed_norm(predicted_signals, 2, 1))
    constraints = [
        cp.norm(measurements - manifold_matrix @ predicted_signals, 'fro') <= err
        #measurements == manifold_matrix @ predicted_signals
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    return predicted_signals.value.T


def solve_l1(measurement: np.ndarray, manifold_matrix: np.ndarray, err: float = 1e-10) -> np.ndarray:
    '''
    Solves constrained l1 optimization problem of the form

        min_x || x ||_1
        s.t.  || y - Ax ||_2 < err

        where the measurement model is 

            y = Ax
    '''
    sensor_count, grid_size = manifold_matrix.shape
    predicted_signal = cp.Variable(shape=(grid_size))
    objective = cp.Minimize(cp.norm(predicted_signal, 1))
    constraints = [
            cp.norm(measurement - manifold_matrix @ predicted_signal, 2) <= err
            #measurement == manifold_matrix @ predicted_signal
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    return predicted_signal.value


def get_largest_k_peaks(signal: np.ndarray, k: int = 1):
    peak_ind, peak_info = sps.find_peaks(np.abs(signal), height=0)
    peak_height = peak_info['peak_heights']
    peak_sortind = peak_ind[peak_height.argsort()]
    pred_peaks = peak_sortind[-k:]
    return pred_peaks


def calculate_support_error(pred_signal, true_signal):
    true_peaks = np.nonzero(true_signal)[0]
    num_peaks = len(true_peaks)
    pred_peaks = get_largest_k_peaks(pred_signal, num_peaks)
    return np.mean(np.in1d(true_peaks, pred_peaks))


def calculate_nmse(pred_signal, true_signal):
    return linalg.norm(pred_signal - true_signal) / linalg.norm(true_signal)
