#!/usr/bin/env python3

import numpy as np
from utils import solve_mmv
from matplotlib import pyplot as plt
import cvxpy as cp

n_bins = 32
n_sensors = 15
k = 4
n_grid = 100

sensors = np.arange(n_sensors)

A = np.random.randn(n_sensors, n_grid)
A = np.exp(1j * np.pi * 1/2 * np.outer(sensors, np.linspace(0, 1, n_grid)))
supp = np.random.choice(n_grid, replace=False, size=k)
n_freq = len(supp)
signal = np.random.randn(n_freq, k)

y = A[:, supp] @ signal 

x = solve_mmv(y, A, err=1e-5 * np.sqrt(n_grid * n_sensors))


x_hat = cp.Variable(shape=(n_grid, k))
objective = cp.Minimize(cp.mixed_norm(x_hat, 2, 1))
constraints = [
    cp.norm(y - A @ x_hat, 'fro') <= 1e-5
    #measurements == manifold_matrix @ predicted_signals
]
problem = cp.Problem(objective, constraints)
problem.solve(verbose=True, solver='SCS')

plt.stem(np.linalg.norm(x_hat.value.T, axis=0))
plt.vlines(supp, ymin=0, ymax=plt.ylim()[-1], linestyles='--', colors='k')
plt.show()


