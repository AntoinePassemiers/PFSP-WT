# -*- coding: utf-8 -*-
# objective.py: Objective functions
# author : Antoine Passemiers

import numpy as np
import numba


@numba.jit('void(i4[:, :], i4[:, :])', nopython=True)
def computation_times(c, p):
    """Calculates computation times inplace.

    Parameters:
        c (:obj:`np.ndarray`): Buffer for storing computation
            times, represented as a matrix of shape (N, M).
        p (:obj:`np.ndarray`): Matrix of shape (N, M)
            where `p[i, j]` is the processing time
            of job i on machine j.
    """
    n = p.shape[0]
    m = p.shape[1]

    c[:, 0] = np.cumsum(p[:, 0])
    c[0, :] = np.cumsum(p[0, :])

    for i in range(1, n):
        for j in range(1, m):
            c[i, j] = max(c[i-1, j], c[i, j-1]) + p[i, j] 


def objective(func):
    """Decorator for objective functions.

    Wrapped function calculates the objective
    function with job computation times already
    precompted by the decorator.

    Parameters:
        fun (function): Wrapped function.

    Returns:
        function: A new objective function.
    """

    def new_func(instance, solution, refresh=True):
        if refresh:
            ordered_p = instance.p[solution, :]
            computation_times(instance.c, ordered_p)
        return func(instance, solution)
    new_func.__name__ = func.__name__
    return new_func


@objective
def weighted_tardiness(instance, solution):
    """Computes weighted tardiness.

    Parameters:
        instance (:obj:`pfsp.Instance`): Problem instance.
        solution (:obj:`np.ndarray`): Array of shape (n,)
            representing the current solution.

    Returns:
        int: Weighted tardiness.
    """
    d = instance.d[solution]
    w = instance.w[solution]
    t = np.maximum(instance.c[:, -1] - d, 0)
    return np.dot(w, t)


@objective
def makespan(instance, solution):
    """Computes makespan.

    Parameters:
        instance (:obj:`pfsp.Instance`): Problem instance.
        solution (:obj:`np.ndarray`): Array of shape (n,)
            representing the current solution.

    Returns:
        int: Makespan.
    """
    return np.max(instance.c[:, -1])
