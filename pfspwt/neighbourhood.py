# -*- coding: utf-8 -*-
# neighbourhood.py: Solution neighbourhoods for local search
# author : Antoine Passemiers

from pfspwt.objective import computation_times

import numpy as np
import numba


@numba.jit('boolean(i4[:, :], i4[:, :], i4[:], f4[:], i4[:], i4[:])', nopython=True)
def __swap_search(c, p, d, w, solution, new_sol):
    """Local search based on swap moves.

    Parameters:
        c (:obj:`np.ndarray`): Completion times.
        p (:obj:`np.ndarray`): Computation times.
        d (:obj:`np.ndarray`): Due dates.
        w (:obj:`np.ndarray`): Weights.
        solution (:obj:`np.ndarray`): Current solution.
        new_sol (:obj:`np.ndarray`): Array to store the new solution.
    """
    new_sol[:] = solution[:]

    # Evaluate weighted tardiness on provided solution
    computation_times(c, p[solution, :])
    t = np.maximum(c[:, -1] - d, 0).astype(np.float32)
    bestZ = np.dot(w, t)
    best_idx = -1
    for i in range(len(solution) - 1):

        # Swap two jobs
        solution[i], solution[i+1] = solution[i+1], solution[i]

        # Compute completion times of new solution
        computation_times(c, p[solution, :])

        # Put jobs back in place
        solution[i], solution[i+1] = solution[i+1], solution[i]

        # Evaluate weighted tardiness of new solution
        t = np.maximum(c[:, -1] - d, 0).astype(np.float32)
        currentZ = np.dot(w, t)
        if currentZ < bestZ:
            bestZ = currentZ
            best_idx = i
    i = best_idx
    if i > -1:
        new_sol[i], new_sol[i+1] = new_sol[i+1], new_sol[i]
        improvement = True
    else:
        improvement = False
    return improvement


def swap_search(instance, solution):
    """Local search based on swap moves.

    This function is a wrapper of the Numba function
    of the same name.

    Parameters:
        instance (:obj:`pfspwt.Instance`): PFSP-WT instance.
        solution (:obj:`np.ndarray`): Array of shape (n,) where
            `solution[i]` is the identifier of the job scheduled
            in position i.
    
    Returns:
        :obj:`np.ndarray`: Improved solution.
    """
    d = instance.d[solution]
    w = instance.w[solution].astype(np.float32)
    p = instance.p
    c = instance.c
    new_solution = np.empty_like(solution)
    improvement = __swap_search(c, p, d, w, solution, new_solution)
    return new_solution, improvement


@numba.jit('boolean(i4[:, :], i4[:, :], i4[:], f4[:], i4[:], i4[:])', nopython=True)
def __interchange_search(c, p, d, w, solution, new_sol):
    """Local search based on interchange moves.

    Parameters:
        c (:obj:`np.ndarray`): Completion times.
        p (:obj:`np.ndarray`): Computation times.
        d (:obj:`np.ndarray`): Due dates.
        w (:obj:`np.ndarray`): Weights.
        solution (:obj:`np.ndarray`): Current solution.
        new_sol (:obj:`np.ndarray`): Array to store the new solution.
    """
    new_sol[:] = solution[:]

    # Compute weighted tardiness of provided solution
    computation_times(c, p[solution, :])
    t = np.maximum(c[:, -1] - d, 0).astype(np.float32)
    bestZ = np.dot(w, t)

    best_i, best_j = -1, -1
    for i in range(len(solution) - 1):
        for j in range(i):

            # Interchange jobs i and j
            solution[i], solution[j] = solution[j], solution[i]

            # Compute completion times of new solution
            computation_times(c, p[solution, :])

            # Put jobs back in place
            solution[i], solution[j] = solution[j], solution[i]

            # Evaluate weighted tardiness of new solution
            t = np.maximum(c[:, -1] - d, 0).astype(np.float32)
            currentZ = np.dot(w, t)
            if currentZ < bestZ:
                bestZ = currentZ
                best_i, best_j = i, j
    i, j = best_i, best_j
    if i > -1:
        i, j = best_i, best_j
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
        return 1
    else:
        return 0


def interchange_search(instance, solution):
    """Local search based on interchange moves.

    This function is a wrapper of the Numba function
    of the same name.

    Parameters:
        instance (:obj:`pfspwt.Instance`): PFSP-WT instance.
        solution (:obj:`np.ndarray`): Array of shape (n,) where
            `solution[i]` is the identifier of the job scheduled
            in position i.
    
    Returns:
        :obj:`np.ndarray`: Improved solution.
    """
    d = instance.d[solution]
    w = instance.w[solution].astype(np.float32)
    p = instance.p
    c = instance.c
    new_solution = np.empty_like(solution)
    improvement = __swap_search(c, p, d, w, solution, new_solution)
    return new_solution, improvement


@numba.jit('boolean(i4[:, :], i4[:, :], i4[:], f4[:], i4[:], i4[:])', nopython=True)
def __insertion_search(c, p, d, w, solution, new_sol):
    """Local search based on insertion moves.

    Parameters:
        c (:obj:`np.ndarray`): Completion times.
        p (:obj:`np.ndarray`): Computation times.
        d (:obj:`np.ndarray`): Due dates.
        w (:obj:`np.ndarray`): Weights.
        solution (:obj:`np.ndarray`): Current solution.
        new_sol (:obj:`np.ndarray`): Array to store the new solution.
    """
    new_sol[:] = solution[:]

    # Evaluate weighted tardiness of the provided solution
    computation_times(c, p[solution, :])
    t = np.maximum(c[:, -1] - d, 0).astype(np.float32)
    bestZ = np.dot(w, t)

    best_i, best_j = -1, -1
    for i in range(len(solution)):
        for j in range(i):

            # Insert job j at position i
            solution[j+1:i+1], solution[j] = solution[j:i], solution[i]

            # Compute completion times of the new solution
            computation_times(c, p[solution, :])

            # Put the jobs back in place
            solution[i:j], solution[j], solution[i+1:j+1], solution[i]

            # Evaluate the weighted tardiness of the new solution
            t = np.maximum(c[:, -1] - d, 0).astype(np.float32)
            currentZ = np.dot(w, t)
            if currentZ < bestZ:
                bestZ = currentZ
                best_i, best_j = i, j
    if best_i > -1:
        i, j = best_i, best_j
        new_sol[j+1:i+1], new_sol[j] = new_sol[j:i], new_sol[i]
        return 1
    else:
        return 0


def insertion_search(instance, solution):
    """Local search based on insertion moves.

    This function is a wrapper of the Numba function
    of the same name.

    Parameters:
        instance (:obj:`pfspwt.Instance`): PFSP-WT instance.
        solution (:obj:`np.ndarray`): Array of shape (n,) where
            `solution[i]` is the identifier of the job scheduled
            in position i.
    
    Returns:
        :obj:`np.ndarray`: Improved solution.
    """
    d = instance.d[solution]
    w = instance.w[solution].astype(np.float32)
    p = instance.p
    c = instance.c
    new_solution = np.empty_like(solution)
    improvement = __insertion_search(c, p, d, w, solution, new_solution)
    return new_solution, improvement
