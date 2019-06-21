# -*- coding: utf-8 -*-
# heuristics.py: Heuristics for initial solutions
# author : Antoine Passemiers

from pfspwt.objective import computation_times

import numpy as np
import numba
import scipy.stats


def neh_algorithm(instance):
    """NEH heuristic for building an initial solution.

    References:
        * A heuristic algorithm for the m-machine,
        n-job flow-shop sequencing problem.
        Nawaz, Muhammad and Enscore Jr, E Emory and Ham, Inyong.
        Omega (1983) 91-95

    Parameters:
        instance (:obj:`instance`): PFSP-WT instance.

    Returns:
        :obj:`np.ndarray`: Array representing the solution.
    """

    # Sort jobs by due date
    _c = instance.d
    indices = np.argsort(_c)

    # Retrieve constants of the problem
    c, w, d = instance.c, instance.w, instance.d
    p = instance.p
    n = len(indices)

    # Add first job to the solution
    solution = [indices[0]]

    for k in range(1, n):
        best_wt, best_idx = np.inf, 0
        for h in range(k + 1):

            # Add next job to the partial sequence
            partial_indices = solution[:h] + [indices[k]] + solution[h:]

            # Sort constants according to the partial sequence
            ordered_p = p[partial_indices, :]
            ordered_w = w[partial_indices]
            ordered_d = d[partial_indices]
            ordered_c = c[:len(partial_indices), :]

            # Compute weighted tardiness
            computation_times(ordered_c, ordered_p)
            ordered_t = np.maximum(ordered_c[:, -1] - ordered_d, 0)
            wt = np.dot(ordered_w, ordered_t)

            # Keep the index of the best position where to insert
            # the next job
            if wt < best_wt:
                best_wt = wt
                best_idx = h

        # Insert the next job at the best position
        solution = solution[:best_idx] + [indices[k]] + solution[best_idx:]
    solution = np.asarray(solution, dtype=np.int32)
    return solution
