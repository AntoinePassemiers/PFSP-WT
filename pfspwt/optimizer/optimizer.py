# -*- coding: utf-8 -*-
# optimizer.py: Simple optimizer based on weighted tardiness
# author : Antoine Passemiers

from pfspwt.objective import weighted_tardiness
from pfspwt.optimizer.base import BaseOptimizer

import numpy as np


class Optimizer(BaseOptimizer):
    """Heuristic optimizer based on weighted tardiness.

    Optimizers are designed for keeping track of the
    best(s) provided solution(s) and to check whether
    some stopping condition has been met.

    Attributes:
        best (:obj:`np.ndarray`): Array of shape (n,)
            representing the best scheduling solution.
        Zbest (float): Weighted tardiness of the `best` solution.
    """

    def __init__(self, *args, **kwargs):
        BaseOptimizer.__init__(self, *args, **kwargs)
        self.best = None
        self.Zbest = np.inf

    def _evaluate(self, instance, solution):
        """Evaluates a new solution.

        Parameters:
            instance (:obj:`pfsp.Instance`): Instance of the PFSP-WT problem.
            new_sol (:obj:`np.ndarray`): Array of shape (n,) representing
                a scheduling / current solution.

        Returns:
            float: Tuple of size 1 where the only element is the
                weighted tardiness.
            bool: Whether `solution` is an improvement with regards
                current best solution.
        """
        wt = weighted_tardiness(instance, solution, refresh=True)
        is_improvement = False
        if wt < self.Zbest:
            self.Zbest = wt
            self.best = solution
            is_improvement = True
        return (wt,), is_improvement

    def solutions(self):
        """Returns the set of Pareto-optimal solutions.

        Returns:
            list: List of Pareto-optimal solutions.
        """
        return [self.best]
