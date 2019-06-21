# -*- coding: utf-8 -*-
# base.py: Base class for heuristic optimizers
# author : Antoine Passemiers

from abc import ABCMeta, abstractmethod
import time
import numpy as np


class BaseOptimizer(metaclass=ABCMeta):
    """Base class for heuristic optimizers.

    Optimizers are designed for keeping track of the
    best(s) provided solution(s) and to check whether
    some stopping condition has been met.

    Attributes:
        _objective (list): Historical values of
            the objective function.
        _n_iterations (int): Current number of iterations.
        _max_n_iterations (int): Maximum number of iterations.
        _early_stopping (int): Number of allowed optimization
            steps without improvement.
        _max_time (float): Maximum execution time allowed.
        _t0 (int): Starting time of the execution.
        _n_steps_without_improvement (int): Number of optimization
            steps without improvement.
        _seed (int): Seed for the random number generator.
    """

    def __init__(self, n_iterations=np.inf, early_stopping=np.inf,
                 max_time=None, seed=None):
        self._objective = list()
        self._max_n_iterations = n_iterations
        self._n_iterations = 0
        self._early_stopping = early_stopping
        self._max_time = max_time
        self._t0 = None
        self._n_steps_without_improvement = 0
        self._seed = seed

    def start(self):
        """Starts optimization.

        Sets the seed and resets the iteration counters.
        Execution time is measured starting
        from the time at which this method is called.
        """
        if self._seed is not None:
            np.random.seed(self._seed)
        if self._max_time is not None:
            self._t0 = time.time()
        self._n_iterations = 0
        self._n_steps_without_improvement = 0
        self._objective = list()

    def evaluate(self, instance, solution):
        """Evaluates a new solution.

        Parameters:
            instance (:obj:`pfspwt.Instance`): Instance of the PFSP-WT problem.
            solution (:obj:`np.ndarray`): Current scheduling solution.

        Returns:
            float: Weighted tardiness of current solution.
        """
        objectives, is_improvement = self._evaluate(instance, solution)
        if not is_improvement:
            self._n_steps_without_improvement += 1
        else:
            self._n_steps_without_improvement = 0
        self._objective.append(objectives)
        return objectives[0]

    def step(self):
        """Accounts for one ACO step.

        The iteration counter is incremented.
        """
        self._n_iterations += 1

    def is_running(self):
        """Checks whether the heuristic algorithm has finished,
        based either on the number of steps without improvement,
        or the maximum execution time.

        Returns:
            bool: Whether the algorithm has finished.
        """
        if self._t0 is not None:
            if self._max_time < time.time() - self._t0:
                return False
        if self._early_stopping is not None:
            if (self._n_steps_without_improvement > self._early_stopping):
                return False
        if self._n_iterations is not None:
            if self._n_iterations >= self._max_n_iterations:
                return False
        return True

    @abstractmethod
    def solutions(self):
        """Returns the best solution / Pareto-optimal solutions.

        Returns:
            list: List of (Pareto-)optimal solutions.
        """
        pass

    @abstractmethod
    def _evaluate(self, instance, solution):
        """Wrapped method for the evaluation of a new solution.

        Parameters:
            instance (:obj:`pfspwt.Instance`): Instance of the PFSP-WT problem.
            solution (:obj:`np.ndarray`): Current scheduling solution.

        Returns:
            float: Weighted tardiness of current solution.
        """
        pass

    @property
    def objective(self):
        """Retrieves the historical values of the objective function(s).

        Returns:
            :obj:`np.ndarray`: Historical values of the objective(s).
        """
        return np.squeeze(np.asarray([list(x) for x in self._objective]))
