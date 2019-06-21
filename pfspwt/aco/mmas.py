# -*- coding: utf-8 -*-
# mmmas.py: MMAS algorithm
# author : Antoine Passemiers

from pfspwt.ant import Ant
from pfspwt.aco.base import ACO
from pfspwt.heuristics import neh_algorithm

import numba
import numpy as np


class MMAS(ACO):
    """MMAS algorithm from Thomas Stützle.

    References:
        * An Ant Approach to the Flow Shop Problem,
          Thomas Stützle.
    """

    def __init__(self, *args, **kwargs):
        ACO.__init__(self, *args, **kwargs)

    def initial_solution(self, instance):
        """Creates initial solution using the NEH heuristic

        Parameters:
            instance (:obj:`pfspwt.Instance`): PFSP-WT instance.

        Returns:
            :obj:`pfspwt.Ant`: Ant representing the initial solution. 
        """
        return Ant(neh_algorithm(instance))

    def init_pheromones(self):
        """Initializes pheromone trails.

        Trail intensities are determined using the initial
        solution (NEH).
        """
        n = self._instance.n
        self._tau = np.full((n, n), self._tau_max, dtype=np.float32)

    def update_pheromones(self, ant):
        """Updates pheromone trails.

        Parameters:
            ant (:obj:`pfspwt.Ant`): Ant that is in charge of
                adding pheromone trails.
        """
        n = self._instance.n
        Zcurrent = self.evaluate(ant)
        indices = ant.asarray()

        # Trail evaporation
        self._tau *= self.rho

        # Add new pheromone trail
        self._tau[indices, np.arange(n)] += (1. / Zcurrent)

        # Set tau_{i, j} to tau_{min} if tau_{i, j} < tau_{min}
        # Set tau_{i, j} to tau_{max} if tau_{i, j} > tau_{max}
        np.clip(self._tau, self._tau_min, self._tau_max, out=self._tau)

    def pheromones_are_individual(self):
        """Whether pheromones are added by all ants or not.

        Returns:
            bool: False because pheromones are added by the best
                ant of the iteration only.
        """
        return False

    def update_parameters(self):
        """Updates parameters.

        In MMAS and M-MMAS, only tau_min and tau_max
        are updated at each iteration.
        """
        self._tau_max =  1. / ((1. - self.rho) * self._Zbest)
        self._tau_min = self._tau_max / 5.

    @numba.jit('void(f4[:, :], i4[:], i4[:], i4[:])', nopython=True)
    def _create_solution(T, solution, best, candidates):
        """Creates a solution by following pheromone trails.

        The algorithm uses the best sequence found so far as
        an heuristic, but also uses cumulative pheromone intensities
        to build a new solution.

        Parameters:
            T (:obj:`np.ndarray`): Cumulative trail intensities,
                represented by an array of shape (n, n) where
                intensities have been summed over the second dimension.
            solution (:obj:`np.ndarray`): Array of shape (n,) where
                the solution will be stored. `solution[i]` is the
                identifier of the job being scheduled at position i.
            best (:obj:`np.ndarray`): Best sequence found so far.
            candidates (:obj:`np.ndarray`): Array of shape (5,) that
                is used to store unscheduled jobs.
        """
        n = T.shape[0]
        idx = np.arange(candidates.shape[0])

        # Store in `candidates` the next 5 job identifiers
        # to be scheduled
        for next_id in range(candidates.shape[0]):
            candidates[next_id] = best[next_id]
        next_id = candidates.shape[0]

        for k in range(n):

            u = np.random.rand()
            if u < (n - 4.) / n:
                # Take the job with maximal cumulative trail
                # intensity among the all unscheduled jobs remaining
                # in the best sequence
                candidate_id = np.argmax(T[candidates, k])
            else:
                # Randomly take a job (based on intensities)
                # among the 5 first unscheduled jobs in the
                # best sequence
                proba = T[candidates, k]
                proba /= proba.sum()
                candidate_id = idx[:5][
                        np.searchsorted(np.cumsum(proba), np.random.rand())]

            # Add the elected candidate job to the solution
            i = candidates[candidate_id]
            solution[k] = i

            # Remove elected candidate from `candidates` and add the next
            # unscheduled job from the best solution
            candidates = candidates[np.arange(candidates.shape[0]) != candidate_id]

    def create_solution(self):
        """Creates a new solution.

        Creates a new solution based on current values
        of trail intensities and returns an ant associated
        with that solution.

        Returns:
            :obj:´pfspwt.Ant´: Ant that found the new solution.
        """
        n = self._instance.n
        solution = np.arange(n, dtype=np.int32)
        candidates = np.zeros(n, dtype=np.int32)

        # Create new solution
        T = self._tau
        MMAS._create_solution(T, solution, self._best.asarray(), candidates)

        # Reject solution if negative job identifiers
        if not (n == len(np.unique(solution)) \
                and np.all(n > solution) and np.all(solution >= 0)):
            solution = self._best.asarray()
        return Ant(solution)
