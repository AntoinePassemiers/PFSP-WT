# -*- coding: utf-8 -*-
# mmmas.py: PACO algorithm
# author : Antoine Passemiers

from pfspwt.ant import Ant
from pfspwt.aco.base import ACO
from pfspwt.heuristics import neh_algorithm

import numba
import numpy as np


class PACO(ACO):
    """ACO algorithm proposed by Rajendran et al.
    
    Attributes:
        _seed_ant (:obj:`pfspwt.Ant`): Seed solution.

    References:
        * Ant-colony algorithms for permutation flowshop scheduling
          to minimize makespan/total flowtime of jobs,
          Chandrasekharan Rajendran and Hans Ziegler,
          European Journal of Operational Research 155 (2004) 426–438.
    """

    def __init__(self, *args, **kwargs):
        ACO.__init__(self, *args, **kwargs)
        self._seed_ant = None

    def initial_solution(self, instance):
        """Generates initial solution using NEH heuristic.

        Parameters:
            instance (:obj:`pfspwt.Instance`): PFSP-WT instance.

        Returns:
            :obj:`pfspwt.Ant`: Initial ant.
        """
        return Ant(neh_algorithm(instance))

    def init_pheromones(self):
        """Initializes pheromone trails.

        Trail intensities are determined using the initial
        solution (NEH).
        """

        # Set seed sequence to the initial solution
        self._seed_ant = self._best

        # Initialize pheromone trails
        n = self._instance.n
        self._tau = np.full((n, n), (1. / self._Zbest), dtype=np.float32)
        indices = self._best.asarray()

        # h[i] is the position of job i in current sequence
        h = np.argsort(indices)[..., np.newaxis]

        # Compute `diff` like in PACO
        k = np.arange(n)[np.newaxis, ...]
        diff = np.abs(h - k) + 1

        # if diff[i, j] > n / 4, then tau[i, j] *= 2
        # if diff[i, j] > n / 2, then tau[i, j] *= 4
        # Since n / 2 > n / 4, diff[i, j] is divided
        # by 2 and not 4 afterwards
        self._tau[n / 4. < diff] /= 2.
        self._tau[n / 2. <= diff] /= 2.

    def update_pheromones(self, ant):
        """Updates pheromone trails.

        Parameters:
            ant (:obj:´pfspwt.Ant`): Ant that is in charge of
                adding pheromone trails.
        """
        n = self._instance.n

        # Evaluate ant
        Zcurrent = self.evaluate(ant)

        # h[i] is the position of job i in current sequence
        h = np.argsort(ant.asarray())[..., np.newaxis]

        # h_best[i] is the position of job i in the best sequence
        # obtained so far
        h_best = self._best.asarray()[..., np.newaxis]

        # Compute `diff` like in PACO
        k = np.arange(n)[np.newaxis, ...]
        diff = np.sqrt(np.abs(h_best - k) + 1.)

        # Pheromone evaporation
        self._tau *= self.rho

        # Add new pheromone trails
        bound = 1 if (n <= 40) else 2
        idx = (np.abs(h - k) <= bound)
        self._tau[idx] += (1. / (diff[idx] * Zcurrent))

    def pheromones_are_individual(self):
        """Whether pheromones are added by all ants or not.

        Returns:
            bool: True because pheromones are added by each
                ant in PACO.
        """
        return True

    def update_parameters(self):
        """Updates parameters.

        In PACO, there are no parameters to be updated
        between iterations.
        """
        pass

    @numba.jit('void(f4[:, :], i4[:], i4[:], i4[:])', nopython=True)
    def _create_solution(T, solution, best, candidates):
        """Creates a solution by following pheromone trails.

        The algorithm uses the seed sequence (sequence obtained
        by running NEH heuristic + eventually local search) found so
        far as an heuristic, but also uses cumulative pheromone
        intensities to build a new solution.

        Parameters:
            T (:obj:`np.ndarray`): Cumulative trail intensities,
                represented by an array of shape (n, n) where
                intensities have been summed over the second dimension.
            solution (:obj:`np.ndarray`): Array of shape (n,) where
                the solution will be stored. `solution[i]` is the
                identifier of the job being scheduled at position i.
            best (:obj:`np.ndarray`): Best sequence found so far.
            candidates (:obj:`np.ndarray`): Array of shape (5,) that
                is used to store the next 5 job identifiers in the
                best sequence that have not been scheduled yet in
                the current solution.
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
            if u <= 0.4:
                # Take the first unscheduled job found
                # in the seed sequence
                candidate_id = 0
            elif u <= 0.8:
                # Take the job with maximal cumulative trail
                # intensity among the 5 first unscheduled jobs
                # in the seed sequence
                candidate_id = np.argmax(T[candidates, k])
            else:
                # Randomly take a job (based on intensities)
                # among the 5 first unscheduled jobs in the
                # seed sequence
                proba = T[candidates, k]
                proba /= proba.sum()
                candidate_id = idx[:candidates.shape[0]][
                        np.searchsorted(np.cumsum(proba), np.random.rand())]

            # Add the elected candidate job to the solution
            i = candidates[candidate_id]
            solution[k] = i

            # Remove elected candidate from `candidates` and add the next
            # unscheduled job from the best solution
            if next_id < n:
                candidates[candidate_id:-1] = candidates[candidate_id+1:]
                candidates[-1] = best[next_id]
                next_id += 1
            else:
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
        candidates = np.zeros(5, dtype=np.int32)

        # Summation of trail intensities
        T = np.cumsum(self._tau, axis=1)
        PACO._create_solution(T, solution, self._best.asarray(), candidates)
        return Ant(solution)
