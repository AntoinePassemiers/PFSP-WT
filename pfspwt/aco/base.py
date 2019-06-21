# -*- coding: utf-8 -*-
# base.py: Base class for Ant Colony Optimizers
# author : Antoine Passemiers

from pfspwt.ant import Ant
from pfspwt.neighbourhood import *
from pfspwt.objective import weighted_tardiness

from abc import ABCMeta, abstractmethod
import numpy as np


class ACO(metaclass=ABCMeta):
    """Base class for Ant Colony Optimization algorithms.

    Attributes:
        _optimizer (:obj:`pfspwt.optimizer.BaseOptimizer`):
            Heuristic optimizer.
        n_ants (int): Number of ants treated at each iteration.
        rho (float): Pheromone trail persistence.
        _ls (str): Local search method to use. "none" corresponds
            to no local search at all.
        _tau (:obj:`np.ndarray`): Array of shape (n, n) where
            element `_tau[i, j]` if the pheromone trail
            associated to the desire of placing job i
            at position j in the solution.
        _best (:obj:`np.ndarray`): Array of shape (n,)
            representing the best scheduling solution.
        _Zbest (float): Weighted tardiness of the `best`
            solution.
        _instance (:obj:`pfspwt.Instance`): Instance of the
            PFSP-WT problem.
    """

    def __init__(self, optimizer, n_ants=40, rho=.75, ls='none'):
        self._optimizer = optimizer
        self.n_ants = n_ants
        self.rho = rho
        self._ls = ls
        if ls is not None:
            self._ls = ls.lower().strip()
            lss = ['none', 'swap', 'interchange', 'insertion']
            assert(self._ls in lss)
        self._tau = None
        self._best = None
        self._Zbest = None
        self._instance = None

    def evaluate(self, ant):
        """Evaluates a solution newly found by an ant.

        Parameters:
            ant (:obj:`pfspwt.Ant`): Ant.

        Returns:
            float: Weighted tardiness
        """
        return self._optimizer.evaluate(self._instance, ant.asarray())

    def initialize(self, instance):
        """Initialize the ant colony for an instance of the PFSP-WT problem.

        Initialization consists of four steps:
        1) Finding an initial solution.
        2) Eventually applying local search on the initial solution.
        3) Initialize the parameters of the algorithm.
        4) Initialize the pheromone trails.

        Parameters:
            instance (:obj:`pfspwt.Instance`): Instance
                of the PFSP-WT problem.
        """
        # Start keeping track of optimal solutions
        self._optimizer.start()

        # Find initial solution
        self._instance = instance
        ant = self.initial_solution(instance)
        self._Zbest = self.evaluate(ant)
        self._best = ant.copy()

        # Apply local search
        ant = self.local_search(ant)
        Z = self.evaluate(ant)
        if Z < self._Zbest:
            self._Zbest = Z
            self._best = ant.copy()

        # Initialize parameter values
        self.update_parameters()

        # Initialize pheromone trails
        self.init_pheromones()

    def step(self):
        """Applies one ACO step.

        An ACO step consists in creating multiple
        ants to find new solutions, eventually
        applying local search on them, and updating
        the pheromones and parameters.
        """
        solutions, scores = list(), list()
        for k in range(self.n_ants):

            # Create ant
            ant = self.create_solution()

            # Apply local search
            ant = self.local_search(ant)

            # Update pheromones with each ant if asked
            if self.pheromones_are_individual():
                self.update_pheromones(ant)

            # Evaluate solution found by the ant
            solutions.append(ant)
            score = self.evaluate(ant)
            scores.append(score)
            if score < self._Zbest:
                self._Zbest = score
                self._best = ant

            # Check for convergence or ressources
            if not self._optimizer.is_running():
                break

        # Best ant for current iteration
        current_ant = solutions[np.argmin(scores)]

        # Update trails of pheromones
        if not self.pheromones_are_individual():
            self.update_pheromones(current_ant)

        # Update algorithm parameters
        self.update_parameters()

    def local_search(self, ant):
        """Applies local search on current solution.

        Parameters:
            ant (:obj:`pfspwt.Ant`): Current ant in
                exploration phase.

        Returns:
            :obj:`pfspwt.Ant`: An ant with possibly
                improved solution.
        """
        solution = ant.asarray()
        for _ in range(3):
            improvement = False
            if self._ls == 'swap':
                solution, improvement = swap_search(
                        self._instance, solution)
            elif self._ls == 'interchange':
                solution, improvement = interchange_search(
                        self._instance, solution)
            elif self._ls == 'insertion':
                solution, improvement = insertion_search(
                        self._instance, solution)
            if not improvement:
                break
        return Ant(solution)

    @abstractmethod
    def initial_solution(self, instance):
        """Creates initial solution using an heuristic.

        Parameters:
            instance (:obj:`pfspwt.Instance`): PFSP-WT instance.

        Returns:
            :obj:`pfspwt.Ant`: Ant representing the initial solution. 
        """
        pass

    @abstractmethod
    def init_pheromones(self):
        """Initializes pheromone trails.

        Pheromone intensities may be computed based
        on the initial solution.
        """
        pass

    @abstractmethod
    def update_pheromones(self, ant):
        """Updates pheromone trails.

        Parameters:
            ant (:obj:`pfspwt.Ant`): Ant that is in charge
                of adding new pheromone trails.
        """
        pass

    @abstractmethod
    def pheromones_are_individual(self):
        """Whether pheromones are added by each ant or not.

        Returns:
            bool: If True, pheromone trails are updated at
                each solution creation (like in M-MMAS).
                If False, trails are updated only by the
                best ant of current iteration.
        """
        pass

    @abstractmethod
    def update_parameters(self):
        """Updates parameters.

        Parameters may be, for example, the lower
        and upper bounds of trail intensities of MMAS.
        """
        pass

    @abstractmethod
    def create_solution(self):
        """Creates a new solution.

        Creates a new solution based on current values
        of trail intensities and returns an ant associated
        with that solution.

        Returns:
            :obj:´pfspwt.Ant´: Ant that found the new solution.
        """
        pass

    @property
    def optimizer(self):
        """Returns the optimizer.

        Returns:
            :obj:´pfspwt.optimizer.BaseOptimizer`: Optimizer
                that keep track of the best solutions and
                the computational resources.
        """
        return self._optimizer
