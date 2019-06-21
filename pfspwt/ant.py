# -*- coding: utf-8 -*-
# ant.py: Base class for ants
# author : Antoine Passemiers

import numpy as np


class Ant:
    """Representation of an ant.

    Attributes:
        solution (:obj:`np.ndarray`): Solution chosen
            by the ant, represented as a NumPy array,
            where element `solution[i]` is the id of
            the ith job to be sequenced. The size of
            the array is the number of jobs to be sequenced.
    """

    def __init__(self, solution):
        self._solution = np.asarray(solution, dtype=np.int32)

    def asarray(self):
        """Converts the ant to a NumPy array.

        Returns:
            :obj:`np.ndarray`: Solution chosen by the ant.
        """
        return self._solution

    def copy(self):
        """Deep-copies an ant.

        Returns:
            :obj:`pfspwt.Ant`: A new ant identical to
                the current one.
        """
        return Ant(np.copy(self._solution))
