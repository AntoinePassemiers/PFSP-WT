# -*- coding: utf-8 -*-
# instance.py: Problem instances
# author : Antoine Passemiers

from pfspwt.objective import weighted_tardiness

import numpy as np


class Instance:
    """Instance of the PFSP-WT problem.

    Attributes:
        n (int): Number of jobs.
        m (int): Number of available machines.
        p (:obj:`np.ndarray`): Matrix of shape (N, M)
            where `p[i, j]` is the processing time
            of job i on machine j.
        c (:obj:`np.ndarray`): Buffer for storing computation
            times, represented as a matrix of shape (N, M).
        d (:obj:`np.ndarray`): Vector of shape (N,)
            where `D[i]` is the deadline of job i.
        w (:obj:`np.ndarray`): Vector of shape (N,)
            where `w[i]` is the priority weight of job i.
    """

    def __init__(self, p, d, w):
        self.p = np.asarray(p, dtype=np.int32)
        self.c = np.empty_like(self.p)
        self.d = np.asarray(d, dtype=np.int32)
        self.w = np.asarray(w)
        self.n = self.p.shape[0]
        self.m = self.p.shape[1]
