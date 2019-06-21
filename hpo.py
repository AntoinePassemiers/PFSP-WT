# -*- coding: utf-8 -*-
# hpo.py: Hyper-parameter optimization
# author : Antoine Passemiers

from pfspwt.aco import MMAS, MMMAS, PACO
from pfspwt.io import PFSPWTIO
from pfspwt.optimizer import Optimizer
from pfspwt.hpo import Hyperoptimizer

import os
import numpy as np


DATA_DIR = 'data'
MAX_TIME = 10.


if __name__ == '__main__':

    # Load all instances from data folder
    instances = dict()
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.txt'):
            name = filename.split('.')[0]
            filepath = os.path.join(DATA_DIR, filename)
            instances[name] = PFSPWTIO.read(filepath)

    # Define HPO objective function as the sum of objective functions
    # across the whole dataset
    def objective(params):
        scores = list()
        for instance in instances.values():
            optimizer = Optimizer(max_time=MAX_TIME)
            aco = MMAS(optimizer, n_ants=params['n_ants'], rho=params['rho'])
            aco.initialize(instance)
            while optimizer.is_running():
                aco.step()
            scores.append(np.min(optimizer.objective))
        return np.mean(scores)

    # Hyper-optimize the model on the number of ants
    # and the pheromone trail persistence
    hpo = Hyperoptimizer(max_evals=50, maximize=False)
    hpo.add_randint('n_ants', 1, 100)
    hpo.add_uniform('rho', 0., 1.)
    best = hpo.run(objective)
    print(best)

    print(list(hpo.objective))
    print(list(hpo.n_ants))
    print(list(hpo.rho))

