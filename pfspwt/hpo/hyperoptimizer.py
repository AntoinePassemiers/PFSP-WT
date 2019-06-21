# -*- coding: utf-8 -*-
# hyperoptimizer.py: Hyper-parameter optimization
# author : Antoine Passemiers

import hyperopt
import time
import numpy as np


class Hyperoptimizer:

    def __init__(self, max_evals=5, maximize=False):
        self.max_evals = max_evals
        self.space = dict()
        self.trials = hyperopt.Trials()
        self.maximize = maximize
        self.best = None

    def run(self, obj):

        def objective(params):
            start_time = time.time()
            if self.maximize:
                result = -obj(params)
            else:
                result = obj(params)
            return {
                'loss': result,
                'status': hyperopt.STATUS_OK,
                'eval_time': start_time,
                'computation_time': time.time() - start_time,
            }

        self.best = hyperopt.fmin(
            objective,
            self.space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials)
        return self.best

    def __getattr__(self, attr):
        if len(list(self.trials)) == 0:
            return getattr(self, attr)
        elif not attr in list(self.trials)[0]['misc']['vals'].keys():
            return getattr(self, attr)
        else:
            return [trial['misc']['vals'][attr] for trial in list(self.trials)]

    @property
    def objective(self):
        values = list()
        for trial in list(self.trials):
            if self.maximize:
                values.append(-trial['result']['loss'])
            else:
                values.append(trial['result']['loss'])
        return np.asarray(values)
    
    def add_categorical(self, name, choices):
        self.space[name] = hyperopt.hp.choice(name, choices)

    def add_randint(self, name, lower, upper):
        _range = upper - lower
        self.space[name] = hyperopt.hp.randint(name, _range) + lower

    def add_uniform(self, name, lower, upper):
        self.space[name] = hyperopt.hp.uniform(name, lower, upper)

    def add_log_uniform(self, name, lower, upper):
        self.space[name] = hyperopt.hp.loguniform(name, lower, upper)
