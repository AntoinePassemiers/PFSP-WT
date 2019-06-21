# -*- coding: utf-8 -*-
# run.py: Entry point of pfspwt
# author : Antoine Passemiers

from pfspwt.aco import MMAS, MMMAS, PACO
from pfspwt.io import PFSPWTIO
from pfspwt.optimizer import Optimizer, BiObjectiveOptimizer

import argparse
import os
import sys
import numpy as np


def parse_arguments():
    """Parses command line arguments.

    All arguments except `path` are optional and
    include, among others: the choice of the algorithm,
    ACO hyper-parameters, the local search method,
    the seed and the limits in computational resources.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'path',
            type=str,
            help='Path to the instance file')
    parser.add_argument(
            '--method',
            choices=['MMAS', 'M-MMAS', 'PACO'],
            default='M-MMAS',
            type=str,
            required=False,
            help='ACO algorithm')
    parser.add_argument(
            '--n-ants',
            default=None,
            type=int,
            required=False,
            help='Number of ants in the colony')
    parser.add_argument(
            '--local-search',
            choices=['none', 'swap', 'interchange', 'insertion'],
            default='none',
            type=str,
            help='Local search method')
    parser.add_argument(
            '--rho',
            default=None,
            type=float,
            required=False,
            help='Persistence of pheromone trails')
    parser.add_argument(
            '--seed',
            default=None,
            type=int,
            required=False,
            help='Seed for the random number generator')
    parser.add_argument(
            '--iterations',
            default=np.inf,
            type=int,
            required=False,
            help='Maximum number of iterations')
    parser.add_argument(
            '--early-stopping',
            default=np.inf,
            type=int,
            required=False,
            help='Maximum number of iterations without improvement')
    parser.add_argument(
            '--time',
            default=30.,
            type=float,
            required=False,
            help='Maximum execution time (in seconds)')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        import sys
        sys.exit(0)
    return args


if __name__ == '__main__':

    # Parse command line arguments
    args = parse_arguments()

    # Load instance from text file
    instance = PFSPWTIO.read(args.path)

    # Initialize optimizer with given resources
    optimizer = Optimizer(
            n_iterations=args.iterations,
            early_stopping=args.early_stopping,
            max_time=args.time,
            seed=args.seed)

    # Create ACO
    kwargs = { 'ls': args.local_search }
    if args.method == 'MMAS':
        kwargs['n_ants'] = 22 if args.n_ants is None else args.n_ants
        kwargs['rho'] = 0.23 if args.rho is None else args.rho
        aco = MMAS(optimizer, **kwargs)
    elif args.method == 'M-MMAS':
        kwargs['n_ants'] = 34 if args.n_ants is None else args.n_ants
        kwargs['rho'] = 0.3 if args.rho is None else args.rho
        aco = MMMAS(optimizer, **kwargs)
    else:
        kwargs['n_ants'] = 50 if args.n_ants is None else args.n_ants
        kwargs['rho'] = 0.4 if args.rho is None else args.rho
        aco = PACO(optimizer, **kwargs)

    # Use ACO for optimization
    aco.initialize(instance)
    k = 0
    while optimizer.is_running():
        aco.step()
        k += 1
    objs = optimizer.objective

    print('Found solution: %s' % str(optimizer.solutions()[0]))
