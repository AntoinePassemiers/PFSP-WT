# -*- coding: utf-8 -*-
# bioptimizer.py: Bi-objective optimization
# author : Antoine Passemiers

from pfspwt.objective import weighted_tardiness, makespan
from pfspwt.optimizer.base import BaseOptimizer


class BiObjectiveOptimizer(BaseOptimizer):
    """Bi-objective optimizer for the PFSP-WT problem.

    Optimizers are designed for keeping track of the
    best(s) provided solution(s) and to check whether
    some stopping condition has been met.

    First objective function is the weighted tardiness
    and second objective function is the makespan.

    Attributes:
        pareto_set (dict): Set of tuples where each
            tuple is an alternative characterized
            by its weighted tardiness, makespan and solution.
    """

    def __init__(self, *args, **kwargs):
        BaseOptimizer.__init__(self, *args, **kwargs)
        self.pareto_set = dict()

    def _evaluate(self, instance, new_sol):
        """Evaluates a new solution.

        Parameters:
            instance (:obj:`pfsp.Instance`): Instance of the PFSP-WT problem.
            new_sol (:obj:`np.ndarray`): Array of shape (n,) representing
                a scheduling / current solution.

        Returns:
            float: Tuple where first element is the weighted tardiness
                and second element is the makespan.
            bool: Whether `new_sol` is an improvement with regards
                current best solution.
        """

        # Compute metrics
        new_wt = weighted_tardiness(instance, new_sol)
        new_m = makespan(instance, new_sol, refresh=False)

        # Update Pareto set with new solution
        new_pareto_set = dict()
        new_sol_is_dominated = False
        is_improvement = False
        for (wt, m), sol in self.pareto_set.items():
            if self.dominates((wt, m, sol), (new_wt, new_m, new_sol)):
                new_sol_is_dominated = True
                break # New solution is dominated and thus not added to the set
            elif not self.dominates((new_wt, new_m, new_sol), (wt, m, sol)):
                new_pareto_set[(wt, m)] = sol
            else:
                is_improvement = True
        if not new_sol_is_dominated:
            # Add new solution to Pareto set (since not dominated)
            new_pareto_set[(new_wt, new_m)] = new_sol
            self.pareto_set = new_pareto_set
        return (new_wt, new_m), is_improvement

    def dominates(self, a, b):
        """Checks whether a candidate is strictly better
        than another candidate.

        Parameters:
            a (tuple): First candidate.
            b (tuple): Second candidate.

        Returns:
            bool: Whether candidate `a` is stricly better than
                candidate `b`.
        """
        (wt_a, m_a, sol_a) = a
        (wt_b, m_b, sol_b) = b
        return wt_a <= wt_b and m_a <= m_b and (wt_a < wt_b or m_a < m_b)

    def solutions(self):
        """Returns the set of Pareto-optimal solutions.

        Returns:
            list: List of Pareto-optimal solutions.
        """
        return [sol for _, _, sol in self.pareto_set]
