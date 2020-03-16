import unittest
import data.test_data as tdd
import solvers.heuristic_graph as heur
import package.instance as inst
import package.solution as sol
import random as rn
import numpy as np
import pytups.superdict as sd


class TestHeuristic(unittest.TestCase):

    def setUp(self):
        self.instance = inst.Instance(tdd.dataset5())
        self.solution = sol.Solution(tdd.solution5())
        self.experiment = heur.GraphOriented(self.instance)
        self.experiment.check_solution()
        self.options = tdd.options5()
        self._prev = self.experiment.instance.get_prev_period
        self._next = self.experiment.instance.get_next_period
        np.random.seed(42)
        rn.seed(42)

    def test_solve(self):
        solution = self.experiment.solve(self.options)
        a = self.solution.data.to_dictup().to_tuplist().to_set()
        b = solution.data.to_dictup().to_tuplist().to_set()
        self.assertEqual(a, b)

    def test_solve_initial(self):
        self.experiment.set_solution(self.solution.data)
        self.experiment.check_solution()
        of1 = self.experiment.get_objective_function()
        # of1 = 6067
        _options = dict(max_iters_initial=10, big_window=True, num_max=50,
                        max_iters=10, timeLimit_cycle=10, mip_start=True)
        _options = {**self.options, **_options}
        solution = self.experiment.solve(_options)
        a = self.solution.data.to_dictup().to_tuplist().to_set()
        b = solution.data.to_dictup().to_tuplist().to_set()
        of2 = self.experiment.get_objective_function()
        self.assertEqual(of1, of2)
        self.assertEqual(a, b)


if __name__ == "__main__":
    t = TestHeuristic()
    t.setUp()
    t.test_solve_initial()
    # t.test_order_1()
    # unittest.main()
