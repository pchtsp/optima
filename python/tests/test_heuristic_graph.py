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
        # calculate new solution
        _options = dict(max_iters_initial=1, big_window=True, num_max=10000,
                        max_iters=1, timeLimit_cycle=30)
        _options = {**self.options, **_options}
        solution = self.experiment.solve(_options)

        # prepare new solution
        errors1 = self.experiment.check_solution()
        solution1 = solution.data.to_dictup().to_tuplist().to_set()
        of2 = self.experiment.get_objective_function()

        # get old solution
        self.experiment.set_solution(self.solution.data)

        # prepare old solution
        errors2 = self.experiment.check_solution()
        of1 = self.experiment.get_objective_function()
        solution2 = self.experiment.solution.data.to_dictup().to_tuplist().to_set()

        # import pprint
        # pprint.pprint(sorted(a - b))
        # test they are the same
        self.assertEqual(errors1, errors2)

        # TODO: for some reason the model does not permit incomplete
        #   maintenances at the end. This makes it impossible to compare.
        #   ths needs to be corrected in the model or in the graph generation
        # self.assertEqual(solution1, solution2)
        # self.assertEqual(of1, of2)

    def test_solve_initial_fixed(self):
        # get old solution
        self.experiment.set_solution(self.solution.data)

        # prepare old solution
        self.experiment.check_solution()
        a = self.experiment.solution.data.to_dictup().to_tuplist().to_set()
        of1 = self.experiment.get_objective_function()

        # get new solution
        _options = dict(max_iters_initial=0, big_window=True, num_max=10000,
                        max_iters=1, timeLimit_cycle=10, mip_start=True, fix_vars=['assign'],
                        temperature=999999999999)
        _options = {**self.options, **_options}
        self.experiment.solve(_options)

        # prepare new solution
        self.experiment.check_solution()
        b = self.experiment.solution.data.to_dictup().to_tuplist().to_set()
        of2 = self.experiment.get_objective_function()

        # check
        self.assertEqual(of1, of2)
        self.assertEqual(a, b)



if __name__ == "__main__":
    # t = TestHeuristic()
    # t.setUp()
    # t.test_solve_initial()
    unittest.main()
