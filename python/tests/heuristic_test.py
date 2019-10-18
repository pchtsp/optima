import unittest
import data.test_data as tdd
import solvers.heuristics_maintfirst as heur
import package.instance as inst
import package.solution as sol
import random as rn
import numpy as np
import pytups.superdict as sd

"""
TODO:
template reading and writing.
 

"""



class TestHeuristic(unittest.TestCase):

    def setUp(self):
        instance = inst.Instance(tdd.dataset2())
        solution = sol.Solution(tdd.solution2())
        self.experiment = heur.MaintenanceFirst(instance, solution)
        self.experiment.check_solution()
        self._prev = self.experiment.instance.get_prev_period
        self._next = self.experiment.instance.get_next_period
        np.random.seed(42)
        rn.seed(42)

    def test_move_success(self):
        res = '1'
        date = '2018-07'
        m = 'VI'

        opts = [(date, 16), (self._prev(date), 2), (self._next(date), 15)]
        ret = self.experiment.solution.data['aux']['ret']
        for opt in opts:
            _ret = ret.get_m(m, res, opt[0])
            self.assertEqual(_ret, opt[1])
        result = self.experiment.move_maintenance(res, date, m, -1)

        self.assertNotIn(date, self.experiment.solution.data['state_m'][res])
        self.assertIn(self._prev(date), self.experiment.solution.data['state_m'][res])
        self.assertIn(m, self.experiment.solution.data['state_m'][res][self._prev(date)])
        opts = [(self._prev(date), 16), (self._prev(self._prev(date)), 3), (date, 15)]
        for opt in opts:
            _ret = ret.get_m(m, res, opt[0])
            self.assertEqual(_ret, opt[1])

    def test_move_bad(self):
        res = '1'
        date = '2018-07'
        m = 'VI'
        result = self.experiment.move_maintenance(res, date, m, 1)
        self.assertIsNone(result)

    def test_del_maint(self):
        res = '1'
        date = '2018-03'
        m = 'VG'
        ret = self.experiment.solution.data['aux']['ret']
        solution_data = self.experiment.solution.data['state_m']
        opts = [(date, 8), (self._prev(date), 2), (self._next(date), 7)]
        for opt in opts:
            _ret = ret.get_m(m, res, opt[0])
            self.assertEqual(_ret, opt[1])
        self.experiment.del_maint(res, date, m)
        self.assertNotIn(date, solution_data[res])
        self.experiment.update_rt_until_next_maint(res, date, m, 'ret')
        opts = [(date, 1), (self._prev(date), 2), (self._next(date), 0)]
        for opt in opts:
            _ret = ret.get_m(m, res, opt[0])
            self.assertEqual(_ret, opt[1])

    def test_assign_maint_bad(self):
        res = '1'
        date = '2018-03'
        m = 'VG'
        self.experiment.assign_maintenance(resource=res, maint=m, maint_start=date)
        pass

    def test_del_assign_maint(self):
        res, date, m = '1', '2018-03', 'VG'
        solution_data = self.experiment.solution.data['state_m']
        # we delete all maints
        [self.experiment.del_maint(res, p) for p in self.experiment.solution.data['state_m'][res].keys_l()]
        # we confirm there are errors
        errors = self.experiment.check_solution()
        errors_sample = \
            {('VG', '1', '2018-04'): 0,
             ('VG', '1', '2018-05'): -1,
             ('VG', '1', '2018-06'): -2,
             ('VG', '1', '2018-07'): -3,
             ('VG', '1', '2018-08'): -4,
             ('VG', '1', '2018-09'): -5,
             ('VG', '1', '2018-10'): -6,
             ('VI', '1', '2018-08'): 0,
             ('VI', '1', '2018-09'): -1,
             ('VI', '1', '2018-10'): -2}
        self.assertDictEqual(errors['elapsed'], errors_sample)
        # we assign other, different, maints.
        opts = [('1', '2018-04', 'VG'), ('1', '2018-08', 'VI')]
        for res, date, m in opts:
            self.experiment.assign_maintenance(resource=res, maint=m, maint_start=date)
        # we confirm there no errors now
        errors = self.experiment.check_solution(recalculate=False)
        self.assertDictEqual(errors, {})
        # we check the existence of the new maintenances
        for opt in opts:
            self.assertEqual(solution_data.get_m(*opt), 1)

if __name__ == "__main__":
    # t = TestHeuristic()
    # t.setUp()
    # t.test_123()
    # t.test_order_1()
    unittest.main()