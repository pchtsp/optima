import unittest
import data.data_input as di
import package.instance as inst
import os
import data.dates as aux

class TestInstance(unittest.TestCase):

    def setUp(self):
        # directory = '/home/pchtsp/Documents/projects/OPTIMA/data/examples/201811092041/'
        directory = './../../data/examples/201811092041/'
        self.model_data = di.load_data(os.path.join(directory, 'data_in.json'))
        self.options = di.load_data(os.path.join(directory, 'options.json'))
        self.instance = inst.Instance(self.model_data)
        self.dates_to_test = \
            [('2018-01', '2023-01'),
             ('2018-01', '2017-01'),
             ('2018-01', '2018-01'),
             ('2018-01', '2017-12'),
             (self.instance.get_param('start'), self.instance.get_param('end')),
             ]

    # def solve_example(self):
    #     exec.execute_solve(model_data, options)

    def test_get_periods(self):
        periods = self.instance.get_periods()
        periods_real = aux.get_months(self.instance.get_param('start'),
                                      self.instance.get_param('end'))
        self.assertEqual(periods, periods_real)

    def test_get_periods_range(self):

        for start, end in self.dates_to_test:
            periods = self.instance.get_periods_range(start, end)
            periods_real = aux.get_months(start, end)
            self.assertEqual(periods, periods_real)

    def test_get_next_period(self):
        for start, end in self.dates_to_test:
            period = self.instance.get_next_period(start)
            period_real = aux.get_next_month(start)
            self.assertEqual(period, period_real)

    def test_get_next_periods(self):
        for start, end in self.dates_to_test:
            period = self.instance.get_prev_period(start)
            period_real = aux.get_prev_month(start)
            self.assertEqual(period, period_real)

    def test_resources_equal(self):
        resources = self.instance.get_resources()
    pass

    if __name__ == "__main__":
        # t = TestHeuristic()
        # t.test_order_1()
        # # t.test_swap17_A15()
        # # t.test_check_swap_defects_sq7()
        unittest.main()
