import unittest
import package.exec as exec
import data.data_input as di
import package.experiment as exp
import os

class TestHeuristic(unittest.TestCase):

    def setUp(self):
        # directory = '/home/pchtsp/Documents/projects/OPTIMA/data/examples/201811092041/'
        directory = './../../data/template/Test3/'
        self.experiment = exp.Experiment.from_template_dir(directory)
        self.options = di.load_data(os.path.join(directory, 'options.json'))

    # def solve_example(self):
    #     exec.execute_solve(model_data, options)
    def dataset1(self):
        pass

if __name__ == "__main__":
    t = TestHeuristic()
    t.setUp()
    # t.test_order_1()
    # unittest.main()