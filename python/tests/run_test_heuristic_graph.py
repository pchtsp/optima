try:
    import tests.test_heuristic_graph as thg
except:
    import test_heuristic_graph as thg
import unittest


def run_one_case(test, data):
    suite = unittest.TestSuite()
    suite.addTest(thg.TestHeuristic(test, data))
    unittest.TextTestRunner(verbosity=0).run(suite)

def run_all():
    # Tests
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    # we get suite with all PuLP tests
    suite_all = thg.suite()
    # we run all tests at the same time
    ret = runner.run(suite_all)

if __name__ == '__main__':
    # run_all()
    import data.test_data as tdd
    run_one_case('test_solve', tdd.dataset6())