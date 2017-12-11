# /usr/bin/python3
import os


class Config(object):

    def __init__(self, options):
        self.gap = options['gap']
        self.path = options['path']
        self.timeLimit = options['timeLimit']

    def config_cbc(self):
        return \
            ["presolve on",
                       "gomory on",
                       "knapsack on",
                       "probing on",
                       "ratio {}".format(self.gap),
                       "sec {}".format(self.timeLimit)]

    def config_gurobi(self):
        # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        result_path = self.path + 'results.sol'.format()
        log_path = self.path + 'results.log'
        return [('TimeLimit', self.timeLimit),
                ('ResultFile', result_path),
                ('LogFile', log_path),
                ('MIPGap', self.gap)]

    def config_cplex(self):
        # CPLEX parameters: https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/settingParams.html
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        # result_path = path + 'results.sol'.format()
        log_path = self.path + 'results.log'
        return ['set logfile {}'.format(log_path),
                         'set timelimit {}'.format(self.timeLimit),
                         'mip tolerances mipgap {}'.format(self.gap)]
