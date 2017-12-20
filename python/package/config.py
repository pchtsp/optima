# /usr/bin/python3
import os
import package.aux as aux
import pulp as pl
import tempfile
from os import dup, dup2, close


class Config(object):

    def __init__(self, options):
        if options is None:
            options = {}

        default_options = {
            'timeLimit': 300
            , 'gap': 0
            , 'solver': "GUROBI"
            , 'path':
                '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.
                    format(aux.get_timestamp())
        }

        # the following merges the two configurations (replace into):
        options = {**default_options, **options}

        self.gap = options['gap']
        self.path = options['path']
        self.timeLimit = options['timeLimit']
        self.solver = options['solver']

    def config_cbc(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_path = self.path + 'results.log'
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
        log_path = self.path + 'results.log'
        return ['set logfile {}'.format(log_path),
                'set timelimit {}'.format(self.timeLimit),
                'set mip tolerances mipgap {}'.format(self.gap)]

    def solve_model(self, model):
        if self.solver == "GUROBI":
            return model.solve(pl.GUROBI_CMD(options=self.config_gurobi()))
        if self.solver == "CPLEX":
            return model.solve(pl.CPLEX_CMD(options=self.config_cplex()))

        with tempfile.TemporaryFile() as tmp_output:
            orig_std_out = dup(1)
            dup2(tmp_output.fileno(), 1)
            result = model.solve(pl.PULP_CBC_CMD(options=self.config_cbc(), msg=True))
            dup2(orig_std_out, 1)
            close(orig_std_out)
            tmp_output.seek(0)
            logFile = [line.decode('ascii') for line in tmp_output.read().splitlines()]
        with open(self.path + "results.log", 'w') as f:
            for item in logFile:
                f.write("{}\n".format(item))
            # f.writelines(["%s\n" % item for item in list])
            # f.write(r'\n'.join(logFile))
        return result

