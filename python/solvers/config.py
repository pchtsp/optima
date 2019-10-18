import os
import package.auxiliar as aux
import pulp as pl
import tempfile
from os import dup, dup2, close
import package.params as params


class Config(object):

    def __init__(self, options):
        if options is None:
            options = {}

        default_options = {
            'timeLimit': 300
            , 'gap': None
            , 'solver': "GUROBI"
            , 'path':
                os.path.join(params.PATHS['experiments'], aux.get_timestamp()) + '/'
            , 'memory': None
            , 'writeMPS': False
            , 'writeLP': False
            , 'gap_abs': None
        }

        # the following merges the two configurations (replace into):
        options = {**default_options, **options}


        self.gap = options['gap']
        self.path = options['path']
        self.timeLimit = options['timeLimit']
        self.solver = options['solver']
        self.solver_add_opts = options.get('solver_add_opts', [])
        self.mip_start = options.get('mip_start', False)
        self.gap_abs = options['gap_abs']
        self.log_path = os.path.join(self.path, 'results.log')
        self.result_path = os.path.join(self.path, 'results.sol')

        if options['memory'] is None:
            if hasattr(os, "sysconf"):
                self.memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 2)
            else: # windows
                self.memory = None
        else:
            self.memory = options['memory']
        self.writeMPS = options['writeMPS']
        self.writeLP = options['writeLP']

        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def config_cbc(self):

        params_eq  = \
            dict(timeLimit="sec {}",
                 gap = "ratio {}",
                 gap_abs = 'allow {}',
                 )
        params = [v.format(getattr(self, k)) for k, v in params_eq.items()
                  if getattr(self, k) is not None] + self.solver_add_opts

        return \
            ["presolve on",
             "gomory on",
             "knapsack on",
             "probing on"] + params + self.solver_add_opts

    def config_gurobi(self):
        # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters

        params_eq  = \
            dict(log_path='LogFile',
                 timeLimit='TimeLimit',
                 gap = 'MIPGap',
                 gap_abs = 'MIPGapAbs',
                 result_path = 'ResultFile'
                 )
        return [(v, getattr(self, k)) for k, v in params_eq.items()
             if getattr(self, k) is not None] + self.solver_add_opts

    def config_cplex(self):
        # CPLEX parameters: https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/settingParams.html

        params_eq  = \
            dict(log_path='set logFile {}',
                 timeLimit='set timelimit {}',
                 gap = 'set mip tolerances mipgap {}',
                 gap_abs = 'set mip tolerances absmipgap {}',
                 memory = 'set mip limits treememory {}'
                 )
        a = [v.format(getattr(self, k)) for k, v in params_eq.items()
            if getattr(self, k) is not None] + \
           self.solver_add_opts
        return a

    def config_choco(self):
        # CHOCO parameters https://github.com/chocoteam/choco-parsers/blob/master/MPS.md
        return [('-tl', self.timeLimit * 1000),
                ('-p', 1)] + self.solver_add_opts

    def solve_model(self, model):

        if self.writeMPS:
            model.writeMPS(filename=os.path.join(self.path, 'formulation.mps'))
        if self.writeLP:
            model.writeLP(filename=os.path.join(self.path, 'formulation.lp'))

        if self.solver == "GUROBI":
            return model.solve(pl.GUROBI_CMD(options=self.config_gurobi(), keepFiles=1))
        if self.solver == "CPLEX":
            try:
                result = model.solve(pl.CPLEX_CMD(options=self.config_cplex(), keepFiles=1, mip_start=self.mip_start),
                                     timeout=self.timeLimit + 60)
            except pl.PulpSolverError:
                result = 0
            return result
        if self.solver == "CHOCO":
            return model.solve(pl.PULP_CHOCO_CMD(options=self.config_choco(), keepFiles=1, msg=0))
        if self.solver == "CBC":
            with tempfile.TemporaryFile() as tmp_output:
                orig_std_out = dup(1)
                dup2(tmp_output.fileno(), 1)
                result = model.solve(
                    pl.PULP_CBC_CMD(options=self.config_cbc(), msg=True, keepFiles=1, mip_start=self.mip_start)
                )
                dup2(orig_std_out, 1)
                close(orig_std_out)
                tmp_output.seek(0)
                logFile = [line.decode('ascii') for line in tmp_output.read().splitlines()]
            with open(self.path + "results.log", 'w') as f:
                for item in logFile:
                    f.write("{}\n".format(item))
            return result
        return 0

