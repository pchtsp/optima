# /usr/bin/python3

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
            , 'path':
                os.path.join(params.PATHS['experiments'], aux.get_timestamp()) + '/'
        }

        # the following merges the two configurations (replace into):
        options = {**default_options, **options}


        self.gap = options.get('gap')
        self.path = options['path']
        self.timeLimit = options['timeLimit']
        self.solver = options.get('solver', 'GUROBI')
        self.solver_add_opts = options.get('solver_add_opts', {}).get(self.solver, [])
        self.mip_start = options.get('mip_start', False)
        self.gap_abs = options.get('gap_abs')
        self.log_path = self.path + 'results.log'
        self.result_path = self.path + 'results.sol'
        self.threads = options.get('threads')
        self.solver_path = options.get('solver_path')

        if options['memory'] is None:
            if hasattr(os, "sysconf"):
                self.memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 2)
            else: # windows
                self.memory = None
        else:
            self.memory = options.get('memory')
        self.writeMPS = options.get('writeMPS', False)
        self.writeLP = options.get('writeLP', False)

    def config_cbc(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        params_eq  = \
            dict(timeLimit="sec {}",
                 gap = "ratio {}",
                 gap_abs = 'allow {}',
                 threads = 'threads {}'
                 )
        params = [v.format(getattr(self, k)) for k, v in params_eq.items()
                  if getattr(self, k) is not None] + self.solver_add_opts

        return params + self.solver_add_opts

    def config_gurobi(self):
        # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters
        if not os.path.exists(self.path):
            os.mkdir(self.path)

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
        if not os.path.exists(self.path):
            os.mkdir(self.path)

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
        # print(a)
        return a

    def config_choco(self):
        # CHOCO parameters https://github.com/chocoteam/choco-parsers/blob/master/MPS.md
        return [('-tl', self.timeLimit * 1000),
                ('-p', 1)] + self.solver_add_opts

    def solve_model(self, model):

        if self.writeMPS:
            model.writeMPS(filename=self.path + 'formulation.mps')
        if self.writeLP:
            model.writeLP(filename=self.path + 'formulation.lp')

        solver = None
        if self.solver == "GUROBI":
            solver = pl.GUROBI_CMD(options=self.config_gurobi(), keepFiles=1)
        if self.solver == "CPLEX":
            solver = pl.CPLEX_CMD(options=self.config_cplex(), keepFiles=1, mip_start=self.mip_start)
        if self.solver == "CHOCO":
            solver = pl.PULP_CHOCO_CMD(options=self.config_choco(), keepFiles=1, msg=0)
        if solver is not None:
            return model.solve(solver)
        if self.solver == "CBC":
            if self.solver_path:
                solver = pl.COIN_CMD(options=self.config_cbc(), msg=True, keepFiles=1,
                                     mip_start=self.mip_start, path=self.solver_path)
            else:
                solver = pl.PULP_CBC_CMD(options=self.config_cbc(), msg=True, keepFiles=1,
                                         mip_start=self.mip_start)
            with tempfile.TemporaryFile() as tmp_output:
                orig_std_out = dup(1)
                dup2(tmp_output.fileno(), 1)
                result = model.solve(solver)
                dup2(orig_std_out, 1)
                close(orig_std_out)
                tmp_output.seek(0)
                logFile = [line.decode('ascii') for line in tmp_output.read().splitlines()]
            with open(self.path + "results.log", 'w') as f:
                for item in logFile:
                    f.write("{}\n".format(item))
            return result
        return 0

