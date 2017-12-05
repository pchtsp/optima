# /usr/bin/python3
import os

def config_cbc(gap, time, path):
    return \
        ["presolve on",
                   "gomory on",
                   "knapsack on",
                   "probing on",
                   "ratio {}".format(gap),
                   "sec {}".format(time)]


def config_gurobi(gap, time, path):
    # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters
    if not os.path.exists(path):
        os.mkdir(path)
    result_path = path + 'results.sol'.format()
    log_path = path + 'results.log'
    return [('TimeLimit', time),
            ('ResultFile', result_path),
            ('LogFile', log_path),
            ('MIPGap', gap)]


def config_cplex(gap, time, path):
    # CPLEX parameters: https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/settingParams.html
    if not os.path.exists(path):
        os.mkdir(path)
    # result_path = path + 'results.sol'.format()
    log_path = path + 'results.log'
    return ['set logfile {}'.format(log_path),
                     'set timelimit {}'.format(time),
                     'mip tolerances mipgap {}'.format(gap)]
