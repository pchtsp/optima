import os
import datetime as dt

path_root = '/home/pchtsp/Documents/projects/'
path_results = '/home/pchtsp/Dropbox/OPTIMA_results/'
path_project = path_root + "OPTIMA/"

PATHS = {
    'root': path_root
    ,'results': path_results
    ,'experiments': path_results + "experiments/"
    ,'img': path_project + "img/"
    ,'latex': path_project + "latex/"
    ,'data': path_project + "data/"
}

PATHS['input'] = PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
PATHS['hist'] = PATHS['data'] + 'raw/Planifs M2000.xlsm'

OPTIONS = {
    'timeLimit': 1800  # seconds
    , 'gap': 0.00049
    , 'solver': "CPLEX"  # HEUR, CPO, CHOCO, CPLEX, GUROBI, CBC
    , 'memory': 15000
    , 'print': True
    , 'integer': False
    , 'black_list': ['O8', 'O10', 'O6']
    , 'white_list': []
    , 'start': '2018-01'
    , 'num_period': 90
    , 'path': os.path.join(
        PATHS['experiments'],
        dt.datetime.now().strftime("%Y%m%d%H%M")
    ) + '/'
    , 'simulate': False
    , 'slack_vars': True
    , 'simulation': {
        'num_resources': 100
        , 'num_parallel_tasks': 2
        , 'maint_duration': 6
        , 'max_used_time': 1000
        , 'max_elapsed_time': 60
        , 'elapsed_time_size': 30
        , 'min_usage_period': 20
        , 'perc_capacity': 0.25
        # , 'seed': 9366
        , 'seed': 500
        # The following are fixed options, not arguments for the scenario:
        , 't_min_assign': [2, 3, 6]
        # , 't_required_hours': [50, 60, 70, 80]
        , 't_required_hours': [r for r in range(30, 90, 10)]
        , 't_num_resource': (2, 5)
        , 't_duration': (12, 36)
        , 'perc_in_maint': 0.15
    }
}

# white_list = ['O1', 'O5']
# black_list = []
# black_list = ['O10', 'O8', 'O6']
# black_list = ['O8']  # this task has less candidates than what it asks.

