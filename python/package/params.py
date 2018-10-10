import os
import datetime as dt

# TODO: TEMP not sure why python doesn't find the ENV VAR from bash.bashrc
os.environ['LD_LIBRARY_PATH'] = os.environ['GUROBI_HOME'] + "/lib"


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
    'timeLimit': 3600  # seconds
    , 'gap': 0
    , 'solver': "GUROBI"  # HEUR, CPO, CHOCO, CPLEX, GUROBI, CBC
    , 'memory': 15000
    , 'print': True
    , 'integer': False
    , 'black_list': ['O8', 'O10', 'O6']
    , 'white_list': []
    , 'start': '2018-01'
    , 'num_period': 50
    , 'path': os.path.join(
        PATHS['experiments'],
        dt.datetime.now().strftime("%Y%m%d%H%M")
    ) + '/'
    , 'simulate': True
    , 'slack_vars': "No"  # ['No', 'Yes', 3, 6]
    , 'writeLP': True
    , 'writeMPS': False
    , 'simulation': {
        'num_resources': 70
        , 'num_parallel_tasks': 3
        , 'maint_duration': 6
        , 'max_used_time': 1000
        , 'max_elapsed_time': 60  # max time without maintenance
        , 'elapsed_time_size': 30  # size of window to do next maintenance
        , 'min_usage_period': 20  # minimum consumption per period
        , 'perc_capacity': 0.15
        , 'min_avail_percent': 0.1  # min percentage of available aircraft per type
        , 'min_avail_value': 1  # min num of available aircraft per type
        , 'min_hours_perc': 0.5  # min percentage of maximum possible hours of fleet type
        # , 'seed': 9366
        , 'seed': None
        # The following are fixed options, not arguments for the scenario:
        , 't_min_assign': [2, 3, 6]
        , 'initial_unbalance': (-3, 3)
        , 't_required_hours': (30, 50, 80) # triangular distribution params
        , 't_num_resource': (2, 5)
        , 't_duration': (6, 12)
        , 'perc_in_maint': 0.07
    }
}

# white_list = ['O1', 'O5']
# black_list = []
# black_list = ['O10', 'O8', 'O6']
# black_list = ['O8']  # this task has less candidates than what it asks.

