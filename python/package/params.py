import os
import datetime as dt

# TODO: TEMP not sure why python doesn't find the ENV VAR from bash.bashrc
if 'GUROBI_HOME' in os.environ:
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = ""
    os.environ['LD_LIBRARY_PATH'] += ':' + os.environ['GUROBI_HOME'] + "/lib"

path_base = r'/home'
# path_base = r'C:\Users'
path_root = path_base + '/pchtsp/Documents/projects/'
path_results = path_base + '/pchtsp/Documents/projects/optima_results/'
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

params_cplex = \
[ 'set mip cuts flowcovers 1'
, 'set mip cuts mircut 1'
, 'set mip strategy backtrack 0.1'
, 'set mip strategy heuristicfreq 100'
, 'set mip strategy presolvenode 2'
, 'set mip strategy probe 3'
, 'set mip limits gomorycand 10000'
, 'set mip limits gomorypass 10'
, 'set mip tolerances mipgap 0']


OPTIONS = {
    'timeLimit': 3600  # seconds
    , 'gap': 0
    , 'solver': "HEUR_mf"  # HEUR, CPO, CHOCO, CPLEX, GUROBI, CBC, HEUR_mf
    , 'memory': None
    , 'solver_add_opts': params_cplex
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
    , 'simulate': True
    , 'slack_vars': "No"  # ['No', 'Yes', 3, 6]
    , 'writeLP': False
    , 'writeMPS': False
    , 'price_rut_end': 0
    , 'seed': None
    , 'simulation': {
        'num_resources': 15  # this depends on the number of tasks actually
        , 'num_parallel_tasks': 1
        , 'maint_duration': 6
        , 'max_used_time': 1000
        , 'max_elapsed_time': 60  # max time without maintenance
        , 'elapsed_time_size': 30  # size of window to do next maintenance
        , 'min_usage_period': 5  # minimum consumption per period
        , 'perc_capacity': 0.15
        , 'min_avail_percent': 0.1  # min percentage of available aircraft per type
        , 'min_avail_value': 1  # min num of available aircraft per type
        , 'min_hours_perc': 0.5  # min percentage of maximum possible hours of fleet type
        , 'seed': 44
        # The following are fixed options, not arguments for the scenario:
        , 't_min_assign': [2, 3, 6]  # minimum assignment time for tasks
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

